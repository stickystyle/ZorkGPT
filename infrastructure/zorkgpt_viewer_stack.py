#!/usr/bin/env python3

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_s3_deployment as s3deploy,
    aws_iam as iam,
    aws_ec2 as ec2,
    Duration,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct


class ZorkGPTViewerStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create S3 bucket for hosting the website and state files
        self.bucket = s3.Bucket(
            self,
            "ZorkGPTViewerBucket",
            bucket_name=f"zorkgpt-viewer-{self.account}-{self.region}",
            public_read_access=False,  # We'll use CloudFront for access
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,  # For development - change for production
            auto_delete_objects=True,  # For development - change for production
            versioning=False,  # State files don't need versioning
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.HEAD],
                    allowed_origins=["*"],
                    allowed_headers=["*"],
                    max_age=3600,
                )
            ],
        )

        # Create Origin Access Control for CloudFront to access S3
        oac = cloudfront.OriginAccessControl(
            self,
            "ZorkGPTViewerOAC",
            description="Origin Access Control for ZorkGPT Viewer",
            origin_access_control_origin_type=cloudfront.OriginAccessControlOriginType.S3,
            signing=cloudfront.Signing.SIGV4_ALWAYS,
        )

        # Create cache behaviors - different caching for different file types
        cache_behaviors = [
            # Don't cache current_state.json at all
            cloudfront.BehaviorOptions(
                path_pattern="/current_state.json",
                origin=origins.S3Origin(
                    self.bucket,
                    origin_access_control=oac,
                ),
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD,
            ),
            # Cache static assets (HTML, CSS, JS) for a reasonable time
            cloudfront.BehaviorOptions(
                path_pattern="*.html",
                origin=origins.S3Origin(
                    self.bucket,
                    origin_access_control=oac,
                ),
                cache_policy=cloudfront.CachePolicy(
                    self,
                    "HTMLCachePolicy",
                    cache_policy_name=f"ZorkGPT-HTML-Cache-{self.stack_name}",
                    comment="Cache policy for HTML files",
                    default_ttl=Duration.minutes(5),  # Short cache for HTML
                    max_ttl=Duration.hours(1),
                    min_ttl=Duration.seconds(0),
                    cookie_behavior=cloudfront.CacheCookieBehavior.none(),
                    header_behavior=cloudfront.CacheHeaderBehavior.none(),
                    query_string_behavior=cloudfront.CacheQueryStringBehavior.all(),  # Allow cache busting
                    enable_accept_encoding_gzip=True,
                    enable_accept_encoding_brotli=True,
                ),
                origin_request_policy=cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD,
            ),
        ]

        # Create CloudFront distribution
        self.distribution = cloudfront.Distribution(
            self,
            "ZorkGPTViewerDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3Origin(
                    self.bucket,
                    origin_access_control=oac,
                ),
                cache_policy=cloudfront.CachePolicy(
                    self,
                    "DefaultCachePolicy",
                    cache_policy_name=f"ZorkGPT-Default-Cache-{self.stack_name}",
                    comment="Default cache policy for ZorkGPT viewer",
                    default_ttl=Duration.hours(1),  # Cache static assets for 1 hour
                    max_ttl=Duration.days(1),
                    min_ttl=Duration.seconds(0),
                    cookie_behavior=cloudfront.CacheCookieBehavior.none(),
                    header_behavior=cloudfront.CacheHeaderBehavior.none(),
                    query_string_behavior=cloudfront.CacheQueryStringBehavior.all(),
                    enable_accept_encoding_gzip=True,
                    enable_accept_encoding_brotli=True,
                ),
                origin_request_policy=cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD,
            ),
            additional_behaviors=dict(
                (behavior.path_pattern, behavior) for behavior in cache_behaviors
            ),
            default_root_object="zork_viewer.html",
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,  # Use only North America and Europe
            comment="CloudFront distribution for ZorkGPT Live Viewer",
            enabled=True,
        )

        # Grant CloudFront access to the S3 bucket
        self.bucket.add_to_resource_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal("cloudfront.amazonaws.com")],
                actions=["s3:GetObject"],
                resources=[f"{self.bucket.bucket_arn}/*"],
                conditions={
                    "StringEquals": {
                        "AWS:SourceArn": f"arn:aws:cloudfront::{self.account}:distribution/{self.distribution.distribution_id}"
                    }
                },
            )
        )

        # Deploy the initial website files
        s3deploy.BucketDeployment(
            self,
            "ZorkGPTViewerDeployment",
            sources=[s3deploy.Source.asset(".")],  # Deploy from current directory
            destination_bucket=self.bucket,
            include=["zork_viewer.html"],  # Only deploy the HTML file initially
            distribution=self.distribution,
            distribution_paths=["/*"],  # Invalidate all paths after deployment
        )

        # Create VPC for EC2 instance (or use default VPC)
        vpc = ec2.Vpc.from_lookup(self, "DefaultVPC", is_default=True)

        # Create security group for ZorkGPT instance
        security_group = ec2.SecurityGroup(
            self,
            "ZorkGPTSecurityGroup",
            vpc=vpc,
            description="Security group for ZorkGPT EC2 instance",
            allow_all_outbound=True,
        )

        # Allow SSH access
        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(22),
            "SSH access"
        )

        # Create IAM role for EC2 instance
        ec2_role = iam.Role(
            self,
            "ZorkGPTEC2Role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="IAM role for ZorkGPT EC2 instance",
        )

        # Grant S3 permissions to the EC2 instance
        self.bucket.grant_read_write(ec2_role)

        # Add CloudWatch logs permissions
        ec2_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("CloudWatchAgentServerPolicy")
        )

        # Create user data script to set up the instance
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "#!/bin/bash",
            "yum update -y",
            "yum install -y git python3 python3-pip",
            
            # Install uv for faster Python package management
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "source $HOME/.cargo/env",
            
            # Create zorkgpt user
            "useradd -m -s /bin/bash zorkgpt",
            "mkdir -p /home/zorkgpt/.ssh",
            "chown zorkgpt:zorkgpt /home/zorkgpt/.ssh",
            
            # Clone ZorkGPT repository
            "cd /home/zorkgpt",
            "git clone https://github.com/stickystyle/ZorkGPT.git",
            "chown -R zorkgpt:zorkgpt /home/zorkgpt/ZorkGPT",
            
            # Set up Python environment using uv
            "cd /home/zorkgpt/ZorkGPT",
            "sudo -u zorkgpt /home/zorkgpt/.cargo/bin/uv sync",
            
            # Set up environment variables
            f"echo 'export ZORK_S3_BUCKET={self.bucket.bucket_name}' >> /home/zorkgpt/.bashrc",
            "echo 'export PATH=/home/zorkgpt/.cargo/bin:$PATH' >> /home/zorkgpt/.bashrc",
            
            # Create a simple startup script
            "cat > /home/zorkgpt/start_zorkgpt.sh << 'EOF'",
            "#!/bin/bash",
            "cd /home/zorkgpt/ZorkGPT",
            "export ZORK_S3_BUCKET=" + self.bucket.bucket_name,
            "/home/zorkgpt/.cargo/bin/uv run python main.py",
            "EOF",
            
            "chmod +x /home/zorkgpt/start_zorkgpt.sh",
            "chown zorkgpt:zorkgpt /home/zorkgpt/start_zorkgpt.sh",
            
            # Create systemd service for auto-start (optional)
            "cat > /etc/systemd/system/zorkgpt.service << 'EOF'",
            "[Unit]",
            "Description=ZorkGPT Game Runner",
            "After=network.target",
            "",
            "[Service]",
            "Type=simple",
            "User=zorkgpt",
            "WorkingDirectory=/home/zorkgpt/ZorkGPT",
            "Environment=ZORK_S3_BUCKET=" + self.bucket.bucket_name,
            "Environment=PATH=/home/zorkgpt/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "ExecStart=/home/zorkgpt/.cargo/bin/uv run python main.py",
            "Restart=always",
            "RestartSec=10",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "EOF",
            
            "systemctl daemon-reload",
            "systemctl enable zorkgpt.service",
            
            # Log completion
            "echo 'ZorkGPT setup completed' > /var/log/zorkgpt-setup.log",
        )

        # Create EC2 instance
        self.ec2_instance = ec2.Instance(
            self,
            "ZorkGPTInstance",
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MICRO
            ),
            machine_image=ec2.AmazonLinuxImage(
                generation=ec2.AmazonLinuxGeneration.AMAZON_LINUX_2
            ),
            vpc=vpc,
            security_group=security_group,
            role=ec2_role,
            user_data=user_data,
            key_name="parrishfamily",  # Your existing key pair
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        # Outputs
        CfnOutput(
            self,
            "BucketName",
            value=self.bucket.bucket_name,
            description="S3 bucket name for ZorkGPT viewer",
        )

        CfnOutput(
            self,
            "DistributionDomainName",
            value=self.distribution.distribution_domain_name,
            description="CloudFront distribution domain name",
        )

        CfnOutput(
            self,
            "DistributionId",
            value=self.distribution.distribution_id,
            description="CloudFront distribution ID",
        )

        CfnOutput(
            self,
            "ViewerURL",
            value=f"https://{self.distribution.distribution_domain_name}",
            description="URL to access the ZorkGPT Live Viewer",
        )

        CfnOutput(
            self,
            "StateFileURL",
            value=f"https://{self.distribution.distribution_domain_name}/current_state.json",
            description="URL for the current state JSON file",
        )

        CfnOutput(
            self,
            "EC2InstanceId",
            value=self.ec2_instance.instance_id,
            description="EC2 instance ID running ZorkGPT",
        )

        CfnOutput(
            self,
            "EC2PublicIP",
            value=self.ec2_instance.instance_public_ip,
            description="Public IP address of the ZorkGPT EC2 instance",
        )

        CfnOutput(
            self,
            "EC2SSHCommand",
            value=f"ssh -i ~/.ssh/parrishfamily.pem ec2-user@{self.ec2_instance.instance_public_ip}",
            description="SSH command to connect to the ZorkGPT instance",
        ) 