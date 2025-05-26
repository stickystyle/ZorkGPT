#!/usr/bin/env python3

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,

    aws_iam as iam,
    aws_ec2 as ec2,
    aws_route53 as route53,
    aws_route53_targets as targets,
    aws_certificatemanager as acm,
    Duration,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct


class ZorkGPTViewerStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Domain configuration
        domain_name = "zorkgpt.com"
        
        # Create a new hosted zone for the domain
        hosted_zone = route53.HostedZone(
            self,
            "ZorkGPTHostedZone",
            zone_name=domain_name,
            comment="Hosted zone for ZorkGPT domain",
        )

        # Create SSL certificate for the domain (must be in us-east-1 for CloudFront)
        certificate = acm.Certificate(
            self,
            "ZorkGPTCertificate",
            domain_name=domain_name,
            subject_alternative_names=[f"www.{domain_name}"],
            validation=acm.CertificateValidation.from_dns(hosted_zone),
        )

        # Create S3 bucket for hosting the website and state files
        self.bucket = s3.Bucket(
            self,
            "ZorkGPTViewerBucket",
            public_read_access=False,  # We'll use CloudFront for access
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.HEAD],
                    allowed_origins=["*"],
                    allowed_headers=["*"],
                    max_age=3600,
                )
            ],
        )

        # Create Origin Access Identity for CloudFront to access S3
        oai = cloudfront.OriginAccessIdentity(
            self,
            "ZorkGPTViewerOAI",
            comment="Origin Access Identity for ZorkGPT Viewer",
        )

        # Create S3 origin for CloudFront
        s3_origin = origins.S3BucketOrigin.with_origin_access_identity(
            bucket=self.bucket,
            origin_access_identity=oai,
        )

        # Create CloudFront distribution
        self.distribution = cloudfront.Distribution(
            self,
            "ZorkGPTViewerDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=s3_origin,
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
            additional_behaviors={
                # Cache current_state.json for 6 seconds to balance freshness with origin protection
                "/current_state.json": cloudfront.BehaviorOptions(
                    origin=s3_origin,
                    cache_policy=cloudfront.CachePolicy(
                        self,
                        "StateFileCachePolicy",
                        cache_policy_name=f"ZorkGPT-State-Cache-{self.stack_name}",
                        comment="Cache policy for current_state.json - 6 second cache for real-time data",
                        default_ttl=Duration.seconds(6),  # Cache for 6 seconds
                        max_ttl=Duration.seconds(10),     # Max 10 seconds
                        min_ttl=Duration.seconds(0),      # Allow no cache if needed
                        cookie_behavior=cloudfront.CacheCookieBehavior.none(),
                        header_behavior=cloudfront.CacheHeaderBehavior.none(),
                        query_string_behavior=cloudfront.CacheQueryStringBehavior.none(),
                        enable_accept_encoding_gzip=True,
                        enable_accept_encoding_brotli=True,
                    ),
                    origin_request_policy=cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD,
                ),
                # Cache static assets (HTML, CSS, JS) for a reasonable time
                "*.html": cloudfront.BehaviorOptions(
                    origin=s3_origin,
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
            },
            default_root_object="zork_viewer.html",
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,  # Use only North America and Europe
            comment="CloudFront distribution for ZorkGPT Live Viewer",
            enabled=True,
            # Custom domain configuration
            domain_names=[domain_name, f"www.{domain_name}"],
            certificate=certificate,
        )

        # Grant CloudFront OAI access to the S3 bucket
        self.bucket.grant_read(oai)

        # Create Route53 A records to point domain to CloudFront distribution
        route53.ARecord(
            self,
            "ZorkGPTARecord",
            zone=hosted_zone,
            record_name=domain_name,
            target=route53.RecordTarget.from_alias(
                targets.CloudFrontTarget(self.distribution)
            ),
        )

        route53.ARecord(
            self,
            "ZorkGPTWWWARecord",
            zone=hosted_zone,
            record_name=f"www.{domain_name}",
            target=route53.RecordTarget.from_alias(
                targets.CloudFrontTarget(self.distribution)
            ),
        )

        # Note: HTML files are now deployed from the EC2 instance using the manage_ec2.py update command
        # This avoids unwanted service restarts caused by CDK BucketDeployment custom resources


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
            ec2.Peer.any_ipv4(), ec2.Port.tcp(22), "SSH access"
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
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "CloudWatchAgentServerPolicy"
            )
        )

        # Create user data script to set up the instance
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "#!/bin/bash",
            "dnf update -y",
            "dnf install -y git python3 python3-pip gcc make ncurses-devel awscli",
            # Install Frotz - compile from source since Amazon Linux 2023 doesn't have frotz in repos
            "cd /tmp",
            "git clone https://gitlab.com/DavidGriffith/frotz.git",
            "cd frotz",
            "make dumb",
            "make install_dumb",
            # Create zorkgpt user
            "useradd -m -s /bin/bash zorkgpt",
            "mkdir -p /home/zorkgpt/.ssh",
            "chown zorkgpt:zorkgpt /home/zorkgpt/.ssh",
            # Install uv for zorkgpt user
            "sudo -u zorkgpt curl -LsSf https://astral.sh/uv/install.sh | sudo -u zorkgpt sh",
            # Clone ZorkGPT repository
            "cd /home/zorkgpt",
            "git clone https://github.com/stickystyle/ZorkGPT.git",
            "chown -R zorkgpt:zorkgpt /home/zorkgpt/ZorkGPT",
            # Set up Python environment using uv with S3 dependencies
            "cd /home/zorkgpt/ZorkGPT",
            "sudo -u zorkgpt /home/zorkgpt/.local/bin/uv sync --extra s3",
            # Ensure zork.z5 file is available (download from S3 bucket if needed)
            f"aws s3 cp s3://{self.bucket.bucket_name}/infrastructure/zork.z5 /home/zorkgpt/ZorkGPT/infrastructure/zork.z5 || echo 'Could not download zork.z5 from S3'",
            "chown zorkgpt:zorkgpt /home/zorkgpt/ZorkGPT/infrastructure/zork.z5",
            # Verify Frotz installation
            "which dfrotz > /var/log/frotz-install.log 2>&1",
            "echo 'Frotz installation check:' >> /var/log/frotz-install.log",
            "ls -la /usr/local/bin/dfrotz >> /var/log/frotz-install.log 2>&1",
            "dfrotz -v >> /var/log/frotz-install.log 2>&1",
            # Verify zork.z5 file is available
            "echo 'Zork.z5 file check:' >> /var/log/frotz-install.log",
            "ls -la /home/zorkgpt/ZorkGPT/infrastructure/zork.z5 >> /var/log/frotz-install.log 2>&1",
            # Verify Python version
            "echo 'Python version check:' >> /var/log/frotz-install.log",
            "python3 --version >> /var/log/frotz-install.log 2>&1",
            # Set up environment variables
            f"echo 'export ZORK_S3_BUCKET={self.bucket.bucket_name}' >> /home/zorkgpt/.bashrc",
            "echo 'export PATH=/home/zorkgpt/.local/bin:$PATH' >> /home/zorkgpt/.bashrc",
            # Create a simple startup script
            "cat > /home/zorkgpt/start_zorkgpt.sh << 'EOF'",
            "#!/bin/bash",
            "cd /home/zorkgpt/ZorkGPT",
            "export ZORK_S3_BUCKET=" + self.bucket.bucket_name,
            "/home/zorkgpt/.local/bin/uv run python main.py",
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
            "Environment=PATH=/home/zorkgpt/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "ExecStart=/home/zorkgpt/.local/bin/uv run python main.py",
            "Restart=always",
            "RestartSec=10",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
            "EOF",
            "systemctl daemon-reload",
            "systemctl enable zorkgpt.service",
            # Upload initial HTML file to S3 for immediate viewer availability
            f"aws s3 cp /home/zorkgpt/ZorkGPT/zork_viewer.html s3://{self.bucket.bucket_name}/zork_viewer.html || echo 'Could not upload initial zork_viewer.html'",
            # Log completion
            "echo 'ZorkGPT setup completed with Frotz' > /var/log/zorkgpt-setup.log",
        )

        # Create EC2 instance
        self.ec2_instance = ec2.Instance(
            self,
            "ZorkGPTInstance",
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.MICRO
            ),
            machine_image=ec2.AmazonLinuxImage(
                generation=ec2.AmazonLinuxGeneration.AMAZON_LINUX_2023
            ),
            vpc=vpc,
            security_group=security_group,
            role=ec2_role,
            user_data=user_data,
            key_pair=ec2.KeyPair.from_key_pair_name(self, "ImportedKeyPair", "parrishfamily"),
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
            value=f"https://{domain_name}",
            description="URL to access the ZorkGPT Live Viewer",
        )

        CfnOutput(
            self,
            "StateFileURL",
            value=f"https://{domain_name}/current_state.json",
            description="URL for the current state JSON file",
        )

        CfnOutput(
            self,
            "CustomDomainName",
            value=domain_name,
            description="Custom domain name for ZorkGPT",
        )

        CfnOutput(
            self,
            "CertificateArn",
            value=certificate.certificate_arn,
            description="ARN of the SSL certificate",
        )

        CfnOutput(
            self,
            "HostedZoneId",
            value=hosted_zone.hosted_zone_id,
            description="Route53 Hosted Zone ID",
        )

        CfnOutput(
            self,
            "NameServers",
            value=cdk.Fn.join(",", hosted_zone.hosted_zone_name_servers),
            description="Route53 nameservers - configure these with your domain registrar",
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
