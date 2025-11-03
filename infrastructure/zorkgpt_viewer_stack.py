#!/usr/bin/env python3

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_sns as sns,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cloudwatch_actions,
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
                        max_ttl=Duration.seconds(10),  # Max 10 seconds
                        min_ttl=Duration.seconds(0),  # Allow no cache if needed
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

        # Add SSM permissions for remote management
        ec2_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMManagedInstanceCore"
            )
        )

        # Create SNS topic for monitoring alerts
        alert_topic = sns.Topic(
            self, "ZorkGPTAlerts", display_name="ZorkGPT Monitoring Alerts"
        )

        # Grant SNS permissions to EC2 role
        alert_topic.grant_publish(ec2_role)

        # Create user data script to set up the instance
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "#!/bin/bash",
            "dnf update -y",
            "dnf install -y git python3 python3-pip gcc make ncurses-devel awscli amazon-cloudwatch-agent",
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
            # Set up monitoring
            f"echo 'export ALERT_TOPIC_ARN={alert_topic.topic_arn}' >> /home/zorkgpt/.bashrc",
            # Create monitoring script
            "cat > /home/zorkgpt/monitor.py << 'EOF'",
            """
import json
import boto3
import datetime
import subprocess
import os
import sys

def log_to_journal(message, priority="info"):
    \"\"\"Log to systemd journal\"\"\"
    try:
        subprocess.run([
            'systemd-cat', '-t', 'zorkgpt-monitor', '-p', priority
        ], input=message, text=True, check=True)
    except:
        print(f"Failed to log to journal: {message}")

def check_health():
    timestamp = datetime.datetime.now().isoformat()
    
    # Check systemd service
    try:
        result = subprocess.run(['systemctl', 'is-active', 'zorkgpt'], 
                              capture_output=True, text=True)
        service_active = result.stdout.strip() == 'active'
    except:
        service_active = False
    
    # Check if process is actually running
    try:
        result = subprocess.run(['pgrep', '-f', 'main.py'], 
                              capture_output=True, text=True)
        process_running = bool(result.stdout.strip())
    except:
        process_running = False
    
    # Check current_state.json age (if it exists locally)
    state_file_fresh = True
    state_age_minutes = 0
    try:
        if os.path.exists('/home/zorkgpt/ZorkGPT/current_state.json'):
            with open('/home/zorkgpt/ZorkGPT/current_state.json', 'r') as f:
                state_data = json.load(f)
                state_timestamp = datetime.datetime.fromisoformat(
                    state_data['metadata']['timestamp'].replace('Z', '+00:00')
                )
                state_age_minutes = (datetime.datetime.now(datetime.timezone.utc) - state_timestamp).total_seconds() / 60
                state_file_fresh = state_age_minutes < 10  # Alert if > 10 minutes old
    except Exception as e:
        state_file_fresh = False
        log_to_journal(f"Error checking state file: {e}", "warning")
    
    health_data = {
        'timestamp': timestamp,
        'service_active': service_active,
        'process_running': process_running,
        'state_file_fresh': state_file_fresh,
        'state_age_minutes': round(state_age_minutes, 1)
    }
    
    # Log to journal (structured logging)
    log_message = json.dumps(health_data)
    
    # Determine log level based on health
    if service_active and process_running and state_file_fresh:
        log_to_journal(f"Health check OK: {log_message}", "info")
    else:
        log_to_journal(f"Health check FAILED: {log_message}", "err")
        send_alert(health_data)
    
    return health_data

def send_alert(health_data):
    try:
        topic_arn = os.environ.get('ALERT_TOPIC_ARN')
        if not topic_arn:
            log_to_journal("No ALERT_TOPIC_ARN set, skipping alert", "warning")
            return
            
        sns = boto3.client('sns')
        
        issues = []
        if not health_data['service_active']:
            issues.append("❌ Systemd service not active")
        if not health_data['process_running']:
            issues.append("❌ Python process not running")
        if not health_data['state_file_fresh']:
            issues.append(f"❌ State file stale ({health_data['state_age_minutes']} min old)")
        
        message = f'''🚨 ZorkGPT Health Alert

Time: {health_data['timestamp']}

Issues Detected:
{chr(10).join(issues)}

Status Summary:
• Service Active: {health_data['service_active']}
• Process Running: {health_data['process_running']}
• State File Fresh: {health_data['state_file_fresh']}
• State Age: {health_data['state_age_minutes']} minutes

Check logs: journalctl -u zorkgpt -f
Check monitor: journalctl -t zorkgpt-monitor -f
'''
        
        sns.publish(
            TopicArn=topic_arn,
            Subject='🚨 ZorkGPT Alert',
            Message=message
        )
        
        log_to_journal(f"Alert sent: {', '.join(issues)}", "notice")
        
    except Exception as e:
        log_to_journal(f"Failed to send alert: {e}", "err")

if __name__ == '__main__':
    check_health()
""",
            "EOF",
            "chown zorkgpt:zorkgpt /home/zorkgpt/monitor.py",
            # Create cron job to check every 5 minutes
            "echo '*/5 * * * * cd /home/zorkgpt && source ~/.bashrc && /home/zorkgpt/.local/bin/uv run python monitor.py' | sudo -u zorkgpt crontab -",
            # Create helpful log viewing scripts
            "cat > /home/zorkgpt/view_logs.sh << 'EOF'",
            "#!/bin/bash",
            "echo '=== ZorkGPT Service Logs (last 20 lines) ==='",
            "journalctl -u zorkgpt --no-pager -n 20",
            "echo ''",
            "echo '=== Monitor Logs (last 10 checks) ==='",
            "journalctl -t zorkgpt-monitor --no-pager -n 10",
            "echo ''",
            "echo '=== Current Service Status ==='",
            "systemctl status zorkgpt --no-pager",
            "EOF",
            "chmod +x /home/zorkgpt/view_logs.sh",
            "chown zorkgpt:zorkgpt /home/zorkgpt/view_logs.sh",
            # Create a script to follow logs in real-time
            "cat > /home/zorkgpt/follow_logs.sh << 'EOF'",
            "#!/bin/bash",
            "echo 'Following ZorkGPT logs... (Ctrl+C to exit)'",
            "echo 'Service logs in one terminal, monitor logs in another'",
            "echo ''",
            'if [ "$1" = "monitor" ]; then',
            "    journalctl -t zorkgpt-monitor -f",
            "else",
            "    journalctl -u zorkgpt -f",
            "fi",
            "EOF",
            "chmod +x /home/zorkgpt/follow_logs.sh",
            "chown zorkgpt:zorkgpt /home/zorkgpt/follow_logs.sh",
            # Configure CloudWatch Agent for system metrics
            "cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'",
            """{
    "metrics": {
        "namespace": "ZorkGPT/EC2",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60,
                "totalcpu": true
            },
            "disk": {
                "measurement": [
                    "used_percent",
                    "inodes_free"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "netstat": {
                "measurement": [
                    "tcp_established",
                    "tcp_time_wait"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}""",
            "EOF",
            # Start CloudWatch Agent
            "systemctl enable amazon-cloudwatch-agent",
            "systemctl start amazon-cloudwatch-agent",
            # Log completion
            "echo 'ZorkGPT setup completed with Frotz, monitoring, and CloudWatch Agent' > /var/log/zorkgpt-setup.log",
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
            key_pair=ec2.KeyPair.from_key_pair_name(
                self, "ImportedKeyPair", "parrishfamily"
            ),
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
        )

        # Create CloudWatch alarms for EC2 monitoring
        # CPU Utilization Alarm
        cpu_alarm = cloudwatch.Alarm(
            self,
            "ZorkGPTHighCPU",
            alarm_name="ZorkGPT-High-CPU-Usage",
            alarm_description="Alert when ZorkGPT EC2 instance CPU usage is high",
            metric=cloudwatch.Metric(
                namespace="ZorkGPT/EC2",
                metric_name="cpu_usage_user",
                dimensions_map={
                    "InstanceId": self.ec2_instance.instance_id,
                    "host": self.ec2_instance.instance_id,
                    "cpu": "cpu-total",
                },
                statistic="Average",
                period=Duration.minutes(5),
            ),
            threshold=80,  # Alert if CPU > 80%
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        cpu_alarm.add_alarm_action(cloudwatch_actions.SnsAction(alert_topic))

        # Memory Utilization Alarm
        memory_alarm = cloudwatch.Alarm(
            self,
            "ZorkGPTHighMemory",
            alarm_name="ZorkGPT-High-Memory-Usage",
            alarm_description="Alert when ZorkGPT EC2 instance memory usage is high",
            metric=cloudwatch.Metric(
                namespace="ZorkGPT/EC2",
                metric_name="mem_used_percent",
                dimensions_map={
                    "InstanceId": self.ec2_instance.instance_id,
                    "host": self.ec2_instance.instance_id,
                },
                statistic="Average",
                period=Duration.minutes(5),
            ),
            threshold=85,  # Alert if Memory > 85%
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        memory_alarm.add_alarm_action(cloudwatch_actions.SnsAction(alert_topic))

        # Disk Space Alarm
        disk_alarm = cloudwatch.Alarm(
            self,
            "ZorkGPTHighDiskUsage",
            alarm_name="ZorkGPT-High-Disk-Usage",
            alarm_description="Alert when ZorkGPT EC2 instance disk usage is high",
            metric=cloudwatch.Metric(
                namespace="ZorkGPT/EC2",
                metric_name="disk_used_percent",
                dimensions_map={
                    "InstanceId": self.ec2_instance.instance_id,
                    "host": self.ec2_instance.instance_id,
                    "device": "/dev/xvda1",
                    "fstype": "xfs",
                    "path": "/",
                },
                statistic="Average",
                period=Duration.minutes(5),
            ),
            threshold=80,  # Alert if Disk > 80%
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        disk_alarm.add_alarm_action(cloudwatch_actions.SnsAction(alert_topic))

        # Instance Status Check Alarm (built-in EC2 metric)
        status_alarm = cloudwatch.Alarm(
            self,
            "ZorkGPTInstanceStatusCheck",
            alarm_name="ZorkGPT-Instance-Status-Check-Failed",
            alarm_description="Alert when ZorkGPT EC2 instance status check fails",
            metric=cloudwatch.Metric(
                namespace="AWS/EC2",
                metric_name="StatusCheckFailed_Instance",
                dimensions_map={"InstanceId": self.ec2_instance.instance_id},
                statistic="Maximum",
                period=Duration.minutes(5),
            ),
            threshold=0,  # Alert if any status check fails
            evaluation_periods=2,
            datapoints_to_alarm=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        status_alarm.add_alarm_action(cloudwatch_actions.SnsAction(alert_topic))

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

        # Monitoring outputs
        CfnOutput(
            self,
            "AlertTopicArn",
            value=alert_topic.topic_arn,
            description="SNS Topic ARN for ZorkGPT alerts - subscribe manually via AWS Console",
        )

        CfnOutput(
            self,
            "MonitoringCommands",
            value=f"View logs: ssh -i ~/.ssh/parrishfamily.pem ec2-user@{self.ec2_instance.instance_public_ip} 'sudo /home/zorkgpt/view_logs.sh'",
            description="SSH command to view monitoring logs and service status",
        )

        CfnOutput(
            self,
            "LogCommands",
            value=f"""Follow service logs: ssh -i ~/.ssh/parrishfamily.pem ec2-user@{self.ec2_instance.instance_public_ip} 'sudo /home/zorkgpt/follow_logs.sh'
Follow monitor logs: ssh -i ~/.ssh/parrishfamily.pem ec2-user@{self.ec2_instance.instance_public_ip} 'sudo /home/zorkgpt/follow_logs.sh monitor'""",
            description="SSH commands to follow ZorkGPT logs in real-time",
        )

        # CloudWatch Alarm outputs
        CfnOutput(
            self,
            "CloudWatchAlarms",
            value=f"""CPU Alarm: {cpu_alarm.alarm_name}
Memory Alarm: {memory_alarm.alarm_name}
Disk Alarm: {disk_alarm.alarm_name}
Status Alarm: {status_alarm.alarm_name}""",
            description="CloudWatch alarms created for EC2 monitoring",
        )

        CfnOutput(
            self,
            "CloudWatchDashboard",
            value=f"https://console.aws.amazon.com/cloudwatch/home?region={self.region}#alarmsV2:alarm/{cpu_alarm.alarm_name}",
            description="CloudWatch console URL to view alarms and metrics",
        )
