#!/bin/bash

# Update monitoring on existing ZorkGPT EC2 instance
# Usage: ./update_monitoring.sh <instance-ip> [alert-topic-arn]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <instance-ip> [alert-topic-arn]"
    echo "Example: $0 1.2.3.4 arn:aws:sns:us-east-1:123456789:ZorkGPTAlerts"
    exit 1
fi

INSTANCE_IP=$1
ALERT_TOPIC_ARN=${2:-""}
SSH_KEY="~/.ssh/parrishfamily.pem"
SSH_USER="ec2-user"

echo "🔧 Updating monitoring on ZorkGPT instance: $INSTANCE_IP"

# Create temporary script to upload and run
cat > /tmp/update_monitoring_remote.sh << 'EOF'
#!/bin/bash

echo "📦 Installing CloudWatch Agent if not present..."
if ! command -v amazon-cloudwatch-agent-ctl &> /dev/null; then
    sudo dnf install -y amazon-cloudwatch-agent
fi

echo "⚙️ Updating CloudWatch Agent configuration..."
sudo cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'CWEOF'
{
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
}
CWEOF

echo "🔄 Restarting CloudWatch Agent..."
sudo systemctl enable amazon-cloudwatch-agent
sudo systemctl restart amazon-cloudwatch-agent

echo "📊 Updating monitoring script..."
cat > /home/zorkgpt/monitor.py << 'MONEOF'
import json
import boto3
import datetime
import subprocess
import os
import sys

def log_to_journal(message, priority="info"):
    """Log to systemd journal"""
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
MONEOF

sudo chown zorkgpt:zorkgpt /home/zorkgpt/monitor.py

echo "⏰ Setting up cron job..."
echo '*/5 * * * * cd /home/zorkgpt && source ~/.bashrc && /home/zorkgpt/.local/bin/uv run python monitor.py' | sudo -u zorkgpt crontab -

echo "📝 Creating log viewing scripts..."
cat > /home/zorkgpt/view_logs.sh << 'VIEWEOF'
#!/bin/bash
echo '=== ZorkGPT Service Logs (last 20 lines) ==='
journalctl -u zorkgpt --no-pager -n 20
echo ''
echo '=== Monitor Logs (last 10 checks) ==='
journalctl -t zorkgpt-monitor --no-pager -n 10
echo ''
echo '=== Current Service Status ==='
systemctl status zorkgpt --no-pager
VIEWEOF

cat > /home/zorkgpt/follow_logs.sh << 'FOLLOWEOF'
#!/bin/bash
echo 'Following ZorkGPT logs... (Ctrl+C to exit)'
echo 'Service logs in one terminal, monitor logs in another'
echo ''
if [ "$1" = "monitor" ]; then
    journalctl -t zorkgpt-monitor -f
else
    journalctl -u zorkgpt -f
fi
FOLLOWEOF

sudo chmod +x /home/zorkgpt/view_logs.sh /home/zorkgpt/follow_logs.sh
sudo chown zorkgpt:zorkgpt /home/zorkgpt/view_logs.sh /home/zorkgpt/follow_logs.sh

echo "✅ Monitoring update completed!"
echo "📊 CloudWatch Agent status:"
sudo systemctl status amazon-cloudwatch-agent --no-pager -l

echo "🔍 Monitor logs:"
sudo journalctl -t zorkgpt-monitor --no-pager -n 5

EOF

# Upload and run the script
echo "📤 Uploading update script..."
scp -i $SSH_KEY /tmp/update_monitoring_remote.sh $SSH_USER@$INSTANCE_IP:/tmp/

echo "🚀 Running update on remote instance..."
ssh -i $SSH_KEY $SSH_USER@$INSTANCE_IP "chmod +x /tmp/update_monitoring_remote.sh && /tmp/update_monitoring_remote.sh"

# Update environment variable if provided
if [ -n "$ALERT_TOPIC_ARN" ]; then
    echo "🔔 Setting ALERT_TOPIC_ARN environment variable..."
    ssh -i $SSH_KEY $SSH_USER@$INSTANCE_IP "echo 'export ALERT_TOPIC_ARN=$ALERT_TOPIC_ARN' | sudo tee -a /home/zorkgpt/.bashrc"
fi

echo "✅ Monitoring update completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Subscribe to SNS alerts: aws sns subscribe --topic-arn $ALERT_TOPIC_ARN --protocol email --notification-endpoint your-email@example.com"
echo "2. View logs: ssh -i $SSH_KEY $SSH_USER@$INSTANCE_IP 'sudo /home/zorkgpt/view_logs.sh'"
echo "3. Follow logs: ssh -i $SSH_KEY $SSH_USER@$INSTANCE_IP 'sudo /home/zorkgpt/follow_logs.sh'"

# Clean up
rm /tmp/update_monitoring_remote.sh 