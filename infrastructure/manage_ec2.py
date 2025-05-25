#!/usr/bin/env python3
"""
EC2 Management script for ZorkGPT instance

This script helps manage the ZorkGPT EC2 instance remotely.
"""

import subprocess
import json
import sys
import argparse
from typing import Optional


def get_stack_output(output_key: str) -> Optional[str]:
    """Get a specific output from the CloudFormation stack."""
    try:
        result = subprocess.run(
            f"aws cloudformation describe-stacks --stack-name ZorkGPTViewerStack --query 'Stacks[0].Outputs[?OutputKey==`{output_key}`].OutputValue' --output text",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        value = result.stdout.strip()
        return value if value and value != "None" else None
    except subprocess.CalledProcessError:
        return None


def run_ssh_command(command: str, public_ip: str) -> bool:
    """Run a command on the EC2 instance via SSH."""
    ssh_cmd = f"ssh -i ~/.ssh/parrishfamily.pem -o StrictHostKeyChecking=no ec2-user@{public_ip} '{command}'"
    
    try:
        result = subprocess.run(ssh_cmd, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ SSH command failed: {e}")
        return False


def get_instance_status(public_ip: str) -> None:
    """Get the status of ZorkGPT on the EC2 instance."""
    print("🔍 Checking ZorkGPT status...")
    
    commands = [
        ("System Status", "sudo systemctl is-active zorkgpt || echo 'inactive'"),
        ("Service Status", "sudo systemctl status zorkgpt --no-pager -l"),
        ("Recent Logs", "sudo journalctl -u zorkgpt --no-pager -n 10"),
    ]
    
    for description, command in commands:
        print(f"\n📋 {description}:")
        print("-" * 40)
        run_ssh_command(command, public_ip)


def start_zorkgpt(public_ip: str) -> None:
    """Start the ZorkGPT service."""
    print("🚀 Starting ZorkGPT service...")
    if run_ssh_command("sudo systemctl start zorkgpt", public_ip):
        print("✅ ZorkGPT service started")
    else:
        print("❌ Failed to start ZorkGPT service")


def stop_zorkgpt(public_ip: str) -> None:
    """Stop the ZorkGPT service."""
    print("🛑 Stopping ZorkGPT service...")
    if run_ssh_command("sudo systemctl stop zorkgpt", public_ip):
        print("✅ ZorkGPT service stopped")
    else:
        print("❌ Failed to stop ZorkGPT service")


def restart_zorkgpt(public_ip: str) -> None:
    """Restart the ZorkGPT service."""
    print("🔄 Restarting ZorkGPT service...")
    if run_ssh_command("sudo systemctl restart zorkgpt", public_ip):
        print("✅ ZorkGPT service restarted")
    else:
        print("❌ Failed to restart ZorkGPT service")


def view_logs(public_ip: str, follow: bool = False) -> None:
    """View ZorkGPT logs."""
    follow_flag = "-f" if follow else ""
    print(f"📜 Viewing ZorkGPT logs{'(following)' if follow else ''}...")
    
    if follow:
        print("Press Ctrl+C to stop following logs")
    
    run_ssh_command(f"sudo journalctl -u zorkgpt {follow_flag} --no-pager", public_ip)


def update_zorkgpt(public_ip: str) -> None:
    """Update ZorkGPT to the latest version."""
    print("📥 Updating ZorkGPT...")
    
    commands = [
        "sudo systemctl stop zorkgpt",
        "cd /home/zorkgpt/ZorkGPT && sudo -u zorkgpt git pull",
        "cd /home/zorkgpt/ZorkGPT && sudo -u zorkgpt /home/zorkgpt/.cargo/bin/uv sync",
        "sudo systemctl start zorkgpt"
    ]
    
    for i, command in enumerate(commands, 1):
        print(f"Step {i}/{len(commands)}: {command.split('&&')[-1].strip()}")
        if not run_ssh_command(command, public_ip):
            print(f"❌ Update failed at step {i}")
            return
    
    print("✅ ZorkGPT updated successfully")


def ssh_connect(public_ip: str) -> None:
    """Open an interactive SSH session."""
    print(f"🔗 Connecting to EC2 instance...")
    ssh_cmd = f"ssh -i ~/.ssh/parrishfamily.pem ec2-user@{public_ip}"
    subprocess.run(ssh_cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description='Manage ZorkGPT EC2 instance')
    parser.add_argument('action', choices=[
        'status', 'start', 'stop', 'restart', 'logs', 'logs-follow', 
        'update', 'ssh', 'info'
    ], help='Action to perform')
    
    args = parser.parse_args()
    
    print("🖥️  ZorkGPT EC2 Management")
    print("=" * 30)
    
    # Get instance information
    public_ip = get_stack_output("EC2PublicIP")
    instance_id = get_stack_output("EC2InstanceId")
    viewer_url = get_stack_output("ViewerURL")
    
    if not public_ip:
        print("❌ Could not find EC2 instance information")
        print("Make sure the ZorkGPT infrastructure is deployed")
        sys.exit(1)
    
    if args.action == 'info':
        print(f"📍 Instance ID: {instance_id}")
        print(f"🌐 Public IP: {public_ip}")
        print(f"🔗 SSH Command: ssh -i ~/.ssh/parrishfamily.pem ec2-user@{public_ip}")
        if viewer_url:
            print(f"👁️  Viewer URL: {viewer_url}")
        return
    
    print(f"🎯 Target: {public_ip} ({instance_id})")
    
    # Execute the requested action
    if args.action == 'status':
        get_instance_status(public_ip)
    elif args.action == 'start':
        start_zorkgpt(public_ip)
    elif args.action == 'stop':
        stop_zorkgpt(public_ip)
    elif args.action == 'restart':
        restart_zorkgpt(public_ip)
    elif args.action == 'logs':
        view_logs(public_ip, follow=False)
    elif args.action == 'logs-follow':
        view_logs(public_ip, follow=True)
    elif args.action == 'update':
        update_zorkgpt(public_ip)
    elif args.action == 'ssh':
        ssh_connect(public_ip)


if __name__ == "__main__":
    main() 