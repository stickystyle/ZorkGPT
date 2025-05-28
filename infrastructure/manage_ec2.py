#!/usr/bin/env python3
"""
EC2 Management script for ZorkGPT instance

This script helps manage the ZorkGPT EC2 instance remotely.
"""

import subprocess
import json
import sys
import argparse
import os
from datetime import datetime
from typing import Optional


def get_stack_output(output_key: str) -> Optional[str]:
    """Get a specific output from the CloudFormation stack."""
    try:
        result = subprocess.run(
            f"aws cloudformation describe-stacks --stack-name ZorkGPTViewerStack --query 'Stacks[0].Outputs[?OutputKey==`{output_key}`].OutputValue' --profile parrishfamily --output text",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        value = result.stdout.strip()
        return value if value and value != "None" else None
    except subprocess.CalledProcessError:
        return None


def run_ssh_command(command: str, public_ip: str, capture_output: bool = True) -> bool:
    """Run a command on the EC2 instance via SSH."""
    ssh_cmd = f"ssh -i ~/.ssh/parrishfamily.pem -o StrictHostKeyChecking=no ec2-user@{public_ip} '{command}'"
    
    try:
        if capture_output:
            result = subprocess.run(ssh_cmd, shell=True, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"⚠️ stderr: {result.stderr}")
            return result.returncode == 0
        else:
            # For interactive commands like logs-follow, don't capture output
            result = subprocess.run(ssh_cmd, shell=True)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ SSH command failed: {e}")
        if capture_output and hasattr(e, 'stdout') and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and hasattr(e, 'stderr') and e.stderr:
            print(f"stderr: {e.stderr}")
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

    # When following logs, don't capture output to allow real-time streaming
    run_ssh_command(f"sudo journalctl -u zorkgpt {follow_flag} --no-pager", public_ip, capture_output=not follow)


def update_zorkgpt(public_ip: str) -> None:
    """Update ZorkGPT to the latest version."""
    print("📥 Updating ZorkGPT...")

    # Get CloudFront distribution ID for cache invalidation
    distribution_id = get_stack_output("DistributionId")
    
    # Get S3 bucket name for uploads
    bucket_name = get_stack_output("BucketName")
    if not bucket_name:
        print("⚠️  Warning: Could not get S3 bucket name from CloudFormation stack")

    commands = [
        "sudo systemctl stop zorkgpt",
        'sudo -u zorkgpt bash -c "cd /home/zorkgpt/ZorkGPT && git pull"',
        'sudo -u zorkgpt bash -c "cd /home/zorkgpt/ZorkGPT && ~/.local/bin/uv sync --extra s3"',
        f'sudo -u zorkgpt bash -c "cd /home/zorkgpt/ZorkGPT && aws s3 cp zork_viewer.html s3://{bucket_name or "BUCKET_NAME_NOT_FOUND"}/zork_viewer.html"',
        "sudo systemctl start zorkgpt",
    ]

    step_descriptions = [
        "Stopping ZorkGPT service",
        "Pulling latest code from git",
        "Syncing Python dependencies with S3 support",
        "Uploading viewer HTML to S3",
        "Starting ZorkGPT service",
    ]

    for i, (command, description) in enumerate(zip(commands, step_descriptions), 1):
        print(f"Step {i}/{len(commands)}: {description}")
        if not run_ssh_command(command, public_ip):
            # Special handling for the HTML upload step - don't fail if upload fails
            if "aws s3 cp zork_viewer.html" in command:
                print("⚠️  zork_viewer.html upload failed, continuing anyway (check bucket permissions)")
                continue
            else:
                print(f"❌ Update failed at step {i}")
                return

    # Invalidate CloudFront cache for the HTML file
    if distribution_id:
        print("🔄 Invalidating CloudFront cache for viewer HTML...")
        invalidation_cmd = f"aws cloudfront create-invalidation --distribution-id {distribution_id} --paths '/zork_viewer.html' --profile parrishfamily"
        try:
            result = subprocess.run(invalidation_cmd, shell=True, check=True, capture_output=True, text=True)
            print("✅ CloudFront cache invalidation initiated")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  CloudFront invalidation failed: {e}")
            print("   The HTML file was uploaded but cache may take time to refresh")
    else:
        print("⚠️  Could not get CloudFront distribution ID for cache invalidation")

    print("✅ ZorkGPT updated successfully")


def update_viewer_only(public_ip: str) -> None:
    """Update only the viewer HTML file without restarting ZorkGPT."""
    print("🌐 Updating viewer HTML file...")

    # Get CloudFront distribution ID for cache invalidation
    distribution_id = get_stack_output("DistributionId")
    
    # Get S3 bucket name for uploads
    bucket_name = get_stack_output("BucketName")
    if not bucket_name:
        print("❌ Could not get S3 bucket name from CloudFormation stack")
        return

    # Pull latest code to get updated HTML file
    print("📥 Pulling latest code from git...")
    pull_cmd = 'sudo -u zorkgpt bash -c "cd /home/zorkgpt/ZorkGPT && git pull"'
    
    if not run_ssh_command(pull_cmd, public_ip):
        print("❌ Failed to pull latest code from git")
        return

    print("✅ Latest code pulled from git")

    # Upload the viewer HTML file
    print("📤 Uploading viewer HTML to S3...")
    upload_cmd = f'sudo -u zorkgpt bash -c "cd /home/zorkgpt/ZorkGPT && aws s3 cp zork_viewer.html s3://{bucket_name}/zork_viewer.html"'
    
    if not run_ssh_command(upload_cmd, public_ip):
        print("❌ Failed to upload zork_viewer.html to S3")
        return

    print("✅ Viewer HTML uploaded to S3")

    # Invalidate CloudFront cache for the HTML file
    if distribution_id:
        print("🔄 Invalidating CloudFront cache for viewer HTML...")
        invalidation_cmd = f"aws cloudfront create-invalidation --distribution-id {distribution_id} --paths '/zork_viewer.html' --profile parrishfamily"
        try:
            result = subprocess.run(invalidation_cmd, shell=True, check=True, capture_output=True, text=True)
            print("✅ CloudFront cache invalidation initiated")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  CloudFront invalidation failed: {e}")
            print("   The HTML file was uploaded but cache may take time to refresh")
    else:
        print("⚠️  Could not get CloudFront distribution ID for cache invalidation")

    print("🎉 Viewer update completed successfully!")
    print("💡 Note: ZorkGPT service was not restarted - only the viewer HTML was updated")


def download_analysis_files(public_ip: str) -> None:
    """Download analysis files from the EC2 instance."""
    print("📥 Downloading analysis files from EC2 instance...")
    
    # Create timestamped directory for analysis files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = f"analysis_{timestamp}"
    
    try:
        os.makedirs(analysis_dir, exist_ok=True)
        print(f"📁 Created analysis directory: {analysis_dir}")
    except OSError as e:
        print(f"❌ Failed to create analysis directory: {e}")
        return
    
    # Files to download from the ZorkGPT working directory
    files_to_download = [
        "current_state.json",
        "knowledgebase.md", 
        "zork_episode_log.jsonl"
    ]
    
    success_count = 0
    
    for filename in files_to_download:
        print(f"📄 Downloading {filename}...")
        
        # First, copy the file to a temporary location accessible by ec2-user
        # Use sudo -u zorkgpt to access the file as the zorkgpt user
        temp_path = f"/tmp/{filename}"
        copy_cmd = f"sudo -u zorkgpt cp /home/zorkgpt/ZorkGPT/{filename} {temp_path} && sudo chown ec2-user:ec2-user {temp_path}"
        
        # Copy file to temp location on EC2
        if not run_ssh_command(copy_cmd, public_ip):
            print(f"⚠️  Failed to copy {filename} to temporary location")
            continue
        
        # Use scp to download the file from temp location
        scp_cmd = f"scp -i ~/.ssh/parrishfamily.pem -o StrictHostKeyChecking=no ec2-user@{public_ip}:{temp_path} {analysis_dir}/"
        
        try:
            result = subprocess.run(scp_cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ Downloaded {filename}")
            success_count += 1
            
            # Clean up temp file on EC2
            cleanup_cmd = f"rm -f {temp_path}"
            run_ssh_command(cleanup_cmd, public_ip)
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to download {filename}: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"   Error details: {e.stderr}")
            
            # Still try to clean up temp file
            cleanup_cmd = f"rm -f {temp_path}"
            run_ssh_command(cleanup_cmd, public_ip)
    
    if success_count > 0:
        print(f"\n🎉 Downloaded {success_count}/{len(files_to_download)} files to {analysis_dir}/")
        print(f"📊 Analysis files ready for review:")
        
        # List what was actually downloaded
        for filename in files_to_download:
            filepath = os.path.join(analysis_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"   ✓ {filename} ({file_size:,} bytes)")
            else:
                print(f"   ✗ {filename} (not downloaded)")
    else:
        print("❌ No files were successfully downloaded")
        # Clean up empty directory
        try:
            os.rmdir(analysis_dir)
        except OSError:
            pass


def ssh_connect(public_ip: str) -> None:
    """Open an interactive SSH session."""
    print(f"🔗 Connecting to EC2 instance...")
    ssh_cmd = f"ssh -i ~/.ssh/parrishfamily.pem ec2-user@{public_ip}"
    subprocess.run(ssh_cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Manage ZorkGPT EC2 instance")
    parser.add_argument(
        "action",
        choices=[
            "status",
            "start",
            "stop",
            "restart",
            "logs",
            "logs-follow",
            "update",
            "update-viewer",
            "download",
            "ssh",
            "info",
        ],
        help="Action to perform",
    )

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

    if args.action == "info":
        print(f"📍 Instance ID: {instance_id}")
        print(f"🌐 Public IP: {public_ip}")
        print(f"🔗 SSH Command: ssh -i ~/.ssh/parrishfamily.pem ec2-user@{public_ip}")
        if viewer_url:
            print(f"👁️  Viewer URL: {viewer_url}")
        return

    print(f"🎯 Target: {public_ip} ({instance_id})")

    # Execute the requested action
    if args.action == "status":
        get_instance_status(public_ip)
    elif args.action == "start":
        start_zorkgpt(public_ip)
    elif args.action == "stop":
        stop_zorkgpt(public_ip)
    elif args.action == "restart":
        restart_zorkgpt(public_ip)
    elif args.action == "logs":
        view_logs(public_ip, follow=False)
    elif args.action == "logs-follow":
        view_logs(public_ip, follow=True)
    elif args.action == "update":
        update_zorkgpt(public_ip)
    elif args.action == "update-viewer":
        update_viewer_only(public_ip)
    elif args.action == "download":
        download_analysis_files(public_ip)
    elif args.action == "ssh":
        ssh_connect(public_ip)


if __name__ == "__main__":
    main()
