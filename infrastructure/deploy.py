#!/usr/bin/env python3
"""
Deployment script for ZorkGPT Viewer Infrastructure

This script helps deploy the S3 bucket and CloudFront distribution
for hosting the ZorkGPT Live Viewer.
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(command, description, check=True, env=None):
    """Run a command and handle errors gracefully."""
    print(f"\n🔄 {description}...")
    print(f"Running: {command}")

    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True, env=env
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr and result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False

        print(f"✅ {description} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def check_prerequisites():
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")

    # Check if AWS CLI is installed and configured
    if not run_command("aws --version", "Checking AWS CLI", check=False):
        print("❌ AWS CLI is not installed. Please install it first:")
        print(
            "   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        )
        return False

    # Check if AWS credentials are configured
    if not run_command(
        "aws sts get-caller-identity", "Checking AWS credentials", check=False
    ):
        print("❌ AWS credentials not configured. Please run 'aws configure' first.")
        return False

    # Check if Node.js is installed (required for CDK)
    if not run_command("node --version", "Checking Node.js", check=False):
        print("❌ Node.js is not installed. Please install it first:")
        print("   https://nodejs.org/")
        return False

    # Check if CDK is installed
    if not run_command("cdk --version", "Checking CDK", check=False):
        print("❌ AWS CDK is not installed. Installing it now...")
        if not run_command("npm install -g aws-cdk", "Installing AWS CDK"):
            return False

    print("✅ All prerequisites are satisfied")
    return True


def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    print("\n🐍 Setting up Python environment...")

    # Create virtual environment if it doesn't exist
    if not Path(".venv").exists():
        if not run_command("python -m venv .venv", "Creating virtual environment"):
            return False

    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        activate_cmd = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
        python_cmd = ".venv\\Scripts\\python"
    else:
        activate_cmd = "source .venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
        python_cmd = ".venv/bin/python"

    if not run_command(
        f"{pip_cmd} install -r requirements.txt", "Installing Python dependencies"
    ):
        return False

    print("✅ Python environment setup complete")
    return True


def get_venv_python():
    """Get the path to the virtual environment's Python interpreter."""
    if sys.platform == "win32":
        return ".venv\\Scripts\\python"
    else:
        return ".venv/bin/python"


def get_venv_environment():
    """Get environment variables configured to use the virtual environment."""
    env = os.environ.copy()
    if sys.platform == "win32":
        venv_path = Path(".venv/Scripts").absolute()
    else:
        venv_path = Path(".venv/bin").absolute()

    env["PATH"] = (
        f"{venv_path}:{env['PATH']}"
        if sys.platform != "win32"
        else f"{venv_path};{env['PATH']}"
    )
    return env


def bootstrap_cdk():
    """Bootstrap CDK in the current AWS account/region."""
    print("\n🚀 Bootstrapping CDK...")

    # Get current AWS account and region
    try:
        result = subprocess.run(
            "aws sts get-caller-identity --query Account --output text",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        account = result.stdout.strip()

        result = subprocess.run(
            "aws configure get region",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        region = result.stdout.strip()

        print(f"Account: {account}")
        print(f"Region: {region}")

    except subprocess.CalledProcessError:
        print("❌ Could not determine AWS account/region")
        return False

    # Bootstrap CDK using virtual environment
    bootstrap_cmd = f"cdk bootstrap aws://{account}/{region}"

    # Update the environment to use our virtual environment python
    env = get_venv_environment()
    env["CDK_DEFAULT_ACCOUNT"] = account
    env["CDK_DEFAULT_REGION"] = region

    if not run_command(bootstrap_cmd, "Bootstrapping CDK", env=env):
        print("ℹ️  CDK might already be bootstrapped, continuing...")

    return True


def deploy_stack():
    """Deploy the ZorkGPT Viewer stack."""
    print("\n🚀 Deploying ZorkGPT Viewer stack...")

    # Update the environment to use our virtual environment python
    env = get_venv_environment()

    # Synthesize the stack first
    if not run_command("cdk synth", "Synthesizing CDK stack", env=env):
        return False

    # Deploy the stack
    if not run_command(
        "cdk deploy --require-approval never", "Deploying stack", env=env
    ):
        return False

    print("✅ Stack deployed successfully!")
    return True


def get_stack_outputs():
    """Get and display the stack outputs."""
    print("\n📋 Getting stack outputs...")

    try:
        result = subprocess.run(
            "aws cloudformation describe-stacks --stack-name ZorkGPTViewerStack --query 'Stacks[0].Outputs'",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )

        outputs = json.loads(result.stdout)

        print("\n🎉 Deployment complete! Here are your resources:")
        print("=" * 60)

        for output in outputs:
            key = output["OutputKey"]
            value = output["OutputValue"]
            description = output.get("Description", "")

            print(f"\n{key}:")
            print(f"  Value: {value}")
            if description:
                print(f"  Description: {description}")

        print("\n" + "=" * 60)
        print("\n📝 Next steps:")
        print("1. Wait 2-3 minutes for EC2 instance to finish setup")
        print("2. SSH to your instance using the EC2SSHCommand")
        print("3. ZorkGPT will start automatically and upload to S3")
        print("4. Access your live viewer at the ViewerURL")
        print("\n💡 Tip: Use 'sudo journalctl -u zorkgpt -f' to watch ZorkGPT logs")

        return True

    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"❌ Could not get stack outputs: {e}")
        return False


def main():
    """Main deployment function."""
    print("🚀 ZorkGPT Viewer Infrastructure Deployment")
    print("=" * 50)

    # Change to infrastructure directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Run deployment steps
    steps = [
        ("Checking prerequisites", check_prerequisites),
        ("Setting up Python environment", setup_python_environment),
        ("Bootstrapping CDK", bootstrap_cdk),
        ("Deploying stack", deploy_stack),
        ("Getting stack outputs", get_stack_outputs),
    ]

    for step_name, step_func in steps:
        print(f"\n{'=' * 20} {step_name} {'=' * 20}")
        if not step_func():
            print(f"\n❌ Deployment failed at step: {step_name}")
            sys.exit(1)

    print("\n🎉 Deployment completed successfully!")
    print("Your ZorkGPT Live Viewer infrastructure is now ready!")


if __name__ == "__main__":
    main()
