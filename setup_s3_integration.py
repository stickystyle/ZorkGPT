#!/usr/bin/env python3
"""
Setup script to configure ZorkGPT with S3 integration

This script shows how to configure the ZorkOrchestrator to automatically
upload state files to the S3 bucket created by the CDK infrastructure.
"""

import subprocess
import json
import os
from zork_orchestrator import ZorkOrchestrator
from zork_api import ZorkInterface


def get_bucket_name_from_stack():
    """Get the S3 bucket name from the deployed CloudFormation stack."""
    try:
        result = subprocess.run(
            "aws cloudformation describe-stacks --stack-name ZorkGPTViewerStack --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' --output text",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        bucket_name = result.stdout.strip()
        if bucket_name and bucket_name != "None":
            return bucket_name
        else:
            print("❌ Could not find BucketName in stack outputs")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting bucket name: {e}")
        return None


def get_viewer_url_from_stack():
    """Get the CloudFront viewer URL from the deployed stack."""
    try:
        result = subprocess.run(
            "aws cloudformation describe-stacks --stack-name ZorkGPTViewerStack --query 'Stacks[0].Outputs[?OutputKey==`ViewerURL`].OutputValue' --output text",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        viewer_url = result.stdout.strip()
        if viewer_url and viewer_url != "None":
            return viewer_url
        else:
            print("❌ Could not find ViewerURL in stack outputs")
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting viewer URL: {e}")
        return None


def main():
    print("🚀 ZorkGPT S3 Integration Setup")
    print("=" * 40)

    # Check if infrastructure is deployed
    print("\n🔍 Checking deployed infrastructure...")
    bucket_name = get_bucket_name_from_stack()
    viewer_url = get_viewer_url_from_stack()

    if not bucket_name:
        print("\n❌ No ZorkGPT infrastructure found!")
        print("Please deploy the infrastructure first:")
        print("  cd infrastructure")
        print("  python deploy.py")
        return

    print(f"✅ Found S3 bucket: {bucket_name}")
    if viewer_url:
        print(f"✅ Found viewer URL: {viewer_url}")

    # Show configuration options
    print("\n📋 Configuration Options:")
    print("\n1. Environment Variable (Recommended):")
    print(f"   export ZORK_S3_BUCKET={bucket_name}")

    print("\n2. Constructor Parameter:")
    print("   orchestrator = ZorkOrchestrator(")
    print(f"       s3_bucket='{bucket_name}',")
    print("       enable_state_export=True")
    print("   )")

    # Offer to set environment variable
    print(f"\n🔧 Would you like to set the environment variable now? (y/n): ", end="")
    response = input().lower().strip()

    if response in ["y", "yes"]:
        # Set environment variable for current session
        os.environ["ZORK_S3_BUCKET"] = bucket_name
        print(f"✅ Environment variable set for this session")

        # Show how to make it permanent
        print("\n💡 To make this permanent, add this to your shell profile:")
        print(f"   echo 'export ZORK_S3_BUCKET={bucket_name}' >> ~/.bashrc")
        print("   # or ~/.zshrc for zsh users")

    # Offer to run a test episode
    print(f"\n🎮 Would you like to run a test episode now? (y/n): ", end="")
    response = input().lower().strip()

    if response in ["y", "yes"]:
        print("\n🎮 Starting test episode...")

        # Create orchestrator with S3 configuration
        orchestrator = ZorkOrchestrator(
            s3_bucket=bucket_name,
            enable_state_export=True,
            max_turns_per_episode=10,  # Short test episode
        )

        try:
            with ZorkInterface(timeout=1.0) as zork_game:
                episode_experiences, final_score = orchestrator.play_episode(zork_game)
                print(f"\n✅ Test episode completed!")
                print(f"   Final score: {final_score}")
                print(f"   Turns taken: {orchestrator.turn_count}")

                if viewer_url:
                    print(f"\n🌐 View your live data at: {viewer_url}")

        except Exception as e:
            print(f"\n❌ Test episode failed: {e}")

    print("\n🎉 Setup complete!")
    if viewer_url:
        print(f"🌐 Your live viewer: {viewer_url}")
    print("\n📝 Next steps:")
    print("1. Run your ZorkGPT episodes normally")
    print("2. State files will automatically upload to S3")
    print("3. View live progress in your browser")


if __name__ == "__main__":
    main()
