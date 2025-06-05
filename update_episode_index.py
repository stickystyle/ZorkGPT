#!/usr/bin/env python3
"""
Simple script to update the episode index for ZorkGPT viewer.

This script can be run periodically (e.g., via cron) to keep the episode
index up to date with new episodes. It generates an episodes.json file
locally and uploads it to S3 if an S3 bucket is configured, making it
available for the ZorkGPT viewer to fetch episode information.
"""

import sys
import os
import tomllib
from pathlib import Path
from generate_episode_index import EpisodeIndexGenerator

def load_config():
    """Load configuration from pyproject.toml if available."""
    config_file = Path("pyproject.toml")
    if config_file.exists():
        try:
            with open(config_file, "rb") as f:
                config = tomllib.load(f)
                return config.get("tool", {}).get("zorkgpt", {}).get("aws", {})
        except Exception as e:
            print(f"Warning: Could not load pyproject.toml: {e}")
    return {}

def main():
    # Load configuration from pyproject.toml
    config = load_config()
    
    # Configuration - adjust these as needed
    S3_BUCKET = None  # Set to your S3 bucket name if using S3
    S3_PREFIX = config.get("s3_key_prefix", "zorkgpt/")  # Default from pyproject.toml
    LOCAL_SNAPSHOTS_DIR = "./zorkgpt/snapshots"
    OUTPUT_FILE = "./zorkgpt/episodes.json"
    
    # Environment variables override config file settings
    if 'ZORKGPT_S3_BUCKET' in os.environ:
        S3_BUCKET = os.environ['ZORKGPT_S3_BUCKET']
        print(f"Using S3 bucket from environment: {S3_BUCKET}")
    
    if 'ZORKGPT_S3_PREFIX' in os.environ:
        S3_PREFIX = os.environ['ZORKGPT_S3_PREFIX']
        print(f"Using S3 prefix from environment: {S3_PREFIX}")
    elif config.get("s3_key_prefix"):
        print(f"Using S3 prefix from config: {S3_PREFIX}")
    
    # Create generator
    generator = EpisodeIndexGenerator(
        s3_bucket=S3_BUCKET,
        s3_key_prefix=S3_PREFIX,
        local_snapshots_dir=LOCAL_SNAPSHOTS_DIR
    )
    
    try:
        print("Generating episode index...")
        index = generator.generate_index()
        
        print(f"Found {index['total_episodes']} episodes")
        
        # Save index locally
        generator.save_index(index, OUTPUT_FILE)
        
        # Upload to S3 if configured
        if S3_BUCKET:
            print("Uploading episode index to S3...")
            upload_success = generator.upload_index_to_s3(index)
            if upload_success:
                print("Episode index uploaded to S3 successfully!")
            else:
                print("Warning: Failed to upload episode index to S3")
        else:
            print("No S3 bucket configured - skipping S3 upload")
        
        print(f"Episode index updated successfully!")
        print(f"Local index file: {OUTPUT_FILE}")
        
        if index['episodes']:
            print(f"Latest episode: {index['episodes'][0]['episode_id']}")
            print(f"Total turns across all episodes: {sum(ep['total_turns'] for ep in index['episodes'])}")
        
        return 0
        
    except Exception as e:
        print(f"Error updating episode index: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())