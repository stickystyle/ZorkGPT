#!/usr/bin/env python3
"""
Simple script to update the episode index for ZorkGPT viewer.

This script can be run periodically (e.g., via cron) to keep the episode
index up to date with new episodes.
"""

import sys
import os
from generate_episode_index import EpisodeIndexGenerator

def main():
    # Configuration - adjust these as needed
    S3_BUCKET = None  # Set to your S3 bucket name if using S3
    S3_PREFIX = ""    # Set to your S3 key prefix if using S3
    LOCAL_SNAPSHOTS_DIR = "./zorkgpt/snapshots"
    OUTPUT_FILE = "./zorkgpt/episodes.json"
    
    # Check if S3 bucket is provided via environment variable
    if 'ZORKGPT_S3_BUCKET' in os.environ:
        S3_BUCKET = os.environ['ZORKGPT_S3_BUCKET']
        print(f"Using S3 bucket from environment: {S3_BUCKET}")
    
    if 'ZORKGPT_S3_PREFIX' in os.environ:
        S3_PREFIX = os.environ['ZORKGPT_S3_PREFIX']
        print(f"Using S3 prefix from environment: {S3_PREFIX}")
    
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
        
        # Save index
        generator.save_index(index, OUTPUT_FILE)
        
        print(f"Episode index updated successfully!")
        print(f"Index file: {OUTPUT_FILE}")
        
        if index['episodes']:
            print(f"Latest episode: {index['episodes'][0]['episode_id']}")
            print(f"Total turns across all episodes: {sum(ep['total_turns'] for ep in index['episodes'])}")
        
        return 0
        
    except Exception as e:
        print(f"Error updating episode index: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())