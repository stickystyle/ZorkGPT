#!/usr/bin/env python3
"""
Episode Index Generator for ZorkGPT Viewer

This script generates an index of available episodes by scanning S3 snapshots
or local snapshot directories. It creates an episodes.json file that the viewer
can use to list and switch between episodes.
"""

import json
import os
import glob
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
import tomllib
from pathlib import Path

# Optional S3 support
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


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


class EpisodeIndexGenerator:
    """Generates an index of available ZorkGPT episodes."""
    
    def __init__(self, s3_bucket: Optional[str] = None, s3_key_prefix: str = "", local_snapshots_dir: str = "./zorkgpt/snapshots"):
        self.s3_bucket = s3_bucket
        self.s3_key_prefix = s3_key_prefix
        self.local_snapshots_dir = local_snapshots_dir
        self.s3_client = None
        
        print(f"S3_AVAILABLE: {S3_AVAILABLE}")
        print(f"s3_bucket provided: {s3_bucket}")
        
        if S3_AVAILABLE and s3_bucket:
            try:
                self.s3_client = boto3.client("s3")
                print(f"S3 client initialized for bucket: {s3_bucket}")
            except Exception as e:
                print(f"Failed to initialize S3 client: {e}")
        else:
            if not S3_AVAILABLE:
                print("boto3 not available - S3 scanning disabled")
            if not s3_bucket:
                print("No S3 bucket specified - S3 scanning disabled")
    
    def generate_index(self) -> Dict[str, Any]:
        """Generate episode index from available sources."""
        episodes = []
        
        # Try S3 first if available
        if self.s3_client and self.s3_bucket:
            print("Scanning S3 for episodes...")
            s3_episodes = self._scan_s3_episodes()
            episodes.extend(s3_episodes)
            print(f"Found {len(s3_episodes)} episodes in S3")
        
        # Scan local snapshots
        if os.path.exists(self.local_snapshots_dir):
            print(f"Scanning local directory: {self.local_snapshots_dir}")
            local_episodes = self._scan_local_episodes()
            episodes.extend(local_episodes)
            print(f"Found {len(local_episodes)} episodes locally")
        else:
            print(f"Local snapshots directory not found: {self.local_snapshots_dir}")
        
        # Remove duplicates and sort by timestamp
        unique_episodes = self._deduplicate_episodes(episodes)
        unique_episodes.sort(key=lambda x: x['start_time'], reverse=True)
        
        # Mark only the most recent episode as not game over (current episode)
        if unique_episodes:
            unique_episodes[0]['game_over'] = False
        
        # Generate index
        index = {
            "generated_at": datetime.now().isoformat(),
            "total_episodes": len(unique_episodes),
            "episodes": unique_episodes
        }
        
        print(f"Generated index with {len(unique_episodes)} unique episodes")
        return index
    
    def _scan_s3_episodes(self) -> List[Dict[str, Any]]:
        """Scan S3 for episode snapshots."""
        episodes = []
        
        try:
            # List objects in the snapshots prefix
            snapshots_prefix = f"{self.s3_key_prefix}snapshots/"
            print(f"Scanning S3 with prefix: {snapshots_prefix}")
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=snapshots_prefix)
            
            episode_dirs = set()
            total_objects = 0
            
            for page in pages:
                if 'Contents' not in page:
                    print("No contents in this page")
                    continue
                
                page_objects = len(page['Contents'])
                total_objects += page_objects
                print(f"Processing page with {page_objects} objects")
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Extract episode ID from path like "snapshots/2024-01-15T10:30:45/turn_1.json"
                    match = re.match(rf"{re.escape(snapshots_prefix)}([^/]+)/turn_(\d+)\.json$", key)
                    if match:
                        episode_id = match.group(1)
                        turn_num = int(match.group(2))
                        episode_dirs.add(episode_id)
                        print(f"Found episode: {episode_id}, turn: {turn_num}")
            
            print(f"Total objects scanned: {total_objects}")
            print(f"Episode directories found: {len(episode_dirs)}")
            
            # For each episode, get metadata from the first and last snapshots
            for episode_id in episode_dirs:
                print(f"Getting info for episode: {episode_id}")
                episode_info = self._get_s3_episode_info(episode_id)
                if episode_info:
                    episodes.append(episode_info)
                    print(f"Successfully added episode: {episode_id}")
                else:
                    print(f"Failed to get info for episode: {episode_id}")
                    
        except Exception as e:
            print(f"Error scanning S3 episodes: {e}")
        
        return episodes
    
    def _get_s3_episode_info(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get episode information from S3 snapshots."""
        try:
            episode_prefix = f"{self.s3_key_prefix}snapshots/{episode_id}/"
            
            # List all snapshots for this episode using pagination
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=episode_prefix)
            
            # Find turn files and extract turn numbers
            turn_files = []
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    match = re.search(r'turn_(\d+)\.json$', key)
                    if match:
                        turn_num = int(match.group(1))
                        turn_files.append({
                            'turn': turn_num,
                            'key': key,
                            'last_modified': obj['LastModified']
                        })
            
            if not turn_files:
                return None
            
            # Sort by turn number
            turn_files.sort(key=lambda x: x['turn'])
            
            # Get first snapshot for episode metadata
            first_snapshot_key = turn_files[0]['key']
            first_snapshot = self._get_s3_snapshot(first_snapshot_key)
            
            # Get last snapshot for final stats
            last_snapshot_key = turn_files[-1]['key']
            last_snapshot = self._get_s3_snapshot(last_snapshot_key)
            
            if not first_snapshot or not last_snapshot:
                return None
            
            # Extract episode information
            episode_info = {
                "episode_id": episode_id,
                "source": "s3",
                "start_time": first_snapshot.get('metadata', {}).get('timestamp', episode_id),
                "end_time": last_snapshot.get('metadata', {}).get('timestamp', ''),
                "total_turns": last_snapshot.get('metadata', {}).get('turn_count', len(turn_files)),
                "final_score": last_snapshot.get('metadata', {}).get('score', 0),
                "game_over": True,  # Will be set to False for current episode later
                "death_count": last_snapshot.get('current_state', {}).get('death_count', 0),
                "models": last_snapshot.get('metadata', {}).get('models', {}),
                "snapshot_count": len(turn_files),
                "first_turn": turn_files[0]['turn'],
                "last_turn": turn_files[-1]['turn']
            }
            
            return episode_info
            
        except Exception as e:
            print(f"Error getting S3 episode info for {episode_id}: {e}")
            return None
    
    def _get_s3_snapshot(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a snapshot from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return json.loads(content)
        except Exception as e:
            print(f"Error reading S3 snapshot {key}: {e}")
            return None
    
    def _scan_local_episodes(self) -> List[Dict[str, Any]]:
        """Scan local directory for episode snapshots."""
        episodes = []
        
        try:
            # Look for episode directories
            episode_pattern = os.path.join(self.local_snapshots_dir, "*")
            episode_dirs = [d for d in glob.glob(episode_pattern) if os.path.isdir(d)]
            
            for episode_dir in episode_dirs:
                episode_id = os.path.basename(episode_dir)
                episode_info = self._get_local_episode_info(episode_id, episode_dir)
                if episode_info:
                    episodes.append(episode_info)
                    
        except Exception as e:
            print(f"Error scanning local episodes: {e}")
        
        return episodes
    
    def _get_local_episode_info(self, episode_id: str, episode_dir: str) -> Optional[Dict[str, Any]]:
        """Get episode information from local snapshots."""
        try:
            # Find all turn files
            turn_pattern = os.path.join(episode_dir, "turn_*.json")
            turn_files = glob.glob(turn_pattern)
            
            if not turn_files:
                return None
            
            # Extract turn numbers and sort
            turn_info = []
            for turn_file in turn_files:
                match = re.search(r'turn_(\d+)\.json$', turn_file)
                if match:
                    turn_num = int(match.group(1))
                    turn_info.append({
                        'turn': turn_num,
                        'file': turn_file,
                        'mtime': os.path.getmtime(turn_file)
                    })
            
            turn_info.sort(key=lambda x: x['turn'])
            
            # Read first and last snapshots
            first_snapshot = self._read_local_snapshot(turn_info[0]['file'])
            last_snapshot = self._read_local_snapshot(turn_info[-1]['file'])
            
            if not first_snapshot or not last_snapshot:
                return None
            
            # Extract episode information
            episode_info = {
                "episode_id": episode_id,
                "source": "local",
                "start_time": first_snapshot.get('metadata', {}).get('timestamp', episode_id),
                "end_time": last_snapshot.get('metadata', {}).get('timestamp', ''),
                "total_turns": last_snapshot.get('metadata', {}).get('turn_count', len(turn_info)),
                "final_score": last_snapshot.get('metadata', {}).get('score', 0),
                "game_over": True,  # Will be set to False for current episode later
                "death_count": last_snapshot.get('current_state', {}).get('death_count', 0),
                "models": last_snapshot.get('metadata', {}).get('models', {}),
                "snapshot_count": len(turn_info),
                "first_turn": turn_info[0]['turn'],
                "last_turn": turn_info[-1]['turn']
            }
            
            return episode_info
            
        except Exception as e:
            print(f"Error getting local episode info for {episode_id}: {e}")
            return None
    
    def _read_local_snapshot(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Read a local snapshot file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading local snapshot {file_path}: {e}")
            return None
    
    def _deduplicate_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate episodes (same episode_id from different sources)."""
        seen = {}
        unique_episodes = []
        
        for episode in episodes:
            episode_id = episode['episode_id']
            
            if episode_id not in seen:
                seen[episode_id] = episode
                unique_episodes.append(episode)
            else:
                # Prefer S3 source over local if both exist
                existing = seen[episode_id]
                if episode['source'] == 's3' and existing['source'] == 'local':
                    # Replace local with S3 version
                    unique_episodes.remove(existing)
                    unique_episodes.append(episode)
                    seen[episode_id] = episode
        
        return unique_episodes
    
    def save_index(self, index: Dict[str, Any], output_file: str = "./zorkgpt/episodes.json") -> None:
        """Save the episode index to a file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Write index file
            with open(output_file, 'w') as f:
                json.dump(index, f, indent=2, default=str)
            
            print(f"Episode index saved to: {output_file}")
            
        except Exception as e:
            print(f"Error saving episode index: {e}")
    
    def upload_index_to_s3(self, index: Dict[str, Any], s3_key: str = "episodes.json") -> bool:
        """Upload the episode index to S3."""
        if not self.s3_client or not self.s3_bucket:
            print("S3 not configured - skipping S3 upload")
            return False
        
        try:
            # Prepare the full S3 key
            full_s3_key = f"{self.s3_key_prefix}{s3_key}"
            print(f"S3 key prefix: '{self.s3_key_prefix}'")
            print(f"S3 key: '{s3_key}'")
            print(f"Full S3 key: '{full_s3_key}'")
            
            # Convert index to JSON string
            index_json = json.dumps(index, indent=2, default=str)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=full_s3_key,
                Body=index_json.encode('utf-8'),
                ContentType='application/json'
            )
            
            print(f"Episode index uploaded to S3: s3://{self.s3_bucket}/{full_s3_key}")
            return True
            
        except Exception as e:
            print(f"Error uploading episode index to S3: {e}")
            return False


def main():
    # Load configuration from pyproject.toml
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Generate ZorkGPT episode index")
    parser.add_argument("--s3-bucket", help="S3 bucket name")
    parser.add_argument("--s3-prefix", default=config.get("s3_key_prefix", ""), help="S3 key prefix")
    parser.add_argument("--local-dir", default="./zorkgpt/snapshots", help="Local snapshots directory")
    parser.add_argument("--output", default="./zorkgpt/episodes.json", help="Output file path")
    parser.add_argument("--upload-s3", action="store_true", help="Upload index to S3 after generating")
    
    args = parser.parse_args()
    
    # Use config defaults if not provided via command line
    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix
    
    # Environment variables override everything
    if 'ZORKGPT_S3_BUCKET' in os.environ:
        s3_bucket = os.environ['ZORKGPT_S3_BUCKET']
        print(f"Using S3 bucket from environment: {s3_bucket}")
    
    if 'ZORKGPT_S3_PREFIX' in os.environ:
        s3_prefix = os.environ['ZORKGPT_S3_PREFIX']
        print(f"Using S3 prefix from environment: {s3_prefix}")
    elif config.get("s3_key_prefix") and args.s3_prefix == config.get("s3_key_prefix", ""):
        print(f"Using S3 prefix from config: {s3_prefix}")
    
    print(f"Final S3 prefix: '{s3_prefix}'")
    
    # Create generator
    generator = EpisodeIndexGenerator(
        s3_bucket=s3_bucket,
        s3_key_prefix=s3_prefix,
        local_snapshots_dir=args.local_dir
    )
    
    # Generate index
    print("Generating episode index...")
    index = generator.generate_index()
    
    # Save index locally
    generator.save_index(index, args.output)
    
    # Upload to S3 if requested and configured
    if args.upload_s3:
        if args.s3_bucket:
            print("Uploading episode index to S3...")
            upload_success = generator.upload_index_to_s3(index)
            if upload_success:
                print("Episode index uploaded to S3 successfully!")
            else:
                print("Warning: Failed to upload episode index to S3")
        else:
            print("Warning: --upload-s3 specified but no S3 bucket configured")
    
    # Print summary
    print(f"\nEpisode Index Summary:")
    print(f"Total episodes: {index['total_episodes']}")
    
    if index['episodes']:
        print(f"Latest episode: {index['episodes'][0]['episode_id']}")
        print(f"Date range: {index['episodes'][-1]['start_time']} to {index['episodes'][0]['start_time']}")
        
        # Show some stats
        total_turns = sum(ep['total_turns'] for ep in index['episodes'])
        total_deaths = sum(ep['death_count'] for ep in index['episodes'])
        max_score = max(ep['final_score'] for ep in index['episodes'])
        
        print(f"Total turns across all episodes: {total_turns}")
        print(f"Total deaths: {total_deaths}")
        print(f"Highest score achieved: {max_score}")


if __name__ == "__main__":
    main()