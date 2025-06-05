"""
AWS Lambda function for generating ZorkGPT episode index.

This function scans S3 snapshots to build an index of all available episodes,
providing metadata like final score, turn count, duration, and status.
"""

import json
import boto3
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3_client = boto3.client('s3')


def lambda_handler(event, context):
    """
    Main Lambda handler for episode index generation.
    
    Returns:
        dict: API Gateway response with episode index data
    """
    try:
        # Get bucket name from environment or event
        bucket_name = event.get('queryStringParameters', {}).get('bucket') if event.get('queryStringParameters') else None
        if not bucket_name:
            bucket_name = context.get('bucket_name') or 'zorkgpt-viewer-bucket'
        
        logger.info(f"Generating episode index for bucket: {bucket_name}")
        
        # Get episode list from S3
        episodes = get_episode_list(bucket_name)
        
        # Build response
        response_body = {
            'episodes': episodes,
            'total_count': len(episodes),
            'generated_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'max-age=300'  # Cache for 5 minutes
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        logger.error(f"Error generating episode index: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }


def get_episode_list(bucket_name: str) -> List[Dict[str, Any]]:
    """
    Scan S3 snapshots directory to build episode index.
    
    Args:
        bucket_name: S3 bucket containing snapshots
        
    Returns:
        List of episode metadata dictionaries
    """
    episodes = []
    
    try:
        # List all episode directories in snapshots/
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix='snapshots/',
            Delimiter='/'
        )
        
        episode_ids = set()
        
        # Extract episode IDs from S3 directory structure
        for page in page_iterator:
            for prefix_info in page.get('CommonPrefixes', []):
                prefix = prefix_info['Prefix']
                # Extract episode ID from path like 'snapshots/2025-06-03T13:01:40/'
                episode_match = re.match(r'snapshots/([^/]+)/', prefix)
                if episode_match:
                    episode_ids.add(episode_match.group(1))
        
        logger.info(f"Found {len(episode_ids)} episodes")
        
        # Get metadata for each episode
        for episode_id in sorted(episode_ids, reverse=True):  # Most recent first
            try:
                episode_data = get_episode_metadata(bucket_name, episode_id)
                if episode_data:
                    episodes.append(episode_data)
            except Exception as e:
                logger.warning(f"Failed to get metadata for episode {episode_id}: {str(e)}")
                # Include episode even if we can't get full metadata
                episodes.append({
                    'episode_id': episode_id,
                    'status': 'unknown',
                    'error': str(e)
                })
        
    except Exception as e:
        logger.error(f"Error scanning S3 for episodes: {str(e)}")
        raise
    
    return episodes


def get_episode_metadata(bucket_name: str, episode_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from the final snapshot of an episode.
    
    Args:
        bucket_name: S3 bucket name
        episode_id: Episode identifier
        
    Returns:
        Episode metadata dictionary or None if not found
    """
    try:
        # Find the highest turn number for this episode
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f'snapshots/{episode_id}/turn_',
            MaxKeys=1000
        )
        
        if 'Contents' not in response:
            logger.warning(f"No snapshots found for episode {episode_id}")
            return None
        
        # Extract turn numbers and find the highest
        turn_files = []
        for obj in response['Contents']:
            key = obj['Key']
            turn_match = re.search(r'turn_(\d+)\.json$', key)
            if turn_match:
                turn_number = int(turn_match.group(1))
                turn_files.append((turn_number, key, obj['LastModified']))
        
        if not turn_files:
            logger.warning(f"No valid turn files found for episode {episode_id}")
            return None
        
        # Get the final turn snapshot
        final_turn, final_key, last_modified = max(turn_files, key=lambda x: x[0])
        
        # Download and parse the final snapshot
        obj_response = s3_client.get_object(Bucket=bucket_name, Key=final_key)
        snapshot_data = json.loads(obj_response['Body'].read().decode('utf-8'))
        
        # Extract metadata
        metadata = snapshot_data.get('metadata', {})
        current_state = snapshot_data.get('current_state', {})
        
        # Calculate episode duration
        start_time = parse_episode_start_time(episode_id)
        duration_minutes = None
        if start_time and last_modified:
            # Convert last_modified to naive datetime for comparison
            end_time = last_modified.replace(tzinfo=None)
            duration = end_time - start_time
            duration_minutes = round(duration.total_seconds() / 60, 1)
        
        # Determine episode status
        status = determine_episode_status(snapshot_data, final_turn)
        
        return {
            'episode_id': episode_id,
            'start_time': episode_id,  # Episode ID is the start timestamp
            'final_turn': final_turn,
            'score': metadata.get('score', 0),
            'max_turns': metadata.get('max_turns', 0),
            'location': current_state.get('location', 'Unknown'),
            'death_count': current_state.get('death_count', 0),
            'status': status,
            'duration_minutes': duration_minutes,
            'last_modified': last_modified.isoformat(),
            'models': metadata.get('models', {}),
            'completion_percent': round((final_turn / metadata.get('max_turns', 1)) * 100, 1) if metadata.get('max_turns') else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting metadata for episode {episode_id}: {str(e)}")
        return None


def parse_episode_start_time(episode_id: str) -> Optional[datetime]:
    """
    Parse episode start time from episode ID.
    
    Args:
        episode_id: Episode ID in format YYYY-MM-DDTHH:MM:SS
        
    Returns:
        datetime object or None if parsing fails
    """
    try:
        return datetime.strptime(episode_id, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        logger.warning(f"Could not parse episode start time from ID: {episode_id}")
        return None


def determine_episode_status(snapshot_data: Dict, final_turn: int) -> str:
    """
    Determine episode status based on final snapshot data.
    
    Args:
        snapshot_data: Final snapshot data
        final_turn: Final turn number
        
    Returns:
        Status string ('completed', 'in_progress', 'died', 'abandoned')
    """
    metadata = snapshot_data.get('metadata', {})
    current_state = snapshot_data.get('current_state', {})
    
    max_turns = metadata.get('max_turns', 0)
    death_count = current_state.get('death_count', 0)
    score = metadata.get('score', 0)
    
    # Check if episode reached max turns
    if max_turns > 0 and final_turn >= max_turns:
        if score > 300:  # High score indicates good completion
            return 'completed'
        else:
            return 'max_turns_reached'
    
    # Check if player died multiple times (likely abandoned)
    if death_count >= 3:
        return 'died'
    
    # Check if it's a very short episode (likely abandoned)
    if final_turn < 10:
        return 'abandoned'
    
    # Check score-based completion (Zork typically ends around 350 points)
    if score >= 350:
        return 'completed'
    
    # If recent activity, consider in progress
    # (This would need timestamp checking in a real implementation)
    return 'in_progress'


def get_episode_summary_stats(episodes: List[Dict]) -> Dict[str, Any]:
    """
    Generate summary statistics for episode list.
    
    Args:
        episodes: List of episode metadata
        
    Returns:
        Summary statistics dictionary
    """
    if not episodes:
        return {}
    
    total_episodes = len(episodes)
    total_turns = sum(ep.get('final_turn', 0) for ep in episodes)
    avg_turns = round(total_turns / total_episodes, 1) if total_episodes > 0 else 0
    
    scores = [ep.get('score', 0) for ep in episodes if ep.get('score', 0) > 0]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    max_score = max(scores) if scores else 0
    
    status_counts = {}
    for episode in episodes:
        status = episode.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        'total_episodes': total_episodes,
        'total_turns': total_turns,
        'average_turns': avg_turns,
        'average_score': avg_score,
        'max_score': max_score,
        'status_distribution': status_counts
    }