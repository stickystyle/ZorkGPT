import json
import os
import boto3
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

def handler(event, context):
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    key_prefix = os.environ.get('S3_KEY_PREFIX', '') # Default to empty string if not set

    if not bucket_name:
        logger.error("S3_BUCKET_NAME environment variable not set.")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'S3_BUCKET_NAME not configured'}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

    # Ensure the prefix ends with a slash if it's not empty
    snapshots_prefix = key_prefix
    if snapshots_prefix and not snapshots_prefix.endswith('/'):
        snapshots_prefix += '/'
    snapshots_prefix += 'snapshots/'

    logger.info(f"Listing episodes in bucket: {bucket_name}, prefix: {snapshots_prefix}")

    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=snapshots_prefix,
            Delimiter='/'
        )

        episode_ids = []
        if 'CommonPrefixes' in response:
            for common_prefix in response['CommonPrefixes']:
                # Extract episode ID from prefix like 'zorkgpt_data/snapshots/YYYY-MM-DDTHH:MM:SS/'
                full_prefix_path = common_prefix.get('Prefix')
                # Remove the base snapshot prefix and the trailing slash
                episode_id = full_prefix_path.replace(snapshots_prefix, '', 1).rstrip('/')
                if episode_id: # Ensure we don't add empty strings
                    episode_ids.append(episode_id)

        logger.info(f"Found {len(episode_ids)} episodes: {episode_ids}")

        return {
            'statusCode': 200,
            'body': json.dumps({'episodes': episode_ids}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

    except Exception as e:
        logger.error(f"Error listing S3 objects: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }
