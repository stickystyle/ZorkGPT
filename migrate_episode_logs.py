#!/usr/bin/env python3
"""
Migration script to split monolithic episode log into per-episode files.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import logging


def migrate_episode_logs(input_file: str, workdir: str = "game_files", dry_run: bool = False):
    """
    Split monolithic episode log into per-episode files.
    
    Args:
        input_file: Path to monolithic episode log file
        workdir: Working directory for game files
        dry_run: If True, show what would be done without making changes
        
    Returns:
        True if migration succeeded, False otherwise
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        return False
    
    # Group log entries by episode_id
    episode_logs = defaultdict(list)
    total_entries = 0
    skipped_entries = 0
    
    logger.info(f"Reading logs from {input_file}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    log_entry = json.loads(line)
                    episode_id = log_entry.get('episode_id', 'unknown')
                    
                    # Skip entries without episode_id or with empty episode_id
                    if not episode_id or episode_id == '':
                        logger.warning(f"Line {line_num}: No episode_id, skipping")
                        skipped_entries += 1
                        continue
                    
                    episode_logs[episode_id].append(log_entry)
                    total_entries += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error: {e}")
                    skipped_entries += 1
                    continue
    
    except IOError as e:
        logger.error(f"Error reading input file: {e}")
        return False
    
    logger.info(f"Processed {total_entries} log entries across {len(episode_logs)} episodes")
    if skipped_entries > 0:
        logger.info(f"Skipped {skipped_entries} entries due to missing episode_id or JSON errors")
    
    if dry_run:
        logger.info("DRY RUN: Would create the following episode files:")
        for episode_id, logs in episode_logs.items():
            episode_dir = Path(workdir) / "episodes" / episode_id
            episode_log_file = episode_dir / "episode_log.jsonl"
            logger.info(f"  {episode_log_file}: {len(logs)} entries")
        return True
    
    # Create episode directories and write files
    episodes_dir = Path(workdir) / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    migrated_episodes = 0
    
    for episode_id, logs in episode_logs.items():
        episode_dir = episodes_dir / episode_id
        episode_dir.mkdir(exist_ok=True)
        
        episode_log_file = episode_dir / "episode_log.jsonl"
        
        try:
            with open(episode_log_file, 'w', encoding='utf-8') as f:
                for log_entry in logs:
                    f.write(json.dumps(log_entry) + '\n')
            
            logger.info(f"Migrated episode {episode_id}: {len(logs)} entries")
            migrated_episodes += 1
            
        except IOError as e:
            logger.error(f"Error writing episode file {episode_log_file}: {e}")
            continue
    
    logger.info(f"Migration complete: {migrated_episodes} episodes migrated")
    
    # Create backup of original file
    if migrated_episodes > 0:
        backup_path = input_path.with_suffix('.jsonl.backup')
        if not backup_path.exists():
            try:
                import shutil
                shutil.copy2(input_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            except IOError as e:
                logger.warning(f"Could not create backup: {e}")
    
    return True


def analyze_log_file(input_file: str):
    """
    Analyze a log file and display statistics about episodes.
    
    Args:
        input_file: Path to log file to analyze
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    episode_stats = defaultdict(lambda: {"entries": 0, "turns": 0, "first_timestamp": None, "last_timestamp": None})
    total_entries = 0
    no_episode_entries = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    log_entry = json.loads(line)
                    episode_id = log_entry.get('episode_id', '')
                    timestamp = log_entry.get('timestamp', '')
                    
                    total_entries += 1
                    
                    if not episode_id:
                        no_episode_entries += 1
                        continue
                    
                    stats = episode_stats[episode_id]
                    stats["entries"] += 1
                    
                    if log_entry.get("event_type") == "turn_start":
                        turn = log_entry.get("turn", 0)
                        stats["turns"] = max(stats["turns"], turn)
                    
                    if timestamp:
                        if not stats["first_timestamp"] or timestamp < stats["first_timestamp"]:
                            stats["first_timestamp"] = timestamp
                        if not stats["last_timestamp"] or timestamp > stats["last_timestamp"]:
                            stats["last_timestamp"] = timestamp
                    
                except json.JSONDecodeError:
                    continue
    
    except IOError as e:
        logger.error(f"Error reading input file: {e}")
        return
    
    # Display statistics
    logger.info(f"\nLog File Analysis: {input_file}")
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Entries without episode_id: {no_episode_entries}")
    logger.info(f"Total episodes: {len(episode_stats)}")
    
    if episode_stats:
        logger.info("\nEpisode Details:")
        for episode_id in sorted(episode_stats.keys()):
            stats = episode_stats[episode_id]
            logger.info(f"\n  Episode: {episode_id}")
            logger.info(f"    Entries: {stats['entries']}")
            logger.info(f"    Turns: {stats['turns']}")
            logger.info(f"    First: {stats['first_timestamp']}")
            logger.info(f"    Last: {stats['last_timestamp']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate monolithic episode log to per-episode files")
    parser.add_argument("input_file", help="Path to monolithic episode log file")
    parser.add_argument("--workdir", default="game_files", help="Working directory for episodes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without doing it")
    parser.add_argument("--analyze", action="store_true", help="Analyze log file and show statistics")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_log_file(args.input_file)
    else:
        success = migrate_episode_logs(args.input_file, args.workdir, args.dry_run)
        exit(0 if success else 1)