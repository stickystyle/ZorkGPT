#!/usr/bin/env python3

from orchestration import ZorkOrchestratorV2
import time
import argparse
import os
import json
import glob
from datetime import datetime


def find_latest_save_episode_id(game_files_dir="game_files"):
    """Find the most recent save file and extract its episode ID.
    
    Args:
        game_files_dir: Directory containing save files
        
    Returns:
        Episode ID string if found, None otherwise
    """
    if not os.path.exists(game_files_dir):
        print(f"📁 Game files directory '{game_files_dir}' not found")
        return None
    
    # Look for metadata files which contain episode info
    metadata_pattern = os.path.join(game_files_dir, "*_metadata.json")
    metadata_files = glob.glob(metadata_pattern)
    
    if not metadata_files:
        print(f"📁 No save metadata files found in '{game_files_dir}'")
        return None
    
    latest_file = None
    latest_time = None
    latest_episode_id = None
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            session_id = metadata.get('session_id')
            last_command_time = metadata.get('last_command_time')
            
            if session_id and last_command_time:
                # Parse the timestamp
                command_time = datetime.fromisoformat(last_command_time.replace('Z', '+00:00'))
                
                if latest_time is None or command_time > latest_time:
                    latest_time = command_time
                    latest_file = metadata_file
                    latest_episode_id = session_id
                    
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️  Warning: Could not parse metadata file {metadata_file}: {e}")
            continue
    
    if latest_episode_id:
        print(f"🔍 Found latest save: {latest_episode_id} (last activity: {latest_time})")
        return latest_episode_id
    else:
        print(f"📁 No valid save files found in '{game_files_dir}'")
        return None


def run_episode(episode_id=None):
    """Run a long episode with adaptive knowledge management."""

    orchestrator = ZorkOrchestratorV2()

    print("🚀 Starting long episode with ZorkOrchestrator v2...")
    print(f"📋 Configuration:")
    print(f"  - Max turns: {orchestrator.config.max_turns_per_episode}")
    print(
        f"  - Knowledge update interval: {orchestrator.config.knowledge_update_interval} turns"
    )
    print(f"  - Map update interval: {orchestrator.config.map_update_interval} turns")
    print(f"  - State export: {orchestrator.config.enable_state_export}")
    print(f"  - Turn delay: {orchestrator.config.turn_delay_seconds} seconds")
    print(f"  - S3 bucket: {orchestrator.config.s3_bucket or 'Not configured'}")
    print(
        f"  - S3 client: {'✅ Available' if orchestrator.state_manager.s3_client else '❌ Not available'}"
    )
    print(f"  - Game server URL: {orchestrator.config.game_server_url}")
    print()

    # Create game interface using the orchestrator's method
    game_interface = orchestrator.create_game_interface()
    
    # Check if we're restoring an existing episode
    if episode_id:
        print(f"🔄 Attempting to restore episode: {episode_id}")
        restore_response = game_interface.start_session(session_id=episode_id)
        if restore_response["success"]:
            print(f"✅ Successfully connected to episode {episode_id}")
            # Update orchestrator state to match the restored episode
            orchestrator.game_state.episode_id = episode_id
            orchestrator.agent.update_episode_id(episode_id)
            orchestrator.extractor.update_episode_id(episode_id)
            orchestrator.critic.update_episode_id(episode_id)
        else:
            print(f"❌ Failed to restore episode {episode_id}: {restore_response.get('error', 'Unknown error')}")
            print("🆕 Starting new episode instead...")
            episode_id = None
    
    try:
        if episode_id:
            # For restored episodes, get current state and continue
            print("🎮 Continuing restored episode...")
            current_state_response = game_interface.send_command("look")
            if current_state_response["success"]:
                current_state = current_state_response["response"]
                # Extract and process current state
                extracted_info = orchestrator.extractor.extract_info(current_state)
                orchestrator._process_extraction(extracted_info, "", current_state)
                # Run the game loop starting from current state
                final_score = orchestrator._run_game_loop(game_interface, current_state)
            else:
                print("❌ Failed to get current state, starting new episode")
                final_score = orchestrator.play_episode(game_interface)
        else:
            final_score = orchestrator.play_episode(game_interface)

        print(f"\n🎯 Episode Complete!")
        print(f"  - Final score: {final_score}")
        print(f"  - Turns played: {orchestrator.game_state.turn_count}")
        print(f"  - Episode ID: {orchestrator.game_state.episode_id}")

        # Calculate knowledge updates more accurately
        regular_updates = (
            orchestrator.game_state.turn_count // orchestrator.config.knowledge_update_interval
        )
        turns_since_last = (
            orchestrator.game_state.turn_count - orchestrator.knowledge_manager.last_knowledge_update_turn
        )
        min_final_threshold = max(10, orchestrator.config.knowledge_update_interval // 4)
        final_update_eligible = turns_since_last >= min_final_threshold

        print(f"  - Regular knowledge updates: {regular_updates}")
        if final_update_eligible:
            print(f"  - Final update: ✅ (analyzed {turns_since_last} turns)")
        else:
            print(
                f"  - Final update: ❌ (only {turns_since_last} turns since last update)"
            )

        # Show orchestrator status
        status = orchestrator.get_orchestrator_status()
        print(f"\n📊 Manager Status:")
        for manager_name, manager_status in status["managers"].items():
            print(f"  - {manager_name}: {manager_status.get('component', 'N/A')}")

        # Show the final knowledge base
        try:
            with open("knowledgebase.md", "r") as f:
                knowledge_content = f.read()
                print(
                    f"\n📚 Final knowledge base ({len(knowledge_content)} characters):"
                )
                print("=" * 60)
                print(
                    knowledge_content[:500] + "..."
                    if len(knowledge_content) > 500
                    else knowledge_content
                )
        except FileNotFoundError:
            print("\n📚 No knowledge base file found")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ZorkGPT episodes")
    parser.add_argument(
        "--episode-id", 
        type=str, 
        help="Episode ID to restore and continue (ISO8601 format: YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--continuous", 
        action="store_true", 
        help="Run episodes continuously in a loop"
    )
    parser.add_argument(
        "--restore-last", 
        action="store_true", 
        help="Automatically restore from the most recent save file in game_files/"
    )
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.episode_id and args.restore_last:
        print("❌ Error: Cannot use both --episode-id and --restore-last")
        parser.print_help()
        exit(1)
    
    print("=" * 60)
    
    # Show mode information
    if args.continuous and args.restore_last:
        print("🔄 CONTINUOUS MODE with restore-last: Will restore latest save, then run new episodes")
    elif args.continuous:
        print("🔄 CONTINUOUS MODE: Will run new episodes indefinitely")
    elif args.episode_id:
        print(f"🎯 SINGLE EPISODE MODE: Restoring specific episode {args.episode_id}")
    elif args.restore_last:
        print("🎯 SINGLE EPISODE MODE: Restoring from latest save")
    else:
        print("🎯 SINGLE EPISODE MODE: Starting new episode")
    print()
    
    def get_episode_id_to_restore():
        """Get the episode ID to restore, considering --restore-last and --episode-id."""
        if args.episode_id:
            return args.episode_id
        elif args.restore_last:
            return find_latest_save_episode_id()
        else:
            return None
    
    if args.continuous:
        # Continuous mode with optional restore capability
        first_episode = True
        while True:
            try:
                # Only attempt restore on the first episode if --restore-last is specified
                episode_id_to_use = get_episode_id_to_restore() if first_episode else None
                run_episode(episode_id=episode_id_to_use)
                first_episode = False  # Subsequent episodes will be new
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print("🔄 Retrying in 5 seconds...")
                time.sleep(5)
                first_episode = False  # Don't try to restore after errors
    else:
        # Single episode mode (with optional restore)
        try:
            episode_id_to_use = get_episode_id_to_restore()
            run_episode(episode_id=episode_id_to_use)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
