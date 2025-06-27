#!/usr/bin/env python3
"""
Test script to validate knowledge base generation workflow.

This script simulates how zork_strategy_generator.py (AdaptiveKnowledgeManager)
processes log entries and generates knowledgebase.md during a running episode.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import shutil

# Import the knowledge manager
from zork_strategy_generator import AdaptiveKnowledgeManager
import logging
from logger import create_zork_logger


def create_sample_episode_log(log_file: str, episode_id: str):
    """Create a sample episode log with realistic game data."""
    
    sample_entries = [
        # Turn 1: Opening mailbox
        {
            "timestamp": "2025-06-13T10:00:00.000000",
            "level": "INFO",
            "message": "Turn 1 starting",
            "event_type": "turn_completed",
            "episode_id": episode_id,
            "turn": 1
        },
        {
            "timestamp": "2025-06-13T10:00:01.000000",
            "level": "INFO",
            "message": "Agent action selected",
            "event_type": "final_action_selection",
            "episode_id": episode_id,
            "turn": 1,
            "agent_action": "open mailbox",
            "agent_reasoning": "Starting the game by opening the mailbox to see what's inside",
            "critic_score": 0.8
        },
        {
            "timestamp": "2025-06-13T10:00:02.000000",
            "level": "INFO",
            "message": "Zork response",
            "event_type": "zork_response",
            "episode_id": episode_id,
            "turn": 1,
            "action": "open mailbox",
            "zork_response": "Opening the small mailbox reveals a leaflet."
        },
        
        # Turn 2: Taking leaflet
        {
            "timestamp": "2025-06-13T10:00:10.000000",
            "level": "INFO",
            "message": "Turn 2 starting",
            "event_type": "turn_completed",
            "episode_id": episode_id,
            "turn": 2
        },
        {
            "timestamp": "2025-06-13T10:00:11.000000",
            "level": "INFO",
            "message": "Agent action selected",
            "event_type": "final_action_selection",
            "episode_id": episode_id,
            "turn": 2,
            "agent_action": "take leaflet",
            "agent_reasoning": "Taking the leaflet to read it and gather information",
            "critic_score": 0.9
        },
        {
            "timestamp": "2025-06-13T10:00:12.000000",
            "level": "INFO",
            "message": "Zork response",
            "event_type": "zork_response",
            "episode_id": episode_id,
            "turn": 2,
            "action": "take leaflet",
            "zork_response": "Taken."
        },
        
        # Turn 3: Reading leaflet
        {
            "timestamp": "2025-06-13T10:00:20.000000",
            "level": "INFO",
            "message": "Turn 3 starting",
            "event_type": "turn_completed",
            "episode_id": episode_id,
            "turn": 3
        },
        {
            "timestamp": "2025-06-13T10:00:21.000000",
            "level": "INFO",
            "message": "Agent action selected",
            "event_type": "final_action_selection",
            "episode_id": episode_id,
            "turn": 3,
            "agent_action": "read leaflet",
            "agent_reasoning": "Reading the leaflet to understand the game's premise",
            "critic_score": 0.85
        },
        {
            "timestamp": "2025-06-13T10:00:22.000000",
            "level": "INFO",
            "message": "Zork response",
            "event_type": "zork_response",
            "episode_id": episode_id,
            "turn": 3,
            "action": "read leaflet",
            "zork_response": "\"WELCOME TO ZORK!\\n\\nZORK is a game of adventure, danger, and low cunning. In it you will explore some of the most amazing territory ever seen by mortals. No computer should be without one!\""
        },
        
        # Turn 10: Death event for testing
        {
            "timestamp": "2025-06-13T10:05:00.000000",
            "level": "INFO",
            "message": "Turn 10 starting",
            "event_type": "turn_completed",
            "episode_id": episode_id,
            "turn": 10
        },
        {
            "timestamp": "2025-06-13T10:05:01.000000",
            "level": "INFO",
            "message": "Agent action selected",
            "event_type": "final_action_selection",
            "episode_id": episode_id,
            "turn": 10,
            "agent_action": "go north",
            "agent_reasoning": "Exploring north into the darkness without a light source",
            "critic_score": -0.5
        },
        {
            "timestamp": "2025-06-13T10:05:02.000000",
            "level": "INFO",
            "message": "Zork response",
            "event_type": "zork_response",
            "episode_id": episode_id,
            "turn": 10,
            "action": "go north",
            "zork_response": "It is pitch black. You are likely to be eaten by a grue.\\n\\nOh, no! You have walked into the slavering fangs of a lurking grue!\\n\\n****  You have died  ****"
        },
        {
            "timestamp": "2025-06-13T10:05:03.000000",
            "level": "INFO",
            "message": "Game over",
            "event_type": "game_over",
            "episode_id": episode_id,
            "turn": 10,
            "reason": "Death: eaten by a grue",
            "final_score": 0,
            "death_count": 1
        }
    ]
    
    # Write entries to log file
    with open(log_file, 'w') as f:
        for entry in sample_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Created sample log with {len(sample_entries)} entries")


def test_knowledge_generation():
    """Test the knowledge base generation workflow."""
    
    # Create temporary directory for test
    test_dir = tempfile.mkdtemp(prefix="zork_test_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Set up test environment
        episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        episode_dir = os.path.join(test_dir, "episodes", episode_id)
        os.makedirs(episode_dir, exist_ok=True)
        
        log_file = os.path.join(episode_dir, "episode_log.jsonl")
        knowledge_file = os.path.join(test_dir, "knowledgebase.md")
        
        # Create sample log data
        print(f"\n1. Creating sample episode log...")
        create_sample_episode_log(log_file, episode_id)
        
        # Initialize logger (simple Python logger for test)
        logger = logging.getLogger("test_knowledge")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(handler)
        
        # Initialize AdaptiveKnowledgeManager
        print(f"\n2. Initializing AdaptiveKnowledgeManager...")
        knowledge_manager = AdaptiveKnowledgeManager(
            log_file=log_file,
            output_file=knowledge_file,
            logger=logger,
            workdir=test_dir
        )
        
        # Test 1: Check quality assessment for turns 1-3
        print(f"\n3. Testing knowledge update for turns 1-3...")
        success = knowledge_manager.update_knowledge_from_turns(
            episode_id=episode_id,
            start_turn=1,
            end_turn=3,
            is_final_update=False
        )
        print(f"   Update success: {success}")
        
        # Check if knowledge base was created
        if os.path.exists(knowledge_file):
            print(f"\n4. Knowledge base created successfully!")
            print(f"   File size: {os.path.getsize(knowledge_file)} bytes")
            
            # Display content
            print(f"\n5. Knowledge base content:")
            print("=" * 80)
            with open(knowledge_file, 'r') as f:
                content = f.read()
                print(content[:1000] + "..." if len(content) > 1000 else content)
            print("=" * 80)
        else:
            print(f"\n4. ERROR: Knowledge base was not created!")
        
        # Test 2: Final update with death event
        print(f"\n6. Testing final knowledge update (with death event)...")
        success = knowledge_manager.update_knowledge_from_turns(
            episode_id=episode_id,
            start_turn=1,
            end_turn=10,
            is_final_update=True
        )
        print(f"   Final update success: {success}")
        
        if os.path.exists(knowledge_file):
            print(f"\n7. Updated knowledge base content:")
            print("=" * 80)
            with open(knowledge_file, 'r') as f:
                content = f.read()
                # Show last 1000 chars to see new content
                if len(content) > 2000:
                    print("...\n" + content[-1000:])
                else:
                    print(content)
            print("=" * 80)
        
        # Test 3: Extract turn window data directly
        print(f"\n8. Testing _extract_turn_window_data method...")
        turn_data = knowledge_manager._extract_turn_window_data(
            episode_id=episode_id,
            start_turn=1,
            end_turn=10
        )
        
        if turn_data:
            print(f"   Extracted data:")
            print(f"   - Actions and responses: {len(turn_data['actions_and_responses'])}")
            print(f"   - Death events: {len(turn_data['death_events'])}")
            print(f"   - Game over events: {len(turn_data['game_over_events'])}")
            
            if turn_data['actions_and_responses']:
                print(f"\n   Sample action-response pair:")
                sample = turn_data['actions_and_responses'][0]
                print(f"   Turn {sample['turn']}: {sample['action']}")
                print(f"   Response: {sample['response'][:100]}...")
        else:
            print(f"   ERROR: No turn data extracted!")
        
        print(f"\n9. Test completed successfully!")
        
    except Exception as e:
        print(f"\nERROR during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up (optional - comment out to inspect files)
        print(f"\nCleaning up test directory...")
        # shutil.rmtree(test_dir)
        print(f"Test files preserved in: {test_dir}")


if __name__ == "__main__":
    print("Testing Knowledge Base Generation")
    print("=" * 80)
    test_knowledge_generation()