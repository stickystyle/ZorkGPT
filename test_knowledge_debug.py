#!/usr/bin/env python3
"""
Debug test to see what data is actually being passed to the LLM.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
import logging

# Monkey patch to intercept LLM calls
original_create = None
captured_prompts = []

def capture_llm_call(self, **kwargs):
    """Capture LLM prompts for inspection."""
    messages = kwargs.get('messages', [])
    for msg in messages:
        if msg['role'] == 'user':
            captured_prompts.append(msg['content'])
            print("\n" + "="*80)
            print("CAPTURED LLM PROMPT:")
            print("="*80)
            print(msg['content'][:2000] + "..." if len(msg['content']) > 2000 else msg['content'])
            print("="*80)
    
    # Return a mock response
    class MockResponse:
        content = """Based on the gameplay data provided:

1. **Game World Mechanics**: The mailbox contains a leaflet that provides game introduction. Opening containers reveals their contents. Dark areas contain grues that are fatal without light.

2. **Strategic Patterns**: Reading informational items (leaflet) provides context. Moving into darkness without light leads to death by grue.

3. **Environmental Knowledge**: The starting area has a mailbox. North leads to a dark area that requires light for safe passage.

4. **Danger Recognition**: Grues inhabit dark areas and will kill unprepared players. The warning "pitch black" indicates immediate danger.

5. **Efficiency Insights**: Always check for light sources before entering dark areas. Read all available text for clues.

6. **Problem-Solving Patterns**: Systematic interaction with objects (open, take, read) reveals information and items.

7. **Learning from Experience**: Light is essential for survival in dark areas. The game provides warnings before fatal encounters."""
    
    return MockResponse()


# Import after defining mock
from zork_strategy_generator import AdaptiveKnowledgeManager
import llm_client


def create_test_log(log_file: str, episode_id: str):
    """Create a test log with specific gameplay data."""
    
    entries = [
        # Turn 1
        {"timestamp": "2025-06-13T10:00:00", "level": "INFO", "event_type": "turn_completed", 
         "episode_id": episode_id, "turn": 1},
        {"timestamp": "2025-06-13T10:00:01", "level": "INFO", "event_type": "final_action_selection",
         "episode_id": episode_id, "turn": 1, "agent_action": "open mailbox",
         "agent_reasoning": "I need to explore the environment starting with the mailbox",
         "critic_score": 0.8},
        {"timestamp": "2025-06-13T10:00:02", "level": "INFO", "event_type": "zork_response",
         "episode_id": episode_id, "turn": 1, "action": "open mailbox",
         "zork_response": "Opening the small mailbox reveals a leaflet."},
         
        # Turn 5
        {"timestamp": "2025-06-13T10:02:00", "level": "INFO", "event_type": "turn_completed",
         "episode_id": episode_id, "turn": 5},
        {"timestamp": "2025-06-13T10:02:01", "level": "INFO", "event_type": "final_action_selection",
         "episode_id": episode_id, "turn": 5, "agent_action": "north",
         "agent_reasoning": "Exploring north to find new areas", "critic_score": 0.5},
        {"timestamp": "2025-06-13T10:02:02", "level": "INFO", "event_type": "zork_response",
         "episode_id": episode_id, "turn": 5, "action": "north",
         "zork_response": "North of House\nYou are facing the north side of a white house."},
         
        # Turn 10 - Death
        {"timestamp": "2025-06-13T10:05:00", "level": "INFO", "event_type": "turn_completed",
         "episode_id": episode_id, "turn": 10},
        {"timestamp": "2025-06-13T10:05:01", "level": "INFO", "event_type": "final_action_selection",
         "episode_id": episode_id, "turn": 10, "agent_action": "east",
         "agent_reasoning": "Going east into the forest", "critic_score": -0.3},
        {"timestamp": "2025-06-13T10:05:02", "level": "INFO", "event_type": "zork_response",
         "episode_id": episode_id, "turn": 10, "action": "east",
         "zork_response": "It is pitch black. You are likely to be eaten by a grue.\n\nOh, no! You have walked into the slavering fangs of a lurking grue!\n\n****  You have died  ****"},
        {"timestamp": "2025-06-13T10:05:03", "level": "INFO", "event_type": "game_over",
         "episode_id": episode_id, "turn": 10, "reason": "Death by grue", "death_count": 1}
    ]
    
    with open(log_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def main():
    """Test knowledge generation with prompt capture."""
    
    test_dir = tempfile.mkdtemp(prefix="zork_debug_")
    print(f"Test directory: {test_dir}")
    
    try:
        # Setup
        episode_id = "test_episode"
        episode_dir = os.path.join(test_dir, "episodes", episode_id)
        os.makedirs(episode_dir, exist_ok=True)
        
        log_file = os.path.join(episode_dir, "episode_log.jsonl")
        knowledge_file = os.path.join(test_dir, "knowledgebase.md")
        
        # Create test log
        print("\n1. Creating test log...")
        create_test_log(log_file, episode_id)
        
        # Setup logger
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        
        # Initialize knowledge manager
        print("\n2. Initializing AdaptiveKnowledgeManager...")
        km = AdaptiveKnowledgeManager(
            log_file=log_file,
            output_file=knowledge_file,
            logger=logger,
            workdir=test_dir
        )
        
        # Monkey patch the create method on the client instance
        original_method = km.client.chat.completions.create
        km.client.chat.completions.create = lambda **kwargs: capture_llm_call(km.client, **kwargs)
        
        # Extract turn data first
        print("\n3. Extracting turn window data...")
        turn_data = km._extract_turn_window_data(episode_id, 1, 10)
        
        if turn_data:
            print(f"\nExtracted turn data:")
            print(f"- Actions: {len(turn_data['actions_and_responses'])}")
            print(f"- Deaths: {len(turn_data['death_events'])}")
            
            print("\nAction details:")
            for action in turn_data['actions_and_responses']:
                print(f"  Turn {action['turn']}: {action['action']} -> {action['response'][:50]}...")
        
        # Trigger knowledge update
        print("\n4. Triggering knowledge update...")
        km.update_knowledge_from_turns(episode_id, 1, 10, is_final_update=True)
        
        print(f"\n5. Number of LLM prompts captured: {len(captured_prompts)}")
        
    finally:
        print(f"\nTest files in: {test_dir}")


if __name__ == "__main__":
    main()