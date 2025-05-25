#!/usr/bin/env python3

from zork_api import ZorkInterface
from zork_orchestrator import ZorkOrchestrator
import time

def run_long_episode():
    """Run a long episode with adaptive knowledge management."""
    
    # Configure for long episodes with adaptive knowledge
    orchestrator = ZorkOrchestrator(
        # Episode configuration
        max_turns_per_episode=5000,        # Very long episodes
        
        enable_adaptive_knowledge=True,    # Enable turn-based updates
        knowledge_update_interval=100,     # Update every 100 turns
        
        # Traditional episode-based updates (optional)
        auto_update_knowledge=False,       # Disable to rely only on adaptive system
    )
    
    print("🚀 Starting long episode with adaptive knowledge management...")
    print(f"📋 Configuration:")
    print(f"  - Max turns: {orchestrator.max_turns_per_episode}")
    print(f"  - Knowledge update interval: {orchestrator.knowledge_update_interval} turns")
    print(f"  - Adaptive knowledge: {orchestrator.enable_adaptive_knowledge}")
    print()
    
    with ZorkInterface(timeout=1.0) as zork_game:
        try:
            episode_experiences, final_score = orchestrator.play_episode(zork_game)
            
            print(f"\n🎯 Episode Complete!")
            print(f"  - Final score: {final_score}")
            print(f"  - Turns played: {orchestrator.turn_count}")
            print(f"  - Knowledge updates: {orchestrator.turn_count // orchestrator.knowledge_update_interval}")
            
            # Show the final knowledge base
            try:
                with open("knowledgebase.md", "r") as f:
                    knowledge_content = f.read()
                    print(f"\n📚 Final knowledge base ({len(knowledge_content)} characters):")
                    print("=" * 60)
                    print(knowledge_content[:500] + "..." if len(knowledge_content) > 500 else knowledge_content)
            except FileNotFoundError:
                print("\n📚 No knowledge base file found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()



if __name__ == "__main__":
    print("=" * 60)
    while True:
        try:
            run_long_episode()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
        