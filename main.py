#!/usr/bin/env python3

from zork_orchestrator import ZorkOrchestrator
import time


def run_episode():
    """Run a long episode with adaptive knowledge management."""

    orchestrator = ZorkOrchestrator()

    print("🚀 Starting long episode with adaptive knowledge management...")
    print(f"📋 Configuration:")
    print(f"  - Max turns: {orchestrator.max_turns_per_episode}")
    print(
        f"  - Knowledge update interval: {orchestrator.knowledge_update_interval} turns"
    )
    print(f"  - Map update interval: {orchestrator.map_update_interval} turns")
    print(f"  - State export: {orchestrator.enable_state_export}")
    print(f"  - Turn delay: {orchestrator.turn_delay_seconds} seconds")
    print(f"  - S3 bucket: {orchestrator.s3_bucket or 'Not configured'}")
    print(
        f"  - S3 client: {'✅ Available' if orchestrator.s3_client else '❌ Not available'}"
    )
    print(f"  - Game server URL: {orchestrator.game_server_url}")
    print(f"  - Game working directory: {orchestrator.zork_workdir_abs_path}")
    print(f"  - Save file template: {orchestrator.zork_save_filename_template}")
    print()

    # Create game interface using the orchestrator's method
    game_interface = orchestrator.create_game_interface()
    
    with game_interface:
        try:
            final_score = orchestrator.play_episode(game_interface)

            print(f"\n🎯 Episode Complete!")
            print(f"  - Final score: {final_score}")
            print(f"  - Turns played: {orchestrator.turn_count}")

            # Calculate knowledge updates more accurately
            regular_updates = (
                orchestrator.turn_count // orchestrator.knowledge_update_interval
            )
            turns_since_last = (
                orchestrator.turn_count - orchestrator.last_knowledge_update_turn
            )
            min_final_threshold = max(10, orchestrator.knowledge_update_interval // 4)
            final_update_eligible = turns_since_last >= min_final_threshold

            print(f"  - Regular knowledge updates: {regular_updates}")
            if final_update_eligible:
                print(f"  - Final update: ✅ (analyzed {turns_since_last} turns)")
            else:
                print(
                    f"  - Final update: ❌ (only {turns_since_last} turns since last update)"
                )

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
    print("=" * 60)
    while True:
        try:
            run_episode()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
            time.sleep(1)
