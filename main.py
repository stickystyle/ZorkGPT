#!/usr/bin/env python3

from orchestration import ZorkOrchestratorV2
import time


def run_episode():
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
    
    try:
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
    print("=" * 60)
    while True:
        try:
            run_episode()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
            time.sleep(1)
