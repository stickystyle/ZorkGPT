#!/usr/bin/env python3

from orchestration import ZorkOrchestratorV2
import time
import argparse
from datetime import datetime


def run_episode(episode_id=None, max_turns=None):
    """Run a long episode with adaptive knowledge management."""

    # Generate episode_id if not provided
    if episode_id is None:
        episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    orchestrator = ZorkOrchestratorV2(
        episode_id=episode_id, max_turns_per_episode=max_turns
    )

    print("🚀 Starting long episode with ZorkOrchestrator v2...")
    print("📋 Configuration:")
    print(f"  - Max turns: {orchestrator.config.max_turns_per_episode}")
    print(
        f"  - Knowledge update interval: {orchestrator.config.knowledge_update_interval} turns"
    )
    print(f"  - State export: {orchestrator.config.enable_state_export}")
    print(f"  - Turn delay: {orchestrator.config.turn_delay_seconds} seconds")
    print(f"  - S3 bucket: {orchestrator.config.s3_bucket or 'Not configured'}")
    print(
        f"  - S3 client: {'✅ Available' if orchestrator.state_manager.s3_client else '❌ Not available'}"
    )
    print()

    try:
        # Play the episode - orchestrator manages Jericho interface internally
        final_score = orchestrator.play_episode()

        print("\n🎯 Episode Complete!")
        print(f"  - Final score: {final_score}")
        print(f"  - Turns played: {orchestrator.game_state.turn_count}")
        print(f"  - Episode ID: {orchestrator.game_state.episode_id}")

        # Calculate knowledge updates more accurately
        regular_updates = (
            orchestrator.game_state.turn_count
            // orchestrator.config.knowledge_update_interval
        )
        turns_since_last = (
            orchestrator.game_state.turn_count
            - orchestrator.knowledge_manager.last_knowledge_update_turn
        )
        min_final_threshold = max(
            10, orchestrator.config.knowledge_update_interval // 4
        )
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
        print("\n📊 Manager Status:")
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
        "--continuous", action="store_true", help="Run episodes continuously in a loop"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns per episode",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )

    args = parser.parse_args()

    print("=" * 60)

    # Show mode information
    if args.continuous:
        print("🔄 CONTINUOUS MODE: Will run new episodes indefinitely")
    elif args.episodes > 1:
        print(f"🎮 MULTIPLE EPISODES MODE: Will run {args.episodes} episodes")
    else:
        print("🎯 SINGLE EPISODE MODE: Starting new episode")

    if args.max_turns:
        print(f"📏 Max turns per episode: {args.max_turns}")
    print()

    if args.continuous:
        # Continuous mode
        while True:
            try:
                run_episode(max_turns=args.max_turns)
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()
                print("🔄 Retrying in 5 seconds...")
                time.sleep(5)
    elif args.episodes > 1:
        # Multiple episodes mode
        for i in range(args.episodes):
            try:
                print(f"\n🎮 Starting episode {i + 1} of {args.episodes}")
                run_episode(max_turns=args.max_turns)
            except Exception as e:
                print(f"❌ Error in episode {i + 1}: {e}")
                import traceback

                traceback.print_exc()
                if i < args.episodes - 1:
                    print("🔄 Starting next episode in 5 seconds...")
                    time.sleep(5)
    else:
        # Single episode mode
        try:
            run_episode(max_turns=args.max_turns)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
