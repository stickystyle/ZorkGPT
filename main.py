#!/usr/bin/env python3

from orchestration import ZorkOrchestratorV2
import time
import argparse
import signal
from datetime import datetime


def run_episode(episode_id=None, max_turns=None):
    """Run a long episode with adaptive knowledge management."""

    # Generate episode_id if not provided
    if episode_id is None:
        episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    orchestrator = ZorkOrchestratorV2(
        episode_id=episode_id, max_turns_per_episode=max_turns
    )

    print("🚀 Starting long episode with ZorkOrchestrator v2...", flush=True)
    print("📋 Configuration:", flush=True)
    print(f"  - Max turns: {orchestrator.config.max_turns_per_episode}", flush=True)
    print(
        f"  - Knowledge update interval: {orchestrator.config.knowledge_update_interval} turns", flush=True
    )
    print(f"  - State export: {orchestrator.config.enable_state_export}", flush=True)
    print(f"  - Turn delay: {orchestrator.config.turn_delay_seconds} seconds", flush=True)
    print(f"  - S3 bucket: {orchestrator.config.s3_bucket or 'Not configured'}", flush=True)
    print(
        f"  - S3 client: {'✅ Available' if orchestrator.state_manager.s3_client else '❌ Not available'}", flush=True
    )
    print(flush=True)

    interrupted_by_user = False
    final_score = None

    try:
        # Play the episode - orchestrator manages Jericho interface internally
        final_score = orchestrator.play_episode()
    except KeyboardInterrupt:
        interrupted_by_user = True
        print("\n\n🛑 Interrupted by user (Ctrl-C)")
        print("What would you like to do?")
        print("  [1] Graceful shutdown (save current progress and finalize episode)")
        print("  [2] Exit immediately (lose progress since last save)")

        choice = None
        while choice not in ["1", "2"]:
            try:
                choice = input("Enter choice (1 or 2): ").strip()
                if choice not in ["1", "2"]:
                    print("Invalid choice. Please enter 1 or 2.")
            except (KeyboardInterrupt, EOFError):
                choice = "2"  # Default to immediate exit if interrupted again
                break

        if choice == "1":
            print("\n📦 Performing graceful shutdown...")
            print("  - Finalizing episode...")
            print("  ⚠ Please wait, do not interrupt again...")

            # Disable interrupts during cleanup to prevent partial saves
            original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

            # Trigger the same cleanup as normal episode end
            try:
                # Get current score (if Jericho is still running)
                if orchestrator.jericho_interface and orchestrator.jericho_interface.env:
                    final_score, _ = orchestrator.jericho_interface.get_score()
                else:
                    final_score = orchestrator.game_state.previous_zork_score
                    print("  ⚠ Jericho not running, using last known score")

                # Finalize episode (knowledge synthesis, etc.)
                orchestrator.episode_synthesizer.finalize_episode(
                    final_score=final_score,
                    critic_confidence_history=orchestrator.critic_confidence_history,
                )
                print("  ✓ Episode finalized")

                # Export final state
                orchestrator._export_coordinated_state()
                print("  ✓ State exported")

                # Save map state
                orchestrator.map_manager.save_map_state()
                print("  ✓ Map state saved")

                # Flush Langfuse traces if available
                if orchestrator.langfuse_client:
                    try:
                        orchestrator.langfuse_client.flush()
                        print("  ✓ Langfuse traces flushed")
                    except Exception as e:
                        print(f"  ⚠ Warning: Failed to flush Langfuse traces: {e}")

                # Close Jericho interface
                orchestrator.jericho_interface.close()
                print("  ✓ Jericho interface closed")

                print("\n✅ Graceful shutdown complete!")
                print(f"  - Final score: {final_score}")
                print(f"  - Turns played: {orchestrator.game_state.turn_count}")
                print(f"  - Episode ID: {orchestrator.game_state.episode_id}")

            except Exception as e:
                print(f"\n❌ Error during graceful shutdown: {e}")
                import traceback
                traceback.print_exc()
                print("\nSome data may not have been saved properly.")
            finally:
                # Restore original interrupt handler
                signal.signal(signal.SIGINT, original_handler)
        else:
            print("\n⚡ Exiting immediately without saving...")

        # Re-raise KeyboardInterrupt to properly exit continuous/multi-episode mode
        raise KeyboardInterrupt()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Normal episode completion (only runs if not interrupted)
    if not interrupted_by_user and final_score is not None:
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

    print("=" * 60, flush=True)

    # Show mode information
    if args.continuous:
        print("🔄 CONTINUOUS MODE: Will run new episodes indefinitely", flush=True)
    elif args.episodes > 1:
        print(f"🎮 MULTIPLE EPISODES MODE: Will run {args.episodes} episodes", flush=True)
    else:
        print("🎯 SINGLE EPISODE MODE: Starting new episode", flush=True)

    if args.max_turns:
        print(f"📏 Max turns per episode: {args.max_turns}", flush=True)
    print(flush=True)

    if args.continuous:
        # Continuous mode
        try:
            while True:
                try:
                    run_episode(max_turns=args.max_turns)
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback

                    traceback.print_exc()
                    print("🔄 Retrying in 5 seconds...")
                    time.sleep(5)
        except KeyboardInterrupt:
            print("\n\n🛑 Continuous mode interrupted by user (Ctrl-C)")
            print("Exiting continuous mode...")
    elif args.episodes > 1:
        # Multiple episodes mode
        i = -1  # Initialize in case interrupt happens before loop starts
        try:
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
        except KeyboardInterrupt:
            completed_count = i + 1 if i >= 0 else 0
            print(f"\n\n🛑 Multiple episodes mode interrupted by user (Ctrl-C)")
            print(f"Completed {completed_count} of {args.episodes} episodes before interruption")
    else:
        # Single episode mode
        try:
            run_episode(max_turns=args.max_turns)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
