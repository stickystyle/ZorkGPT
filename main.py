"""
Main entry point for ZorkGPT using the modular architecture.

This uses the modular design:
- ZorkOrchestrator: Coordinates all subsystems
- ZorkAgent: Handles action generation
- ZorkExtractor: Handles information extraction
- ZorkCritic: Handles action evaluation
- ZorkStrategyGenerator: Generates strategic guides
"""

from zork_api import ZorkInterface
from zork_orchestrator import ZorkOrchestrator


def main():
    """Main entry point for playing Zork episodes."""
    # Create ZorkOrchestrator instance with default settings
    orchestrator = ZorkOrchestrator()

    with ZorkInterface(timeout=1.0) as zork_game:
        try:
            episode_experiences, final_score = orchestrator.play_episode(zork_game)
            print(f"\nPlayed one episode. Final Zork score: {final_score}")
            print(f"Turns taken: {orchestrator.turn_count}")
            print(f"Base max turns: {orchestrator.base_max_turns_per_episode}")
            print(f"Final max turns: {orchestrator.max_turns_per_episode}")
            print(f"Turn limit increases: {orchestrator.turn_limit_increases}")
            if orchestrator.critic_scores_history:
                avg_critic_score = sum(orchestrator.critic_scores_history) / len(
                    orchestrator.critic_scores_history
                )
                print(f"Average critic score: {avg_critic_score:.3f}")
            print(orchestrator.game_map.render_ascii())
        except RuntimeError as e:
            print(f"ZorkInterface runtime error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("Ensuring Zork process is closed.")


if __name__ == "__main__":
    main()
