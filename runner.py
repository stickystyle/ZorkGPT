from zork_api import ZorkInterface
from zork_orchestrator import ZorkOrchestrator

orchestrator = ZorkOrchestrator()
for x in range(10):
    print(f"Playing episode {x+1}")
    with ZorkInterface(timeout=1.0) as zork_game:
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
    print(f"Episode {x+1} complete")
