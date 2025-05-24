from main import ZorkAgent
from zork_api import ZorkInterface

agent = ZorkAgent()

for i in range(0, 100):
    print(f"Starting episode {i}...")

    with ZorkInterface(timeout=1.0) as zork_game:
        try:
            episode_experiences, final_score = agent.play_episode(zork_game)
            print(f"\nPlayed one episode. Final Zork score: {final_score}")
            print(f"Turns taken: {agent.turn_count}")
            print(f"Base max turns: {agent.base_max_turns_per_episode}")
            print(f"Final max turns: {agent.max_turns_per_episode}")
            print(f"Turn limit increases: {agent.turn_limit_increases}")
            if agent.critic_scores_history:
                avg_critic_score = sum(agent.critic_scores_history) / len(agent.critic_scores_history)
                print(f"Average critic score: {avg_critic_score:.3f}")
            print(agent.game_map.render_ascii())
        except RuntimeError as e:
            print(f"ZorkInterface runtime error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("Ensuring Zork process is closed.")