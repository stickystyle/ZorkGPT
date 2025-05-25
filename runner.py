from zork_api import ZorkInterface
from zork_orchestrator import ZorkOrchestrator

orchestrator = ZorkOrchestrator(
    max_turns_per_episode=5000,
    enable_adaptive_knowledge=True,
    knowledge_update_interval=100,
    auto_update_knowledge=False,
)  # Use only adaptive system


with ZorkInterface(timeout=1.0) as zork_game:
    episode_experiences, final_score = orchestrator.play_episode(zork_game)
    print(f"\nPlayed one episode. Final Zork score: {final_score}")
    print(f"Turns taken: {orchestrator.turn_count}")
    print(f"Max turns: {orchestrator.max_turns_per_episode}")
