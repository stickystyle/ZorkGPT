#!/usr/bin/env python3
"""
Test script to demonstrate the dynamic turn limit functionality.
This script creates a ZorkAgent and simulates performance metrics to show
how the turn limit adjustment would work in practice.
"""

from main import ZorkAgent
from datetime import datetime


def test_dynamic_turn_limits():
    """Test the dynamic turn limit functionality with simulated data."""
    
    print("Testing Dynamic Turn Limit Functionality")
    print("=" * 50)
    
    # Create agent with custom dynamic turn limit settings for testing
    agent = ZorkAgent(
        max_turns_per_episode=50,  # Lower starting point for testing
        absolute_max_turns=200,    # Lower absolute max for testing
        turn_limit_increment=25,   # Smaller increments for testing
        performance_check_interval=10,  # Check more frequently
        performance_threshold=0.6,  # Lower threshold for easier triggering
        min_turns_for_increase=20,  # Lower minimum for testing
    )
    
    # Reset episode state
    agent.reset_episode_state()
    agent.episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    print(f"Initial Configuration:")
    print(f"  Base max turns: {agent.base_max_turns_per_episode}")
    print(f"  Current max turns: {agent.max_turns_per_episode}")
    print(f"  Absolute max turns: {agent.absolute_max_turns}")
    print(f"  Turn limit increment: {agent.turn_limit_increment}")
    print(f"  Performance check interval: {agent.performance_check_interval}")
    print(f"  Performance threshold: {agent.performance_threshold}")
    print(f"  Min turns for increase: {agent.min_turns_for_increase}")
    print()
    
    # Simulate some turns with good performance
    print("Simulating episode with good performance...")
    
    # Simulate turns 1-19 (before min_turns_for_increase)
    for turn in range(1, 20):
        agent.turn_count = turn
        agent.critic_scores_history.append(0.7)  # Good score
        # Simulate some experiences for reward calculation
        if hasattr(agent, 'experience_tracker'):
            agent.experience_tracker.add_experience(
                state=f"test_state_{turn}",
                action=f"test_action_{turn}",
                reward=0.5,  # Positive reward
                next_state=f"test_next_state_{turn}",
                done=False,
                critic_score=0.7,
                critic_justification="Good performance",
                zork_score=turn
            )
    
    print(f"Turn {agent.turn_count}: No adjustment yet (below min_turns_for_increase)")
    increased = agent.evaluate_performance_and_adjust_turn_limit()
    print(f"  Turn limit increased: {increased}")
    print(f"  Current max turns: {agent.max_turns_per_episode}")
    print()
    
    # Simulate turns 20-29 (first potential increase)
    for turn in range(20, 30):
        agent.turn_count = turn
        agent.critic_scores_history.append(0.8)  # Even better score
        agent.action_history.append((f"go north", f"response_{turn}"))  # Movement action
        if hasattr(agent, 'experience_tracker'):
            agent.experience_tracker.add_experience(
                state=f"test_state_{turn}",
                action=f"go north",
                reward=0.6,
                next_state=f"test_next_state_{turn}",
                done=False,
                critic_score=0.8,
                critic_justification="Excellent performance",
                zork_score=turn
            )
    
    print(f"Turn {agent.turn_count}: First performance check with good metrics")
    increased = agent.evaluate_performance_and_adjust_turn_limit()
    print(f"  Turn limit increased: {increased}")
    print(f"  Current max turns: {agent.max_turns_per_episode}")
    print(f"  Total increases: {agent.turn_limit_increases}")
    print()
    
    # Simulate more turns (30-39) with continued good performance
    for turn in range(30, 40):
        agent.turn_count = turn
        agent.critic_scores_history.append(0.75)
        agent.action_history.append((f"examine object", f"response_{turn}"))
        if hasattr(agent, 'experience_tracker'):
            agent.experience_tracker.add_experience(
                state=f"test_state_{turn}",
                action=f"examine object",
                reward=0.4,
                next_state=f"test_next_state_{turn}",
                done=False,
                critic_score=0.75,
                critic_justification="Good exploration",
                zork_score=turn + 5
            )
    
    print(f"Turn {agent.turn_count}: Second performance check")
    increased = agent.evaluate_performance_and_adjust_turn_limit()
    print(f"  Turn limit increased: {increased}")
    print(f"  Current max turns: {agent.max_turns_per_episode}")
    print(f"  Total increases: {agent.turn_limit_increases}")
    print()
    
    # Show final statistics
    if agent.critic_scores_history:
        avg_score = sum(agent.critic_scores_history) / len(agent.critic_scores_history)
        print(f"Final Statistics:")
        print(f"  Average critic score: {avg_score:.3f}")
        print(f"  Total critic evaluations: {len(agent.critic_scores_history)}")
        print(f"  Final max turns: {agent.max_turns_per_episode}")
        print(f"  Turn limit increases: {agent.turn_limit_increases}")
        print(f"  Potential additional turns earned: {agent.max_turns_per_episode - agent.base_max_turns_per_episode}")


if __name__ == "__main__":
    test_dynamic_turn_limits() 