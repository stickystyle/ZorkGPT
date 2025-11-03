"""
Test that critic does not reject valid exits from available exits list.

This test validates the fix for the issue where the critic was incorrectly
rejecting valid movement commands (e.g., "north") even when they appeared
in the available_exits list.
"""

import pytest
from zork_critic import ZorkCritic


class TestCriticValidExits:
    """Test critic handling of valid exits."""

    def test_critic_approves_exit_in_available_exits_list(self, test_config):
        """Critic should approve movements that are in the available exits list."""
        critic = ZorkCritic(config=test_config)

        # Simulate the exact scenario from the bug report
        game_state = (
            "West of House You are standing in an open field west of a white house, "
            "with a boarded front door. There is a small mailbox here."
        )
        proposed_action = "north"
        available_exits = ["north", "south", "west"]

        # Evaluate the action
        response = critic.evaluate_action(
            game_state_text=game_state,
            proposed_action=proposed_action,
            available_exits=available_exits,
        )

        # The critic should approve this action with a positive score
        assert response.score > 0, (
            f"Critic rejected valid exit 'north' with score {response.score}. "
            f"Justification: {response.justification}"
        )
        assert response.score >= 0.5, (
            f"Critic score for valid exit should be >= 0.5, got {response.score}"
        )

    def test_critic_rejects_exit_not_in_available_exits_list(self, test_config):
        """Critic should reject movements that are NOT in the available exits list."""
        critic = ZorkCritic(config=test_config)

        game_state = (
            "West of House You are standing in an open field west of a white house, "
            "with a boarded front door. There is a small mailbox here."
        )
        proposed_action = "east"  # Not in available exits
        available_exits = ["north", "south", "west"]

        response = critic.evaluate_action(
            game_state_text=game_state,
            proposed_action=proposed_action,
            available_exits=available_exits,
        )

        # The critic should reject this action with a negative score
        assert response.score < 0, (
            f"Critic approved invalid exit 'east' with score {response.score}. "
            f"Justification: {response.justification}"
        )
        assert response.score <= -0.5, (
            f"Critic score for invalid exit should be <= -0.5, got {response.score}"
        )

    def test_critic_approves_all_valid_cardinal_directions(self, test_config):
        """Critic should approve all cardinal directions when they're in the exits list."""
        critic = ZorkCritic(config=test_config)

        game_state = "Junction You are at a crossroads."
        available_exits = ["north", "south", "east", "west"]

        for direction in available_exits:
            response = critic.evaluate_action(
                game_state_text=game_state,
                proposed_action=direction,
                available_exits=available_exits,
            )

            assert response.score > 0, (
                f"Critic rejected valid exit '{direction}' with score {response.score}. "
                f"Justification: {response.justification}"
            )

    def test_critic_doesnt_leak_ground_truth_in_justification(self, test_config):
        """Critic should not reveal ground-truth info in justifications."""
        critic = ZorkCritic(config=test_config)

        game_state = "West of House"
        proposed_action = "east"  # Invalid exit
        available_exits = ["north", "south", "west"]

        response = critic.evaluate_action(
            game_state_text=game_state,
            proposed_action=proposed_action,
            available_exits=available_exits,
        )

        # Check that justification doesn't leak ground-truth
        justification_lower = response.justification.lower()
        forbidden_phrases = [
            "available exits",
            "ground truth",
            "game engine",
            "will definitely fail",
            "guaranteed",
            "confirmed",
        ]

        for phrase in forbidden_phrases:
            assert phrase not in justification_lower, (
                f"Critic justification leaked ground-truth info: '{phrase}' found in "
                f"'{response.justification}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
