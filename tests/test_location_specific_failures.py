"""
Integration tests for location-specific failure tracking system.
Tests the enhanced ActionRejectionSystem and LLM-based failure detection.
"""

import pytest
from unittest.mock import Mock, patch
from collections import Counter

from zork_critic import (
    ZorkCritic,
    ActionRejectionSystem,
    FailureDetectionResponse,
    CriticResponse,
)
from session.game_configuration import GameConfiguration


class TestActionRejectionSystem:
    """Test the enhanced ActionRejectionSystem with location-specific logic."""

    def setup_method(self):
        self.rejection_system = ActionRejectionSystem()

    def test_location_specific_failure_prevention(self):
        """Test that actions that failed in current location are blocked."""
        action = "go north"
        current_location = "Kitchen"
        failed_actions_by_location = {
            "Kitchen": {"go north", "open door"},
            "Living Room": {"go east"},
        }
        context = {}

        should_override, reason = self.rejection_system.should_override_rejection(
            action, current_location, failed_actions_by_location, context
        )

        # Should NOT override because action failed in current location
        assert not should_override
        assert reason == "action_failed_in_current_location"

    def test_location_specific_failure_allows_other_locations(self):
        """Test that actions can be tried in different locations even if they failed elsewhere."""
        action = "go east"
        current_location = "Kitchen"
        failed_actions_by_location = {
            "Living Room": {"go east"},  # Failed in different location
            "Kitchen": {"go north"},  # Current location has different failures
        }
        context = {
            "recent_locations": ["Kitchen"] * 3,  # Not enough for loop detection
            "recent_actions": ["look", "examine table"],
            "previous_actions_and_responses": [],
            "turns_since_movement": 1,
            "critic_confidence": 0.6,
        }

        should_override, reason = self.rejection_system.should_override_rejection(
            action, current_location, failed_actions_by_location, context
        )

        # Should check other conditions (not immediately blocked)
        # This will go through to _check_other_override_conditions
        # Since we have limited context, it should return False
        assert not should_override

    def test_novel_action_in_current_location(self):
        """Test that novel actions in current location can be overridden."""
        action = "examine chandelier"
        current_location = "Ballroom"
        failed_actions_by_location = {
            "Ballroom": {"go north", "go south"},  # Different actions failed
            "Kitchen": {"examine chandelier"},  # Same action failed elsewhere
        }
        context = {
            "recent_locations": ["Ballroom"] * 6,  # Enough for loop detection
            "recent_actions": [
                "go north",
                "go south",
                "look",
                "go north",
                "go south",
                "look",
            ],
            "previous_actions_and_responses": [
                ("go north", "There is a wall there."),
                ("go south", "You can't go that way."),
                ("look", "You are in a ballroom."),
            ],
            "turns_since_movement": 3,
            "critic_confidence": 0.5,
            "current_location": current_location,
            "failed_actions_by_location": failed_actions_by_location,
        }

        should_override, reason = self.rejection_system.should_override_rejection(
            action, current_location, failed_actions_by_location, context
        )

        # Should override due to novel action logic in _check_other_override_conditions
        assert should_override
        assert "novel_action" in reason

    def test_loop_detection_with_location_specific_failures(self):
        """Test that loop detection works with location-specific failure tracking."""
        action = "look"  # Use an action that's already been repeated
        current_location = "Garden"
        failed_actions_by_location = {
            "Garden": {"go north"},  # One action failed here
        }
        context = {
            "recent_locations": ["Garden"] * 6,  # Loop detected
            "recent_actions": [
                "go north",
                "go north",
                "go north",
                "look",
                "go north",
                "look",
            ],
            "previous_actions_and_responses": [
                ("go north", "There is a wall there."),
                ("go north", "There is a wall there."),
                ("go north", "There is a wall there."),
                ("look", "You are in a garden."),
                ("go north", "There is a wall there."),
                ("look", "You are in a garden."),
            ],
            "turns_since_movement": 4,
            "critic_confidence": 0.5,  # Add this to prevent other overrides
        }

        should_override, reason = self.rejection_system.should_override_rejection(
            action, current_location, failed_actions_by_location, context
        )

        # Should override due to being stuck in a loop or trying a novel action
        assert should_override
        assert any(
            word in reason for word in ["loop", "diversity", "progress", "novel_action"]
        )


class TestFailureDetection:
    """Test the LLM-based failure detection system."""

    def setup_method(self):
        # Mock the LLM client and other dependencies
        self.mock_client = Mock()
        self.mock_logger = Mock()

        # Create critic with config
        config = GameConfiguration.from_toml()
        self.critic = ZorkCritic(
            config=config,
            client=self.mock_client,
            logger=self.mock_logger,
            episode_id="test_episode",
        )

    def test_detect_parser_failure(self):
        """Test detection of parser failures."""
        action = "flibbergibbet"
        game_response = "I don't understand that."

        # Mock LLM response indicating failure
        mock_response = Mock()
        mock_response.content = (
            '{"action_failed": true, "reason": "Parser did not understand the command"}'
        )

        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.critic.detect_action_failure(action, game_response)

        assert isinstance(result, FailureDetectionResponse)
        assert result.action_failed is True
        assert (
            "parser" in result.reason.lower() or "understand" in result.reason.lower()
        )

    def test_detect_blocked_movement(self):
        """Test detection of blocked movement."""
        action = "go north"
        game_response = "There is a wall there."

        # Mock LLM response indicating failure
        mock_response = Mock()
        mock_response.content = (
            '{"action_failed": true, "reason": "Movement blocked by wall"}'
        )

        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.critic.detect_action_failure(action, game_response)

        assert result.action_failed is True
        assert "wall" in result.reason.lower() or "blocked" in result.reason.lower()

    def test_detect_successful_action(self):
        """Test detection of successful actions."""
        action = "take lamp"
        game_response = "Taken."

        # Mock LLM response indicating success
        mock_response = Mock()
        mock_response.content = (
            '{"action_failed": false, "reason": "Action succeeded - item was taken"}'
        )

        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.critic.detect_action_failure(action, game_response)

        assert result.action_failed is False
        assert "succeed" in result.reason.lower() or "taken" in result.reason.lower()

    def test_detect_informational_action(self):
        """Test that informational actions are not marked as failures."""
        action = "look"
        game_response = "You are in a kitchen. There is a table here."

        # Mock LLM response indicating success (informational)
        mock_response = Mock()
        mock_response.content = '{"action_failed": false, "reason": "Action provided useful information about the location"}'

        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.critic.detect_action_failure(action, game_response)

        assert result.action_failed is False
        assert "information" in result.reason.lower()

    def test_failure_detection_error_handling(self):
        """Test that LLM errors default to action succeeded."""
        action = "examine table"
        game_response = "The table is wooden."

        # Mock LLM call failure
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = self.critic.detect_action_failure(action, game_response)

        # Should default to action succeeded on error
        assert result.action_failed is False
        assert "error" in result.reason.lower()


class TestEnhancedCriticEvaluation:
    """Test the enhanced critic evaluation with location-specific context."""

    def setup_method(self):
        self.mock_client = Mock()
        self.mock_logger = Mock()

        # Create critic with config
        config = GameConfiguration.from_toml()
        self.critic = ZorkCritic(
            config=config,
            client=self.mock_client,
            logger=self.mock_logger,
            episode_id="test_episode",
        )

    def test_location_specific_context_in_evaluation(self):
        """Test that location-specific failure context is included in critic evaluation."""
        # Mock critic response
        mock_response = Mock()
        mock_response.content = '{"score": -0.8, "justification": "Action failed in this location before", "confidence": 0.9}'

        self.mock_client.chat.completions.create.return_value = mock_response

        game_state = "You are in a dark cave."
        action = "go north"
        current_location = "Dark Cave"
        failed_actions_by_location = {
            "Dark Cave": {"go north", "go east"},
            "Other Room": {"go south"},
        }
        action_counts = Counter({"go north": 3})

        result = self.critic.evaluate_action(
            game_state_text=game_state,
            proposed_action=action,
            action_counts=action_counts,
            current_location_name=current_location,
            failed_actions_by_location=failed_actions_by_location,
        )

        # Verify the call was made
        assert self.mock_client.chat.completions.create.called

        # Check that the prompt included location-specific context
        call_args = self.mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_prompt = messages[-1]["content"]

        # Should mention that the action failed in this specific location
        assert "FAILED" in user_prompt
        assert current_location in user_prompt
        assert action in user_prompt

        # Check the result
        assert isinstance(result, CriticResponse)
        assert result.score == -0.8


class TestIntegrationWithOrchestrator:
    """Test integration with ZorkOrchestrator workflow."""

    def test_failed_actions_tracking_workflow(self):
        """Test the complete workflow of failed actions tracking."""
        # This would require a more complex setup with a mock orchestrator
        # For now, we'll test the key components individually

        rejection_system = ActionRejectionSystem()

        # Simulate a sequence of actions in the same location
        location = "Kitchen"
        failed_actions_by_location = {}

        # First attempt at "go north" - would fail and be added to tracking
        failed_actions_by_location[location] = {"go north"}

        # Second attempt should be blocked
        action = "go north"
        context = {"recent_locations": [location] * 2}

        should_override, reason = rejection_system.should_override_rejection(
            action, location, failed_actions_by_location, context
        )

        assert not should_override
        assert reason == "action_failed_in_current_location"

        # But same action in different location should be allowed to proceed to other checks
        different_location = "Living Room"
        should_override, reason = rejection_system.should_override_rejection(
            action, different_location, failed_actions_by_location, context
        )

        # This will go to other override conditions since it's not blocked by location-specific failures
        # The exact result depends on the context, but it shouldn't be immediately blocked
        assert reason != "action_failed_in_current_location"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
