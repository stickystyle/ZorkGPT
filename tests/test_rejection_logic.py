"""Test rejection logic implementation."""

from unittest.mock import Mock
from managers.rejection_manager import RejectionManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
import logging


class TestRejectionManager:
    """Test the RejectionManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.config = Mock(spec=GameConfiguration)
        self.config.critic_rejection_threshold = -0.2
        self.game_state = GameState()
        self.rejection_manager = RejectionManager(
            self.logger, self.config, self.game_state
        )

    def test_rejection_threshold_calculation(self):
        """Test that rejection threshold is calculated correctly based on trust level."""
        # Default trust level is 0.8
        assert self.rejection_manager.get_rejection_threshold() == -0.2 * 0.8

        # Lower trust level
        self.rejection_manager.state.trust_level = 0.5
        assert self.rejection_manager.get_rejection_threshold() == -0.2 * 0.5

    def test_trust_update(self):
        """Test trust level updates based on rejection outcomes."""
        # Add some correct rejections
        for _ in range(10):
            self.rejection_manager.update_trust(was_rejection_correct=True)

        # Trust should be high
        assert self.rejection_manager.state.trust_level > 0.8
        assert self.rejection_manager.state.correct_rejections == 10

        # Add some incorrect rejections
        for _ in range(10):
            self.rejection_manager.update_trust(was_rejection_correct=False)

        # Trust should decrease
        assert self.rejection_manager.state.trust_level < 0.8
        assert self.rejection_manager.state.incorrect_rejections == 10

    def test_should_override_rejection_failed_action(self):
        """Test that rejection is not overridden for previously failed actions."""
        action = "take sword"
        location = "West of House"
        failed_actions = {location: {"take sword"}}

        should_override, reason = self.rejection_manager.should_override_rejection(
            action=action,
            current_location=location,
            failed_actions_by_location=failed_actions,
            context={},
        )

        assert not should_override
        assert reason == "action_failed_in_current_location"

    def test_should_override_rejection_exploring_new_locations(self):
        """Test override when exploring new locations."""
        context = {
            "recent_locations": ["Room1", "Room2", "Room3", "Room4", "Room5"],
            "recent_actions": ["north", "south", "east", "west", "up"],
            "previous_actions_and_responses": [],
            "turns_since_movement": 2,
        }

        should_override, reason = self.rejection_manager.should_override_rejection(
            action="go north",
            current_location="Room5",
            failed_actions_by_location={},
            context=context,
        )

        assert should_override
        assert reason == "exploring_new_locations"

    def test_movement_tracking(self):
        """Test movement tracking updates."""
        assert self.rejection_manager.state.turns_since_movement == 0

        # No movement
        self.rejection_manager.update_movement_tracking(moved=False)
        assert self.rejection_manager.state.turns_since_movement == 1

        # Movement occurred
        self.rejection_manager.update_movement_tracking(moved=True)
        assert self.rejection_manager.state.turns_since_movement == 0

    def test_rejected_actions_tracking(self):
        """Test tracking of rejected actions."""
        self.rejection_manager.start_new_turn()
        assert len(self.rejection_manager.rejected_actions_this_turn) == 0

        self.rejection_manager.add_rejected_action("take lamp", -0.5, "Bad action")
        assert len(self.rejection_manager.rejected_actions_this_turn) == 1
        assert self.rejection_manager.rejected_actions_this_turn[0] == "take lamp"

        # New turn should clear rejected actions
        self.rejection_manager.start_new_turn()
        assert len(self.rejection_manager.rejected_actions_this_turn) == 0

    def test_state_export_and_restore(self):
        """Test state export and restoration."""
        # Set up some state
        self.rejection_manager.state.trust_level = 0.6
        self.rejection_manager.state.turns_since_movement = 5
        self.rejection_manager.add_rejected_action("go north", -0.3, "Blocked")

        # Export state
        exported = self.rejection_manager.get_state_for_export()
        assert exported["trust_level"] == 0.6
        assert exported["turns_since_movement"] == 5
        assert len(exported["rejected_actions_this_turn"]) == 1

        # Reset and restore
        self.rejection_manager.reset_episode()
        assert self.rejection_manager.state.trust_level == 0.8  # Reset to default

        self.rejection_manager.restore_state(exported)
        assert self.rejection_manager.state.trust_level == 0.6
        assert self.rejection_manager.state.turns_since_movement == 5
