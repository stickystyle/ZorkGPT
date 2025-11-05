"""Integration tests for Phase 1 loop break system.

This test suite validates that all three loop break phases work together:
- Phase 1A: Progress velocity detection
- Phase 1B: Location revisit penalty
- Phase 1C: Exploration guidance + stuck countdown warnings
"""

import pytest
from collections import deque
from unittest.mock import Mock, patch, MagicMock
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
from session.game_configuration import GameConfiguration


class TestLoopBreakIntegration:
    """Test that all three loop break phases work together."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            max_turns_stuck=40,
            stuck_check_interval=10,
            stuck_warning_threshold=20,
            enable_location_penalty=True,
            location_revisit_penalty=-0.2,
            enable_stuck_warnings=True,
            enable_exploration_hints=True,
            zork_game_workdir="/tmp/test_game_files",
        )

    @pytest.fixture
    def orchestrator(self, config):
        """Create test orchestrator."""
        with patch('orchestration.zork_orchestrator_v2.JerichoInterface'):
            orch = ZorkOrchestratorV2(episode_id="test_integration")
            orch.config = config
            return orch

    def test_all_phases_active_together(self, orchestrator):
        """Test that all three phases are active simultaneously."""
        # Phase 1A: Progress velocity
        assert orchestrator.config.max_turns_stuck == 40
        assert orchestrator.config.stuck_check_interval == 10

        # Phase 1B: Location revisit penalty
        assert orchestrator.config.enable_location_penalty is True
        assert orchestrator.config.location_revisit_penalty == -0.2

        # Phase 1C: Stuck warnings
        assert orchestrator.config.enable_stuck_warnings is True
        assert orchestrator.config.stuck_warning_threshold == 20
        assert orchestrator.config.enable_exploration_hints is True

    def test_stuck_episode_with_all_mechanisms(self, orchestrator):
        """Test a stuck episode with all three mechanisms active."""
        # Simulate stuck episode
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0

        # Track score for progress velocity
        orchestrator._track_score_for_progress_detection()

        # Simulate 20 turns stuck
        orchestrator.game_state.turn_count = 20

        # Check Phase 1C: Warnings should appear
        warning = orchestrator._build_stuck_countdown_warning()
        assert warning != ""
        assert "DIE in 20 turns" in warning

        # Simulate 40 turns stuck
        orchestrator.game_state.turn_count = 40

        # Check Phase 1A: Should detect stuck
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 40

        # Would terminate at turn 40 check
        assert turns_stuck >= orchestrator.config.max_turns_stuck

    def test_location_revisit_with_stuck_warnings(self, orchestrator):
        """Test that location penalties and stuck warnings coexist."""
        # Setup: Agent stuck and revisiting location
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25  # Stuck for 25 turns
        orchestrator.game_state.current_room_id = 20
        orchestrator.game_state.current_room_name = "Test Room"

        # Track score
        orchestrator._track_score_for_progress_detection()

        # Track location history (simulate revisits)
        # Window size is 5, checks last 5 locations excluding current
        # History: [15, 18, 20, 15, 20] - current is 20, check [15, 18, 20, 15]
        # Room 20 appears once in the window (excluding current)
        orchestrator._location_id_history = deque([15, 18, 20, 15, 20], maxlen=20)

        # Phase 1C: Warning should appear (25 > 20 threshold)
        warning = orchestrator._build_stuck_countdown_warning()
        assert warning != ""
        assert "DIE in 15 turns" in warning

        # Phase 1B: Revisit should be detected
        revisit_info = orchestrator._detect_location_revisit()
        assert revisit_info["detected"] is True
        assert revisit_info["recent_visits"] == 1  # Seen once in window (excluding current)

        # Apply penalty
        base_score = 0.8
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score, revisit_info
        )
        expected = 0.8 + (-0.2 * 1)  # 0.6
        assert adjusted_score == expected
        assert reason != ""
        assert "Location revisit penalty" in reason

    def test_warnings_disabled_does_not_break_velocity(self, orchestrator):
        """Test that disabling warnings doesn't break progress velocity."""
        orchestrator.config.enable_stuck_warnings = False

        # Setup stuck episode
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 40
        orchestrator._track_score_for_progress_detection()

        # Phase 1C: Warnings disabled
        warning = orchestrator._build_stuck_countdown_warning()
        assert warning == ""

        # Phase 1A: Still detects stuck
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 40

    def test_penalties_disabled_does_not_break_velocity(self, orchestrator):
        """Test that disabling penalties doesn't break progress velocity."""
        orchestrator.config.enable_location_penalty = False

        # Setup stuck episode with revisits
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 40
        orchestrator.game_state.current_room_id = 20
        orchestrator._location_id_history = deque([20, 20, 20, 20, 20], maxlen=20)

        orchestrator._track_score_for_progress_detection()

        # Phase 1B: Detection still works (penalty application is disabled separately)
        revisit_info = orchestrator._detect_location_revisit()
        # Detection happens regardless, but penalty won't be applied in apply function
        assert revisit_info["detected"] is True
        assert revisit_info["recent_visits"] == 4

        # Phase 1A: Still detects stuck
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 40

    def test_configuration_loading_from_toml(self, tmp_path):
        """Test that all loop break config loads from pyproject.toml."""
        from pathlib import Path
        config = GameConfiguration.from_toml(Path("pyproject.toml"))

        # Phase 1A config
        assert hasattr(config, 'max_turns_stuck')
        assert hasattr(config, 'stuck_check_interval')
        assert config.max_turns_stuck == 40
        assert config.stuck_check_interval == 10

        # Phase 1B config
        assert hasattr(config, 'enable_location_penalty')
        assert hasattr(config, 'location_revisit_penalty')
        assert hasattr(config, 'location_revisit_window')
        assert config.enable_location_penalty is True
        assert config.location_revisit_penalty == -0.2
        assert config.location_revisit_window == 5

        # Phase 1C config
        assert hasattr(config, 'enable_stuck_warnings')
        assert hasattr(config, 'stuck_warning_threshold')
        assert hasattr(config, 'action_novelty_window')
        assert config.enable_stuck_warnings is True
        assert config.stuck_warning_threshold == 20
        assert config.action_novelty_window == 15  # Actual value from pyproject.toml

    def test_all_mechanisms_work_independently(self, orchestrator):
        """Test that each mechanism can work independently of others."""
        # Scenario 1: Stuck but not revisiting locations
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25
        orchestrator._location_id_history = deque([1, 2, 3, 4, 5], maxlen=20)  # All unique

        orchestrator._track_score_for_progress_detection()

        # Phase 1A: Detects stuck
        assert orchestrator._get_turns_since_score_change() == 25

        # Phase 1B: No revisits
        revisit_info = orchestrator._detect_location_revisit()
        assert revisit_info["detected"] is False

        # Phase 1C: Shows warnings
        warning = orchestrator._build_stuck_countdown_warning()
        assert warning != ""

        # Scenario 2: Revisiting locations but making progress
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 30
        orchestrator._track_score_for_progress_detection()

        # Make progress
        orchestrator.game_state.previous_zork_score = 20
        orchestrator.game_state.turn_count = 31
        orchestrator._track_score_for_progress_detection()

        # Still revisiting
        orchestrator._location_id_history = deque([10, 20, 10, 20, 10], maxlen=20)

        # Phase 1A: Not stuck (score just changed at turn 31)
        # The tracking happens at turn 31, so turns_since_change should be 0
        assert orchestrator._get_turns_since_score_change() == 0

        # Phase 1B: Detects revisits
        revisit_info = orchestrator._detect_location_revisit()
        assert revisit_info["detected"] is True

        # Phase 1C: No warnings (not stuck long enough)
        warning = orchestrator._build_stuck_countdown_warning()
        assert warning == ""

    def test_stuck_termination_event_structure(self, orchestrator):
        """Test that stuck termination events include all relevant data."""
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 50
        orchestrator._track_score_for_progress_detection()

        turns_stuck = orchestrator._get_turns_since_score_change()

        # Build event data
        event = {
            "event_type": "stuck_termination",
            "turn": orchestrator.game_state.turn_count,
            "turns_stuck": turns_stuck,
            "score": orchestrator.game_state.previous_zork_score,
            "max_turns_stuck": orchestrator.config.max_turns_stuck,
        }

        # Validate structure
        assert event["event_type"] == "stuck_termination"
        assert event["turns_stuck"] == 50
        assert event["turns_stuck"] >= orchestrator.config.max_turns_stuck

    def test_location_penalty_event_structure(self, orchestrator):
        """Test that location penalty events include all relevant data."""
        orchestrator.game_state.current_room_id = 20
        orchestrator.game_state.current_room_name = "Kitchen"
        # History: [15, 20, 18, 20, 19, 20]
        # Current is 20 (last element), window checks [15, 20, 18, 20, 19] (5 elements)
        # Room 20 appears 2 times in window (excluding current)
        orchestrator._location_id_history = deque([15, 20, 18, 20, 19, 20], maxlen=20)

        revisit_info = orchestrator._detect_location_revisit()
        base_score = 0.9
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score, revisit_info
        )

        # Build event data
        event = {
            "event_type": "location_penalty_applied",
            "turn": orchestrator.game_state.turn_count,
            "location_id": orchestrator.game_state.current_room_id,
            "location_name": orchestrator.game_state.current_room_name,
            "recent_visits": revisit_info["recent_visits"],
            "base_score": base_score,
            "adjusted_score": adjusted_score,
            "penalty": adjusted_score - base_score,
        }

        # Validate structure
        assert event["event_type"] == "location_penalty_applied"
        assert event["location_id"] == 20
        assert event["location_name"] == "Kitchen"
        assert event["recent_visits"] == 2  # Appears 2 times in window
        assert event["penalty"] < 0
