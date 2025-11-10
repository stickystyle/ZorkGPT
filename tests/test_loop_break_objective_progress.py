"""Unit tests for Phase 4: Objective-Based Progress Tracking.

Tests the implementation of objective completion as progress metric alongside score changes.
Validates the OR logic: progress = (score change OR objective completion) within 40-turn window.

Specification: ideas/objective_progress.md Section 6 (Testing Strategy)
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
from session.game_configuration import GameConfiguration
from session.game_state import GameState


@pytest.fixture
def orchestrator_with_objectives(tmp_path):
    """Create orchestrator for testing progress tracking with objectives enabled."""
    # Create config with objective-based progress enabled
    config = GameConfiguration(
        max_turns_per_episode=100,
        game_file_path="test.z5",
        max_turns_stuck=40,
        stuck_check_interval=10,
        stuck_warning_threshold=20,
        enable_objective_based_progress=True,  # Feature enabled
        zork_game_workdir=str(tmp_path),
    )

    # Mock Jericho interface to avoid needing actual game file
    with patch('orchestration.zork_orchestrator_v2.JerichoInterface'):
        orch = ZorkOrchestratorV2(episode_id="test_objective_progress")
        orch.config = config

        # Initialize tracking variables
        orch._last_score_change_turn = 0
        orch._last_tracked_score = 0
        orch.game_state.previous_zork_score = 0
        orch.game_state.turn_count = 0

        return orch


@pytest.fixture
def orchestrator_without_objectives(tmp_path):
    """Create orchestrator for testing progress tracking with objectives disabled."""
    config = GameConfiguration(
        max_turns_per_episode=100,
        game_file_path="test.z5",
        max_turns_stuck=40,
        stuck_check_interval=10,
        stuck_warning_threshold=20,
        enable_objective_based_progress=False,  # Feature disabled
        zork_game_workdir=str(tmp_path),
    )

    with patch('orchestration.zork_orchestrator_v2.JerichoInterface'):
        orch = ZorkOrchestratorV2(episode_id="test_no_objectives")
        orch.config = config

        # Initialize tracking variables
        orch._last_score_change_turn = 0
        orch._last_tracked_score = 0
        orch.game_state.previous_zork_score = 0
        orch.game_state.turn_count = 0

        return orch


def create_completed_objective(objective: str, turn: int) -> dict:
    """Helper to create a properly formatted completed objective dict."""
    return {
        "objective": objective,
        "completed_turn": turn,
        "completion_action": "test action",
        "completion_response": "test response",
        "completion_location": "Test Room",
        "completion_score": 10,
    }


class TestObjectiveBasedProgress:
    """Test suite for objective-based progress tracking."""

    def test_score_increase_resets_timer(self, orchestrator_with_objectives):
        """Test that score increases reset the stuck timer (existing behavior)."""
        # Arrange: Simulate being stuck for 20 turns
        orchestrator_with_objectives.game_state.turn_count = 20
        orchestrator_with_objectives.game_state.previous_zork_score = 0
        orchestrator_with_objectives._last_score_change_turn = 0
        orchestrator_with_objectives._last_tracked_score = 0

        # Act: Score increases
        orchestrator_with_objectives.game_state.previous_zork_score = 10
        orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Timer resets (should show 0 turns stuck)
        turns_stuck = orchestrator_with_objectives._get_turns_since_score_change()
        assert turns_stuck == 0, f"Expected timer to reset to 0, got {turns_stuck}"

    def test_objective_completion_resets_timer(self, orchestrator_with_objectives):
        """Test that objective completion resets the stuck timer (new behavior)."""
        # Arrange: Simulate being stuck for 30 turns with no score change
        orchestrator_with_objectives.game_state.turn_count = 30
        orchestrator_with_objectives.game_state.previous_zork_score = 0
        orchestrator_with_objectives._last_score_change_turn = 0
        orchestrator_with_objectives._last_tracked_score = 0

        # Act: Complete an objective at turn 30 (this turn)
        orchestrator_with_objectives.game_state.completed_objectives = [
            create_completed_objective("Test objective", 30)
        ]
        orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Timer resets due to objective completion
        turns_stuck = orchestrator_with_objectives._get_turns_since_score_change()
        assert turns_stuck == 0, f"Expected timer to reset to 0 due to objective, got {turns_stuck}"

    def test_both_score_and_objective_reset_timer(self, orchestrator_with_objectives):
        """Test that both score and objective completion in same turn reset timer (OR logic)."""
        # Arrange: Simulate being stuck for 25 turns
        orchestrator_with_objectives.game_state.turn_count = 25
        orchestrator_with_objectives.game_state.previous_zork_score = 0
        orchestrator_with_objectives._last_score_change_turn = 0
        orchestrator_with_objectives._last_tracked_score = 0

        # Act: Both score increases and objective completes at turn 25
        orchestrator_with_objectives.game_state.previous_zork_score = 10
        orchestrator_with_objectives.game_state.completed_objectives = [
            create_completed_objective("Test objective", 25)
        ]
        orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Timer resets (either condition sufficient)
        turns_stuck = orchestrator_with_objectives._get_turns_since_score_change()
        assert turns_stuck == 0, f"Expected timer to reset with both progress types, got {turns_stuck}"

    def test_no_progress_increments_timer(self, orchestrator_with_objectives):
        """Test that no score increase and no objective completion increments stuck timer."""
        # Arrange: Initial state at turn 0
        orchestrator_with_objectives.game_state.turn_count = 0
        orchestrator_with_objectives.game_state.previous_zork_score = 10
        orchestrator_with_objectives._last_score_change_turn = 0
        orchestrator_with_objectives._last_tracked_score = 10
        orchestrator_with_objectives.game_state.completed_objectives = []

        # Act: Advance 15 turns with no progress
        orchestrator_with_objectives.game_state.turn_count = 15
        orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Timer shows 15 turns stuck
        turns_stuck = orchestrator_with_objectives._get_turns_since_score_change()
        assert turns_stuck == 15, f"Expected 15 turns stuck, got {turns_stuck}"

    def test_fallback_to_score_only_when_no_objectives(self, orchestrator_with_objectives):
        """Test that empty objectives list falls back to score-only tracking."""
        # Arrange: No objectives exist, stuck for 20 turns
        orchestrator_with_objectives.game_state.turn_count = 20
        orchestrator_with_objectives.game_state.previous_zork_score = 10
        orchestrator_with_objectives._last_score_change_turn = 0
        orchestrator_with_objectives._last_tracked_score = 10
        orchestrator_with_objectives.game_state.completed_objectives = []  # Empty list

        # Act: Track progress
        orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Timer continues (no objective progress to reset it)
        turns_stuck = orchestrator_with_objectives._get_turns_since_score_change()
        assert turns_stuck == 20, f"Expected 20 turns stuck with no objectives, got {turns_stuck}"

    def test_feature_flag_disabled_ignores_objectives(self, orchestrator_without_objectives):
        """Test that feature flag disabled causes objective completions to be ignored."""
        # Arrange: Objective completed at turn 10, now at turn 30 (20 turns since objective)
        orchestrator_without_objectives.game_state.turn_count = 30
        orchestrator_without_objectives.game_state.previous_zork_score = 10
        orchestrator_without_objectives._last_score_change_turn = 0
        orchestrator_without_objectives._last_tracked_score = 10
        orchestrator_without_objectives.game_state.completed_objectives = [
            create_completed_objective("Ignored objective", 10)
        ]

        # Act: Track progress (should ignore objective)
        orchestrator_without_objectives._track_score_for_progress_detection()

        # Assert: Timer continues (objective ignored due to feature flag)
        turns_stuck = orchestrator_without_objectives._get_turns_since_score_change()
        assert turns_stuck == 30, f"Expected 30 turns stuck (objectives ignored), got {turns_stuck}"

    def test_objective_removal_does_not_count_as_progress(self, orchestrator_with_objectives):
        """Test that objective removal doesn't reset the stuck timer.

        Note: Only discovered_objectives can be removed (via staleness).
        Completed objectives persist forever and continue counting as progress.
        """
        # Arrange: One objective completed at turn 5, now at turn 10
        orchestrator_with_objectives.game_state.turn_count = 10
        orchestrator_with_objectives.game_state.previous_zork_score = 10
        orchestrator_with_objectives._last_score_change_turn = 0
        orchestrator_with_objectives._last_tracked_score = 10
        orchestrator_with_objectives.game_state.completed_objectives = [
            create_completed_objective("Keep me", 5)
        ]

        # Act: Track progress with the objective still present
        orchestrator_with_objectives._track_score_for_progress_detection()
        turns_stuck_before = orchestrator_with_objectives._get_turns_since_score_change()

        # Simulate progression to turn 50 with objective still present
        # (completed objectives are never removed)
        orchestrator_with_objectives.game_state.turn_count = 50
        orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Timer continues (old objective completion outside window)
        turns_stuck_after = orchestrator_with_objectives._get_turns_since_score_change()
        assert turns_stuck_after == 50, (
            f"Expected 50 turns stuck (old objective outside window), got {turns_stuck_after}"
        )

        # Validate the objective is still present (never removed)
        assert len(orchestrator_with_objectives.game_state.completed_objectives) == 1, (
            "Completed objectives should never be removed"
        )

    def test_objective_completion_within_window(self, orchestrator_with_objectives, caplog):
        """Test objective completion within 40-turn window counts as progress.

        This test validates the window-based progress detection logic by checking
        the objective_progress flag in debug logs, not timer resets (which only
        occur for very recent completions within 1 turn).
        """
        # Scenario 1: 30 turns ago (within window)
        orchestrator_with_objectives.game_state.turn_count = 40
        orchestrator_with_objectives.game_state.previous_zork_score = 10
        orchestrator_with_objectives._last_score_change_turn = 0
        orchestrator_with_objectives._last_tracked_score = 10
        orchestrator_with_objectives.game_state.completed_objectives = [
            create_completed_objective("Recent objective", 10)  # Completed 30 turns ago
        ]

        # Act: Track progress - capture logs from the 'zorkgpt' logger
        with caplog.at_level(logging.DEBUG, logger='zorkgpt'):
            caplog.clear()
            orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: objective_progress=True in logs (within window)
        assert "objective_progress=True" in caplog.text, (
            f"Expected objective_progress=True for objective 30 turns ago (within 40-turn window). "
            f"Got logs: {caplog.text[:500]}"
        )
        assert "progress_made=True" in caplog.text, (
            "Expected progress_made=True when objective within window"
        )

        # Scenario 2: 39 turns ago (edge of window)
        orchestrator_with_objectives.game_state.turn_count = 42
        orchestrator_with_objectives.game_state.completed_objectives = [
            create_completed_objective("Edge objective", 3)  # Completed 39 turns ago
        ]

        with caplog.at_level(logging.DEBUG, logger='zorkgpt'):
            caplog.clear()
            orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Still within window (42 - 3 = 39 turns ago)
        assert "objective_progress=True" in caplog.text, (
            f"Expected objective_progress=True for objective 39 turns ago (at edge of window). "
            f"Got logs: {caplog.text[:500]}"
        )
        assert "progress_made=True" in caplog.text, (
            "Expected progress_made=True when objective at edge of window"
        )

        # Scenario 3: 41 turns ago (outside window)
        orchestrator_with_objectives.game_state.turn_count = 42
        orchestrator_with_objectives._last_score_change_turn = 0  # Reset to force stuck state
        orchestrator_with_objectives.game_state.completed_objectives = [
            create_completed_objective("Old objective", 1)  # Completed 41 turns ago
        ]

        with caplog.at_level(logging.DEBUG, logger='zorkgpt'):
            caplog.clear()
            orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Outside window (42 - 1 = 41 turns ago)
        assert "objective_progress=False" in caplog.text, (
            f"Expected objective_progress=False for objective 41 turns ago (outside 40-turn window). "
            f"Got logs: {caplog.text[:500]}"
        )
        assert "progress_made=False" in caplog.text, (
            "Expected progress_made=False when no progress in window"
        )

    def test_logging_includes_both_metrics(self, orchestrator_with_objectives, caplog):
        """Verify logging includes both score_progress and objective_progress flags."""
        # Arrange: Setup state with both metrics available
        orchestrator_with_objectives.game_state.turn_count = 20
        orchestrator_with_objectives.game_state.previous_zork_score = 10
        orchestrator_with_objectives._last_score_change_turn = 0
        orchestrator_with_objectives._last_tracked_score = 10
        orchestrator_with_objectives.game_state.completed_objectives = [
            create_completed_objective("Log test", 10)
        ]

        # Act: Track progress with logging enabled
        with caplog.at_level(logging.DEBUG, logger='zorkgpt'):
            caplog.clear()
            orchestrator_with_objectives._track_score_for_progress_detection()

        # Assert: Log contains both metrics
        log_text = caplog.text
        assert "score_progress=" in log_text, f"Log should include score_progress flag. Got: {log_text[:500]}"
        assert "objective_progress=" in log_text, f"Log should include objective_progress flag. Got: {log_text[:500]}"
        assert "progress_made=" in log_text, f"Log should include progress_made flag. Got: {log_text[:500]}"
        assert "turns_stuck=" in log_text, f"Log should include turns_stuck metric. Got: {log_text[:500]}"
