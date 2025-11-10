"""Integration tests for Phase 5: Objective-Based Progress Tracking.

This test suite validates end-to-end behavior of the loop break system
with objective-based progress tracking. Tests cover:
- Episode continuation with objective completions
- Episode termination without any progress
- Warning message adaptation based on objectives
- Mixed progress scenarios (score + objectives)
- Integration of all three loop break phases
"""

import pytest
from collections import deque
from unittest.mock import Mock, patch
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
from session.game_configuration import GameConfiguration
from session.game_state import GameState


class TestObjectiveBasedProgressIntegration:
    """Integration tests for objective-based progress tracking."""

    @pytest.fixture
    def config(self):
        """Create test configuration with objective-based progress enabled."""
        return GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            max_turns_stuck=40,
            stuck_check_interval=10,
            stuck_warning_threshold=20,
            enable_objective_based_progress=True,
            enable_location_penalty=True,
            location_revisit_penalty=-0.2,
            location_revisit_window=5,
            enable_stuck_warnings=True,
            enable_exploration_hints=True,
            zork_game_workdir="/tmp/test_game_files",
        )

    @pytest.fixture
    def orchestrator(self, config):
        """Create test orchestrator with mocked dependencies."""
        with patch('orchestration.zork_orchestrator_v2.JerichoInterface'):
            orch = ZorkOrchestratorV2(episode_id="test_objective_integration")
            orch.config = config
            # Initialize tracking variables
            orch._last_score_change_turn = 0
            orch._last_tracked_score = 0
            return orch

    def simulate_turns(
        self,
        orchestrator,
        num_turns,
        start_turn=None,
        score=None,
        objectives=None
    ):
        """Helper to simulate multiple turns with optional state changes.

        Args:
            orchestrator: The orchestrator instance
            num_turns: Number of turns to simulate
            start_turn: Starting turn number (default: current turn + 1)
            score: Score to set (if None, keeps current score)
            objectives: List of objectives to add to completed_objectives
        """
        if start_turn is None:
            start_turn = orchestrator.game_state.turn_count + 1

        for i in range(num_turns):
            turn = start_turn + i
            orchestrator.game_state.turn_count = turn

            if score is not None:
                orchestrator.game_state.previous_zork_score = score

            if objectives is not None:
                orchestrator.game_state.completed_objectives = objectives

            # Track progress each turn
            orchestrator._track_score_for_progress_detection()

    def create_completed_objective(self, objective_text, turn_num):
        """Create a completed objective record."""
        return {
            "objective": objective_text,
            "completed_turn": turn_num,
            "completion_action": "test action",
            "completion_response": "test response",
            "completion_location": "Test Location",
            "completion_score": 10
        }

    def test_episode_continues_with_objective_completions(self, orchestrator):
        """Test that objective completion prevents termination.

        Scenario:
        - 30 turns without score increase
        - Complete an objective at turn 31
        - Episode continues (stuck timer resets)
        - Continue for 20 more turns with no progress
        - Warnings start appearing (turns_stuck=20)
        """
        # Setup: Initialize with score of 10
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()

        # Phase 1: 30 turns without score increase
        self.simulate_turns(orchestrator, 30, start_turn=1, score=10)

        # Verify we're stuck for 30 turns
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 30, "Should be stuck for 30 turns"

        # Phase 2: Complete an objective at turn 31
        objective = self.create_completed_objective("explore north", turn_num=31)
        orchestrator.game_state.completed_objectives = [objective]
        orchestrator.game_state.turn_count = 31
        orchestrator._track_score_for_progress_detection()

        # Verify stuck timer reset
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 0, "Stuck timer should reset after objective completion"

        # Phase 3: Continue for 21 more turns without progress
        # Note: The objective at turn 31 is considered "recent" (within 1 turn) at turn 32,
        # so the timer gets reset again at turn 32. We need to go to turn 52 to get 20 turns stuck.
        self.simulate_turns(orchestrator, 21, start_turn=32, score=10)

        # Verify warnings start appearing (turn 52 is 20 turns after turn 32's reset)
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 20, "Should be stuck for 20 turns after reset"

        warning = orchestrator._build_stuck_countdown_warning()
        assert warning != "", "Warning should appear at 20 turns stuck"
        assert "20 turns" in warning, "Warning should mention 20 turns remaining"

    def test_episode_terminates_without_any_progress(self, orchestrator):
        """Test episode terminates after 40 turns without progress.

        Scenario:
        - 40 turns without score increase or objective completion
        - Expected: Episode should be ready to terminate at turn 40
        """
        # Setup: Initialize with score of 10
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()

        # Simulate 40 turns without any progress
        self.simulate_turns(orchestrator, 40, start_turn=1, score=10)

        # Check if stuck timer reached termination threshold
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck >= orchestrator.config.max_turns_stuck, \
            f"Should be stuck for {orchestrator.config.max_turns_stuck}+ turns (got {turns_stuck})"

    def test_warnings_mention_objectives_when_present(self, orchestrator):
        """Test warnings mention objectives when they exist.

        Scenario:
        - 20 turns stuck with 3 active objectives
        - Expected: Warning includes "score or complete an objective" and lists objectives
        """
        # Setup: 3 active objectives
        orchestrator.game_state.discovered_objectives = [
            "explore north to Location 81",
            "open the window",
            "find the treasure"
        ]

        # Setup: Initialize and simulate 20 turns stuck
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()

        self.simulate_turns(orchestrator, 20, start_turn=1, score=10)

        # Get warning message
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify warning mentions both paths
        assert warning != "", "Warning should be present"
        assert "score or complete an objective" in warning.lower(), \
            "Warning should mention both progress paths"

        # Verify objectives are listed
        assert "CURRENT OBJECTIVES:" in warning or "objectives:" in warning.lower(), \
            "Warning should list current objectives"
        assert "explore north to Location 81" in warning, \
            "Warning should include first objective"

    def test_warnings_omit_objectives_when_absent(self, orchestrator):
        """Test warnings omit objectives when none exist.

        Scenario:
        - 20 turns stuck with no objectives
        - Expected: Warning includes "increase your score" only (no objective mention)
        """
        # Setup: No objectives
        orchestrator.game_state.discovered_objectives = []

        # Setup: Initialize and simulate 20 turns stuck
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()

        self.simulate_turns(orchestrator, 20, start_turn=1, score=10)

        # Get warning message
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify warning mentions only score
        assert warning != "", "Warning should be present"
        assert "increase your score" in warning.lower(), \
            "Warning should mention score increases"

        # Verify objectives are NOT mentioned
        assert "complete an objective" not in warning.lower(), \
            "Warning should NOT mention objectives when none exist"
        assert "CURRENT OBJECTIVES:" not in warning, \
            "Warning should NOT list objectives when none exist"

    def test_mixed_progress_scenario(self, orchestrator):
        """Test realistic alternating progress pattern.

        Scenario:
        - Turn 10: Score increase (reset)
        - Turn 30: No progress for 20 turns (warning appears)
        - Turn 35: Objective completion (reset)
        - Turn 55: No progress for 20 turns (warning appears)
        - Turn 75: Score increase (reset)
        - Expected: Episode continues past turn 75
        """
        # Initialize
        orchestrator.game_state.previous_zork_score = 0
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()

        # Phase 1: Turns 1-10, score increases at turn 10
        self.simulate_turns(orchestrator, 9, start_turn=1, score=0)
        orchestrator.game_state.turn_count = 10
        orchestrator.game_state.previous_zork_score = 5
        orchestrator._track_score_for_progress_detection()

        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 0, "Stuck timer should reset after score increase"

        # Phase 2: Turns 11-30, no progress
        self.simulate_turns(orchestrator, 20, start_turn=11, score=5)

        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 20, "Should be stuck for 20 turns"

        warning = orchestrator._build_stuck_countdown_warning()
        assert warning != "", "Warning should appear at 20 turns stuck"

        # Phase 3: Turn 35, objective completion
        objective = self.create_completed_objective("open door", turn_num=35)
        orchestrator.game_state.completed_objectives = [objective]
        orchestrator.game_state.turn_count = 35
        orchestrator._track_score_for_progress_detection()

        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 0, "Stuck timer should reset after objective completion"

        # Phase 4: Turns 36-56, no progress
        # Same issue: objective at turn 35 is "recent" at turn 36, causing another reset
        self.simulate_turns(orchestrator, 21, start_turn=36, score=5)

        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 20, "Should be stuck for 20 turns again"

        warning = orchestrator._build_stuck_countdown_warning()
        assert warning != "", "Warning should appear again at 20 turns stuck"

        # Phase 5: Turn 75, score increase
        # Adjusted: phase 4 now ends at turn 56, so we start at 57
        self.simulate_turns(orchestrator, 18, start_turn=57, score=5)
        orchestrator.game_state.turn_count = 75
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 0, "Stuck timer should reset after score increase"

        # Episode should continue past turn 75
        assert orchestrator.game_state.turn_count == 75
        assert turns_stuck < orchestrator.config.max_turns_stuck

    def test_objective_completion_after_score_increase(self, orchestrator):
        """Test that both events reset timer independently.

        Scenario:
        - Turn 5: Score increase
        - Turn 10: Objective completion
        - Expected: Both events reset timer independently
        """
        # Initialize
        orchestrator.game_state.previous_zork_score = 0
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()

        # Turn 5: Score increase
        self.simulate_turns(orchestrator, 4, start_turn=1, score=0)
        orchestrator.game_state.turn_count = 5
        orchestrator.game_state.previous_zork_score = 5
        orchestrator._track_score_for_progress_detection()

        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 0, "Stuck timer should reset after score increase"

        # Turn 10: Objective completion
        objective = self.create_completed_objective("find lamp", turn_num=10)
        orchestrator.game_state.completed_objectives = [objective]
        orchestrator.game_state.turn_count = 10
        orchestrator.game_state.previous_zork_score = 5  # Score unchanged
        orchestrator._track_score_for_progress_detection()

        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 0, "Stuck timer should reset after objective completion"

        # Verify both events are tracked independently
        assert orchestrator._last_score_change_turn == 10, \
            "Timer should be updated by objective completion"

    def test_phase_1a_1b_1c_integration(self, orchestrator):
        """Test all loop break phases work together without conflicts.

        Validates:
        - Phase 1A uses new progress metric (score OR objectives)
        - Phase 1B applies location penalties independently
        - Phase 1C warnings mention objectives when present
        - Phases don't interfere with each other
        """
        # Setup: Initialize state
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0
        orchestrator.game_state.current_room_id = 20
        orchestrator.game_state.current_room_name = "Test Room"
        orchestrator._track_score_for_progress_detection()

        # Setup objectives for Phase 1C
        orchestrator.game_state.discovered_objectives = [
            "explore the kitchen",
            "find the key"
        ]

        # Phase 1A: Simulate 25 turns stuck (with objective progress at turn 15)
        self.simulate_turns(orchestrator, 14, start_turn=1, score=10)

        # Add objective completion at turn 15
        objective = self.create_completed_objective("explore west", turn_num=15)
        orchestrator.game_state.completed_objectives = [objective]
        orchestrator.game_state.turn_count = 15
        orchestrator._track_score_for_progress_detection()

        # Verify Phase 1A reset works
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 0, "Phase 1A: Timer should reset from objective"

        # Continue to turn 36 (20 turns after the second reset at turn 16)
        # Objective at turn 15 is "recent" at turn 16, causing another reset
        self.simulate_turns(orchestrator, 21, start_turn=16, score=10)
        turns_stuck = orchestrator._get_turns_since_score_change()
        assert turns_stuck == 20, "Phase 1A: Should be stuck for 20 turns"

        # Phase 1B: Setup location revisit history
        # History: [15, 20, 18, 20, 19, 20]
        # Current is 20, window checks last 5 excluding current
        orchestrator._location_id_history = deque([15, 20, 18, 20, 19, 20], maxlen=20)

        # Verify Phase 1B detection works
        revisit_info = orchestrator._detect_location_revisit()
        assert revisit_info["detected"] is True, \
            "Phase 1B: Should detect location revisit"
        assert revisit_info["recent_visits"] == 2, \
            "Phase 1B: Should count 2 visits in window"

        # Verify Phase 1B penalty application
        base_score = 0.9
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score, revisit_info
        )
        expected_penalty = -0.2 * 2  # -0.4
        expected_score = 0.9 + expected_penalty  # 0.5
        assert adjusted_score == expected_score, \
            f"Phase 1B: Expected score {expected_score}, got {adjusted_score}"
        assert "Location revisit penalty" in reason, \
            "Phase 1B: Reason should mention penalty"

        # Phase 1C: Verify warnings work with objectives
        warning = orchestrator._build_stuck_countdown_warning()
        assert warning != "", "Phase 1C: Warning should be present"
        assert "score or complete an objective" in warning.lower(), \
            "Phase 1C: Warning should mention both progress paths"
        assert "explore the kitchen" in warning or "CURRENT OBJECTIVES:" in warning, \
            "Phase 1C: Warning should include objectives"

        # Verify phases don't interfere
        # - Location penalty doesn't affect progress tracking
        assert turns_stuck == 20, "Phases don't interfere: Penalty doesn't affect progress"

        # - Warning includes objectives but doesn't change stuck timer
        warning_turns = orchestrator._get_turns_since_score_change()
        assert warning_turns == 20, "Phases don't interfere: Warning doesn't affect timer"

        # - Objective progress resets timer without affecting location tracking
        objective2 = self.create_completed_objective("new objective", turn_num=36)
        orchestrator.game_state.completed_objectives = [objective, objective2]
        orchestrator.game_state.turn_count = 36
        orchestrator._track_score_for_progress_detection()

        turns_stuck_after_objective = orchestrator._get_turns_since_score_change()
        assert turns_stuck_after_objective == 0, \
            "Phases don't interfere: Objective resets timer"

        # Location history should still be intact
        assert len(orchestrator._location_id_history) > 0, \
            "Phases don't interfere: Location history preserved"
