# ABOUTME: Tests for objective-based progress tracking in loop break system
# ABOUTME: Validates window-based checking and smart timer reset logic

"""
Tests for Phase 2: Core Progress Tracking Logic

This test suite validates the enhanced _track_score_for_progress_detection()
method that supports both score-based and objective-based progress detection.

Test Coverage:
- Window-based progress checking (both score and objective)
- Smart timer reset logic
- Edge cases (empty objectives, disabled feature, boundary conditions)
- Backward compatibility with score-only tracking
- Comprehensive logging validation
"""

import logging
from typing import Dict, Any
from unittest.mock import Mock, MagicMock, patch

import pytest

from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class TestObjectiveProgressTracking:
    """Test suite for objective-based progress tracking."""

    @pytest.fixture
    def mock_config(self) -> GameConfiguration:
        """Create a real configuration with objective progress enabled."""
        # Load real config and override specific settings
        with patch.dict('os.environ', {}, clear=False):
            config = GameConfiguration.from_toml()
            config.max_turns_stuck = 40
            config.enable_objective_based_progress = True
            config.stuck_check_interval = 5
            config.stuck_warning_threshold = 20
            return config

    @pytest.fixture
    def mock_game_state(self) -> GameState:
        """Create a mock game state."""
        state = Mock(spec=GameState)
        state.turn_count = 1
        state.previous_zork_score = 0
        state.completed_objectives = []
        return state

    @pytest.fixture
    def orchestrator(self, mock_config, mock_game_state) -> ZorkOrchestratorV2:
        """Create orchestrator with mocked dependencies."""
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config), \
             patch('orchestration.zork_orchestrator_v2.GameState', return_value=mock_game_state), \
             patch('orchestration.zork_orchestrator_v2.JerichoInterface'), \
             patch('orchestration.zork_orchestrator_v2.MapManager'), \
             patch('orchestration.zork_orchestrator_v2.SimpleMemoryManager'), \
             patch('orchestration.zork_orchestrator_v2.KnowledgeManager'), \
             patch('orchestration.zork_orchestrator_v2.ObjectiveManager'), \
             patch('orchestration.zork_orchestrator_v2.ContextManager'), \
             patch('orchestration.zork_orchestrator_v2.StateManager'):

            orch = ZorkOrchestratorV2(episode_id="test-progress-tracking")
            # Override with our mock instances
            orch.config = mock_config
            orch.game_state = mock_game_state
            orch.logger = Mock(spec=logging.Logger)
            return orch

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_initialization_on_first_call(self, orchestrator):
        """Test that tracking variables are initialized on first call."""
        # First call should initialize tracking
        orchestrator._track_score_for_progress_detection()

        assert hasattr(orchestrator, '_last_score_change_turn')
        assert hasattr(orchestrator, '_last_tracked_score')
        assert orchestrator._last_score_change_turn == 0
        assert orchestrator._last_tracked_score == 0

    def test_initialization_respects_starting_score(self, orchestrator):
        """Test that initialization works with non-zero starting score."""
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        assert orchestrator._last_tracked_score == 10

    # ========================================================================
    # Score-Based Progress Tests
    # ========================================================================

    def test_score_change_resets_timer(self, orchestrator):
        """Test that score change resets the stuck timer."""
        # Initialize
        orchestrator._track_score_for_progress_detection()

        # Advance turns without score change
        orchestrator.game_state.turn_count = 10
        orchestrator._track_score_for_progress_detection()

        # Check stuck turns
        assert orchestrator._get_turns_since_score_change() == 10

        # Change score
        orchestrator.game_state.turn_count = 15
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        # Timer should be reset
        assert orchestrator._get_turns_since_score_change() == 0

    def test_score_decrease_resets_timer(self, orchestrator):
        """Test that score decrease (death/penalty) also resets timer."""
        orchestrator._track_score_for_progress_detection()
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 5
        orchestrator._track_score_for_progress_detection()

        # Score decreases (death)
        orchestrator.game_state.turn_count = 10
        orchestrator.game_state.previous_zork_score = 0
        orchestrator._track_score_for_progress_detection()

        assert orchestrator._get_turns_since_score_change() == 0

    def test_score_progress_within_window(self, orchestrator):
        """Test window-based score progress detection."""
        # Initialize
        orchestrator._track_score_for_progress_detection()

        # Score change at turn 5
        orchestrator.game_state.turn_count = 5
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        # Check at turn 44 (39 turns after change, within 40-turn window)
        orchestrator.game_state.turn_count = 44
        orchestrator._track_score_for_progress_detection()

        # Verify debug logging shows score_progress=True
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        assert len(debug_calls) > 0
        last_debug = debug_calls[-1]
        assert 'score_progress=True' in str(last_debug)

    def test_score_progress_outside_window(self, orchestrator):
        """Test that score progress expires outside window."""
        # Initialize
        orchestrator._track_score_for_progress_detection()

        # Score change at turn 5
        orchestrator.game_state.turn_count = 5
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        # Check at turn 46 (41 turns after change, outside 40-turn window)
        orchestrator.game_state.turn_count = 46
        orchestrator._track_score_for_progress_detection()

        # Verify debug logging shows score_progress=False
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        assert len(debug_calls) > 0
        last_debug = debug_calls[-1]
        assert 'score_progress=False' in str(last_debug)

    # ========================================================================
    # Objective-Based Progress Tests
    # ========================================================================

    def test_objective_completion_prevents_termination(self, orchestrator):
        """Test that objective completion resets timer when no recent score change."""
        # Initialize
        orchestrator._track_score_for_progress_detection()

        # Advance without score change (stuck for 30 turns)
        orchestrator.game_state.turn_count = 30
        orchestrator._track_score_for_progress_detection()

        assert orchestrator._get_turns_since_score_change() == 30

        # Complete objective at turn 35 (no score change)
        orchestrator.game_state.turn_count = 35
        orchestrator.game_state.completed_objectives = [{
            "objective": "Open mailbox",
            "completed_turn": 35,
            "completion_action": "open mailbox",
            "completion_response": "You open the mailbox...",
            "completion_location": "West of House",
            "completion_score": 0
        }]
        orchestrator._track_score_for_progress_detection()

        # Timer should be reset by objective progress
        assert orchestrator._get_turns_since_score_change() == 0

    def test_objective_progress_within_window(self, orchestrator):
        """Test window-based objective progress detection."""
        # Initialize and complete objective at turn 5
        orchestrator._track_score_for_progress_detection()
        orchestrator.game_state.turn_count = 5
        orchestrator.game_state.completed_objectives = [{
            "objective": "Get lamp",
            "completed_turn": 5,
            "completion_action": "take lamp",
            "completion_response": "Taken.",
            "completion_location": "Living Room",
            "completion_score": 0
        }]
        orchestrator._track_score_for_progress_detection()

        # Check at turn 44 (39 turns after completion, within window)
        orchestrator.game_state.turn_count = 44
        orchestrator._track_score_for_progress_detection()

        # Verify debug logging shows objective_progress=True
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        assert len(debug_calls) > 0
        last_debug = debug_calls[-1]
        assert 'objective_progress=True' in str(last_debug)

    def test_objective_progress_outside_window(self, orchestrator):
        """Test that objective progress expires outside window."""
        # Initialize and complete objective at turn 5
        orchestrator._track_score_for_progress_detection()
        orchestrator.game_state.turn_count = 5
        orchestrator.game_state.completed_objectives = [{
            "objective": "Get lamp",
            "completed_turn": 5,
            "completion_action": "take lamp",
            "completion_response": "Taken.",
            "completion_location": "Living Room",
            "completion_score": 0
        }]
        orchestrator._track_score_for_progress_detection()

        # Check at turn 46 (41 turns after completion, outside window)
        orchestrator.game_state.turn_count = 46
        orchestrator._track_score_for_progress_detection()

        # Verify debug logging shows objective_progress=False
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        assert len(debug_calls) > 0
        last_debug = debug_calls[-1]
        assert 'objective_progress=False' in str(last_debug)

    def test_multiple_objectives_uses_most_recent(self, orchestrator):
        """Test that multiple objectives use the most recent completion."""
        orchestrator._track_score_for_progress_detection()

        # Complete multiple objectives
        orchestrator.game_state.completed_objectives = [
            {"objective": "First", "completed_turn": 5, "completion_action": "action1",
             "completion_response": "resp1", "completion_location": "loc1", "completion_score": 0},
            {"objective": "Second", "completed_turn": 10, "completion_action": "action2",
             "completion_response": "resp2", "completion_location": "loc2", "completion_score": 0},
            {"objective": "Third", "completed_turn": 15, "completion_action": "action3",
             "completion_response": "resp3", "completion_location": "loc3", "completion_score": 0},
        ]

        # Check at turn 54 (39 turns after most recent, within window)
        orchestrator.game_state.turn_count = 54
        orchestrator._track_score_for_progress_detection()

        # Should use turn 15 (most recent) for window calculation
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        assert len(debug_calls) > 0
        last_debug = debug_calls[-1]
        assert 'objective_progress=True' in str(last_debug)

    # ========================================================================
    # Combined Progress Tests
    # ========================================================================

    def test_either_score_or_objective_progress_sufficient(self, orchestrator):
        """Test that either score or objective progress prevents termination."""
        orchestrator._track_score_for_progress_detection()

        # Score change at turn 5, no objectives
        orchestrator.game_state.turn_count = 5
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        # At turn 44: score progress=True, objective progress=False, overall=True
        orchestrator.game_state.turn_count = 44
        orchestrator._track_score_for_progress_detection()

        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        last_debug = debug_calls[-1]
        assert 'progress_made=True' in str(last_debug)

        # Now test with only objective progress
        orchestrator.game_state.turn_count = 50  # Score progress expired
        orchestrator.game_state.completed_objectives = [{
            "objective": "Get lamp", "completed_turn": 49,
            "completion_action": "take lamp", "completion_response": "Taken.",
            "completion_location": "Living Room", "completion_score": 10
        }]
        orchestrator._track_score_for_progress_detection()

        # At turn 50: score progress=False, objective progress=True, overall=True
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        last_debug = debug_calls[-1]
        assert 'progress_made=True' in str(last_debug)

    def test_no_progress_when_both_expired(self, orchestrator):
        """Test no progress when both score and objective windows expired."""
        orchestrator._track_score_for_progress_detection()

        # Score change at turn 5
        orchestrator.game_state.turn_count = 5
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        # Objective completion at turn 10
        orchestrator.game_state.completed_objectives = [{
            "objective": "Get lamp", "completed_turn": 10,
            "completion_action": "take lamp", "completion_response": "Taken.",
            "completion_location": "Living Room", "completion_score": 10
        }]

        # Check at turn 100 (both windows expired)
        orchestrator.game_state.turn_count = 100
        orchestrator._track_score_for_progress_detection()

        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        last_debug = debug_calls[-1]
        assert 'score_progress=False' in str(last_debug)
        assert 'objective_progress=False' in str(last_debug)
        assert 'progress_made=False' in str(last_debug)

    def test_score_change_takes_precedence_over_objective(self, orchestrator):
        """Test that score change takes precedence when both occur simultaneously."""
        orchestrator._track_score_for_progress_detection()

        # Get stuck for 10 turns first
        orchestrator.game_state.turn_count = 10
        orchestrator._track_score_for_progress_detection()

        # At turn 15: Both score change AND objective completion happen
        # Score change should take precedence (first branch)
        orchestrator.game_state.turn_count = 15
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.completed_objectives = [{
            "objective": "Get lamp", "completed_turn": 15,
            "completion_action": "take lamp", "completion_response": "Taken.",
            "completion_location": "Living Room", "completion_score": 10
        }]
        orchestrator._track_score_for_progress_detection()

        # Verify only score_change event logged at turn 15, not objective_progress_reset
        # (score change takes precedence via if/elif structure)
        info_calls = orchestrator.logger.info.call_args_list

        # Check for score change events (may use different format)
        score_change_events = [c for c in info_calls if 'Score changed' in str(c)]
        objective_reset_events = [c for c in info_calls if 'Objective progress detected' in str(c)]

        assert len(score_change_events) >= 1, f"Should log score change. Got: {info_calls}"
        assert len(objective_reset_events) == 0, "Should not log objective reset when score changes"

    # ========================================================================
    # Edge Cases
    # ========================================================================

    def test_empty_completed_objectives_list(self, orchestrator):
        """Test that empty objectives list is handled gracefully."""
        orchestrator._track_score_for_progress_detection()
        orchestrator.game_state.turn_count = 10
        orchestrator.game_state.completed_objectives = []

        # Should not raise exception
        orchestrator._track_score_for_progress_detection()

        # objective_progress should be False
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        last_debug = debug_calls[-1]
        assert 'objective_progress=False' in str(last_debug)

    def test_feature_disabled_ignores_objectives(self, orchestrator):
        """Test that objectives are ignored when feature disabled."""
        orchestrator.config.enable_objective_based_progress = False
        orchestrator._track_score_for_progress_detection()

        # Complete objective
        orchestrator.game_state.turn_count = 10
        orchestrator.game_state.completed_objectives = [{
            "objective": "Get lamp", "completed_turn": 10,
            "completion_action": "take lamp", "completion_response": "Taken.",
            "completion_location": "Living Room", "completion_score": 0
        }]
        orchestrator._track_score_for_progress_detection()

        # objective_progress should be False despite completion
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        last_debug = debug_calls[-1]
        assert 'objective_progress=False' in str(last_debug)

    def test_objective_at_turn_1_checked_at_turn_42(self, orchestrator):
        """Test edge case: objective at turn 1, checked at turn 42."""
        orchestrator._track_score_for_progress_detection()

        # Complete objective at turn 1
        orchestrator.game_state.turn_count = 1
        orchestrator.game_state.completed_objectives = [{
            "objective": "Open mailbox", "completed_turn": 1,
            "completion_action": "open mailbox", "completion_response": "You open...",
            "completion_location": "West of House", "completion_score": 0
        }]
        orchestrator._track_score_for_progress_detection()

        # Check at turn 42 (41 turns after, outside 40-turn window)
        orchestrator.game_state.turn_count = 42
        orchestrator._track_score_for_progress_detection()

        # objective_progress should be False (outside window)
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        last_debug = debug_calls[-1]
        assert 'objective_progress=False' in str(last_debug)

    # ========================================================================
    # Logging Tests
    # ========================================================================

    def test_debug_logging_every_turn(self, orchestrator):
        """Test that debug logging occurs every turn."""
        orchestrator._track_score_for_progress_detection()

        # Run for 5 turns
        for turn in range(1, 6):
            orchestrator.game_state.turn_count = turn
            orchestrator._track_score_for_progress_detection()

        # Should have 5 debug logs (excluding initialization)
        debug_calls = [call for call in orchestrator.logger.debug.call_args_list
                       if 'Progress check' in str(call)]
        assert len(debug_calls) == 5

    def test_info_logging_on_score_change_when_stuck(self, orchestrator):
        """Test info logging when score changes after being stuck."""
        orchestrator._track_score_for_progress_detection()

        # Get stuck for 10 turns
        orchestrator.game_state.turn_count = 10
        orchestrator._track_score_for_progress_detection()

        # Change score
        orchestrator.game_state.turn_count = 15
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        # Should have info log about score change
        info_calls = orchestrator.logger.info.call_args_list
        score_change_logs = [c for c in info_calls if 'Score changed' in str(c)]
        assert len(score_change_logs) > 0
        assert 'resetting stuck timer' in str(score_change_logs[-1])

    def test_info_logging_on_objective_progress_reset(self, orchestrator):
        """Test info logging when timer reset by objective progress."""
        orchestrator._track_score_for_progress_detection()

        # Get stuck for 30 turns (no score change)
        orchestrator.game_state.turn_count = 30
        orchestrator._track_score_for_progress_detection()

        # Complete objective (triggers reset)
        orchestrator.game_state.turn_count = 35
        orchestrator.game_state.completed_objectives = [{
            "objective": "Open mailbox", "completed_turn": 35,
            "completion_action": "open mailbox", "completion_response": "You open...",
            "completion_location": "West of House", "completion_score": 0
        }]
        orchestrator._track_score_for_progress_detection()

        # Should have info log about objective progress reset
        info_calls = orchestrator.logger.info.call_args_list
        objective_reset_logs = [c for c in info_calls if 'Objective progress detected' in str(c)]
        assert len(objective_reset_logs) > 0
        assert 'resetting stuck timer' in str(objective_reset_logs[-1])

    def test_no_info_logging_during_initialization(self, orchestrator):
        """Test that initialization doesn't log info messages."""
        # First call initializes - should return early without logging
        orchestrator._track_score_for_progress_detection()

        # No info logs during initialization
        info_calls = orchestrator.logger.info.call_args_list
        assert len(info_calls) == 0, "Initialization should not log info messages"

    # ========================================================================
    # Backward Compatibility Tests
    # ========================================================================

    def test_get_turns_since_score_change_still_works(self, orchestrator):
        """Test that existing helper method still works correctly."""
        orchestrator._track_score_for_progress_detection()

        # Advance turns
        orchestrator.game_state.turn_count = 25
        orchestrator._track_score_for_progress_detection()

        assert orchestrator._get_turns_since_score_change() == 25

    def test_score_only_mode_unaffected(self, orchestrator):
        """Test that disabling objectives doesn't break score tracking."""
        orchestrator.config.enable_objective_based_progress = False
        orchestrator._track_score_for_progress_detection()

        # Score change should still work
        orchestrator.game_state.turn_count = 10
        orchestrator.game_state.previous_zork_score = 10
        orchestrator._track_score_for_progress_detection()

        assert orchestrator._get_turns_since_score_change() == 0
        assert orchestrator._last_tracked_score == 10


# ============================================================================
# Integration Tests
# ============================================================================

class TestProgressTrackingIntegration:
    """Integration tests for progress tracking with other systems."""

    @pytest.fixture
    def config(self) -> GameConfiguration:
        """Create real configuration for integration tests."""
        with patch.dict('os.environ', {}, clear=False):
            config = GameConfiguration.from_toml()
            config.max_turns_stuck = 40
            config.enable_objective_based_progress = True
            config.stuck_check_interval = 5
            config.stuck_warning_threshold = 20
            return config

    def test_realistic_game_scenario(self, config):
        """Test a realistic game scenario with mixed progress."""
        state = Mock(spec=GameState)
        state.turn_count = 1
        state.previous_zork_score = 0
        state.completed_objectives = []

        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=config), \
             patch('orchestration.zork_orchestrator_v2.GameState', return_value=state), \
             patch('orchestration.zork_orchestrator_v2.JerichoInterface'), \
             patch('orchestration.zork_orchestrator_v2.MapManager'), \
             patch('orchestration.zork_orchestrator_v2.SimpleMemoryManager'), \
             patch('orchestration.zork_orchestrator_v2.KnowledgeManager'), \
             patch('orchestration.zork_orchestrator_v2.ObjectiveManager'), \
             patch('orchestration.zork_orchestrator_v2.ContextManager'), \
             patch('orchestration.zork_orchestrator_v2.StateManager'):

            orch = ZorkOrchestratorV2(episode_id="test-realistic-scenario")
            orch.config = config
            orch.game_state = state
            orch.logger = Mock(spec=logging.Logger)

            # Initialize
            orch._track_score_for_progress_detection()

            # Early score gain (turn 5)
            state.turn_count = 5
            state.previous_zork_score = 10
            orch._track_score_for_progress_detection()
            assert orch._get_turns_since_score_change() == 0

            # Stuck for 20 turns
            state.turn_count = 25
            orch._track_score_for_progress_detection()
            assert orch._get_turns_since_score_change() == 20

            # Complete objective (no score) - should reset
            state.turn_count = 30
            state.completed_objectives = [{
                "objective": "Open mailbox", "completed_turn": 30,
                "completion_action": "open mailbox", "completion_response": "...",
                "completion_location": "West of House", "completion_score": 10
            }]
            orch._track_score_for_progress_detection()
            assert orch._get_turns_since_score_change() == 0

            # More stuck time
            state.turn_count = 50
            orch._track_score_for_progress_detection()
            assert orch._get_turns_since_score_change() == 20

            # Another objective
            state.turn_count = 55
            state.completed_objectives.append({
                "objective": "Get lamp", "completed_turn": 55,
                "completion_action": "take lamp", "completion_response": "Taken.",
                "completion_location": "Living Room", "completion_score": 10
            })
            orch._track_score_for_progress_detection()
            assert orch._get_turns_since_score_change() == 0

            # Final score gain
            state.turn_count = 70
            state.previous_zork_score = 20
            orch._track_score_for_progress_detection()
            assert orch._get_turns_since_score_change() == 0
