"""Tests for Phase 3: Objective-aware stuck countdown warnings.

This test suite validates that stuck countdown warnings adapt based on
objective presence/absence and include appropriate guidance.
"""

import pytest
from unittest.mock import patch
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
from session.game_configuration import GameConfiguration


class TestObjectiveAwareWarnings:
    """Test objective-aware stuck countdown warnings."""

    @pytest.fixture
    def config(self):
        """Create test configuration with warnings enabled."""
        return GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            max_turns_stuck=40,
            stuck_check_interval=10,
            stuck_warning_threshold=20,
            enable_stuck_warnings=True,
            enable_exploration_hints=True,
            zork_game_workdir="/tmp/test_game_files",
        )

    @pytest.fixture
    def orchestrator(self, config):
        """Create test orchestrator."""
        with patch('orchestration.zork_orchestrator_v2.JerichoInterface'):
            orch = ZorkOrchestratorV2(episode_id="test_warnings")
            orch.config = config
            return orch

    def test_warning_without_objectives_omits_objective_section(self, orchestrator):
        """Warning without objectives should not include CURRENT OBJECTIVES section."""
        # Setup: Agent stuck for 25 turns, no objectives
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.discovered_objectives = []

        # Track score to establish baseline
        orchestrator._track_score_for_progress_detection()

        # Generate warning
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify warning exists
        assert warning != "", "Warning should be generated when stuck"

        # Verify no objective section
        assert "CURRENT OBJECTIVES:" not in warning, "Should not include objectives section"
        assert "NO SCORE PROGRESS" in warning, "Should mention score progress"
        assert "If you do not increase your score, you will DIE in 15 turns" in warning
        assert "SURVIVAL DEPENDS ON SCORE INCREASE" in warning

        # Verify no objective-specific suggestions
        assert "Try working on the objectives" not in warning
        assert "complete objectives" not in warning

    def test_warning_with_objectives_includes_objective_section(self, orchestrator):
        """Warning with objectives should include CURRENT OBJECTIVES section."""
        # Setup: Agent stuck for 25 turns, with objectives
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.discovered_objectives = [
            "explore north to Location 81",
            "open the window",
            "find the treasure"
        ]

        # Track score to establish baseline
        orchestrator._track_score_for_progress_detection()

        # Generate warning
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify warning exists
        assert warning != "", "Warning should be generated when stuck"

        # Verify objective section exists
        assert "CURRENT OBJECTIVES:" in warning, "Should include objectives section"
        assert "• explore north to Location 81" in warning
        assert "• open the window" in warning
        assert "• find the treasure" in warning

        # Verify adapted progress message
        assert "NO PROGRESS" in warning, "Should mention general progress"
        assert "If you do not increase your score or complete an objective" in warning
        assert "SURVIVAL DEPENDS ON MAKING PROGRESS" in warning

        # Verify objective-specific suggestions
        assert "Try working on the objectives listed above" in warning
        assert "complete objectives" in warning

    def test_warning_limits_objectives_to_five(self, orchestrator):
        """Warning should limit displayed objectives to 5 maximum."""
        # Setup: Agent stuck with 7 objectives
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.discovered_objectives = [
            "objective 1",
            "objective 2",
            "objective 3",
            "objective 4",
            "objective 5",
            "objective 6",
            "objective 7"
        ]

        # Track score to establish baseline
        orchestrator._track_score_for_progress_detection()

        # Generate warning
        warning = orchestrator._build_stuck_countdown_warning()

        # Count objective bullets
        objective_count = warning.count("• objective")
        assert objective_count == 5, f"Should show exactly 5 objectives, got {objective_count}"

        # Verify first 5 are included
        assert "• objective 1" in warning
        assert "• objective 2" in warning
        assert "• objective 3" in warning
        assert "• objective 4" in warning
        assert "• objective 5" in warning

        # Verify 6 and 7 are not included
        assert "• objective 6" not in warning
        assert "• objective 7" not in warning

    def test_warning_urgency_levels_preserved(self, orchestrator):
        """Warning should maintain urgency levels regardless of objectives."""
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.discovered_objectives = ["test objective"]

        # Track score to establish baseline
        orchestrator._track_score_for_progress_detection()

        # Test critical emergency (5 turns left)
        orchestrator.game_state.turn_count = 35  # 40 - 35 = 5 left
        warning = orchestrator._build_stuck_countdown_warning()
        assert "🚨 CRITICAL EMERGENCY" in warning
        assert "DIE in 5 turns" in warning

        # Test urgent warning (10 turns left)
        orchestrator.game_state.turn_count = 30  # 40 - 30 = 10 left
        warning = orchestrator._build_stuck_countdown_warning()
        assert "⚠️ URGENT WARNING" in warning
        assert "DIE in 10 turns" in warning

        # Test normal warning (20 turns left)
        orchestrator.game_state.turn_count = 20  # 40 - 20 = 20 left
        warning = orchestrator._build_stuck_countdown_warning()
        assert "⚠️ SCORE STAGNATION DETECTED" in warning
        assert "DIE in 20 turns" in warning

    def test_warnings_disabled_when_config_flag_false(self, orchestrator):
        """Warnings should not appear when disabled in config."""
        # Disable warnings
        orchestrator.config.enable_stuck_warnings = False

        # Setup stuck condition
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.discovered_objectives = ["test objective"]

        # Track score
        orchestrator._track_score_for_progress_detection()

        # Generate warning
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify no warning
        assert warning == "", "Warning should be empty when disabled"

    def test_warnings_not_shown_until_threshold(self, orchestrator):
        """Warnings should not appear until threshold is reached."""
        # Setup: Agent stuck but below threshold (threshold is 20)
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 15  # Below threshold
        orchestrator.game_state.discovered_objectives = ["test objective"]

        # Track score
        orchestrator._track_score_for_progress_detection()

        # Generate warning
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify no warning
        assert warning == "", "Warning should not appear below threshold"

        # Now reach threshold
        orchestrator.game_state.turn_count = 20  # At threshold
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify warning appears
        assert warning != "", "Warning should appear at threshold"
        assert "DIE in 20 turns" in warning

    def test_warning_with_empty_objective_list_behaves_like_no_objectives(self, orchestrator):
        """Empty objective list should behave the same as no objectives."""
        # Setup: Agent stuck with empty objectives list
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.discovered_objectives = []  # Empty list

        # Track score
        orchestrator._track_score_for_progress_detection()

        # Generate warning
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify no objective section
        assert "CURRENT OBJECTIVES:" not in warning
        assert "NO SCORE PROGRESS" in warning
        assert "If you do not increase your score, you will DIE in 15 turns" in warning
        assert "SURVIVAL DEPENDS ON SCORE INCREASE" in warning

    def test_warning_general_suggestions_always_included(self, orchestrator):
        """General suggestions should appear regardless of objectives."""
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25

        # Track score
        orchestrator._track_score_for_progress_detection()

        # Test with objectives
        orchestrator.game_state.discovered_objectives = ["test objective"]
        warning_with_obj = orchestrator._build_stuck_countdown_warning()

        # Test without objectives
        orchestrator.game_state.discovered_objectives = []
        warning_without_obj = orchestrator._build_stuck_countdown_warning()

        # Verify general suggestions in both
        general_suggestions = [
            "Try a completely different location (move 3+ rooms away)",
            "Attempt a different puzzle approach",
            "Explore unexplored exits",
            "Consider abandoning your current strategy"
        ]

        for suggestion in general_suggestions:
            assert suggestion in warning_with_obj, f"Missing suggestion with objectives: {suggestion}"
            assert suggestion in warning_without_obj, f"Missing suggestion without objectives: {suggestion}"

    def test_warning_objective_suggestions_only_with_objectives(self, orchestrator):
        """Objective-specific suggestions should only appear with objectives."""
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25

        # Track score
        orchestrator._track_score_for_progress_detection()

        # Test with objectives
        orchestrator.game_state.discovered_objectives = ["test objective"]
        warning_with_obj = orchestrator._build_stuck_countdown_warning()

        # Test without objectives
        orchestrator.game_state.discovered_objectives = []
        warning_without_obj = orchestrator._build_stuck_countdown_warning()

        # Verify objective suggestions only with objectives
        objective_suggestions = [
            "Try working on the objectives listed above",
            "Prioritize actions that might increase your score or complete objectives"
        ]

        for suggestion in objective_suggestions:
            assert suggestion in warning_with_obj, f"Missing objective suggestion: {suggestion}"
            assert suggestion not in warning_without_obj, f"Should not include objective suggestion: {suggestion}"

    def test_warning_formatting_consistent(self, orchestrator):
        """Warning should maintain consistent formatting structure."""
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.discovered_objectives = ["test objective"]

        # Track score
        orchestrator._track_score_for_progress_detection()

        # Generate warning
        warning = orchestrator._build_stuck_countdown_warning()

        # Verify structure elements
        assert warning.startswith("=" * 70), "Should start with separator line"
        assert warning.endswith("=" * 70), "Should end with separator line"
        assert "Your PRIMARY GOAL is to INCREASE YOUR SCORE." in warning
        assert "SUGGESTED STRATEGIES TO BREAK FREE:" in warning

        # Verify proper line breaks (not all on one line)
        lines = warning.split("\n")
        assert len(lines) > 10, "Warning should be multi-line"
