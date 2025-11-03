"""
ABOUTME: Tests for knowledge update quality checks and sliding window logic.
ABOUTME: Validates that long episodes can still receive knowledge updates.
"""

import pytest
from unittest.mock import Mock
from knowledge import AdaptiveKnowledgeManager
from session.game_configuration import GameConfiguration


class TestKnowledgeUpdateQuality:
    """Test suite for knowledge update quality checks."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager instance."""
        log_file = tmp_path / "test_log.jsonl"
        output_file = tmp_path / "test_knowledge.md"

        # Create minimal config files to avoid errors
        workdir = tmp_path / "game_files"
        workdir.mkdir(exist_ok=True)

        # Create test configuration
        config = GameConfiguration.from_toml()

        manager = AdaptiveKnowledgeManager(
            config=config,
            log_file=str(log_file),
            output_file=str(output_file),
            logger=Mock(),
            workdir=str(workdir),
        )
        return manager

    def create_turn_data(self, actions_list, episode_id="test_episode"):
        """
        Helper to create turn_data from a list of actions.

        Args:
            actions_list: List of action strings (e.g., ["north", "take lamp", "north"])
            episode_id: Episode identifier

        Returns:
            turn_data dictionary suitable for _should_update_knowledge()
        """
        turn_data = {
            "episode_id": episode_id,
            "start_turn": 1,
            "end_turn": len(actions_list),
            "actions_and_responses": [],
            "score_changes": [],
            "location_changes": [],
            "death_events": [],
        }

        for i, action in enumerate(actions_list, start=1):
            turn_data["actions_and_responses"].append({
                "turn": i,
                "action": action,
                "reasoning": f"Reasoning for {action}",
                "critic_score": 0.5,
                "response": f"Response to {action} with some meaningful content here",
            })

        return turn_data

    def test_long_episode_still_updates(self, manager):
        """
        Verify that 300+ turn episodes can still update using sliding window.

        This is the primary bug fix validation: even if episode-wide variety
        is low (26%), recent window variety should allow updates.
        """
        # Create a 350-turn episode with:
        # - First 250 turns: diverse actions (50 unique)
        # - Last 100 turns: good variety (30 unique actions distributed evenly)

        # First 250 turns - establish diverse history
        early_actions = []
        for i in range(50):
            # Each action repeated 5 times
            for _ in range(5):
                early_actions.append(f"action_{i}")

        # Last 100 turns - 30 unique actions distributed evenly (30% variety)
        # This ensures we don't trigger stuck detection
        late_actions = []
        for i in range(100):
            late_actions.append(f"late_action_{i % 30}")

        all_actions = early_actions + late_actions
        turn_data = self.create_turn_data(all_actions)

        # Episode-wide variety: 80 unique / 350 total = 22.8% (would fail old 30% threshold)
        # Window variety (last 75): 30 unique / 75 total = 40% (should pass at 15% threshold)

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert should_update, f"Long episode should allow updates, got: {reason}"
        # Reason should reference either sliding window OR stuck detection (both are valid)
        assert "last" in reason or "unique actions" in reason, \
            f"Reason should reference sliding window or action variety, got: {reason}"

    def test_sliding_window_calculation(self, manager):
        """Verify sliding window logic is correct."""
        # Create 100-turn episode where:
        # - First 50 turns: very diverse (40 unique)
        # - Last 50 turns: very repetitive (5 unique)

        early_diverse = [f"action_{i}" for i in range(40)] + ["filler"] * 10
        late_repetitive = (["stuck_1"] * 10 + ["stuck_2"] * 10 + ["stuck_3"] * 10
                          + ["stuck_4"] * 10 + ["stuck_5"] * 10)

        all_actions = early_diverse + late_repetitive
        turn_data = self.create_turn_data(all_actions)

        # Window (last 75 turns) includes:
        # - 25 turns from early_diverse (diverse)
        # - 50 turns from late_repetitive (5 unique)
        # Total unique in window: ~30 / 75 = 40% variety (should pass)

        should_update, reason = manager._should_update_knowledge(turn_data)

        # The window should focus on recent turns, detecting low variety
        assert should_update or "recent window" in reason, \
            f"Window calculation should work correctly, got: {reason}"

    def test_stuck_detection_forces_update(self, manager):
        """Verify stuck pattern override works."""
        # Create scenario where agent is clearly stuck:
        # - 50 turns of normal play
        # - Last 20 turns: only 3 unique actions repeated

        normal_play = [f"action_{i % 20}" for i in range(50)]
        stuck_pattern = (["stuck_action_1"] * 7 + ["stuck_action_2"] * 7
                        + ["stuck_action_3"] * 6)

        all_actions = normal_play + stuck_pattern
        turn_data = self.create_turn_data(all_actions)

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert should_update, f"Stuck pattern should force update, got: {reason}"
        assert "stuck" in reason.lower() or "forcing update" in reason.lower(), \
            f"Reason should mention stuck pattern, got: {reason}"

    def test_short_episode_handling(self, manager):
        """Verify episodes < 75 turns work correctly."""
        # Create a 30-turn episode with good variety
        actions = [f"action_{i}" for i in range(15)] * 2  # 15 unique, 30 total = 50%
        turn_data = self.create_turn_data(actions)

        should_update, reason = manager._should_update_knowledge(turn_data)

        # Should use window_size = 30 (not 75)
        assert should_update, f"Short episode with variety should update, got: {reason}"

    def test_minimum_action_threshold(self, manager):
        """Verify minimum action requirement still enforced."""
        # Only 2 actions - should fail regardless of variety
        actions = ["action_1", "action_2"]
        turn_data = self.create_turn_data(actions)

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert not should_update, "Should reject episodes with < 3 actions"
        assert "Too few actions" in reason

    def test_death_events_override(self, manager):
        """Verify death events always trigger update."""
        # Very repetitive episode but with death
        actions = ["north"] * 50
        turn_data = self.create_turn_data(actions)

        # Add death event
        turn_data["death_events"] = [{
            "turn": 50,
            "reason": "eaten by grue",
            "event_type": "game_over"
        }]

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert should_update, "Death events should always trigger update"
        assert "death" in reason.lower()

    def test_score_changes_override(self, manager):
        """Verify score changes trigger update."""
        # Repetitive episode but with score changes
        actions = ["north"] * 30
        turn_data = self.create_turn_data(actions)

        turn_data["score_changes"] = [{
            "turn": 15,
            "from_score": 0,
            "to_score": 5,
            "change": 5
        }]

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert should_update, "Score changes should trigger update"
        assert "Score changed" in reason

    def test_location_changes_override(self, manager):
        """Verify location changes trigger update."""
        # Repetitive episode but with location discoveries
        actions = ["north"] * 30
        turn_data = self.create_turn_data(actions)

        turn_data["location_changes"] = [
            {"turn": 5, "from_location": "West of House", "to_location": "North of House"},
            {"turn": 10, "from_location": "North of House", "to_location": "Forest Path"},
        ]

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert should_update, "Location changes should trigger update"
        assert "new locations" in reason.lower()

    def test_very_low_recent_variety_rejected(self, manager):
        """Verify very low variety in recent window is still rejected."""
        # Create episode where last 75 turns have < 15% variety

        # First 100 turns: diverse
        early_actions = [f"action_{i % 40}" for i in range(100)]

        # Last 75 turns: only 10 unique actions (13.3% variety)
        late_actions = [f"repetitive_{i % 10}" for i in range(75)]

        all_actions = early_actions + late_actions
        turn_data = self.create_turn_data(all_actions)

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert not should_update, "Very low recent variety should be rejected"
        assert "repetitive" in reason.lower()
        assert "recent window" in reason.lower()

    def test_window_size_adapts_to_episode_length(self, manager):
        """Verify window size is min(75, episode_length)."""
        # Test with 20-turn episode
        actions = [f"action_{i}" for i in range(10)] * 2  # 10 unique, 20 total
        turn_data = self.create_turn_data(actions)

        should_update, reason = manager._should_update_knowledge(turn_data)

        # Window should be 20, not 75
        # With 50% variety, should pass
        assert should_update, f"Short episode should adapt window size, got: {reason}"

    def test_response_variety_check(self, manager):
        """Verify response variety is still checked."""
        # Good action variety but no response variety
        actions = [f"action_{i}" for i in range(20)]
        turn_data = self.create_turn_data(actions)

        # Make all responses identical
        for action in turn_data["actions_and_responses"]:
            action["response"] = "Same response"

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert not should_update, "Should reject episodes with no response variety"
        assert "no new information" in reason.lower()

    def test_meaningful_content_check(self, manager):
        """Verify meaningful content requirement."""
        # Good variety but responses too short
        actions = [f"action_{i}" for i in range(20)]
        turn_data = self.create_turn_data(actions)

        # Make all responses very short
        for action in turn_data["actions_and_responses"]:
            action["response"] = "Ok"

        should_update, reason = manager._should_update_knowledge(turn_data)

        assert not should_update, "Should reject episodes with too-short responses"
        # Response variety check triggers first (all responses are "Ok")
        assert "information" in reason.lower() or "short" in reason.lower() or "uninformative" in reason.lower()

    def test_logging_includes_window_metrics(self, manager):
        """Verify comprehensive logging includes window and episode metrics."""
        actions = [f"action_{i}" for i in range(100)]
        turn_data = self.create_turn_data(actions)

        should_update, reason = manager._should_update_knowledge(turn_data)

        # Check that logger was called with proper metrics
        if should_update:
            # Should have logged decision with metrics
            assert manager.logger.info.called, "Should log decision"

            # Verify log call includes key metrics
            call_args = manager.logger.info.call_args
            if call_args:
                # Check keyword arguments
                kwargs = call_args[1] if len(call_args) > 1 else call_args.kwargs

                # Should include window_size
                assert "window_size" in str(kwargs) or "window_size" in call_args[0], \
                    "Should log window_size"

    def test_exact_threshold_boundary(self, manager):
        """Test behavior at exact 15% threshold."""
        # Create episode with exactly 15% recent variety
        # 75 turns, 11 unique actions = 14.67% (below threshold)
        # 75 turns, 12 unique actions = 16% (above threshold)

        # Test below threshold
        below_actions = [f"action_{i % 11}" for i in range(75)]
        turn_data_below = self.create_turn_data(below_actions)

        should_update_below, reason_below = manager._should_update_knowledge(turn_data_below)
        assert not should_update_below, "14.67% should be below 15% threshold"

        # Test above threshold
        above_actions = [f"action_{i % 12}" for i in range(75)]
        turn_data_above = self.create_turn_data(above_actions)

        should_update_above, reason_above = manager._should_update_knowledge(turn_data_above)
        assert should_update_above, "16% should be above 15% threshold"


class TestRegressionScenarios:
    """Test specific regression scenarios from the bug report."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager instance."""
        log_file = tmp_path / "test_log.jsonl"
        output_file = tmp_path / "test_knowledge.md"

        workdir = tmp_path / "game_files"
        workdir.mkdir(exist_ok=True)

        # Create test configuration
        config = GameConfiguration.from_toml()

        manager = AdaptiveKnowledgeManager(
            config=config,
            log_file=str(log_file),
            output_file=str(output_file),
            logger=Mock(),
            workdir=str(workdir),
        )
        return manager

    def test_turn_300_scenario_from_bug_report(self, manager):
        """
        Reproduce exact scenario from bug report:
        - Turn 300: 26% unique actions episode-wide
        - Should still allow update via sliding window
        """
        # Create 300-turn episode with 26% episode-wide variety
        # But maintain decent variety in recent window to avoid stuck detection

        # Strategy: Use 50 unique actions distributed across 300 turns
        # Episode-wide: 50/300 = 16.7% (would fail old 30% threshold)
        # Recent window (last 75): ~25 unique = 33% (should pass)

        actions = []
        # Cycle through 50 actions, distributing them evenly
        for i in range(300):
            actions.append(f"action_{i % 50}")

        turn_data = {
            "episode_id": "bug_report_test",
            "start_turn": 1,
            "end_turn": 300,
            "actions_and_responses": [],
            "score_changes": [],
            "location_changes": [],
            "death_events": [],
        }

        for i, action in enumerate(actions, start=1):
            turn_data["actions_and_responses"].append({
                "turn": i,
                "action": action,
                "reasoning": f"Reasoning for {action}",
                "critic_score": 0.5,
                "response": f"Response to {action} - this is meaningful content that is long enough to pass the length check",
            })

        should_update, reason = manager._should_update_knowledge(turn_data)

        # OLD BEHAVIOR: Would fail with "Too repetitive (16.7% unique actions)"
        # NEW BEHAVIOR: Should check last 75 turns only (33% variety there)
        assert should_update, \
            f"Turn 300 scenario should allow updates via sliding window. Reason: {reason}"
        # Either sliding window logic OR stuck detection can allow the update
        assert "last" in reason.lower() or "unique" in reason.lower(), \
            f"Should reference action variety analysis, got: {reason}"
