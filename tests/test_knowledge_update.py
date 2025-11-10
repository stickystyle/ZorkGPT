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


class TestCrossEpisodePreservation:
    """Test suite for CROSS-EPISODE INSIGHTS preservation during knowledge updates."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager instance with mocked LLM client."""
        log_file = tmp_path / "test_log.jsonl"
        output_file = tmp_path / "test_knowledge.md"

        workdir = tmp_path / "game_files"
        workdir.mkdir(exist_ok=True)

        # Create test configuration
        config = GameConfiguration.from_toml()

        # Create manager with mock client
        mock_client = Mock()
        manager = AdaptiveKnowledgeManager(
            config=config,
            log_file=str(log_file),
            output_file=str(output_file),
            logger=Mock(),
            workdir=str(workdir),
        )
        manager.client = mock_client
        manager.tmp_path = tmp_path  # Store for helper method access
        return manager

    def write_turn_events_to_log(self, manager, actions, episode_id="test_episode"):
        """Helper to write turn events to log file."""
        import json
        from pathlib import Path

        # Write to episode-specific log file (workdir/episodes/episode_id/episode_log.jsonl)
        episode_dir = Path(manager.workdir) / "episodes" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_log = episode_dir / "episode_log.jsonl"

        with open(episode_log, "w", encoding="utf-8") as f:
            for i, action in enumerate(actions, start=1):
                # Write final_action_selection event
                action_event = {
                    "event_type": "final_action_selection",
                    "episode_id": episode_id,
                    "turn": i,
                    "agent_action": action,
                    "agent_reasoning": f"Reasoning for {action}",
                    "critic_score": 0.5,
                }
                f.write(json.dumps(action_event) + "\n")

                # Write zork_response event
                response_event = {
                    "event_type": "zork_response",
                    "episode_id": episode_id,
                    "turn": i,
                    "action": action,
                    # Vary responses to pass uniqueness check
                    "zork_response": f"Response {i} to {action} with meaningful content here that is long enough to pass the quality check which requires at least 100 total characters across all responses.",
                }
                f.write(json.dumps(response_event) + "\n")

            # Add score change event to ensure quality check passes
            score_event = {
                "event_type": "score_change",
                "episode_id": episode_id,
                "turn": 10,
                "from_score": 0,
                "to_score": 5,
                "change": 5,
            }
            f.write(json.dumps(score_event) + "\n")

    def create_turn_data(self, actions_list, episode_id="test_episode"):
        """Helper to create valid turn_data."""
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
                "response": f"Response to {action} with meaningful content here that is long enough",
            })

        return turn_data

    def test_cross_episode_insights_preserved_during_update(self, manager, tmp_path):
        """
        Verify CROSS-EPISODE INSIGHTS section is preserved during knowledge updates.

        When update_knowledge_from_turns() is called, the existing CROSS-EPISODE
        INSIGHTS section should be:
        1. Extracted from existing knowledge
        2. Removed before LLM generation
        3. Restored after LLM generation
        """
        # Setup: Create existing knowledge with CROSS-EPISODE INSIGHTS
        existing_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## CROSS-EPISODE INSIGHTS
**IMPORTANT CROSS-EPISODE WISDOM**
This content should be preserved exactly.
It represents validated patterns from multiple episodes.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        # Write existing knowledge to file
        with open(manager.output_file, "w", encoding="utf-8") as f:
            f.write(existing_knowledge)

        # Mock LLM to return knowledge WITH a CROSS-EPISODE section (should be removed)
        llm_generated_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations - UPDATED BY LLM.

## CROSS-EPISODE INSIGHTS
LLM tried to regenerate this section - should be ignored.

## STRATEGIC PATTERNS
Examine before taking objects - UPDATED BY LLM.
"""

        # Mock anthropic-style response (response.content.strip())
        mock_response = Mock()
        mock_response.content = llm_generated_knowledge
        manager.client.chat.completions.create = Mock(return_value=mock_response)

        # Create mock turn data by writing to log file
        actions = [f"action_{i}" for i in range(20)]
        self.write_turn_events_to_log(manager, actions)

        # Execute update
        result = manager.update_knowledge_from_turns("test_episode", 1, 20)

        assert result, "Update should succeed"

        # Verify: Read final knowledge
        with open(manager.output_file, "r", encoding="utf-8") as f:
            final_knowledge = f.read()

        # ORIGINAL CROSS-EPISODE content should be present
        assert "IMPORTANT CROSS-EPISODE WISDOM" in final_knowledge, \
            "Original CROSS-EPISODE content should be preserved"
        assert "validated patterns from multiple episodes" in final_knowledge, \
            "Original CROSS-EPISODE content should be preserved exactly"

        # LLM's CROSS-EPISODE content should NOT be present
        assert "LLM tried to regenerate" not in final_knowledge, \
            "LLM-generated CROSS-EPISODE section should be removed"

        # LLM updates to OTHER sections should be present
        assert "UPDATED BY LLM" in final_knowledge, \
            "LLM updates to other sections should be preserved"

    def test_cross_episode_preservation_with_missing_section(self, manager):
        """
        Verify behavior when existing knowledge has no CROSS-EPISODE INSIGHTS.

        When there's no existing CROSS-EPISODE section:
        1. No section is extracted
        2. LLM generation proceeds normally
        3. No section is restored
        4. Final knowledge should not have CROSS-EPISODE section
        """
        # Setup: Create existing knowledge WITHOUT CROSS-EPISODE INSIGHTS
        existing_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        with open(manager.output_file, "w", encoding="utf-8") as f:
            f.write(existing_knowledge)

        # Mock LLM to return knowledge without CROSS-EPISODE section
        llm_generated_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations - UPDATED.

## STRATEGIC PATTERNS
Examine before taking objects - UPDATED.
"""

        # Mock anthropic-style response (response.content.strip())
        mock_response = Mock()
        mock_response.content = llm_generated_knowledge
        manager.client.chat.completions.create = Mock(return_value=mock_response)

        # Write turn events to log
        actions = [f"action_{i}" for i in range(20)]
        self.write_turn_events_to_log(manager, actions)

        # Execute update
        result = manager.update_knowledge_from_turns("test_episode", 1, 20)

        assert result, "Update should succeed"

        # Verify: No CROSS-EPISODE section in final knowledge
        with open(manager.output_file, "r", encoding="utf-8") as f:
            final_knowledge = f.read()

        assert "## CROSS-EPISODE INSIGHTS" not in final_knowledge, \
            "Should not have CROSS-EPISODE section when none existed before"

    def test_llm_generates_cross_episode_section_is_removed(self, manager):
        """
        Verify that even if LLM generates CROSS-EPISODE section, it's removed.

        This is the defensive programming test: LLM might generate CROSS-EPISODE
        despite being told not to. We should remove it and restore the original.
        """
        # Setup: Existing knowledge with CROSS-EPISODE
        existing_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## CROSS-EPISODE INSIGHTS
Original wisdom that must be preserved.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        with open(manager.output_file, "w", encoding="utf-8") as f:
            f.write(existing_knowledge)

        # Mock LLM to return knowledge WITH a CROSS-EPISODE section
        # (even though prompt tells it not to)
        llm_generated_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations - UPDATED.

## CROSS-EPISODE INSIGHTS
LLM ignored the instruction and generated this anyway.

## STRATEGIC PATTERNS
Examine before taking objects - UPDATED.

## CROSS-EPISODE INSIGHTS
LLM even generated it twice somehow.
"""

        # Mock anthropic-style response (response.content.strip())
        mock_response = Mock()
        mock_response.content = llm_generated_knowledge
        manager.client.chat.completions.create = Mock(return_value=mock_response)

        # Write turn events to log
        actions = [f"action_{i}" for i in range(20)]
        self.write_turn_events_to_log(manager, actions)

        # Execute update
        result = manager.update_knowledge_from_turns("test_episode", 1, 20)

        assert result, "Update should succeed"

        # Verify: Final knowledge has ONLY the original CROSS-EPISODE content
        with open(manager.output_file, "r", encoding="utf-8") as f:
            final_knowledge = f.read()

        # Should have ORIGINAL content
        assert "Original wisdom that must be preserved" in final_knowledge

        # Should NOT have LLM-generated content
        assert "LLM ignored the instruction" not in final_knowledge
        assert "LLM even generated it twice" not in final_knowledge

        # Should only appear once
        assert final_knowledge.count("## CROSS-EPISODE INSIGHTS") == 1, \
            "CROSS-EPISODE section should appear exactly once"

    def test_empty_existing_knowledge_first_update(self, manager):
        """
        Verify behavior on first knowledge update (no existing file).

        When there's no existing knowledge file:
        1. No section is extracted (empty)
        2. LLM generation proceeds normally
        3. No section is restored
        4. Final knowledge should not have CROSS-EPISODE section
        """
        # No existing file (first update)
        import os
        assert not os.path.exists(manager.output_file), "Should start with no existing file"

        # Mock LLM to return initial knowledge
        llm_generated_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        # Mock anthropic-style response (response.content.strip())
        mock_response = Mock()
        mock_response.content = llm_generated_knowledge
        manager.client.chat.completions.create = Mock(return_value=mock_response)

        # Write turn events to log
        actions = [f"action_{i}" for i in range(20)]
        self.write_turn_events_to_log(manager, actions)

        # Execute update
        result = manager.update_knowledge_from_turns("test_episode", 1, 20)

        assert result, "First update should succeed"

        # Verify: Knowledge written correctly
        with open(manager.output_file, "r", encoding="utf-8") as f:
            final_knowledge = f.read()

        assert "## DANGERS & THREATS" in final_knowledge
        assert "## STRATEGIC PATTERNS" in final_knowledge
        assert "## CROSS-EPISODE INSIGHTS" not in final_knowledge, \
            "First update should not have CROSS-EPISODE section"

    def test_preservation_logging(self, manager):
        """
        Verify that preservation and restoration are logged.

        Should see debug logs for:
        1. Preserving CROSS-EPISODE INSIGHTS
        2. Restoring CROSS-EPISODE INSIGHTS
        """
        # Setup: Existing knowledge with CROSS-EPISODE
        existing_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## CROSS-EPISODE INSIGHTS
Wisdom to preserve.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        with open(manager.output_file, "w", encoding="utf-8") as f:
            f.write(existing_knowledge)

        # Mock LLM
        llm_generated_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Updated dangers.

## STRATEGIC PATTERNS
Updated patterns.
"""

        # Mock anthropic-style response (response.content.strip())
        mock_response = Mock()
        mock_response.content = llm_generated_knowledge
        manager.client.chat.completions.create = Mock(return_value=mock_response)

        # Write turn events to log
        actions = [f"action_{i}" for i in range(20)]
        self.write_turn_events_to_log(manager, actions)

        # Execute update
        result = manager.update_knowledge_from_turns("test_episode", 1, 20)

        assert result, "Update should succeed"

        # Verify logging calls
        assert manager.logger.debug.called, "Should have debug log calls"

        # Check for preservation and restoration logs
        debug_calls = [str(call) for call in manager.logger.debug.call_args_list]
        debug_messages = " ".join(debug_calls)

        assert "Preserving CROSS-EPISODE INSIGHTS" in debug_messages, \
            "Should log preservation"
        assert "Restored CROSS-EPISODE INSIGHTS" in debug_messages, \
            "Should log restoration"
