"""
ABOUTME: Integration tests for cross-episode synthesis writing to knowledgebase.md.
ABOUTME: Validates that synthesize_inter_episode_wisdom updates the correct file and section.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from knowledge import AdaptiveKnowledgeManager
from session.game_configuration import GameConfiguration


class TestCrossEpisodeSynthesisIntegration:
    """Integration tests for cross-episode synthesis workflow."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager with temporary files."""
        log_file = tmp_path / "test_log.jsonl"
        output_file = tmp_path / "knowledgebase.md"
        workdir = tmp_path / "game_files"
        workdir.mkdir(exist_ok=True)

        # Create a minimal episode log file
        with open(log_file, "w") as f:
            f.write(
                '{"episode": 1, "turn": 1, "action": "north", "response": "You are in a forest."}\n'
            )

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

    @pytest.fixture
    def episode_data(self):
        """Create sample episode data."""
        return {
            "episode_id": 1,
            "turn_count": 50,
            "final_score": 60,
            "death_count": 1,
            "episode_ended_in_death": True,
            "discovered_objectives": ["explore west of house", "find lamp"],
            "completed_objectives": ["explore west of house"],
            "avg_critic_score": 0.75,
        }

    def test_synthesis_creates_knowledge_base_if_missing(self, manager, episode_data):
        """Test that synthesis creates knowledgebase.md if it doesn't exist."""
        # Mock the LLM client
        mock_response = Mock()
        mock_response.content = "New cross-episode insights content."

        with patch.object(
            manager.client.chat.completions, "create", return_value=mock_response
        ):
            # Mock the turn data extraction
            with patch.object(
                manager,
                "_extract_turn_window_data",
                return_value={
                    "episode_id": 1,
                    "start_turn": 1,
                    "end_turn": 50,
                    "actions_and_responses": [
                        {"turn": 1, "action": "north", "response": "Forest"}
                    ],
                    "death_events": [
                        {
                            "turn": 50,
                            "reason": "Eaten by grue",
                            "death_location": "Dark cellar",
                            "action_taken": "go east",
                        }
                    ],
                },
            ):
                result = manager.synthesize_inter_episode_wisdom(episode_data)

        assert result is True
        assert os.path.exists(manager.output_file)

        # Verify content
        with open(manager.output_file, "r") as f:
            content = f.read()

        assert "# Zork Game World Knowledge Base" in content
        assert "## CROSS-EPISODE INSIGHTS" in content
        assert "New cross-episode insights content." in content

    def test_synthesis_updates_existing_knowledge_base(self, manager, episode_data):
        """Test that synthesis updates existing knowledgebase.md correctly."""
        # Create existing knowledge base
        existing_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Watch out for grues.

## CROSS-EPISODE INSIGHTS
Old insights that should be replaced.
"""
        with open(manager.output_file, "w") as f:
            f.write(existing_content)

        # Mock the LLM client
        mock_response = Mock()
        mock_response.content = "Updated cross-episode insights."

        with patch.object(
            manager.client.chat.completions, "create", return_value=mock_response
        ):
            # Mock the turn data extraction
            with patch.object(
                manager,
                "_extract_turn_window_data",
                return_value={
                    "episode_id": 1,
                    "start_turn": 1,
                    "end_turn": 50,
                    "actions_and_responses": [
                        {"turn": 1, "action": "north", "response": "Forest"}
                    ],
                    "death_events": [],
                },
            ):
                result = manager.synthesize_inter_episode_wisdom(episode_data)

        assert result is True

        # Verify content
        with open(manager.output_file, "r") as f:
            content = f.read()

        # Verify structure is maintained
        assert "## DANGERS & THREATS" in content
        assert "Watch out for grues." in content

        # Verify cross-episode section is updated
        assert "## CROSS-EPISODE INSIGHTS" in content
        assert "Updated cross-episode insights." in content
        assert "Old insights that should be replaced." not in content

    def test_synthesis_skips_insignificant_episodes(self, manager):
        """Test that synthesis skips episodes that don't meet criteria."""
        insignificant_episode = {
            "episode_id": 1,
            "turn_count": 10,
            "final_score": 5,
            "death_count": 0,
            "episode_ended_in_death": False,
            "discovered_objectives": [],
            "completed_objectives": [],
            "avg_critic_score": 0.1,
        }

        result = manager.synthesize_inter_episode_wisdom(insignificant_episode)

        assert result is False
        # Knowledge base should not be created or modified
        if os.path.exists(manager.output_file):
            pytest.fail("Knowledge base should not be created for insignificant episode")

    def test_synthesis_handles_death_episodes(self, manager, episode_data):
        """Test that synthesis always processes episodes ending in death."""
        death_episode = {
            "episode_id": 1,
            "turn_count": 5,  # Very short
            "final_score": 0,  # No score
            "death_count": 1,
            "episode_ended_in_death": True,  # But death occurred
            "discovered_objectives": [],
            "completed_objectives": [],
            "avg_critic_score": 0.0,
        }

        # Mock the LLM client
        mock_response = Mock()
        mock_response.content = "Death analysis insights."

        with patch.object(
            manager.client.chat.completions, "create", return_value=mock_response
        ):
            # Mock the turn data extraction
            with patch.object(
                manager,
                "_extract_turn_window_data",
                return_value={
                    "episode_id": 1,
                    "start_turn": 1,
                    "end_turn": 5,
                    "actions_and_responses": [
                        {"turn": 5, "action": "go east", "response": "Eaten by grue"}
                    ],
                    "death_events": [
                        {
                            "turn": 5,
                            "reason": "Eaten by grue",
                            "death_location": "Dark cellar",
                            "action_taken": "go east",
                        }
                    ],
                },
            ):
                result = manager.synthesize_inter_episode_wisdom(death_episode)

        # Should process even though other criteria aren't met
        assert result is True

    def test_synthesis_preserves_section_order(self, manager, episode_data):
        """Test that synthesis preserves the order of knowledge base sections."""
        # Create knowledge base with multiple sections
        existing_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Danger content.

## PUZZLE SOLUTIONS
Puzzle content.

## STRATEGIC PATTERNS
Strategy content.
"""
        with open(manager.output_file, "w") as f:
            f.write(existing_content)

        # Mock the LLM client
        mock_response = Mock()
        mock_response.content = "New insights."

        with patch.object(
            manager.client.chat.completions, "create", return_value=mock_response
        ):
            with patch.object(
                manager,
                "_extract_turn_window_data",
                return_value={
                    "episode_id": 1,
                    "start_turn": 1,
                    "end_turn": 50,
                    "actions_and_responses": [{"turn": 1, "action": "n", "response": "ok"}],
                    "death_events": [],
                },
            ):
                result = manager.synthesize_inter_episode_wisdom(episode_data)

        assert result is True

        # Verify content and order
        with open(manager.output_file, "r") as f:
            content = f.read()

        # Find positions of sections
        dangers_pos = content.find("## DANGERS & THREATS")
        puzzles_pos = content.find("## PUZZLE SOLUTIONS")
        strategy_pos = content.find("## STRATEGIC PATTERNS")
        insights_pos = content.find("## CROSS-EPISODE INSIGHTS")

        # Verify CROSS-EPISODE INSIGHTS is after other content
        assert dangers_pos < insights_pos
        assert puzzles_pos < insights_pos
        assert strategy_pos < insights_pos

    def test_synthesis_handles_llm_failure(self, manager, episode_data):
        """Test that synthesis handles LLM failures gracefully."""
        # Mock the LLM client to raise an exception
        with patch.object(
            manager.client.chat.completions,
            "create",
            side_effect=Exception("LLM API error"),
        ):
            with patch.object(
                manager,
                "_extract_turn_window_data",
                return_value={
                    "episode_id": 1,
                    "start_turn": 1,
                    "end_turn": 50,
                    "actions_and_responses": [{"turn": 1, "action": "n", "response": "ok"}],
                    "death_events": [],
                },
            ):
                result = manager.synthesize_inter_episode_wisdom(episode_data)

        # Should return False on failure
        assert result is False

    def test_synthesis_handles_missing_turn_data(self, manager, episode_data):
        """Test that synthesis handles missing turn data gracefully."""
        # Mock turn data extraction to return None
        with patch.object(manager, "_extract_turn_window_data", return_value=None):
            result = manager.synthesize_inter_episode_wisdom(episode_data)

        # Should return False when turn data is missing
        assert result is False

    def test_synthesis_prompt_includes_death_analysis(self, manager, episode_data):
        """Test that synthesis prompt includes death event analysis."""
        # Mock the LLM client and capture the prompt
        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get("messages", [{}])[1].get("content", "")
            mock_response = Mock()
            mock_response.content = "New insights."
            return mock_response

        with patch.object(
            manager.client.chat.completions, "create", side_effect=capture_prompt
        ):
            with patch.object(
                manager,
                "_extract_turn_window_data",
                return_value={
                    "episode_id": 1,
                    "start_turn": 1,
                    "end_turn": 50,
                    "actions_and_responses": [{"turn": 1, "action": "n", "response": "ok"}],
                    "death_events": [
                        {
                            "turn": 50,
                            "reason": "Eaten by grue",
                            "death_location": "Dark cellar",
                            "action_taken": "go east",
                            "death_context": "No light source",
                        }
                    ],
                },
            ):
                manager.synthesize_inter_episode_wisdom(episode_data)

        # Verify death analysis is in the prompt
        assert captured_prompt is not None
        assert "DEATH EVENT ANALYSIS:" in captured_prompt
        assert "Eaten by grue" in captured_prompt
        assert "Dark cellar" in captured_prompt
        assert "go east" in captured_prompt

    def test_synthesis_prompt_includes_existing_insights(self, manager, episode_data):
        """Test that synthesis prompt includes existing cross-episode insights."""
        # Create knowledge base with existing insights
        existing_content = """# Zork Game World Knowledge Base

## CROSS-EPISODE INSIGHTS
Previous wisdom:
- Always carry lamp in dark areas
- Avoid trolls without weapons
"""
        with open(manager.output_file, "w") as f:
            f.write(existing_content)

        # Mock the LLM client and capture the prompt
        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get("messages", [{}])[1].get("content", "")
            mock_response = Mock()
            mock_response.content = "Updated insights."
            return mock_response

        with patch.object(
            manager.client.chat.completions, "create", side_effect=capture_prompt
        ):
            with patch.object(
                manager,
                "_extract_turn_window_data",
                return_value={
                    "episode_id": 1,
                    "start_turn": 1,
                    "end_turn": 50,
                    "actions_and_responses": [{"turn": 1, "action": "n", "response": "ok"}],
                    "death_events": [],
                },
            ):
                manager.synthesize_inter_episode_wisdom(episode_data)

        # Verify existing insights are in the prompt
        assert captured_prompt is not None
        assert "EXISTING CROSS-EPISODE INSIGHTS:" in captured_prompt
        assert "Always carry lamp in dark areas" in captured_prompt
        assert "Avoid trolls without weapons" in captured_prompt
