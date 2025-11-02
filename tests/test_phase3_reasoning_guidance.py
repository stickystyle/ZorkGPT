"""
Tests for Phase 3: Agent Prompt Reasoning Guidance.

Verifies that the agent prompt includes explicit guidance on using
previous reasoning to maintain strategic continuity across turns.
"""

import pytest
from unittest.mock import Mock
from zork_agent import ZorkAgent


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.error = Mock()
    logger.warning = Mock()
    logger.debug = Mock()
    return logger


class TestAgentPromptReasoningGuidance:
    """Test that agent prompt includes reasoning guidance section."""

    def test_agent_prompt_includes_reasoning_guidance_section(self, mock_logger):
        """Test that agent system prompt includes 'USING YOUR PREVIOUS REASONING' section."""
        agent = ZorkAgent(logger=mock_logger)

        # Verify the guidance section header is present
        assert "USING YOUR PREVIOUS REASONING:" in agent.system_prompt, \
            "Agent prompt should include reasoning guidance section header"

    def test_reasoning_guidance_appears_before_output_format(self, mock_logger):
        """Test that reasoning guidance appears before OUTPUT FORMAT section."""
        agent = ZorkAgent(logger=mock_logger)

        guidance_pos = agent.system_prompt.find("USING YOUR PREVIOUS REASONING:")
        output_format_pos = agent.system_prompt.find("OUTPUT FORMAT")

        assert guidance_pos > 0, "Reasoning guidance section should be present"
        assert output_format_pos > 0, "OUTPUT FORMAT section should be present"
        assert guidance_pos < output_format_pos, \
            "Reasoning guidance should appear before OUTPUT FORMAT section"

    def test_reasoning_guidance_includes_three_scenarios(self, mock_logger):
        """Test that reasoning guidance includes three key scenarios."""
        agent = ZorkAgent(logger=mock_logger)

        # Find the guidance section
        guidance_start = agent.system_prompt.find("USING YOUR PREVIOUS REASONING:")
        guidance_end = agent.system_prompt.find("OUTPUT FORMAT")
        guidance_section = agent.system_prompt[guidance_start:guidance_end]

        # Verify three key scenarios are present
        assert "Continuing a plan?" in guidance_section, \
            "Guidance should mention continuing existing plans"

        assert "New information requires revision?" in guidance_section, \
            "Guidance should mention revising plans based on new information"

        assert "Starting fresh?" in guidance_section, \
            "Guidance should mention starting new plans"

    def test_reasoning_guidance_emphasizes_continuity(self, mock_logger):
        """Test that reasoning guidance emphasizes building on previous thinking."""
        agent = ZorkAgent(logger=mock_logger)

        guidance_start = agent.system_prompt.find("USING YOUR PREVIOUS REASONING:")
        guidance_end = agent.system_prompt.find("OUTPUT FORMAT")
        guidance_section = agent.system_prompt[guidance_start:guidance_end]

        # Verify emphasis on continuity
        assert "build on" in guidance_section.lower() or "revise" in guidance_section.lower(), \
            "Guidance should emphasize building on or revising previous thinking"

        assert "previous thinking" in guidance_section.lower() or "previous reasoning" in guidance_section.lower(), \
            "Guidance should reference previous thinking/reasoning"

    def test_reasoning_guidance_mentions_context_section(self, mock_logger):
        """Test that guidance mentions the '## Previous Reasoning and Actions' context section."""
        agent = ZorkAgent(logger=mock_logger)

        guidance_start = agent.system_prompt.find("USING YOUR PREVIOUS REASONING:")
        guidance_end = agent.system_prompt.find("OUTPUT FORMAT")
        guidance_section = agent.system_prompt[guidance_start:guidance_end]

        # Verify it references the context section
        assert "## Previous Reasoning and Actions" in guidance_section, \
            "Guidance should mention the '## Previous Reasoning and Actions' context section"

    def test_reasoning_guidance_structure_is_clear(self, mock_logger):
        """Test that reasoning guidance is structured clearly with numbered scenarios."""
        agent = ZorkAgent(logger=mock_logger)

        guidance_start = agent.system_prompt.find("USING YOUR PREVIOUS REASONING:")
        guidance_end = agent.system_prompt.find("OUTPUT FORMAT")
        guidance_section = agent.system_prompt[guidance_start:guidance_end]

        # Verify numbered structure
        assert "1." in guidance_section, "Guidance should include numbered list item 1"
        assert "2." in guidance_section, "Guidance should include numbered list item 2"
        assert "3." in guidance_section, "Guidance should include numbered list item 3"

    def test_full_prompt_structure_intact(self, mock_logger):
        """Test that adding reasoning guidance didn't break existing prompt structure."""
        agent = ZorkAgent(logger=mock_logger)

        # Verify key existing sections are still present
        assert "CRITICAL RULES:" in agent.system_prompt, \
            "CRITICAL RULES section should still be present"

        assert "NAVIGATION PROTOCOL:" in agent.system_prompt, \
            "NAVIGATION PROTOCOL section should still be present"

        assert "COMMAND SYNTAX:" in agent.system_prompt, \
            "COMMAND SYNTAX section should still be present"

        assert "OUTPUT FORMAT" in agent.system_prompt, \
            "OUTPUT FORMAT section should still be present"

    def test_reasoning_guidance_placement_in_workflow(self, mock_logger):
        """Test that reasoning guidance is positioned logically in the prompt workflow."""
        agent = ZorkAgent(logger=mock_logger)

        # Find positions of key sections
        common_actions_pos = agent.system_prompt.find("COMMON ACTIONS:")
        reasoning_guidance_pos = agent.system_prompt.find("USING YOUR PREVIOUS REASONING:")
        output_format_pos = agent.system_prompt.find("OUTPUT FORMAT")
        anti_patterns_pos = agent.system_prompt.find("ANTI-PATTERNS TO AVOID:")

        # Verify logical ordering
        assert common_actions_pos < reasoning_guidance_pos, \
            "Reasoning guidance should come after COMMON ACTIONS"

        assert reasoning_guidance_pos < output_format_pos, \
            "Reasoning guidance should come before OUTPUT FORMAT"

        assert output_format_pos < anti_patterns_pos, \
            "OUTPUT FORMAT should come before ANTI-PATTERNS"


class TestReasoningGuidanceIntegration:
    """Test integration of reasoning guidance with context."""

    def test_agent_receives_reasoning_in_context_and_guidance_in_prompt(self, mock_logger):
        """
        Integration test: Verify the complete flow works together.

        The agent should receive:
        1. Previous reasoning in the context (from ContextManager)
        2. Guidance on using that reasoning in the prompt (from agent.md)
        """
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration

        # Create test configuration
        config = GameConfiguration(
            max_turns_per_episode=1000,
            turn_delay_seconds=0.0,
            game_file_path="test.z5",
            critic_rejection_threshold=0.5,
            episode_log_file="test.log",
            json_log_file="test.jsonl",
            state_export_file="test_state.json",
            zork_game_workdir="test",
            client_base_url="http://localhost:1234",
            client_api_key="test_key",
            agent_model="test-agent",
            critic_model="test-critic",
            info_ext_model="test-extractor",
            analysis_model="test-analysis",
            memory_model="test-memory",
            condensation_model="test-condensation",
            knowledge_update_interval=100,
            objective_update_interval=20,
            enable_objective_refinement=True,
            objective_refinement_interval=200,
            max_objectives_before_forced_refinement=15,
            refined_objectives_target_count=10,
            max_context_tokens=100000,
            context_overflow_threshold=0.8,
            enable_state_export=True,
            s3_bucket="test-bucket",
            s3_key_prefix="test/",
            simple_memory_file="Memories.md",
            simple_memory_max_shown=10,
            map_state_file="test_map.json",
            knowledge_file="test_knowledgebase.md",
            agent_sampling={},
            critic_sampling={},
            extractor_sampling={},
            analysis_sampling={},
            memory_sampling={},
            condensation_sampling={},
        )

        # Create game state with reasoning history
        game_state = GameState()
        game_state.action_reasoning_history.append({
            "turn": 1,
            "reasoning": "I should explore north to find the treasure room.",
            "action": "go north",
            "timestamp": "2025-11-02T10:00:00"
        })
        game_state.action_history.append(("go north", "You enter a dark room."))

        # Create context manager and agent
        context_manager = ContextManager(mock_logger, config, game_state)
        agent = ZorkAgent(logger=mock_logger)

        # Get context with reasoning
        context = context_manager.get_agent_context(
            current_state="You are in a dark room.",
            inventory=[],
            location="Dark Room",
        )
        formatted_context = context_manager.get_formatted_agent_prompt_context(context)

        # Verify both pieces are present:
        # 1. Context includes previous reasoning
        assert "## Previous Reasoning and Actions" in formatted_context, \
            "Context should include previous reasoning section"
        assert "I should explore north to find the treasure room." in formatted_context, \
            "Context should include the actual previous reasoning"

        # 2. Agent prompt includes guidance on using reasoning
        assert "USING YOUR PREVIOUS REASONING:" in agent.system_prompt, \
            "Agent prompt should include reasoning guidance"
        assert "Continuing a plan?" in agent.system_prompt, \
            "Agent prompt should guide on continuing plans"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
