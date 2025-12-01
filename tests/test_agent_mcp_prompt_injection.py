# ABOUTME: Tests for MCP prompt injection in ZorkAgent
# ABOUTME: Validates conditional injection of thoughtbox guidance when MCP enabled

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from zork_agent import ZorkAgent
from session.game_configuration import GameConfiguration


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_config():
    """Base GameConfiguration with MCP disabled."""
    config = GameConfiguration.from_toml()
    config.mcp_enabled = False
    return config


@pytest.fixture
def mcp_enabled_config():
    """GameConfiguration with MCP enabled."""
    config = GameConfiguration.from_toml()
    config.mcp_enabled = True
    return config


@pytest.fixture
def mock_llm_client():
    """Mock LLM client wrapper for testing."""
    wrapper = MagicMock()
    wrapper.client = MagicMock()
    wrapper.client._supports_tool_calling = MagicMock(return_value=True)
    wrapper.client.chat_completions_create = MagicMock()
    return wrapper


@pytest.fixture
def sample_base_prompt():
    """Sample base agent prompt for testing."""
    return """You are an intelligent agent playing Zork.

**CRITICAL RULES:**
1. Do something important

**OUTPUT FORMAT (REQUIRED):**

You must respond with valid JSON.
"""


@pytest.fixture
def sample_base_prompt_lowercase():
    """Sample base agent prompt with lowercase OUTPUT FORMAT marker."""
    return """You are an intelligent agent playing Zork.

**CRITICAL RULES:**
1. Do something important

**Output Format (REQUIRED):**

You must respond with valid JSON.
"""


# ============================================================================
# Test Classes
# ============================================================================


class TestEnhancePromptWithMCP:
    """Test _enhance_prompt_with_mcp method."""

    def test_injects_guidance_when_mcp_enabled(
        self, mcp_enabled_config, mock_llm_client, sample_base_prompt
    ):
        """
        Verify MCP guidance is injected when mcp_enabled=True.

        Test approach:
        1. Create agent with mcp_enabled=True
        2. Call _enhance_prompt_with_mcp with sample prompt
        3. Verify thoughtbox guidance is present in output
        """
        # Arrange
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=mcp_enabled_config,
                model="gpt-4",
                client=mock_llm_client,
            )

        # Act
        result = agent._enhance_prompt_with_mcp(sample_base_prompt)

        # Assert
        assert "STRUCTURED REASONING TOOL" in result, "Should inject MCP section header"
        assert "thoughtbox.clear_thought" in result, "Should mention thoughtbox tool"
        assert "Use thoughtbox when:" in result, "Should include when to use guidance"
        assert "Do NOT use thoughtbox for:" in result, "Should include when not to use"

    def test_no_injection_when_mcp_disabled(
        self, base_config, mock_llm_client, sample_base_prompt
    ):
        """
        Verify MCP guidance is NOT injected when mcp_enabled=False.

        Test approach:
        1. Create agent with mcp_enabled=False
        2. Call _enhance_prompt_with_mcp with sample prompt
        3. Verify thoughtbox guidance is absent
        """
        # Arrange
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=base_config,
                model="gpt-4",
                client=mock_llm_client,
            )

        # Act
        result = agent._enhance_prompt_with_mcp(sample_base_prompt)

        # Assert
        assert result == sample_base_prompt, "Prompt should be unchanged when MCP disabled"
        assert "STRUCTURED REASONING TOOL" not in result, "Should not inject MCP section"
        assert "thoughtbox" not in result, "Should not mention thoughtbox"

    def test_inserts_before_output_format_uppercase(
        self, mcp_enabled_config, mock_llm_client, sample_base_prompt
    ):
        """
        Verify MCP guidance is inserted before **OUTPUT FORMAT section.

        Test approach:
        1. Create agent with mcp_enabled=True
        2. Call _enhance_prompt_with_mcp with prompt containing **OUTPUT FORMAT
        3. Verify MCP section appears before OUTPUT FORMAT
        """
        # Arrange
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=mcp_enabled_config,
                model="gpt-4",
                client=mock_llm_client,
            )

        # Act
        result = agent._enhance_prompt_with_mcp(sample_base_prompt)

        # Assert
        mcp_pos = result.find("STRUCTURED REASONING TOOL")
        output_format_pos = result.find("**OUTPUT FORMAT")
        assert mcp_pos < output_format_pos, "MCP section should appear before OUTPUT FORMAT"

    def test_appends_when_no_output_format_marker(
        self, mcp_enabled_config, mock_llm_client
    ):
        """
        Verify MCP guidance is appended when no OUTPUT FORMAT marker exists.

        Test approach:
        1. Create agent with mcp_enabled=True
        2. Call _enhance_prompt_with_mcp with prompt without OUTPUT FORMAT
        3. Verify MCP section is appended at end
        """
        # Arrange
        prompt_without_marker = "You are an agent. Do stuff."
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=mcp_enabled_config,
                model="gpt-4",
                client=mock_llm_client,
            )

        # Act
        result = agent._enhance_prompt_with_mcp(prompt_without_marker)

        # Assert
        assert result.startswith("You are an agent"), "Original content should be preserved"
        assert "STRUCTURED REASONING TOOL" in result, "MCP section should be present"
        assert result.endswith("```\n\n"), "MCP section should be at end"

    def test_contains_example_json_output(
        self, mcp_enabled_config, mock_llm_client, sample_base_prompt
    ):
        """
        Verify MCP guidance contains example JSON output format.

        Test approach:
        1. Create agent with mcp_enabled=True
        2. Call _enhance_prompt_with_mcp
        3. Verify example JSON with thinking/action/new_objective is present
        """
        # Arrange
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=mcp_enabled_config,
                model="gpt-4",
                client=mock_llm_client,
            )

        # Act
        result = agent._enhance_prompt_with_mcp(sample_base_prompt)

        # Assert
        assert '"thinking":' in result, "Example should include thinking field"
        assert '"action":' in result, "Example should include action field"
        assert '"new_objective":' in result, "Example should include new_objective field"


class TestLoadSystemPromptWithMCP:
    """Test that _load_system_prompt integrates MCP enhancement."""

    def test_system_prompt_includes_mcp_when_enabled(
        self, mcp_enabled_config, mock_llm_client, tmp_path
    ):
        """
        Verify system prompt includes MCP guidance when loaded with MCP enabled.

        Test approach:
        1. Create temp agent.md file
        2. Create agent with mcp_enabled=True
        3. Verify loaded system_prompt contains MCP guidance
        """
        # Arrange - create temp agent.md
        agent_md = tmp_path / "agent.md"
        agent_md.write_text("""You are an agent.

**OUTPUT FORMAT (REQUIRED):**

Respond with JSON.
""")
        original_cwd = os.getcwd()

        try:
            os.chdir(tmp_path)

            # Act
            agent = ZorkAgent(
                config=mcp_enabled_config,
                model="gpt-4",
                client=mock_llm_client,
            )

            # Assert
            assert "STRUCTURED REASONING TOOL" in agent.system_prompt
            assert "thoughtbox.clear_thought" in agent.system_prompt

        finally:
            os.chdir(original_cwd)

    def test_system_prompt_excludes_mcp_when_disabled(
        self, base_config, mock_llm_client, tmp_path
    ):
        """
        Verify system prompt excludes MCP guidance when loaded with MCP disabled.

        Test approach:
        1. Create temp agent.md file
        2. Create agent with mcp_enabled=False
        3. Verify loaded system_prompt does not contain MCP guidance
        """
        # Arrange - create temp agent.md
        agent_md = tmp_path / "agent.md"
        agent_md.write_text("""You are an agent.

**OUTPUT FORMAT (REQUIRED):**

Respond with JSON.
""")
        original_cwd = os.getcwd()

        try:
            os.chdir(tmp_path)

            # Act
            agent = ZorkAgent(
                config=base_config,
                model="gpt-4",
                client=mock_llm_client,
            )

            # Assert
            assert "STRUCTURED REASONING TOOL" not in agent.system_prompt
            assert "thoughtbox" not in agent.system_prompt

        finally:
            os.chdir(original_cwd)


class TestMCPPromptContent:
    """Test the content of the MCP prompt guidance."""

    def test_includes_puzzle_guidance(
        self, mcp_enabled_config, mock_llm_client, sample_base_prompt
    ):
        """
        Verify MCP guidance includes puzzle-solving use case.

        Test approach:
        1. Create agent with MCP enabled
        2. Get enhanced prompt
        3. Verify puzzle-related guidance is present
        """
        # Arrange
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=mcp_enabled_config,
                model="gpt-4",
                client=mock_llm_client,
            )

        # Act
        result = agent._enhance_prompt_with_mcp(sample_base_prompt)

        # Assert
        assert "Puzzle feedback" in result, "Should mention puzzle use case"
        assert "systematic experimentation" in result, "Should reference experimentation"

    def test_includes_anti_overuse_guidance(
        self, mcp_enabled_config, mock_llm_client, sample_base_prompt
    ):
        """
        Verify MCP guidance includes when NOT to use thoughtbox.

        Test approach:
        1. Create agent with MCP enabled
        2. Get enhanced prompt
        3. Verify anti-overuse guidance is present
        """
        # Arrange
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=mcp_enabled_config,
                model="gpt-4",
                client=mock_llm_client,
            )

        # Act
        result = agent._enhance_prompt_with_mcp(sample_base_prompt)

        # Assert
        assert "Simple movement" in result, "Should warn against use for simple movement"
        assert "Combat situations" in result, "Should warn against use in combat"
        assert "Obvious single actions" in result, "Should warn against obvious actions"

    def test_includes_iteration_guidance(
        self, mcp_enabled_config, mock_llm_client, sample_base_prompt
    ):
        """
        Verify MCP guidance includes iteration count guidance.

        Test approach:
        1. Create agent with MCP enabled
        2. Get enhanced prompt
        3. Verify iteration guidance is present
        """
        # Arrange
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=mcp_enabled_config,
                model="gpt-4",
                client=mock_llm_client,
            )

        # Act
        result = agent._enhance_prompt_with_mcp(sample_base_prompt)

        # Assert
        assert "2-4 thoughts" in result, "Should suggest typical thought count for puzzles"
        assert "1-2 for strategic" in result, "Should suggest count for strategic decisions"
        assert "nextThoughtNeeded" in result, "Should explain continuation flag"
