# ABOUTME: Tests for ZorkAgent async action generation setup (Task #8)
# ABOUTME: Validates MCP session connection, tool schema retrieval, message building, and compatibility checks

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import is_dataclass

from zork_agent import ZorkAgent, MCPContext
from managers.mcp_config import MCPError
from session.game_configuration import GameConfiguration


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_mcp_manager():
    """Mock MCPManager for testing."""
    manager = MagicMock()
    manager.connect_session = AsyncMock()
    manager.disconnect_session = AsyncMock()
    manager.get_tool_schemas = AsyncMock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "test.tool",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    manager.is_disabled = False
    return manager


@pytest.fixture
def mock_llm_client():
    """Mock LLM client wrapper for testing."""
    wrapper = MagicMock()
    # LLMClientWrapper has .client which is the LLMClient
    wrapper.client = MagicMock()
    wrapper.client._supports_tool_calling = MagicMock(return_value=True)
    return wrapper


@pytest.fixture
def test_config():
    """GameConfiguration for testing - loads from pyproject.toml."""
    return GameConfiguration.from_toml()


@pytest.fixture
def agent_without_mcp(test_config, mock_llm_client):
    """Agent instance without MCP manager."""
    with patch.object(ZorkAgent, "_load_system_prompt"):
        agent = ZorkAgent(
            config=test_config,
            model="gpt-4",
            client=mock_llm_client,
            mcp_manager=None,
        )
        agent.system_prompt = "You are a Zork agent."
    return agent


@pytest.fixture
def agent_with_mcp(test_config, mock_llm_client, mock_mcp_manager):
    """Agent instance with MCP manager."""
    with patch.object(ZorkAgent, "_load_system_prompt"):
        agent = ZorkAgent(
            config=test_config,
            model="gpt-4",
            client=mock_llm_client,
            mcp_manager=mock_mcp_manager,
        )
        agent.system_prompt = "You are a Zork agent."
    return agent


@pytest.fixture
def agent_o1_model(test_config, mock_llm_client, mock_mcp_manager):
    """Agent with o1 model (requires system prompt in user role)."""
    with patch.object(ZorkAgent, "_load_system_prompt"):
        agent = ZorkAgent(
            config=test_config,
            model="o1-preview",
            client=mock_llm_client,
            mcp_manager=mock_mcp_manager,
        )
        agent.system_prompt = "You are a Zork agent."
    return agent


@pytest.fixture
def agent_o3_model(test_config, mock_llm_client, mock_mcp_manager):
    """Agent with o3 model (requires system prompt in user role)."""
    with patch.object(ZorkAgent, "_load_system_prompt"):
        agent = ZorkAgent(
            config=test_config,
            model="o3-mini",
            client=mock_llm_client,
            mcp_manager=mock_mcp_manager,
        )
        agent.system_prompt = "You are a Zork agent."
    return agent


# ============================================================================
# TestMessageHistoryBuilding
# ============================================================================


class TestMessageHistoryBuilding:
    """Tests for message history construction with cache control (Req 4.7)."""

    @pytest.mark.asyncio
    async def test_system_message_has_cache_control(self, agent_without_mcp):
        """System message should have ephemeral cache control."""
        result = await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Find system message
        system_msg = next(
            (m for m in result.messages if m.get("role") == "system"), None
        )

        assert system_msg is not None, "System message should exist"
        assert "cache_control" in system_msg, "System message should have cache_control"
        assert system_msg["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_user_message_has_cache_control(self, agent_without_mcp):
        """User message should have ephemeral cache control."""
        result = await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Find user message (the one with game state, not system prompt)
        user_messages = [m for m in result.messages if m.get("role") == "user"]
        # For non-o1 models, there's one user message with game state
        user_msg = user_messages[-1] if user_messages else None

        assert user_msg is not None, "User message should exist"
        assert "cache_control" in user_msg, "User message should have cache_control"
        assert user_msg["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_memories_combined_with_game_state(self, agent_without_mcp):
        """Memories should be combined with game state in user message."""
        memories = "Memory 1: Found a lamp\nMemory 2: Opened mailbox"

        result = await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=memories
        )

        # Find user message with game state
        user_messages = [m for m in result.messages if m.get("role") == "user"]
        user_msg = user_messages[-1] if user_messages else None

        assert user_msg is not None
        content = user_msg.get("content", "")

        # Both game state and memories should be in the message
        assert "You are in a room." in content
        assert "Memory 1: Found a lamp" in content
        assert "Memory 2: Opened mailbox" in content

    @pytest.mark.asyncio
    async def test_o1_model_system_prompt_in_user_role(self, agent_o1_model):
        """O1 models should have system prompt in user role, not system role."""
        result = await agent_o1_model._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Should NOT have a system message
        system_msg = next(
            (m for m in result.messages if m.get("role") == "system"), None
        )
        assert system_msg is None, "O1 models should not have system message"

        # First user message should contain system prompt
        user_messages = [m for m in result.messages if m.get("role") == "user"]
        assert len(user_messages) >= 1, "Should have at least one user message"
        first_user_msg = user_messages[0]
        assert (
            "You are a Zork agent." in first_user_msg.get("content", "")
        ), "System prompt should be in user message"

    @pytest.mark.asyncio
    async def test_o3_model_system_prompt_in_user_role(self, agent_o3_model):
        """O3 models should have system prompt in user role, not system role."""
        result = await agent_o3_model._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Should NOT have a system message
        system_msg = next(
            (m for m in result.messages if m.get("role") == "system"), None
        )
        assert system_msg is None, "O3 models should not have system message"

        # First user message should contain system prompt
        user_messages = [m for m in result.messages if m.get("role") == "user"]
        assert len(user_messages) >= 1, "Should have at least one user message"
        first_user_msg = user_messages[0]
        assert (
            "You are a Zork agent." in first_user_msg.get("content", "")
        ), "System prompt should be in user message"


# ============================================================================
# TestMCPSessionConnection
# ============================================================================


class TestMCPSessionConnection:
    """Tests for MCP session connection during async setup (Req 3.1, 3.3, 3.4)."""

    @pytest.mark.asyncio
    async def test_mcp_session_connects_when_enabled(
        self, agent_with_mcp, mock_mcp_manager
    ):
        """MCP session should connect when manager is provided and enabled."""
        await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Verify connect_session was called
        mock_mcp_manager.connect_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mcp_session_not_connected_when_disabled(self, agent_without_mcp):
        """MCP session should not connect when manager is None."""
        result = await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Should indicate MCP is not connected
        assert result.mcp_connected is False

    @pytest.mark.asyncio
    async def test_mcp_session_not_connected_when_manager_disabled(
        self, agent_with_mcp, mock_mcp_manager
    ):
        """MCP session should not connect when manager.is_disabled is True."""
        mock_mcp_manager.is_disabled = True

        result = await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Should not call connect_session
        mock_mcp_manager.connect_session.assert_not_awaited()

        # Should indicate MCP is not connected
        assert result.mcp_connected is False


# ============================================================================
# TestToolSchemaRetrieval
# ============================================================================


class TestToolSchemaRetrieval:
    """Tests for tool schema retrieval during async setup (Req 1.1, 3.4)."""

    @pytest.mark.asyncio
    async def test_tool_schemas_retrieved_when_mcp_enabled(
        self, agent_with_mcp, mock_mcp_manager
    ):
        """Tool schemas should be retrieved when MCP is enabled."""
        await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Verify get_tool_schemas was called
        mock_mcp_manager.get_tool_schemas.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tool_schemas_none_when_mcp_disabled(self, agent_without_mcp):
        """Tool schemas should be None when MCP is disabled."""
        result = await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        assert result.tool_schemas is None

    @pytest.mark.asyncio
    async def test_tool_schemas_in_result(self, agent_with_mcp, mock_mcp_manager):
        """Tool schemas should be included in MCPContext result."""
        expected_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "test.tool",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        mock_mcp_manager.get_tool_schemas = AsyncMock(return_value=expected_schemas)

        result = await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        assert result.tool_schemas == expected_schemas


# ============================================================================
# TestModelCompatibilityCheck
# ============================================================================


class TestModelCompatibilityCheck:
    """Tests for model compatibility checking with MCP (Req 4.5, 12.5)."""

    @pytest.mark.asyncio
    async def test_incompatible_model_raises_mcp_error(
        self, test_config, mock_llm_client, mock_mcp_manager
    ):
        """Incompatible model should raise MCPError when MCP is enabled."""
        # Make the client report no tool support
        mock_llm_client.client._supports_tool_calling = MagicMock(return_value=False)

        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=test_config,
                model="o1-preview",  # Incompatible model
                client=mock_llm_client,
                mcp_manager=mock_mcp_manager,
            )
            agent.system_prompt = "You are a Zork agent."

        with pytest.raises(MCPError) as exc_info:
            await agent._generate_action_async(
                game_state_text="You are in a room.", relevant_memories=None
            )

        assert "does not support tool calling" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_compatible_model_no_error(self, agent_with_mcp, mock_llm_client):
        """Compatible model should not raise error."""
        # Make the client report tool support
        mock_llm_client.client._supports_tool_calling = MagicMock(return_value=True)

        # Should not raise
        result = await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_force_tool_support_bypasses_check(
        self, test_config, mock_llm_client, mock_mcp_manager
    ):
        """mcp_force_tool_support=True should bypass compatibility check."""
        # Make the client report no tool support
        mock_llm_client.client._supports_tool_calling = MagicMock(return_value=False)

        # Enable force_tool_support on config
        test_config.mcp_force_tool_support = True

        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=test_config,
                model="o1-preview",
                client=mock_llm_client,
                mcp_manager=mock_mcp_manager,
            )
            agent.system_prompt = "You are a Zork agent."

        # Should not raise despite incompatible model
        result = await agent._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        assert result is not None


# ============================================================================
# TestMCPContextResult
# ============================================================================


class TestMCPContextResult:
    """Tests for MCPContext dataclass result (Req 10.3)."""

    @pytest.mark.asyncio
    async def test_returns_mcp_context_dataclass(self, agent_without_mcp):
        """Should return an MCPContext dataclass instance."""
        result = await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        assert isinstance(result, MCPContext)
        assert is_dataclass(result)

    @pytest.mark.asyncio
    async def test_mcp_context_has_messages(self, agent_without_mcp):
        """MCPContext should have messages field populated."""
        result = await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        assert hasattr(result, "messages")
        assert isinstance(result.messages, list)
        assert len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_mcp_context_mcp_connected_true_when_connected(
        self, agent_with_mcp, mock_mcp_manager
    ):
        """MCPContext.mcp_connected should be True when MCP connects."""
        result = await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        assert hasattr(result, "mcp_connected")
        assert result.mcp_connected is True

    @pytest.mark.asyncio
    async def test_mcp_context_mcp_connected_false_when_disabled(
        self, agent_with_mcp, mock_mcp_manager
    ):
        """MCPContext.mcp_connected should be False when manager is disabled."""
        mock_mcp_manager.is_disabled = True

        result = await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        assert result.mcp_connected is False

    @pytest.mark.asyncio
    async def test_mcp_context_has_all_required_fields(self, agent_with_mcp):
        """MCPContext should have all required fields."""
        result = await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.", relevant_memories=None
        )

        # Check all required fields exist
        assert hasattr(result, "messages")
        assert hasattr(result, "tool_schemas")
        assert hasattr(result, "mcp_connected")

        # Check types
        assert isinstance(result.messages, list)
        assert result.tool_schemas is None or isinstance(result.tool_schemas, list)
        assert isinstance(result.mcp_connected, bool)
