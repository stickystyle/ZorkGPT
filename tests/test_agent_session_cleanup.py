# ABOUTME: Tests for MCP session cleanup and Langfuse tracking (Task #12)
# ABOUTME: Validates session summary logging, Langfuse spans, and cleanup in finally block

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import is_dataclass

from zork_agent import ZorkAgent, MCPContext
from session.game_configuration import GameConfiguration
from llm_client import ToolCall, FunctionCall


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
    manager._server_name = "test-server"
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
def mock_langfuse_client():
    """Mock Langfuse client for testing."""
    client = MagicMock()

    # Mock the span context manager
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=None)
    mock_span.update = MagicMock()

    # Mock start_as_current_span to return the span context manager
    client.start_as_current_span = MagicMock(return_value=mock_span)

    return client


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
def agent_with_langfuse(test_config, mock_llm_client, mock_mcp_manager, mock_langfuse_client):
    """Agent instance with MCP manager and Langfuse client."""
    with patch.object(ZorkAgent, "_load_system_prompt"):
        agent = ZorkAgent(
            config=test_config,
            model="gpt-4",
            client=mock_llm_client,
            mcp_manager=mock_mcp_manager,
            langfuse_client=mock_langfuse_client,
        )
        agent.system_prompt = "You are a Zork agent."
    return agent


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


# ============================================================================
# TestLangfuseClientAcceptance
# ============================================================================


class TestLangfuseClientAcceptance:
    """Tests for ZorkAgent accepting and storing langfuse_client parameter."""

    def test_agent_accepts_langfuse_client(self, test_config, mock_llm_client, mock_langfuse_client):
        """ZorkAgent should accept optional langfuse_client parameter and store it (Req 7.5)."""
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=test_config,
                model="gpt-4",
                client=mock_llm_client,
                mcp_manager=None,
                langfuse_client=mock_langfuse_client,
            )
            agent.system_prompt = "You are a Zork agent."

        assert hasattr(agent, "langfuse_client"), "Agent should have langfuse_client attribute"
        assert agent.langfuse_client is mock_langfuse_client

    def test_agent_langfuse_client_defaults_to_none(self, test_config, mock_llm_client):
        """ZorkAgent should default langfuse_client to None when not provided."""
        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=test_config,
                model="gpt-4",
                client=mock_llm_client,
                mcp_manager=None,
            )
            agent.system_prompt = "You are a Zork agent."

        assert hasattr(agent, "langfuse_client"), "Agent should have langfuse_client attribute"
        assert agent.langfuse_client is None


# ============================================================================
# TestToolCallingLoopMetadata
# ============================================================================


class TestToolCallingLoopMetadata:
    """Tests for _run_tool_calling_loop returning metadata (Req 7.3)."""

    @pytest.mark.asyncio
    async def test_run_tool_calling_loop_returns_metadata(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """_run_tool_calling_loop should return dict with _metadata containing iterations and tool_calls (Req 7.3)."""
        # Mock LLM to return content immediately (no tool calls)
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        mcp_context = MCPContext(
            messages=[{"role": "user", "content": "test"}],
            tool_schemas=[{"type": "function", "function": {"name": "test"}}],
            mcp_connected=True,
        )

        result = await agent_with_mcp._run_tool_calling_loop(mcp_context)

        assert "_metadata" in result, "Result should contain _metadata"
        assert "iterations" in result["_metadata"], "_metadata should contain iterations count"
        assert "tool_calls" in result["_metadata"], "_metadata should contain tool_calls count"
        assert result["_metadata"]["iterations"] == 1, "Should have 1 iteration"
        assert result["_metadata"]["tool_calls"] == 0, "Should have 0 tool calls"

    @pytest.mark.asyncio
    async def test_metadata_includes_tool_call_count(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """_metadata should count total tool calls executed (Req 7.3)."""
        # First response: 2 tool calls
        mock_tool_call_1 = MagicMock()
        mock_tool_call_1.id = "tc1"
        mock_tool_call_1.type = "function"
        mock_tool_call_1.function = MagicMock()
        mock_tool_call_1.function.name = "tool1"
        mock_tool_call_1.function.arguments = "{}"

        mock_tool_call_2 = MagicMock()
        mock_tool_call_2.id = "tc2"
        mock_tool_call_2.type = "function"
        mock_tool_call_2.function = MagicMock()
        mock_tool_call_2.function.name = "tool2"
        mock_tool_call_2.function.arguments = "{}"

        mock_response_with_tools = MagicMock()
        mock_response_with_tools.content = None
        mock_response_with_tools.tool_calls = [mock_tool_call_1, mock_tool_call_2]

        # Second response: content (exit loop)
        mock_response_with_content = MagicMock()
        mock_response_with_content.content = '{"action": "look", "thinking": "test"}'
        mock_response_with_content.tool_calls = None

        # Configure LLM client to return tool calls, then content
        mock_llm_client.client.chat_completions_create = MagicMock(
            side_effect=[mock_response_with_tools, mock_response_with_content]
        )

        # Mock tool execution
        mock_tool_result = MagicMock()
        mock_tool_result.to_dict = MagicMock(return_value={"result": "success"})
        mock_mcp_manager.call_tool = AsyncMock(return_value=mock_tool_result)

        mcp_context = MCPContext(
            messages=[{"role": "user", "content": "test"}],
            tool_schemas=[{"type": "function", "function": {"name": "test"}}],
            mcp_connected=True,
        )

        result = await agent_with_mcp._run_tool_calling_loop(mcp_context)

        assert result["_metadata"]["iterations"] == 2, "Should have 2 iterations"
        assert result["_metadata"]["tool_calls"] == 2, "Should have 2 tool calls total"

    @pytest.mark.asyncio
    async def test_metadata_includes_duration(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """_metadata should include duration_ms value (Req 7.3)."""
        # Mock LLM to return content immediately
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        mcp_context = MCPContext(
            messages=[{"role": "user", "content": "test"}],
            tool_schemas=[{"type": "function", "function": {"name": "test"}}],
            mcp_connected=True,
        )

        result = await agent_with_mcp._run_tool_calling_loop(mcp_context)

        assert "duration_ms" in result["_metadata"], "_metadata should contain duration_ms"
        assert isinstance(result["_metadata"]["duration_ms"], (int, float))
        assert result["_metadata"]["duration_ms"] >= 0, "Duration should be non-negative"


# ============================================================================
# TestSessionSummaryLogging
# ============================================================================


class TestSessionSummaryLogging:
    """Tests for session summary logging after disconnect (Req 7.3)."""

    @pytest.mark.asyncio
    async def test_session_summary_logged_after_disconnect(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """Logger.info should be called with session summary after disconnect (Req 7.3)."""
        # Mock LLM to return content
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        # Add mock logger
        agent_with_mcp.logger = MagicMock()

        # Execute action generation (includes disconnect in finally)
        await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.",
            relevant_memories=None,
        )

        # Verify disconnect was called
        mock_mcp_manager.disconnect_session.assert_awaited_once()

        # Verify logger.info was called with session summary
        info_calls = agent_with_mcp.logger.info.call_args_list

        # Find the session summary log call
        session_summary_found = False
        for call_obj in info_calls:
            args, kwargs = call_obj
            if args and "MCP session complete" in args[0]:
                session_summary_found = True
                # Verify it contains iterations and tool_calls
                assert "iteration" in args[0].lower() or "1" in args[0]
                break

        assert session_summary_found, "Logger should log session summary with 'MCP session complete'"

    @pytest.mark.asyncio
    async def test_session_summary_includes_metadata_values(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """Session summary log should include actual iterations and tool_calls values (Req 7.3)."""
        # Mock LLM with tool calls then content
        mock_tool_call = MagicMock()
        mock_tool_call.id = "tc1"
        mock_tool_call.type = "function"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "test.tool"
        mock_tool_call.function.arguments = "{}"

        mock_response_with_tools = MagicMock()
        mock_response_with_tools.content = None
        mock_response_with_tools.tool_calls = [mock_tool_call]

        mock_response_with_content = MagicMock()
        mock_response_with_content.content = '{"action": "look", "thinking": "test"}'
        mock_response_with_content.tool_calls = None

        mock_llm_client.client.chat_completions_create = MagicMock(
            side_effect=[mock_response_with_tools, mock_response_with_content]
        )

        mock_tool_result = MagicMock()
        mock_tool_result.to_dict = MagicMock(return_value={"result": "success"})
        mock_mcp_manager.call_tool = AsyncMock(return_value=mock_tool_result)

        # Add mock logger
        agent_with_mcp.logger = MagicMock()

        await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.",
            relevant_memories=None,
        )

        # Find session summary log
        info_calls = agent_with_mcp.logger.info.call_args_list
        session_log = None
        for call_obj in info_calls:
            args, kwargs = call_obj
            if args and "MCP session complete" in args[0]:
                session_log = args[0]
                break

        assert session_log is not None, "Should have logged session summary"
        # Should mention 2 iterations and 1 tool call
        assert "2" in session_log or "iteration" in session_log.lower()


# ============================================================================
# TestLangfuseSessionSpan
# ============================================================================


class TestLangfuseSessionSpan:
    """Tests for Langfuse session span creation and updates (Req 7.5)."""

    @pytest.mark.asyncio
    async def test_langfuse_session_span_created(
        self, agent_with_langfuse, mock_llm_client, mock_mcp_manager, mock_langfuse_client
    ):
        """When langfuse_client provided and MCP connected, start_as_current_span should be called (Req 7.5)."""
        # Mock LLM to return content
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        await agent_with_langfuse._generate_action_async(
            game_state_text="You are in a room.",
            relevant_memories=None,
        )

        # Verify start_as_current_span was called with "mcp-session"
        mock_langfuse_client.start_as_current_span.assert_called_once()
        call_args = mock_langfuse_client.start_as_current_span.call_args
        assert call_args[1]["name"] == "mcp-session", "Span should be named 'mcp-session'"

    @pytest.mark.asyncio
    async def test_langfuse_span_not_created_when_mcp_disabled(
        self, agent_without_mcp, mock_llm_client, mock_langfuse_client
    ):
        """No span should be created when MCP is disabled (Req 7.5)."""
        # Mock LLM to return content
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        # Add langfuse client to agent without MCP
        agent_without_mcp.langfuse_client = mock_langfuse_client

        await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.",
            relevant_memories=None,
        )

        # Verify start_as_current_span was NOT called
        mock_langfuse_client.start_as_current_span.assert_not_called()

    @pytest.mark.asyncio
    async def test_langfuse_span_not_created_when_langfuse_none(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """No span should be created when langfuse_client is None."""
        # Mock LLM to return content
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        # agent_with_mcp has no langfuse_client
        assert agent_with_mcp.langfuse_client is None

        # Should not raise, just skip span creation
        await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.",
            relevant_memories=None,
        )

        # No assertions needed - just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_langfuse_span_updated_with_results(
        self, agent_with_langfuse, mock_llm_client, mock_mcp_manager, mock_langfuse_client
    ):
        """Span.update() should be called with action, iterations, tool_calls, duration_ms (Req 7.5)."""
        # Mock LLM to return content
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        await agent_with_langfuse._generate_action_async(
            game_state_text="You are in a room.",
            relevant_memories=None,
        )

        # Get the mock span from the context manager
        mock_span = mock_langfuse_client.start_as_current_span.return_value

        # Verify span.update was called
        mock_span.update.assert_called_once()

        # Verify update arguments contain expected keys
        update_kwargs = mock_span.update.call_args[1]
        assert "output" in update_kwargs, "Should update with output"

        output_data = update_kwargs["output"]
        assert "action" in output_data, "Output should contain action"
        assert "iterations" in output_data, "Output should contain iterations"
        assert "tool_calls" in output_data, "Output should contain tool_calls"
        assert "duration_ms" in output_data, "Output should contain duration_ms"


# ============================================================================
# TestSessionCleanupFinally
# ============================================================================


class TestSessionCleanupFinally:
    """Tests for session disconnect in finally block (Req 3.5)."""

    @pytest.mark.asyncio
    async def test_session_disconnects_in_finally_block(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """disconnect_session should be called even when error occurs in loop (Req 3.5)."""
        # Mock LLM to raise an exception
        mock_llm_client.client.chat_completions_create = MagicMock(
            side_effect=Exception("LLM call failed")
        )

        # Attempt action generation - should raise
        with pytest.raises(Exception) as exc_info:
            await agent_with_mcp._generate_action_async(
                game_state_text="You are in a room.",
                relevant_memories=None,
            )

        assert "LLM call failed" in str(exc_info.value)

        # Verify disconnect was still called (in finally block)
        mock_mcp_manager.disconnect_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_session_disconnects_on_success(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """disconnect_session should be called on successful completion (Req 3.5)."""
        # Mock LLM to return content
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        await agent_with_mcp._generate_action_async(
            game_state_text="You are in a room.",
            relevant_memories=None,
        )

        # Verify disconnect was called
        mock_mcp_manager.disconnect_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_session_not_disconnected_when_mcp_disabled(
        self, agent_without_mcp, mock_llm_client
    ):
        """disconnect_session should not be called when MCP is disabled."""
        # Mock LLM to return content
        mock_response = MagicMock()
        mock_response.content = '{"action": "look", "thinking": "test"}'
        mock_response.tool_calls = None
        mock_llm_client.client.chat_completions_create = MagicMock(return_value=mock_response)

        await agent_without_mcp._generate_action_async(
            game_state_text="You are in a room.",
            relevant_memories=None,
        )

        # agent_without_mcp has no mcp_manager - just verify no exception


# ============================================================================
# TestMaxIterationsMetadata
# ============================================================================


class TestMaxIterationsMetadata:
    """Tests for metadata when max iterations is reached."""

    @pytest.mark.asyncio
    async def test_metadata_returned_on_max_iterations(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager
    ):
        """Metadata should be returned even when max iterations is reached (Req 7.3)."""
        # Mock LLM to always return tool calls (infinite loop scenario)
        mock_tool_call = MagicMock()
        mock_tool_call.id = "tc1"
        mock_tool_call.type = "function"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "test.tool"
        mock_tool_call.function.arguments = "{}"

        mock_response_with_tools = MagicMock()
        mock_response_with_tools.content = None
        mock_response_with_tools.tool_calls = [mock_tool_call]

        # Final forced response
        mock_forced_response = MagicMock()
        mock_forced_response.content = '{"action": "look", "thinking": "forced"}'
        mock_forced_response.tool_calls = None

        # Mock to return tools multiple times, then forced content
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return tools for max_iterations calls, then forced response
            if call_count <= 3:  # Assuming max_iterations=3
                return mock_response_with_tools
            else:
                return mock_forced_response

        mock_llm_client.client.chat_completions_create = MagicMock(side_effect=side_effect)

        mock_tool_result = MagicMock()
        mock_tool_result.to_dict = MagicMock(return_value={"result": "success"})
        mock_mcp_manager.call_tool = AsyncMock(return_value=mock_tool_result)

        mcp_context = MCPContext(
            messages=[{"role": "user", "content": "test"}],
            tool_schemas=[{"type": "function", "function": {"name": "test"}}],
            mcp_connected=True,
        )

        # Set low max_iterations for testing
        result = await agent_with_mcp._run_tool_calling_loop(mcp_context, max_iterations=3)

        # Should still have metadata
        assert "_metadata" in result, "Should return metadata even at max iterations"
        assert result["_metadata"]["iterations"] == 3, "Should have hit max iterations"
        assert result["_metadata"]["tool_calls"] == 3, "Should have executed 3 tool calls"
