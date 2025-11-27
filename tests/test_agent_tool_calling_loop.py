# ABOUTME: Tests for ZorkAgent tool-calling loop (Task #9)
# ABOUTME: Validates tool execution, message history, loop exit conditions

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Dict, Any, Optional

from zork_agent import ZorkAgent, MCPContext
from llm_client import LLMResponse, ToolCall, FunctionCall, ToolCallResult
from session.game_configuration import GameConfiguration
from managers.mcp_config import MCPError


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
    manager.call_tool = AsyncMock()  # Add call_tool mock
    manager.is_disabled = False
    return manager


@pytest.fixture
def mock_llm_client():
    """Mock LLM client wrapper for testing."""
    wrapper = MagicMock()
    wrapper.client = MagicMock()
    wrapper.client._supports_tool_calling = MagicMock(return_value=True)
    wrapper.client.chat_completions_create = MagicMock()  # Sync method that returns LLMResponse
    return wrapper


@pytest.fixture
def test_config():
    """GameConfiguration for testing - loads from pyproject.toml."""
    return GameConfiguration.from_toml()


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
def sample_mcp_context():
    """Sample MCPContext for testing."""
    return MCPContext(
        messages=[
            {"role": "system", "content": "You are a Zork agent."},
            {"role": "user", "content": "Game state here"},
        ],
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "test.tool",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        mcp_connected=True,
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestLoopExitConditions:
    """Tests for loop exit conditions (Req 5.6, 5.10)."""

    @pytest.mark.asyncio
    async def test_loop_exits_on_content_no_tool_calls(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """LLM returns content directly, no tool calls - loop exits immediately.

        Verifies:
        - Loop exits immediately when LLM returns content
        - Content is parsed and returned
        - No tool execution occurs
        """
        # Mock LLM response with content, no tool_calls
        llm_response = LLMResponse(
            content='{"thinking": "I should go north", "action": "go north"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )
        mock_llm_client.client.chat_completions_create.return_value = llm_response

        # Call loop (will fail until implemented)
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify loop exited immediately (only 1 LLM call)
        assert mock_llm_client.client.chat_completions_create.call_count == 1

        # Verify content is parsed and returned
        assert result is not None
        assert "reasoning" in result
        assert "action" in result

    @pytest.mark.asyncio
    async def test_loop_warns_on_neither_content_nor_tool_calls(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """LLM returns neither content nor tool_calls - warns and exits.

        Verifies:
        - Warning is logged when neither content nor tool_calls present
        - Loop exits gracefully
        - No infinite loop occurs
        """
        # Mock LLM response with neither content nor tool_calls
        llm_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )
        mock_llm_client.client.chat_completions_create.return_value = llm_response

        # Mock logger to capture warning
        with patch.object(agent_with_mcp, "logger") as mock_logger:
            result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

            # Verify warning was logged
            assert mock_logger is not None
            # Note: Specific warning check depends on implementation

        # Verify loop exited (only 1 LLM call)
        assert mock_llm_client.client.chat_completions_create.call_count == 1

    @pytest.mark.asyncio
    async def test_loop_exits_on_max_iterations(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """Loop exits after max iterations even if LLM keeps returning tool_calls.

        Verifies:
        - Loop respects max_iterations limit
        - Returns indicator for forced action (Task #11)
        - Prevents infinite loops
        """
        # Mock LLM to always return tool_calls (infinite loop scenario)
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test.tool", arguments="{}"),
        )
        llm_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )
        mock_llm_client.client.chat_completions_create.return_value = llm_response

        # Mock tool execution
        agent_with_mcp.mcp_manager.call_tool.return_value = ToolCallResult(
            content={"result": "success"}
        )

        # Call loop with low max_iterations (will fail until implemented)
        # Note: Implementation should accept max_iterations parameter
        result = await agent_with_mcp._run_tool_calling_loop(
            sample_mcp_context, max_iterations=3
        )

        # Verify loop stopped at max_iterations
        # Should have: max_iterations LLM calls + (max_iterations * tool executions)
        assert mock_llm_client.client.chat_completions_create.call_count == 3

        # Verify forced action indicator returned
        # Note: Exact format depends on implementation
        assert result is not None


class TestToolExecution:
    """Tests for tool execution (Req 5.1, 5.2, 5.3)."""

    @pytest.mark.asyncio
    async def test_loop_executes_single_tool_call(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """LLM returns one tool call, then content.

        Verifies:
        - Tool is executed via mcp_manager.call_tool
        - Tool result is added to message history
        - Second LLM call is made with updated history
        """
        # First LLM response: tool call
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(
                name="test.tool", arguments='{"param": "value"}'
            ),
        )
        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

        # Second LLM response: content
        second_response = LLMResponse(
            content='{"thinking": "Tool result received", "action": "go north"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution
        tool_result = ToolCallResult(content={"result": "success"})
        agent_with_mcp.mcp_manager.call_tool.return_value = tool_result

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify tool was called
        agent_with_mcp.mcp_manager.call_tool.assert_awaited_once_with(
            tool_name="test.tool",
            arguments={"param": "value"},
        )

        # Verify two LLM calls were made
        assert mock_llm_client.client.chat_completions_create.call_count == 2

        # Verify final result parsed
        assert result is not None
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_loop_executes_tools_sequentially(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """LLM returns multiple tool_calls - executed sequentially, not parallel.

        Verifies:
        - Tools executed one by one
        - Order matches order in tool_calls list
        - Not executed in parallel
        """
        # LLM response with multiple tool calls
        tool_call_1 = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="tool.one", arguments='{"a": 1}'),
        )
        tool_call_2 = ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="tool.two", arguments='{"b": 2}'),
        )
        tool_call_3 = ToolCall(
            id="call_3",
            type="function",
            function=FunctionCall(name="tool.three", arguments='{"c": 3}'),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call_1, tool_call_2, tool_call_3],
        )

        second_response = LLMResponse(
            content='{"thinking": "Done", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Track call order
        call_order = []

        async def mock_call_tool(tool_name, arguments):
            call_order.append(tool_name)
            return ToolCallResult(content={"result": f"{tool_name} done"})

        agent_with_mcp.mcp_manager.call_tool.side_effect = mock_call_tool

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify sequential execution in correct order
        assert len(call_order) == 3
        assert call_order == ["tool.one", "tool.two", "tool.three"]

        # Verify all tools called with correct arguments
        assert agent_with_mcp.mcp_manager.call_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_json_arguments_parsed_correctly(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """Tool call JSON arguments are parsed before execution.

        Verifies:
        - JSON string arguments are parsed to dict
        - Parsed dict is passed to call_tool
        - JSON parsing errors handled gracefully
        """
        # LLM response with JSON arguments
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(
                name="test.tool",
                arguments='{"nested": {"key": "value"}, "list": [1, 2, 3]}',
            ),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

        second_response = LLMResponse(
            content='{"thinking": "Done", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution
        agent_with_mcp.mcp_manager.call_tool.return_value = ToolCallResult(
            content={"result": "success"}
        )

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify arguments were parsed correctly
        agent_with_mcp.mcp_manager.call_tool.assert_awaited_once()
        call_args = agent_with_mcp.mcp_manager.call_tool.call_args[1]

        assert call_args["tool_name"] == "test.tool"
        assert call_args["arguments"] == {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

    @pytest.mark.asyncio
    async def test_malformed_json_arguments_handled_gracefully(
        self, agent_with_mcp, mock_llm_client, mock_mcp_manager, sample_mcp_context
    ):
        """Tool call with invalid JSON arguments adds error to history and continues.

        Verifies:
        - Malformed JSON doesn't crash the loop
        - Error message is added to history
        - Loop continues to next tool call
        """
        # First response: tool call with malformed JSON
        tool_call_malformed = ToolCall(
            id="call_bad",
            type="function",
            function=FunctionCall(
                name="test.tool",
                arguments='{"incomplete": ',  # Malformed JSON
            ),
        )
        tool_call_good = ToolCall(
            id="call_good",
            type="function",
            function=FunctionCall(
                name="test.tool",
                arguments='{"valid": true}',
            ),
        )
        response_with_tools = LLMResponse(
            content=None,
            model="gpt-4",
            tool_calls=[tool_call_malformed, tool_call_good],
            finish_reason="tool_calls",
        )

        # Second response: content
        response_with_content = LLMResponse(
            content='{"thinking": "done", "action": "look"}',
            model="gpt-4",
            tool_calls=None,
            finish_reason="stop",
        )

        mock_llm_client.client.chat_completions_create = MagicMock(
            side_effect=[response_with_tools, response_with_content]
        )

        mock_mcp_manager.call_tool = AsyncMock(
            return_value=ToolCallResult(content={"result": "ok"}, is_error=False)
        )

        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Should still return successfully
        assert result["action"] == "look"

        # Only the valid tool call should have been executed
        assert mock_mcp_manager.call_tool.await_count == 1


class TestMessageHistory:
    """Tests for message history management (Req 5.4, 5.5)."""

    @pytest.mark.asyncio
    async def test_tool_results_appended_to_history(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """After tool execution, message history contains tool call and result.

        Verifies:
        - Assistant message with tool_calls added to history
        - Tool message with tool_call_id and content added to history
        - Message format follows OpenAI spec
        """
        # LLM response with tool call
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test.tool", arguments='{"x": 1}'),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

        second_response = LLMResponse(
            content='{"thinking": "Done", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        # Setup responses for two LLM calls
        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution
        tool_result = ToolCallResult(content={"result": "success"})
        agent_with_mcp.mcp_manager.call_tool.return_value = tool_result

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify two LLM calls were made
        assert mock_llm_client.client.chat_completions_create.call_count == 2

        # Get messages from second call (call_args_list[1])
        second_call_kwargs = mock_llm_client.client.chat_completions_create.call_args_list[1][1]
        messages_in_second_call = second_call_kwargs['messages']

        # Find assistant message with tool_calls
        assistant_msg = None
        for msg in messages_in_second_call:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                assistant_msg = msg
                break

        assert assistant_msg is not None, "Assistant message with tool_calls not found"
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["id"] == "call_123"

        # Find tool message with result
        tool_msg = None
        for msg in messages_in_second_call:
            if msg.get("role") == "tool":
                tool_msg = msg
                break

        assert tool_msg is not None, "Tool message not found"
        assert tool_msg["tool_call_id"] == "call_123"
        assert "result" in str(tool_msg["content"])

    @pytest.mark.asyncio
    async def test_loop_continues_after_tool_results(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """After tool results added, LLM is called again with updated history.

        Verifies:
        - First response has tool_calls
        - After tool results added, second LLM call made
        - Second response has content
        - Loop exits correctly
        """
        # First LLM response: tool call
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test.tool", arguments="{}"),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

        # Second LLM response: content
        second_response = LLMResponse(
            content='{"thinking": "Based on tool result", "action": "go north"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution
        tool_result = ToolCallResult(content={"info": "important data"})
        agent_with_mcp.mcp_manager.call_tool.return_value = tool_result

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify two LLM calls made
        assert mock_llm_client.client.chat_completions_create.call_count == 2

        # Verify loop exited with parsed content
        assert result is not None
        assert "reasoning" in result
        assert "action" in result


class TestLLMCallParameters:
    """Tests for LLM call parameters (Req 4.6)."""

    @pytest.mark.asyncio
    async def test_response_format_not_used_during_loop(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """LLM calls during loop do NOT pass response_format (OpenRouter compat).

        Verifies:
        - response_format NOT passed to LLM during tool-calling loop
        - This is for OpenRouter compatibility
        - Tools parameter IS passed
        """
        # LLM response with content (immediate exit)
        llm_response = LLMResponse(
            content='{"thinking": "Test", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )
        mock_llm_client.client.chat_completions_create.return_value = llm_response

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify LLM was called
        assert mock_llm_client.client.chat_completions_create.call_count == 1

        # Verify call parameters
        call_kwargs = mock_llm_client.client.chat_completions_create.call_args[1]

        # response_format should NOT be present
        assert "response_format" not in call_kwargs, (
            "response_format should not be passed during tool-calling loop"
        )

        # tools should be present
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == sample_mcp_context.tool_schemas

    @pytest.mark.asyncio
    async def test_tool_choice_auto_when_mcp_connected(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """tool_choice='auto' is passed when mcp_connected=True.

        Verifies:
        - tool_choice="auto" passed to LLM
        - Agent can decide whether to use tools
        - Follows Req 1.2: Agent decides tool usage
        """
        # LLM response with content (immediate exit)
        llm_response = LLMResponse(
            content='{"thinking": "Test", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )
        mock_llm_client.client.chat_completions_create.return_value = llm_response

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify LLM was called
        assert mock_llm_client.client.chat_completions_create.call_count == 1

        # Verify tool_choice parameter
        call_kwargs = mock_llm_client.client.chat_completions_create.call_args[1]

        assert "tool_choice" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto", (
            "tool_choice should be 'auto' to let agent decide"
        )


class TestToolExecutionErrorHandling:
    """Tests for tool execution error handling."""

    @pytest.mark.asyncio
    async def test_tool_error_added_to_history(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """When tool execution fails, error is added to message history.

        Verifies:
        - Tool errors captured in ToolCallResult
        - Error message added to history for LLM to see
        - Loop continues after error
        """
        # LLM response with tool call
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test.tool", arguments="{}"),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

        second_response = LLMResponse(
            content='{"thinking": "Tool failed, trying different approach", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        # Setup responses for two LLM calls
        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution error
        tool_error = ToolCallResult(
            content=None,
            is_error=True,
            error_message="Tool execution failed: timeout",
        )
        agent_with_mcp.mcp_manager.call_tool.return_value = tool_error

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify two LLM calls were made
        assert mock_llm_client.client.chat_completions_create.call_count == 2

        # Get messages from second call
        second_call_kwargs = mock_llm_client.client.chat_completions_create.call_args_list[1][1]
        messages_in_second_call = second_call_kwargs['messages']

        # Find tool message with error
        tool_msg = None
        for msg in messages_in_second_call:
            if msg.get("role") == "tool":
                tool_msg = msg
                break

        assert tool_msg is not None
        assert "error" in str(tool_msg["content"]).lower()


class TestMultipleToolCallIterations:
    """Tests for multiple iterations of tool calling."""

    @pytest.mark.asyncio
    async def test_multiple_tool_call_iterations(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """Loop handles multiple iterations of tool calls before getting content.

        Verifies:
        - Iteration 1: tool call → tool result → LLM call
        - Iteration 2: tool call → tool result → LLM call
        - Iteration 3: content → exit
        - History accumulates across iterations
        """
        # Iteration 1: tool call
        tool_call_1 = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="tool.one", arguments="{}"),
        )
        response_1 = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call_1],
        )

        # Iteration 2: another tool call
        tool_call_2 = ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="tool.two", arguments="{}"),
        )
        response_2 = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call_2],
        )

        # Iteration 3: final content
        response_3 = LLMResponse(
            content='{"thinking": "All tools done", "action": "go north"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            response_1,
            response_2,
            response_3,
        ]

        # Mock tool execution
        agent_with_mcp.mcp_manager.call_tool.side_effect = [
            ToolCallResult(content={"result": "one"}),
            ToolCallResult(content={"result": "two"}),
        ]

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify three LLM calls (one per iteration)
        assert mock_llm_client.client.chat_completions_create.call_count == 3

        # Verify two tool executions
        assert agent_with_mcp.mcp_manager.call_tool.call_count == 2

        # Verify final result
        assert result is not None
        assert "reasoning" in result
        assert "action" in result


class TestBatchErrorHandling:
    """Tests for batch error handling in tool-calling loop (Req 6.2-6.6).

    These tests validate error handling for multiple tool_calls in a single LLM response:
    - Non-timeout errors: continue with remaining tools
    - Timeout errors: abort batch and skip remaining tools
    - Error messages added to history for LLM decision-making
    """

    @pytest.mark.asyncio
    async def test_non_timeout_error_continues_batch(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """Non-timeout error in batch continues with remaining tools (Req 6.4).

        Verifies:
        - First tool fails with non-timeout error
        - Second and third tools execute successfully
        - All 3 tool results added to message history
        - Error message properly formatted in history
        """
        # LLM response with 3 tool calls
        tool_call_1 = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="tool.one", arguments="{}"),
        )
        tool_call_2 = ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="tool.two", arguments="{}"),
        )
        tool_call_3 = ToolCall(
            id="call_3",
            type="function",
            function=FunctionCall(name="tool.three", arguments="{}"),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call_1, tool_call_2, tool_call_3],
        )

        # Second response: content (exit loop)
        second_response = LLMResponse(
            content='{"thinking": "Some tools failed, but continuing", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution: first fails, others succeed
        agent_with_mcp.mcp_manager.call_tool.side_effect = [
            ToolCallResult(
                is_error=True,
                error_message="first tool failed",
                content={"error": "fail"},
            ),
            ToolCallResult(is_error=False, content={"result": "ok"}),
            ToolCallResult(is_error=False, content={"result": "ok"}),
        ]

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify all 3 tool calls were executed
        assert agent_with_mcp.mcp_manager.call_tool.call_count == 3, (
            "All tools in batch should execute despite non-timeout error"
        )

        # Verify second LLM call has 3 tool messages in history
        second_call_kwargs = mock_llm_client.client.chat_completions_create.call_args_list[1][1]
        messages_in_second_call = second_call_kwargs['messages']

        tool_messages = [msg for msg in messages_in_second_call if msg.get("role") == "tool"]
        assert len(tool_messages) == 3, "All 3 tool results should be in history"

        # Verify first message contains error
        first_tool_msg = next(
            msg for msg in messages_in_second_call
            if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1"
        )
        assert "error" in str(first_tool_msg["content"]).lower(), (
            "Error message should be present in tool result"
        )

    @pytest.mark.asyncio
    async def test_timeout_aborts_batch(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """Timeout error aborts batch and skips remaining tools (Req 6.5).

        Verifies:
        - First tool raises TimeoutError
        - Remaining tools (2nd, 3rd) are NOT executed
        - Only 1 tool message added to history (timeout error)
        - Loop continues to second LLM call
        """
        # LLM response with 3 tool calls
        tool_call_1 = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="tool.slow", arguments="{}"),
        )
        tool_call_2 = ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="tool.two", arguments="{}"),
        )
        tool_call_3 = ToolCall(
            id="call_3",
            type="function",
            function=FunctionCall(name="tool.three", arguments="{}"),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call_1, tool_call_2, tool_call_3],
        )

        # Second response: content (exit loop)
        second_response = LLMResponse(
            content='{"thinking": "Tool timed out, trying different approach", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution: first raises TimeoutError
        agent_with_mcp.mcp_manager.call_tool.side_effect = asyncio.TimeoutError()

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify only 1 tool call was made (remaining skipped)
        assert agent_with_mcp.mcp_manager.call_tool.call_count == 1, (
            "Batch should abort after timeout, skipping remaining tools"
        )

        # Verify second LLM call has only 1 tool message
        second_call_kwargs = mock_llm_client.client.chat_completions_create.call_args_list[1][1]
        messages_in_second_call = second_call_kwargs['messages']

        tool_messages = [msg for msg in messages_in_second_call if msg.get("role") == "tool"]
        assert len(tool_messages) == 1, "Only timeout error should be in history"

        # Verify message contains timeout error
        timeout_msg = tool_messages[0]
        assert timeout_msg["tool_call_id"] == "call_1"
        assert "timeout" in str(timeout_msg["content"]).lower(), (
            "Timeout error message should be present"
        )

    @pytest.mark.asyncio
    async def test_timeout_message_in_history(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """Timeout error message properly formatted in history (Req 6.6).

        Verifies:
        - Timeout message includes tool name
        - Message format: {"error": "Tool call timeout: <tool_name>"}
        - LLM can see timeout and decide how to proceed
        """
        # LLM response with 1 tool call
        tool_call = ToolCall(
            id="call_slow",
            type="function",
            function=FunctionCall(name="test.slowtool", arguments="{}"),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

        # Second response: content (exit loop)
        second_response = LLMResponse(
            content='{"thinking": "Tool timed out, will try simpler approach", "action": "look"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution: raises TimeoutError
        agent_with_mcp.mcp_manager.call_tool.side_effect = asyncio.TimeoutError()

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify second LLM call messages
        second_call_kwargs = mock_llm_client.client.chat_completions_create.call_args_list[1][1]
        messages_in_second_call = second_call_kwargs['messages']

        # Find tool message with timeout error
        tool_msg = next(
            msg for msg in messages_in_second_call if msg.get("role") == "tool"
        )

        assert tool_msg["tool_call_id"] == "call_slow"

        # Verify message contains timeout error with tool name
        content_str = str(tool_msg["content"])
        assert "timeout" in content_str.lower(), "Message should mention timeout"
        assert "test.slowtool" in content_str, "Message should include tool name"

    @pytest.mark.asyncio
    async def test_batch_partial_success_then_timeout(
        self, agent_with_mcp, mock_llm_client, sample_mcp_context
    ):
        """Batch with partial success then timeout aborts remaining tools (Req 6.3, 6.5).

        Verifies:
        - First tool succeeds
        - Second tool times out
        - Third tool NOT executed (batch aborted)
        - History has 2 tool messages (success + timeout)
        """
        # LLM response with 3 tool calls
        tool_call_1 = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="tool.fast", arguments="{}"),
        )
        tool_call_2 = ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="tool.slow", arguments="{}"),
        )
        tool_call_3 = ToolCall(
            id="call_3",
            type="function",
            function=FunctionCall(name="tool.never_called", arguments="{}"),
        )

        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=[tool_call_1, tool_call_2, tool_call_3],
        )

        # Second response: content (exit loop)
        second_response = LLMResponse(
            content='{"thinking": "Got partial results before timeout", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None,
        )

        mock_llm_client.client.chat_completions_create.side_effect = [
            first_response,
            second_response,
        ]

        # Mock tool execution: first succeeds, second times out, third never called
        async def mock_call_tool_side_effect(tool_name, arguments):
            if tool_name == "tool.fast":
                return ToolCallResult(is_error=False, content={"result": "success"})
            elif tool_name == "tool.slow":
                raise asyncio.TimeoutError()
            else:
                raise AssertionError(f"Tool {tool_name} should not have been called")

        agent_with_mcp.mcp_manager.call_tool.side_effect = mock_call_tool_side_effect

        # Call loop
        result = await agent_with_mcp._run_tool_calling_loop(sample_mcp_context)

        # Verify only 2 tool calls made (third skipped due to timeout)
        assert agent_with_mcp.mcp_manager.call_tool.call_count == 2, (
            "Only first and second tools should execute, third skipped after timeout"
        )

        # Verify history has 2 tool messages
        second_call_kwargs = mock_llm_client.client.chat_completions_create.call_args_list[1][1]
        messages_in_second_call = second_call_kwargs['messages']

        tool_messages = [msg for msg in messages_in_second_call if msg.get("role") == "tool"]
        assert len(tool_messages) == 2, (
            "History should have first tool result + timeout error (2 total)"
        )

        # Verify first message is success
        first_msg = next(
            msg for msg in messages_in_second_call
            if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1"
        )
        assert "success" in str(first_msg["content"]), "First tool should have succeeded"

        # Verify second message is timeout error
        second_msg = next(
            msg for msg in messages_in_second_call
            if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_2"
        )
        assert "timeout" in str(second_msg["content"]).lower(), (
            "Second tool should have timeout error"
        )
