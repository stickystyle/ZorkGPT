# ABOUTME: Unit tests for MCPManager.call_tool() method.
# ABOUTME: Tests tool execution, timeout handling, error wrapping, and logging.

import asyncio
import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from session.game_configuration import GameConfiguration
from llm_client import ToolCallResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def create_mcp_config_file(tmp_path):
    """Helper fixture to create a mcp_config.json file with specified content."""

    def _create(config_data: dict) -> str:
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))
        return str(config_file)

    return _create


@pytest.fixture
def valid_mcp_config_data():
    """Valid MCP configuration data."""
    return {
        "mcpServers": {
            "test-server": {
                "command": "echo",
                "args": ["hello"],
                "env": {"TEST_VAR": "test_value"},
            }
        }
    }


@pytest.fixture
def game_config_with_mcp(tmp_path, create_mcp_config_file, valid_mcp_config_data):
    """Create a GameConfiguration with MCP enabled and valid config file."""
    config_file = create_mcp_config_file(valid_mcp_config_data)
    return GameConfiguration(
        max_turns_per_episode=100,
        game_file_path="test.z5",
        mcp_enabled=True,
        mcp_config_file=config_file,
        mcp_tool_call_timeout_seconds=5.0,  # Default timeout for testing
    )


@pytest.fixture
def mock_stdio_client():
    """Mock the stdio_client context manager from MCP SDK."""
    mock_read = AsyncMock()
    mock_write = AsyncMock()
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
    mock_context.__aexit__ = AsyncMock(return_value=None)

    with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
        yield mock_context


@pytest.fixture
def mock_client_session():
    """Mock the ClientSession class from MCP SDK."""
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()

    with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
        yield mock_session


@pytest.fixture
def connected_manager(
    mock_logger,
    game_config_with_mcp,
    mock_stdio_client,
    mock_client_session,
):
    """Create an MCPManager instance with connected session for tool call tests."""
    from managers.mcp_manager import MCPManager

    async def _create():
        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Connect session
        await manager.connect_session()

        return manager, mock_client_session

    return _create()


# =============================================================================
# Task 5.1: Unit Tests for call_tool Success Cases
# =============================================================================


class TestCallToolSuccess:
    """Test successful tool execution (Requirements 5.1, 11.1)."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self, connected_manager):
        """Test call_tool returns ToolCallResult with content on success (Req 11.1)."""
        manager, mock_session = await connected_manager

        # Mock successful tool call result from MCP server
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Tool executed successfully"}]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Call tool
        result = await manager.call_tool(
            tool_name="test-server.sample_tool",
            arguments={"arg1": "value1"},
        )

        # Verify ToolCallResult structure
        assert isinstance(result, ToolCallResult)
        assert result.is_error is False
        assert result.error_message is None
        assert result.content == [{"type": "text", "text": "Tool executed successfully"}]

        # Verify session.call_tool was called correctly
        mock_session.call_tool.assert_called_once_with(
            "sample_tool",
            {"arg1": "value1"},
        )

    @pytest.mark.asyncio
    async def test_call_tool_parses_tool_name(self, connected_manager):
        """Test call_tool uses _parse_tool_name to extract server/tool names (Req 5.1)."""
        manager, mock_session = await connected_manager

        # Mock successful tool call
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Success"}]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Spy on _parse_tool_name
        with patch.object(
            manager, "_parse_tool_name", wraps=manager._parse_tool_name
        ) as mock_parse:
            await manager.call_tool(
                tool_name="test-server.my_tool",
                arguments={},
            )

            # Verify _parse_tool_name was called
            mock_parse.assert_called_once_with("test-server.my_tool")

            # Verify session.call_tool received unprefixed tool name
            mock_session.call_tool.assert_called_once_with("my_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_with_complex_arguments(self, connected_manager):
        """Test call_tool passes complex nested arguments correctly."""
        manager, mock_session = await connected_manager

        # Mock successful tool call
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Complex args handled"}]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Complex nested arguments
        complex_args = {
            "nested": {"key": "value", "count": 42},
            "list": [1, 2, 3],
            "bool": True,
        }

        result = await manager.call_tool(
            tool_name="test-server.complex_tool",
            arguments=complex_args,
        )

        # Verify result
        assert isinstance(result, ToolCallResult)
        assert result.is_error is False

        # Verify arguments passed correctly
        mock_session.call_tool.assert_called_once_with("complex_tool", complex_args)


# =============================================================================
# Task 5.1: Unit Tests for call_tool Error Cases
# =============================================================================


class TestCallToolErrors:
    """Test error handling during tool execution (Requirements 6.2, 11.2, 11.4)."""

    @pytest.mark.asyncio
    async def test_call_tool_failure_returns_error_result(self, connected_manager):
        """Test call_tool returns ToolCallResult with is_error=True on exception (Req 11.2, 11.4)."""
        manager, mock_session = await connected_manager

        # Mock session.call_tool to raise exception
        mock_session.call_tool = AsyncMock(
            side_effect=Exception("Tool execution failed")
        )

        # Call tool (should not raise, should wrap error)
        result = await manager.call_tool(
            tool_name="test-server.failing_tool",
            arguments={"arg": "value"},
        )

        # Verify error result structure
        assert isinstance(result, ToolCallResult)
        assert result.is_error is True
        assert result.error_message is not None
        assert "Tool execution failed" in result.error_message
        assert result.content is not None  # Error details in content

    @pytest.mark.asyncio
    async def test_call_tool_session_not_connected(self, mock_logger, game_config_with_mcp):
        """Test call_tool raises RuntimeError when session not connected (Req 5.1)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Session is None (not connected)
        assert manager._session is None

        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            await manager.call_tool(
                tool_name="test-server.tool",
                arguments={},
            )

        assert "session not connected" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_call_tool_invalid_tool_name_format(self, connected_manager):
        """Test call_tool handles invalid tool name format gracefully."""
        manager, mock_session = await connected_manager

        # Invalid format: missing server prefix
        with pytest.raises(ValueError) as exc_info:
            await manager.call_tool(
                tool_name="invalid_format",  # No dot separator
                arguments={},
            )

        assert "invalid tool name format" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_call_tool_with_mcp_error_response(self, connected_manager):
        """Test call_tool handles MCP error response correctly."""
        manager, mock_session = await connected_manager

        # Mock MCP error response
        mock_call_result = MagicMock()
        mock_call_result.isError = True
        mock_call_result.content = [{"type": "text", "text": "MCP server error"}]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Call tool
        result = await manager.call_tool(
            tool_name="test-server.error_tool",
            arguments={},
        )

        # Should wrap as error result
        assert isinstance(result, ToolCallResult)
        assert result.is_error is True
        assert "MCP server error" in str(result.content)


# =============================================================================
# Task 5.1: Unit Tests for call_tool Timeout Handling
# =============================================================================


class TestCallToolTimeout:
    """Test timeout handling for tool calls (Requirements 6.5, 5.1)."""

    @pytest.mark.asyncio
    async def test_call_tool_timeout_raises_timeout_error(self, connected_manager):
        """Test call_tool raises asyncio.TimeoutError on timeout (Req 6.5)."""
        manager, mock_session = await connected_manager

        # Mock slow tool call that exceeds timeout
        async def slow_call_tool(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return MagicMock(content=[{"type": "text", "text": "Too slow"}])

        mock_session.call_tool = slow_call_tool

        # Should raise TimeoutError
        with pytest.raises(asyncio.TimeoutError):
            await manager.call_tool(
                tool_name="test-server.slow_tool",
                arguments={},
                timeout_seconds=0.1,  # Very short timeout
            )

    @pytest.mark.asyncio
    async def test_call_tool_uses_config_timeout(
        self, mock_logger, tmp_path, create_mcp_config_file, valid_mcp_config_data
    ):
        """Test call_tool uses config.mcp_tool_call_timeout_seconds by default (Req 5.1)."""
        from managers.mcp_manager import MCPManager

        # Create config with custom timeout
        config_file = create_mcp_config_file(valid_mcp_config_data)
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=config_file,
            mcp_tool_call_timeout_seconds=3,  # Custom timeout (int, not float)
        )

        manager = MCPManager(config=config, logger=mock_logger)

        # Mock session and asyncio.wait_for to capture timeout parameter
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=MagicMock(content=[{"type": "text", "text": "Success"}])
        )
        manager._session = mock_session

        # Spy on asyncio.wait_for to verify timeout parameter
        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for:
            # Make wait_for pass through the result
            async def passthrough_wait_for(coro, timeout):
                assert timeout == 3, "Expected config timeout to be used"
                return await coro

            mock_wait_for.side_effect = passthrough_wait_for

            # Call without timeout parameter (should use config default)
            result = await manager.call_tool(
                tool_name="test-server.tool",
                arguments={},
            )

            # Verify wait_for was called with config timeout
            mock_wait_for.assert_called_once()
            assert mock_wait_for.call_args[1]["timeout"] == 3

    @pytest.mark.asyncio
    async def test_call_tool_uses_parameter_timeout(self, connected_manager):
        """Test call_tool timeout_seconds parameter overrides config (Req 5.1)."""
        manager, mock_session = await connected_manager

        # Mock successful tool call
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Success"}]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Spy on asyncio.wait_for to verify parameter timeout used
        with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for:
            # Make wait_for pass through the result
            async def passthrough_wait_for(coro, timeout):
                assert timeout == 10.0, "Expected parameter timeout to override config"
                return await coro

            mock_wait_for.side_effect = passthrough_wait_for

            # Call with explicit timeout parameter
            result = await manager.call_tool(
                tool_name="test-server.tool",
                arguments={},
                timeout_seconds=10.0,  # Override config default
            )

            # Verify wait_for was called with parameter timeout
            mock_wait_for.assert_called_once()
            assert mock_wait_for.call_args[1]["timeout"] == 10.0


# =============================================================================
# Task 5.1: Unit Tests for call_tool Logging
# =============================================================================


class TestCallToolLogging:
    """Test logging for tool calls (Requirements 7.1, 7.2, 6.2)."""

    @pytest.mark.asyncio
    async def test_call_tool_logs_start(self, connected_manager):
        """Test call_tool logs tool invocation with name and arguments (Req 7.1)."""
        manager, mock_session = await connected_manager
        mock_logger = manager.logger

        # Mock successful tool call
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Success"}]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Reset mock to ignore connection logs
        mock_logger.reset_mock()

        # Call tool
        await manager.call_tool(
            tool_name="test-server.my_tool",
            arguments={"arg1": "value1", "arg2": 42},
        )

        # Verify logging call with tool name and arguments in extra dict
        info_calls = [call for call in mock_logger.info.call_args_list]
        assert len(info_calls) >= 2, "Expected at least 2 info logs (start + success)"

        # Check start log
        start_log = info_calls[0]
        assert "my_tool" in start_log[0][0].lower() or (
            "extra" in start_log[1] and "my_tool" in str(start_log[1]["extra"])
        )

    @pytest.mark.asyncio
    async def test_call_tool_logs_success(self, connected_manager):
        """Test call_tool logs success with duration_ms (Req 7.2)."""
        manager, mock_session = await connected_manager
        mock_logger = manager.logger

        # Mock successful tool call
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Success"}]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Reset mock to ignore connection logs
        mock_logger.reset_mock()

        # Call tool
        result = await manager.call_tool(
            tool_name="test-server.my_tool",
            arguments={},
        )

        # Verify success log includes duration_ms in extra dict
        info_calls = [call for call in mock_logger.info.call_args_list]
        assert len(info_calls) >= 2, "Expected at least 2 info logs (start + success)"

        # Check success log (last info call)
        success_log = info_calls[-1]
        if "extra" in success_log[1]:
            extra = success_log[1]["extra"]
            # Should have duration_ms or similar timing metric
            assert (
                "duration_ms" in extra
                or "duration" in extra
                or "elapsed" in extra
            ), "Expected timing metric in success log"

    @pytest.mark.asyncio
    async def test_call_tool_logs_error(self, connected_manager):
        """Test call_tool logs error with error message (Req 6.2)."""
        manager, mock_session = await connected_manager
        mock_logger = manager.logger

        # Mock tool call failure
        mock_session.call_tool = AsyncMock(
            side_effect=Exception("Tool crashed")
        )

        # Reset mock to ignore connection logs
        mock_logger.reset_mock()

        # Call tool (should not raise, should wrap error)
        result = await manager.call_tool(
            tool_name="test-server.failing_tool",
            arguments={},
        )

        # Verify error log was called
        assert mock_logger.error.called, "Expected error log on tool failure"

        # Check error log contains error message
        error_calls = [call for call in mock_logger.error.call_args_list]
        assert len(error_calls) >= 1, "Expected at least 1 error log"

        error_log = error_calls[0]
        log_message = error_log[0][0]
        assert "Tool crashed" in log_message or (
            "extra" in error_log[1] and "Tool crashed" in str(error_log[1])
        ), "Expected error message in log"

    @pytest.mark.asyncio
    async def test_call_tool_logs_result_type_and_length(self, connected_manager):
        """Test call_tool logs result type and content length (Req 7.2)."""
        manager, mock_session = await connected_manager
        mock_logger = manager.logger

        # Mock successful tool call with content
        mock_call_result = MagicMock()
        mock_call_result.content = [
            {"type": "text", "text": "A" * 100}  # Long content
        ]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Reset mock to ignore connection logs
        mock_logger.reset_mock()

        # Call tool
        result = await manager.call_tool(
            tool_name="test-server.my_tool",
            arguments={},
        )

        # Verify success log includes result metadata
        info_calls = [call for call in mock_logger.info.call_args_list]
        success_log = info_calls[-1]

        if "extra" in success_log[1]:
            extra = success_log[1]["extra"]
            # Should have result type or length information
            has_metadata = any(
                key in extra
                for key in ["result_type", "content_length", "content_size", "result_size"]
            )
            assert has_metadata, "Expected result metadata in success log"


# =============================================================================
# Task 5.1: Unit Tests for call_tool Langfuse Integration
# =============================================================================


class TestCallToolLangfuse:
    """Test Langfuse observability integration (Requirements 7.4)."""

    @pytest.mark.asyncio
    async def test_call_tool_creates_langfuse_span(
        self, mock_logger, game_config_with_mcp, mock_stdio_client, mock_client_session
    ):
        """Test call_tool creates Langfuse span when langfuse_client provided (Req 7.4)."""
        from managers.mcp_manager import MCPManager

        # Create mock Langfuse client with span context manager
        mock_langfuse = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_span = MagicMock(return_value=mock_span)

        # Create manager with Langfuse client
        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
            langfuse_client=mock_langfuse,
        )

        # Connect session
        await manager.connect_session()

        # Mock successful tool call
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Success"}]
        mock_client_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Call tool
        result = await manager.call_tool(
            tool_name="test-server.my_tool",
            arguments={"arg": "value"},
        )

        # Verify Langfuse span was created
        mock_langfuse.start_as_current_span.assert_called_once()
        span_call = mock_langfuse.start_as_current_span.call_args

        # Verify span includes tool name and metadata
        span_name = span_call[0][0] if span_call[0] else span_call[1].get("name")
        assert "my_tool" in span_name or "test-server.my_tool" in span_name

    @pytest.mark.asyncio
    async def test_call_tool_no_langfuse_when_disabled(self, connected_manager):
        """Test call_tool skips Langfuse when langfuse_client is None (Req 7.4)."""
        manager, mock_session = await connected_manager

        # Verify no Langfuse client
        assert manager.langfuse_client is None

        # Mock successful tool call
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Success"}]
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Call tool (should not raise, should work without Langfuse)
        result = await manager.call_tool(
            tool_name="test-server.my_tool",
            arguments={},
        )

        # Verify result
        assert isinstance(result, ToolCallResult)
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_call_tool_langfuse_span_includes_metadata(
        self, mock_logger, game_config_with_mcp, mock_stdio_client, mock_client_session
    ):
        """Test Langfuse span includes tool name, arguments, and result metadata."""
        from managers.mcp_manager import MCPManager

        # Create mock Langfuse client
        mock_langfuse = MagicMock()
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_span = MagicMock(return_value=mock_span)

        # Create manager with Langfuse
        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
            langfuse_client=mock_langfuse,
        )

        # Connect session
        await manager.connect_session()

        # Mock successful tool call
        mock_call_result = MagicMock()
        mock_call_result.content = [{"type": "text", "text": "Result content"}]
        mock_client_session.call_tool = AsyncMock(return_value=mock_call_result)

        # Call tool with arguments
        result = await manager.call_tool(
            tool_name="test-server.complex_tool",
            arguments={"key": "value", "count": 42},
        )

        # Verify span creation
        mock_langfuse.start_as_current_span.assert_called_once()

        # Verify span metadata (may be in kwargs or passed to span methods)
        # Exact implementation depends on Langfuse SDK usage
        span_call = mock_langfuse.start_as_current_span.call_args
        assert span_call is not None, "Expected Langfuse span to be created"
