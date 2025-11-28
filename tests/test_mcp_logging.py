# ABOUTME: Tests for MCP logging and observability (Task #17, Req 7)
# ABOUTME: Validates JSON log structure, human-readable formatting, and Langfuse integration

import asyncio
import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List, Dict, Any

from managers.mcp_manager import MCPManager
from managers.mcp_config import MCPConfig
from session.game_configuration import GameConfiguration
from logger import HumanReadableFormatter, JSONFormatter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_langfuse_client():
    """Mock Langfuse client for tracing tests."""
    client = MagicMock()
    client.start_as_current_span = MagicMock()
    span = MagicMock()
    span.update = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    client.start_as_current_span.return_value = span
    return client


@pytest.fixture
def mock_mcp_session():
    """Mock MCP session for testing."""
    session = MagicMock()
    session.call_tool = AsyncMock()

    # Mock successful tool result
    mock_result = MagicMock()
    mock_result.content = [{"type": "text", "text": "Success"}]
    mock_result.isError = False
    session.call_tool.return_value = mock_result

    return session


@pytest.fixture
def mcp_manager_with_mocks(tmp_path, mock_langfuse_client, mock_mcp_session):
    """MCPManager with mocked dependencies."""
    logger = logging.getLogger("test_mcp_logging")
    logger.setLevel(logging.INFO)

    # Create minimal mcp_config.json for MCPManager
    mcp_config_content = {
        "mcpServers": {
            "test": {
                "command": "echo",
                "args": ["test"]
            }
        }
    }
    mcp_config_path = tmp_path / "mcp_config.json"
    mcp_config_path.write_text(json.dumps(mcp_config_content))

    config = GameConfiguration.from_toml()
    config.mcp_config_file = str(mcp_config_path)
    config.mcp_tool_call_timeout_seconds = 30

    manager = MCPManager(
        config=config,
        logger=logger,
        langfuse_client=mock_langfuse_client,
    )

    # Inject mock session
    manager._session = mock_mcp_session

    return manager


@pytest.fixture
def log_capture():
    """Capture log records for testing."""
    class LogCapture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    return LogCapture()


# ============================================================================
# JSON Log Structure Tests (Req 7.1, 7.2, 7.3)
# ============================================================================


class TestJSONLogStructure:
    """Test JSON log structure for MCP events."""

    @pytest.mark.asyncio
    async def test_tool_call_start_includes_iteration(
        self, mcp_manager_with_mocks, log_capture
    ):
        """Test that mcp_tool_call_start log includes iteration number."""
        manager = mcp_manager_with_mocks
        manager.logger.addHandler(log_capture)

        # Call tool with iteration number
        await manager.call_tool(
            tool_name="test.tool",
            arguments={"arg": "value"},
            iteration=5
        )

        # Find the start log
        start_logs = [
            r for r in log_capture.records
            if hasattr(r, 'event_type') and r.event_type == 'mcp_tool_call_start'
        ]

        assert len(start_logs) == 1, "Should have one mcp_tool_call_start log"
        record = start_logs[0]

        # Verify required fields (Req 7.1)
        assert hasattr(record, 'tool_name')
        assert record.tool_name == "test.tool"  # Full prefixed name
        assert hasattr(record, 'arguments')
        assert record.arguments == {"arg": "value"}
        assert hasattr(record, 'iteration')
        assert record.iteration == 5, "Should include iteration number"

    @pytest.mark.asyncio
    async def test_tool_call_success_includes_metadata(
        self, mcp_manager_with_mocks, log_capture
    ):
        """Test that mcp_tool_call_success log includes result metadata."""
        manager = mcp_manager_with_mocks
        manager.logger.addHandler(log_capture)

        # Call tool
        await manager.call_tool(
            tool_name="test.tool",
            arguments={"arg": "value"}
        )

        # Find the success log
        success_logs = [
            r for r in log_capture.records
            if hasattr(r, 'event_type') and r.event_type == 'mcp_tool_call_success'
        ]

        assert len(success_logs) == 1, "Should have one mcp_tool_call_success log"
        record = success_logs[0]

        # Verify required fields (Req 7.2)
        assert hasattr(record, 'tool_name')
        assert hasattr(record, 'result_type')
        assert hasattr(record, 'result_length')
        assert hasattr(record, 'duration_ms')
        assert record.duration_ms > 0, "Should measure duration"

    def test_json_formatter_preserves_extra_fields(self):
        """Test that JSONFormatter preserves all extra fields."""
        formatter = JSONFormatter()

        # Create log record with MCP fields
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Add MCP-specific fields
        record.event_type = "mcp_tool_call_start"
        record.tool_name = "test_tool"
        record.iteration = 5
        record.arguments = {"arg": "value"}

        # Format and parse
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Verify all fields present
        assert parsed["event_type"] == "mcp_tool_call_start"
        assert parsed["tool_name"] == "test_tool"
        assert parsed["iteration"] == 5
        assert parsed["arguments"] == {"arg": "value"}


# ============================================================================
# Iteration Parameter Tests
# ============================================================================


class TestIterationParameter:
    """Test iteration parameter in call_tool()."""

    @pytest.mark.asyncio
    async def test_call_tool_accepts_iteration_parameter(
        self, mcp_manager_with_mocks
    ):
        """Test that call_tool() accepts optional iteration parameter."""
        manager = mcp_manager_with_mocks

        # Should not raise exception
        result = await manager.call_tool(
            tool_name="test.tool",
            arguments={},
            iteration=10
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_call_tool_iteration_defaults_to_none(
        self, mcp_manager_with_mocks, log_capture
    ):
        """Test that iteration defaults to None if not provided."""
        manager = mcp_manager_with_mocks
        manager.logger.addHandler(log_capture)

        # Call without iteration
        await manager.call_tool(
            tool_name="test.tool",
            arguments={}
        )

        # Find start log
        start_logs = [
            r for r in log_capture.records
            if hasattr(r, 'event_type') and r.event_type == 'mcp_tool_call_start'
        ]

        assert len(start_logs) == 1
        record = start_logs[0]

        # Should have iteration field, but it may be None
        assert hasattr(record, 'iteration')


# ============================================================================
# Human-Readable Formatting Tests
# ============================================================================


class TestHumanReadableFormatting:
    """Test human-readable formatting for MCP events."""

    def test_mcp_session_connected_hidden(self):
        """Test that mcp_session_connected is hidden from console."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Connected to MCP server",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_session_connected"
        record.server_name = "test-server"

        formatted = formatter.format(record)

        assert formatted is None, "Should be hidden from console for cleaner output"

    def test_mcp_tool_call_start_hidden(self):
        """Test that mcp_tool_call_start is hidden from console (success shows usage)."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Tool call start",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_tool_call_start"
        record.tool_name = "search"
        record.iteration = 3

        formatted = formatter.format(record)

        assert formatted is None, "Should be hidden; success event shows usage instead"

    def test_mcp_tool_call_success_format(self):
        """Test formatting for mcp_tool_call_success event."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Tool call success",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_tool_call_success"
        record.tool_name = "search"
        record.duration_ms = 1234.56

        formatted = formatter.format(record)

        # Compact format with thought bubble emoji
        assert formatted == "  💭 search (1235ms)"

    def test_mcp_tool_call_error_format(self):
        """Test formatting for mcp_tool_call_error event."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Tool call error",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_tool_call_error"
        record.tool_name = "search"
        record.error = "Connection timeout"

        formatted = formatter.format(record)

        assert formatted == "  ✗ MCP Tool error: search - Connection timeout"

    def test_mcp_tool_call_timeout_format(self):
        """Test formatting for mcp_tool_call_timeout event."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Tool call timeout",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_tool_call_timeout"
        record.tool_name = "search"
        record.timeout_seconds = 30

        formatted = formatter.format(record)

        assert formatted == "  ⏱ MCP Tool timeout: search (30s)"

    def test_mcp_session_summary_hidden(self):
        """Test that mcp_session_summary is hidden from console."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Session summary",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_session_summary"
        record.iterations = 5
        record.tool_calls = 3
        record.duration_ms = 5678.90

        formatted = formatter.format(record)

        assert formatted is None, "Should be hidden from console for cleaner output"

    def test_mcp_session_disconnected_hidden(self):
        """Test that mcp_session_disconnected is hidden from console."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Disconnected",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_session_disconnected"

        formatted = formatter.format(record)

        assert formatted is None, "Should be hidden from console"

    def test_mcp_session_connect_retry_format(self):
        """Test formatting for mcp_session_connect_retry event."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Connection retry",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_session_connect_retry"

        formatted = formatter.format(record)

        assert formatted == "⚠️ MCP: Connection retry..."

    def test_mcp_session_degradation_format(self):
        """Test formatting for mcp_session_degradation event."""
        formatter = HumanReadableFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Session degradation",
            args=(),
            exc_info=None
        )
        record.event_type = "mcp_session_degradation"

        formatted = formatter.format(record)

        assert formatted == "⚠️ MCP: Disabled for remainder of episode"


# ============================================================================
# Langfuse Span Tests (Req 7.4, 7.5)
# ============================================================================


class TestLangfuseSpans:
    """Test Langfuse span creation for tool calls."""

    @pytest.mark.asyncio
    async def test_langfuse_span_created_for_tool_call(
        self, mcp_manager_with_mocks, mock_langfuse_client
    ):
        """Test that Langfuse span is created for each tool call."""
        manager = mcp_manager_with_mocks

        await manager.call_tool(
            tool_name="test.search",
            arguments={"query": "test"}
        )

        # Verify span was created
        mock_langfuse_client.start_as_current_span.assert_called_once()

        call_args = mock_langfuse_client.start_as_current_span.call_args
        assert call_args[1]["name"] == "mcp-tool-search"
        assert call_args[1]["input"] == {"query": "test"}
        assert "tool_name" in call_args[1]["metadata"]
        assert "server_name" in call_args[1]["metadata"]

    @pytest.mark.asyncio
    async def test_langfuse_span_updated_on_success(
        self, mcp_manager_with_mocks, mock_langfuse_client
    ):
        """Test that Langfuse span is updated with result on success."""
        manager = mcp_manager_with_mocks

        await manager.call_tool(
            tool_name="test.search",
            arguments={"query": "test"}
        )

        # Get the span that was created
        span = mock_langfuse_client.start_as_current_span.return_value.__enter__.return_value

        # Verify span was updated
        span.update.assert_called_once()

        call_args = span.update.call_args
        assert "output" in call_args[1]
        assert "metadata" in call_args[1]
        assert call_args[1]["metadata"]["is_error"] is False
        assert "duration_ms" in call_args[1]["metadata"]

    @pytest.mark.asyncio
    async def test_langfuse_span_updated_on_error(
        self, mcp_manager_with_mocks, mock_langfuse_client, mock_mcp_session
    ):
        """Test that Langfuse span is updated with error details on failure."""
        manager = mcp_manager_with_mocks

        # Mock error result
        error_result = MagicMock()
        error_result.content = "Error occurred"
        error_result.isError = True
        mock_mcp_session.call_tool.return_value = error_result

        await manager.call_tool(
            tool_name="test.search",
            arguments={"query": "test"}
        )

        # Get the span that was created
        span = mock_langfuse_client.start_as_current_span.return_value.__enter__.return_value

        # Verify span was updated with error
        span.update.assert_called_once()

        call_args = span.update.call_args
        assert call_args[1]["metadata"]["is_error"] is True


# ============================================================================
# Error Message Formatting Tests
# ============================================================================


class TestErrorMessageFormatting:
    """Test error message formatting for tool failures."""

    @pytest.mark.asyncio
    async def test_mcp_error_log_includes_details(
        self, mcp_manager_with_mocks, mock_mcp_session, log_capture
    ):
        """Test that MCP error logs include error details."""
        manager = mcp_manager_with_mocks
        manager.logger.addHandler(log_capture)

        # Mock error result
        error_result = MagicMock()
        error_result.content = "Connection refused"
        error_result.isError = True
        mock_mcp_session.call_tool.return_value = error_result

        await manager.call_tool(
            tool_name="test.tool",
            arguments={}
        )

        # Find error log
        error_logs = [
            r for r in log_capture.records
            if hasattr(r, 'event_type') and r.event_type == 'mcp_tool_call_error'
        ]

        assert len(error_logs) == 1
        record = error_logs[0]

        assert hasattr(record, 'error')
        assert record.error == "MCP server error"
        assert hasattr(record, 'duration_ms')

    @pytest.mark.asyncio
    async def test_timeout_error_log_includes_timeout_value(
        self, mcp_manager_with_mocks, mock_mcp_session, log_capture
    ):
        """Test that timeout error logs include timeout value."""
        manager = mcp_manager_with_mocks
        manager.logger.addHandler(log_capture)

        # Mock timeout
        mock_mcp_session.call_tool.side_effect = asyncio.TimeoutError()

        # TimeoutError is re-raised by MCPManager (Req 6.5)
        with pytest.raises(asyncio.TimeoutError):
            await manager.call_tool(
                tool_name="test.tool",
                arguments={},
                timeout_seconds=15
            )

        # Find timeout log
        timeout_logs = [
            r for r in log_capture.records
            if hasattr(r, 'event_type') and r.event_type == 'mcp_tool_call_timeout'
        ]

        assert len(timeout_logs) == 1
        record = timeout_logs[0]

        assert hasattr(record, 'timeout_seconds')
        assert record.timeout_seconds == 15


# ============================================================================
# Integration Tests
# ============================================================================


class TestMCPLoggingIntegration:
    """Integration tests for MCP logging."""

    @pytest.mark.asyncio
    async def test_full_tool_call_lifecycle_logging(
        self, mcp_manager_with_mocks, log_capture
    ):
        """Test that all log events are created for a successful tool call."""
        manager = mcp_manager_with_mocks
        manager.logger.addHandler(log_capture)

        await manager.call_tool(
            tool_name="test.search",
            arguments={"query": "test"},
            iteration=5
        )

        # Verify we got start and success logs
        event_types = [
            getattr(r, 'event_type', None)
            for r in log_capture.records
            if hasattr(r, 'event_type')
        ]

        assert 'mcp_tool_call_start' in event_types
        assert 'mcp_tool_call_success' in event_types

        # Verify start log has iteration
        start_log = next(
            r for r in log_capture.records
            if hasattr(r, 'event_type') and r.event_type == 'mcp_tool_call_start'
        )
        assert start_log.iteration == 5
