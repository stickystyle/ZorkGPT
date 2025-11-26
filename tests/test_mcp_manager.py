# ABOUTME: Unit tests for MCPManager initialization, environment merging, and session lifecycle.
# ABOUTME: Tests MCP server connection management and graceful degradation tracking.

import json
import logging
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from session.game_configuration import GameConfiguration
from managers.mcp_config import MCPConfigError, MCPServerStartupError


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


# =============================================================================
# Task 3.1: Unit Tests for MCPManager Initialization
# =============================================================================


class TestMCPManagerInitialization:
    """Test MCPManager initialization (Requirements 2.1, 2.2, 2.3, 3.2)."""

    def test_initialization_with_valid_config(
        self, mock_logger, game_config_with_mcp
    ):
        """Test MCPManager accepts config and logger (Req 2.1)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        assert manager.config == game_config_with_mcp
        assert manager.logger == mock_logger
        assert manager.langfuse_client is None  # Default

    def test_initialization_with_langfuse_client(
        self, mock_logger, game_config_with_mcp
    ):
        """Test MCPManager accepts optional langfuse_client parameter."""
        from managers.mcp_manager import MCPManager

        mock_langfuse = MagicMock()
        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
            langfuse_client=mock_langfuse,
        )

        assert manager.langfuse_client == mock_langfuse

    def test_initialization_loads_mcp_config(
        self, mock_logger, game_config_with_mcp
    ):
        """Test MCPManager loads server config on initialization (Req 2.1)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Verify server config was loaded
        assert manager._server_name == "test-server"
        assert manager._server_config.command == "echo"
        assert manager._server_config.args == ["hello"]

    def test_initialization_raises_on_missing_config_file(
        self, mock_logger, tmp_path
    ):
        """Test MCPManager raises MCPConfigError when config file missing (Req 2.2)."""
        from managers.mcp_manager import MCPManager

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(tmp_path / "nonexistent.json"),
        )

        with pytest.raises(MCPConfigError) as exc_info:
            MCPManager(config=config, logger=mock_logger)

        assert "config file not found" in str(exc_info.value).lower()

    def test_initialization_raises_on_invalid_json(
        self, mock_logger, tmp_path
    ):
        """Test MCPManager raises MCPConfigError on invalid JSON (Req 2.3)."""
        from managers.mcp_manager import MCPManager

        # Create invalid JSON file
        config_file = tmp_path / "bad_config.json"
        config_file.write_text("{ invalid json }")

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        with pytest.raises(MCPConfigError) as exc_info:
            MCPManager(config=config, logger=mock_logger)

        assert "invalid json" in str(exc_info.value).lower()

    def test_is_disabled_defaults_to_false(
        self, mock_logger, game_config_with_mcp
    ):
        """Test is_disabled property defaults to False (Req 6.9)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        assert manager.is_disabled is False


# =============================================================================
# Task 3.1: Unit Tests for Environment Variable Merging
# =============================================================================


class TestMCPManagerEnvironmentMerging:
    """Test environment variable merging for subprocess (Requirement 3.2)."""

    def test_env_vars_merged_with_system(
        self, mock_logger, game_config_with_mcp
    ):
        """Test config env vars merged with system environment (Req 3.2)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Build subprocess env
        env = manager._build_subprocess_env()

        # Should have system PATH (assuming it exists)
        assert "PATH" in env or len(env) > 0  # System env should be present

        # Should have config env var
        assert env.get("TEST_VAR") == "test_value"

    def test_config_env_overrides_system(
        self, mock_logger, tmp_path, create_mcp_config_file
    ):
        """Test config env vars override system vars on collision (Req 3.2)."""
        from managers.mcp_manager import MCPManager

        # Create config that overrides a known env var
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": [],
                    "env": {"PATH": "/custom/path"},
                }
            }
        }
        config_file = create_mcp_config_file(config_data)

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=config_file,
        )

        manager = MCPManager(config=config, logger=mock_logger)
        env = manager._build_subprocess_env()

        # Config should override system PATH
        assert env["PATH"] == "/custom/path"

    def test_no_env_uses_system_only(
        self, mock_logger, tmp_path, create_mcp_config_file
    ):
        """Test fallback to system environment when no config env (Req 3.2)."""
        from managers.mcp_manager import MCPManager

        # Create config without env vars
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": [],
                    # No env field
                }
            }
        }
        config_file = create_mcp_config_file(config_data)

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=config_file,
        )

        manager = MCPManager(config=config, logger=mock_logger)
        env = manager._build_subprocess_env()

        # Should have system environment
        assert "PATH" in env or len(env) > 0
        # Should NOT have any extra test vars
        assert "TEST_VAR" not in env


# =============================================================================
# Task 3.1: Unit Tests for Session Lifecycle
# =============================================================================


class TestMCPManagerSessionLifecycle:
    """Test MCP session connect/disconnect lifecycle (Requirements 3.1, 3.3, 3.5, 3.7)."""

    @pytest.mark.asyncio
    async def test_connect_session_creates_session(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """Test connect_session creates session object (Req 3.1)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        await manager.connect_session()

        assert manager._session is not None
        assert manager._stdio_context is not None

    @pytest.mark.asyncio
    async def test_connect_session_calls_initialize(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """Test connect_session calls session.initialize() for handshake (Req 3.3)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        await manager.connect_session()

        # Verify initialize was called
        mock_client_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_session_logs_connection(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """Test connect_session logs successful connection."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        await manager.connect_session()

        # Verify logging
        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0]
        assert "connected" in log_message.lower()

    @pytest.mark.asyncio
    async def test_disconnect_session_closes_session(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """Test disconnect_session closes session and context (Req 3.5, 3.7)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        await manager.connect_session()
        await manager.disconnect_session()

        # Verify context exit was called
        mock_stdio_client.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_none_after_disconnect(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """Test session is None after disconnect (state cleared)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        await manager.connect_session()
        assert manager._session is not None

        await manager.disconnect_session()
        assert manager._session is None
        assert manager._stdio_context is None

    @pytest.mark.asyncio
    async def test_disconnect_session_logs_disconnection(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """Test disconnect_session logs successful disconnection."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        await manager.connect_session()
        mock_logger.reset_mock()
        await manager.disconnect_session()

        # Verify logging
        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0]
        assert "disconnected" in log_message.lower()

    @pytest.mark.asyncio
    async def test_disconnect_session_noop_when_not_connected(
        self,
        mock_logger,
        game_config_with_mcp,
    ):
        """Test disconnect_session is no-op when not connected."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Should not raise
        await manager.disconnect_session()

        # Session should still be None
        assert manager._session is None

    @pytest.mark.asyncio
    async def test_connect_session_skipped_when_disabled(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """Test connect_session is no-op when manager is disabled (Req 6.9)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Manually disable
        manager._disabled = True

        await manager.connect_session()

        # Should not have connected
        assert manager._session is None
        mock_stdio_client.__aenter__.assert_not_called()


# =============================================================================
# Task 3.1: Unit Tests for Error Handling
# =============================================================================


class TestMCPManagerErrorHandling:
    """Test MCP error handling during connection."""

    @pytest.mark.asyncio
    async def test_connect_session_raises_startup_error_on_failure(
        self,
        mock_logger,
        game_config_with_mcp,
    ):
        """Test connect_session raises MCPServerStartupError on failure."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Mock stdio_client to raise an exception
        with patch(
            "managers.mcp_manager.stdio_client",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(MCPServerStartupError) as exc_info:
                await manager.connect_session()

            error_msg = str(exc_info.value)
            assert "failed to start" in error_msg.lower()
            assert "test-server" in error_msg

    @pytest.mark.asyncio
    async def test_disconnect_session_handles_error_gracefully(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_client_session,
    ):
        """Test disconnect_session handles errors without raising."""
        from managers.mcp_manager import MCPManager

        # Create a mock context that raises on exit
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(side_effect=Exception("Disconnect error"))

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            manager = MCPManager(
                config=game_config_with_mcp,
                logger=mock_logger,
            )

            await manager.connect_session()

            # Should not raise, just log warning
            await manager.disconnect_session()

            # Should have logged warning
            mock_logger.warning.assert_called()

            # State should still be cleaned up
            assert manager._session is None
            assert manager._stdio_context is None


# =============================================================================
# Task 6.1: Unit Tests for MCP Graceful Degradation
# =============================================================================


class TestMCPManagerGracefulDegradation:
    """Test MCP graceful degradation and retry logic (Requirements 6.1, 6.7, 6.8, 6.9)."""

    def test_successful_connections_counter_starts_at_zero(
        self, mock_logger, game_config_with_mcp
    ):
        """
        Test _successful_connections counter initialized to zero.

        Verifies initial state before any connection attempts.
        """
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        assert manager._successful_connections == 0

    @pytest.mark.asyncio
    async def test_successful_connections_increments_after_connect(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """
        Test _successful_connections increments after successful connection.

        Verifies counter increments to distinguish first turn from subsequent turns.
        """
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # First successful connection
        await manager.connect_session()
        assert manager._successful_connections == 1

        # Disconnect and reconnect
        await manager.disconnect_session()
        await manager.connect_session()
        assert manager._successful_connections == 2

    @pytest.mark.asyncio
    async def test_first_turn_failure_no_retry(
        self,
        mock_logger,
        game_config_with_mcp,
    ):
        """
        Test first turn failure fails fast without retry (Requirement 6.1).

        When _successful_connections == 0 and connection fails:
        - Should raise MCPServerStartupError immediately
        - Should NOT attempt retry
        - Should NOT set _disabled flag
        """
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Verify initial state
        assert manager._successful_connections == 0
        assert manager._retry_attempted is False

        # Mock connection failure
        with patch(
            "managers.mcp_manager.stdio_client",
            side_effect=Exception("First turn connection failed"),
        ):
            with pytest.raises(MCPServerStartupError) as exc_info:
                await manager.connect_session()

            # Verify error details
            error_msg = str(exc_info.value)
            assert "failed to start" in error_msg.lower()

        # Verify no retry was attempted
        assert manager._retry_attempted is False
        assert manager._disabled is False
        assert manager._successful_connections == 0

    @pytest.mark.asyncio
    async def test_subsequent_turn_failure_with_successful_retry(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_client_session,
    ):
        """
        Test subsequent turn failure with successful retry (Requirement 6.7).

        When _successful_connections > 0 and connection fails once then succeeds:
        - Should retry exactly once
        - Should succeed on retry
        - Should log warning for retry attempt
        - Should log info for retry success
        - Should increment successful_connections counter
        """
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Simulate previous successful connection
        manager._successful_connections = 1

        # Mock stdio_client: fail first time, succeed second time
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "managers.mcp_manager.stdio_client",
            side_effect=[
                Exception("Transient failure"),  # First attempt fails
                mock_context,  # Second attempt succeeds
            ],
        ):
            # Should succeed after retry
            await manager.connect_session()

        # Verify session was established
        assert manager._session is not None
        assert manager._successful_connections == 2

        # Verify retry warning was logged
        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "retry" in str(call).lower()
        ]
        assert len(warning_calls) >= 1, "Should log warning for retry attempt"

        # Verify retry success was logged
        info_calls = [
            call for call in mock_logger.info.call_args_list
            if "retry" in str(call).lower() and "success" in str(call).lower()
        ]
        assert len(info_calls) >= 1, "Should log info for retry success"

    @pytest.mark.asyncio
    async def test_subsequent_turn_failure_with_failed_retry(
        self,
        mock_logger,
        game_config_with_mcp,
    ):
        """
        Test subsequent turn failure with failed retry (Requirements 6.8, 6.9).

        When _successful_connections > 0 and both connection attempts fail:
        - Should retry exactly once
        - Should NOT raise exception (graceful degradation)
        - Should set _disabled = True
        - Should log warning for degradation
        - Should NOT increment successful_connections counter
        """
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Simulate previous successful connection
        manager._successful_connections = 1

        # Mock stdio_client: fail both times
        with patch(
            "managers.mcp_manager.stdio_client",
            side_effect=[
                Exception("First failure"),
                Exception("Second failure"),
            ],
        ):
            # Should NOT raise exception
            await manager.connect_session()

        # Verify manager is disabled
        assert manager._disabled is True
        assert manager.is_disabled is True

        # Verify session was NOT established
        assert manager._session is None

        # Verify counter did NOT increment
        assert manager._successful_connections == 1

        # Verify degradation warning was logged
        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "disabl" in str(call).lower() or "degrad" in str(call).lower()
        ]
        assert len(warning_calls) >= 1, "Should log warning for degradation"

    @pytest.mark.asyncio
    async def test_disabled_flag_prevents_connection_attempts(
        self,
        mock_logger,
        game_config_with_mcp,
    ):
        """
        Test _disabled flag prevents future connection attempts (Requirement 6.9).

        When _disabled = True:
        - connect_session should return immediately
        - Should not attempt any connection
        - Should not modify state
        """
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Manually disable manager
        manager._disabled = True

        # Mock stdio_client to track if it's called
        with patch("managers.mcp_manager.stdio_client") as mock_stdio:
            await manager.connect_session()

            # Should not have attempted connection
            mock_stdio.assert_not_called()

        # State should remain unchanged
        assert manager._session is None
        assert manager._successful_connections == 0

    @pytest.mark.asyncio
    async def test_retry_attempted_flag_resets_on_success(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """
        Test _retry_attempted flag resets after successful connection.

        Ensures retry flag doesn't persist across successful connections.
        """
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Simulate previous retry state
        manager._retry_attempted = True
        manager._successful_connections = 1

        # Successful connection should reset flag
        await manager.connect_session()

        assert manager._retry_attempted is False
        assert manager._successful_connections == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_first_subsequent_failure_if_retry_used(
        self,
        mock_logger,
        game_config_with_mcp,
    ):
        """
        Test only one retry attempt per episode.

        If retry already used (_retry_attempted=True), subsequent failures
        should disable immediately without additional retry.
        """
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Simulate state after a retry was already used
        manager._successful_connections = 1
        manager._retry_attempted = True

        # Connection failure should disable immediately
        with patch(
            "managers.mcp_manager.stdio_client",
            side_effect=Exception("Connection failed"),
        ):
            await manager.connect_session()

        # Should be disabled without additional retry
        assert manager._disabled is True
        assert manager._retry_attempted is True  # Should remain True
        assert manager._successful_connections == 1  # Should not increment
