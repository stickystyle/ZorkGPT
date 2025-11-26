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
