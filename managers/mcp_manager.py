# ABOUTME: MCPManager handles MCP server connections and session lifecycle.
# ABOUTME: Uses MCP SDK for stdio transport, manages per-turn sessions.

import logging
import os
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from session.game_configuration import GameConfiguration
from managers.mcp_config import (
    MCPConfig,
    MCPServerConfig,
    MCPServerStartupError,
    load_mcp_config,
)


class MCPManager:
    """Manages MCP server connections and tool execution.

    Lifecycle: Per-turn session connect/disconnect via stdio transport.
    Graceful degradation: Tracks failures and disables MCP after retries.

    This class is NOT a traditional ZorkGPT manager (doesn't extend BaseManager).
    It's an async utility class that manages MCP protocol interactions.
    """

    def __init__(
        self,
        config: GameConfiguration,
        logger: logging.Logger,
        langfuse_client: Optional[Any] = None,
    ):
        """Initialize MCP manager.

        Args:
            config: Game configuration with MCP settings
            logger: Logger instance for structured logging
            langfuse_client: Optional Langfuse client for observability

        Raises:
            MCPConfigError: If mcp_config.json missing or invalid
        """
        self.config = config
        self.logger = logger
        self.langfuse_client = langfuse_client

        # Session state
        self._session: Optional[ClientSession] = None
        self._stdio_context: Optional[Any] = None
        self._disabled: bool = False
        self._retry_attempted: bool = False

        # Load server configuration
        self._mcp_config: MCPConfig = self._load_mcp_config()
        self._server_name: str
        self._server_config: MCPServerConfig
        self._server_name, self._server_config = self._mcp_config.get_server_config()

    def _load_mcp_config(self) -> MCPConfig:
        """Load MCP server configuration from config file.

        Returns:
            Validated MCPConfig instance

        Raises:
            MCPConfigError: If file missing or invalid
        """
        return load_mcp_config(self.config.mcp_config_file)

    @property
    def is_disabled(self) -> bool:
        """Check if MCP has been disabled due to repeated failures."""
        return self._disabled

    async def connect_session(self) -> None:
        """Connect MCP session and spawn subprocess.

        Called at turn start. Spawns subprocess via stdio transport,
        creates ClientSession, and performs protocol handshake.

        Raises:
            MCPServerStartupError: If server fails to start
        """
        if self._disabled:
            return

        try:
            await self._connect_session_impl()
        except Exception as e:
            # Re-raise as MCPServerStartupError
            raise MCPServerStartupError(
                f"Failed to start MCP server '{self._server_name}': {e}\n"
                f"Command: {self._server_config.command} {' '.join(self._server_config.args)}\n"
                f"Check that the server command is available and properly configured."
            ) from e

    async def _connect_session_impl(self) -> None:
        """Internal implementation of session connection."""
        # Build environment with merging
        env = self._build_subprocess_env()

        # Create server parameters
        server_params = StdioServerParameters(
            command=self._server_config.command,
            args=self._server_config.args,
            env=env,
        )

        # Start stdio transport (spawns subprocess)
        self._stdio_context = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()

        # Create session
        self._session = ClientSession(read_stream, write_stream)

        # Protocol handshake (Req 3.3)
        await self._session.initialize()

        self.logger.info(
            f"MCP session connected to '{self._server_name}'",
            extra={
                "event_type": "mcp_session_connected",
                "server_name": self._server_name,
            },
        )

    def _build_subprocess_env(self) -> Dict[str, str]:
        """Build environment variables for subprocess.

        Merges system environment with config-specified env vars.
        Config env vars override system vars (Req 3.2).

        Returns:
            Merged environment dictionary
        """
        # Start with system environment
        env = dict(os.environ)

        # Merge config env vars (override on collision)
        if self._server_config.env:
            env.update(self._server_config.env)

        return env

    async def disconnect_session(self) -> None:
        """Disconnect MCP session and terminate subprocess.

        Called at turn end. Closes session and terminates subprocess
        to ensure clean state (Req 3.5, 3.7).
        """
        if self._session is None:
            return

        try:
            # Close session context (terminates subprocess)
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)

            self.logger.info(
                f"MCP session disconnected from '{self._server_name}'",
                extra={
                    "event_type": "mcp_session_disconnected",
                    "server_name": self._server_name,
                },
            )
        except Exception as e:
            self.logger.warning(
                f"Error during MCP session disconnect: {e}",
                extra={
                    "event_type": "mcp_session_disconnect_error",
                    "error": str(e),
                },
            )
        finally:
            self._session = None
            self._stdio_context = None
