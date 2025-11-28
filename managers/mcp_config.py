# ABOUTME: MCP server configuration models for validation and loading.
# ABOUTME: Defines MCPServerConfig and MCPConfig Pydantic models.

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    pass


class MCPConfigError(MCPError):
    """Exception for MCP configuration errors."""

    pass


class MCPServerStartupError(MCPError):
    """Raised when an MCP server fails to start."""

    pass


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    command: str = Field(
        description="Command to launch the server (e.g., 'npx', 'uvx')"
    )
    args: List[str] = Field(
        default_factory=list, description="Arguments for the command"
    )
    env: Optional[Dict[str, str]] = Field(
        default=None, description="Environment variables to set for the subprocess"
    )


class MCPConfig(BaseModel):
    """Root MCP configuration schema (loaded from mcp_config.json).

    Note: V1 supports a single MCP server only. The config format uses
    mcpServers dict for compatibility with Claude Desktop/Cline, but
    only the first server entry is used.
    """

    mcpServers: Dict[str, MCPServerConfig] = Field(
        description="Map of server name to server configuration"
    )

    def get_server_config(self) -> Tuple[str, MCPServerConfig]:
        """Get the single server config (V1: first entry only).

        Returns:
            Tuple of (server_name, server_config)

        Raises:
            ValueError: If no servers configured
        """
        if not self.mcpServers:
            raise ValueError("No MCP servers configured in mcpServers")
        server_name = next(iter(self.mcpServers))
        return server_name, self.mcpServers[server_name]


def load_mcp_config(config_file: str) -> MCPConfig:
    """Load MCP server configuration from JSON file.

    Args:
        config_file: Path to mcp_config.json

    Returns:
        Validated MCPConfig instance

    Raises:
        MCPConfigError: If file missing or invalid JSON
    """
    config_path = Path(config_file)

    if not config_path.exists():
        raise MCPConfigError(
            f"MCP is enabled but config file not found: {config_file}\n"
            f"Please either:\n"
            f"  1. Create {config_file} in project root, or\n"
            f"  2. Set mcp.enabled = false in pyproject.toml"
        )

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise MCPConfigError(
            f"Invalid JSON in MCP config file: {config_file}\n" f"Parse error: {e}"
        )

    return MCPConfig.model_validate(data)
