# ABOUTME: Tests for MCP configuration loading and validation.
# ABOUTME: Validates pyproject.toml MCP settings, mcp_config.json loading, and Pydantic models.

import pytest
import json
from pathlib import Path

from session.game_configuration import GameConfiguration


class TestMCPConfigurationDefaults:
    """Test MCP configuration default values (Requirements 13.1-13.5)."""

    def test_mcp_enabled_defaults_to_false(self):
        """Test mcp_enabled defaults to False (implicit from Requirements)."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )
        assert config.mcp_enabled is False

    def test_mcp_config_file_defaults_to_mcp_config_json(self):
        """Test mcp_config_file defaults to 'mcp_config.json' (Req 13.4)."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )
        assert config.mcp_config_file == "mcp_config.json"

    def test_mcp_max_tool_iterations_defaults_to_20(self):
        """Test mcp_max_tool_iterations defaults to 20 (Req 13.1)."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )
        assert config.mcp_max_tool_iterations == 20

    def test_mcp_tool_call_timeout_seconds_defaults_to_30(self):
        """Test mcp_tool_call_timeout_seconds defaults to 30 (Req 13.2)."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )
        assert config.mcp_tool_call_timeout_seconds == 30

    def test_mcp_server_startup_timeout_seconds_defaults_to_10(self):
        """Test mcp_server_startup_timeout_seconds defaults to 10 (Req 13.3)."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )
        assert config.mcp_server_startup_timeout_seconds == 10

    def test_mcp_force_tool_support_defaults_to_false(self):
        """Test mcp_force_tool_support defaults to False (Req 13.5)."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )
        assert config.mcp_force_tool_support is False


def _get_minimal_toml_content(mcp_section: str = "") -> str:
    """Generate minimal valid TOML content with optional MCP section."""
    return f"""
[tool.zorkgpt.orchestrator]
max_turns_per_episode = 100
knowledge_update_interval = 50
objective_update_interval = 25
enable_objective_refinement = true
objective_refinement_interval = 75
max_objectives_before_forced_refinement = 20
refined_objectives_target_count = 10
enable_state_export = false
enable_inter_episode_synthesis = true

[tool.zorkgpt.gameplay]
turn_delay_seconds = 0.5
turn_window_size = 50
min_knowledge_quality = 7.0
critic_rejection_threshold = -0.1
enable_exit_pruning = true
exit_failure_threshold = 3
zork_save_filename_template = "test_save_{{timestamp}}"
zork_game_workdir = "test_game_files"

[tool.zorkgpt.files]
episode_log_file = "test_episode.log"
json_log_file = "test_episode.jsonl"
state_export_file = "test_state.json"
map_state_file = "test_map.json"
knowledge_file = "test_knowledge.md"
game_file_path = "test_game.z5"

[tool.zorkgpt.llm]
client_base_url = "http://localhost:1234"
agent_model = "test-agent"
critic_model = "test-critic"
info_ext_model = "test-extractor"
analysis_model = "test-analysis"
memory_model = "test-memory"

[tool.zorkgpt.aws]
s3_key_prefix = "test-prefix/"

[tool.zorkgpt.simple_memory]
memory_file = "TestMemories.md"
max_memories_shown = 5
{mcp_section}
"""


class TestMCPConfigurationFromTOML:
    """Test MCP settings loading from pyproject.toml (Requirement 2.4)."""

    def test_mcp_settings_load_from_toml(self, tmp_path):
        """Test loading MCP settings from [tool.zorkgpt.mcp] section (Req 2.4)."""
        mcp_section = """
[tool.zorkgpt.mcp]
enabled = true
config_file = "custom_mcp.json"
max_tool_iterations = 15
tool_call_timeout_seconds = 45
server_startup_timeout_seconds = 15
force_tool_support = true
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(_get_minimal_toml_content(mcp_section))

        config = GameConfiguration.from_toml(toml_file)

        assert config.mcp_enabled is True
        assert config.mcp_config_file == "custom_mcp.json"
        assert config.mcp_max_tool_iterations == 15
        assert config.mcp_tool_call_timeout_seconds == 45
        assert config.mcp_server_startup_timeout_seconds == 15
        assert config.mcp_force_tool_support is True

    def test_mcp_settings_use_defaults_when_section_absent(self, tmp_path):
        """Test MCP defaults when [tool.zorkgpt.mcp] section is absent."""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(_get_minimal_toml_content())

        config = GameConfiguration.from_toml(toml_file)

        # Should use all defaults
        assert config.mcp_enabled is False
        assert config.mcp_config_file == "mcp_config.json"
        assert config.mcp_max_tool_iterations == 20
        assert config.mcp_tool_call_timeout_seconds == 30
        assert config.mcp_server_startup_timeout_seconds == 10
        assert config.mcp_force_tool_support is False

    def test_mcp_partial_settings_from_toml(self, tmp_path):
        """Test partial MCP settings load correctly with others using defaults."""
        mcp_section = """
[tool.zorkgpt.mcp]
enabled = true
max_tool_iterations = 25
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(_get_minimal_toml_content(mcp_section))

        config = GameConfiguration.from_toml(toml_file)

        # Explicit values
        assert config.mcp_enabled is True
        assert config.mcp_max_tool_iterations == 25
        # Defaults for others
        assert config.mcp_config_file == "mcp_config.json"
        assert config.mcp_tool_call_timeout_seconds == 30
        assert config.mcp_server_startup_timeout_seconds == 10
        assert config.mcp_force_tool_support is False


class TestMCPConfigurationValidation:
    """Test MCP configuration validation constraints."""

    def test_mcp_max_tool_iterations_minimum_boundary(self):
        """Test mcp_max_tool_iterations must be >= 1."""
        with pytest.raises(Exception) as exc_info:
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                mcp_max_tool_iterations=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value).lower() or "ge" in str(exc_info.value).lower()

    def test_mcp_max_tool_iterations_maximum_boundary(self):
        """Test mcp_max_tool_iterations must be <= 100."""
        with pytest.raises(Exception) as exc_info:
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                mcp_max_tool_iterations=101,
            )
        assert "less than or equal to 100" in str(exc_info.value).lower() or "le" in str(exc_info.value).lower()

    def test_mcp_tool_call_timeout_minimum_boundary(self):
        """Test mcp_tool_call_timeout_seconds must be >= 1."""
        with pytest.raises(Exception):
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                mcp_tool_call_timeout_seconds=0,
            )

    def test_mcp_server_startup_timeout_minimum_boundary(self):
        """Test mcp_server_startup_timeout_seconds must be >= 1."""
        with pytest.raises(Exception):
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                mcp_server_startup_timeout_seconds=0,
            )


class TestMCPServerConfigModels:
    """Test MCP server configuration Pydantic models."""

    def test_mcp_server_config_validates_required_fields(self):
        """Test MCPServerConfig requires command field."""
        from managers.mcp_config import MCPServerConfig

        # Valid config
        config = MCPServerConfig(command="npx", args=["-y", "thoughtbox"])
        assert config.command == "npx"
        assert config.args == ["-y", "thoughtbox"]
        assert config.env is None

    def test_mcp_server_config_with_env_vars(self):
        """Test MCPServerConfig with environment variables."""
        from managers.mcp_config import MCPServerConfig

        config = MCPServerConfig(
            command="npx",
            args=["-y", "thoughtbox"],
            env={"DEBUG": "true", "LOG_LEVEL": "info"},
        )
        assert config.env == {"DEBUG": "true", "LOG_LEVEL": "info"}

    def test_mcp_server_config_missing_command_raises_error(self):
        """Test MCPServerConfig requires command field."""
        from managers.mcp_config import MCPServerConfig

        with pytest.raises(Exception):
            MCPServerConfig(args=["-y", "thoughtbox"])

    def test_mcp_config_validates_structure(self):
        """Test MCPConfig validates mcpServers structure."""
        from managers.mcp_config import MCPConfig, MCPServerConfig

        config = MCPConfig(
            mcpServers={
                "thoughtbox": MCPServerConfig(
                    command="npx",
                    args=["-y", "@anthropic/mcp-server-sequential-thinking"],
                )
            }
        )
        assert "thoughtbox" in config.mcpServers

    def test_mcp_config_get_server_config_returns_first_entry(self):
        """Test get_server_config() returns (name, config) tuple (Req 2.6)."""
        from managers.mcp_config import MCPConfig, MCPServerConfig

        config = MCPConfig(
            mcpServers={
                "server1": MCPServerConfig(command="cmd1", args=[]),
                "server2": MCPServerConfig(command="cmd2", args=[]),
            }
        )
        server_name, server_config = config.get_server_config()

        # Should return first entry (V1 limitation)
        assert server_name == "server1"
        assert server_config.command == "cmd1"

    def test_mcp_config_get_server_config_raises_on_empty(self):
        """Test get_server_config() raises ValueError if no servers."""
        from managers.mcp_config import MCPConfig

        config = MCPConfig(mcpServers={})

        with pytest.raises(ValueError, match="No MCP servers configured"):
            config.get_server_config()


class TestMCPConfigLoading:
    """Test loading MCP configuration from mcp_config.json file."""

    def test_load_valid_mcp_config_json(self, tmp_path, monkeypatch):
        """Test loading valid mcp_config.json (Req 2.1)."""
        from managers.mcp_config import load_mcp_config

        monkeypatch.chdir(tmp_path)

        config_data = {
            "mcpServers": {
                "thoughtbox": {
                    "command": "npx",
                    "args": ["-y", "@anthropic/mcp-server-sequential-thinking"],
                    "env": {"DEBUG": "false"},
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        mcp_config = load_mcp_config(str(config_file))

        assert "thoughtbox" in mcp_config.mcpServers
        assert mcp_config.mcpServers["thoughtbox"].command == "npx"

    def test_missing_config_file_raises_error(self, tmp_path, monkeypatch):
        """Test error when mcp_config.json missing and MCP enabled (Req 2.2)."""
        from managers.mcp_config import load_mcp_config, MCPConfigError

        monkeypatch.chdir(tmp_path)

        with pytest.raises(MCPConfigError) as exc_info:
            load_mcp_config("nonexistent_mcp_config.json")

        error_msg = str(exc_info.value)
        assert "config file not found" in error_msg.lower()
        assert "nonexistent_mcp_config.json" in error_msg

    def test_invalid_json_raises_error(self, tmp_path, monkeypatch):
        """Test error when mcp_config.json has invalid JSON (Req 2.3)."""
        from managers.mcp_config import load_mcp_config, MCPConfigError

        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "mcp_config.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(MCPConfigError) as exc_info:
            load_mcp_config(str(config_file))

        error_msg = str(exc_info.value)
        assert "invalid json" in error_msg.lower()

    def test_v1_uses_first_server_only(self, tmp_path, monkeypatch):
        """Test only first server entry used (Req 2.6)."""
        from managers.mcp_config import load_mcp_config

        monkeypatch.chdir(tmp_path)

        config_data = {
            "mcpServers": {
                "first_server": {"command": "first", "args": []},
                "second_server": {"command": "second", "args": []},
                "third_server": {"command": "third", "args": []},
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        mcp_config = load_mcp_config(str(config_file))
        server_name, server_config = mcp_config.get_server_config()

        # V1 limitation: only first server used
        assert server_name == "first_server"
        assert server_config.command == "first"


class TestMCPDisabledBehavior:
    """Test behavior when MCP is disabled (Requirement 2.5)."""

    def test_mcp_disabled_does_not_require_config_file(self, tmp_path):
        """Test that disabled MCP doesn't need mcp_config.json (Req 2.5)."""
        # Create config with MCP disabled - should not fail even without mcp_config.json
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=False,
        )
        assert config.mcp_enabled is False

    def test_mcp_disabled_from_toml_does_not_require_config_file(self, tmp_path):
        """Test disabled MCP from TOML doesn't require config file (Req 2.5)."""
        mcp_section = """
[tool.zorkgpt.mcp]
enabled = false
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(_get_minimal_toml_content(mcp_section))

        # Should load without error even without mcp_config.json
        config = GameConfiguration.from_toml(toml_file)
        assert config.mcp_enabled is False
