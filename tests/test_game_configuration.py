# ABOUTME: Tests for GameConfiguration class
# ABOUTME: Validates configuration loading, backward compatibility, and validation

import pytest
import os
from pathlib import Path
from session.game_configuration import GameConfiguration


class TestGameConfiguration:
    """Test suite for GameConfiguration class."""

    def test_backward_compatibility_client_api_key(self, monkeypatch):
        """Test that CLIENT_API_KEY (legacy) works as fallback."""
        # Clear any existing env vars
        monkeypatch.delenv("ZORKGPT_CLIENT_API_KEY", raising=False)
        monkeypatch.delenv("CLIENT_API_KEY", raising=False)

        # Set legacy env var
        monkeypatch.setenv("CLIENT_API_KEY", "legacy-api-key")

        # Create config (should pick up legacy env var in model_post_init)
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )

        assert config.client_api_key == "legacy-api-key"

    def test_backward_compatibility_s3_bucket(self, monkeypatch):
        """Test that ZORK_S3_BUCKET (legacy) works as fallback."""
        # Clear any existing env vars
        monkeypatch.delenv("ZORKGPT_S3_BUCKET", raising=False)
        monkeypatch.delenv("ZORK_S3_BUCKET", raising=False)

        # Set legacy env var
        monkeypatch.setenv("ZORK_S3_BUCKET", "legacy-bucket")

        # Create config (should pick up legacy env var in model_post_init)
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )

        assert config.s3_bucket == "legacy-bucket"

    def test_new_env_vars_take_precedence(self, monkeypatch):
        """Test that ZORKGPT_* env vars override legacy ones."""
        # Set both legacy and new env vars
        monkeypatch.setenv("CLIENT_API_KEY", "legacy-api-key")
        monkeypatch.setenv("ZORKGPT_CLIENT_API_KEY", "new-api-key")
        monkeypatch.setenv("ZORK_S3_BUCKET", "legacy-bucket")
        monkeypatch.setenv("ZORKGPT_S3_BUCKET", "new-bucket")

        # Create config (new vars should win)
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            client_api_key="new-api-key",  # Pydantic will load from ZORKGPT_ env
            s3_bucket="new-bucket",
        )

        assert config.client_api_key == "new-api-key"
        assert config.s3_bucket == "new-bucket"

    def test_zork_save_filename_template_field_exists(self):
        """Test that zork_save_filename_template field is present."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )

        assert hasattr(config, "zork_save_filename_template")
        assert config.zork_save_filename_template == "zorkgpt_save_{timestamp}"

    def test_extra_fields_are_forbidden(self):
        """Test that extra='forbid' catches typos in field names."""
        with pytest.raises(Exception) as exc_info:
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                invalid_field_name="should_fail",  # Typo in field name
            )

        assert "extra_forbidden" in str(exc_info.value) or "Extra inputs are not permitted" in str(exc_info.value)

    def test_from_toml_loads_configuration(self, tmp_path):
        """Test that from_toml() properly loads configuration."""
        # Create a minimal pyproject.toml
        toml_content = """
[tool.zorkgpt.orchestrator]
max_turns_per_episode = 150
knowledge_update_interval = 50
objective_update_interval = 25
enable_objective_refinement = true
objective_refinement_interval = 75
max_objectives_before_forced_refinement = 20
refined_objectives_target_count = 10
max_context_tokens = 100000
context_overflow_threshold = 0.8
enable_state_export = false
enable_inter_episode_synthesis = true

[tool.zorkgpt.gameplay]
turn_delay_seconds = 0.5
turn_window_size = 50
min_knowledge_quality = 7.0
critic_rejection_threshold = -0.1
enable_exit_pruning = true
exit_failure_threshold = 3
enable_knowledge_condensation = true
knowledge_condensation_threshold = 20000
zork_save_filename_template = "test_save_{timestamp}"
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
condensation_model = "test-condensation"

[tool.zorkgpt.aws]
s3_key_prefix = "test-prefix/"

[tool.zorkgpt.simple_memory]
memory_file = "TestMemories.md"
max_memories_shown = 5

[tool.zorkgpt.retry]
max_retries = 3
initial_delay = 0.5
max_delay = 30.0
exponential_base = 2.0
jitter_factor = 0.1
retry_on_timeout = true
retry_on_rate_limit = true
retry_on_server_error = true
timeout_seconds = 60.0
circuit_breaker_enabled = true
circuit_breaker_failure_threshold = 5
circuit_breaker_recovery_timeout = 120.0
circuit_breaker_success_threshold = 2
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(toml_content)

        # Load config from TOML
        config = GameConfiguration.from_toml(toml_file)

        # Verify loaded values
        assert config.max_turns_per_episode == 150
        assert config.knowledge_update_interval == 50
        assert config.turn_delay_seconds == 0.5
        assert config.zork_save_filename_template == "test_save_{timestamp}"
        assert config.agent_model == "test-agent"
        assert config.critic_model == "test-critic"
        assert config.simple_memory_max_shown == 5

    def test_workdir_creation(self, tmp_path, monkeypatch):
        """Test that game workdir is created if it doesn't exist."""
        # Change to tmp directory
        monkeypatch.chdir(tmp_path)

        workdir = tmp_path / "test_workdir"
        assert not workdir.exists()

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            zork_game_workdir=str(workdir),
        )

        # Workdir should be created in model_post_init
        assert workdir.exists()
        assert workdir.is_dir()
