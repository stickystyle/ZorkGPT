# ABOUTME: Tests for objective-based progress configuration loading
# ABOUTME: Validates enable_objective_based_progress field loads correctly from TOML

import pytest
from pathlib import Path
from session.game_configuration import GameConfiguration


class TestObjectiveBasedProgressConfig:
    """Test suite for objective-based progress configuration."""

    def test_default_value_is_true(self):
        """Test that enable_objective_based_progress defaults to True."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
        )

        assert hasattr(config, "enable_objective_based_progress")
        assert config.enable_objective_based_progress is True

    def test_explicit_value_true(self):
        """Test that enable_objective_based_progress can be set to True."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            enable_objective_based_progress=True,
        )

        assert config.enable_objective_based_progress is True

    def test_explicit_value_false(self):
        """Test that enable_objective_based_progress can be set to False."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            enable_objective_based_progress=False,
        )

        assert config.enable_objective_based_progress is False

    def test_from_toml_loads_true(self, tmp_path):
        """Test that enable_objective_based_progress loads True from TOML."""
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

[tool.zorkgpt.loop_break]
max_turns_stuck = 40
stuck_check_interval = 10
enable_objective_based_progress = true
enable_location_penalty = true
location_revisit_penalty = -0.2
location_revisit_window = 5
enable_exploration_hints = true
action_novelty_window = 15
enable_stuck_warnings = true
stuck_warning_threshold = 20
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(toml_content)

        # Load config from TOML
        config = GameConfiguration.from_toml(toml_file)

        # Verify enable_objective_based_progress loaded correctly
        assert config.enable_objective_based_progress is True

    def test_from_toml_loads_false(self, tmp_path):
        """Test that enable_objective_based_progress loads False from TOML."""
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

[tool.zorkgpt.loop_break]
max_turns_stuck = 40
stuck_check_interval = 10
enable_objective_based_progress = false
enable_location_penalty = true
location_revisit_penalty = -0.2
location_revisit_window = 5
enable_exploration_hints = true
action_novelty_window = 15
enable_stuck_warnings = true
stuck_warning_threshold = 20
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(toml_content)

        # Load config from TOML
        config = GameConfiguration.from_toml(toml_file)

        # Verify enable_objective_based_progress loaded correctly
        assert config.enable_objective_based_progress is False

    def test_from_toml_defaults_to_true_when_missing(self, tmp_path):
        """Test that enable_objective_based_progress defaults to True if not in TOML."""
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

[tool.zorkgpt.loop_break]
max_turns_stuck = 40
stuck_check_interval = 10
enable_location_penalty = true
location_revisit_penalty = -0.2
location_revisit_window = 5
enable_exploration_hints = true
action_novelty_window = 15
enable_stuck_warnings = true
stuck_warning_threshold = 20
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(toml_content)

        # Load config from TOML (without enable_objective_based_progress in loop_break)
        config = GameConfiguration.from_toml(toml_file)

        # Should default to True
        assert config.enable_objective_based_progress is True
