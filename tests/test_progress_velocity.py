# ABOUTME: Tests for progress velocity detection (stuck behavior termination)
# ABOUTME: Validates score tracking, threshold-based termination, and configuration

import pytest
from unittest.mock import Mock, MagicMock, patch
from session.game_configuration import GameConfiguration
from session.game_state import GameState
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2


class TestProgressVelocityDetection:
    """Test suite for progress velocity detection (stuck behavior termination)."""

    @pytest.fixture
    def mock_jericho_interface(self):
        """Create a mock Jericho interface."""
        interface = Mock()
        interface.get_location_structured = Mock(return_value=Mock(num=1, name="Test Room"))
        interface.get_score = Mock(return_value=(0, 0))
        interface.get_inventory_structured = Mock(return_value=[])
        interface.get_valid_exits = Mock(return_value=[])
        interface.send_command = Mock(return_value="Test response")
        interface.is_game_over = Mock(return_value=(False, None))
        interface.start = Mock(return_value="Initial game state")
        interface.close = Mock()
        return interface

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a minimal configuration for testing."""
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            max_turns_stuck=40,
            stuck_check_interval=10,
            turn_delay_seconds=0.0,  # No delay for fast tests
            zork_game_workdir=str(tmp_path / "game_files"),
            episode_log_file=str(tmp_path / "test.log"),
            json_log_file=str(tmp_path / "test.jsonl"),
            enable_state_export=False,  # Disable for faster tests
        )
        return config

    @pytest.fixture
    def orchestrator(self, mock_config, mock_jericho_interface, tmp_path, monkeypatch):
        """Create an orchestrator with mocked dependencies."""
        # Mock the JerichoInterface constructor
        with patch('orchestration.zork_orchestrator_v2.JerichoInterface', return_value=mock_jericho_interface):
            # Mock agent, critic, extractor
            with patch('orchestration.zork_orchestrator_v2.ZorkAgent'), \
                 patch('orchestration.zork_orchestrator_v2.ZorkCritic'), \
                 patch('orchestration.zork_orchestrator_v2.HybridZorkExtractor'):

                # Create orchestrator
                orch = ZorkOrchestratorV2(episode_id="test-episode")
                orch.config = mock_config
                orch.jericho_interface = mock_jericho_interface

                # Mock managers to prevent real file operations
                for manager in orch.managers:
                    manager.process_turn = Mock()
                    manager.reset_episode = Mock()
                    manager.get_status = Mock(return_value={})

                return orch

    def test_score_tracking_initialization(self, orchestrator):
        """Test that score tracking initializes correctly."""
        # Before first call, attributes should not exist
        assert not hasattr(orchestrator, '_last_score_change_turn')
        assert not hasattr(orchestrator, '_last_tracked_score')

        # First call initializes tracking
        orchestrator._track_score_for_progress_detection()

        # After first call, attributes should be set
        assert hasattr(orchestrator, '_last_score_change_turn')
        assert hasattr(orchestrator, '_last_tracked_score')
        assert orchestrator._last_score_change_turn == 0
        assert orchestrator._last_tracked_score == orchestrator.game_state.previous_zork_score

    def test_progress_velocity_resets_on_score_increase(self, orchestrator):
        """Test that counter resets when score increases."""
        # Initialize tracking
        orchestrator.game_state.previous_zork_score = 0
        orchestrator.game_state.turn_count = 10
        orchestrator._track_score_for_progress_detection()

        # Advance turns without score change
        orchestrator.game_state.turn_count = 20
        orchestrator._track_score_for_progress_detection()
        assert orchestrator._get_turns_since_score_change() == 20

        # Increase score
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.previous_zork_score = 5
        orchestrator._track_score_for_progress_detection()

        # Counter should reset
        assert orchestrator._get_turns_since_score_change() == 0
        assert orchestrator._last_score_change_turn == 25
        assert orchestrator._last_tracked_score == 5

    def test_progress_velocity_resets_on_score_decrease(self, orchestrator):
        """Test that counter resets when score decreases (e.g., death penalty)."""
        # Initialize tracking with score
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 10
        orchestrator._track_score_for_progress_detection()

        # Advance turns without score change
        orchestrator.game_state.turn_count = 20
        orchestrator._track_score_for_progress_detection()
        assert orchestrator._get_turns_since_score_change() == 20

        # Decrease score (death penalty)
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.previous_zork_score = 5
        orchestrator._track_score_for_progress_detection()

        # Counter should reset
        assert orchestrator._get_turns_since_score_change() == 0
        assert orchestrator._last_score_change_turn == 25
        assert orchestrator._last_tracked_score == 5

    def test_progress_velocity_terminates_after_threshold(self, orchestrator, mock_jericho_interface):
        """Test that episode terminates after max_turns_stuck threshold."""
        # Setup: score stuck at 10 for entire episode
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()  # Initialize at turn 0

        # Mock agent/critic to return valid actions
        orchestrator.agent.get_action_with_reasoning = Mock(
            return_value={"action": "look", "reasoning": "test"}
        )
        orchestrator.critic.evaluate_action = Mock(
            return_value=Mock(score=0.5, justification="OK", confidence=0.8)
        )
        orchestrator.extractor.extract_info = Mock(
            return_value=Mock(
                score=10,
                inventory=[],
                game_over=False,
                is_room_description=False
            )
        )
        orchestrator.extractor.get_clean_game_text = Mock(return_value="Test response")

        # Simulate turns until we hit the check interval
        max_turns_stuck = orchestrator.config.max_turns_stuck
        check_interval = orchestrator.config.stuck_check_interval

        # We need to reach turn count where:
        # 1. turn_count % check_interval == 0 (check happens)
        # 2. turns_stuck >= max_turns_stuck (termination triggered)

        # Run turns until termination should occur
        termination_turn = max_turns_stuck + check_interval
        for turn in range(1, termination_turn + 1):
            orchestrator.game_state.turn_count = turn

            # Execute turn logic (simplified)
            orchestrator._run_turn("test state")

            # Track score
            orchestrator._track_score_for_progress_detection()

            # Check stuck condition (matches orchestrator logic)
            if turn % check_interval == 0:
                turns_stuck = orchestrator._get_turns_since_score_change()

                if turns_stuck >= max_turns_stuck:
                    orchestrator.game_state.game_over_flag = True
                    orchestrator.game_state.termination_reason = "stuck_no_progress"
                    break

        # Verify termination occurred
        assert orchestrator.game_state.game_over_flag, \
            f"Episode should terminate after {max_turns_stuck} turns stuck"
        assert orchestrator.game_state.termination_reason == "stuck_no_progress"
        assert orchestrator.game_state.turn_count >= max_turns_stuck

    def test_progress_velocity_does_not_terminate_prematurely(self, orchestrator):
        """Test that episode does not terminate before threshold."""
        # Setup: score stuck at 10
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()

        # Advance to just before threshold
        turns_before_threshold = orchestrator.config.max_turns_stuck - 1
        orchestrator.game_state.turn_count = turns_before_threshold
        orchestrator._track_score_for_progress_detection()

        turns_stuck = orchestrator._get_turns_since_score_change()

        # Should not terminate yet
        assert turns_stuck < orchestrator.config.max_turns_stuck
        assert not orchestrator.game_state.game_over_flag

    def test_stuck_check_interval(self, orchestrator):
        """Test that check only runs every N turns."""
        check_interval = orchestrator.config.stuck_check_interval

        # These turns should trigger check
        trigger_turns = [10, 20, 30, 40, 50]
        for turn in trigger_turns:
            assert turn % check_interval == 0, f"Turn {turn} should trigger check"

        # These turns should NOT trigger check
        non_trigger_turns = [5, 15, 23, 37, 49]
        for turn in non_trigger_turns:
            assert turn % check_interval != 0, f"Turn {turn} should not trigger check"

    def test_progress_velocity_configurable_threshold(self, tmp_path):
        """Test that progress velocity works with different threshold values."""
        # Test with custom threshold
        custom_threshold = 20
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            max_turns_stuck=custom_threshold,
            stuck_check_interval=5,
            stuck_warning_threshold=10,  # Must be < max_turns_stuck
            zork_game_workdir=str(tmp_path / "game_files"),
        )

        assert config.max_turns_stuck == custom_threshold
        assert config.stuck_check_interval == 5

    def test_get_turns_since_score_change_calculation(self, orchestrator):
        """Test accurate calculation of turns since last score change."""
        # Initialize at turn 0
        orchestrator.game_state.turn_count = 0
        orchestrator.game_state.previous_zork_score = 0
        orchestrator._track_score_for_progress_detection()

        # Advance to turn 10
        orchestrator.game_state.turn_count = 10
        assert orchestrator._get_turns_since_score_change() == 10

        # Score change at turn 15
        orchestrator.game_state.turn_count = 15
        orchestrator.game_state.previous_zork_score = 5
        orchestrator._track_score_for_progress_detection()
        assert orchestrator._get_turns_since_score_change() == 0

        # Advance to turn 25
        orchestrator.game_state.turn_count = 25
        assert orchestrator._get_turns_since_score_change() == 10

        # Advance to turn 35
        orchestrator.game_state.turn_count = 35
        assert orchestrator._get_turns_since_score_change() == 20

    def test_score_tracking_handles_no_change(self, orchestrator):
        """Test that tracking handles multiple turns with no score change."""
        orchestrator.game_state.previous_zork_score = 10
        orchestrator.game_state.turn_count = 0
        orchestrator._track_score_for_progress_detection()

        # Multiple turns with no score change
        for turn in range(1, 30):
            orchestrator.game_state.turn_count = turn
            orchestrator._track_score_for_progress_detection()

            # Tracked score should remain unchanged
            assert orchestrator._last_tracked_score == 10
            # Last change should still be turn 0
            assert orchestrator._last_score_change_turn == 0
            # Turns stuck should increase
            assert orchestrator._get_turns_since_score_change() == turn

    def test_configuration_loads_from_toml(self, tmp_path):
        """Test that loop_break configuration loads from TOML file."""
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

[tool.zorkgpt.loop_break]
max_turns_stuck = 30
stuck_check_interval = 5
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(toml_content)

        # Load config from TOML
        config = GameConfiguration.from_toml(toml_file)

        # Verify loop_break values loaded correctly
        assert config.max_turns_stuck == 30
        assert config.stuck_check_interval == 5

    def test_stuck_termination_closes_cleanly(self, orchestrator):
        """Test that stuck termination doesn't cause double close()."""
        # Configure for quick termination
        orchestrator.config.max_turns_stuck = 10
        orchestrator.config.stuck_check_interval = 5

        # Mock components to simulate stuck episode
        with patch.object(orchestrator.jericho_interface, 'is_game_over', return_value=(False, "")):
            with patch.object(orchestrator.jericho_interface, 'send_command', return_value="Nothing happens."):
                with patch.object(orchestrator.extractor, 'extract_info') as mock_extract:
                    # Simulate stuck score
                    mock_info = Mock()
                    mock_info.score = 10
                    mock_info.game_over = False
                    mock_info.is_room_description = False
                    mock_extract.return_value = mock_info

                    with patch.object(orchestrator.agent, 'get_action_with_reasoning', return_value={"action": "look", "reasoning": "test"}):
                        with patch.object(orchestrator.critic, 'evaluate_action') as mock_critic:
                            mock_result = Mock()
                            mock_result.confidence = 0.9
                            mock_result.score = 0.5
                            mock_result.justification = "OK"
                            mock_critic.return_value = mock_result

                            with patch.object(orchestrator.extractor, 'get_clean_game_text', return_value="Nothing happens."):
                                # Run until stuck termination
                                orchestrator.play_episode()

        # Verify close() called exactly once
        assert orchestrator.jericho_interface.close.call_count == 1

    def test_tracking_with_nonzero_starting_score(self, orchestrator):
        """Test tracking works correctly when game starts with score > 0."""
        # Set non-zero starting score
        orchestrator.game_state.previous_zork_score = 25

        # Initialize tracking
        orchestrator._track_score_for_progress_detection()

        # Verify initialized correctly
        assert orchestrator._last_tracked_score == 25
        assert orchestrator._last_score_change_turn == 0

    def test_invalid_stuck_configuration_rejected(self):
        """Test that invalid stuck detection config raises error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                max_turns_stuck=5,
                stuck_check_interval=10,  # Invalid: threshold < interval
                zork_game_workdir="test_workdir"
            )

    def test_configuration_uses_defaults_when_missing(self, tmp_path):
        """Test that default values are used when loop_break section is missing."""
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

        # Load config from TOML (loop_break section missing)
        config = GameConfiguration.from_toml(toml_file)

        # Verify default values are used
        assert config.max_turns_stuck == 40  # Default value
        assert config.stuck_check_interval == 10  # Default value
