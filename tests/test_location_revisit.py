# ABOUTME: Tests for location revisit penalty (anti-loop mechanism)
# ABOUTME: Validates location tracking, revisit detection, and penalty calculation

import pytest
from unittest.mock import Mock, MagicMock, patch
from session.game_configuration import GameConfiguration
from session.game_state import GameState
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2


class TestLocationRevisitPenalty:
    """Test suite for location revisit penalty (anti-loop mechanism)."""

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
            turn_delay_seconds=0.0,
            zork_game_workdir=str(tmp_path / "game_files"),
            episode_log_file=str(tmp_path / "test.log"),
            json_log_file=str(tmp_path / "test.jsonl"),
            enable_state_export=False,
            # Location penalty settings
            enable_location_penalty=True,
            location_revisit_penalty=-0.2,
            location_revisit_window=5,
        )
        return config

    @pytest.fixture
    def orchestrator(self, mock_config, mock_jericho_interface, tmp_path):
        """Create an orchestrator with mocked dependencies."""
        with patch('orchestration.zork_orchestrator_v2.JerichoInterface', return_value=mock_jericho_interface):
            with patch('orchestration.zork_orchestrator_v2.ZorkAgent'), \
                 patch('orchestration.zork_orchestrator_v2.ZorkCritic'), \
                 patch('orchestration.zork_orchestrator_v2.HybridZorkExtractor'):

                orch = ZorkOrchestratorV2(episode_id="test-episode")
                orch.config = mock_config
                orch.jericho_interface = mock_jericho_interface

                # Mock managers
                for manager in orch.managers:
                    manager.process_turn = Mock()
                    manager.reset_episode = Mock()
                    manager.get_status = Mock(return_value={})

                # Setup map manager to have rooms for name lookup
                orch.map_manager.game_map.rooms = {
                    1: Mock(name="Test Room"),
                    2: Mock(name="Other Room"),
                    3: Mock(name="Third Room"),
                }

                return orch

    def test_location_tracking_uses_ids_not_names(self, orchestrator, mock_jericho_interface):
        """Test that location tracking uses integer IDs, not room names."""
        # Setup: Mock returns location ID 42
        mock_jericho_interface.get_location_structured.return_value = Mock(num=42, name="Test Room")

        # Track location
        orchestrator._track_location_history()

        # Verify: Should store integer ID, not name
        assert hasattr(orchestrator, '_location_id_history')
        assert len(orchestrator._location_id_history) == 1
        assert orchestrator._location_id_history[0] == 42
        assert isinstance(orchestrator._location_id_history[0], int)

    def test_location_revisit_detection_simple(self, orchestrator, mock_jericho_interface):
        """Test simple revisit detection: visit location A, then B, then return to A."""
        # Visit location 1
        mock_jericho_interface.get_location_structured.return_value = Mock(num=1, name="Room 1")
        orchestrator._track_location_history()

        # Visit location 2
        mock_jericho_interface.get_location_structured.return_value = Mock(num=2, name="Room 2")
        orchestrator._track_location_history()

        # Return to location 1 (revisit!)
        mock_jericho_interface.get_location_structured.return_value = Mock(num=1, name="Room 1")
        orchestrator._track_location_history()

        # Detect revisit
        revisit_info = orchestrator._detect_location_revisit()

        # Verify detection
        assert revisit_info["detected"] is True
        assert revisit_info["location_id"] == 1
        assert revisit_info["recent_visits"] == 1  # Seen once in history (excluding current)
        assert revisit_info["window_size"] == 2  # Checked 2 previous locations

    def test_location_revisit_penalty_calculation(self, orchestrator):
        """Test penalty calculation: -0.2 per revisit."""
        # Setup: 2 recent visits detected
        revisit_info = {
            "detected": True,
            "location_id": 1,
            "recent_visits": 2,
            "window_size": 5
        }

        base_score = 0.8

        # Apply penalty
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score=base_score,
            revisit_info=revisit_info
        )

        # Verify: 0.8 + (-0.2 * 2) = 0.4
        assert adjusted_score == 0.4
        assert "penalty -0.4" in reason.lower()
        assert "2x return" in reason.lower()

    def test_no_penalty_on_first_visit(self, orchestrator, mock_jericho_interface):
        """Test that no penalty applied on first visit to a location."""
        # First visit to location 1
        mock_jericho_interface.get_location_structured.return_value = Mock(num=1, name="Room 1")
        orchestrator._track_location_history()

        # Detect revisit
        revisit_info = orchestrator._detect_location_revisit()

        # Verify: No revisit detected
        assert revisit_info["detected"] is False
        assert revisit_info["recent_visits"] == 0

        # Apply penalty
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score=0.8,
            revisit_info=revisit_info
        )

        # Verify: No penalty applied
        assert adjusted_score == 0.8
        assert reason == ""

    def test_penalty_stacks_on_multiple_revisits(self, orchestrator, mock_jericho_interface):
        """Test that penalty increases with multiple revisits."""
        # Visit pattern: 1 -> 2 -> 1 -> 3 -> 1 (location 1 visited 3 times total)
        locations = [1, 2, 1, 3, 1]

        for loc_id in locations:
            mock_jericho_interface.get_location_structured.return_value = Mock(num=loc_id, name=f"Room {loc_id}")
            orchestrator._track_location_history()

        # Detect revisit at final position (location 1)
        revisit_info = orchestrator._detect_location_revisit()

        # Verify: 2 previous visits to location 1 in window (turns 0 and 2)
        assert revisit_info["detected"] is True
        assert revisit_info["location_id"] == 1
        assert revisit_info["recent_visits"] == 2  # Seen 2 times in recent history

        # Apply penalty
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score=0.9,
            revisit_info=revisit_info
        )

        # Verify: 0.9 + (-0.2 * 2) = 0.5
        assert adjusted_score == 0.5

    def test_sliding_window_forgets_old_revisits(self, orchestrator, mock_jericho_interface):
        """Test that old revisits outside window are forgotten."""
        # Visit pattern with window size 5:
        # Positions: 0   1   2   3   4   5   6   7   (indices)
        # Locations: 1 | 2 | 3 | 4 | 5 | 6 | 1 | (current)
        #            ^window starts here (5 back)   ^current

        # Initial visit to location 1 (outside window)
        mock_jericho_interface.get_location_structured.return_value = Mock(num=1, name="Room 1")
        orchestrator._track_location_history()

        # Fill window with other locations
        for loc_id in [2, 3, 4, 5, 6]:
            mock_jericho_interface.get_location_structured.return_value = Mock(num=loc_id, name=f"Room {loc_id}")
            orchestrator._track_location_history()

        # Return to location 1
        mock_jericho_interface.get_location_structured.return_value = Mock(num=1, name="Room 1")
        orchestrator._track_location_history()

        # Detect revisit
        revisit_info = orchestrator._detect_location_revisit()

        # Verify: First visit to location 1 is outside window, so no revisit detected
        # Window contains: [2, 3, 4, 5, 6] (previous 5 locations)
        # Location 1 at index 0 is outside the window
        assert revisit_info["detected"] is False
        assert revisit_info["recent_visits"] == 0

    def test_revisit_detection_disabled_by_config(self, orchestrator):
        """Test that penalty can be disabled via configuration."""
        # Disable penalty
        orchestrator.config.enable_location_penalty = False

        # Setup: Revisit detected
        revisit_info = {
            "detected": True,
            "location_id": 1,
            "recent_visits": 2,
            "window_size": 5
        }

        # Apply penalty
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score=0.8,
            revisit_info=revisit_info
        )

        # Verify: No penalty applied
        assert adjusted_score == 0.8
        assert reason == ""

    def test_penalty_clamped_to_zero(self, orchestrator):
        """Test that penalty cannot reduce score below 0.0."""
        # Setup: Many revisits would create large penalty
        revisit_info = {
            "detected": True,
            "location_id": 1,
            "recent_visits": 5,  # Would be -1.0 penalty
            "window_size": 5
        }

        base_score = 0.3  # 0.3 - 1.0 = -0.7, should clamp to 0.0

        # Apply penalty
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score=base_score,
            revisit_info=revisit_info
        )

        # Verify: Clamped to 0.0
        assert adjusted_score == 0.0
        assert adjusted_score >= 0.0

    def test_penalty_clamped_to_one(self, orchestrator):
        """Test that adjusted score cannot exceed 1.0 (even with positive adjustment)."""
        # Setup: This shouldn't happen in practice, but test clamping logic
        revisit_info = {
            "detected": False,  # No penalty
            "location_id": 1,
            "recent_visits": 0,
            "window_size": 5
        }

        base_score = 1.5  # Invalid high score (shouldn't happen, but test clamping)

        # Apply penalty (no penalty, but should still clamp)
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score=base_score,
            revisit_info=revisit_info
        )

        # Verify: No change (no penalty detected)
        assert adjusted_score == 1.5  # Returns unchanged when no penalty

        # Test explicit clamping with penalty
        orchestrator.config.location_revisit_penalty = 0.0  # Temporarily set to 0 for clamping test
        revisit_info["detected"] = True
        revisit_info["recent_visits"] = 1

        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score=1.5,
            revisit_info=revisit_info
        )

        # Verify: Clamped to 1.0 when penalty applied
        assert adjusted_score == 1.0

    def test_configuration_validation_penalty_must_be_negative(self):
        """Test that configuration rejects positive penalty values."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                location_revisit_penalty=0.2,  # Invalid: must be negative
                zork_game_workdir="test_workdir"
            )

    def test_configuration_validation_penalty_within_bounds(self):
        """Test that configuration rejects penalty values outside [-1.0, 0.0]."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                location_revisit_penalty=-1.5,  # Invalid: below -1.0
                zork_game_workdir="test_workdir"
            )

    def test_configuration_validation_window_size_reasonable(self):
        """Test that configuration validates window size bounds."""
        from pydantic import ValidationError

        # Too small (< 2)
        with pytest.raises(ValidationError):
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                location_revisit_window=1,  # Invalid: need at least 2
                zork_game_workdir="test_workdir"
            )

        # Too large (> 20)
        with pytest.raises(ValidationError):
            GameConfiguration(
                max_turns_per_episode=100,
                game_file_path="test.z5",
                location_revisit_window=25,  # Invalid: max is 20
                zork_game_workdir="test_workdir"
            )

    def test_location_tracking_sliding_window_max_20(self, orchestrator, mock_jericho_interface):
        """Test that location history maintains max 20 entries."""
        # Track 25 locations
        for loc_id in range(1, 26):
            mock_jericho_interface.get_location_structured.return_value = Mock(num=loc_id, name=f"Room {loc_id}")
            orchestrator._track_location_history()

        # Verify: Only last 20 stored
        assert len(orchestrator._location_id_history) == 20
        # Verify: First 5 locations dropped, last 20 kept
        assert orchestrator._location_id_history[0] == 6  # Started at 1, dropped first 5
        assert orchestrator._location_id_history[-1] == 25

    def test_detect_revisit_handles_empty_history(self, orchestrator):
        """Test that revisit detection handles empty history gracefully."""
        # Don't track any locations
        revisit_info = orchestrator._detect_location_revisit()

        # Verify: Returns safe defaults
        assert revisit_info["detected"] is False
        assert revisit_info["location_id"] is None
        assert revisit_info["recent_visits"] == 0
        assert revisit_info["window_size"] == 0

    def test_detect_revisit_handles_single_location(self, orchestrator, mock_jericho_interface):
        """Test that revisit detection handles single location (no history to check)."""
        # Track only one location
        mock_jericho_interface.get_location_structured.return_value = Mock(num=1, name="Room 1")
        orchestrator._track_location_history()

        # Detect revisit
        revisit_info = orchestrator._detect_location_revisit()

        # Verify: No revisit (need history to compare against)
        assert revisit_info["detected"] is False
        assert revisit_info["location_id"] == 1
        assert revisit_info["recent_visits"] == 0
        assert revisit_info["window_size"] == 0

    def test_configuration_loads_from_toml(self, tmp_path):
        """Test that location penalty configuration loads from TOML file."""
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
max_turns_stuck = 30
stuck_check_interval = 5
enable_location_penalty = false
location_revisit_penalty = -0.3
location_revisit_window = 8
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(toml_content)

        # Load config from TOML
        config = GameConfiguration.from_toml(toml_file)

        # Verify location penalty values loaded correctly
        assert config.enable_location_penalty is False
        assert config.location_revisit_penalty == -0.3
        assert config.location_revisit_window == 8

    def test_penalty_uses_location_name_in_reason(self, orchestrator, mock_jericho_interface):
        """Test that penalty reason includes human-readable location name."""
        # Setup map with known location
        orchestrator.map_manager.game_map.rooms = {
            1: Mock(name="West of House"),
        }

        # Detect revisit to location 1
        revisit_info = {
            "detected": True,
            "location_id": 1,
            "recent_visits": 1,
            "window_size": 5
        }

        # Apply penalty
        adjusted_score, reason = orchestrator._apply_location_revisit_penalty(
            base_score=0.8,
            revisit_info=revisit_info
        )

        # Verify: Reason includes location name
        assert "West of House" in reason
        assert "1x return" in reason

    def test_multiple_sequential_revisits(self, orchestrator, mock_jericho_interface):
        """Test detection when same location visited multiple times in sequence."""
        # Visit pattern: 1 -> 1 -> 1 (staying in same location)
        for _ in range(3):
            mock_jericho_interface.get_location_structured.return_value = Mock(num=1, name="Room 1")
            orchestrator._track_location_history()

        # Detect revisit
        revisit_info = orchestrator._detect_location_revisit()

        # Verify: Detects 2 previous visits (excluding current)
        assert revisit_info["detected"] is True
        assert revisit_info["location_id"] == 1
        assert revisit_info["recent_visits"] == 2

    def test_configuration_uses_defaults_when_missing(self, tmp_path):
        """Test that default values are used when loop_break section omits location penalty."""
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
max_turns_stuck = 30
stuck_check_interval = 5
"""
        toml_file = tmp_path / "pyproject.toml"
        toml_file.write_text(toml_content)

        # Load config from TOML (location penalty fields missing)
        config = GameConfiguration.from_toml(toml_file)

        # Verify default values are used
        assert config.enable_location_penalty is True  # Default
        assert config.location_revisit_penalty == -0.2  # Default
        assert config.location_revisit_window == 5  # Default
