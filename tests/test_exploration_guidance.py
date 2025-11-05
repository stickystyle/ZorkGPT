"""
Tests for exploration guidance features (Phase 1C).

Tests action novelty detection, unexplored exit detection,
stuck countdown warnings, and exploration hints.
"""

import pytest
from unittest.mock import Mock, patch
from session.game_configuration import GameConfiguration
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2


@pytest.fixture
def mock_config():
    """Create a real configuration for testing with adjusted values."""
    config = GameConfiguration.from_toml()
    # Override values for testing
    config.action_novelty_window = 15
    config.enable_exploration_hints = True
    config.enable_stuck_warnings = True
    config.stuck_warning_threshold = 20
    config.max_turns_stuck = 40
    config.stuck_check_interval = 10
    config.turn_delay_seconds = 0
    return config


@pytest.fixture
def mock_jericho_interface():
    """Create a mock Jericho interface."""
    mock = Mock()
    mock.get_location_structured.return_value = Mock(num=1, name="Test Room")
    mock.get_score.return_value = (0, 0)
    mock.get_inventory_structured.return_value = []
    mock.get_valid_exits.return_value = ["north", "south", "east", "west"]
    mock.is_game_over.return_value = (False, "")
    mock.start.return_value = "Game started"
    mock.send_command.return_value = "OK"
    return mock


@pytest.fixture
def orchestrator_with_game(mock_config, mock_jericho_interface, tmp_path, monkeypatch):
    """Create an orchestrator with mocked dependencies for testing."""
    # Set temp directory for game files
    monkeypatch.setattr("orchestration.zork_orchestrator_v2.GameConfiguration.from_toml",
                       lambda: mock_config)

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

            # Setup map manager to have rooms for testing
            orch.map_manager.game_map.rooms = {
                1: Mock(name="Test Room", exits={}),
                2: Mock(name="Other Room", exits={}),
            }

            # Initialize game state
            orch.game_state.current_room_id = 1
            orch.game_state.current_room_name_for_map = "Test Room"
            orch.game_state.turn_count = 0
            orch.game_state.previous_zork_score = 0

            return orch


class TestActionTracking:
    """Test action history tracking and novelty detection."""

    def test_action_tracking_and_novelty_detection(self, orchestrator_with_game):
        """Test that actions are tracked and novelty is detected correctly."""
        orch = orchestrator_with_game

        # Initially empty history
        assert not hasattr(orch, '_action_history')

        # Track first action
        orch._track_action_history("look")
        assert len(orch._action_history) == 1
        assert orch._action_history[0] == "look"

        # Track more actions
        orch._track_action_history("go north")
        orch._track_action_history("examine lamp")
        assert len(orch._action_history) == 3

        # Test novelty detection - new action should be novel
        novelty_info = orch._detect_action_novelty("open mailbox")
        assert novelty_info["is_novel"] is True
        assert novelty_info["recent_actions"] == 3
        assert novelty_info["window_size"] == 3

        # Test novelty detection - repeated action should not be novel
        novelty_info = orch._detect_action_novelty("look")
        assert novelty_info["is_novel"] is False
        assert novelty_info["recent_actions"] == 3

    def test_action_novelty_with_empty_history(self, orchestrator_with_game):
        """Test novelty detection with no action history."""
        orch = orchestrator_with_game

        # Before any tracking
        novelty_info = orch._detect_action_novelty("look")
        assert novelty_info["is_novel"] is True
        assert novelty_info["recent_actions"] == 0
        assert novelty_info["window_size"] == 0

    def test_action_tracking_window_limit(self, orchestrator_with_game):
        """Test that action history respects window size."""
        orch = orchestrator_with_game

        # Track more actions than window size
        for i in range(30):
            orch._track_action_history(f"action_{i}")

        # Should keep only max window (15 from config + max 20)
        max_window = max(orch.config.action_novelty_window, 20)
        assert len(orch._action_history) == max_window

    def test_action_normalization(self, orchestrator_with_game):
        """Test that actions are normalized for comparison."""
        orch = orchestrator_with_game

        orch._track_action_history("LOOK AROUND")
        novelty_info = orch._detect_action_novelty("look around")
        assert novelty_info["is_novel"] is False  # Same action, different case


class TestUnexploredExitDetection:
    """Test detection of unexplored exits."""

    def test_unexplored_exit_detection(self, orchestrator_with_game):
        """Test detection of unexplored exits from current location."""
        orch = orchestrator_with_game

        # Start game to get initial location
        orch.jericho_interface.start()
        orch.jericho_interface.send_command("verbose")

        # Get initial location
        location_obj = orch.jericho_interface.get_location_structured()
        location_id = location_obj.num

        # Add room to map but with unexplored exits
        orch.map_manager.game_map.add_room(location_id, "West of House")
        orch.game_state.current_room_id = location_id

        # Add a connection to an unexplored room (first create destination room, then connect)
        orch.map_manager.game_map.add_room(999, "Unexplored Room")
        orch.map_manager.game_map.add_connection(location_id, "north", 999)

        # Now remove the destination room to make it unexplored
        del orch.map_manager.game_map.rooms[999]

        # Detect unexplored exits
        unexplored_info = orch._detect_unexplored_exits()
        assert unexplored_info["has_unexplored"] is True
        assert "north" in unexplored_info["unexplored_exits"]

    def test_unexplored_exits_when_location_not_in_map(self, orchestrator_with_game):
        """Test unexplored detection when current location not in map."""
        orch = orchestrator_with_game

        # Set location that's not in map
        orch.game_state.current_room_id = 12345

        unexplored_info = orch._detect_unexplored_exits()
        assert unexplored_info["has_unexplored"] is False
        assert unexplored_info["unexplored_exits"] == []
        assert unexplored_info["all_exits"] == []

    def test_no_unexplored_exits(self, orchestrator_with_game):
        """Test when all exits have been explored."""
        orch = orchestrator_with_game

        # Start game
        orch.jericho_interface.start()
        location_obj = orch.jericho_interface.get_location_structured()
        location_id = location_obj.num

        # Add room and explored exit
        orch.map_manager.game_map.add_room(location_id, "West of House")
        orch.map_manager.game_map.add_room(2, "North of House")
        orch.map_manager.game_map.add_connection(location_id, "north", 2)
        orch.game_state.current_room_id = location_id

        unexplored_info = orch._detect_unexplored_exits()
        assert unexplored_info["has_unexplored"] is False
        assert unexplored_info["unexplored_exits"] == []


class TestStuckCountdownWarnings:
    """Test stuck countdown warning generation."""

    def test_stuck_warning_not_shown_before_threshold(self, orchestrator_with_game):
        """Test that warning is not shown before threshold is reached."""
        orch = orchestrator_with_game

        # Simulate being stuck for 19 turns (below threshold of 20)
        orch._last_score_change_turn = 0
        orch.game_state.turn_count = 19
        orch._last_tracked_score = 0

        warning = orch._build_stuck_countdown_warning()
        assert warning == ""

    def test_stuck_warning_shown_at_threshold(self, orchestrator_with_game):
        """Test that warning is shown when threshold is reached."""
        orch = orchestrator_with_game

        # Simulate being stuck for exactly 20 turns (threshold)
        orch._last_score_change_turn = 0
        orch.game_state.turn_count = 20
        orch._last_tracked_score = 0

        warning = orch._build_stuck_countdown_warning()
        assert warning != ""
        assert "SCORE STAGNATION DETECTED" in warning
        assert "20 turns" in warning
        assert "DIE in 20 turns" in warning

    def test_stuck_warning_escalates_urgency(self, orchestrator_with_game):
        """Test that warning urgency escalates as termination approaches."""
        orch = orchestrator_with_game
        orch._last_score_change_turn = 0
        orch._last_tracked_score = 0

        # Mid-range warning (20 turns stuck, 20 until death)
        orch.game_state.turn_count = 20
        warning = orch._build_stuck_countdown_warning()
        assert "SCORE STAGNATION DETECTED" in warning
        assert "SUGGESTED STRATEGIES TO BREAK FREE" in warning

        # Urgent warning (30 turns stuck, 10 until death)
        orch.game_state.turn_count = 30
        warning = orch._build_stuck_countdown_warning()
        assert "URGENT WARNING" in warning
        assert "SUGGESTED STRATEGIES TO BREAK FREE" in warning

        # Critical warning (35 turns stuck, 5 until death)
        orch.game_state.turn_count = 35
        warning = orch._build_stuck_countdown_warning()
        assert "CRITICAL EMERGENCY" in warning
        assert "SUGGESTED STRATEGIES TO BREAK FREE" in warning

    def test_stuck_warning_shows_correct_countdown(self, orchestrator_with_game):
        """Test that warning shows correct turns until termination."""
        orch = orchestrator_with_game

        # Test various countdown values
        test_cases = [
            (20, 20),  # 20 turns stuck, 20 until death
            (25, 15),  # 25 turns stuck, 15 until death
            (30, 10),  # 30 turns stuck, 10 until death
            (35, 5),   # 35 turns stuck, 5 until death
            (38, 2),   # 38 turns stuck, 2 until death
        ]

        for turns_stuck, expected_countdown in test_cases:
            orch._last_score_change_turn = 0
            orch.game_state.turn_count = turns_stuck
            orch._last_tracked_score = 0

            warning = orch._build_stuck_countdown_warning()
            assert f"DIE in {expected_countdown} turns" in warning

    def test_stuck_warning_disabled_by_config(self, orchestrator_with_game):
        """Test that warnings can be disabled via configuration."""
        orch = orchestrator_with_game

        # Disable warnings
        orch.config.enable_stuck_warnings = False

        # Simulate being stuck
        orch._last_score_change_turn = 0
        orch.game_state.turn_count = 30
        orch._last_tracked_score = 0

        warning = orch._build_stuck_countdown_warning()
        assert warning == ""


class TestExplorationHints:
    """Test exploration hint generation."""

    def test_exploration_hints_disabled_by_config(self, orchestrator_with_game):
        """Test that hints can be disabled via configuration."""
        orch = orchestrator_with_game
        orch.config.enable_exploration_hints = False

        novelty_info = {"is_novel": False, "recent_actions": 3, "window_size": 3}
        unexplored_info = {"has_unexplored": True, "unexplored_exits": ["north"], "all_exits": ["north", "south"]}

        hints = orch._build_exploration_hints("look", novelty_info, unexplored_info)
        assert hints == ""

    def test_exploration_hints_for_novel_actions(self, orchestrator_with_game):
        """Test that no hints are shown for novel actions."""
        orch = orchestrator_with_game

        novelty_info = {"is_novel": True, "recent_actions": 5, "window_size": 5}
        unexplored_info = {"has_unexplored": False, "unexplored_exits": [], "all_exits": ["north"]}

        hints = orch._build_exploration_hints("examine mailbox", novelty_info, unexplored_info)
        assert hints == ""  # No hints for novel action without unexplored exits

    def test_exploration_hints_for_repeated_actions(self, orchestrator_with_game):
        """Test hints for repeated actions."""
        orch = orchestrator_with_game

        novelty_info = {"is_novel": False, "recent_actions": 5, "window_size": 5}
        unexplored_info = {"has_unexplored": False, "unexplored_exits": [], "all_exits": []}

        hints = orch._build_exploration_hints("look", novelty_info, unexplored_info)
        assert "recently tried 'look'" in hints
        assert "Consider trying something different" in hints

    def test_exploration_hints_for_unexplored_exits(self, orchestrator_with_game):
        """Test hints for unexplored exits."""
        orch = orchestrator_with_game

        novelty_info = {"is_novel": True, "recent_actions": 0, "window_size": 0}
        unexplored_info = {
            "has_unexplored": True,
            "unexplored_exits": ["north", "west"],
            "all_exits": ["north", "south", "west"]
        }

        hints = orch._build_exploration_hints("examine", novelty_info, unexplored_info)
        assert "unexplored exits" in hints
        assert "north" in hints
        assert "west" in hints

    def test_exploration_hints_combined(self, orchestrator_with_game):
        """Test combined hints for both novelty and unexplored exits."""
        orch = orchestrator_with_game

        novelty_info = {"is_novel": False, "recent_actions": 5, "window_size": 5}
        unexplored_info = {
            "has_unexplored": True,
            "unexplored_exits": ["east"],
            "all_exits": ["north", "east"]
        }

        hints = orch._build_exploration_hints("look", novelty_info, unexplored_info)
        assert "recently tried" in hints
        assert "unexplored exits" in hints
        assert "east" in hints


class TestConfigurationValidation:
    """Test configuration validation for exploration guidance."""

    def test_configuration_warning_threshold_validation(self):
        """Test that warning threshold must be less than max_turns_stuck."""
        # Valid configuration
        config_dict = {
            "max_turns_per_episode": 100,
            "max_turns_stuck": 40,
            "stuck_warning_threshold": 20,
        }
        config = GameConfiguration.model_validate(config_dict)
        assert config.stuck_warning_threshold == 20

        # Invalid configuration (threshold >= max_turns_stuck)
        invalid_config_dict = {
            "max_turns_per_episode": 100,
            "max_turns_stuck": 40,
            "stuck_warning_threshold": 40,  # Equal to max_turns_stuck
        }
        with pytest.raises(ValueError, match="stuck_warning_threshold.*must be <"):
            GameConfiguration.model_validate(invalid_config_dict)

        # Invalid configuration (threshold > max_turns_stuck)
        invalid_config_dict2 = {
            "max_turns_per_episode": 100,
            "max_turns_stuck": 40,
            "stuck_warning_threshold": 50,  # Greater than max_turns_stuck
        }
        with pytest.raises(ValueError, match="stuck_warning_threshold.*must be <"):
            GameConfiguration.model_validate(invalid_config_dict2)

    def test_action_novelty_window_bounds(self):
        """Test action novelty window respects configured bounds."""
        config_dict = {
            "max_turns_per_episode": 100,
            "action_novelty_window": 15,
        }
        config = GameConfiguration.model_validate(config_dict)
        assert config.action_novelty_window == 15

        # Test minimum bound
        config_dict_min = {
            "max_turns_per_episode": 100,
            "action_novelty_window": 5,
        }
        config_min = GameConfiguration.model_validate(config_dict_min)
        assert config_min.action_novelty_window == 5

        # Test maximum bound
        config_dict_max = {
            "max_turns_per_episode": 100,
            "action_novelty_window": 50,
        }
        config_max = GameConfiguration.model_validate(config_dict_max)
        assert config_max.action_novelty_window == 50


class TestIntegration:
    """Integration tests for exploration guidance in actual flow."""

    def test_stuck_warning_appears_in_agent_context(self, orchestrator_with_game):
        """Test that stuck warnings actually appear in agent context."""
        orch = orchestrator_with_game

        # Setup: Agent stuck for 20 turns
        orch.config.stuck_warning_threshold = 20
        orch._last_score_change_turn = 0
        orch.game_state.turn_count = 20
        orch.game_state.previous_zork_score = 10
        orch._last_tracked_score = 10

        # Build warning
        stuck_warning = orch._build_stuck_countdown_warning()

        # Verify warning is generated
        assert stuck_warning != ""
        assert "DIE in" in stuck_warning
        assert "20 turns" in stuck_warning  # Turns until death
        assert "NO SCORE PROGRESS for 20 turns" in stuck_warning

    def test_exploration_hints_are_generated(self, orchestrator_with_game):
        """Test that exploration hints are properly generated."""
        orch = orchestrator_with_game

        # Setup: Agent proposes repeated action
        from collections import deque
        max_window = max(orch.config.action_novelty_window, 20)
        orch._action_history = deque(['look', 'examine lamp', 'take lamp'], maxlen=max_window)
        proposed_action = "examine lamp"

        # Start game to initialize map
        orch.jericho_interface.start()
        location_obj = orch.jericho_interface.get_location_structured()
        location_id = location_obj.num

        # Add room with unexplored exit
        orch.map_manager.game_map.add_room(location_id, "Test Room")
        orch.map_manager.game_map.add_room(999, "Unexplored Room")
        orch.map_manager.game_map.add_connection(location_id, "north", 999)
        del orch.map_manager.game_map.rooms[999]  # Make it unexplored
        orch.game_state.current_room_id = location_id

        # Detect and build hints
        novelty_info = orch._detect_action_novelty(proposed_action)
        unexplored_info = orch._detect_unexplored_exits()
        hints = orch._build_exploration_hints(
            proposed_action, novelty_info, unexplored_info
        )

        # Verify hints are generated
        assert hints != ""
        assert "recently tried" in hints.lower()
        assert "unexplored exits" in hints.lower()

    def test_hints_integrated_into_critic_evaluation(self, orchestrator_with_game):
        """Test that hints are passed to critic during evaluation."""
        orch = orchestrator_with_game

        # Mock critic evaluation to capture the game_state_text passed to it
        from unittest.mock import Mock
        from zork_critic import CriticResponse

        # Setup: Agent proposes repeated action
        from collections import deque
        max_window = max(orch.config.action_novelty_window, 20)
        orch._action_history = deque(['look', 'examine lamp', 'take lamp'], maxlen=max_window)
        proposed_action = "look"

        # Start game
        orch.jericho_interface.start()
        location_obj = orch.jericho_interface.get_location_structured()
        location_id = location_obj.num

        # Add room with unexplored exit
        orch.map_manager.game_map.add_room(location_id, "Test Room")
        orch.map_manager.game_map.add_room(999, "Unexplored Room")
        orch.map_manager.game_map.add_connection(location_id, "north", 999)
        del orch.map_manager.game_map.rooms[999]  # Make it unexplored
        orch.game_state.current_room_id = location_id
        orch.game_state.current_room_name_for_map = "Test Room"

        # Track what game_state_text is passed to critic
        captured_game_state_text = []

        def mock_evaluate_action(**kwargs):
            captured_game_state_text.append(kwargs.get('game_state_text', ''))
            return CriticResponse(
                score=0.8,
                justification="Test evaluation",
                confidence=0.8
            )

        orch.critic.evaluate_action = Mock(side_effect=mock_evaluate_action)

        # Execute critic evaluation loop
        current_state = "You are in a test room."
        agent_context = {}
        formatted_context = "Test context"

        # Call the critic evaluation loop
        try:
            action_to_take, _, _, _, _, _, _ = orch._execute_critic_evaluation_loop(
                current_state=current_state,
                proposed_action=proposed_action,
                agent_context=agent_context,
                formatted_context=formatted_context,
            )
        except Exception as e:
            # Log any errors for debugging
            print(f"Error during critic evaluation: {e}")
            raise

        # Verify hints were passed to critic
        assert len(captured_game_state_text) > 0
        game_state_with_hints = captured_game_state_text[0]

        # Check that exploration hints were appended
        # Hints should mention novelty and unexplored exits
        assert "recently tried" in game_state_with_hints.lower() or \
               "unexplored" in game_state_with_hints.lower(), \
               f"Exploration hints not found in critic context: {game_state_with_hints}"

    def test_deque_sliding_window_efficiency(self, orchestrator_with_game):
        """Test that deque is used for efficient sliding windows."""
        orch = orchestrator_with_game
        from collections import deque

        # Track 100 actions
        for i in range(100):
            orch._track_action_history(f"action_{i}")

        # Verify deque is used
        assert isinstance(orch._action_history, deque)

        # Verify maxlen is set correctly
        max_window = max(orch.config.action_novelty_window, 20)
        assert orch._action_history.maxlen == max_window

        # Verify only max_window items are kept
        assert len(orch._action_history) == max_window

    def test_location_tracking_uses_deque(self, orchestrator_with_game):
        """Test that location tracking uses deque for efficiency."""
        orch = orchestrator_with_game
        from collections import deque

        # Track multiple locations
        for i in range(30):
            orch._track_location_history()

        # Verify deque is used
        assert isinstance(orch._location_id_history, deque)

        # Verify maxlen is set to 20
        assert orch._location_id_history.maxlen == 20

        # Verify only 20 items are kept
        assert len(orch._location_id_history) == 20
