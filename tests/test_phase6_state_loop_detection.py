# ABOUTME: Tests for Phase 6 state loop detection in StateManager
# ABOUTME: Validates state hash tracking and loop detection functionality

import pytest
from unittest.mock import Mock, MagicMock
from managers.state_manager import StateManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class TestPhase6StateLoopDetection:
    """Test state loop detection functionality (Phase 6)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return GameConfiguration(
            # Core game settings
            max_turns_per_episode=1000,
            turn_delay_seconds=0.0,
            game_file_path="test_game.z5",
            critic_rejection_threshold=0.5,
            # File paths
            episode_log_file="test_episode.log",
            json_log_file="test_episode.jsonl",
            state_export_file="test_state.json",
            zork_game_workdir="test_game_files",
            # LLM client settings
            client_base_url="http://localhost:1234",
            client_api_key="test_api_key",
            # Model specifications
            agent_model="test-agent-model",
            critic_model="test-critic-model",
            info_ext_model="test-extractor-model",
            analysis_model="test-analysis-model",
            memory_model="test-memory-model",
            condensation_model="test-condensation-model",
            # Update intervals
            knowledge_update_interval=100,
            objective_update_interval=20,
            # Objective refinement
            enable_objective_refinement=True,
            objective_refinement_interval=200,
            max_objectives_before_forced_refinement=15,
            refined_objectives_target_count=10,
            # Context management
            max_context_tokens=100000,
            context_overflow_threshold=0.8,
            # State export
            enable_state_export=False,  # Disable export for tests
            s3_bucket="test-bucket",
            s3_key_prefix="test/",
            # Simple Memory
            simple_memory_file="Memories.md",
            simple_memory_max_shown=10,
            map_state_file="test_map_state.json",
            # Sampling parameters
            agent_sampling={},
            critic_sampling={},
            extractor_sampling={},
            analysis_sampling={},
            memory_sampling={},
            condensation_sampling={},
        )

    @pytest.fixture
    def game_state(self):
        """Create test game state."""
        state = GameState()
        state.episode_id = "test-episode"
        state.turn_count = 1
        return state

    @pytest.fixture
    def logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def state_manager(self, logger, config, game_state):
        """Create StateManager instance."""
        return StateManager(logger, config, game_state, llm_client=None)

    @pytest.fixture
    def mock_jericho(self):
        """Create mock JerichoInterface."""
        mock = Mock()
        # Create distinct state tuples for testing
        mock.save_state.side_effect = [
            (1, 2, 3),  # State 1
            (4, 5, 6),  # State 2
            (1, 2, 3),  # State 1 again (loop!)
        ]
        return mock

    def test_state_hash_initialization(self, state_manager):
        """Test that state history is initialized empty."""
        assert state_manager.state_history == []
        assert state_manager.max_state_history_size == 1000

    def test_state_hash_reset_on_episode_reset(self, state_manager):
        """Test that state history is cleared on episode reset."""
        # Add some state hashes
        state_manager.state_history = [123, 456, 789]

        # Reset episode
        state_manager.reset_episode()

        # History should be cleared
        assert state_manager.state_history == []

    def test_track_state_hash_no_loop(self, state_manager):
        """Test tracking unique states (no loop detected)."""
        mock_jericho = Mock()
        mock_jericho.save_state.return_value = (1, 2, 3)

        # Track state (first time)
        result = state_manager.track_state_hash(mock_jericho)

        assert result is False  # No loop
        assert len(state_manager.state_history) == 1

    def test_track_state_hash_detects_loop(self, state_manager, game_state):
        """Test that identical states are detected as loops."""
        mock_jericho = Mock()
        state_tuple = (1, 2, 3)
        mock_jericho.save_state.return_value = state_tuple

        # Track state first time
        result1 = state_manager.track_state_hash(mock_jericho)
        assert result1 is False

        # Track same state again (loop!)
        result2 = state_manager.track_state_hash(mock_jericho)
        assert result2 is True  # Loop detected!

    def test_track_state_hash_multiple_unique_states(self, state_manager):
        """Test tracking multiple unique states."""
        mock_jericho = Mock()

        # Track 3 different states
        for i in range(3):
            mock_jericho.save_state.return_value = (i, i + 1, i + 2)
            result = state_manager.track_state_hash(mock_jericho)
            assert result is False  # No loop

        assert len(state_manager.state_history) == 3

    def test_state_history_size_limit(self, state_manager):
        """Test that state history respects max size limit."""
        mock_jericho = Mock()

        # Add more states than the limit
        for i in range(1500):
            mock_jericho.save_state.return_value = (i, i + 1, i + 2)
            state_manager.track_state_hash(mock_jericho)

        # Should not exceed max size
        assert len(state_manager.state_history) == 1000

    def test_track_state_hash_error_handling(self, state_manager):
        """Test graceful error handling when state access fails."""
        mock_jericho = Mock()
        mock_jericho.save_state.side_effect = Exception("State access failed")

        # Should return None on error
        result = state_manager.track_state_hash(mock_jericho)
        assert result is None

    def test_state_manager_status_includes_loop_detection(self, state_manager):
        """Test that status reporting includes loop detection info."""
        # Add some state hashes
        state_manager.state_history = [123, 456]

        status = state_manager.get_status()

        assert "state_history_size" in status
        assert status["state_history_size"] == 2
        assert status["loop_detection_enabled"] is True

    def test_track_state_hash_with_different_hash_collisions(self, state_manager):
        """Test that hash collisions are properly handled."""
        mock_jericho = Mock()

        # First state
        mock_jericho.save_state.return_value = (1, 2, 3)
        result1 = state_manager.track_state_hash(mock_jericho)
        assert result1 is False

        # Different state (will have different hash)
        mock_jericho.save_state.return_value = (3, 2, 1)
        result2 = state_manager.track_state_hash(mock_jericho)
        assert result2 is False

        # First state again (should detect loop)
        mock_jericho.save_state.return_value = (1, 2, 3)
        result3 = state_manager.track_state_hash(mock_jericho)
        assert result3 is True

    def test_track_state_hash_loop_index_calculation(self, state_manager, logger):
        """Test that loop index and turns are correctly calculated."""
        mock_jericho = Mock()

        # Add 5 unique states
        for i in range(5):
            mock_jericho.save_state.return_value = (i, i + 1, i + 2)
            state_manager.track_state_hash(mock_jericho)

        # Add the second state again (index 1)
        mock_jericho.save_state.return_value = (1, 2, 3)
        result = state_manager.track_state_hash(mock_jericho)

        assert result is True
        # Verify logger was called with loop detection
        logger.warning.assert_called()

    def test_state_history_maintains_order(self, state_manager):
        """Test that state history maintains insertion order."""
        import pickle
        mock_jericho = Mock()

        states = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        hashes = []

        for state in states:
            mock_jericho.save_state.return_value = state
            state_manager.track_state_hash(mock_jericho)
            # Compute hash the same way StateManager does: hash(pickle.dumps(state_tuple))
            hashes.append(hash(pickle.dumps(state)))

        # State history should match the order of hashes
        assert state_manager.state_history == hashes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
