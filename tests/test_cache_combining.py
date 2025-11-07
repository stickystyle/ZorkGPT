# ABOUTME: Tests for get_location_memory() cache combining (Phase 3.2 of ephemeral memory system)
# ABOUTME: Verifies that get_location_memory() combines both persistent and ephemeral caches

import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus
from session.game_state import GameState
from session.game_configuration import GameConfiguration


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock(spec=logging.Logger)


@pytest.fixture
def game_config(tmp_path):
    """Create a test game configuration with temporary work directory."""
    return GameConfiguration(
        max_turns_per_episode=1000,
        turn_delay_seconds=0.0,
        game_file_path="test_game.z5",
        critic_rejection_threshold=0.5,
        episode_log_file="test_episode.log",
        json_log_file="test_episode.jsonl",
        state_export_file="test_state.json",
        map_state_file="test_map_state.json",
        zork_game_workdir=str(tmp_path),
        client_base_url="http://localhost:1234",
        client_api_key="test_api_key",
        agent_model="test-agent-model",
        critic_model="test-critic-model",
        info_ext_model="test-extractor-model",
        analysis_model="test-analysis-model",
        memory_model="test-memory-model",
        knowledge_update_interval=100,
        objective_update_interval=20,
        enable_objective_refinement=True,
        objective_refinement_interval=200,
        max_objectives_before_forced_refinement=15,
        refined_objectives_target_count=10,
        max_context_tokens=100000,
        context_overflow_threshold=0.8,
        enable_state_export=True,
        s3_bucket="test-bucket",
        s3_key_prefix="test/",
        simple_memory_file="Memories.md",
        simple_memory_max_shown=10,
        knowledge_file="test_knowledgebase.md",
        agent_sampling={},
        critic_sampling={},
        extractor_sampling={},
        analysis_sampling={},
        memory_sampling={'temperature': 0.3, 'max_tokens': 1000},
    )


@pytest.fixture
def game_state():
    """Create a test game state."""
    state = GameState()
    state.episode_id = "test_episode_001"
    state.turn_count = 10
    state.current_room_name_for_map = "Living Room"
    state.previous_zork_score = 50
    state.current_inventory = ["lamp", "sword"]
    return state


class TestCacheCombining:
    """Test suite for get_location_memory() combining both caches."""

    def test_get_location_memory_returns_persistent_only(
        self, mock_logger, game_config, game_state, tmp_path
    ):
        """
        Test that get_location_memory() returns persistent memories when only persistent exist.

        Validates baseline behavior (no regression) - should pass with current implementation.

        Test approach:
        1. Create SimpleMemoryManager
        2. Add 2 permanent memories to location 10
        3. Call get_location_memory(10)
        4. Should return both memories
        5. Validates existing behavior still works
        """
        # Arrange - Use real temp directory
        game_config.zork_game_workdir = str(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Add 2 permanent memories to location 10
        memory1 = Memory(
            category="SUCCESS",
            title="Door can be opened",
            episode=1,
            turns="10",
            score_change=None,
            text="The door opens when unlocked.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE
        )
        memory2 = Memory(
            category="DISCOVERY",
            title="Key found in mailbox",
            episode=1,
            turns="12",
            score_change=None,
            text="The brass key is in the mailbox.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE
        )

        # Add memories with real file operations
        success1 = manager.add_memory(10, "Test Room", memory1)
        success2 = manager.add_memory(10, "Test Room", memory2)

        # Validate setup succeeded
        assert success1, "First memory should be added successfully"
        assert success2, "Second memory should be added successfully"

        # Act
        result = manager.get_location_memory(10)

        # Assert
        assert result != "", "Should return non-empty string for location with memories"
        assert "Door can be opened" in result, "Should include first permanent memory"
        assert "Key found in mailbox" in result, "Should include second permanent memory"
        assert len([line for line in result.split("\n") if line.startswith("[")]) == 2, \
            "Should have 2 memory entries"

    def test_get_location_memory_returns_ephemeral_only(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that get_location_memory() returns ephemeral memories when only ephemeral exist.

        EXPECTED TO FAIL - validates new ephemeral retrieval behavior.

        Test approach:
        1. Create SimpleMemoryManager
        2. Add 2 ephemeral memories to location 20
        3. Call get_location_memory(20)
        4. Should return both ephemeral memories
        5. Validates new cache retrieval logic
        """
        # Arrange
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Add 2 ephemeral memories to location 20
        memory1 = Memory(
            category="NOTE",
            title="Current room seems safe",
            episode=1,
            turns="5",
            score_change=None,
            text="No obvious threats detected on first pass.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        memory2 = Memory(
            category="NOTE",
            title="Temporary observation",
            episode=1,
            turns="6",
            score_change=None,
            text="Strange noise heard from north.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )

        manager.add_memory(20, "Dark Room", memory1)
        manager.add_memory(20, "Dark Room", memory2)

        # Act
        result = manager.get_location_memory(20)

        # Assert
        assert result != "", "Should return non-empty string for location with ephemeral memories"
        assert "Current room seems safe" in result, "Should include first ephemeral memory"
        assert "Temporary observation" in result, "Should include second ephemeral memory"
        assert len([line for line in result.split("\n") if line.startswith("[")]) == 2, \
            "Should have 2 memory entries"

    def test_get_location_memory_combines_both_caches(
        self, mock_logger, game_config, game_state, tmp_path
    ):
        """
        Test that get_location_memory() combines persistent and ephemeral memories.

        EXPECTED TO FAIL - main test for cache combining feature.

        Test approach:
        1. Create SimpleMemoryManager
        2. Add 2 permanent memories to location 30
        3. Add 2 ephemeral memories to same location 30
        4. Call get_location_memory(30)
        5. Should return all 4 memories (2 persistent + 2 ephemeral)
        6. Main validation of cache combining
        """
        # Arrange - Use real temp directory
        game_config.zork_game_workdir = str(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Add 2 permanent memories
        permanent1 = Memory(
            category="SUCCESS",
            title="Window can be entered",
            episode=1,
            turns="15",
            score_change=None,
            text="To enter kitchen: (1) open window, (2) enter window.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE
        )
        permanent2 = Memory(
            category="DANGER",
            title="Grue attacks in darkness",
            episode=1,
            turns="20",
            score_change=None,
            text="Grue will attack if in dark room without light.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE
        )

        # Add 2 ephemeral memories
        ephemeral1 = Memory(
            category="NOTE",
            title="Troll seems distracted",
            episode=1,
            turns="8",
            score_change=None,
            text="Troll is currently looking away from path.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        ephemeral2 = Memory(
            category="NOTE",
            title="Heard sound from east",
            episode=1,
            turns="9",
            score_change=None,
            text="Strange scratching noise coming from east exit.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )

        # Add memories with real file operations
        success1 = manager.add_memory(30, "Troll Room", permanent1)
        success2 = manager.add_memory(30, "Troll Room", permanent2)
        success3 = manager.add_memory(30, "Troll Room", ephemeral1)
        success4 = manager.add_memory(30, "Troll Room", ephemeral2)

        # Validate setup succeeded
        assert success1, "First permanent memory should be added successfully"
        assert success2, "Second permanent memory should be added successfully"
        assert success3, "First ephemeral memory should be added successfully"
        assert success4, "Second ephemeral memory should be added successfully"

        # Act
        result = manager.get_location_memory(30)

        # Assert
        assert result != "", "Should return non-empty string for location with memories"
        # Check all 4 memories are present
        assert "Window can be entered" in result, "Should include first permanent memory"
        assert "Grue attacks in darkness" in result, "Should include second permanent memory"
        assert "Troll seems distracted" in result, "Should include first ephemeral memory"
        assert "Heard sound from east" in result, "Should include second ephemeral memory"

        # Count memory entries
        memory_lines = [line for line in result.split("\n") if line.startswith("[")]
        assert len(memory_lines) == 4, \
            f"Should have 4 memory entries (2 persistent + 2 ephemeral), got {len(memory_lines)}"

    def test_get_location_memory_separates_locations(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that get_location_memory() separates memories by location.

        EXPECTED TO FAIL - validates location filtering works with both caches.

        Test approach:
        1. Create SimpleMemoryManager
        2. Add permanent memory to location 10
        3. Add ephemeral memory to location 20
        4. Call get_location_memory(10)
        5. Should return only location 10 memory
        6. Validates location filtering still works correctly
        """
        # Arrange
        with patch("builtins.open", create=True), \
             patch("managers.simple_memory_manager.FileLock", MagicMock()):

            manager = SimpleMemoryManager(
                logger=mock_logger,
                config=game_config,
                game_state=game_state
            )

            # Add permanent memory to location 10
            permanent_mem = Memory(
                category="SUCCESS",
                title="Location 10 memory",
                episode=1,
                turns="10",
                score_change=None,
                text="This is at location 10.",
                persistence="permanent",
                status=MemoryStatus.ACTIVE
            )

            # Add ephemeral memory to location 20
            ephemeral_mem = Memory(
                category="NOTE",
                title="Location 20 memory",
                episode=1,
                turns="11",
                score_change=None,
                text="This is at location 20.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )

            manager.add_memory(10, "Room Ten", permanent_mem)
            manager.add_memory(20, "Room Twenty", ephemeral_mem)

            # Act
            result = manager.get_location_memory(10)

            # Assert
            assert result != "", "Should return non-empty string for location 10"
            assert "Location 10 memory" in result, "Should include location 10 memory"
            assert "Location 20 memory" not in result, "Should NOT include location 20 memory"
            assert len([line for line in result.split("\n") if line.startswith("[")]) == 1, \
                "Should have exactly 1 memory entry"

    def test_get_location_memory_respects_status_filter(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that get_location_memory() respects status filtering across both caches.

        EXPECTED TO FAIL - validates status filtering works when combining caches.

        Test approach:
        1. Create SimpleMemoryManager
        2. Add ACTIVE permanent memory to location 40
        3. Add SUPERSEDED permanent memory to location 40
        4. Add ACTIVE ephemeral memory to location 40
        5. Add SUPERSEDED ephemeral memory to location 40
        6. Call get_location_memory(40)
        7. Should return only ACTIVE memories from both caches
        8. Validates filtering works across both caches
        """
        # Arrange
        with patch("builtins.open", create=True), \
             patch("managers.simple_memory_manager.FileLock", MagicMock()):

            manager = SimpleMemoryManager(
                logger=mock_logger,
                config=game_config,
                game_state=game_state
            )

            # Add ACTIVE permanent memory
            active_permanent = Memory(
                category="SUCCESS",
                title="Active permanent memory",
                episode=1,
                turns="10",
                score_change=None,
                text="This permanent memory is active.",
                persistence="permanent",
                status=MemoryStatus.ACTIVE
            )

            # Add SUPERSEDED permanent memory
            superseded_permanent = Memory(
                category="FAILURE",
                title="Superseded permanent memory",
                episode=1,
                turns="11",
                score_change=None,
                text="This permanent memory was superseded.",
                persistence="permanent",
                status=MemoryStatus.SUPERSEDED,
                superseded_by="Better approach",
                superseded_at_turn=15
            )

            # Add ACTIVE ephemeral memory
            active_ephemeral = Memory(
                category="NOTE",
                title="Active ephemeral memory",
                episode=1,
                turns="12",
                score_change=None,
                text="This ephemeral memory is active.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )

            # Add SUPERSEDED ephemeral memory
            superseded_ephemeral = Memory(
                category="NOTE",
                title="Superseded ephemeral memory",
                episode=1,
                turns="13",
                score_change=None,
                text="This ephemeral memory was superseded.",
                persistence="ephemeral",
                status=MemoryStatus.SUPERSEDED,
                superseded_by="Better observation",
                superseded_at_turn=16
            )

            manager.add_memory(40, "Test Room", active_permanent)
            manager.add_memory(40, "Test Room", superseded_permanent)
            manager.add_memory(40, "Test Room", active_ephemeral)
            manager.add_memory(40, "Test Room", superseded_ephemeral)

            # Act
            result = manager.get_location_memory(40)

            # Assert
            assert result != "", "Should return non-empty string with active memories"
            # Check only ACTIVE memories are present
            assert "Active permanent memory" in result, "Should include active permanent memory"
            assert "Active ephemeral memory" in result, "Should include active ephemeral memory"
            # Check SUPERSEDED memories are NOT present
            assert "Superseded permanent memory" not in result, \
                "Should NOT include superseded permanent memory"
            assert "Superseded ephemeral memory" not in result, \
                "Should NOT include superseded ephemeral memory"
            # Should have exactly 2 entries (2 ACTIVE)
            memory_lines = [line for line in result.split("\n") if line.startswith("[")]
            assert len(memory_lines) == 2, \
                f"Should have 2 active memory entries, got {len(memory_lines)}"

    def test_get_location_memory_empty_location(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that get_location_memory() returns empty string for location with no memories.

        Validates edge case handling.

        Test approach:
        1. Create SimpleMemoryManager
        2. Call get_location_memory(99) for location with no memories
        3. Should return empty string
        4. Validates edge case
        """
        # Arrange
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Act
        result = manager.get_location_memory(99)

        # Assert
        assert result == "", "Should return empty string for location with no memories"

    def test_get_location_memory_after_episode_reset(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that get_location_memory() returns only persistent memories after reset.

        EXPECTED TO FAIL - validates reset behavior works with combined retrieval.

        Test approach:
        1. Create SimpleMemoryManager
        2. Add permanent memory to location 50
        3. Add ephemeral memory to location 50
        4. Call reset_episode()
        5. Call get_location_memory(50)
        6. Should return only permanent memory (ephemeral cleared)
        7. Validates reset behavior works correctly with combined retrieval
        """
        # Arrange
        with patch("builtins.open", create=True), \
             patch("managers.simple_memory_manager.FileLock", MagicMock()):

            manager = SimpleMemoryManager(
                logger=mock_logger,
                config=game_config,
                game_state=game_state
            )

            # Add permanent memory
            permanent_mem = Memory(
                category="SUCCESS",
                title="Permanent knowledge",
                episode=1,
                turns="10",
                score_change=None,
                text="This should survive reset.",
                persistence="permanent",
                status=MemoryStatus.ACTIVE
            )

            # Add ephemeral memory
            ephemeral_mem = Memory(
                category="NOTE",
                title="Ephemeral observation",
                episode=1,
                turns="11",
                score_change=None,
                text="This should be cleared on reset.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )

            manager.add_memory(50, "Reset Test Room", permanent_mem)
            manager.add_memory(50, "Reset Test Room", ephemeral_mem)

            # Verify both are present before reset
            result_before = manager.get_location_memory(50)
            assert "Permanent knowledge" in result_before
            assert "Ephemeral observation" in result_before

            # Act
            manager.reset_episode()

            # Get memories after reset
            result_after = manager.get_location_memory(50)

            # Assert
            assert result_after != "", "Should return non-empty string with permanent memory"
            assert "Permanent knowledge" in result_after, \
                "Should include permanent memory after reset"
            assert "Ephemeral observation" not in result_after, \
                "Should NOT include ephemeral memory after reset"
            # Should have exactly 1 entry (permanent only)
            memory_lines = [line for line in result_after.split("\n") if line.startswith("[")]
            assert len(memory_lines) == 1, \
                f"Should have 1 memory entry after reset, got {len(memory_lines)}"

    def test_get_location_memory_tentative_status_from_both_caches(
        self, mock_logger, game_config, game_state, tmp_path
    ):
        """
        Test that get_location_memory() properly formats TENTATIVE memories from both caches.

        EXPECTED TO FAIL - validates TENTATIVE status handling across both caches.

        Test approach:
        1. Create SimpleMemoryManager
        2. Add TENTATIVE permanent memory
        3. Add TENTATIVE ephemeral memory
        4. Add ACTIVE permanent memory for comparison
        5. Call get_location_memory()
        6. Should return all memories with proper formatting
        7. TENTATIVE memories should be grouped and marked
        """
        # Arrange - Use real temp directory
        game_config.zork_game_workdir = str(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Add ACTIVE permanent memory
        active_mem = Memory(
            category="SUCCESS",
            title="Confirmed fact",
            episode=1,
            turns="10",
            score_change=None,
            text="This is confirmed.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE
        )

        # Add TENTATIVE permanent memory
        tentative_permanent = Memory(
            category="NOTE",
            title="Uncertain permanent",
            episode=1,
            turns="11",
            score_change=None,
            text="This permanent memory is tentative.",
            persistence="permanent",
            status=MemoryStatus.TENTATIVE
        )

        # Add TENTATIVE ephemeral memory
        tentative_ephemeral = Memory(
            category="NOTE",
            title="Uncertain ephemeral",
            episode=1,
            turns="12",
            score_change=None,
            text="This ephemeral memory is tentative.",
            persistence="ephemeral",
            status=MemoryStatus.TENTATIVE
        )

        # Add memories with real file operations
        success1 = manager.add_memory(60, "Tentative Room", active_mem)
        success2 = manager.add_memory(60, "Tentative Room", tentative_permanent)
        success3 = manager.add_memory(60, "Tentative Room", tentative_ephemeral)

        # Validate setup succeeded
        assert success1, "Active memory should be added successfully"
        assert success2, "Tentative permanent memory should be added successfully"
        assert success3, "Tentative ephemeral memory should be added successfully"

        # Act
        result = manager.get_location_memory(60)

        # Assert
        assert result != "", "Should return non-empty string"
        # Check ACTIVE memory is present normally
        assert "Confirmed fact" in result, "Should include active memory"
        # Check both TENTATIVE memories are present
        assert "Uncertain permanent" in result, "Should include tentative permanent memory"
        assert "Uncertain ephemeral" in result, "Should include tentative ephemeral memory"
        # Check TENTATIVE section marker is present
        assert "TENTATIVE MEMORIES" in result, "Should have TENTATIVE section marker"
        # Should have 3 total memories
        memory_lines = [line for line in result.split("\n")
                      if line.strip().startswith("[") or "  [" in line]
        assert len(memory_lines) == 3, \
            f"Should have 3 memory entries (1 active + 2 tentative), got {len(memory_lines)}"
