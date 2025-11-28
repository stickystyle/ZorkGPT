# ABOUTME: Tests for add_memory() persistence routing (Phase 3.1 of ephemeral memory system)
# ABOUTME: Verifies routing logic based on persistence level (ephemeral vs core/permanent)

import pytest
from pathlib import Path
from managers.simple_memory_manager import SimpleMemoryManager
from managers.memory.models import Memory, MemoryStatus
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class TestMemoryRouting:
    """Test suite for add_memory() routing by persistence level."""

    def test_ephemeral_memory_not_written_to_file(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that ephemeral memories are NOT written to Memories.md.

        Test approach:
        1. Create SimpleMemoryManager with clean temp directory
        2. Create ephemeral memory
        3. Call add_memory() with ephemeral memory
        4. Verify memory added to ephemeral_cache
        5. Verify memory NOT in persistent memory_cache
        6. Verify Memories.md file NOT created (no file write occurred)
        """
        # Arrange
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        ephemeral_memory = Memory(
            category="NOTE",
            title="Temporary observation",
            episode=1,
            turns="10",
            score_change=None,
            text="This is a temporary observation that should not persist.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )

        memories_file = Path(game_config.zork_game_workdir) / "Memories.md"

        # Verify file does not exist before test
        assert not memories_file.exists(), "Memories.md should not exist initially"

        # Act
        success = manager.add_memory(
            location_id=5,
            location_name="Kitchen",
            memory=ephemeral_memory
        )

        # Assert
        assert success is True, "add_memory() should return True for ephemeral memory"

        # Verify memory in ephemeral cache
        assert 5 in manager.ephemeral_cache, "Location 5 should be in ephemeral_cache"
        assert len(manager.ephemeral_cache[5]) == 1, "Should have exactly 1 ephemeral memory"
        cached_memory = manager.ephemeral_cache[5][0]
        assert cached_memory.title == "Temporary observation"
        assert cached_memory.persistence == "ephemeral"

        # Verify memory NOT in persistent cache
        assert 5 not in manager.memory_cache, "Location 5 should NOT be in memory_cache"

        # Verify file NOT created (no file write occurred)
        assert not memories_file.exists(), (
            "Memories.md should NOT be created for ephemeral memories"
        )

    def test_core_memory_written_to_file_and_cache(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that core memories are written to file AND added to cache.

        Test approach:
        1. Create SimpleMemoryManager with clean temp directory
        2. Create core memory
        3. Call add_memory() with core memory
        4. Verify memory added to memory_cache
        5. Verify memory NOT in ephemeral_cache
        6. Verify Memories.md file created and contains memory
        """
        # Arrange
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        core_memory = Memory(
            category="SUCCESS",
            title="Fundamental game mechanic",
            episode=1,
            turns="15",
            score_change=5,
            text="Opening doors requires keys. Core game mechanic.",
            persistence="core",
            status=MemoryStatus.ACTIVE
        )

        memories_file = Path(game_config.zork_game_workdir) / "Memories.md"

        # Act
        success = manager.add_memory(
            location_id=10,
            location_name="Hallway",
            memory=core_memory
        )

        # Assert
        assert success is True, "add_memory() should return True for core memory"

        # Verify memory in persistent cache
        assert 10 in manager.memory_cache, "Location 10 should be in memory_cache"
        assert len(manager.memory_cache[10]) == 1, "Should have exactly 1 core memory"
        cached_memory = manager.memory_cache[10][0]
        assert cached_memory.title == "Fundamental game mechanic"
        assert cached_memory.persistence == "core"

        # Verify memory NOT in ephemeral cache
        assert 10 not in manager.ephemeral_cache, (
            "Location 10 should NOT be in ephemeral_cache"
        )

        # Verify file created and contains memory
        assert memories_file.exists(), "Memories.md should be created for core memory"

        file_content = memories_file.read_text(encoding="utf-8")
        assert "## Location 10: Hallway" in file_content, (
            "File should contain location header"
        )
        assert "[SUCCESS - CORE] Fundamental game mechanic" in file_content, (
            "File should contain memory title with category and CORE marker"
        )
        assert "Ep1, T15, +5" in file_content, (
            "File should contain episode, turn, and score change"
        )
        assert "Opening doors requires keys" in file_content, (
            "File should contain memory text"
        )

    def test_permanent_memory_written_to_file_and_cache(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that permanent memories are written to file AND added to cache.

        Test approach:
        1. Create SimpleMemoryManager with clean temp directory
        2. Create permanent memory
        3. Call add_memory() with permanent memory
        4. Verify memory added to memory_cache
        5. Verify memory NOT in ephemeral_cache
        6. Verify Memories.md file created and contains memory
        """
        # Arrange
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        permanent_memory = Memory(
            category="DISCOVERY",
            title="Hidden treasure location",
            episode=2,
            turns="45-48",
            score_change=10,
            text="Secret room contains valuable treasure. Remember this location.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE
        )

        memories_file = Path(game_config.zork_game_workdir) / "Memories.md"

        # Act
        success = manager.add_memory(
            location_id=25,
            location_name="Secret Chamber",
            memory=permanent_memory
        )

        # Assert
        assert success is True, "add_memory() should return True for permanent memory"

        # Verify memory in persistent cache
        assert 25 in manager.memory_cache, "Location 25 should be in memory_cache"
        assert len(manager.memory_cache[25]) == 1, "Should have exactly 1 permanent memory"
        cached_memory = manager.memory_cache[25][0]
        assert cached_memory.title == "Hidden treasure location"
        assert cached_memory.persistence == "permanent"

        # Verify memory NOT in ephemeral cache
        assert 25 not in manager.ephemeral_cache, (
            "Location 25 should NOT be in ephemeral_cache"
        )

        # Verify file created and contains memory
        assert memories_file.exists(), "Memories.md should be created for permanent memory"

        file_content = memories_file.read_text(encoding="utf-8")
        assert "## Location 25: Secret Chamber" in file_content, (
            "File should contain location header"
        )
        assert "[DISCOVERY - PERMANENT] Hidden treasure location" in file_content, (
            "File should contain memory title with category and PERMANENT marker"
        )
        assert "Ep2, T45-48, +10" in file_content, (
            "File should contain episode, turn range, and score change"
        )
        assert "Secret room contains valuable treasure" in file_content, (
            "File should contain memory text"
        )

    def test_ephemeral_logging_mentions_in_memory(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that ephemeral memory logging mentions 'in-memory only'.

        Test approach:
        1. Create SimpleMemoryManager with mock logger
        2. Create ephemeral memory
        3. Call add_memory() with ephemeral memory
        4. Verify log_info called with message containing 'in-memory only'
        5. Verify log message includes 'ephemeral' persistence level
        """
        # Arrange
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        ephemeral_memory = Memory(
            category="NOTE",
            title="Quick note",
            episode=1,
            turns="5",
            score_change=None,
            text="Ephemeral observation.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )

        # Act
        manager.add_memory(
            location_id=8,
            location_name="Forest",
            memory=ephemeral_memory
        )

        # Assert - Check that log_info was called with correct message
        assert mock_logger.info.called, "log_info should be called"

        # Get the log message from the call
        log_calls = mock_logger.info.call_args_list
        assert len(log_calls) > 0, "Should have at least one log_info call"

        # Find the log call with the memory addition message
        memory_log_found = False
        for call in log_calls:
            log_message = call[0][0]  # First positional argument is the message
            if "ephemeral memory" in log_message.lower() and "in-memory only" in log_message.lower():
                memory_log_found = True
                # Verify message mentions the memory title
                assert "Quick note" in log_message, (
                    f"Log message should mention memory title, got: {log_message}"
                )
                break

        assert memory_log_found, (
            "Should have log message mentioning 'ephemeral memory' and 'in-memory only'"
        )

    def test_persistent_logging_mentions_file(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that core/permanent memory logging mentions 'to file'.

        Test approach:
        1. Create SimpleMemoryManager with mock logger
        2. Create core memory
        3. Call add_memory() with core memory
        4. Verify log_info called with message containing 'to file'
        5. Verify log message includes persistence level
        """
        # Arrange
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        core_memory = Memory(
            category="SUCCESS",
            title="Important discovery",
            episode=1,
            turns="20",
            score_change=5,
            text="Core game mechanic discovered.",
            persistence="core",
            status=MemoryStatus.ACTIVE
        )

        # Act
        manager.add_memory(
            location_id=12,
            location_name="Library",
            memory=core_memory
        )

        # Assert - Check that log_info was called with correct message
        assert mock_logger.info.called, "log_info should be called"

        # Get the log message from the call
        log_calls = mock_logger.info.call_args_list
        assert len(log_calls) > 0, "Should have at least one log_info call"

        # Find the log call with the memory addition message
        memory_log_found = False
        for call in log_calls:
            log_message = call[0][0]  # First positional argument is the message
            if "core" in log_message.lower() and "memory to file" in log_message.lower():
                memory_log_found = True
                # Verify message mentions the memory title
                assert "Important discovery" in log_message, (
                    f"Log message should mention memory title, got: {log_message}"
                )
                break

        assert memory_log_found, (
            "Should have log message mentioning persistence level and 'to file'"
        )

    def test_multiple_ephemeral_memories_same_location(
        self, mock_logger, game_config, game_state
    ):
        """
        Test that multiple ephemeral memories at same location are cached correctly.

        Test approach:
        1. Create SimpleMemoryManager with clean temp directory
        2. Add 3 ephemeral memories to location 5
        3. Verify ephemeral_cache[5] has 3 memories
        4. Verify all memories have correct data
        5. Verify file NOT modified (remains non-existent)
        """
        # Arrange
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        ephemeral_memories = [
            Memory(
                category="NOTE",
                title="First observation",
                episode=1,
                turns="10",
                score_change=None,
                text="First ephemeral note.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            ),
            Memory(
                category="NOTE",
                title="Second observation",
                episode=1,
                turns="11",
                score_change=None,
                text="Second ephemeral note.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            ),
            Memory(
                category="NOTE",
                title="Third observation",
                episode=1,
                turns="12",
                score_change=None,
                text="Third ephemeral note.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )
        ]

        memories_file = Path(game_config.zork_game_workdir) / "Memories.md"

        # Act - Add all three ephemeral memories
        for memory in ephemeral_memories:
            success = manager.add_memory(
                location_id=5,
                location_name="Kitchen",
                memory=memory
            )
            assert success is True, f"add_memory() should succeed for {memory.title}"

        # Assert
        # Verify ephemeral cache has all 3 memories
        assert 5 in manager.ephemeral_cache, "Location 5 should be in ephemeral_cache"
        assert len(manager.ephemeral_cache[5]) == 3, (
            f"Should have exactly 3 ephemeral memories, got {len(manager.ephemeral_cache[5])}"
        )

        # Verify all memories are stored correctly
        cached_titles = [m.title for m in manager.ephemeral_cache[5]]
        assert "First observation" in cached_titles, "Should contain first memory"
        assert "Second observation" in cached_titles, "Should contain second memory"
        assert "Third observation" in cached_titles, "Should contain third memory"

        # Verify all cached memories have ephemeral persistence
        for cached_memory in manager.ephemeral_cache[5]:
            assert cached_memory.persistence == "ephemeral", (
                f"Memory '{cached_memory.title}' should have ephemeral persistence"
            )

        # Verify location NOT in persistent cache
        assert 5 not in manager.memory_cache, (
            "Location 5 should NOT be in memory_cache for ephemeral memories"
        )

        # Verify file NOT created (no file writes occurred)
        assert not memories_file.exists(), (
            "Memories.md should NOT be created for ephemeral memories"
        )
