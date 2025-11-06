"""
ABOUTME: Tests for supersession validation preventing ephemeral downgrades.
ABOUTME: Validates runtime checks that block permanent/core → ephemeral transitions.

This module tests the runtime validation added to supersede_memory() that blocks
invalid persistence level transitions that would cause data loss after episode reset.

Test Coverage:
1. Permanent → Ephemeral downgrade rejection
2. Core → Ephemeral downgrade rejection
3. Core → Permanent lateral move allowed
4. Original memory preservation when downgrade rejected
"""

import pytest
from pathlib import Path
from managers.simple_memory_manager import (
    SimpleMemoryManager,
    Memory,
    MemoryStatus
)


class TestSupersessionValidation:
    """Test supersession validation prevents ephemeral downgrades."""

    def test_supersede_permanent_to_ephemeral_rejected(
        self, mock_logger, game_config, game_state, tmp_path
    ):
        """
        Test that permanent → ephemeral downgrade is rejected.

        Validates that attempting to supersede a permanent memory with an ephemeral
        memory fails and logs a warning. The original permanent memory must remain
        unchanged (ACTIVE status) and the new ephemeral memory must not be added.

        Test approach:
        1. Add PERMANENT memory to location
        2. Attempt to supersede with EPHEMERAL memory
        3. Verify supersession rejected (returns False)
        4. Verify old PERMANENT memory remains ACTIVE
        5. Verify new EPHEMERAL memory NOT added to ephemeral_cache
        6. Verify warning logged with appropriate message
        """
        # Setup: Use real temp directory
        game_config.zork_game_workdir = str(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Add PERMANENT memory to location 10
        old_memory = Memory(
            category="NOTE",
            title="Door requires brass key",
            episode=1,
            turns="10",
            score_change=0,
            text="The wooden door requires the brass key to unlock.",
            persistence="permanent"
        )
        success = manager.add_memory(10, "Test Room", old_memory)
        assert success, "Failed to add permanent memory"

        # Verify old memory in memory_cache (persistent)
        assert 10 in manager.memory_cache, "Location should be in memory_cache"
        assert len(manager.memory_cache[10]) == 1, "Should have one permanent memory"
        assert manager.memory_cache[10][0].title == "Door requires brass key"
        assert manager.memory_cache[10][0].status == MemoryStatus.ACTIVE
        assert manager.memory_cache[10][0].persistence == "permanent"

        # Create EPHEMERAL memory (attempted downgrade)
        new_memory = Memory(
            category="NOTE",
            title="Door is open now",
            episode=1,
            turns="20",
            score_change=0,
            text="The door is currently open after I unlocked it.",
            persistence="ephemeral"
        )

        # Attempt to supersede (should fail)
        success = manager.supersede_memory(10, "Test Room", "Door requires brass key", new_memory)

        # Verify supersession rejected
        assert not success, "Supersession should be rejected for permanent → ephemeral downgrade"

        # Verify old PERMANENT memory remains ACTIVE (unchanged)
        old_mem = next((m for m in manager.memory_cache[10] if m.title == "Door requires brass key"), None)
        assert old_mem is not None, "Old permanent memory should still exist"
        assert old_mem.status == MemoryStatus.ACTIVE, "Old memory should remain ACTIVE (not superseded)"
        assert old_mem.superseded_by is None, "Old memory should not be marked as superseded"
        assert old_mem.superseded_at_turn is None, "Old memory should have no supersession turn"

        # Verify new EPHEMERAL memory NOT added to ephemeral_cache
        assert 10 not in manager.ephemeral_cache, "No ephemeral memories should be added"

        # Verify memory_cache unchanged (still has 1 permanent memory)
        assert len(manager.memory_cache[10]) == 1, "Should still have exactly one permanent memory"

        # Verify warning logged
        assert mock_logger.warning.called, "Should log warning for rejected downgrade"
        warning_call = mock_logger.warning.call_args
        assert "Cannot downgrade permanent memory to ephemeral" in str(warning_call), \
            "Warning should mention downgrade rejection"

    def test_supersede_core_to_ephemeral_rejected(
        self, mock_logger, game_config, game_state, tmp_path
    ):
        """
        Test that core → ephemeral downgrade is rejected.

        Validates that attempting to supersede a core memory (spawn state) with an
        ephemeral memory fails and logs a warning. Core memories represent permanent
        game state that resets each episode and cannot be replaced with temporary state.

        Test approach:
        1. Add CORE memory to location
        2. Attempt to supersede with EPHEMERAL memory
        3. Verify supersession rejected (returns False)
        4. Verify old CORE memory remains ACTIVE
        5. Verify new EPHEMERAL memory NOT added to ephemeral_cache
        6. Verify warning logged with appropriate message
        """
        # Setup
        game_config.zork_game_workdir = str(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Add CORE memory to location 10 (spawn state)
        old_memory = Memory(
            category="NOTE",
            title="Mailbox at spawn",
            episode=1,
            turns="1",
            score_change=0,
            text="The small mailbox is present at this location in spawn state.",
            persistence="core"
        )
        success = manager.add_memory(10, "Test Room", old_memory)
        assert success, "Failed to add core memory"

        # Verify old memory in memory_cache
        assert 10 in manager.memory_cache, "Location should be in memory_cache"
        assert len(manager.memory_cache[10]) == 1, "Should have one core memory"
        assert manager.memory_cache[10][0].title == "Mailbox at spawn"
        assert manager.memory_cache[10][0].status == MemoryStatus.ACTIVE
        assert manager.memory_cache[10][0].persistence == "core"

        # Create EPHEMERAL memory (attempted downgrade)
        new_memory = Memory(
            category="NOTE",
            title="Mailbox moved",
            episode=1,
            turns="30",
            score_change=0,
            text="I moved the mailbox to a different location.",
            persistence="ephemeral"
        )

        # Attempt to supersede (should fail)
        success = manager.supersede_memory(10, "Test Room", "Mailbox at spawn", new_memory)

        # Verify supersession rejected
        assert not success, "Supersession should be rejected for core → ephemeral downgrade"

        # Verify old CORE memory remains ACTIVE
        old_mem = next((m for m in manager.memory_cache[10] if m.title == "Mailbox at spawn"), None)
        assert old_mem is not None, "Old core memory should still exist"
        assert old_mem.status == MemoryStatus.ACTIVE, "Old memory should remain ACTIVE"
        assert old_mem.superseded_by is None, "Old memory should not be marked as superseded"
        assert old_mem.persistence == "core", "Old memory should still be core persistence"

        # Verify new EPHEMERAL memory NOT added
        assert 10 not in manager.ephemeral_cache, "No ephemeral memories should be added"

        # Verify memory_cache unchanged
        assert len(manager.memory_cache[10]) == 1, "Should still have exactly one core memory"

        # Verify warning logged with core-specific message
        assert mock_logger.warning.called, "Should log warning for rejected downgrade"
        warning_call = mock_logger.warning.call_args
        assert "Cannot downgrade core memory to ephemeral" in str(warning_call), \
            "Warning should mention core downgrade rejection"

    def test_supersede_core_to_permanent_allowed(
        self, mock_logger, game_config, game_state, tmp_path
    ):
        """
        Test that core → permanent lateral move is allowed (not a downgrade).

        Validates that superseding a core memory with a permanent memory succeeds
        because both are persistent (written to file). This is not a downgrade to
        ephemeral, so it should be permitted.

        Test approach:
        1. Add CORE memory to location
        2. Supersede with PERMANENT memory
        3. Verify supersession succeeds (returns True)
        4. Verify old CORE memory marked SUPERSEDED
        5. Verify new PERMANENT memory added to memory_cache
        6. Verify file contains both memories with correct status
        """
        # Setup
        game_config.zork_game_workdir = str(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Add CORE memory to location 10
        old_memory = Memory(
            category="NOTE",
            title="Troll at spawn location",
            episode=1,
            turns="5",
            score_change=0,
            text="The troll is present at north passage in spawn state.",
            persistence="core"
        )
        success = manager.add_memory(10, "Test Room", old_memory)
        assert success, "Failed to add core memory"

        # Verify old memory in memory_cache
        assert 10 in manager.memory_cache
        assert len(manager.memory_cache[10]) == 1
        old_mem_before = manager.memory_cache[10][0]
        assert old_mem_before.title == "Troll at spawn location"
        assert old_mem_before.status == MemoryStatus.ACTIVE
        assert old_mem_before.persistence == "core"

        # Create PERMANENT memory (lateral move, not downgrade)
        new_memory = Memory(
            category="NOTE",
            title="Troll behavior confirmed",
            episode=1,
            turns="25",
            score_change=0,
            text="The troll is always hostile and blocks passage permanently.",
            persistence="permanent"
        )

        # Supersede (should succeed)
        success = manager.supersede_memory(10, "Test Room", "Troll at spawn location", new_memory)

        # Verify supersession succeeded
        assert success, "Supersession should succeed for core → permanent (lateral move)"

        # Verify old CORE memory marked SUPERSEDED
        old_mem = next((m for m in manager.memory_cache[10] if m.title == "Troll at spawn location"), None)
        assert old_mem is not None, "Old core memory should still exist"
        assert old_mem.status == MemoryStatus.SUPERSEDED, "Old memory should be SUPERSEDED"
        assert old_mem.superseded_by == "Troll behavior confirmed", "Should reference new memory"
        assert old_mem.superseded_at_turn is not None, "Should have supersession turn"

        # Verify new PERMANENT memory added to memory_cache
        new_mem = next((m for m in manager.memory_cache[10] if m.title == "Troll behavior confirmed"), None)
        assert new_mem is not None, "New permanent memory should exist"
        assert new_mem.status == MemoryStatus.ACTIVE, "New memory should be ACTIVE"
        assert new_mem.persistence == "permanent", "New memory should be permanent"

        # Verify both memories in memory_cache (old SUPERSEDED, new ACTIVE)
        assert len(manager.memory_cache[10]) == 2, "Should have both old and new memories"

        # Verify file contains both memories
        memories_file = tmp_path / "Memories.md"
        assert memories_file.exists(), "Memories.md should exist"
        file_content = memories_file.read_text()
        assert "Troll at spawn location" in file_content, "Old core memory should be in file"
        assert "Troll behavior confirmed" in file_content, "New permanent memory should be in file"
        assert "SUPERSEDED" in file_content, "File should show SUPERSEDED status for old memory"
        assert "CORE" in file_content, "File should show CORE persistence marker"
        assert "PERMANENT" in file_content, "File should show PERMANENT persistence marker"

    def test_downgrade_rejection_preserves_original(
        self, mock_logger, game_config, game_state, tmp_path
    ):
        """
        Test that when downgrade is rejected, the original memory is completely unchanged.

        Validates that a rejected supersession leaves the original memory in its exact
        original state with no modifications to any fields (status, superseded_by,
        superseded_at_turn, etc.).

        Test approach:
        1. Add PERMANENT memory with specific attributes
        2. Capture original memory state (all fields)
        3. Attempt to supersede with EPHEMERAL memory (should fail)
        4. Retrieve memory again
        5. Assert ALL attributes unchanged (deep equality check)
        6. Assert memory still in memory_cache at correct location
        """
        # Setup
        game_config.zork_game_workdir = str(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Add PERMANENT memory with specific attributes
        original_memory = Memory(
            category="SUCCESS",
            title="Lamp acquisition procedure",
            episode=1,
            turns="45",
            score_change=5,
            text="Take the brass lantern from the trophy case in Living Room.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE,
            superseded_by=None,
            superseded_at_turn=None,
            invalidation_reason=None
        )
        success = manager.add_memory(15, "Living Room", original_memory)
        assert success, "Failed to add permanent memory"

        # Capture original state (before attempted supersession)
        original_state = manager.memory_cache[15][0]
        original_category = original_state.category
        original_title = original_state.title
        original_episode = original_state.episode
        original_turns = original_state.turns
        original_score_change = original_state.score_change
        original_text = original_state.text
        original_persistence = original_state.persistence
        original_status = original_state.status
        original_superseded_by = original_state.superseded_by
        original_superseded_at_turn = original_state.superseded_at_turn
        original_invalidation_reason = original_state.invalidation_reason

        # Create EPHEMERAL memory (attempted downgrade)
        new_memory = Memory(
            category="NOTE",
            title="Lamp moved elsewhere",
            episode=1,
            turns="100",
            score_change=0,
            text="I moved the lamp to a different room.",
            persistence="ephemeral"
        )

        # Attempt to supersede (should fail)
        success = manager.supersede_memory(15, "Living Room", "Lamp acquisition procedure", new_memory)
        assert not success, "Supersession should be rejected"

        # Retrieve memory again
        preserved_memory = manager.memory_cache[15][0]

        # Assert ALL attributes unchanged (deep equality check)
        assert preserved_memory.category == original_category, "Category should be unchanged"
        assert preserved_memory.title == original_title, "Title should be unchanged"
        assert preserved_memory.episode == original_episode, "Episode should be unchanged"
        assert preserved_memory.turns == original_turns, "Turns should be unchanged"
        assert preserved_memory.score_change == original_score_change, "Score change should be unchanged"
        assert preserved_memory.text == original_text, "Text should be unchanged"
        assert preserved_memory.persistence == original_persistence, "Persistence should be unchanged"
        assert preserved_memory.status == original_status, "Status should be unchanged (ACTIVE)"
        assert preserved_memory.superseded_by == original_superseded_by, "Superseded_by should remain None"
        assert preserved_memory.superseded_at_turn == original_superseded_at_turn, \
            "Superseded_at_turn should remain None"
        assert preserved_memory.invalidation_reason == original_invalidation_reason, \
            "Invalidation_reason should remain None"

        # Verify memory still in memory_cache at correct location
        assert 15 in manager.memory_cache, "Location should still be in memory_cache"
        assert len(manager.memory_cache[15]) == 1, "Should still have exactly one memory"
        assert manager.memory_cache[15][0].title == "Lamp acquisition procedure", \
            "Original memory should still be present"

        # Verify new ephemeral memory NOT added
        assert 15 not in manager.ephemeral_cache, "No ephemeral memories should be added"

        # Verify file unchanged (only original permanent memory)
        memories_file = tmp_path / "Memories.md"
        assert memories_file.exists(), "Memories.md should exist"
        file_content = memories_file.read_text()
        assert "Lamp acquisition procedure" in file_content, "Original memory should be in file"
        assert "Lamp moved elsewhere" not in file_content, "New ephemeral memory should NOT be in file"
        assert file_content.count("Location 15: Living Room") == 1, \
            "Should have exactly one location header (no duplication)"
