# ABOUTME: Tests for ephemeral cache initialization and public API methods (Phase 2.1 of ephemeral memory system)
# ABOUTME: Verifies dual cache architecture and introspection methods

import pytest
from pathlib import Path

from managers.simple_memory_manager import SimpleMemoryManager
from managers.memory.models import Memory, MemoryStatus
from session.game_configuration import GameConfiguration
from session.game_state import GameState


class TestEphemeralCacheInitialization:
    """Test suite for ephemeral cache initialization and API methods."""

    def test_ephemeral_cache_initializes_empty(self, mock_logger, game_config, game_state):
        """
        Test that SimpleMemoryManager initializes with empty ephemeral_cache.

        Verifies:
        - ephemeral_cache exists as empty dict
        - get_ephemeral_count() returns 0 for total
        """
        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Verify ephemeral_cache exists and is empty
        assert hasattr(manager, 'ephemeral_cache'), "Manager should have ephemeral_cache attribute"
        assert isinstance(manager.ephemeral_cache, dict), "ephemeral_cache should be a dict"
        assert len(manager.ephemeral_cache) == 0, "ephemeral_cache should initialize empty"

        # Verify get_ephemeral_count() returns 0
        assert manager.get_ephemeral_count() == 0, "get_ephemeral_count() should return 0 for empty cache"

    def test_get_ephemeral_count_specific_location(self, mock_logger, game_config, game_state):
        """
        Test get_ephemeral_count() for specific location.

        Verifies:
        - Returns 0 for empty location
        - Returns correct count after adding ephemeral memories
        """
        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Test empty location returns 0
        assert manager.get_ephemeral_count(location_id=5) == 0, "Empty location should have 0 ephemeral memories"

        # Add ephemeral memories to location 5
        memory1 = Memory(
            category="NOTE",
            title="Ephemeral observation 1",
            episode=1,
            turns="10",
            score_change=0,
            text="Temporary note about location.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        memory2 = Memory(
            category="NOTE",
            title="Ephemeral observation 2",
            episode=1,
            turns="11",
            score_change=0,
            text="Another temporary note.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        manager.ephemeral_cache[5] = [memory1, memory2]

        # Verify count is correct
        assert manager.get_ephemeral_count(location_id=5) == 2, "Should count 2 ephemeral memories at location 5"

        # Test different location still returns 0
        assert manager.get_ephemeral_count(location_id=8) == 0, "Different location should still have 0 ephemeral memories"

    def test_get_ephemeral_count_all_locations(self, mock_logger, game_config, game_state):
        """
        Test get_ephemeral_count() without location_id returns total across all locations.

        Verifies:
        - Total count aggregates memories from multiple locations
        """
        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add 3 ephemeral memories to location 5
        manager.ephemeral_cache[5] = [
            Memory(
                category="NOTE",
                title=f"Ephemeral note {i}",
                episode=1,
                turns=str(10 + i),
                score_change=0,
                text=f"Temporary note {i}.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )
            for i in range(3)
        ]

        # Add 2 ephemeral memories to location 8
        manager.ephemeral_cache[8] = [
            Memory(
                category="NOTE",
                title=f"Another ephemeral note {i}",
                episode=1,
                turns=str(20 + i),
                score_change=0,
                text=f"Different temporary note {i}.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )
            for i in range(2)
        ]

        # Verify total count
        total = manager.get_ephemeral_count()
        assert total == 5, f"Total ephemeral count should be 5 (3 at loc 5 + 2 at loc 8), got {total}"

    def test_get_persistent_count_specific_location(self, mock_logger, game_config, game_state, create_memories_file):
        """
        Test get_persistent_count() for specific location.

        Verifies:
        - Returns correct count from memory_cache
        - Counts CORE and PERMANENT memories
        """
        # Create memories file with persistent memories
        memories_content = """# Location Memories

## Location 15: West of House
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Open window** *(Ep1, T23, +0)*
Window can be opened successfully.

**[DISCOVERY] Mailbox location** *(Ep1, T20, +0)*
Small mailbox located here.

---
"""
        create_memories_file(memories_content)

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Test location 15 has 2 persistent memories
        count = manager.get_persistent_count(location_id=15)
        assert count == 2, f"Location 15 should have 2 persistent memories, got {count}"

        # Test empty location returns 0
        assert manager.get_persistent_count(location_id=99) == 0, "Empty location should have 0 persistent memories"

    def test_get_persistent_count_all_locations(self, mock_logger, game_config, game_state, create_memories_file):
        """
        Test get_persistent_count() without location_id returns total across all locations.

        Verifies:
        - Total count aggregates persistent memories from multiple locations
        """
        # Create memories file with persistent memories at multiple locations
        memories_content = """# Location Memories

## Location 15: West of House
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Open window** *(Ep1, T23, +0)*
Window can be opened successfully.

**[DISCOVERY] Mailbox location** *(Ep1, T20, +0)*
Small mailbox located here.

---

## Location 23: Living Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Acquire lamp** *(Ep1, T45, +5)*
Brass lantern is takeable.

---
"""
        create_memories_file(memories_content)

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Verify total count (2 at location 15 + 1 at location 23 = 3)
        total = manager.get_persistent_count()
        assert total == 3, f"Total persistent count should be 3 (2 at loc 15 + 1 at loc 23), got {total}"

    def test_get_memory_breakdown_counts_all_types(self, mock_logger, game_config, game_state, create_memories_file):
        """
        Test get_memory_breakdown() returns correct counts for all memory types.

        Verifies:
        - Returns dict with keys: "core", "permanent", "ephemeral"
        - Counts from both memory_cache and ephemeral_cache
        - Only counts ACTIVE/TENTATIVE memories (excludes SUPERSEDED)
        """
        # Create memories file with core and permanent memories
        memories_content = """# Location Memories

## Location 15: West of House
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS - CORE] Core memory** *(Ep1, T10, +0)*
A core game mechanic.

**[SUCCESS] Permanent memory** *(Ep1, T20, +0)*
A validated strategy.

---
"""
        create_memories_file(memories_content)

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add ephemeral memories to same location
        manager.ephemeral_cache[15] = [
            Memory(
                category="NOTE",
                title="Ephemeral note 1",
                episode=1,
                turns="30",
                score_change=0,
                text="Temporary observation.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            ),
            Memory(
                category="NOTE",
                title="Ephemeral note 2",
                episode=1,
                turns="31",
                score_change=0,
                text="Another temporary observation.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )
        ]

        # Get breakdown
        breakdown = manager.get_memory_breakdown(location_id=15)

        # Verify structure
        assert isinstance(breakdown, dict), "Breakdown should be a dict"
        assert set(breakdown.keys()) == {"core", "permanent", "ephemeral"}, "Breakdown should have keys: core, permanent, ephemeral"

        # Verify counts
        assert breakdown["core"] == 1, f"Should have 1 core memory, got {breakdown['core']}"
        assert breakdown["permanent"] == 1, f"Should have 1 permanent memory, got {breakdown['permanent']}"
        assert breakdown["ephemeral"] == 2, f"Should have 2 ephemeral memories, got {breakdown['ephemeral']}"

    def test_get_memory_breakdown_excludes_superseded(self, mock_logger, game_config, game_state):
        """
        Test get_memory_breakdown() excludes SUPERSEDED memories from counts.

        Verifies:
        - SUPERSEDED memories are not counted in breakdown
        - Only ACTIVE and TENTATIVE memories are counted
        """
        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add mix of ACTIVE, TENTATIVE, and SUPERSEDED memories
        manager.memory_cache[15] = [
            Memory(
                category="SUCCESS",
                title="Active core memory",
                episode=1,
                turns="10",
                score_change=0,
                text="Active core knowledge.",
                persistence="core",
                status=MemoryStatus.ACTIVE
            ),
            Memory(
                category="SUCCESS",
                title="Superseded core memory",
                episode=1,
                turns="11",
                score_change=0,
                text="Outdated core knowledge.",
                persistence="core",
                status=MemoryStatus.SUPERSEDED,
                superseded_by="Active core memory",
                superseded_at_turn=12
            ),
            Memory(
                category="SUCCESS",
                title="Tentative permanent memory",
                episode=1,
                turns="20",
                score_change=0,
                text="Unconfirmed strategy.",
                persistence="permanent",
                status=MemoryStatus.TENTATIVE
            )
        ]

        manager.ephemeral_cache[15] = [
            Memory(
                category="NOTE",
                title="Active ephemeral note",
                episode=1,
                turns="30",
                score_change=0,
                text="Current temporary observation.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            ),
            Memory(
                category="NOTE",
                title="Superseded ephemeral note",
                episode=1,
                turns="31",
                score_change=0,
                text="Outdated temporary observation.",
                persistence="ephemeral",
                status=MemoryStatus.SUPERSEDED,
                superseded_by="Active ephemeral note",
                superseded_at_turn=32
            )
        ]

        # Get breakdown
        breakdown = manager.get_memory_breakdown(location_id=15)

        # Verify only ACTIVE/TENTATIVE memories are counted (exclude SUPERSEDED)
        assert breakdown["core"] == 1, f"Should count 1 ACTIVE core (exclude SUPERSEDED), got {breakdown['core']}"
        assert breakdown["permanent"] == 1, f"Should count 1 TENTATIVE permanent (exclude SUPERSEDED), got {breakdown['permanent']}"
        assert breakdown["ephemeral"] == 1, f"Should count 1 ACTIVE ephemeral (exclude SUPERSEDED), got {breakdown['ephemeral']}"


class TestEpisodeReset:
    """Test suite for reset_episode() cache clearing behavior."""

    def test_ephemeral_cache_cleared_on_reset(self, mock_logger, game_config, game_state):
        """
        Test that reset_episode() clears ephemeral_cache.

        Test approach:
        1. Add ephemeral memories to ephemeral_cache
        2. Call reset_episode()
        3. Verify ephemeral_cache is empty
        """
        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add ephemeral memories to location 5
        memory1 = Memory(
            category="NOTE",
            title="Ephemeral observation 1",
            episode=1,
            turns="10",
            score_change=0,
            text="Temporary note about location.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        memory2 = Memory(
            category="NOTE",
            title="Ephemeral observation 2",
            episode=1,
            turns="11",
            score_change=0,
            text="Another temporary note.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        manager.ephemeral_cache[5] = [memory1, memory2]

        # Add ephemeral memories to location 8
        memory3 = Memory(
            category="NOTE",
            title="Ephemeral observation 3",
            episode=1,
            turns="12",
            score_change=0,
            text="Third temporary note.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        manager.ephemeral_cache[8] = [memory3]

        # Verify ephemeral_cache has memories
        assert len(manager.ephemeral_cache) == 2, "Should have 2 locations with ephemeral memories"
        assert len(manager.ephemeral_cache[5]) == 2, "Location 5 should have 2 ephemeral memories"
        assert len(manager.ephemeral_cache[8]) == 1, "Location 8 should have 1 ephemeral memory"

        # Act: Call reset_episode()
        manager.reset_episode()

        # Assert: ephemeral_cache should be empty
        assert len(manager.ephemeral_cache) == 0, "ephemeral_cache should be empty after reset_episode()"

    def test_persistent_cache_unchanged_on_reset(self, mock_logger, game_config, game_state, create_memories_file):
        """
        Test that reset_episode() preserves memory_cache.

        Test approach:
        1. Add persistent memories to memory_cache (via file)
        2. Add ephemeral memories to ephemeral_cache
        3. Call reset_episode()
        4. Verify memory_cache is unchanged
        5. Verify ephemeral_cache is empty
        """
        # Create memories file with persistent memories
        memories_content = """# Location Memories

## Location 15: West of House
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Open window** *(Ep1, T23, +0)*
Window can be opened successfully.

**[DISCOVERY] Mailbox location** *(Ep1, T20, +0)*
Small mailbox located here.

---
"""
        create_memories_file(memories_content)

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Verify persistent memories loaded
        assert 15 in manager.memory_cache, "Location 15 should have persistent memories"
        persistent_count_before = len(manager.memory_cache[15])
        assert persistent_count_before == 2, f"Should have 2 persistent memories, got {persistent_count_before}"

        # Add ephemeral memories to same location
        memory1 = Memory(
            category="NOTE",
            title="Ephemeral note 1",
            episode=1,
            turns="30",
            score_change=0,
            text="Temporary observation.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        memory2 = Memory(
            category="NOTE",
            title="Ephemeral note 2",
            episode=1,
            turns="31",
            score_change=0,
            text="Another temporary observation.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        manager.ephemeral_cache[15] = [memory1, memory2]

        # Verify both caches populated
        assert len(manager.ephemeral_cache[15]) == 2, "Should have 2 ephemeral memories before reset"
        assert len(manager.memory_cache[15]) == 2, "Should have 2 persistent memories before reset"

        # Act: Call reset_episode()
        manager.reset_episode()

        # Assert: memory_cache unchanged, ephemeral_cache cleared
        assert len(manager.memory_cache[15]) == 2, "Persistent cache should be unchanged after reset_episode()"
        assert len(manager.ephemeral_cache) == 0, "Ephemeral cache should be empty after reset_episode()"

        # Verify persistent memory contents unchanged
        persistent_titles = [m.title for m in manager.memory_cache[15]]
        assert "Open window" in persistent_titles, "Persistent memory 'Open window' should remain"
        assert "Mailbox location" in persistent_titles, "Persistent memory 'Mailbox location' should remain"

    def test_reset_logs_ephemeral_count(self, mock_logger, game_config, game_state):
        """
        Test that reset_episode() logs number of cleared memories.

        Test approach:
        1. Add 5 ephemeral memories (3 at loc 5, 2 at loc 8)
        2. Call reset_episode()
        3. Verify log_info called with count=5
        """
        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add 3 ephemeral memories to location 5
        manager.ephemeral_cache[5] = [
            Memory(
                category="NOTE",
                title=f"Ephemeral note {i}",
                episode=1,
                turns=str(10 + i),
                score_change=0,
                text=f"Temporary note {i}.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )
            for i in range(3)
        ]

        # Add 2 ephemeral memories to location 8
        manager.ephemeral_cache[8] = [
            Memory(
                category="NOTE",
                title=f"Another ephemeral note {i}",
                episode=1,
                turns=str(20 + i),
                score_change=0,
                text=f"Different temporary note {i}.",
                persistence="ephemeral",
                status=MemoryStatus.ACTIVE
            )
            for i in range(2)
        ]

        # Verify setup: 5 total ephemeral memories
        total_ephemeral = sum(len(mems) for mems in manager.ephemeral_cache.values())
        assert total_ephemeral == 5, f"Should have 5 ephemeral memories before reset, got {total_ephemeral}"

        # Act: Call reset_episode()
        manager.reset_episode()

        # Assert: Verify log_info called with correct count
        # Check that log_info was called with message containing "5 ephemeral memories"
        mock_logger.info.assert_called()

        # Check if ANY call contains the expected message
        calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("5 ephemeral memories" in call for call in calls), (
            f"Expected log message about '5 ephemeral memories' not found. Calls: {calls}"
        )

    def test_reset_with_empty_ephemeral_cache(self, mock_logger, game_config, game_state):
        """
        Test that reset_episode() handles empty cache gracefully.

        Test approach:
        1. Don't add any ephemeral memories
        2. Call reset_episode()
        3. Verify no errors
        4. Verify log shows count=0
        """
        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Verify ephemeral_cache is empty
        assert len(manager.ephemeral_cache) == 0, "ephemeral_cache should start empty"

        # Act: Call reset_episode() with empty cache (should not error)
        try:
            manager.reset_episode()
        except Exception as e:
            pytest.fail(f"reset_episode() should handle empty cache without error, but raised: {e}")

        # Assert: Verify log shows count=0
        mock_logger.info.assert_called()

        # Check if ANY call contains the expected message
        calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("0 ephemeral memories" in call for call in calls), (
            f"Expected log message about '0 ephemeral memories' not found. Calls: {calls}"
        )

    def test_multiple_resets_work_correctly(self, mock_logger, game_config, game_state):
        """
        Test that multiple reset_episode() calls work correctly.

        Test approach:
        1. Add ephemeral memories
        2. Call reset_episode() (clears cache)
        3. Add different ephemeral memories
        4. Call reset_episode() again (clears again)
        5. Verify cache empty after both resets
        """
        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # First batch of ephemeral memories
        memory1 = Memory(
            category="NOTE",
            title="First batch memory 1",
            episode=1,
            turns="10",
            score_change=0,
            text="First episode temporary note.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        memory2 = Memory(
            category="NOTE",
            title="First batch memory 2",
            episode=1,
            turns="11",
            score_change=0,
            text="Another first episode note.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        manager.ephemeral_cache[5] = [memory1, memory2]

        # Verify first batch loaded
        assert len(manager.ephemeral_cache[5]) == 2, "Should have 2 ephemeral memories in first batch"

        # First reset
        manager.reset_episode()

        # Verify cache cleared after first reset
        assert len(manager.ephemeral_cache) == 0, "ephemeral_cache should be empty after first reset"

        # Second batch of ephemeral memories (different content)
        memory3 = Memory(
            category="NOTE",
            title="Second batch memory 1",
            episode=2,
            turns="5",
            score_change=0,
            text="Second episode temporary note.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        memory4 = Memory(
            category="NOTE",
            title="Second batch memory 2",
            episode=2,
            turns="6",
            score_change=0,
            text="Another second episode note.",
            persistence="ephemeral",
            status=MemoryStatus.ACTIVE
        )
        manager.ephemeral_cache[8] = [memory3, memory4]

        # Verify second batch loaded
        assert len(manager.ephemeral_cache[8]) == 2, "Should have 2 ephemeral memories in second batch"

        # Second reset
        manager.reset_episode()

        # Verify cache cleared after second reset
        assert len(manager.ephemeral_cache) == 0, "ephemeral_cache should be empty after second reset"

        # Verify log was called twice (once for each reset)
        assert mock_logger.info.call_count >= 2, \
            f"log_info should be called at least twice (once per reset), got {mock_logger.info.call_count} calls"


# Import fixtures from conftest
pytest_plugins = ['tests.simple_memory.conftest']
