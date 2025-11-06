"""
ABOUTME: Tests for supersede_memory() cache migration in SimpleMemoryManager.
ABOUTME: Validates three cases: permanent→permanent, ephemeral→ephemeral, ephemeral→permanent.

Phase 4.1 of Ephemeral Memory System: supersede_memory() must handle cache migration
when an ephemeral memory is superseded by a permanent memory (upgrade case).

Test Coverage:
1. Permanent → Permanent (existing behavior, no regression)
2. Ephemeral → Ephemeral (stays in ephemeral cache, no file write)
3. Ephemeral → Permanent (migration: ephemeral_cache → memory_cache + file)
4. Cache search across both caches
5. File isolation for ephemeral supersession
6. File persistence for migration case
7. Episode reset behavior with migrated memories
"""

import pytest
from pathlib import Path
from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus


def test_supersede_permanent_with_permanent(mock_logger, game_config, game_state, tmp_path):
    """
    Test superseding permanent memory with permanent memory.
    This validates existing behavior (no regression test).

    Expected behavior:
    - Old permanent memory marked SUPERSEDED in memory_cache
    - New permanent memory ACTIVE in memory_cache
    - Both memories written to file
    - Both appear in get_location_memory() output
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add old permanent memory
    old_memory = Memory(
        category="NOTE",
        title="Door locked",
        episode=1,
        turns="10",
        score_change=0,
        text="The wooden door is locked and won't budge.",
        persistence="permanent"
    )
    success = manager.add_memory(10, "Test Room", old_memory)
    assert success, "Failed to add old memory"

    # Verify old memory in memory_cache
    assert 10 in manager.memory_cache, "Location should be in memory_cache"
    assert len(manager.memory_cache[10]) == 1, "Should have one memory"
    assert manager.memory_cache[10][0].title == "Door locked"
    assert manager.memory_cache[10][0].status == MemoryStatus.ACTIVE

    # Create new permanent memory
    new_memory = Memory(
        category="NOTE",
        title="Door has complex lock",
        episode=1,
        turns="20",
        score_change=0,
        text="The door has a complex three-tumbler lock mechanism.",
        persistence="permanent"
    )

    # Supersede old memory with new memory
    success = manager.supersede_memory(10, "Test Room", "Door locked", new_memory)
    assert success, "Supersession should succeed"

    # Verify old memory marked SUPERSEDED
    old_mem = next((m for m in manager.memory_cache[10] if m.title == "Door locked"), None)
    assert old_mem is not None, "Old memory should still exist"
    assert old_mem.status == MemoryStatus.SUPERSEDED, "Old memory should be SUPERSEDED"
    assert old_mem.superseded_by == "Door has complex lock", "Should reference new memory"

    # Verify new memory ACTIVE
    new_mem = next((m for m in manager.memory_cache[10] if m.title == "Door has complex lock"), None)
    assert new_mem is not None, "New memory should exist"
    assert new_mem.status == MemoryStatus.ACTIVE, "New memory should be ACTIVE"

    # Verify both in get_location_memory() output
    output = manager.get_location_memory(10)
    assert "Door has complex lock" in output, "New memory should appear in output"
    # Old memory should NOT appear (SUPERSEDED memories are hidden)
    assert "Door locked" not in output, "Old SUPERSEDED memory should be hidden"

    # Verify file persistence
    memories_file = tmp_path / "Memories.md"
    assert memories_file.exists(), "Memories.md should exist"
    file_content = memories_file.read_text()
    assert "Door locked" in file_content, "Old memory should be in file"
    assert "Door has complex lock" in file_content, "New memory should be in file"
    assert "SUPERSEDED" in file_content, "File should show SUPERSEDED status"


def test_supersede_ephemeral_with_ephemeral(mock_logger, game_config, game_state, tmp_path):
    """
    Test superseding ephemeral memory with ephemeral memory.

    Expected behavior:
    - Old ephemeral memory marked SUPERSEDED in ephemeral_cache
    - New ephemeral memory ACTIVE in ephemeral_cache
    - Old memory NOT in memory_cache (stayed ephemeral)
    - Neither memory written to file (ephemeral only)
    """
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add old ephemeral memory
    old_memory = Memory(
        category="NOTE",
        title="Sword on ground",
        episode=1,
        turns="5",
        score_change=0,
        text="The rusty sword is lying on the ground here.",
        persistence="ephemeral"
    )
    success = manager.add_memory(10, "Test Room", old_memory)
    assert success, "Failed to add old ephemeral memory"

    # Verify old memory in ephemeral_cache
    assert 10 in manager.ephemeral_cache, "Location should be in ephemeral_cache"
    assert len(manager.ephemeral_cache[10]) == 1, "Should have one ephemeral memory"
    assert manager.ephemeral_cache[10][0].title == "Sword on ground"

    # Verify NOT in memory_cache
    assert 10 not in manager.memory_cache, "Ephemeral memory should not be in memory_cache"

    # Create new ephemeral memory
    new_memory = Memory(
        category="NOTE",
        title="Sword in inventory",
        episode=1,
        turns="15",
        score_change=0,
        text="The rusty sword is now in my inventory.",
        persistence="ephemeral"
    )

    # Supersede
    success = manager.supersede_memory(10, "Test Room", "Sword on ground", new_memory)
    assert success, "Supersession should succeed"

    # Verify old memory marked SUPERSEDED in ephemeral_cache
    old_mem = next((m for m in manager.ephemeral_cache[10] if m.title == "Sword on ground"), None)
    assert old_mem is not None, "Old ephemeral memory should still exist"
    assert old_mem.status == MemoryStatus.SUPERSEDED, "Old memory should be SUPERSEDED"
    assert old_mem.superseded_by == "Sword in inventory"

    # Verify new memory ACTIVE in ephemeral_cache
    new_mem = next((m for m in manager.ephemeral_cache[10] if m.title == "Sword in inventory"), None)
    assert new_mem is not None, "New ephemeral memory should exist"
    assert new_mem.status == MemoryStatus.ACTIVE

    # Verify neither in memory_cache (stayed ephemeral)
    assert 10 not in manager.memory_cache, "No memories should be in memory_cache"

    # Verify NOT written to file
    memories_file = tmp_path / "Memories.md"
    if memories_file.exists():
        file_content = memories_file.read_text()
        assert "Sword on ground" not in file_content, "Old ephemeral memory should not be in file"
        assert "Sword in inventory" not in file_content, "New ephemeral memory should not be in file"


def test_supersede_ephemeral_with_permanent_migration(mock_logger, game_config, game_state, tmp_path):
    """
    Test superseding ephemeral memory with permanent memory (MIGRATION CASE).
    This is the key feature: upgrading ephemeral to permanent.

    Expected behavior:
    - Old ephemeral memory marked SUPERSEDED in ephemeral_cache
    - New permanent memory ACTIVE in memory_cache (MIGRATED!)
    - New memory written to file (PERSISTED!)
    - Both appear in get_location_memory() (different caches)
    """
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add old ephemeral memory
    old_memory = Memory(
        category="NOTE",
        title="Item dropped here",
        episode=1,
        turns="5",
        score_change=0,
        text="I dropped a brass key here temporarily.",
        persistence="ephemeral"
    )
    success = manager.add_memory(10, "Test Room", old_memory)
    assert success, "Failed to add ephemeral memory"

    # Verify in ephemeral_cache only
    assert 10 in manager.ephemeral_cache
    assert len(manager.ephemeral_cache[10]) == 1
    assert 10 not in manager.memory_cache

    # Create new permanent memory (upgrade)
    new_memory = Memory(
        category="NOTE",
        title="Item is game mechanic, always here",
        episode=1,
        turns="25",
        score_change=0,
        text="Brass key always respawns here after being dropped - part of game mechanic.",
        persistence="permanent"
    )

    # Supersede (migration should occur)
    success = manager.supersede_memory(10, "Test Room", "Item dropped here", new_memory)
    assert success, "Migration supersession should succeed"

    # Verify old memory marked SUPERSEDED in ephemeral_cache
    old_mem = next((m for m in manager.ephemeral_cache[10] if m.title == "Item dropped here"), None)
    assert old_mem is not None, "Old ephemeral memory should still exist in ephemeral_cache"
    assert old_mem.status == MemoryStatus.SUPERSEDED
    assert old_mem.superseded_by == "Item is game mechanic, always here"

    # CRITICAL: Verify new memory ACTIVE in memory_cache (MIGRATED!)
    assert 10 in manager.memory_cache, "Location should now be in memory_cache"
    new_mem = next((m for m in manager.memory_cache[10] if m.title == "Item is game mechanic, always here"), None)
    assert new_mem is not None, "New permanent memory should exist in memory_cache"
    assert new_mem.status == MemoryStatus.ACTIVE
    assert new_mem.persistence == "permanent"

    # Verify new memory written to file (PERSISTED!)
    memories_file = tmp_path / "Memories.md"
    assert memories_file.exists(), "Memories.md should exist after migration"
    file_content = memories_file.read_text()
    assert "Item is game mechanic, always here" in file_content, "New permanent memory should be in file"
    assert "Item dropped here" not in file_content, "Old ephemeral memory should NOT be in file"

    # Verify get_location_memory() shows both (from different caches)
    output = manager.get_location_memory(10)
    assert "Item is game mechanic, always here" in output, "New permanent memory should appear"
    # Old SUPERSEDED memory should be hidden
    assert "Item dropped here" not in output, "Old SUPERSEDED memory should be hidden"


def test_supersede_checks_both_caches(mock_logger, game_config, game_state, tmp_path):
    """
    Test that supersede_memory() searches both caches for the old memory.

    Expected behavior:
    - Can find and supersede ephemeral memory in ephemeral_cache
    - Can find and supersede permanent memory in memory_cache
    - Returns False if memory not found in either cache
    """
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add ephemeral memory to location 10
    ephemeral_mem = Memory(
        category="NOTE",
        title="Old ephemeral",
        episode=1,
        turns="5",
        score_change=0,
        text="Temporary ephemeral note.",
        persistence="ephemeral"
    )
    manager.add_memory(10, "Test Room", ephemeral_mem)

    # Add permanent memory to same location
    permanent_mem = Memory(
        category="NOTE",
        title="Old permanent",
        episode=1,
        turns="10",
        score_change=0,
        text="Permanent note.",
        persistence="permanent"
    )
    manager.add_memory(10, "Test Room", permanent_mem)

    # Try to supersede ephemeral memory (should find in ephemeral_cache)
    new_ephemeral = Memory(
        category="NOTE",
        title="New ephemeral",
        episode=1,
        turns="15",
        score_change=0,
        text="Updated ephemeral note.",
        persistence="ephemeral"
    )
    success = manager.supersede_memory(10, "Test Room", "Old ephemeral", new_ephemeral)
    assert success, "Should find and supersede ephemeral memory"

    # Verify supersession in ephemeral_cache
    old_eph = next((m for m in manager.ephemeral_cache[10] if m.title == "Old ephemeral"), None)
    assert old_eph.status == MemoryStatus.SUPERSEDED

    # Try to supersede permanent memory (should find in memory_cache)
    new_permanent = Memory(
        category="NOTE",
        title="New permanent",
        episode=1,
        turns="20",
        score_change=0,
        text="Updated permanent note.",
        persistence="permanent"
    )
    success = manager.supersede_memory(10, "Test Room", "Old permanent", new_permanent)
    assert success, "Should find and supersede permanent memory"

    # Verify supersession in memory_cache
    old_perm = next((m for m in manager.memory_cache[10] if m.title == "Old permanent"), None)
    assert old_perm.status == MemoryStatus.SUPERSEDED

    # Try to supersede non-existent memory
    fake_memory = Memory(
        category="NOTE",
        title="Fake new",
        episode=1,
        turns="25",
        score_change=0,
        text="This supersedes nothing.",
        persistence="permanent"
    )
    success = manager.supersede_memory(10, "Test Room", "Non-existent", fake_memory)
    assert not success, "Should fail when memory not found in either cache"


def test_supersede_ephemeral_not_in_file(mock_logger, game_config, game_state, tmp_path):
    """
    Test that ephemeral-to-ephemeral supersession never touches the file.

    Expected behavior:
    - Both old and new ephemeral memories NOT in file
    - Supersession happens in ephemeral_cache only
    - File remains unchanged (or empty if no other memories)
    """
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add ephemeral memory
    old_memory = Memory(
        category="NOTE",
        title="Ephemeral note 1",
        episode=1,
        turns="5",
        score_change=0,
        text="First ephemeral note.",
        persistence="ephemeral"
    )
    manager.add_memory(10, "Test Room", old_memory)

    # Supersede with another ephemeral memory
    new_memory = Memory(
        category="NOTE",
        title="Ephemeral note 2",
        episode=1,
        turns="15",
        score_change=0,
        text="Second ephemeral note.",
        persistence="ephemeral"
    )
    manager.supersede_memory(10, "Test Room", "Ephemeral note 1", new_memory)

    # Read file directly
    memories_file = tmp_path / "Memories.md"

    # File should either not exist or not contain ephemeral memories
    if memories_file.exists():
        file_content = memories_file.read_text()
        assert "Ephemeral note 1" not in file_content, "Old ephemeral should not be in file"
        assert "Ephemeral note 2" not in file_content, "New ephemeral should not be in file"
    else:
        # File doesn't exist - this is also valid (no permanent memories)
        pass


def test_supersede_migration_writes_to_file(mock_logger, game_config, game_state, tmp_path):
    """
    Test that ephemeral-to-permanent migration writes new memory to file.

    Expected behavior:
    - New permanent memory appears in file
    - Old ephemeral memory does NOT appear in file
    - File contains proper formatting for new permanent memory
    """
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add ephemeral memory
    old_memory = Memory(
        category="NOTE",
        title="Temporary state",
        episode=1,
        turns="5",
        score_change=0,
        text="This is a temporary state.",
        persistence="ephemeral"
    )
    manager.add_memory(10, "Test Room", old_memory)

    # Verify file doesn't contain ephemeral memory
    memories_file = tmp_path / "Memories.md"
    if memories_file.exists():
        before_content = memories_file.read_text()
        assert "Temporary state" not in before_content

    # Supersede with permanent memory (migration)
    new_memory = Memory(
        category="NOTE",
        title="Permanent game mechanic",
        episode=1,
        turns="25",
        score_change=0,
        text="This is a permanent game mechanic discovered through testing.",
        persistence="permanent"
    )
    manager.supersede_memory(10, "Test Room", "Temporary state", new_memory)

    # Read file directly
    assert memories_file.exists(), "Memories.md should exist after migration"
    file_content = memories_file.read_text()

    # Verify new permanent memory in file
    assert "Permanent game mechanic" in file_content, "New permanent memory should be in file"
    assert "Location 10: Test Room" in file_content, "Location header should be in file"

    # Verify old ephemeral memory NOT in file
    assert "Temporary state" not in file_content, "Old ephemeral memory should NOT be in file"


def test_supersede_after_episode_reset(mock_logger, game_config, game_state, tmp_path):
    """
    Test behavior of superseded memories after episode reset.

    Expected behavior:
    - Ephemeral cache cleared (including superseded old memory)
    - Permanent new memory survives in memory_cache
    - get_location_memory() returns only permanent memory after reset
    """
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add ephemeral memory
    old_memory = Memory(
        category="NOTE",
        title="Ephemeral assumption",
        episode=1,
        turns="5",
        score_change=0,
        text="Initial ephemeral assumption.",
        persistence="ephemeral"
    )
    manager.add_memory(10, "Test Room", old_memory)

    # Supersede with permanent memory (migration)
    new_memory = Memory(
        category="NOTE",
        title="Confirmed permanent fact",
        episode=1,
        turns="25",
        score_change=5,
        text="Confirmed permanent game mechanic.",
        persistence="permanent"
    )
    manager.supersede_memory(10, "Test Room", "Ephemeral assumption", new_memory)

    # Verify both exist before reset
    assert 10 in manager.ephemeral_cache, "Ephemeral cache should have location"
    assert 10 in manager.memory_cache, "Memory cache should have location"
    assert len(manager.ephemeral_cache[10]) == 1, "Should have old ephemeral (SUPERSEDED)"
    assert len(manager.memory_cache[10]) == 1, "Should have new permanent (ACTIVE)"

    # Call reset_episode()
    manager.reset_episode()

    # Verify ephemeral cache cleared
    assert len(manager.ephemeral_cache) == 0, "Ephemeral cache should be cleared"

    # Verify permanent memory still in memory_cache
    assert 10 in manager.memory_cache, "Permanent memory should survive reset"
    assert len(manager.memory_cache[10]) == 1

    # Verify get_location_memory() returns only permanent memory
    output = manager.get_location_memory(10)
    assert "Confirmed permanent fact" in output, "Permanent memory should appear"
    assert "Ephemeral assumption" not in output, "Old ephemeral memory should be gone"
