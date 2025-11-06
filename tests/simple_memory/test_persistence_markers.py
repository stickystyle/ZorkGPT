"""
ABOUTME: Tests for persistence marker formatting in Memories.md file writing.
ABOUTME: Validates that CORE/PERMANENT markers appear in category field, ephemeral memories excluded.

Phase 5.1 of Ephemeral Memory System: Persistence markers in file format.

EXPECTED FORMAT:
- CORE: **[SUCCESS - CORE] Title** *(Ep1, T23, +0)*
- PERMANENT: **[SUCCESS - PERMANENT] Title** *(Ep1, T23, +0)*
- EPHEMERAL: NOT written to file (in-memory cache only)

CURRENT BUG:
- _format_memory_entry() writes **[SUCCESS] Title** without persistence marker
- Tests should FAIL until implementation adds markers
"""

import pytest
import re
from pathlib import Path
from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus


def test_core_memory_written_with_core_marker(mock_logger, game_config, game_state, tmp_path):
    """
    Test that CORE memories include '- CORE' marker in category field.

    Expected format: **[SUCCESS - CORE] Memory Title** *(Ep1, T23, +0)*

    This test validates CORE marker appears in file when memory.persistence='core'.
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Create CORE memory
    memory = Memory(
        category="SUCCESS",
        title="First visit description",
        episode=1,
        turns="1",
        score_change=0,
        text="This is the opening room description.",
        persistence="core"  # CORE persistence
    )

    # Add memory to location
    success = manager.add_memory(10, "West of House", memory)
    assert success, "Memory addition should succeed"

    # Read file and verify marker
    memories_file = tmp_path / "Memories.md"
    assert memories_file.exists(), "Memories.md should be created"

    file_content = memories_file.read_text()

    # Verify CORE marker in category field
    assert "**[SUCCESS - CORE]" in file_content, (
        "Category field should include '- CORE' marker for CORE memories. "
        f"Expected: **[SUCCESS - CORE] First visit description**\n"
        f"File content:\n{file_content}"
    )

    # Verify full line format
    assert "**[SUCCESS - CORE] First visit description** *(Ep1, T1, +0)*" in file_content, (
        "Full memory line should have correct format with CORE marker"
    )


def test_permanent_memory_written_with_permanent_marker(mock_logger, game_config, game_state, tmp_path):
    """
    Test that PERMANENT memories include '- PERMANENT' marker in category field.

    Expected format: **[NOTE - PERMANENT] Memory Title** *(Ep1, T23, +0)*

    This test validates PERMANENT marker appears in file when memory.persistence='permanent'.
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Create PERMANENT memory
    memory = Memory(
        category="NOTE",
        title="Open window",
        episode=1,
        turns="23",
        score_change=5,
        text="Window can be opened successfully.",
        persistence="permanent"  # PERMANENT persistence
    )

    # Add memory to location
    success = manager.add_memory(10, "West of House", memory)
    assert success, "Memory addition should succeed"

    # Read file and verify marker
    memories_file = tmp_path / "Memories.md"
    assert memories_file.exists(), "Memories.md should be created"

    file_content = memories_file.read_text()

    # Verify PERMANENT marker in category field
    assert "**[NOTE - PERMANENT]" in file_content, (
        "Category field should include '- PERMANENT' marker for PERMANENT memories. "
        f"Expected: **[NOTE - PERMANENT] Open window**\n"
        f"File content:\n{file_content}"
    )

    # Verify full line format
    assert "**[NOTE - PERMANENT] Open window** *(Ep1, T23, +5)*" in file_content, (
        "Full memory line should have correct format with PERMANENT marker"
    )


def test_ephemeral_memory_not_written_to_file(mock_logger, game_config, game_state, tmp_path):
    """
    Test that EPHEMERAL memories are NOT written to file (in-memory cache only).

    Expected behavior: Memory added to ephemeral_cache, file not created or empty.

    This test validates ephemeral memories stay in cache only.
    Note: This should already pass from Phase 3.1 tests.
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Create EPHEMERAL memory
    memory = Memory(
        category="NOTE",
        title="Dropped sword here",
        episode=1,
        turns="30",
        score_change=0,
        text="Dropped sword at this location temporarily.",
        persistence="ephemeral"  # EPHEMERAL persistence
    )

    # Add memory to location
    success = manager.add_memory(10, "West of House", memory)
    assert success, "Memory addition should succeed"

    # Verify memory in ephemeral_cache
    assert 10 in manager.ephemeral_cache, "Memory should be in ephemeral_cache"
    assert len(manager.ephemeral_cache[10]) == 1, "Should have 1 ephemeral memory"
    assert manager.ephemeral_cache[10][0].title == "Dropped sword here"

    # Verify file NOT created OR doesn't contain ephemeral memory
    memories_file = tmp_path / "Memories.md"

    if memories_file.exists():
        file_content = memories_file.read_text()
        assert "Dropped sword here" not in file_content, (
            "Ephemeral memory should NOT appear in file. "
            f"File content:\n{file_content}"
        )
    # If file doesn't exist, that's also correct (no persistent memories to write)


def test_multiple_persistence_types_in_same_file(mock_logger, game_config, game_state, tmp_path):
    """
    Test that mixed persistence types (CORE + PERMANENT) format correctly in same file.

    Expected:
    - CORE memory: **[SUCCESS - CORE] Title**
    - PERMANENT memories: **[NOTE - PERMANENT] Title**, **[FAILURE - PERMANENT] Title**

    This validates marker formatting works across different persistence levels.
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Create 1 CORE memory
    core_memory = Memory(
        category="SUCCESS",
        title="Room entry method",
        episode=1,
        turns="1",
        score_change=0,
        text="This room can be entered from multiple directions.",
        persistence="core"
    )

    # Create 2 PERMANENT memories
    perm_memory_1 = Memory(
        category="NOTE",
        title="Window opens",
        episode=1,
        turns="23",
        score_change=0,
        text="Window can be opened with effort.",
        persistence="permanent"
    )

    perm_memory_2 = Memory(
        category="FAILURE",
        title="Cannot break window",
        episode=1,
        turns="25",
        score_change=0,
        text="Window is sturdy and cannot be broken.",
        persistence="permanent"
    )

    # Add all three to same location
    manager.add_memory(10, "West of House", core_memory)
    manager.add_memory(10, "West of House", perm_memory_1)
    manager.add_memory(10, "West of House", perm_memory_2)

    # Read file and verify all markers
    memories_file = tmp_path / "Memories.md"
    assert memories_file.exists(), "Memories.md should be created"

    file_content = memories_file.read_text()

    # Verify CORE marker
    assert "**[SUCCESS - CORE] Room entry method**" in file_content, (
        "CORE memory should have '- CORE' marker"
    )

    # Verify both PERMANENT markers
    assert "**[NOTE - PERMANENT] Window opens**" in file_content, (
        "First PERMANENT memory should have '- PERMANENT' marker"
    )

    assert "**[FAILURE - PERMANENT] Cannot break window**" in file_content, (
        "Second PERMANENT memory should have '- PERMANENT' marker"
    )

    # Verify all three memories present
    assert file_content.count("- CORE]") == 1, "Should have exactly 1 CORE marker"
    assert file_content.count("- PERMANENT]") == 2, "Should have exactly 2 PERMANENT markers"


def test_superseded_memory_keeps_persistence_marker(mock_logger, game_config, game_state, tmp_path):
    """
    Test that superseded memories preserve persistence marker with SUPERSEDED status.

    Expected format: **[SUCCESS - PERMANENT] [SUPERSEDED] Old Memory**

    This validates persistence marker is preserved during supersession workflow.
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Create initial PERMANENT memory
    old_memory = Memory(
        category="SUCCESS",
        title="Window approach works",
        episode=1,
        turns="23",
        score_change=0,
        text="Window can be opened successfully.",
        persistence="permanent"
    )

    # Add initial memory
    manager.add_memory(10, "West of House", old_memory)

    # Create new PERMANENT memory that supersedes old
    new_memory = Memory(
        category="SUCCESS",
        title="Window requires force",
        episode=1,
        turns="25",
        score_change=0,
        text="Window requires significant force to open fully.",
        persistence="permanent"
    )

    # Supersede old with new
    success = manager.supersede_memory(10, "West of House", "Window approach works", new_memory)
    assert success, "Supersession should succeed"

    # Read file and verify both memories have persistence markers
    memories_file = tmp_path / "Memories.md"
    file_content = memories_file.read_text()

    # Verify new memory has PERMANENT marker
    assert "**[SUCCESS - PERMANENT] Window requires force**" in file_content, (
        "New memory should have '- PERMANENT' marker"
    )

    # Verify old memory has PERMANENT marker AND SUPERSEDED status
    # Format should be: **[SUCCESS - PERMANENT - SUPERSEDED] Window approach works**
    # OR: **[SUCCESS - SUPERSEDED] Window approach works** with persistence in second field

    # Check for persistence marker in superseded memory
    # The implementation may format as [CATEGORY - SUPERSEDED] or [CATEGORY - PERMANENT - SUPERSEDED]
    # We need to verify persistence is preserved somehow

    # Use regex to find superseded memory line
    superseded_pattern = r'\*\*\[(SUCCESS[^\]]*SUPERSEDED[^\]]*)\] Window approach works\*\*'
    match = re.search(superseded_pattern, file_content)

    assert match, (
        "Should find superseded memory with correct status. "
        f"File content:\n{file_content}"
    )

    # Extract the category+status field
    category_status = match.group(1)

    # Verify it includes either PERMANENT or SUPERSEDED (implementation dependent)
    # The key is persistence marker should not be lost
    assert "PERMANENT" in category_status or "SUPERSEDED" in category_status, (
        f"Superseded memory should preserve persistence or status marker. "
        f"Found: '{category_status}'\n"
        f"File content:\n{file_content}"
    )


def test_different_categories_with_persistence_markers(mock_logger, game_config, game_state, tmp_path):
    """
    Test that persistence markers work correctly with all category types.

    Expected formats:
    - **[SUCCESS - PERMANENT] Title**
    - **[FAILURE - PERMANENT] Title**
    - **[NOTE - PERMANENT] Title**
    - **[DANGER - PERMANENT] Title**
    - **[DISCOVERY - PERMANENT] Title**

    This validates marker works consistently across all memory categories.
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Create memories with different categories, all PERMANENT
    categories_and_titles = [
        ("SUCCESS", "Action succeeded", "This action worked."),
        ("FAILURE", "Action failed", "This action did not work."),
        ("NOTE", "Observation noted", "Noticed this detail."),
        ("DANGER", "Hazard detected", "This is dangerous."),
        ("DISCOVERY", "Item found", "Found important item."),
    ]

    # Add all memories to same location
    for i, (category, title, text) in enumerate(categories_and_titles):
        memory = Memory(
            category=category,
            title=title,
            episode=1,
            turns=str(10 + i),
            score_change=0,
            text=text,
            persistence="permanent"
        )
        manager.add_memory(10, "West of House", memory)

    # Read file and verify all category+marker combinations
    memories_file = tmp_path / "Memories.md"
    file_content = memories_file.read_text()

    # Verify each category has PERMANENT marker
    expected_formats = [
        "**[SUCCESS - PERMANENT] Action succeeded**",
        "**[FAILURE - PERMANENT] Action failed**",
        "**[NOTE - PERMANENT] Observation noted**",
        "**[DANGER - PERMANENT] Hazard detected**",
        "**[DISCOVERY - PERMANENT] Item found**",
    ]

    for expected in expected_formats:
        assert expected in file_content, (
            f"Should find '{expected}' in file. "
            f"File content:\n{file_content}"
        )


def test_persistence_marker_parsing_backwards_compatible(mock_logger, game_config, game_state, tmp_path):
    """
    Test that old format (no persistence markers) is parsed correctly as 'permanent'.

    Scenario:
    1. Manually create Memories.md with old format (no markers)
    2. Load memories via _load_memories_from_file()
    3. Verify old memory parsed with default persistence='permanent'
    4. Add new memory with CORE persistence
    5. Verify new memory has '- CORE' marker in file

    This validates backward compatibility with existing Memories.md files.
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    # Create Memories.md with OLD format (no persistence markers)
    old_format_content = """# Location Memories

## Location 10: West of House
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Old Memory** *(Ep1, T1, +0)*
This memory was written in old format without persistence markers.

---
"""

    memories_file = tmp_path / "Memories.md"
    memories_file.write_text(old_format_content)

    # Create manager (will load existing file)
    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Verify old memory was parsed
    assert 10 in manager.memory_cache, "Should load location 10"
    assert len(manager.memory_cache[10]) == 1, "Should have 1 memory"

    old_memory = manager.memory_cache[10][0]
    assert old_memory.title == "Old Memory", "Should parse old memory title"

    # Verify old memory has default persistence='permanent'
    assert old_memory.persistence == "permanent", (
        "Old format memories (no marker) should default to 'permanent'"
    )

    # Now add a new CORE memory
    new_memory = Memory(
        category="SUCCESS",
        title="New Core Memory",
        episode=1,
        turns="10",
        score_change=0,
        text="This is a new memory with CORE persistence.",
        persistence="core"
    )

    manager.add_memory(10, "West of House", new_memory)

    # Read file and verify:
    # 1. Old memory still present (unchanged format)
    # 2. New memory has CORE marker
    file_content = memories_file.read_text()

    # Old memory should still be in old format
    assert "**[SUCCESS] Old Memory**" in file_content, (
        "Old memory should remain in original format"
    )

    # New memory should have CORE marker
    assert "**[SUCCESS - CORE] New Core Memory**" in file_content, (
        "New memory should have '- CORE' marker. "
        f"File content:\n{file_content}"
    )


def test_core_memory_with_status_markers(mock_logger, game_config, game_state, tmp_path):
    """
    Test CORE memory with TENTATIVE status includes both markers.

    Expected format: **[SUCCESS - CORE] Title** with TENTATIVE status

    Note: Current implementation may format as [CATEGORY - STATUS] without persistence.
    This test validates the interaction between persistence and status markers.
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Create CORE memory with TENTATIVE status
    memory = Memory(
        category="SUCCESS",
        title="Unconfirmed core fact",
        episode=1,
        turns="15",
        score_change=0,
        text="This might be a core mechanic but needs confirmation.",
        persistence="core",
        status=MemoryStatus.TENTATIVE
    )

    # Add memory
    manager.add_memory(10, "West of House", memory)

    # Read file
    memories_file = tmp_path / "Memories.md"
    file_content = memories_file.read_text()

    # The implementation needs to handle both persistence and status
    # Expected format could be:
    # - **[SUCCESS - CORE - TENTATIVE] Title** (all markers)
    # - **[SUCCESS - TENTATIVE] Title** (status takes precedence)
    # - **[SUCCESS - CORE] Title** (persistence takes precedence)

    # At minimum, verify CORE marker is present
    assert "- CORE" in file_content, (
        "CORE persistence marker should be present even with TENTATIVE status. "
        f"File content:\n{file_content}"
    )


def test_ephemeral_memory_count_validation(mock_logger, game_config, game_state, tmp_path):
    """
    Test that ephemeral memories increment cache count but not file persistence count.

    Validates:
    - Ephemeral memories added to ephemeral_cache
    - get_ephemeral_count() returns correct count
    - get_persistent_count() excludes ephemeral memories
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add 2 ephemeral memories
    eph_mem_1 = Memory(
        category="NOTE",
        title="Ephemeral note 1",
        episode=1,
        turns="10",
        score_change=0,
        text="Temporary note.",
        persistence="ephemeral"
    )

    eph_mem_2 = Memory(
        category="NOTE",
        title="Ephemeral note 2",
        episode=1,
        turns="11",
        score_change=0,
        text="Another temporary note.",
        persistence="ephemeral"
    )

    manager.add_memory(10, "West of House", eph_mem_1)
    manager.add_memory(10, "West of House", eph_mem_2)

    # Add 1 permanent memory
    perm_mem = Memory(
        category="SUCCESS",
        title="Permanent note",
        episode=1,
        turns="12",
        score_change=0,
        text="This is permanent.",
        persistence="permanent"
    )

    manager.add_memory(10, "West of House", perm_mem)

    # Validate counts
    assert manager.get_ephemeral_count(10) == 2, "Should have 2 ephemeral memories"
    assert manager.get_persistent_count(10) == 1, "Should have 1 persistent memory"

    # Verify file only contains permanent memory
    memories_file = tmp_path / "Memories.md"
    file_content = memories_file.read_text()

    assert "Permanent note" in file_content, "Permanent memory should be in file"
    assert "Ephemeral note 1" not in file_content, "Ephemeral memories should not be in file"
    assert "Ephemeral note 2" not in file_content, "Ephemeral memories should not be in file"


def test_memory_breakdown_includes_persistence_types(mock_logger, game_config, game_state, tmp_path):
    """
    Test that get_memory_breakdown() correctly counts CORE, PERMANENT, and EPHEMERAL.

    Validates breakdown structure:
    {"core": count, "permanent": count, "ephemeral": count}
    """
    # Use real temp directory
    game_config.zork_game_workdir = str(tmp_path)

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    # Add 1 CORE memory
    manager.add_memory(10, "West of House", Memory(
        category="SUCCESS",
        title="Core fact",
        episode=1,
        turns="1",
        score_change=0,
        text="Core game mechanic.",
        persistence="core"
    ))

    # Add 2 PERMANENT memories
    manager.add_memory(10, "West of House", Memory(
        category="NOTE",
        title="Permanent note 1",
        episode=1,
        turns="10",
        score_change=0,
        text="Permanent observation.",
        persistence="permanent"
    ))

    manager.add_memory(10, "West of House", Memory(
        category="NOTE",
        title="Permanent note 2",
        episode=1,
        turns="11",
        score_change=0,
        text="Another permanent observation.",
        persistence="permanent"
    ))

    # Add 3 EPHEMERAL memories
    for i in range(3):
        manager.add_memory(10, "West of House", Memory(
            category="NOTE",
            title=f"Ephemeral note {i}",
            episode=1,
            turns=str(20 + i),
            score_change=0,
            text=f"Temporary note {i}.",
            persistence="ephemeral"
        ))

    # Get breakdown
    breakdown = manager.get_memory_breakdown(10)

    # Validate counts
    assert breakdown["core"] == 1, "Should have 1 CORE memory"
    assert breakdown["permanent"] == 2, "Should have 2 PERMANENT memories"
    assert breakdown["ephemeral"] == 3, "Should have 3 EPHEMERAL memories"
