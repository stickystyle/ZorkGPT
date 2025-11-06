"""
ABOUTME: Tests for Phase 3 public API methods (invalidate_memory, invalidate_memories).
ABOUTME: Validates single and batch invalidation operations with comprehensive coverage.
"""

import pytest
from pathlib import Path
from managers.simple_memory_manager import (
    SimpleMemoryManager,
    Memory,
    MemoryStatus,
    INVALIDATION_MARKER
)
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from unittest.mock import MagicMock


@pytest.fixture
def setup_manager(tmp_path):
    """Create SimpleMemoryManager with test configuration."""
    # Create test directory
    game_dir = tmp_path / "game_files"
    game_dir.mkdir()

    # Create configuration
    config = GameConfiguration(
        max_turns_per_episode=500,
        zork_game_workdir=str(game_dir)
    )

    # Create game state
    game_state = GameState()
    game_state.episode_id = "ep01"
    game_state.turn_count = 25

    # Create logger mock
    logger = MagicMock()

    # Create manager
    manager = SimpleMemoryManager(
        logger=logger,
        config=config,
        game_state=game_state
    )

    return manager, game_state, game_dir


def test_invalidate_memory_single_success(setup_manager):
    """Test invalidate_memory() successfully invalidates a single memory."""
    manager, game_state, game_dir = setup_manager

    # Add initial memory
    memory = Memory(
        category="NOTE",
        title="Troll is friendly",
        episode=1,
        turns="20",
        score_change=0,
        text="Troll seems approachable.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    manager.add_memory(
        location_id=152,
        location_name="Troll Room",
        memory=memory
    )

    # Invalidate the memory
    success = manager.invalidate_memory(
        location_id=152,
        memory_title="Troll is friendly",
        reason="Proven false by death at turn 25",
        turn=25
    )

    # Assertions
    assert success is True

    # Verify memory was updated in file
    memories_path = game_dir / "Memories.md"
    content = memories_path.read_text()
    assert "[NOTE - PERMANENT - SUPERSEDED] Troll is friendly" in content  # With persistence marker
    assert '[Invalidated at T25: "Proven false by death at turn 25"]' in content


def test_invalidate_memory_default_turn(setup_manager):
    """Test invalidate_memory() defaults to current game turn."""
    manager, game_state, game_dir = setup_manager

    # Set game state turn
    game_state.turn_count = 42

    # Add initial memory
    memory = Memory(
        category="SUCCESS",
        title="Key unlocks door",
        episode=1,
        turns="30",
        score_change=5,
        text="Brass key opens the stone door.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    manager.add_memory(
        location_id=134,
        location_name="Hallway",
        memory=memory
    )

    # Invalidate without specifying turn
    success = manager.invalidate_memory(
        location_id=134,
        memory_title="Key unlocks door",
        reason="Door was already unlocked"
    )

    # Assertions
    assert success is True

    # Verify turn defaulted to game_state.turn_count
    memories_path = game_dir / "Memories.md"
    content = memories_path.read_text()
    assert "[Invalidated at T42:" in content


def test_invalidate_memory_empty_reason_fails(setup_manager):
    """Test invalidate_memory() fails when reason is empty."""
    manager, game_state, game_dir = setup_manager

    # Add initial memory
    memory = Memory(
        category="DANGER",
        title="Pit is deep",
        episode=1,
        turns="10",
        score_change=0,
        text="The pit looks very deep.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    manager.add_memory(
        location_id=100,
        location_name="Cellar",
        memory=memory
    )

    # Try to invalidate with empty reason
    success = manager.invalidate_memory(
        location_id=100,
        memory_title="Pit is deep",
        reason="",
        turn=15
    )

    # Assertions
    assert success is False

    # Verify memory was NOT updated
    memories_path = game_dir / "Memories.md"
    content = memories_path.read_text()
    assert "[DANGER - PERMANENT - SUPERSEDED]" not in content
    assert "[DANGER - PERMANENT] Pit is deep" in content  # Still ACTIVE (with PERMANENT marker)


def test_invalidate_memory_nonexistent_memory(setup_manager):
    """Test invalidate_memory() handles nonexistent memory gracefully."""
    manager, game_state, game_dir = setup_manager

    # Don't add any memories, try to invalidate one
    success = manager.invalidate_memory(
        location_id=999,
        memory_title="Nonexistent memory",
        reason="This memory doesn't exist",
        turn=20
    )

    # Assertions
    assert success is False


def test_invalidate_memories_batch_success(setup_manager):
    """Test invalidate_memories() successfully invalidates multiple memories."""
    manager, game_state, game_dir = setup_manager

    # Add multiple memories
    memories = [
        Memory(
            category="NOTE",
            title="Troll is friendly",
            episode=1,
            turns="20",
            score_change=0,
            text="Troll seems friendly.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE
        ),
        Memory(
            category="NOTE",
            title="Troll accepts gifts",
            episode=1,
            turns="21",
            score_change=0,
            text="Troll accepts lunch gift.",
            persistence="permanent",
            status=MemoryStatus.TENTATIVE
        )
    ]

    for memory in memories:
        manager.add_memory(
            location_id=152,
            location_name="Troll Room",
            memory=memory
        )

    # Batch invalidate
    results = manager.invalidate_memories(
        location_id=152,
        memory_titles=["Troll is friendly", "Troll accepts gifts"],
        reason="Both proven false by troll attack",
        turn=25
    )

    # Assertions
    assert results["Troll is friendly"] is True
    assert results["Troll accepts gifts"] is True

    # Verify both memories were updated
    memories_path = game_dir / "Memories.md"
    content = memories_path.read_text()
    assert "[NOTE - PERMANENT - SUPERSEDED] Troll is friendly" in content  # With persistence marker
    assert "[NOTE - PERMANENT - SUPERSEDED] Troll accepts gifts" in content  # With persistence marker
    assert '[Invalidated at T25: "Both proven false by troll attack"]' in content


def test_invalidate_memories_empty_list(setup_manager):
    """Test invalidate_memories() handles empty list gracefully."""
    manager, game_state, game_dir = setup_manager

    # Call with empty list
    results = manager.invalidate_memories(
        location_id=152,
        memory_titles=[],
        reason="Some reason",
        turn=25
    )

    # Assertions
    assert results == {}


def test_invalidate_memories_empty_reason(setup_manager):
    """Test invalidate_memories() fails when reason is empty."""
    manager, game_state, game_dir = setup_manager

    # Add memories
    memory = Memory(
        category="NOTE",
        title="Test memory",
        episode=1,
        turns="20",
        score_change=0,
        text="Test text.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    manager.add_memory(
        location_id=152,
        location_name="Troll Room",
        memory=memory
    )

    # Try to invalidate with empty reason
    results = manager.invalidate_memories(
        location_id=152,
        memory_titles=["Test memory"],
        reason="",
        turn=25
    )

    # Assertions
    assert results["Test memory"] is False


def test_invalidate_memories_partial_success(setup_manager):
    """Test invalidate_memories() returns per-memory status on partial success."""
    manager, game_state, game_dir = setup_manager

    # Add one memory
    memory = Memory(
        category="NOTE",
        title="Memory A",
        episode=1,
        turns="20",
        score_change=0,
        text="Test text A.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    manager.add_memory(
        location_id=152,
        location_name="Troll Room",
        memory=memory
    )

    # Try to invalidate one existing and one non-existing
    results = manager.invalidate_memories(
        location_id=152,
        memory_titles=["Memory A", "Memory B (nonexistent)"],
        reason="Testing partial success",
        turn=30
    )

    # Assertions
    assert results["Memory A"] is True
    assert results["Memory B (nonexistent)"] is False


def test_invalidate_memory_substring_match(setup_manager):
    """Test invalidate_memory() works with substring matching."""
    manager, game_state, game_dir = setup_manager

    # Add memory with longer title
    memory = Memory(
        category="SUCCESS",
        title="The brass key unlocks the stone door",
        episode=1,
        turns="20",
        score_change=5,
        text="Successfully unlocked door.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    manager.add_memory(
        location_id=134,
        location_name="Hallway",
        memory=memory
    )

    # Invalidate using substring
    success = manager.invalidate_memory(
        location_id=134,
        memory_title="brass key unlocks",
        reason="Key was wrong key",
        turn=25
    )

    # Assertions
    assert success is True

    # Verify memory was updated
    memories_path = game_dir / "Memories.md"
    content = memories_path.read_text()
    assert "[SUCCESS - PERMANENT - SUPERSEDED]" in content  # With persistence marker
    assert '[Invalidated at T25: "Key was wrong key"]' in content


def test_invalidate_memory_logging(setup_manager):
    """Test invalidate_memory() produces appropriate log messages."""
    manager, game_state, game_dir = setup_manager

    # Add memory
    memory = Memory(
        category="NOTE",
        title="Test memory",
        episode=1,
        turns="20",
        score_change=0,
        text="Test text.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    manager.add_memory(
        location_id=152,
        location_name="Troll Room",
        memory=memory
    )

    # Reset logger mock to clear add_memory calls
    manager.logger.reset_mock()

    # Invalidate memory
    manager.invalidate_memory(
        location_id=152,
        memory_title="Test memory",
        reason="Testing logging",
        turn=25
    )

    # Verify info log was called (multiple calls expected: one from _update_memory_status, one from invalidate_memory)
    assert manager.logger.info.called
    info_calls = [call[0][0] for call in manager.logger.info.call_args_list]
    assert any("Invalidated memory: 'Test memory'" in msg for msg in info_calls)


def test_invalidate_memories_logging_summary(setup_manager):
    """Test invalidate_memories() logs summary of batch operation."""
    manager, game_state, game_dir = setup_manager

    # Add memories
    for i in range(3):
        memory = Memory(
            category="NOTE",
            title=f"Memory {i}",
            episode=1,
            turns="20",
            score_change=0,
            text=f"Test text {i}.",
            persistence="permanent",
            status=MemoryStatus.ACTIVE
        )
        manager.add_memory(
            location_id=152,
            location_name="Troll Room",
            memory=memory
        )

    # Reset logger
    manager.logger.reset_mock()

    # Batch invalidate
    manager.invalidate_memories(
        location_id=152,
        memory_titles=["Memory 0", "Memory 1", "Memory 2"],
        reason="Batch test",
        turn=30
    )

    # Verify summary log was called
    info_calls = [call[0][0] for call in manager.logger.info.call_args_list]
    assert any("Batch invalidation: 3/3 succeeded" in msg for msg in info_calls)
