"""
Test memory deduplication in add_memory()

Verifies that exact duplicate titles are rejected to prevent
LLM hallucination from creating redundant memories.
"""
import tempfile
from pathlib import Path
import pytest
from managers.simple_memory_manager import SimpleMemoryManager
from managers.memory import Memory
from session.game_state import GameState
from session.game_configuration import GameConfiguration


def test_exact_duplicate_title_rejected(mock_logger, game_state, test_config):
    """Test that exact duplicate titles are rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_config = GameConfiguration(
            max_turns_per_episode=100,
            zork_game_workdir=str(tmp_path)
        )

        manager = SimpleMemoryManager(mock_logger, test_config, game_state)

        # Create first memory
        memory1 = Memory(
            category="DISCOVERY",
            title="Platinum bar in Loud Room",
            episode=1,
            turns="10",
            score_change=0,
            text="A large platinum bar is on the ground.",
            persistence="permanent",
            status="ACTIVE"
        )

        # Add first memory - should succeed
        success1 = manager.add_memory(40, "Deep Canyon", memory1)
        assert success1, "First memory should be added successfully"

        # Create exact duplicate (same title)
        memory2 = Memory(
            category="DISCOVERY",
            title="Platinum bar in Loud Room",  # EXACT DUPLICATE
            episode=1,
            turns="25",
            score_change=0,
            text="The platinum bar is still here.",
            persistence="permanent",
            status="ACTIVE"
        )

        # Try to add duplicate - should be REJECTED
        success2 = manager.add_memory(40, "Deep Canyon", memory2)
        assert not success2, "Duplicate memory should be rejected"

        # Verify only ONE memory exists
        memories = manager.cache_manager.get_from_cache(40, persistent=None, include_superseded=False)
        assert len(memories) == 1, f"Should have exactly 1 memory, got {len(memories)}"
        assert memories[0].title == "Platinum bar in Loud Room"
        assert memories[0].turns == "10", "Should keep first memory, not duplicate"


def test_different_titles_both_accepted(mock_logger, game_state, test_config):
    """Test that different titles are both accepted."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_config = GameConfiguration(
            max_turns_per_episode=100,
            zork_game_workdir=str(tmp_path)
        )

        manager = SimpleMemoryManager(mock_logger, test_config, game_state)

        memory1 = Memory(
            category="DISCOVERY",
            title="Platinum bar in Loud Room",
            episode=1,
            turns="10",
            score_change=0,
            text="A large platinum bar is on the ground.",
            persistence="permanent",
            status="ACTIVE"
        )

        memory2 = Memory(
            category="DISCOVERY",
            title="Room is extremely loud",  # DIFFERENT title
            episode=1,
            turns="11",
            score_change=0,
            text="The room has deafening noise.",
            persistence="permanent",
            status="ACTIVE"
        )

        success1 = manager.add_memory(40, "Deep Canyon", memory1)
        success2 = manager.add_memory(40, "Deep Canyon", memory2)

        assert success1, "First memory should be added"
        assert success2, "Second memory with different title should be added"

        memories = manager.cache_manager.get_from_cache(40, persistent=None, include_superseded=False)
        assert len(memories) == 2, f"Should have 2 memories, got {len(memories)}"


def test_duplicate_allowed_if_first_superseded(mock_logger, game_state, test_config):
    """Test that same title is allowed if first was superseded."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_config = GameConfiguration(
            max_turns_per_episode=100,
            zork_game_workdir=str(tmp_path)
        )

        manager = SimpleMemoryManager(mock_logger, test_config, game_state)

        # Add first memory
        memory1 = Memory(
            category="NOTE",
            title="Troll might be friendly",
            episode=1,
            turns="10",
            score_change=0,
            text="Troll seems calm.",
            persistence="permanent",
            status="ACTIVE"
        )

        manager.add_memory(152, "Troll Room", memory1)

        # Supersede it
        memory2 = Memory(
            category="DANGER",
            title="Troll attacks on sight",
            episode=1,
            turns="15",
            score_change=0,
            text="Troll is hostile.",
            persistence="permanent",
            status="ACTIVE"
        )

        manager.supersede_memory(152, "Troll Room", "Troll might be friendly", memory2)

        # Now try to add memory with SAME title as the superseded one
        # This should be REJECTED because we check for ACTIVE/TENTATIVE only
        # But if we want to re-add after supersession, we'd need to check only ACTIVE
        memory3 = Memory(
            category="NOTE",
            title="Troll might be friendly",  # Same as superseded
            episode=1,
            turns="20",
            score_change=0,
            text="Maybe troll is friendly after all?",
            persistence="permanent",
            status="TENTATIVE"
        )

        # This should succeed because original is SUPERSEDED (not checked)
        success = manager.add_memory(152, "Troll Room", memory3)
        assert success, "Should allow same title if original was superseded"
