"""
ABOUTME: Unit tests for SimpleMemoryManager memory status handling.
ABOUTME: Tests ACTIVE/TENTATIVE/SUPERSEDED status filtering, parsing, and updates.
"""

import pytest
from pathlib import Path

from tests.simple_memory.conftest import Memory, MemoryStatus


class TestMemoryStatusFiltering:
    """Test get_location_memory() filters memories by status (Step 5)."""

    def test_active_memories_shown_normally(self, mock_logger, game_config, game_state):
        """Test ACTIVE memories are displayed without markers."""
        from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Create ACTIVE memory
        active_mem = Memory(
            category="SUCCESS",
            title="Active Memory",
            episode=1,
            turns="10",
            score_change=5,
            text="This is an active memory.",
            status=MemoryStatus.ACTIVE
        )

        # Add to cache
        location_id = 15
        manager.memory_cache[location_id] = [active_mem]

        # Get output
        output = manager.get_location_memory(location_id)

        # Verify
        assert "Active Memory" in output
        assert "[SUCCESS]" in output
        assert "This is an active memory." in output
        assert "⚠️" not in output  # No warning marker for ACTIVE

    def test_tentative_memories_shown_with_warning(self, mock_logger, game_config, game_state):
        """Test TENTATIVE memories are shown with warning marker."""
        from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Create TENTATIVE memory
        tentative_mem = Memory(
            category="DISCOVERY",
            title="Tentative Memory",
            episode=1,
            turns="11",
            score_change=0,
            text="This is a tentative memory.",
            status=MemoryStatus.TENTATIVE
        )

        # Add to cache
        location_id = 15
        manager.memory_cache[location_id] = [tentative_mem]

        # Get output
        output = manager.get_location_memory(location_id)

        # Verify
        assert "Tentative Memory" in output
        assert "⚠️  TENTATIVE MEMORIES" in output
        assert "unconfirmed, may be invalidated" in output
        assert "[DISCOVERY]" in output

    def test_superseded_memories_hidden(self, mock_logger, game_config, game_state):
        """Test SUPERSEDED memories are completely hidden."""
        from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Create SUPERSEDED memory
        superseded_mem = Memory(
            category="FAILURE",
            title="Superseded Memory",
            episode=1,
            turns="12",
            score_change=-5,
            text="This memory was proven wrong.",
            status=MemoryStatus.SUPERSEDED
        )

        # Add to cache
        location_id = 15
        manager.memory_cache[location_id] = [superseded_mem]

        # Get output
        output = manager.get_location_memory(location_id)

        # Verify - should be empty
        assert output == ""
        assert "Superseded Memory" not in output

    def test_mixed_status_memories_filtered_correctly(self, mock_logger, game_config, game_state):
        """Test mix of ACTIVE, TENTATIVE, and SUPERSEDED memories."""
        from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Create memories with different statuses
        active_mem = Memory(
            category="SUCCESS",
            title="Active Memory",
            episode=1,
            turns="10",
            score_change=5,
            text="This is active.",
            status=MemoryStatus.ACTIVE
        )

        tentative_mem = Memory(
            category="DISCOVERY",
            title="Tentative Memory",
            episode=1,
            turns="11",
            score_change=0,
            text="This is tentative.",
            status=MemoryStatus.TENTATIVE
        )

        superseded_mem = Memory(
            category="FAILURE",
            title="Superseded Memory",
            episode=1,
            turns="12",
            score_change=-5,
            text="This was proven wrong.",
            status=MemoryStatus.SUPERSEDED
        )

        # Add to cache
        location_id = 15
        manager.memory_cache[location_id] = [active_mem, tentative_mem, superseded_mem]

        # Get output
        output = manager.get_location_memory(location_id)

        # Verify ACTIVE shown
        assert "Active Memory" in output
        assert "This is active." in output

        # Verify TENTATIVE shown with warning
        assert "Tentative Memory" in output
        assert "This is tentative." in output
        assert "⚠️  TENTATIVE MEMORIES" in output

        # Verify SUPERSEDED hidden
        assert "Superseded Memory" not in output
        assert "This was proven wrong." not in output

    def test_empty_location_returns_empty_string(self, mock_logger, game_config, game_state):
        """Test location with no memories returns empty string."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Get output for non-existent location
        output = manager.get_location_memory(999)

        # Verify empty
        assert output == ""

    def test_blank_line_separator_between_active_and_tentative(self, mock_logger, game_config, game_state):
        """Test blank line separates ACTIVE and TENTATIVE sections."""
        from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Create ACTIVE and TENTATIVE memories
        active_mem = Memory(
            category="SUCCESS",
            title="Active",
            episode=1,
            turns="10",
            score_change=5,
            text="Active.",
            status=MemoryStatus.ACTIVE
        )

        tentative_mem = Memory(
            category="DISCOVERY",
            title="Tentative",
            episode=1,
            turns="11",
            score_change=0,
            text="Tentative.",
            status=MemoryStatus.TENTATIVE
        )

        # Add to cache
        location_id = 15
        manager.memory_cache[location_id] = [active_mem, tentative_mem]

        # Get output
        output = manager.get_location_memory(location_id)

        # Verify blank line between sections
        lines = output.split("\n")
        assert len(lines) >= 3  # At least: active line, blank line, tentative header
        assert any(line == "" for line in lines)  # Has blank line


class TestMemoryStatusParsing:
    """Test parsing of memory status from file format."""

    def test_parse_active_status_default(self, mock_logger, game_config, game_state):
        """Old format without status should default to ACTIVE."""
        from managers.simple_memory_manager import SimpleMemoryManager, MemoryStatus

        content = """# Location Memories

## Location 15: West of House
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Open window** *(Ep1, T23, +0)*
Window can be opened successfully.

---
"""
        # Create temp file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")

        # Parse
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Verify
        assert 15 in manager.memory_cache
        memories = manager.memory_cache[15]
        assert len(memories) == 1
        assert memories[0].status == MemoryStatus.ACTIVE
        assert memories[0].title == "Open window"

    def test_parse_tentative_status(self, mock_logger, game_config, game_state):
        """Parse memory with TENTATIVE status."""
        from managers.simple_memory_manager import SimpleMemoryManager, MemoryStatus

        content = """# Location Memories

## Location 102: Troll Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS - TENTATIVE] Troll accepts gift** *(Ep1, T45, +0)*
Troll accepted lunch but reaction uncertain.

---
"""
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        assert 102 in manager.memory_cache
        memories = manager.memory_cache[102]
        assert len(memories) == 1
        assert memories[0].status == MemoryStatus.TENTATIVE
        assert memories[0].title == "Troll accepts gift"

    def test_parse_superseded_status_with_strikethrough(self, mock_logger, game_config, game_state):
        """Parse memory with SUPERSEDED status and strikethrough text."""
        from managers.simple_memory_manager import SimpleMemoryManager, MemoryStatus

        content = """# Location Memories

## Location 102: Troll Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[DISCOVERY - SUPERSEDED] Troll is friendly** *(Ep1, T40, +0)*
[Superseded at T45 by "Troll attacks after accepting gifts"]
~~Troll accepts gifts without attacking immediately.~~

---
"""
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        assert 102 in manager.memory_cache
        memories = manager.memory_cache[102]
        assert len(memories) == 1
        assert memories[0].status == MemoryStatus.SUPERSEDED
        assert memories[0].title == "Troll is friendly"
        # Strikethrough should be stripped from text
        assert "~~" not in memories[0].text
        assert "Troll accepts gifts without attacking immediately" in memories[0].text

    def test_skip_supersession_reference_line(self, mock_logger, game_config, game_state):
        """Supersession reference line should not be included in memory text."""
        from managers.simple_memory_manager import SimpleMemoryManager

        content = """# Location Memories

## Location 102: Troll Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[DISCOVERY - SUPERSEDED] Troll is friendly** *(Ep1, T40, +0)*
[Superseded at T45 by "Troll attacks after accepting gifts"]
~~Troll seems peaceful.~~

---
"""
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        memories = manager.memory_cache[102]
        # Supersession reference should NOT be in memory text
        assert "[Superseded at" not in memories[0].text
        assert "Troll attacks after accepting gifts" not in memories[0].text
        assert "Troll seems peaceful" in memories[0].text


class TestMemoryStatusUpdates:
    """Test updating memory status in file and cache."""

    def test_update_memory_status_to_superseded(self, mock_logger, game_config, game_state):
        """Test updating a memory status to SUPERSEDED."""
        from managers.simple_memory_manager import SimpleMemoryManager, MemoryStatus

        # Setup: Create initial memory
        content = """# Location Memories

## Location 102: Troll Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Troll accepts lunch gift** *(Ep1, T45, +0)*
Troll accepted lunch without attacking.

---
"""
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Update status
        success = manager._update_memory_status(
            location_id=102,
            memory_title="Troll accepts lunch gift",
            new_status=MemoryStatus.SUPERSEDED,
            superseded_by="Troll attacks after accepting gifts",
            superseded_at_turn=50
        )

        assert success is True

        # Verify cache updated
        memories = manager.memory_cache[102]
        assert memories[0].status == MemoryStatus.SUPERSEDED
        assert memories[0].superseded_by == "Troll attacks after accepting gifts"
        assert memories[0].superseded_at_turn == 50

        # Verify file updated
        updated_content = memories_path.read_text(encoding="utf-8")
        assert "**[SUCCESS - SUPERSEDED] Troll accepts lunch gift**" in updated_content
        assert '[Superseded at T50 by "Troll attacks after accepting gifts"]' in updated_content
        assert "~~Troll accepted lunch without attacking.~~" in updated_content

    def test_update_nonexistent_memory_returns_false(self, mock_logger, game_config, game_state):
        """Test updating a memory that doesn't exist returns False."""
        from managers.simple_memory_manager import SimpleMemoryManager, MemoryStatus

        content = """# Location Memories

## Location 102: Troll Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Some other memory** *(Ep1, T45, +0)*
Some text.

---
"""
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state
        )

        # Try to update non-existent memory
        success = manager._update_memory_status(
            location_id=102,
            memory_title="This memory does not exist",
            new_status=MemoryStatus.SUPERSEDED,
            superseded_by="New memory",
            superseded_at_turn=50
        )

        assert success is False
