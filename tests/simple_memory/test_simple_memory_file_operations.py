"""
ABOUTME: Unit tests for SimpleMemoryManager file write operations.
ABOUTME: Tests writing memories to file, file safety, backup, atomicity, and locking.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
import threading

from tests.simple_memory.conftest import (
    SAMPLE_MEMORIES_SINGLE_LOCATION,
    SAMPLE_MEMORIES_FULL,
    SAMPLE_MEMORIES_EMPTY_FILE,
    Memory
)


class TestWriteNewMemoryToFile:
    """Test writing first memory to empty or non-existent file."""

    def test_write_first_memory_to_empty_file(self, mock_logger, game_config, game_state, sample_memory, create_memories_file):
        """Test writing first memory to an existing empty file."""
        # Create empty file
        create_memories_file(SAMPLE_MEMORIES_EMPTY_FILE)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Write first memory
        result = manager.add_memory(
            location_id=15,
            location_name="West of House",
            memory=sample_memory
        )

        # Should succeed
        assert result is True

        # Verify file content
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Should contain location header
        assert "## Location 15: West of House" in content

        # Should contain memory entry (with PERMANENT marker since persistence='permanent')
        assert "**[SUCCESS - PERMANENT] Open window**" in content
        assert "*(Ep1, T23, +0)*" in content
        assert "Window can be opened successfully." in content

    def test_write_memory_creates_file_if_not_exists(self, mock_logger, game_config, game_state, sample_memory):
        """Test that writing memory creates Memories.md if it doesn't exist."""
        # Don't create file - it should not exist

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"

        # File should not exist initially
        assert not memories_path.exists()

        # Write memory
        result = manager.add_memory(
            location_id=15,
            location_name="West of House",
            memory=sample_memory
        )

        # Should succeed
        assert result is True

        # File should now exist
        assert memories_path.exists()

        # Verify content (with PERMANENT marker since persistence='permanent')
        content = memories_path.read_text(encoding="utf-8")
        assert "## Location 15: West of House" in content
        assert "**[SUCCESS - PERMANENT] Open window**" in content

    def test_memory_formatted_correctly_in_markdown(self, mock_logger, game_config, game_state, sample_memory):
        """Test that memory entry is formatted correctly in markdown."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Write memory
        manager.add_memory(
            location_id=15,
            location_name="West of House",
            memory=sample_memory
        )

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Should match exact format: **[CATEGORY - PERMANENT] Title** *(metadata)*\ntext
        assert "**[SUCCESS - PERMANENT] Open window** *(Ep1, T23, +0)*" in content

        # Text should be on next line
        lines = content.split("\n")
        header_idx = next(i for i, line in enumerate(lines) if "**[SUCCESS - PERMANENT] Open window**" in line)
        text_line = lines[header_idx + 1]
        assert "Window can be opened successfully." in text_line

    def test_file_contains_proper_location_header(self, mock_logger, game_config, game_state, sample_memory):
        """Test that file contains properly formatted location header."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Write memory
        manager.add_memory(
            location_id=15,
            location_name="West of House",
            memory=sample_memory
        )

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Should have location header
        assert "## Location 15: West of House" in content

        # Should have visits and episodes metadata
        assert "**Visits:**" in content
        assert "**Episodes:**" in content

        # Should have Memories subheader
        assert "### Memories" in content

    def test_memory_entry_has_correct_format(self, mock_logger, game_config, game_state):
        """Test memory entry format with different metadata variations."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Test with turn range
        memory_range = Memory(
            category="SUCCESS",
            title="Open and enter window",
            episode=1,
            turns="23-24",
            score_change=0,
            text="Window can be opened with effort.",
            persistence="permanent"
        )

        manager.add_memory(15, "West of House", memory_range)

        # Test with positive score
        memory_score = Memory(
            category="SUCCESS",
            title="Acquire lamp",
            episode=2,
            turns="45",
            score_change=5,
            text="Lamp provides light.",
            persistence="permanent"
        )

        manager.add_memory(23, "Living Room", memory_score)

        # Test without score
        memory_no_score = Memory(
            category="DANGER",
            title="Deadly grue",
            episode=1,
            turns="100",
            score_change=None,
            text="Grue will kill you.",
            persistence="permanent"
        )

        manager.add_memory(10, "Dark Room", memory_no_score)

        # Read file and verify formats
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Turn range format
        assert "*(Ep1, T23-24, +0)*" in content

        # Positive score format
        assert "*(Ep2, T45, +5)*" in content

        # No score format (should not have score)
        assert "*(Ep1, T100)*" in content
        # Should NOT have +0 or any score indicator
        danger_section = content[content.find("DANGER"):]
        danger_line = danger_section.split("\n")[0]
        assert "+0" not in danger_line


class TestAppendToExistingLocationSection:
    """Test appending memories to existing location section."""

    def test_append_second_memory_to_same_location(self, mock_logger, game_config, game_state, create_memories_file):
        """Test appending a second memory to same location."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Verify initial state
        assert len(manager.memory_cache[15]) == 1

        # Add second memory to same location
        new_memory = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )

        result = manager.add_memory(15, "West of House", new_memory)
        assert result is True

        # Cache should have 2 memories now
        assert len(manager.memory_cache[15]) == 2

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Both memories should be present (with PERMANENT markers since persistence='permanent')
        assert "**[SUCCESS] Open window**" in content  # From sample data (old format, backward compat)
        assert "**[FAILURE - PERMANENT] Take window**" in content  # Newly written (new format)

    def test_memories_in_correct_order_newest_last(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that memories are in correct order with newest last."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add second memory
        memory2 = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", memory2)

        # Add third memory
        memory3 = Memory(
            category="DISCOVERY",
            title="Mailbox location",
            episode=1,
            turns="20",
            score_change=0,
            text="Mailbox contains leaflet.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", memory3)

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Find positions of each memory
        success_pos = content.find("**[SUCCESS] Open window**")
        failure_pos = content.find("**[FAILURE - PERMANENT] Take window**")  # Newly written
        discovery_pos = content.find("**[DISCOVERY - PERMANENT] Mailbox location**")  # Newly written

        # Should be in order: SUCCESS < FAILURE < DISCOVERY
        assert success_pos < failure_pos < discovery_pos

    def test_location_header_not_duplicated(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that location header is not duplicated when appending."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add second memory
        new_memory = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", new_memory)

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Count occurrences of location header
        header_count = content.count("## Location 15: West of House")
        assert header_count == 1, f"Expected 1 location header, found {header_count}"

    def test_existing_memories_preserved(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that existing memories are preserved when appending."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Get original content
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        original_content = memories_path.read_text(encoding="utf-8")

        # Add new memory
        new_memory = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", new_memory)

        # Read updated content
        updated_content = memories_path.read_text(encoding="utf-8")

        # Original memory should still be present
        assert "**[SUCCESS] Open window**" in updated_content
        assert "Window can be opened successfully." in updated_content


class TestCreateNewLocationSection:
    """Test adding memory to new location when file has other locations."""

    def test_add_memory_to_new_location(self, mock_logger, game_config, game_state, create_memories_file):
        """Test adding memory to a new location when file has other locations."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add memory to new location
        new_memory = Memory(
            category="SUCCESS",
            title="Acquire lamp",
            episode=1,
            turns="45",
            score_change=5,
            text="Lamp provides light.",
            persistence="permanent"
        )

        result = manager.add_memory(23, "Living Room", new_memory)
        assert result is True

        # Cache should have both locations
        assert 15 in manager.memory_cache
        assert 23 in manager.memory_cache

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Both location headers should be present
        assert "## Location 15: West of House" in content
        assert "## Location 23: Living Room" in content

    def test_new_location_section_added_at_end(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that new location section is added at end of file."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add memory to new location
        new_memory = Memory(
            category="SUCCESS",
            title="Acquire lamp",
            episode=1,
            turns="45",
            score_change=5,
            text="Lamp provides light.",
            persistence="permanent"
        )
        manager.add_memory(23, "Living Room", new_memory)

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Find positions
        location_15_pos = content.find("## Location 15: West of House")
        location_23_pos = content.find("## Location 23: Living Room")

        # Location 23 should be after location 15
        assert location_23_pos > location_15_pos

    def test_proper_separation_between_location_sections(self, mock_logger, game_config, game_state, create_memories_file):
        """Test proper markdown separation between location sections."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Add memory to new location
        new_memory = Memory(
            category="SUCCESS",
            title="Acquire lamp",
            episode=1,
            turns="45",
            score_change=5,
            text="Lamp provides light.",
            persistence="permanent"
        )
        manager.add_memory(23, "Living Room", new_memory)

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Should have separator (---) between sections
        sections = content.split("---")
        assert len(sections) >= 2, "Should have at least one separator between sections"

    def test_existing_locations_not_affected(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that existing locations are not affected by adding new location."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Get original location 15 memories
        original_location_15_count = len(manager.memory_cache[15])
        original_location_23_count = len(manager.memory_cache[23])

        # Add memory to new location
        new_memory = Memory(
            category="DANGER",
            title="Deadly grue",
            episode=1,
            turns="100",
            score_change=None,
            text="Grue will kill you.",
            persistence="permanent"
        )
        manager.add_memory(10, "Dark Room", new_memory)

        # Original locations should be unchanged
        assert len(manager.memory_cache[15]) == original_location_15_count
        assert len(manager.memory_cache[23]) == original_location_23_count

        # New location should exist
        assert 10 in manager.memory_cache
        assert len(manager.memory_cache[10]) == 1


class TestFileBackupMechanism:
    """Test backup file creation before write operations."""

    def test_backup_file_created_before_write(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that backup file is created before writing."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        backup_path = Path(game_config.zork_game_workdir) / "Memories.md.backup"

        # Backup should not exist yet
        assert not backup_path.exists()

        # Write memory
        new_memory = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", new_memory)

        # Backup should now exist
        assert backup_path.exists()

    def test_backup_contains_old_content(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that backup contains the old content before write."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Get original content
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        original_content = memories_path.read_text(encoding="utf-8")

        # Write memory
        new_memory = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", new_memory)

        # Read backup
        backup_path = Path(game_config.zork_game_workdir) / "Memories.md.backup"
        backup_content = backup_path.read_text(encoding="utf-8")

        # Backup should match original
        assert backup_content == original_content

    def test_backup_not_created_if_original_doesnt_exist(self, mock_logger, game_config, game_state):
        """Test that backup is not created if original file doesn't exist."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        backup_path = Path(game_config.zork_game_workdir) / "Memories.md.backup"
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"

        # Neither should exist initially
        assert not memories_path.exists()
        assert not backup_path.exists()

        # Write memory
        new_memory = Memory(
            category="SUCCESS",
            title="Open window",
            episode=1,
            turns="23",
            score_change=0,
            text="Window can be opened.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", new_memory)

        # Memories.md should exist now
        assert memories_path.exists()

        # Backup should NOT exist (no original to back up)
        assert not backup_path.exists()

    def test_backup_overwritten_on_each_write(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that backup is overwritten on each write operation."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        backup_path = Path(game_config.zork_game_workdir) / "Memories.md.backup"

        # First write
        memory1 = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", memory1)

        # Read backup after first write
        backup_after_first = backup_path.read_text(encoding="utf-8")

        # Second write
        memory2 = Memory(
            category="DISCOVERY",
            title="Mailbox",
            episode=1,
            turns="20",
            score_change=0,
            text="Mailbox contains leaflet.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", memory2)

        # Read backup after second write
        backup_after_second = backup_path.read_text(encoding="utf-8")

        # Backups should be different
        assert backup_after_first != backup_after_second

        # Second backup should contain memory1 but not memory2
        assert "Take window" in backup_after_second
        assert "Mailbox" not in backup_after_second


class TestAtomicCacheUpdates:
    """Test that cache is updated atomically with file writes."""

    def test_cache_updated_immediately_after_write(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that cache is updated immediately after successful write."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Initial cache state
        assert len(manager.memory_cache[15]) == 1

        # Write memory
        new_memory = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        result = manager.add_memory(15, "West of House", new_memory)

        # Should succeed
        assert result is True

        # Cache should be updated immediately
        assert len(manager.memory_cache[15]) == 2

        # New memory should be in cache
        cached_memory = manager.memory_cache[15][-1]
        assert cached_memory.title == "Take window"

    def test_cache_matches_file_content_after_write(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that cache matches file content after write."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Write memory
        new_memory = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", new_memory)

        # Create new manager to read from file
        manager2 = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Cache should match between both managers
        assert len(manager.memory_cache[15]) == len(manager2.memory_cache[15])

        # Memories should match
        for m1, m2 in zip(manager.memory_cache[15], manager2.memory_cache[15]):
            assert m1.category == m2.category
            assert m1.title == m2.title
            assert m1.episode == m2.episode
            assert m1.turns == m2.turns
            assert m1.score_change == m2.score_change
            assert m1.text == m2.text

    def test_cache_consistent_even_if_write_fails(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that cache remains consistent even if write operation fails."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Get initial cache state
        initial_cache_count = len(manager.memory_cache[15])

        # Mock Path.write_text to fail
        with patch("pathlib.Path.write_text", side_effect=IOError("Disk full")):
            new_memory = Memory(
                category="FAILURE",
                title="Take window",
                episode=1,
                turns="25",
                score_change=None,
                text="Window cannot be taken.",
            persistence="permanent"
            )
            result = manager.add_memory(15, "West of House", new_memory)

            # Should fail
            assert result is False

        # Cache should NOT be updated (rollback)
        assert len(manager.memory_cache[15]) == initial_cache_count

    def test_reading_from_cache_reflects_new_memories(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that reading from cache reflects newly added memories."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Write memory
        new_memory = Memory(
            category="FAILURE",
            title="Take window",
            episode=1,
            turns="25",
            score_change=None,
            text="Window cannot be taken.",
            persistence="permanent"
        )
        manager.add_memory(15, "West of House", new_memory)

        # Get memories from cache (Phase 2 will have a method for this)
        # For now, directly access cache
        memories = manager.memory_cache[15]

        # Should have 2 memories
        assert len(memories) == 2

        # New memory should be last
        assert memories[-1].title == "Take window"


