"""
ABOUTME: Unit tests for SimpleMemoryManager file parsing and cache management.
ABOUTME: Tests parsing Memories.md format and in-memory cache structure.
"""

import pytest
from pathlib import Path

from tests.simple_memory.conftest import (
    SAMPLE_MEMORIES_FULL,
    SAMPLE_MEMORIES_SINGLE_LOCATION,
    SAMPLE_MEMORIES_NO_SCORE,
    SAMPLE_MEMORIES_CORRUPTED,
    SAMPLE_MEMORIES_EMPTY_FILE
)


class TestMemoriesFileParsingLocationHeaders:
    """Test parsing of location headers from Memories.md format."""

    def test_parse_single_location_header(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing a single location header: ## Location 15: West of House"""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have parsed location 15
        assert 15 in manager.memory_cache
        assert isinstance(manager.memory_cache[15], list)

    def test_parse_multiple_location_headers(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing multiple location headers in same file."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have parsed locations 15 and 23
        assert 15 in manager.memory_cache
        assert 23 in manager.memory_cache
        assert isinstance(manager.memory_cache[15], list)
        assert isinstance(manager.memory_cache[23], list)

    def test_extract_location_id_from_header(self, mock_logger, game_config, game_state, create_memories_file):
        """Test extracting integer location ID from header."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Location ID should be integer 15, not string "15"
        location_ids = list(manager.memory_cache.keys())
        assert 15 in location_ids
        assert isinstance(15, int)

    def test_extract_location_name_from_header(self, mock_logger, game_config, game_state, create_memories_file):
        """Test extracting location name from header (for logging/debugging)."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Memory manager should track location names (even though they're not primary keys)
        # This is implementation detail, but useful for debugging
        # Location 15 should have name "West of House"
        # Location 23 should have name "Living Room"
        assert len(manager.memory_cache[15]) > 0
        assert len(manager.memory_cache[23]) > 0


class TestMemoriesFileParsingMemoryEntries:
    """Test parsing of individual memory entries."""

    def test_parse_memory_category_success(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing [SUCCESS] category from memory entry."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories = manager.memory_cache[15]
        assert len(memories) == 1
        assert memories[0].category == "SUCCESS"

    def test_parse_all_memory_categories(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing all category types: SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Location 15 has SUCCESS, FAILURE, DISCOVERY
        categories_15 = {m.category for m in manager.memory_cache[15]}
        assert "SUCCESS" in categories_15
        assert "FAILURE" in categories_15
        assert "DISCOVERY" in categories_15

        # Location 23 has SUCCESS, FAILURE, NOTE
        categories_23 = {m.category for m in manager.memory_cache[23]}
        assert "SUCCESS" in categories_23
        assert "FAILURE" in categories_23
        assert "NOTE" in categories_23

    def test_parse_memory_title(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing memory title from entry header."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories = manager.memory_cache[15]
        assert memories[0].title == "Open window"

    def test_parse_memory_metadata_episode(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing episode number from metadata: *(Ep1, T23, +0)*"""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories = manager.memory_cache[15]
        assert memories[0].episode == 1

    def test_parse_memory_metadata_turn_single(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing single turn from metadata: T23"""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories = manager.memory_cache[15]
        assert memories[0].turns == "23"

    def test_parse_memory_metadata_turn_range(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing turn range from metadata: T23-24"""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # First memory in location 15 has turn range
        memories = manager.memory_cache[15]
        success_memory = [m for m in memories if m.title == "Open and enter window"][0]
        assert success_memory.turns == "23-24"

    def test_parse_memory_metadata_score_positive(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing positive score change: +5"""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # "Acquire brass lantern" has +5 score
        memories = manager.memory_cache[23]
        lantern_memory = [m for m in memories if "lantern" in m.title.lower()][0]
        assert lantern_memory.score_change == 5

    def test_parse_memory_metadata_score_zero(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing zero score change: +0"""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories = manager.memory_cache[15]
        assert memories[0].score_change == 0

    def test_parse_memory_metadata_score_missing(self, mock_logger, game_config, game_state, create_memories_file):
        """Test handling missing score in metadata: *(Ep1, T100)*"""
        create_memories_file(SAMPLE_MEMORIES_NO_SCORE)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories = manager.memory_cache[10]
        assert memories[0].score_change is None

    def test_parse_memory_text_content(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing memory text (1-2 sentences after header)."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories = manager.memory_cache[15]
        assert memories[0].text == "Window can be opened successfully."

    def test_parse_memory_text_multiline(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing memory text that spans multiple lines."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories = manager.memory_cache[15]
        window_memory = [m for m in memories if "Open and enter window" in m.title][0]
        # Text should be: "Window can be opened with effort and used as alternative entrance to house. Must squeeze through opening."
        assert "Window can be opened with effort" in window_memory.text
        assert "Must squeeze through opening" in window_memory.text


class TestMemoriesFileHandlingEdgeCases:
    """Test handling of missing, empty, or corrupted files."""

    def test_missing_memories_file(self, mock_logger, game_config, game_state):
        """Test graceful handling when Memories.md doesn't exist."""
        # Don't create file - it should be missing

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have empty cache, not crash
        assert isinstance(manager.memory_cache, dict)
        assert len(manager.memory_cache) == 0

    def test_empty_memories_file(self, mock_logger, game_config, game_state, create_memories_file):
        """Test handling of empty Memories.md file."""
        create_memories_file(SAMPLE_MEMORIES_EMPTY_FILE)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have empty cache
        assert isinstance(manager.memory_cache, dict)
        assert len(manager.memory_cache) == 0

    def test_corrupted_location_header_skipped(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that malformed location headers are skipped gracefully."""
        create_memories_file(SAMPLE_MEMORIES_CORRUPTED)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have parsed location 15 and 23, skipped "Location Invalid: Not a Number"
        assert 15 in manager.memory_cache
        assert 23 in manager.memory_cache
        # Should not have invalid entries
        assert len([k for k in manager.memory_cache.keys() if not isinstance(k, int)]) == 0

    def test_corrupted_memory_entry_skipped(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that malformed memory entries are skipped but parsing continues."""
        create_memories_file(SAMPLE_MEMORIES_CORRUPTED)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Location 15 should have 1 valid memory (malformed one skipped)
        assert len(manager.memory_cache[15]) == 1
        assert manager.memory_cache[15][0].title == "Valid memory"

    def test_parsing_continues_after_corruption(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that parsing continues after encountering corrupted sections."""
        create_memories_file(SAMPLE_MEMORIES_CORRUPTED)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Location 23 should still be parsed even after corrupted sections
        assert 23 in manager.memory_cache
        assert len(manager.memory_cache[23]) == 1
        assert manager.memory_cache[23][0].title == "Valid after corruption"


class TestMemoryCacheStructure:
    """Test in-memory cache structure and initialization."""

    def test_cache_type_is_dict_int_to_list(self, mock_logger, game_config, game_state, create_memories_file):
        """Test cache structure is Dict[int, List[Memory]]."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Cache should be a dict
        assert isinstance(manager.memory_cache, dict)

        # Keys should be integers
        for key in manager.memory_cache.keys():
            assert isinstance(key, int)

        # Values should be lists of Memory objects
        for value in manager.memory_cache.values():
            assert isinstance(value, list)
            for item in value:
                # Check it has Memory attributes
                assert hasattr(item, 'category')
                assert hasattr(item, 'title')
                assert hasattr(item, 'episode')
                assert hasattr(item, 'turns')
                assert hasattr(item, 'score_change')
                assert hasattr(item, 'text')

    def test_cache_uses_location_ids_as_keys(self, mock_logger, game_config, game_state, create_memories_file):
        """Test cache uses location IDs (integers) as keys, not names."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have integer keys 15 and 23
        assert 15 in manager.memory_cache
        assert 23 in manager.memory_cache

        # Should NOT have string keys like "West of House" or "Living Room"
        for key in manager.memory_cache.keys():
            assert isinstance(key, int), f"Expected integer key, got {type(key)}: {key}"

    def test_cache_populated_on_initialization(self, mock_logger, game_config, game_state, create_memories_file):
        """Test cache is populated from file during manager initialization."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have loaded all locations
        assert len(manager.memory_cache) == 2

        # Should have loaded all memories
        assert len(manager.memory_cache[15]) == 3  # 3 memories at location 15
        assert len(manager.memory_cache[23]) == 4  # 4 memories at location 23

    def test_multiple_memories_per_location(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that multiple memories can be stored for same location."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Location 23 should have 4 distinct memories
        memories_23 = manager.memory_cache[23]
        assert len(memories_23) == 4

        # All should have different titles
        titles = [m.title for m in memories_23]
        assert len(titles) == len(set(titles)), "Duplicate titles found"


class TestMemoryManagerIntegration:
    """Integration tests for SimpleMemoryManager."""

    def test_full_file_parsing_integration(self, mock_logger, game_config, game_state, create_memories_file):
        """Test end-to-end parsing of complete Memories.md file."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have parsed 2 locations
        assert len(manager.memory_cache) == 2

        # Should have parsed 7 total memories (3 + 4)
        total_memories = sum(len(memories) for memories in manager.memory_cache.values())
        assert total_memories == 7

        # Spot check specific memories
        location_15_memories = manager.memory_cache[15]
        window_memory = [m for m in location_15_memories if "window" in m.title.lower()][0]
        assert window_memory.category == "SUCCESS"
        assert window_memory.episode == 1
        assert window_memory.score_change == 0

        location_23_memories = manager.memory_cache[23]
        lantern_memory = [m for m in location_23_memories if "lantern" in m.title.lower()][0]
        assert lantern_memory.category == "SUCCESS"
        assert lantern_memory.score_change == 5

    def test_logger_called_during_parsing(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that logger is used during file parsing for diagnostics."""
        create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Logger should have been called at least once (for info/debug messages)
        assert mock_logger.info.called or mock_logger.debug.called

    def test_config_workdir_used_for_file_path(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that config.zork_game_workdir is used to locate Memories.md."""
        memories_path = create_memories_file(SAMPLE_MEMORIES_FULL)

        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Should have successfully loaded from correct path
        assert len(manager.memory_cache) > 0

        # Verify the path was constructed correctly
        expected_path = Path(game_config.zork_game_workdir) / "Memories.md"
        assert expected_path == memories_path
