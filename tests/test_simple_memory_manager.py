"""
ABOUTME: Unit tests for SimpleMemoryManager - location-based memory system.
ABOUTME: Tests parsing Memories.md format and in-memory cache management.

This module tests Phase 1 of the Simple Memory System, which provides:
- Parsing of Memories.md file format with location headers and memory entries
- In-memory cache using Dict[int, List[Memory]] structure
- Graceful handling of missing, empty, or corrupted files
- Memory metadata parsing (episode, turns, score changes)
- Memory category extraction (SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE)
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal

from session.game_state import GameState
from session.game_configuration import GameConfiguration


# Memory status constants (must match manager implementation)
MemoryStatusType = Literal["ACTIVE", "TENTATIVE", "SUPERSEDED"]

class MemoryStatus:
    """Memory status constants."""
    ACTIVE: MemoryStatusType = "ACTIVE"
    TENTATIVE: MemoryStatusType = "TENTATIVE"
    SUPERSEDED: MemoryStatusType = "SUPERSEDED"


# Memory dataclass for testing (must match manager implementation)
@dataclass
class Memory:
    """Represents a single location memory entry."""
    category: str  # SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE
    title: str  # Short title of the memory
    episode: int  # Episode number
    turns: str  # Turn range (e.g., "23-24" or "23")
    score_change: Optional[int]  # Score change (+5, +0, None if not specified)
    text: str  # 1-2 sentence synthesized insight
    status: MemoryStatusType = MemoryStatus.ACTIVE  # Memory status
    superseded_by: Optional[str] = None  # Title of memory that superseded this
    superseded_at_turn: Optional[int] = None  # Turn when superseded


# Sample Memories.md content for testing
SAMPLE_MEMORIES_FULL = """# Location Memories

## Location 15: West of House
**Visits:** 3 | **Episodes:** 1, 2, 3

### Memories

**[SUCCESS] Open and enter window** *(Ep1, T23-24, +0)*
Window can be opened with effort and used as alternative entrance to house. Must squeeze through opening.

**[FAILURE] Take or break window** *(Ep1, T25-26)*
Window is part of house structure - cannot be taken, moved, or broken. Violence not effective.

**[DISCOVERY] Mailbox location** *(Ep1, T20, +0)*
Small mailbox located here contains advertising leaflet. Likely tutorial document.

---

## Location 23: Living Room
**Visits:** 5 | **Episodes:** 1, 2, 3, 4

### Memories

**[SUCCESS] Acquire brass lantern** *(Ep1, T45, +5)*
Brass lantern is takeable and provides light source. CRITICAL item for dark areas - always take before exploring.

**[SUCCESS] Light lantern** *(Ep1, T46, +0)*
Lantern can be lit with simple command. Enables safe navigation of dark rooms.

**[FAILURE] Take sword** *(Ep1, T47)*
Ornamental sword is securely mounted and cannot be taken directly. Likely requires puzzle solution.

**[NOTE] Navigation options** *(Ep1, T50, +0)*
West exit leads to Kitchen. Room serves as central hub with multiple exits.

---
"""

SAMPLE_MEMORIES_SINGLE_LOCATION = """# Location Memories

## Location 15: West of House
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Open window** *(Ep1, T23, +0)*
Window can be opened successfully.

---
"""

SAMPLE_MEMORIES_NO_SCORE = """# Location Memories

## Location 10: Forest Path
**Visits:** 2 | **Episodes:** 1, 2

### Memories

**[DANGER] Deadly grue** *(Ep1, T100)*
Dark areas contain lethal grue. Never enter without light source or instant death.

---
"""

SAMPLE_MEMORIES_CORRUPTED = """# Location Memories

## Location 15: West of House
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Valid memory** *(Ep1, T10, +0)*
This memory is valid and should be parsed.

**MALFORMED Missing bracket** *(Ep1, T11, +0)*
This memory has malformed category.

## Location Invalid: Not a Number
**Visits:** 1 | **Episodes:** 1

### Memories

**[NOTE] Should be skipped** *(Ep1, T12, +0)*
This memory is in a location with invalid ID.

## Location 23: Living Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[SUCCESS] Valid after corruption** *(Ep1, T15, +0)*
This memory should still be parsed after corrupted sections.

---
"""

SAMPLE_MEMORIES_EMPTY_FILE = """# Location Memories

---
"""


class TestBaseSetup:
    """Common setup for SimpleMemoryManager tests."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        logger = Mock(spec=logging.Logger)
        return logger

    @pytest.fixture
    def game_config(self, tmp_path):
        """Create a test game configuration with temporary work directory."""
        return GameConfiguration(
            max_turns_per_episode=1000,
            turn_delay_seconds=0.0,
            game_file_path="test_game.z5",
            critic_rejection_threshold=0.5,
            episode_log_file="test_episode.log",
            json_log_file="test_episode.jsonl",
            state_export_file="test_state.json",
            zork_game_workdir=str(tmp_path),  # Use pytest temp directory
            client_base_url="http://localhost:1234",
            client_api_key="test_api_key",
            agent_model="test-agent-model",
            critic_model="test-critic-model",
            info_ext_model="test-extractor-model",
            analysis_model="test-analysis-model",
            memory_model="test-memory-model",
            condensation_model="test-condensation-model",
            knowledge_update_interval=100,
            objective_update_interval=20,
            enable_objective_refinement=True,
            objective_refinement_interval=200,
            max_objectives_before_forced_refinement=15,
            refined_objectives_target_count=10,
            max_context_tokens=100000,
            context_overflow_threshold=0.8,
            enable_state_export=True,
            s3_bucket="test-bucket",
            s3_key_prefix="test/",
            simple_memory_enabled=True,
            simple_memory_file="Memories.md",
            simple_memory_max_shown=10,
            # Sampling parameters
            agent_sampling={},
            critic_sampling={},
            extractor_sampling={},
            analysis_sampling={},
            memory_sampling={'temperature': 0.3, 'max_tokens': 1000},
            condensation_sampling={},
        )

    @pytest.fixture
    def game_state(self):
        """Create a test game state."""
        state = GameState()
        state.episode_id = "test_episode_001"
        state.turn_count = 10
        state.current_room_name_for_map = "Living Room"
        state.previous_zork_score = 50
        state.current_inventory = ["lamp", "sword"]
        return state

    @pytest.fixture
    def create_memories_file(self, game_config):
        """Helper fixture to create a Memories.md file with specified content."""
        def _create(content: str) -> Path:
            memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
            memories_path.write_text(content, encoding="utf-8")
            return memories_path
        return _create


class TestMemoriesFileParsingLocationHeaders(TestBaseSetup):
    """Test parsing of location headers from Memories.md format."""

    def test_parse_single_location_header(self, mock_logger, game_config, game_state, create_memories_file):
        """Test parsing a single location header: ## Location 15: West of House"""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        # Import after fixture setup (manager will be implemented)
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


class TestMemoriesFileParsingMemoryEntries(TestBaseSetup):
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


class TestMemoriesFileHandlingEdgeCases(TestBaseSetup):
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


class TestMemoryCacheStructure(TestBaseSetup):
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


class TestMemoryManagerIntegration(TestBaseSetup):
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


# ============================================================================
# Phase 1.2: File Writing Tests (TDD - Tests First)
# ============================================================================


class TestWriteNewMemoryToFile(TestBaseSetup):
    """Test writing first memory to empty or non-existent file."""

    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory for testing."""
        return Memory(
            category="SUCCESS",
            title="Open window",
            episode=1,
            turns="23",
            score_change=0,
            text="Window can be opened successfully."
        )

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

        # Should contain memory entry
        assert "**[SUCCESS] Open window**" in content
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

        # Verify content
        content = memories_path.read_text(encoding="utf-8")
        assert "## Location 15: West of House" in content
        assert "**[SUCCESS] Open window**" in content

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

        # Should match exact format: **[CATEGORY] Title** *(metadata)*\ntext
        assert "**[SUCCESS] Open window** *(Ep1, T23, +0)*" in content

        # Text should be on next line
        lines = content.split("\n")
        header_idx = next(i for i, line in enumerate(lines) if "**[SUCCESS] Open window**" in line)
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
            text="Window can be opened with effort."
        )

        manager.add_memory(15, "West of House", memory_range)

        # Test with positive score
        memory_score = Memory(
            category="SUCCESS",
            title="Acquire lamp",
            episode=2,
            turns="45",
            score_change=5,
            text="Lamp provides light."
        )

        manager.add_memory(23, "Living Room", memory_score)

        # Test without score
        memory_no_score = Memory(
            category="DANGER",
            title="Deadly grue",
            episode=1,
            turns="100",
            score_change=None,
            text="Grue will kill you."
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


class TestAppendToExistingLocationSection(TestBaseSetup):
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
            text="Window cannot be taken."
        )

        result = manager.add_memory(15, "West of House", new_memory)
        assert result is True

        # Cache should have 2 memories now
        assert len(manager.memory_cache[15]) == 2

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Both memories should be present
        assert "**[SUCCESS] Open window**" in content
        assert "**[FAILURE] Take window**" in content

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
            text="Window cannot be taken."
        )
        manager.add_memory(15, "West of House", memory2)

        # Add third memory
        memory3 = Memory(
            category="DISCOVERY",
            title="Mailbox location",
            episode=1,
            turns="20",
            score_change=0,
            text="Mailbox contains leaflet."
        )
        manager.add_memory(15, "West of House", memory3)

        # Read file
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Find positions of each memory
        success_pos = content.find("**[SUCCESS] Open window**")
        failure_pos = content.find("**[FAILURE] Take window**")
        discovery_pos = content.find("**[DISCOVERY] Mailbox location**")

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
            text="Window cannot be taken."
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
            text="Window cannot be taken."
        )
        manager.add_memory(15, "West of House", new_memory)

        # Read updated content
        updated_content = memories_path.read_text(encoding="utf-8")

        # Original memory should still be present
        assert "**[SUCCESS] Open window**" in updated_content
        assert "Window can be opened successfully." in updated_content


class TestCreateNewLocationSection(TestBaseSetup):
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
            text="Lamp provides light."
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
            text="Lamp provides light."
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
            text="Lamp provides light."
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
            text="Grue will kill you."
        )
        manager.add_memory(10, "Dark Room", new_memory)

        # Original locations should be unchanged
        assert len(manager.memory_cache[15]) == original_location_15_count
        assert len(manager.memory_cache[23]) == original_location_23_count

        # New location should exist
        assert 10 in manager.memory_cache
        assert len(manager.memory_cache[10]) == 1


class TestFileBackupMechanism(TestBaseSetup):
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
            text="Window cannot be taken."
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
            text="Window cannot be taken."
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
            text="Window can be opened."
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
            text="Window cannot be taken."
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
            text="Mailbox contains leaflet."
        )
        manager.add_memory(15, "West of House", memory2)

        # Read backup after second write
        backup_after_second = backup_path.read_text(encoding="utf-8")

        # Backups should be different
        assert backup_after_first != backup_after_second

        # Second backup should contain memory1 but not memory2
        assert "Take window" in backup_after_second
        assert "Mailbox" not in backup_after_second


class TestAtomicCacheUpdates(TestBaseSetup):
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
            text="Window cannot be taken."
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
            text="Window cannot be taken."
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
                text="Window cannot be taken."
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
            text="Window cannot be taken."
        )
        manager.add_memory(15, "West of House", new_memory)

        # Get memories from cache (Phase 2 will have a method for this)
        # For now, directly access cache
        memories = manager.memory_cache[15]

        # Should have 2 memories
        assert len(memories) == 2

        # New memory should be last
        assert memories[-1].title == "Take window"


class TestFileLocking(TestBaseSetup):
    """Test concurrent write safety with file locking."""

    def test_concurrent_writes_dont_corrupt_file(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that concurrent writes don't corrupt file using threading."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager
        import threading

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Create multiple memories to write concurrently
        memories_to_write = [
            Memory(
                category="FAILURE",
                title=f"Memory {i}",
                episode=1,
                turns=str(20 + i),
                score_change=None,
                text=f"Memory text {i}."
            )
            for i in range(10)
        ]

        # Write function for thread
        def write_memory(memory):
            manager.add_memory(15, "West of House", memory)

        # Create threads
        threads = [
            threading.Thread(target=write_memory, args=(mem,))
            for mem in memories_to_write
        ]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify file is not corrupted
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Should be valid markdown
        assert "## Location 15: West of House" in content

        # All memories should be present (cache should have original + 10)
        assert len(manager.memory_cache[15]) == 11

        # File should be parseable
        manager2 = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)
        assert 15 in manager2.memory_cache
        # Should have all memories (parsing validates structure)
        assert len(manager2.memory_cache[15]) >= 10

    def test_lock_released_after_write_completes(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that file lock is released after write completes."""
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
            text="Window cannot be taken."
        )
        manager.add_memory(15, "West of House", new_memory)

        # Should be able to write again immediately (lock released)
        memory2 = Memory(
            category="DISCOVERY",
            title="Mailbox",
            episode=1,
            turns="20",
            score_change=0,
            text="Mailbox contains leaflet."
        )
        result = manager.add_memory(15, "West of House", memory2)

        # Should succeed
        assert result is True

    def test_lock_released_even_if_write_fails(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that lock is released even if write operation fails."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Mock write to fail
        with patch("pathlib.Path.write_text", side_effect=IOError("Disk full")):
            new_memory = Memory(
                category="FAILURE",
                title="Take window",
                episode=1,
                turns="25",
                score_change=None,
                text="Window cannot be taken."
            )
            result = manager.add_memory(15, "West of House", new_memory)

            # Should fail
            assert result is False

        # Should be able to write successfully now (lock was released)
        memory2 = Memory(
            category="DISCOVERY",
            title="Mailbox",
            episode=1,
            turns="20",
            score_change=0,
            text="Mailbox contains leaflet."
        )
        result = manager.add_memory(15, "West of House", memory2)

        # Should succeed
        assert result is True

    def test_timeout_if_lock_held_too_long(self, mock_logger, game_config, game_state, create_memories_file):
        """Test that write times out if lock is held too long."""
        create_memories_file(SAMPLE_MEMORIES_SINGLE_LOCATION)

        from managers.simple_memory_manager import SimpleMemoryManager
        from filelock import FileLock
        import time
        import threading

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        lock_path = str(memories_path) + ".lock"

        # Acquire lock in separate thread and hold it
        lock = FileLock(lock_path, timeout=1)
        lock.acquire()

        try:
            # Try to write (should timeout)
            new_memory = Memory(
                category="FAILURE",
                title="Take window",
                episode=1,
                turns="25",
                score_change=None,
                text="Window cannot be taken."
            )

            # Should fail due to timeout
            result = manager.add_memory(15, "West of House", new_memory)
            assert result is False

            # Logger should have been called with timeout error
            # Check that error was logged
            error_calls = [call for call in mock_logger.error.call_args_list]
            assert len(error_calls) > 0

        finally:
            # Release lock
            lock.release()


# ============================================================================
# Phase 1.3: Trigger Detection Tests (TDD - Tests First)
# ============================================================================


class TestTriggerDetectionPositiveCases(TestBaseSetup):
    """Test cases where _should_synthesize_memory() should return True."""

    @pytest.fixture
    def base_context(self):
        """Create a base Z-machine context with no changes."""
        return {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp', 'sword'],
            'inventory_after': ['lamp', 'sword'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

    def test_trigger_on_positive_score_change(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when score increases."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Score increases by 5
        context = base_context.copy()
        context['score_after'] = 55
        context['score_delta'] = 5

        result = manager._should_synthesize_memory(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("score" in call.lower() for call in debug_calls)

    def test_trigger_on_negative_score_change(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when score decreases."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Score decreases by 10
        context = base_context.copy()
        context['score_after'] = 40
        context['score_delta'] = -10

        result = manager._should_synthesize_memory(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("score" in call.lower() for call in debug_calls)

    def test_trigger_on_location_change(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when location changes."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Location changes from 15 to 23
        context = base_context.copy()
        context['location_after'] = 23
        context['location_changed'] = True

        result = manager._should_synthesize_memory(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("location" in call.lower() for call in debug_calls)

    def test_trigger_on_inventory_item_added(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when inventory gains an item."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Inventory gains 'key'
        context = base_context.copy()
        context['inventory_after'] = ['lamp', 'sword', 'key']
        context['inventory_changed'] = True

        result = manager._should_synthesize_memory(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("inventory" in call.lower() for call in debug_calls)

    def test_trigger_on_inventory_item_removed(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when inventory loses an item."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Inventory loses 'sword'
        context = base_context.copy()
        context['inventory_after'] = ['lamp']
        context['inventory_changed'] = True

        result = manager._should_synthesize_memory(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("inventory" in call.lower() for call in debug_calls)

    def test_trigger_on_death(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when death occurs."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Death occurred
        context = base_context.copy()
        context['died'] = True

        result = manager._should_synthesize_memory(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("death" in call.lower() for call in debug_calls)

    def test_trigger_on_first_visit(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires on first visit to location."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # First visit to location
        context = base_context.copy()
        context['first_visit'] = True

        result = manager._should_synthesize_memory(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("first visit" in call.lower() for call in debug_calls)

    def test_trigger_on_substantial_response(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when response length exceeds 100 characters."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Substantial response (>100 chars)
        context = base_context.copy()
        context['response_length'] = 150

        result = manager._should_synthesize_memory(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("substantial" in call.lower() or "response" in call.lower() for call in debug_calls)


class TestTriggerDetectionNegativeCases(TestBaseSetup):
    """Test cases where _should_synthesize_memory() should return False."""

    @pytest.fixture
    def base_context(self):
        """Create a base Z-machine context with no changes."""
        return {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp', 'sword'],
            'inventory_after': ['lamp', 'sword'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

    def test_no_trigger_on_trivial_action(self, mock_logger, game_config, game_state, base_context):
        """Test no trigger when nothing significant happens."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # No changes at all
        context = base_context.copy()

        result = manager._should_synthesize_memory(context)

        # Should NOT trigger
        assert result is False

    def test_no_trigger_when_multiple_conditions_false(self, mock_logger, game_config, game_state):
        """Test no trigger when all conditions are false."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Explicitly false conditions
        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 20,
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should NOT trigger
        assert result is False

    def test_no_trigger_on_short_response_only(self, mock_logger, game_config, game_state, base_context):
        """Test no trigger when response is short and nothing else changes."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Short response with no other changes
        context = base_context.copy()
        context['response_length'] = 30

        result = manager._should_synthesize_memory(context)

        # Should NOT trigger
        assert result is False


class TestTriggerDetectionEdgeCases(TestBaseSetup):
    """Test edge cases for trigger detection."""

    def test_edge_case_score_change_of_zero(self, mock_logger, game_config, game_state):
        """Test that score delta of 0 does not trigger."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,  # Explicit zero delta
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should NOT trigger (no actual change)
        assert result is False

    def test_edge_case_location_change_to_same_location(self, mock_logger, game_config, game_state):
        """Test that location staying the same does not trigger."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,  # Same location
            'location_changed': False,  # Explicit false
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should NOT trigger (no actual change)
        assert result is False

    def test_edge_case_inventory_change_with_same_items(self, mock_logger, game_config, game_state):
        """Test that inventory with same items does not trigger."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp', 'sword'],
            'inventory_after': ['lamp', 'sword'],  # Same items
            'inventory_changed': False,  # Explicit false
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should NOT trigger (no actual change)
        assert result is False

    def test_edge_case_response_exactly_100_chars(self, mock_logger, game_config, game_state):
        """Test boundary condition: exactly 100 characters should NOT trigger."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 100,  # Exactly 100
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should NOT trigger (must be > 100, not >= 100)
        assert result is False

    def test_edge_case_response_exactly_101_chars(self, mock_logger, game_config, game_state):
        """Test boundary condition: 101 characters should trigger."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 101,  # Just over threshold
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should trigger (> 100)
        assert result is True


# ============================================================================
# Phase 1.4-6: LLM Synthesis Pipeline Tests (TDD - Tests First)
# ============================================================================


# Test fixtures for LLM synthesis
@pytest.fixture
def mock_llm_client_synthesis():
    """Mock LLM client that returns valid synthesis response."""
    from unittest.mock import Mock
    import json

    client = Mock()
    mock_response = Mock()
    mock_response.content = json.dumps({
        "should_remember": True,
        "category": "SUCCESS",
        "memory_title": "Acquired lamp",
        "memory_text": "Brass lantern provides light for dark areas.",
        "reasoning": "Significant item acquisition"
    })
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def sample_z_machine_context():
    """Sample Z-machine context for testing."""
    return {
        'score_before': 0,
        'score_after': 5,
        'score_delta': 5,
        'location_before': 23,
        'location_after': 23,
        'location_changed': False,
        'inventory_before': [],
        'inventory_after': ['lamp'],
        'inventory_changed': True,
        'died': False,
        'response_length': 150,
        'first_visit': False
    }


# ============================================================================
# Part A: Pydantic Schema Tests (5 tests)
# ============================================================================


class TestMemorySynthesisResponseSchema(TestBaseSetup):
    """Test the MemorySynthesisResponse Pydantic model."""

    def test_valid_response_parsing(self, mock_logger, game_config, game_state):
        """Test valid response parsing with all fields present."""
        from managers.simple_memory_manager import MemorySynthesisResponse
        import json

        # Valid JSON with all fields
        valid_json = {
            "should_remember": True,
            "category": "SUCCESS",
            "memory_title": "Test Title",
            "memory_text": "Test memory text.",
            "reasoning": "Test reasoning"
        }

        # Parse using Pydantic
        response = MemorySynthesisResponse.model_validate(valid_json)

        # Verify all fields
        assert response.should_remember is True
        assert response.category == "SUCCESS"
        assert response.memory_title == "Test Title"
        assert response.memory_text == "Test memory text."
        assert response.reasoning == "Test reasoning"

    def test_missing_optional_field_reasoning(self, mock_logger, game_config, game_state):
        """Test missing optional field (reasoning) defaults to empty string."""
        from managers.simple_memory_manager import MemorySynthesisResponse

        # JSON without reasoning field
        json_data = {
            "should_remember": True,
            "category": "FAILURE",
            "memory_title": "Test",
            "memory_text": "Test text."
        }

        # Parse
        response = MemorySynthesisResponse.model_validate(json_data)

        # Reasoning should default to empty string
        assert response.reasoning == ""

    def test_invalid_category_still_parses(self, mock_logger, game_config, game_state):
        """Test invalid category still parses (validation happens later)."""
        from managers.simple_memory_manager import MemorySynthesisResponse

        # JSON with invalid category
        json_data = {
            "should_remember": True,
            "category": "INVALID_CATEGORY",  # Not in [SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE]
            "memory_title": "Test",
            "memory_text": "Test text.",
            "reasoning": "Test"
        }

        # Should still parse (we don't enforce enum validation at Pydantic level)
        response = MemorySynthesisResponse.model_validate(json_data)
        assert response.category == "INVALID_CATEGORY"

    def test_field_type_validation(self, mock_logger, game_config, game_state):
        """Test field types are validated correctly."""
        from managers.simple_memory_manager import MemorySynthesisResponse
        from pydantic import ValidationError

        # should_remember must be bool
        with pytest.raises(ValidationError):
            MemorySynthesisResponse.model_validate({
                "should_remember": "yes",  # Wrong type
                "category": "SUCCESS",
                "memory_title": "Test",
                "memory_text": "Test."
            })

        # Other fields must be strings
        with pytest.raises(ValidationError):
            MemorySynthesisResponse.model_validate({
                "should_remember": True,
                "category": 123,  # Wrong type
                "memory_title": "Test",
                "memory_text": "Test."
            })

    def test_model_validate_json_from_string(self, mock_logger, game_config, game_state):
        """Test parsing from JSON string directly."""
        from managers.simple_memory_manager import MemorySynthesisResponse
        import json

        # JSON string
        json_string = json.dumps({
            "should_remember": True,
            "category": "DISCOVERY",
            "memory_title": "Found key",
            "memory_text": "Key unlocks door.",
            "reasoning": "Important discovery"
        })

        # Parse from string
        response = MemorySynthesisResponse.model_validate_json(json_string)

        # Verify
        assert response.should_remember is True
        assert response.category == "DISCOVERY"
        assert response.memory_title == "Found key"


# ============================================================================
# Part B: LLM Synthesis Method Tests (10 tests)
# ============================================================================


class TestSynthesizeMemoryMethod(TestBaseSetup):
    """Test _synthesize_memory() LLM synthesis method."""

    def test_successful_synthesis_should_remember(self, mock_logger, game_config, game_state,
                                                   mock_llm_client_synthesis, sample_z_machine_context):
        """Test successful synthesis when LLM returns should_remember=True."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Patch LLMClientWrapper to return our mock
        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Call synthesis
            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the brass lantern.",
                z_machine_context=sample_z_machine_context
            )

            # Should return MemorySynthesisResponse object
            assert result is not None
            assert result.should_remember is True
            assert result.category == "SUCCESS"
            assert result.memory_title == "Acquired lamp"

            # Verify LLM was called
            assert mock_llm_client_synthesis.chat.completions.create.called

    def test_synthesis_decides_not_to_remember(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test synthesis returns None when LLM says should_remember=False."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch
        import json

        # Mock LLM client that returns should_remember=False
        client = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "should_remember": False,
            "category": "NOTE",
            "memory_title": "Trivial action",
            "memory_text": "Not worth remembering.",
            "reasoning": "Duplicate of existing memory"
        })
        client.chat.completions.create.return_value = mock_response

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="look",
                response="You see nothing special.",
                z_machine_context=sample_z_machine_context
            )

            # Should return None (don't store)
            assert result is None

    def test_existing_memories_passed_to_llm(self, mock_logger, game_config, game_state,
                                            mock_llm_client_synthesis, sample_z_machine_context, create_memories_file):
        """Test existing memories are passed to LLM for deduplication."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Create file with existing memories
        create_memories_file(SAMPLE_MEMORIES_FULL)

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Call synthesis for location that has existing memories
            result = manager._synthesize_memory(
                location_id=23,  # Has existing memories
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Verify LLM was called
            assert mock_llm_client_synthesis.chat.completions.create.called

            # Get the call arguments
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args

            # Verify messages passed to LLM
            messages = call_args[1]['messages']
            prompt = messages[0]['content']

            # Should include existing memories
            assert "Existing Memories" in prompt or "existing memories" in prompt.lower()

    def test_z_machine_context_in_prompt(self, mock_logger, game_config, game_state,
                                        mock_llm_client_synthesis, sample_z_machine_context):
        """Test Z-machine context is included in LLM prompt."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Get call arguments
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            messages = call_args[1]['messages']
            prompt = messages[0]['content']

            # Should include Z-machine context data
            assert "score" in prompt.lower() or "Score" in prompt
            assert "inventory" in prompt.lower() or "Inventory" in prompt

    def test_action_and_response_in_prompt(self, mock_logger, game_config, game_state,
                                          mock_llm_client_synthesis, sample_z_machine_context):
        """Test action text and game response are included in prompt."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            test_action = "take brass lantern"
            test_response = "You pick up the heavy brass lantern."

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action=test_action,
                response=test_response,
                z_machine_context=sample_z_machine_context
            )

            # Get call arguments
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            messages = call_args[1]['messages']
            prompt = messages[0]['content']

            # Should include action and response
            assert test_action in prompt
            assert test_response in prompt

    def test_uses_info_ext_model(self, mock_logger, game_config, game_state,
                                 mock_llm_client_synthesis, sample_z_machine_context):
        """Test synthesis uses config.memory_model."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Set a specific model in config
        game_config.memory_model = "test-memory-model-v2"

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Verify correct model was used
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            assert call_args[1]['model'] == "test-memory-model-v2"

    def test_structured_output_with_json_schema(self, mock_logger, game_config, game_state,
                                               mock_llm_client_synthesis, sample_z_machine_context):
        """Test uses response_format with Pydantic schema for structured output."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Verify response_format was used
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            assert 'response_format' in call_args[1]

            # Should be JSON schema format
            response_format = call_args[1]['response_format']
            assert 'type' in response_format
            assert response_format['type'] == 'json_schema'

    def test_handles_llm_error_gracefully(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test handles LLM error gracefully without crashing."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch

        # Mock LLM client that raises exception
        client = Mock()
        client.chat.completions.create.side_effect = Exception("LLM API error")

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should return None (error handled)
            assert result is None

            # Should log error
            assert mock_logger.error.called

    def test_handles_invalid_json_gracefully(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test handles invalid JSON response gracefully."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch

        # Mock LLM client that returns malformed JSON
        client = Mock()
        mock_response = Mock()
        mock_response.content = "{ invalid json here"  # Malformed
        client.chat.completions.create.return_value = mock_response

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should return None (error handled)
            assert result is None

            # Should log error
            assert mock_logger.error.called

    def test_reasoning_field_captured(self, mock_logger, game_config, game_state,
                                     mock_llm_client_synthesis, sample_z_machine_context):
        """Test reasoning field is captured and logged."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should have reasoning field
            assert result.reasoning == "Significant item acquisition"

            # Should be logged for debugging
            assert mock_logger.debug.called


# ============================================================================
# Part C: record_action_outcome Method Tests (12 tests)
# ============================================================================


class TestRecordActionOutcomeMethod(TestBaseSetup):
    """Test record_action_outcome() complete flow."""

    def test_complete_flow_trigger_synthesize_write_cache(self, mock_logger, game_config, game_state,
                                                          mock_llm_client_synthesis, sample_z_machine_context):
        """Test complete flow: trigger → synthesize → write → cache."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch
        from pathlib import Path

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Record action outcome (should trigger synthesis)
            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the brass lantern.",
                z_machine_context=sample_z_machine_context
            )

            # Should have called LLM
            assert mock_llm_client_synthesis.chat.completions.create.called

            # Should have written to file
            memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
            assert memories_path.exists()
            content = memories_path.read_text()
            assert "Location 23" in content
            assert "Acquired lamp" in content

            # Should have updated cache
            assert 23 in manager.memory_cache
            assert len(manager.memory_cache[23]) == 1
            assert manager.memory_cache[23][0].title == "Acquired lamp"

    def test_skip_when_no_trigger(self, mock_logger, game_config, game_state, mock_llm_client_synthesis):
        """Test skips synthesis when no trigger fires."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Context with no triggers
            no_trigger_context = {
                'score_before': 50,
                'score_after': 50,
                'score_delta': 0,
                'location_before': 15,
                'location_after': 15,
                'location_changed': False,
                'inventory_before': ['lamp'],
                'inventory_after': ['lamp'],
                'inventory_changed': False,
                'died': False,
                'response_length': 30,
                'first_visit': False
            }

            manager.record_action_outcome(
                location_id=15,
                location_name="West of House",
                action="look",
                response="Nothing special.",
                z_machine_context=no_trigger_context
            )

            # Should NOT have called LLM
            assert not mock_llm_client_synthesis.chat.completions.create.called

    def test_skip_when_llm_says_dont_remember(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test skips write when LLM says should_remember=False."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch
        from pathlib import Path
        import json

        # Mock LLM that says don't remember
        client = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "should_remember": False,
            "category": "NOTE",
            "memory_title": "Trivial",
            "memory_text": "Not worth it.",
            "reasoning": "Duplicate"
        })
        client.chat.completions.create.return_value = mock_response

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="look",
                response="Nothing new.",
                z_machine_context=sample_z_machine_context
            )

            # Should have called LLM
            assert client.chat.completions.create.called

            # Should NOT have written to file
            memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
            assert not memories_path.exists()

            # Cache should be empty
            assert 23 not in manager.memory_cache

    def test_memory_formatted_correctly_before_write(self, mock_logger, game_config, game_state,
                                                    mock_llm_client_synthesis, sample_z_machine_context):
        """Test MemorySynthesisResponse converted to Memory dataclass correctly."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Set game state for metadata
        game_state.episode_id = "ep_001"
        game_state.turn_count = 45

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Check cache has correct Memory format
            memory = manager.memory_cache[23][0]
            assert memory.category == "SUCCESS"
            assert memory.title == "Acquired lamp"
            assert memory.episode == 1  # Extracted from episode_id
            assert memory.turns == "45"  # From turn_count
            assert memory.score_change == 5  # From z_machine_context
            assert memory.text == "Brass lantern provides light for dark areas."

    def test_add_memory_called_with_correct_args(self, mock_logger, game_config, game_state,
                                                 mock_llm_client_synthesis, sample_z_machine_context):
        """Test add_memory is called with correct arguments."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch, Mock

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Mock add_memory method
            manager.add_memory = Mock(return_value=True)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Verify add_memory was called
            assert manager.add_memory.called

            # Check arguments
            call_args = manager.add_memory.call_args
            assert call_args[0][0] == 23  # location_id
            assert call_args[0][1] == "Living Room"  # location_name
            assert hasattr(call_args[0][2], 'category')  # Memory object

    def test_cache_updated_immediately(self, mock_logger, game_config, game_state,
                                      mock_llm_client_synthesis, sample_z_machine_context):
        """Test cache is updated immediately after write."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Cache should be empty initially
            assert 23 not in manager.memory_cache

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Cache should be updated immediately
            assert 23 in manager.memory_cache
            assert len(manager.memory_cache[23]) == 1

            # Should be able to retrieve immediately
            memory_text = manager.get_location_memory(23)
            assert "Acquired lamp" in memory_text

    def test_handles_synthesis_failure_gracefully(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test handles synthesis failure gracefully (returns None)."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch

        # Mock LLM that fails
        client = Mock()
        client.chat.completions.create.side_effect = Exception("LLM error")

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Should not crash
            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should log error
            assert mock_logger.error.called

            # Cache should be empty (no write)
            assert 23 not in manager.memory_cache

    def test_handles_write_failure_gracefully(self, mock_logger, game_config, game_state,
                                              mock_llm_client_synthesis, sample_z_machine_context):
        """Test handles write failure gracefully (add_memory returns False)."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch, Mock

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Mock add_memory to fail
            manager.add_memory = Mock(return_value=False)

            # Should not crash
            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should log warning/error
            assert mock_logger.warning.called or mock_logger.error.called

    def test_existing_memories_retrieved_for_deduplication(self, mock_logger, game_config, game_state,
                                                          mock_llm_client_synthesis, sample_z_machine_context,
                                                          create_memories_file):
        """Test existing memories are retrieved from cache for deduplication."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Create file with existing memories
        create_memories_file(SAMPLE_MEMORIES_FULL)

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Verify cache has existing memories
            assert 23 in manager.memory_cache
            initial_count = len(manager.memory_cache[23])

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # LLM should have received existing memories in prompt
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            messages = call_args[1]['messages']
            prompt = messages[0]['content']

            # Should mention existing memories
            assert "existing" in prompt.lower() or "Existing" in prompt

    def test_logging_at_each_step(self, mock_logger, game_config, game_state,
                                  mock_llm_client_synthesis, sample_z_machine_context):
        """Test logging happens at each step of the pipeline."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should have debug logs for trigger
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("trigger" in call.lower() or "synthesis" in call.lower() for call in debug_calls)

            # Should have info log for storage
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("memory" in call.lower() or "stored" in call.lower() for call in info_calls)

    def test_metadata_extracted_from_z_machine_context(self, mock_logger, game_config, game_state,
                                                       mock_llm_client_synthesis, sample_z_machine_context):
        """Test metadata is correctly extracted from z_machine_context."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Set game state
        game_state.episode_id = "ep_003"
        game_state.turn_count = 127

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Check memory has correct metadata
            memory = manager.memory_cache[23][0]
            assert memory.episode == 3  # From episode_id
            assert memory.turns == "127"  # From turn_count
            assert memory.score_change == 5  # From z_machine_context

    def test_end_to_end_with_realistic_context(self, mock_logger, game_config, game_state,
                                               mock_llm_client_synthesis):
        """Test end-to-end with realistic Z-machine context."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch
        from pathlib import Path

        # Realistic context for acquiring an item
        realistic_context = {
            'score_before': 0,
            'score_after': 5,
            'score_delta': 5,
            'location_before': 23,
            'location_after': 23,
            'location_changed': False,
            'inventory_before': [],
            'inventory_after': ['brass lantern'],
            'inventory_changed': True,
            'died': False,
            'response_length': 87,
            'first_visit': False
        }

        game_state.episode_id = "ep_001"
        game_state.turn_count = 45

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="Taken. The brass lantern is now in your possession and could prove useful for lighting your way.",
                z_machine_context=realistic_context
            )

            # Should complete successfully
            assert 23 in manager.memory_cache
            assert len(manager.memory_cache[23]) == 1

            # File should exist
            memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
            assert memories_path.exists()

            # Content should be correct
            content = memories_path.read_text()
            assert "Location 23: Living Room" in content
            assert "[SUCCESS]" in content
            assert "Acquired lamp" in content
            assert "*(Ep1, T45, +5)*" in content


class TestTriggerDetectionMultipleTriggers(TestBaseSetup):
    """Test behavior when multiple triggers fire simultaneously."""

    def test_multiple_triggers_score_and_location(self, mock_logger, game_config, game_state):
        """Test multiple triggers fire together (score + location)."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 55,  # Score changed
            'score_delta': 5,
            'location_before': 15,
            'location_after': 23,  # Location changed
            'location_changed': True,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should trigger (multiple conditions met)
        assert result is True

        # Logger should log at least one trigger reason
        mock_logger.debug.assert_called()

    def test_multiple_triggers_inventory_and_death(self, mock_logger, game_config, game_state):
        """Test multiple triggers fire together (inventory + death)."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp', 'sword'],
            'inventory_after': ['lamp'],  # Lost sword
            'inventory_changed': True,
            'died': True,  # Also died
            'response_length': 50,
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should trigger (multiple conditions met)
        assert result is True

        # Logger should log at least one trigger reason
        mock_logger.debug.assert_called()

    def test_multiple_triggers_all_conditions(self, mock_logger, game_config, game_state):
        """Test when ALL trigger conditions are met."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 45,  # Score changed (negative)
            'score_delta': -5,
            'location_before': 15,
            'location_after': 23,  # Location changed
            'location_changed': True,
            'inventory_before': ['lamp'],
            'inventory_after': [],  # Lost item
            'inventory_changed': True,
            'died': True,  # Died
            'response_length': 200,  # Substantial response
            'first_visit': True  # First visit
        }

        result = manager._should_synthesize_memory(context)

        # Should trigger (all conditions met)
        assert result is True

        # Logger should log trigger reason
        mock_logger.debug.assert_called()


class TestTriggerDetectionPerformance(TestBaseSetup):
    """Test that trigger detection is fast (no LLM calls)."""

    def test_trigger_detection_is_fast(self, mock_logger, game_config, game_state):
        """Test that trigger detection completes quickly without LLM calls."""
        from managers.simple_memory_manager import SimpleMemoryManager
        import time

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 55,
            'score_delta': 5,
            'location_before': 15,
            'location_after': 23,
            'location_changed': True,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp', 'key'],
            'inventory_changed': True,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        # Measure execution time
        start_time = time.time()
        result = manager._should_synthesize_memory(context)
        elapsed_time = time.time() - start_time

        # Should complete in under 1ms (boolean logic only)
        assert elapsed_time < 0.001, f"Trigger detection took {elapsed_time*1000:.2f}ms (should be <1ms)"

        # Should still return correct result
        assert result is True

    def test_no_llm_client_called_during_trigger_detection(self, mock_logger, game_config, game_state):
        """Test that LLM client is not called during trigger detection."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Verify manager doesn't have llm_client attribute (not needed for triggers)
        # Trigger detection should be pure logic - no external calls

        context = {
            'score_before': 50,
            'score_after': 55,
            'score_delta': 5,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        # Should work without any external dependencies
        result = manager._should_synthesize_memory(context)

        # Should trigger on score change
        assert result is True


class TestMemoryStatusFiltering(TestBaseSetup):
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


class TestMemoryStatusParsing(TestBaseSetup):
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


class TestMemoryStatusUpdates(TestBaseSetup):
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
