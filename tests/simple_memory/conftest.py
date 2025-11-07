"""
ABOUTME: Shared test fixtures and sample data for SimpleMemoryManager tests.
ABOUTME: Includes Memory/MemoryStatus classes, sample file content, and pytest fixtures.
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import json

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
    persistence: str  # "core" | "permanent" | "ephemeral" - REQUIRED
    status: MemoryStatusType = MemoryStatus.ACTIVE  # Memory status
    superseded_by: Optional[str] = None  # Title of memory that superseded this
    superseded_at_turn: Optional[int] = None  # Turn when superseded
    invalidation_reason: Optional[str] = None  # Reason for standalone invalidation


# ============================================================================
# Sample Memories.md content for testing (as module-level constants)
# ============================================================================

# These are module-level constants that can be imported by test files
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


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    return logger


@pytest.fixture
def game_config(tmp_path):
    """Create a test game configuration with temporary work directory."""
    return GameConfiguration(
        max_turns_per_episode=1000,
        turn_delay_seconds=0.0,
        game_file_path="test_game.z5",
        critic_rejection_threshold=0.5,
        episode_log_file="test_episode.log",
        json_log_file="test_episode.jsonl",
        state_export_file="test_state.json",
        map_state_file="test_map_state.json",
        zork_game_workdir=str(tmp_path),  # Use pytest temp directory
        client_base_url="http://localhost:1234",
        client_api_key="test_api_key",
        agent_model="test-agent-model",
        critic_model="test-critic-model",
        info_ext_model="test-extractor-model",
        analysis_model="test-analysis-model",
        memory_model="test-memory-model",
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
        simple_memory_file="Memories.md",
        simple_memory_max_shown=10,
        knowledge_file="test_knowledgebase.md",
        # Sampling parameters
        agent_sampling={},
        critic_sampling={},
        extractor_sampling={},
        analysis_sampling={},
        memory_sampling={'temperature': 0.3, 'max_tokens': 1000},
    )


@pytest.fixture
def game_state():
    """Create a test game state."""
    state = GameState()
    state.episode_id = "test_episode_001"
    state.turn_count = 10
    state.current_room_name_for_map = "Living Room"
    state.previous_zork_score = 50
    state.current_inventory = ["lamp", "sword"]
    return state


@pytest.fixture
def create_memories_file(game_config):
    """Helper fixture to create a Memories.md file with specified content."""
    def _create(content: str) -> Path:
        memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")
        return memories_path
    return _create


@pytest.fixture
def mock_llm_client_synthesis():
    """Mock LLM client that returns valid synthesis response."""
    client = Mock()
    mock_response = Mock()
    mock_response.content = json.dumps({
        "should_remember": True,
        "category": "SUCCESS",
        "memory_title": "Acquired lamp",
        "memory_text": "Brass lantern provides light for dark areas.",
        "persistence": "permanent",
        "status": "ACTIVE",
        "supersedes_memory_titles": [],
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


@pytest.fixture
def base_context():
    """Create a base Z-machine context with no changes (for trigger tests)."""
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


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        category="SUCCESS",
        title="Open window",
        episode=1,
        turns="23",
        score_change=0,
        text="Window can be opened successfully.",
        persistence="permanent"
    )
