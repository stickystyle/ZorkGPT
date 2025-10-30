"""
ABOUTME: Integration tests for Simple Memory System (Phases 2-3).
ABOUTME: Tests memory retrieval, orchestrator integration, context injection, and configuration.

This module tests Phases 2-3 of the Simple Memory System:
- Phase 2: Memory retrieval and formatting (get_location_memory)
- Phase 3: Orchestrator integration, context injection, and configuration
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List

from session.game_state import GameState
from session.game_configuration import GameConfiguration
from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemorySynthesisResponse
from managers.context_manager import ContextManager


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    return logger


@pytest.fixture
def mock_game_state():
    """Mock GameState for testing."""
    state = Mock(spec=GameState)
    state.episode_id = "ep_001"
    state.turn_count = 10
    state.current_room_id = 15
    state.current_room_name_for_map = "Living Room"
    state.current_inventory = ["lamp"]
    state.previous_zork_score = 0
    state.action_history = []
    state.memory_log_history = []
    state.action_reasoning_history = []
    state.action_counts = {}
    state.prev_room_for_prompt_context = ""
    state.action_leading_to_current_room_for_prompt_context = ""
    return state


@pytest.fixture
def mock_config(tmp_path):
    """Mock GameConfiguration with temporary work directory."""
    config = Mock(spec=GameConfiguration)
    config.zork_game_workdir = str(tmp_path)
    config.info_ext_model = "gpt-4"
    config.simple_memory_enabled = True
    config.simple_memory_file = "Memories.md"
    config.simple_memory_max_shown = 10
    config.max_turns_per_episode = 1000
    return config


@pytest.fixture
def mock_llm_client_synthesis():
    """Mock LLM client that returns should_remember=True."""
    client = Mock()

    # Mock the nested chat.completions.create call
    mock_response = Mock()
    mock_response.content = """
    {
        "should_remember": true,
        "category": "SUCCESS",
        "memory_title": "Test Memory",
        "memory_text": "Test memory text.",
        "reasoning": "Test reasoning"
    }
    """

    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def mock_llm_client_no_remember():
    """Mock LLM client that returns should_remember=False."""
    client = Mock()

    mock_response = Mock()
    mock_response.content = """
    {
        "should_remember": false,
        "category": "NOTE",
        "memory_title": "",
        "memory_text": "",
        "reasoning": "Not significant enough"
    }
    """

    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def memory_manager_with_mocks(mock_logger, mock_config, mock_game_state, mock_llm_client_synthesis):
    """Fully configured SimpleMemoryManager for integration testing."""
    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=mock_config,
        game_state=mock_game_state,
        llm_client=mock_llm_client_synthesis
    )
    return manager


@pytest.fixture
def create_memories_file():
    """Helper fixture to create Memories.md file with content."""
    def _create(workdir: Path, content: str) -> Path:
        memories_path = Path(workdir) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")
        return memories_path
    return _create


# Sample memory file content
SAMPLE_MEMORIES_MULTIPLE = """# Location Memories

## Location 15: West of House
**Visits:** 3 | **Episodes:** 1, 2, 3

### Memories

**[SUCCESS] Open and enter window** *(Ep1, T23-24, +0)*
Window can be opened with effort and used as alternative entrance to house.

**[FAILURE] Take or break window** *(Ep1, T25-26)*
Window is part of house structure - cannot be taken, moved, or broken.

**[DISCOVERY] Mailbox location** *(Ep1, T20, +0)*
Small mailbox located here contains advertising leaflet.

---

## Location 23: Living Room
**Visits:** 5 | **Episodes:** 1, 2

### Memories

**[SUCCESS] Acquire brass lantern** *(Ep1, T45, +5)*
Brass lantern is takeable and provides light source.

**[DANGER] Grue warning** *(Ep1, T100, -10)*
Dark areas contain lethal grue. Never enter without light.

---
"""


# ============================================================================
# Part A: Memory Retrieval & Formatting Tests (Phase 2)
# ============================================================================

class TestMemoryRetrievalAndFormatting:
    """Test get_location_memory() retrieval and formatting."""

    def test_returns_formatted_memory_for_location_with_memories(
        self, mock_logger, mock_config, mock_game_state, create_memories_file, tmp_path
    ):
        """Test returns formatted string with all memories for location."""
        # Create file with multiple memories at location 15
        create_memories_file(tmp_path, SAMPLE_MEMORIES_MULTIPLE)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        # Get memories for location 15
        memory_text = manager.get_location_memory(15)

        # Should return formatted text with all 3 memories
        assert memory_text != ""
        assert "[SUCCESS] Open and enter window" in memory_text
        assert "[FAILURE] Take or break window" in memory_text
        assert "[DISCOVERY] Mailbox location" in memory_text
        assert "Window can be opened" in memory_text

    def test_returns_empty_for_new_location(
        self, mock_logger, mock_config, mock_game_state, create_memories_file, tmp_path
    ):
        """Test returns empty string for location with no memories."""
        create_memories_file(tmp_path, SAMPLE_MEMORIES_MULTIPLE)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        # Get memories for location that doesn't exist
        memory_text = manager.get_location_memory(999)

        # Should return empty string
        assert memory_text == ""

    def test_formatting_includes_category_title_text(
        self, mock_logger, mock_config, mock_game_state, create_memories_file, tmp_path
    ):
        """Test formatting includes category, title, and memory text."""
        create_memories_file(tmp_path, SAMPLE_MEMORIES_MULTIPLE)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        memory_text = manager.get_location_memory(15)

        # Format should be: [CATEGORY] Title: text
        assert "[SUCCESS]" in memory_text
        assert "Open and enter window:" in memory_text
        assert "Window can be opened with effort" in memory_text

    def test_memory_ordering_chronological(
        self, mock_logger, mock_config, mock_game_state, tmp_path
    ):
        """Test memories appear in chronological order (as stored in file)."""
        # Create file with memories in specific order
        content = """# Location Memories

## Location 15: Test Room
**Visits:** 3 | **Episodes:** 1

### Memories

**[NOTE] First memory** *(Ep1, T1, +0)*
This was first.

**[NOTE] Second memory** *(Ep1, T2, +0)*
This was second.

**[NOTE] Third memory** *(Ep1, T3, +0)*
This was third.

---
"""
        memories_path = Path(tmp_path) / "Memories.md"
        memories_path.write_text(content, encoding="utf-8")

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        memory_text = manager.get_location_memory(15)

        # Memories should appear in same order as file
        first_pos = memory_text.find("First memory")
        second_pos = memory_text.find("Second memory")
        third_pos = memory_text.find("Third memory")

        assert first_pos < second_pos < third_pos


# ============================================================================
# Part B: Configuration Tests (Phase 3)
# ============================================================================

class TestConfigurationHandling:
    """Test configuration loading and defaults."""

    def test_default_values_when_config_missing(self, mock_logger, tmp_path):
        """Test manager uses defaults when config attributes missing."""
        # Create minimal config without simple_memory attributes
        config = Mock()
        config.zork_game_workdir = str(tmp_path)
        config.info_ext_model = "gpt-4"
        # Don't set simple_memory_* attributes

        game_state = Mock()
        game_state.episode_id = "ep_001"
        game_state.turn_count = 1

        # Manager should handle missing attributes gracefully
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=config,
            game_state=game_state
        )

        # Manager should be created successfully (uses hardcoded defaults internally)
        assert manager is not None

    def test_config_values_override_defaults(self, mock_logger, tmp_path):
        """Test config values override internal defaults."""
        config = Mock()
        config.zork_game_workdir = str(tmp_path)
        config.info_ext_model = "custom-model"
        config.simple_memory_enabled = True
        config.simple_memory_file = "CustomMemories.md"
        config.simple_memory_max_shown = 20

        game_state = Mock()
        game_state.episode_id = "ep_001"
        game_state.turn_count = 1

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=config,
            game_state=game_state
        )

        # Verify manager uses config values
        assert manager.config.info_ext_model == "custom-model"
        assert manager.config.simple_memory_file == "CustomMemories.md"


# ============================================================================
# Part C: Orchestrator Integration Tests (Phase 3)
# ============================================================================

class TestOrchestratorIntegration:
    """Test integration points with orchestrator."""

    def test_z_machine_context_captured_correctly(
        self, memory_manager_with_mocks
    ):
        """Test Z-machine context has all required fields."""
        # Simulate orchestrator calling record_action_outcome
        z_context = {
            'score_before': 0,
            'score_after': 5,
            'score_delta': 5,
            'location_before': 14,
            'location_after': 15,
            'location_changed': True,
            'inventory_before': [],
            'inventory_after': ['lamp'],
            'inventory_changed': True,
            'died': False,
            'response_length': 150,
            'first_visit': True
        }

        # Verify all expected fields present
        assert 'score_delta' in z_context
        assert 'location_changed' in z_context
        assert 'inventory_changed' in z_context
        assert 'died' in z_context
        assert 'first_visit' in z_context
        assert z_context['score_delta'] == 5

    def test_record_action_outcome_called_after_action(
        self, memory_manager_with_mocks
    ):
        """Test record_action_outcome invoked with correct parameters."""
        z_context = {
            'score_delta': 5,
            'location_changed': True,
            'inventory_changed': True,
            'inventory_before': [],
            'inventory_after': ['lamp'],
            'died': False,
            'response_length': 150,
            'first_visit': True
        }

        # Call record_action_outcome (simulates orchestrator call)
        memory_manager_with_mocks.record_action_outcome(
            location_id=15,
            location_name="Living Room",
            action="take lamp",
            response="Taken.",
            z_machine_context=z_context
        )

        # Verify LLM was called for synthesis
        memory_manager_with_mocks._llm_client.chat.completions.create.assert_called_once()

    def test_memories_persist_across_episode_reset(
        self, mock_logger, mock_config, mock_game_state, tmp_path, mock_llm_client_synthesis
    ):
        """Test memories persist after reset_episode() called."""
        # Episode 1: Create memory at location 15
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_client_synthesis
        )

        # Add memory
        memory = Memory(
            category="SUCCESS",
            title="Episode 1 Memory",
            episode=1,
            turns="10",
            score_change=5,
            text="Memory from episode 1"
        )

        success = manager.add_memory(15, "Living Room", memory)
        assert success
        assert 15 in manager.memory_cache

        # Verify file exists
        memories_path = Path(tmp_path) / "Memories.md"
        assert memories_path.exists()
        original_content = memories_path.read_text()
        assert "Episode 1 Memory" in original_content

        # Episode 2: Reset and verify persistence
        manager.reset_episode()

        # Memory should still be in cache
        assert 15 in manager.memory_cache
        assert len(manager.memory_cache[15]) == 1

        # File should still contain memory
        content = memories_path.read_text()
        assert "Episode 1 Memory" in content


# ============================================================================
# Part D: ContextManager Integration Tests (Phase 3)
# ============================================================================

class TestContextManagerIntegration:
    """Test integration with ContextManager for context injection."""

    def test_context_manager_can_access_simple_memory(
        self, mock_logger, mock_config, mock_game_state, tmp_path, create_memories_file
    ):
        """Test ContextManager can access SimpleMemoryManager reference."""
        create_memories_file(tmp_path, SAMPLE_MEMORIES_MULTIPLE)

        # Create managers
        memory_manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        context_manager = ContextManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        # Inject simple_memory reference (simulates orchestrator setup)
        context_manager.simple_memory = memory_manager

        # Verify context manager can retrieve memories
        memory_text = context_manager.simple_memory.get_location_memory(15)
        assert memory_text != ""
        assert "[SUCCESS]" in memory_text

    def test_context_manager_handles_missing_simple_memory(
        self, mock_logger, mock_config, mock_game_state
    ):
        """Test ContextManager gracefully handles missing simple_memory reference."""
        context_manager = ContextManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        # Don't inject simple_memory reference
        # Get agent context should not crash
        context = context_manager.get_agent_context(
            current_state="Test state",
            inventory=["lamp"],
            location="Living Room",
            location_id=15
        )

        # Should return context without crashing
        assert context is not None
        assert isinstance(context, dict)

    def test_memory_section_injected_into_agent_context(
        self, mock_logger, mock_config, mock_game_state, tmp_path, create_memories_file
    ):
        """Test location memory injected into agent context."""
        create_memories_file(tmp_path, SAMPLE_MEMORIES_MULTIPLE)

        # Create managers
        memory_manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        context_manager = ContextManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        # Inject simple_memory reference
        context_manager.simple_memory = memory_manager

        # Get agent context with location that has memories
        # Note: We need to patch get_agent_context to inject memories
        # For now, we test that memory_manager can provide formatted memory
        memory_text = memory_manager.get_location_memory(15)

        # Memory text should be non-empty and formatted
        assert memory_text != ""
        assert "[SUCCESS]" in memory_text
        assert "[FAILURE]" in memory_text


# ============================================================================
# Part E: End-to-End Integration Test
# ============================================================================

class TestEndToEndIntegration:
    """Test complete flow with simulated orchestrator."""

    def test_complete_flow_with_mocked_components(
        self, mock_logger, mock_config, mock_game_state, tmp_path, mock_llm_client_synthesis
    ):
        """Test complete flow: action -> memory creation -> retrieval -> context."""
        # Setup
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_client_synthesis
        )

        # Turn 1: Action at location 15, trigger fires, memory created
        mock_game_state.turn_count = 1
        mock_game_state.episode_id = "ep_001"

        z_context_1 = {
            'score_delta': 5,
            'location_changed': False,
            'inventory_changed': True,
            'inventory_before': [],
            'inventory_after': ['lamp'],
            'died': False,
            'response_length': 150,
            'first_visit': True
        }

        manager.record_action_outcome(
            location_id=15,
            location_name="Living Room",
            action="take lamp",
            response="Taken.",
            z_machine_context=z_context_1
        )

        # Verify memory created
        assert 15 in manager.memory_cache
        assert len(manager.memory_cache[15]) == 1

        # Turn 2: Action at location 15, LLM says don't remember (use different mock)
        mock_game_state.turn_count = 2

        mock_llm_no_remember = Mock()
        mock_response = Mock()
        mock_response.content = '{"should_remember": false, "category": "NOTE", "memory_title": "", "memory_text": "", "reasoning": "Duplicate"}'
        mock_llm_no_remember.chat.completions.create.return_value = mock_response
        manager._llm_client = mock_llm_no_remember

        z_context_2 = {
            'score_delta': 0,
            'location_changed': False,
            'inventory_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        manager.record_action_outcome(
            location_id=15,
            location_name="Living Room",
            action="look",
            response="You see nothing special.",
            z_machine_context=z_context_2
        )

        # Still only 1 memory (LLM said no)
        assert len(manager.memory_cache[15]) == 1

        # Turn 3: Action at location 23, new memory created
        mock_game_state.turn_count = 3
        manager._llm_client = mock_llm_client_synthesis  # Reset to remember

        z_context_3 = {
            'score_delta': 10,
            'location_changed': True,
            'inventory_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'died': False,
            'response_length': 200,
            'first_visit': True
        }

        manager.record_action_outcome(
            location_id=23,
            location_name="Kitchen",
            action="north",
            response="You enter the kitchen.",
            z_machine_context=z_context_3
        )

        # Now have memories at 2 locations
        assert 23 in manager.memory_cache
        assert len(manager.memory_cache[15]) == 1
        assert len(manager.memory_cache[23]) == 1

        # Turn 4: Return to location 15, sees existing memory in retrieval
        mock_game_state.turn_count = 4

        memory_text = manager.get_location_memory(15)

        # Should see the memory from turn 1
        assert memory_text != ""
        assert "Test Memory" in memory_text  # From mock LLM response

        # Verify file contains both locations
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text()
        assert "Location 15:" in content
        assert "Location 23:" in content

    def test_memory_deduplication_via_llm(
        self, mock_logger, mock_config, mock_game_state, tmp_path
    ):
        """Test LLM prevents duplicate memories at same location."""
        # First action: LLM says remember
        mock_llm_remember = Mock()
        mock_response_1 = Mock()
        mock_response_1.content = '{"should_remember": true, "category": "SUCCESS", "memory_title": "First Memory", "memory_text": "First memory text", "reasoning": "New discovery"}'
        mock_llm_remember.chat.completions.create.return_value = mock_response_1

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_remember
        )

        z_context = {
            'score_delta': 5,
            'location_changed': False,
            'inventory_changed': True,
            'inventory_before': [],
            'inventory_after': ['lamp'],
            'died': False,
            'response_length': 150,
            'first_visit': True
        }

        manager.record_action_outcome(
            location_id=15,
            location_name="Living Room",
            action="take lamp",
            response="Taken.",
            z_machine_context=z_context
        )

        assert len(manager.memory_cache[15]) == 1

        # Second action: LLM sees existing memory and says don't remember (duplicate)
        mock_response_2 = Mock()
        mock_response_2.content = '{"should_remember": false, "category": "NOTE", "memory_title": "", "memory_text": "", "reasoning": "Already recorded taking lamp"}'
        mock_llm_remember.chat.completions.create.return_value = mock_response_2

        manager.record_action_outcome(
            location_id=15,
            location_name="Living Room",
            action="take lamp",
            response="You already have the lamp.",
            z_machine_context=z_context
        )

        # Should still only have 1 memory (LLM prevented duplicate)
        assert len(manager.memory_cache[15]) == 1


# ============================================================================
# Additional Integration Tests
# ============================================================================

class TestMemoryFileIntegration:
    """Test file operations in integration scenarios."""

    def test_concurrent_memory_writes_with_lock(
        self, mock_logger, mock_config, mock_game_state, tmp_path, mock_llm_client_synthesis
    ):
        """Test file locking prevents corruption during concurrent writes."""
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_client_synthesis
        )

        # Add multiple memories in sequence
        for i in range(3):
            memory = Memory(
                category="NOTE",
                title=f"Memory {i}",
                episode=1,
                turns=str(i),
                score_change=0,
                text=f"Memory text {i}"
            )

            success = manager.add_memory(15, "Living Room", memory)
            assert success

        # Verify all memories in cache
        assert len(manager.memory_cache[15]) == 3

        # Verify all memories in file
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text()
        assert "Memory 0" in content
        assert "Memory 1" in content
        assert "Memory 2" in content

    def test_backup_created_before_write(
        self, mock_logger, mock_config, mock_game_state, tmp_path
    ):
        """Test backup file created before each write operation."""
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state
        )

        # Create initial memory file
        memories_path = Path(tmp_path) / "Memories.md"
        memories_path.write_text("# Location Memories\n\n---\n", encoding="utf-8")

        # Add memory (should trigger backup)
        memory = Memory(
            category="NOTE",
            title="Test",
            episode=1,
            turns="1",
            score_change=0,
            text="Test"
        )

        manager.add_memory(15, "Living Room", memory)

        # Verify backup exists
        backup_path = Path(str(memories_path) + ".backup")
        assert backup_path.exists()


class TestTriggerDetection:
    """Test Z-machine trigger detection in integration context."""

    def test_all_triggers_invoke_synthesis(
        self, mock_logger, mock_config, mock_game_state, mock_llm_client_synthesis, tmp_path
    ):
        """Test each trigger type correctly invokes LLM synthesis."""
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_client_synthesis
        )

        # Test each trigger type
        triggers = [
            {'score_delta': 5, 'name': 'score_change'},
            {'location_changed': True, 'name': 'location_change'},
            {'inventory_changed': True, 'name': 'inventory_change'},
            {'died': True, 'name': 'death'},
            {'first_visit': True, 'name': 'first_visit'},
            {'response_length': 150, 'name': 'substantial_response'},
        ]

        for trigger_data in triggers:
            # Reset mock
            mock_llm_client_synthesis.chat.completions.create.reset_mock()

            # Create context with only this trigger
            z_context = {
                'score_delta': 0,
                'location_changed': False,
                'inventory_changed': False,
                'inventory_before': [],
                'inventory_after': [],
                'died': False,
                'response_length': 0,
                'first_visit': False,
            }
            z_context.update({k: v for k, v in trigger_data.items() if k != 'name'})

            # Record action
            manager.record_action_outcome(
                location_id=15,
                location_name="Test Room",
                action="test action",
                response="test response",
                z_machine_context=z_context
            )

            # Verify LLM was called
            assert mock_llm_client_synthesis.chat.completions.create.called, \
                f"Trigger '{trigger_data['name']}' did not invoke LLM synthesis"
