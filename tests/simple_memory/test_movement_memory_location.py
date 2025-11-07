"""
ABOUTME: Unit tests for movement memory location storage correctness.
ABOUTME: Tests that memories are stored at SOURCE location (where action was taken), not DESTINATION.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from managers.simple_memory_manager import SimpleMemoryManager


class TestMovementMemoryStoredAtSource:
    """Test cases verifying movement memories are stored at the SOURCE location."""

    def test_movement_memory_stored_at_source_not_destination(
        self, mock_logger, game_config, game_state
    ):
        """Test that movement memory is stored at source location (79), not destination (203)."""
        # Setup: Create manager with empty memory cache
        manager = SimpleMemoryManager(
            logger=mock_logger, config=game_config, game_state=game_state
        )
        # Clear memory cache via cache_manager
        manager.cache_manager._memory_cache = {}

        # Mock LLM client to return a memory synthesis
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "should_remember": True,
            "category": "SUCCESS",
            "memory_title": "Enter through window",
            "memory_text": "Window provides entrance to kitchen from behind house.",
            "persistence": "permanent",
            "status": "ACTIVE",
            "supersedes_memory_titles": [],
            "reasoning": "Successful movement through window"
        })
        mock_response.usage = {"completion_tokens": 100, "prompt_tokens": 500}
        mock_llm_client.chat.completions.create.return_value = mock_response
        manager._llm_client = mock_llm_client
        manager._llm_client_initialized = True
        manager.synthesizer.llm_client = mock_llm_client  # Update synthesizer too

        # Create Z-machine context showing movement from location 79 to 203
        z_machine_context = {
            'score_before': 0,
            'score_after': 0,
            'score_delta': 0,
            'location_before': 79,  # Behind House
            'location_after': 203,  # Kitchen
            'location_changed': True,  # Movement occurred
            'inventory_before': [],
            'inventory_after': [],
            'inventory_changed': False,
            'died': False,
            'response_length': 100,
            'first_visit': True
        }

        # Action: Record action outcome
        # CRITICAL: location_id should be 79 (source), NOT 203 (destination)
        manager.record_action_outcome(
            location_id=79,  # Source location (Behind House)
            location_name="Behind House",
            action="enter window",
            response="You squeeze through the window into the kitchen.",
            z_machine_context=z_machine_context
        )

        # Assert: Memory should be stored at location 79 (source), not 203 (destination)
        assert 79 in manager.memory_cache, "Memory should exist at source location 79"
        assert 203 not in manager.memory_cache, "Memory should NOT exist at destination location 203"

        # Verify memory content
        memories_at_source = manager.memory_cache[79]
        assert len(memories_at_source) == 1, "Should have exactly one memory at source"
        memory = memories_at_source[0]
        assert memory.title == "Enter through window"
        assert memory.category == "SUCCESS"

    def test_non_movement_memory_stored_at_current_location(
        self, mock_logger, game_config, game_state
    ):
        """Test that non-movement memory is stored at current location (no movement)."""
        # Setup: Create manager with empty memory cache
        manager = SimpleMemoryManager(
            logger=mock_logger, config=game_config, game_state=game_state
        )
        # Clear memory cache via cache_manager
        manager.cache_manager._memory_cache = {}

        # Mock LLM client to return a memory synthesis
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "should_remember": True,
            "category": "DISCOVERY",
            "memory_title": "Mailbox contains leaflet",
            "memory_text": "Small mailbox here contains advertising leaflet.",
            "persistence": "permanent",
            "status": "ACTIVE",
            "supersedes_memory_titles": [],
            "reasoning": "Item discovery"
        })
        mock_response.usage = {"completion_tokens": 100, "prompt_tokens": 500}
        mock_llm_client.chat.completions.create.return_value = mock_response
        manager._llm_client = mock_llm_client
        manager._llm_client_initialized = True
        manager.synthesizer.llm_client = mock_llm_client  # Update synthesizer too

        # Create Z-machine context showing NO movement (stayed at location 79)
        z_machine_context = {
            'score_before': 0,
            'score_after': 0,
            'score_delta': 0,
            'location_before': 79,  # Behind House
            'location_after': 79,   # Still Behind House
            'location_changed': False,  # No movement
            'inventory_before': [],
            'inventory_after': ['leaflet'],
            'inventory_changed': True,
            'died': False,
            'response_length': 120,
            'first_visit': False
        }

        # Action: Record action outcome
        manager.record_action_outcome(
            location_id=79,  # Current location (Behind House)
            location_name="Behind House",
            action="examine mailbox",
            response="You open the mailbox and find an advertising leaflet inside.",
            z_machine_context=z_machine_context
        )

        # Assert: Memory should be stored at location 79 (current location)
        assert 79 in manager.memory_cache, "Memory should exist at location 79"

        # Verify memory content
        memories_at_location = manager.memory_cache[79]
        assert len(memories_at_location) == 1, "Should have exactly one memory at location"
        memory = memories_at_location[0]
        assert memory.title == "Mailbox contains leaflet"
        assert memory.category == "DISCOVERY"

    def test_memory_retrieval_at_source_location(
        self, mock_logger, game_config, game_state
    ):
        """Test that movement memory can be retrieved when agent returns to source location."""
        # Setup: Create manager with a movement memory at location 79
        manager = SimpleMemoryManager(
            logger=mock_logger, config=game_config, game_state=game_state
        )

        # Pre-populate memory cache with a movement memory at location 79
        from managers.simple_memory_manager import Memory
        movement_memory = Memory(
            category="SUCCESS",
            title="Enter through window",
            episode=1,
            turns="23",
            score_change=0,
            text="Window provides entrance to kitchen from behind house.",
            persistence="permanent",
            status="ACTIVE"
        )
        manager.memory_cache = {79: [movement_memory]}

        # Action: Retrieve memories for location 79 (returns formatted string)
        memory_text = manager.get_location_memory(location_id=79)

        # Assert: Memory should be retrievable
        assert memory_text is not None, "Should retrieve memories for location 79"
        assert len(memory_text) > 0, "Should have non-empty memory text"

        # Verify memory content appears in the formatted text
        assert "Enter through window" in memory_text, "Memory title should appear in text"
        assert "SUCCESS" in memory_text, "Memory category should appear in text"
        assert "window" in memory_text.lower(), "Memory text should contain 'window'"
        assert "kitchen" in memory_text.lower(), "Memory text should reference kitchen"


class TestMultipleMemoriesAtSameLocation:
    """Test cases for multiple memories at the same location."""

    def test_multiple_memories_accumulate_at_source(
        self, mock_logger, game_config, game_state
    ):
        """Test that multiple memories can accumulate at the same source location."""
        # Setup: Create manager with empty memory cache
        manager = SimpleMemoryManager(
            logger=mock_logger, config=game_config, game_state=game_state
        )
        # Clear memory cache via cache_manager
        manager.cache_manager._memory_cache = {}

        # Mock LLM client to return different memories
        mock_llm_client = Mock()
        manager._llm_client = mock_llm_client
        manager._llm_client_initialized = True

        # First memory: Enter window
        mock_response_1 = Mock()
        mock_response_1.content = json.dumps({
            "should_remember": True,
            "category": "SUCCESS",
            "memory_title": "Enter through window",
            "memory_text": "Window provides entrance to kitchen.",
            "persistence": "permanent",
            "status": "ACTIVE",
            "supersedes_memory_titles": [],
            "reasoning": "Movement success"
        })
        mock_response_1.usage = {"completion_tokens": 100, "prompt_tokens": 500}

        # Second memory: Examine mailbox
        mock_response_2 = Mock()
        mock_response_2.content = json.dumps({
            "should_remember": True,
            "category": "DISCOVERY",
            "memory_title": "Mailbox contains leaflet",
            "memory_text": "Small mailbox contains advertising leaflet.",
            "persistence": "permanent",
            "status": "ACTIVE",
            "supersedes_memory_titles": [],
            "reasoning": "Item discovery"
        })
        mock_response_2.usage = {"completion_tokens": 100, "prompt_tokens": 500}

        # Set up mock to return different responses on consecutive calls
        mock_llm_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        manager.synthesizer.llm_client = mock_llm_client  # Update synthesizer too

        # Record first action (movement through window)
        z_machine_context_1 = {
            'score_before': 0,
            'score_after': 0,
            'score_delta': 0,
            'location_before': 79,
            'location_after': 203,
            'location_changed': True,
            'inventory_before': [],
            'inventory_after': [],
            'inventory_changed': False,
            'died': False,
            'response_length': 100,
            'first_visit': True
        }

        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="enter window",
            response="You squeeze through the window into the kitchen.",
            z_machine_context=z_machine_context_1
        )

        # Record second action (examine mailbox, after returning to location 79)
        z_machine_context_2 = {
            'score_before': 0,
            'score_after': 0,
            'score_delta': 0,
            'location_before': 79,
            'location_after': 79,
            'location_changed': False,
            'inventory_before': [],
            'inventory_after': ['leaflet'],
            'inventory_changed': True,
            'died': False,
            'response_length': 120,
            'first_visit': False
        }

        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="examine mailbox",
            response="You find a leaflet in the mailbox.",
            z_machine_context=z_machine_context_2
        )

        # Assert: Both memories should be at location 79
        assert 79 in manager.memory_cache, "Memory should exist at location 79"
        assert len(manager.memory_cache[79]) == 2, "Should have two memories at location 79"

        # Verify both memory titles
        memory_titles = {mem.title for mem in manager.memory_cache[79]}
        assert "Enter through window" in memory_titles
        assert "Mailbox contains leaflet" in memory_titles


class TestEdgeCases:
    """Test edge cases for memory location storage."""

    def test_memory_not_stored_when_synthesis_returns_should_not_remember(
        self, mock_logger, game_config, game_state
    ):
        """Test that no memory is stored when LLM returns should_remember=False."""
        # Setup: Create manager with empty memory cache
        manager = SimpleMemoryManager(
            logger=mock_logger, config=game_config, game_state=game_state
        )
        # Clear memory cache via cache_manager
        manager.cache_manager._memory_cache = {}

        # Mock LLM client to return should_remember=False
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "should_remember": False,
            "reasoning": "Trivial action, not worth remembering"
        })
        mock_response.usage = {"completion_tokens": 100, "prompt_tokens": 500}
        mock_llm_client.chat.completions.create.return_value = mock_response
        manager._llm_client = mock_llm_client
        manager._llm_client_initialized = True
        manager.synthesizer.llm_client = mock_llm_client  # Update synthesizer too

        # Create Z-machine context showing movement
        z_machine_context = {
            'score_before': 0,
            'score_after': 0,
            'score_delta': 0,
            'location_before': 79,
            'location_after': 203,
            'location_changed': True,
            'inventory_before': [],
            'inventory_after': [],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        # Action: Record action outcome
        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="look",
            response="You see nothing special.",
            z_machine_context=z_machine_context
        )

        # Assert: No memory should be stored
        assert 79 not in manager.memory_cache or len(manager.memory_cache[79]) == 0

    def test_memory_stored_at_location_zero_for_invalid_location(
        self, mock_logger, game_config, game_state
    ):
        """Test that memory is stored at location 0 when location_id is invalid."""
        # Setup: Create manager with empty memory cache
        manager = SimpleMemoryManager(
            logger=mock_logger, config=game_config, game_state=game_state
        )
        # Clear memory cache via cache_manager
        manager.cache_manager._memory_cache = {}

        # Mock LLM client to return a memory synthesis (need to set on both manager and synthesizer)
        mock_llm_client = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "should_remember": True,
            "category": "NOTE",
            "memory_title": "Invalid location test",
            "memory_text": "Testing invalid location handling.",
            "persistence": "permanent",
            "status": "ACTIVE",
            "supersedes_memory_titles": [],
            "reasoning": "Test case"
        })
        mock_response.usage = {"completion_tokens": 100, "prompt_tokens": 500}
        mock_response.usage = {"completion_tokens": 100, "prompt_tokens": 500}
        mock_llm_client.chat.completions.create.return_value = mock_response
        manager._llm_client = mock_llm_client
        manager._llm_client_initialized = True
        manager.synthesizer.llm_client = mock_llm_client  # Update synthesizer too
        manager.synthesizer.llm_client = mock_llm_client  # Update synthesizer too

        # Create Z-machine context with location 0
        z_machine_context = {
            'score_before': 0,
            'score_after': 0,
            'score_delta': 0,
            'location_before': 0,
            'location_after': 0,
            'location_changed': False,
            'inventory_before': [],
            'inventory_after': [],
            'inventory_changed': True,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        # Action: Record action outcome with location_id=0
        manager.record_action_outcome(
            location_id=0,
            location_name="Unknown",
            action="test action",
            response="Test response.",
            z_machine_context=z_machine_context
        )

        # Assert: Memory should be stored at location 0
        assert 0 in manager.memory_cache, "Memory should exist at location 0"

    def test_no_memory_stored_when_trigger_not_fired(
        self, mock_logger, game_config, game_state
    ):
        """Test that no memory is stored when no trigger fires (no significant event)."""
        # Setup: Create manager with empty memory cache
        manager = SimpleMemoryManager(
            logger=mock_logger, config=game_config, game_state=game_state
        )
        # Clear memory cache via cache_manager
        manager.cache_manager._memory_cache = {}

        # Create Z-machine context with NO triggers (no score change, no movement, no inventory change)
        z_machine_context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 79,
            'location_after': 79,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        # Action: Record action outcome (should short-circuit before LLM call)
        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="look",
            response="You see the same scene.",
            z_machine_context=z_machine_context
        )

        # Assert: No memory should be stored (no trigger fired)
        assert 79 not in manager.memory_cache or len(manager.memory_cache[79]) == 0

        # Verify that debug log was called with "No trigger fired"
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("No trigger fired" in call or "skipping synthesis" in call.lower() for call in debug_calls)
