"""
ABOUTME: Integration test for multi-step memory synthesis - Delayed consequence scenario.
ABOUTME: Tests supersession logic when initial success is contradicted by later consequence.

This test validates Phase 5 of the multi-step memory enhancement:
- Delayed consequence: Turn N action seems successful → Turn N+1 reveals failure
- Example: "give lunch to troll" (accepted) → troll attacks (gift strategy fails)
- Memory system must SUPERSEDE the optimistic TENTATIVE memory
- Tests that history context enables LLM to recognize contradictions
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List

from session.game_state import GameState
from session.game_configuration import GameConfiguration
from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemorySynthesisResponse


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    return logger


@pytest.fixture
def mock_game_state_turn1():
    """Mock GameState for Turn 1: give lunch to troll (seems successful)."""
    state = Mock(spec=GameState)
    state.episode_id = "ep_test_troll"
    state.turn_count = 1
    state.current_room_id = 52  # Troll Room
    state.current_room_name_for_map = "Troll Room"
    state.current_inventory = []  # Lunch removed from inventory
    state.previous_zork_score = 0

    # Action history: Just turn 1
    state.action_history = [
        ("give lunch to troll", "The troll, who is not overly proud, graciously accepts the gift and eats it hungrily.")
    ]

    # Reasoning history
    state.action_reasoning_history = [
        {
            "turn": 1,
            "reasoning": "The troll is blocking the path. I'll try giving it the lunch to see if it becomes friendly.",
            "action": "give lunch to troll",
            "timestamp": "2025-01-03T10:00:00Z"
        }
    ]

    state.memory_log_history = []
    state.action_counts = {}
    state.prev_room_for_prompt_context = "Troll Room"
    state.action_leading_to_current_room_for_prompt_context = ""
    return state


@pytest.fixture
def mock_game_state_turn2():
    """Mock GameState for Turn 2: troll attacks (delayed consequence)."""
    state = Mock(spec=GameState)
    state.episode_id = "ep_test_troll"
    state.turn_count = 2
    state.current_room_id = 52  # Still in Troll Room
    state.current_room_name_for_map = "Troll Room"
    state.current_inventory = []
    state.previous_zork_score = 0

    # Action history: Both turns
    state.action_history = [
        ("give lunch to troll", "The troll, who is not overly proud, graciously accepts the gift and eats it hungrily."),
        ("north", "The troll strikes you with a nasty blow from his axe.")  # Delayed attack
    ]

    # Reasoning history
    state.action_reasoning_history = [
        {
            "turn": 1,
            "reasoning": "The troll is blocking the path. I'll try giving it the lunch to see if it becomes friendly.",
            "action": "give lunch to troll",
            "timestamp": "2025-01-03T10:00:00Z"
        },
        {
            "turn": 2,
            "reasoning": "The troll accepted the lunch. I'll try moving north now that it seems pacified.",
            "action": "north",
            "timestamp": "2025-01-03T10:01:00Z"
        }
    ]

    state.memory_log_history = []
    state.action_counts = {}
    state.prev_room_for_prompt_context = "Troll Room"
    state.action_leading_to_current_room_for_prompt_context = ""
    return state


@pytest.fixture
def mock_config(tmp_path):
    """Mock GameConfiguration with temporary work directory."""
    config = Mock(spec=GameConfiguration)
    config.zork_game_workdir = str(tmp_path)
    config.info_ext_model = "gpt-4"
    config.memory_model = "gpt-4"
    config.simple_memory_file = "Memories.md"
    config.simple_memory_max_shown = 10
    config.max_turns_per_episode = 1000
    config.get_memory_history_window = Mock(return_value=3)  # Default window size
    config.memory_sampling = {"temperature": 0.3, "max_tokens": 1000}
    return config


@pytest.fixture
def mock_llm_tentative_memory():
    """Mock LLM client that creates TENTATIVE memory for Turn 1."""
    client = Mock()

    # Turn 1: Troll accepts gift (uncertain outcome, use TENTATIVE)
    mock_response = Mock()
    mock_response.content = """
    {
        "should_remember": true,
        "category": "NOTE",
        "memory_title": "Troll accepts lunch gift",
        "memory_text": "Troll accepts lunch offering graciously and eats it. Reaction to gift unclear, might allow passage.",
        "status": "TENTATIVE",
        "reasoning": "Troll accepted the gift, but long-term consequence unclear. Marking TENTATIVE until we know if it's pacified or still hostile."
    }
    """

    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def mock_llm_supersession_memory():
    """Mock LLM client that recognizes contradiction and supersedes previous memory."""
    client = Mock()

    # Turn 2: Troll attacks (contradiction detected, supersede turn 1 memory)
    mock_response = Mock()
    mock_response.content = """
    {
        "should_remember": true,
        "category": "DANGER",
        "memory_title": "Troll attacks after accepting gift",
        "memory_text": "Troll accepts lunch gift but then becomes hostile and attacks when attempting to pass. Gift strategy does not pacify troll.",
        "status": "ACTIVE",
        "supersedes_memory_titles": ["Troll accepts lunch gift"],
        "reasoning": "This contradicts the previous TENTATIVE memory. Troll acceptance did not lead to safe passage. Must supersede optimistic memory and warn against this strategy."
    }
    """

    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def create_memories_file():
    """Helper fixture to create Memories.md file with content."""
    def _create(tmp_path: Path, content: str = ""):
        memories_path = Path(tmp_path) / "Memories.md"
        if content:
            memories_path.write_text(content, encoding="utf-8")
        else:
            # Create minimal valid structure
            memories_path.write_text("# Location Memories\n\n", encoding="utf-8")
        return memories_path
    return _create


# ============================================================================
# Delayed Consequence Integration Test
# ============================================================================

class TestDelayedConsequenceTroll:
    """Test delayed consequence detection and supersession logic."""

    def test_turn1_creates_tentative_memory(
        self,
        mock_logger,
        mock_config,
        mock_game_state_turn1,
        mock_llm_tentative_memory,
        create_memories_file,
        tmp_path
    ):
        """
        Test Turn 1: Troll accepts lunch gift (creates TENTATIVE memory).

        Scenario:
        - Turn 1: "give lunch to troll" → troll accepts gift graciously
        - Action seems successful but outcome uncertain
        - Expected: TENTATIVE memory (not yet confirmed safe)
        """
        create_memories_file(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state_turn1,
            llm_client=mock_llm_tentative_memory
        )

        # Turn 1: Give lunch to troll (inventory changes, lunch removed)
        z_machine_context = {
            "location_id": 52,
            "location_name": "Troll Room",
            "inventory": [],  # Lunch removed
            "score": 0,
            "moves": 1,
            "visible_objects": ["troll"],
            "score_before": 0,
            "score_after": 0,
            "score_delta": 0,
            "location_before": 52,
            "location_after": 52,
            "location_changed": False,
            "inventory_before": ["lunch"],  # Had lunch
            "inventory_after": [],  # Lunch gone
            "inventory_changed": True,  # TRIGGER: Inventory changed
            "died": False,
            "response_length": 88,  # Length of troll acceptance response
            "first_visit": False
        }

        manager.record_action_outcome(
            location_id=52,
            location_name="Troll Room",
            action="give lunch to troll",
            response="The troll, who is not overly proud, graciously accepts the gift and eats it hungrily.",
            z_machine_context=z_machine_context
        )

        # Verify synthesis was triggered
        assert mock_llm_tentative_memory.chat.completions.create.called

        # Verify history included in prompt (Turn 1 only)
        call_args = mock_llm_tentative_memory.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        synthesis_prompt = messages[0]["content"]

        assert "Turn 1: give lunch to troll" in synthesis_prompt
        assert "graciously accepts the gift" in synthesis_prompt
        assert "try giving it the lunch" in synthesis_prompt  # Reasoning

        # Verify memory was stored at Troll Room (52)
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        assert "## Location 52: Troll Room" in content
        assert "[NOTE - TENTATIVE] Troll accepts lunch gift" in content  # Format includes status
        assert "accepts lunch offering" in content

    def test_turn2_supersedes_tentative_memory(
        self,
        mock_logger,
        mock_config,
        mock_game_state_turn2,
        mock_llm_supersession_memory,
        create_memories_file,
        tmp_path
    ):
        """
        Test Turn 2: Troll attacks (supersedes Turn 1 TENTATIVE memory).

        Scenario:
        - Turn 1: "give lunch to troll" → accepted (TENTATIVE memory exists)
        - Turn 2: "north" → troll attacks (contradiction detected)
        - Expected: Turn 1 memory marked SUPERSEDED, new DANGER memory created
        """
        # Create Memories.md with Turn 1 TENTATIVE memory already present
        turn1_memory = """# Location Memories

## Location 52: Troll Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[NOTE - TENTATIVE] Troll accepts lunch gift** *(Epep_test_troll, T1, +0)*
Troll accepts lunch offering graciously and eats it. Reaction to gift unclear, might allow passage.

---
"""
        create_memories_file(tmp_path, turn1_memory)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state_turn2,
            llm_client=mock_llm_supersession_memory
        )

        # Turn 2: Try to move north → troll attacks
        z_machine_context = {
            "location_id": 52,
            "location_name": "Troll Room",
            "inventory": [],
            "score": 0,
            "moves": 2,
            "visible_objects": ["troll"],
            "score_before": 0,
            "score_after": 0,
            "score_delta": 0,
            "location_before": 52,
            "location_after": 52,
            "location_changed": False,  # Attack prevented movement
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,  # Didn't die (yet), but took damage
            "response_length": 55,  # Length of attack response
            "first_visit": False
        }

        # Need a trigger - use substantial response length (>100)
        z_machine_context["response_length"] = 101  # Trigger synthesis

        manager.record_action_outcome(
            location_id=52,
            location_name="Troll Room",
            action="north",
            response="The troll strikes you with a nasty blow from his axe.",
            z_machine_context=z_machine_context
        )

        # Verify synthesis was triggered
        assert mock_llm_supersession_memory.chat.completions.create.called

        # Verify history included BOTH turns
        call_args = mock_llm_supersession_memory.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        synthesis_prompt = messages[0]["content"]

        # Turn 1 and Turn 2 should be in history
        assert "Turn 1: give lunch to troll" in synthesis_prompt
        assert "Turn 2: north" in synthesis_prompt
        assert "graciously accepts the gift" in synthesis_prompt  # Turn 1 response
        assert "nasty blow" in synthesis_prompt  # Turn 2 response (current action)

        # Verify multi-step guidance section about delayed consequences
        assert "Delayed Consequences" in synthesis_prompt

        # Verify memory file shows supersession
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Original memory should be marked SUPERSEDED
        assert "Troll accepts lunch gift" in content  # Original title still present
        assert "SUPERSEDED" in content  # Status changed

        # New DANGER memory should be present
        assert "[DANGER] Troll attacks after accepting gift" in content
        assert "does not pacify troll" in content
        assert "becomes hostile and attacks" in content

        # New memory should be ACTIVE (not TENTATIVE)
        # Check that the DANGER memory is ACTIVE by looking for the pattern
        danger_section_start = content.find("[DANGER] Troll attacks after accepting gift")
        danger_section = content[danger_section_start:danger_section_start + 500]
        assert "ACTIVE" not in danger_section or "TENTATIVE" not in danger_section  # Default is ACTIVE

    def test_delayed_consequence_with_death(
        self,
        mock_logger,
        mock_config,
        mock_game_state_turn2,
        create_memories_file,
        tmp_path
    ):
        """
        Test delayed consequence when attack causes death.

        Scenario:
        - Turn 1: Give lunch (TENTATIVE memory exists)
        - Turn 2: Troll attack kills player (death trigger + contradiction)
        - Expected: DANGER memory with death warning
        """
        # Create Turn 1 memory
        turn1_memory = """# Location Memories

## Location 52: Troll Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[NOTE - TENTATIVE] Troll accepts lunch gift** *(Epep_test_troll, T1, +0)*
Troll accepts lunch offering graciously and eats it. Reaction to gift unclear, might allow passage.

---
"""
        create_memories_file(tmp_path, turn1_memory)

        # Mock LLM that creates DANGER memory for death
        mock_llm_death = Mock()
        mock_response = Mock()
        mock_response.content = """
        {
            "should_remember": true,
            "category": "DANGER",
            "memory_title": "Troll kills after accepting gift",
            "memory_text": "Troll accepts lunch but then attacks and kills when attempting passage. Gift strategy is FATAL. Do not attempt.",
            "status": "ACTIVE",
            "supersedes_memory_titles": ["Troll accepts lunch gift"],
            "reasoning": "Death occurred after gift acceptance. This is a critical DANGER memory - gift does not work, leads to death."
        }
        """
        mock_llm_death.chat.completions.create.return_value = mock_response

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state_turn2,
            llm_client=mock_llm_death
        )

        # Turn 2: Troll kills player
        z_machine_context = {
            "location_id": 52,
            "location_name": "Troll Room",
            "inventory": [],
            "score": 0,
            "moves": 2,
            "visible_objects": ["troll"],
            "score_before": 0,
            "score_after": 0,
            "score_delta": 0,
            "location_before": 52,
            "location_after": 52,
            "location_changed": False,
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": True,  # TRIGGER: Death occurred
            "response_length": 75,
            "first_visit": False
        }

        manager.record_action_outcome(
            location_id=52,
            location_name="Troll Room",
            action="north",
            response="The troll's axe stroke cleaves you nearly in two. You die.",
            z_machine_context=z_machine_context
        )

        # Verify synthesis triggered (death is a trigger)
        assert mock_llm_death.chat.completions.create.called

        # Verify memory shows FATAL danger
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        assert "[DANGER] Troll kills after accepting gift" in content
        assert "FATAL" in content
        assert "Do not attempt" in content

        # Verify Turn 1 memory marked SUPERSEDED
        assert "Troll accepts lunch gift" in content
        assert "SUPERSEDED" in content


# ============================================================================
# Edge Cases and Validation Tests
# ============================================================================

class TestDelayedConsequenceEdgeCases:
    """Test edge cases for delayed consequence detection."""

    def test_no_supersession_without_history(
        self,
        mock_logger,
        mock_config,
        create_memories_file,
        tmp_path
    ):
        """
        Test that without history, Turn 2 cannot detect delayed consequence.

        This demonstrates the value of Phase 3 history integration.
        Without Turn 1 in history, the system can't connect the attack to the gift.
        """
        create_memories_file(tmp_path)

        # Game state with NO action history (simulating pre-Phase 3 system)
        state = Mock(spec=GameState)
        state.episode_id = "ep_no_history"
        state.turn_count = 2
        state.current_room_id = 52
        state.current_room_name_for_map = "Troll Room"
        state.current_inventory = []
        state.previous_zork_score = 0
        state.action_history = []  # Empty - no history
        state.action_reasoning_history = []  # Empty
        state.memory_log_history = []
        state.action_counts = {}
        state.prev_room_for_prompt_context = ""
        state.action_leading_to_current_room_for_prompt_context = ""

        # Mock LLM that creates simple attack memory (no supersession awareness)
        mock_llm_no_context = Mock()
        mock_response = Mock()
        mock_response.content = """
        {
            "should_remember": true,
            "category": "DANGER",
            "memory_title": "Troll attacks when moving north",
            "memory_text": "Troll attacks with axe when attempting to move north.",
            "reasoning": "Dangerous action, should remember."
        }
        """
        mock_llm_no_context.chat.completions.create.return_value = mock_response

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=state,
            llm_client=mock_llm_no_context
        )

        z_machine_context = {
            "location_id": 52,
            "location_name": "Troll Room",
            "inventory": [],
            "score": 0,
            "moves": 2,
            "visible_objects": ["troll"],
            "score_before": 0,
            "score_after": 0,
            "score_delta": 0,
            "location_before": 52,
            "location_after": 52,
            "location_changed": False,
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,
            "response_length": 101,  # Trigger synthesis
            "first_visit": False
        }

        manager.record_action_outcome(
            location_id=52,
            location_name="Troll Room",
            action="north",
            response="The troll strikes you with a nasty blow from his axe.",
            z_machine_context=z_machine_context
        )

        # Verify synthesis occurred
        assert mock_llm_no_context.chat.completions.create.called

        # Verify prompt did NOT include Turn 1 context
        call_args = mock_llm_no_context.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        synthesis_prompt = messages[0]["content"]

        assert "Turn 1: give lunch to troll" not in synthesis_prompt
        assert "graciously accepts" not in synthesis_prompt

        # Memory created but CANNOT connect to gift strategy
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        assert "[DANGER] Troll attacks when moving north" in content
        # Should NOT mention gift strategy (no context to connect them)
        assert "gift" not in content.lower() or "lunch" not in content.lower()

    def test_supersession_field_in_llm_response(
        self,
        mock_logger,
        mock_config,
        mock_game_state_turn2,
        create_memories_file,
        tmp_path
    ):
        """
        Test that supersedes_memory_titles field is properly used.

        Validates that the LLM response includes the supersession list
        and that SimpleMemoryManager processes it correctly.
        """
        # Create Turn 1 memory with specific title
        turn1_memory = """# Location Memories

## Location 52: Troll Room
**Visits:** 1 | **Episodes:** 1

### Memories

**[NOTE - TENTATIVE] Troll accepts lunch gift** *(Epep_test_troll, T1, +0)*
Troll accepts lunch offering graciously.

---
"""
        create_memories_file(tmp_path, turn1_memory)

        # Mock LLM with explicit supersession list
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
        {
            "should_remember": true,
            "category": "DANGER",
            "memory_title": "Troll attacks after gift",
            "memory_text": "Gift does not work.",
            "status": "ACTIVE",
            "supersedes_memory_titles": ["Troll accepts lunch gift"],
            "reasoning": "Contradicts previous memory."
        }
        """
        mock_llm.chat.completions.create.return_value = mock_response

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state_turn2,
            llm_client=mock_llm
        )

        z_machine_context = {
            "location_id": 52,
            "location_name": "Troll Room",
            "inventory": [],
            "score": 0,
            "moves": 2,
            "visible_objects": ["troll"],
            "score_before": 0,
            "score_after": 0,
            "score_delta": 0,
            "location_before": 52,
            "location_after": 52,
            "location_changed": False,
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,
            "response_length": 101,
            "first_visit": False
        }

        manager.record_action_outcome(
            location_id=52,
            location_name="Troll Room",
            action="north",
            response="The troll strikes you with a nasty blow from his axe.",
            z_machine_context=z_machine_context
        )

        # Verify supersession occurred
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Original memory should be SUPERSEDED
        assert "Troll accepts lunch gift" in content
        assert "SUPERSEDED" in content

        # Should show supersession metadata
        assert "Superseded at T2 by" in content  # Correct format: [Superseded at T2 by "..."]
        assert "Troll attacks after gift" in content  # New memory title in supersession note
