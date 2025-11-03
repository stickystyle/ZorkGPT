"""
ABOUTME: Integration test for multi-step memory synthesis - Window sequence scenario.
ABOUTME: Tests that the system captures multi-step procedures (open → enter window).

This test validates Phase 4 of the multi-step memory enhancement:
- Window sequence: examine window → open window → enter window → kitchen
- Memory should capture the full procedure at SOURCE location (Behind House)
- Tests that history context enables LLM to recognize multi-step patterns
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
def mock_game_state():
    """Mock GameState for window sequence testing."""
    state = Mock(spec=GameState)
    state.episode_id = "ep_test_window"
    state.turn_count = 3  # We'll be at turn 3 when entering window
    state.current_room_id = 79  # Behind House
    state.current_room_name_for_map = "Behind House"
    state.current_inventory = []
    state.previous_zork_score = 0

    # Action history for 3-turn window sequence
    state.action_history = [
        ("examine window", "The window is slightly ajar, but not open enough to allow entry."),
        ("open window", "With great effort, you open the window far enough to allow entry."),
        ("enter window", "You carefully climb through the window.")  # This action causes movement
    ]

    # Reasoning history - agent's strategic thinking
    state.action_reasoning_history = [
        {
            "turn": 1,
            "reasoning": "I should examine the window to understand if it can be opened or entered.",
            "action": "examine window",
            "timestamp": "2025-01-03T10:00:00Z"
        },
        {
            "turn": 2,
            "reasoning": "Window is ajar but needs to be opened more. I'll try opening it before attempting entry.",
            "action": "open window",
            "timestamp": "2025-01-03T10:01:00Z"
        },
        {
            "turn": 3,
            "reasoning": "Window is now open. I can enter it to reach the kitchen.",
            "action": "enter window",
            "timestamp": "2025-01-03T10:02:00Z"
        }
    ]

    state.memory_log_history = []
    state.action_counts = {}
    state.prev_room_for_prompt_context = "Behind House"
    state.action_leading_to_current_room_for_prompt_context = "enter window"
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
    config.get_memory_history_window = Mock(return_value=3)  # Window includes all 3 actions
    config.memory_sampling = {"temperature": 0.3, "max_tokens": 1000}
    return config


@pytest.fixture
def mock_llm_client_multi_step_synthesis():
    """Mock LLM client that recognizes multi-step procedure."""
    client = Mock()

    # The LLM should recognize the multi-step pattern from history
    mock_response = Mock()
    mock_response.content = """
    {
        "should_remember": true,
        "category": "SUCCESS",
        "memory_title": "Window entry to kitchen",
        "memory_text": "To enter kitchen from behind house: (1) examine window to confirm it's usable, (2) open window to make it passable, (3) enter window to reach kitchen. Window requires opening before entry is possible.",
        "reasoning": "This is a multi-step procedure that requires specific sequence. History shows three distinct actions were needed: examination revealed the window was ajar but not open enough, opening made it passable, then entry succeeded. This causal chain is important for future visits."
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
# Window Sequence Integration Test
# ============================================================================

class TestWindowSequenceMultiStep:
    """Test multi-step memory synthesis for window entry sequence."""

    def test_window_sequence_captures_full_procedure(
        self,
        mock_logger,
        mock_config,
        mock_game_state,
        mock_llm_client_multi_step_synthesis,
        create_memories_file,
        tmp_path
    ):
        """
        Test that window sequence (examine → open → enter) is captured as multi-step procedure.

        Scenario:
        - Turn 1: "examine window" - reveals window state
        - Turn 2: "open window" - makes window passable
        - Turn 3: "enter window" - causes movement to kitchen (location 203)

        Expected:
        - Memory synthesized at turn 3
        - Memory stored at location 79 (Behind House - SOURCE)
        - Memory includes full procedure: examine → open → enter
        - Memory captures causal relationship (opening required for entry)
        """
        # Create empty memories file
        create_memories_file(tmp_path)

        # Create manager with mocked LLM client
        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_client_multi_step_synthesis  # Inject mock
        )

        # Simulate the "enter window" action at Behind House (79)
        # This action causes movement to Kitchen (203)
        z_machine_context = {
            "location_id": 203,  # Destination (Kitchen)
            "location_name": "Kitchen",
            "inventory": [],
            "score": 5,  # New score
            "moves": 3,
            "visible_objects": [],
            # Required trigger fields
            "score_before": 0,
            "score_after": 5,
            "score_delta": 5,
            "location_before": 79,  # Behind House
            "location_after": 203,  # Kitchen
            "location_changed": True,  # Movement occurred
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,
            "response_length": 45,  # Length of "You carefully climb through the window."
            "first_visit": False
        }

        # Record the action outcome
        # CRITICAL: This should store at location 79 (source), NOT 203 (destination)
        manager.record_action_outcome(
            location_id=79,  # Behind House - SOURCE location
            location_name="Behind House",
            action="enter window",
            response="You carefully climb through the window.",
            z_machine_context=z_machine_context
        )

        # Verify LLM was called for synthesis
        assert mock_llm_client_multi_step_synthesis.chat.completions.create.called

        # Verify synthesis prompt included history context
        call_args = mock_llm_client_multi_step_synthesis.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        synthesis_prompt = messages[0]["content"]

        # Prompt should include action history section
        assert "RECENT ACTION SEQUENCE" in synthesis_prompt
        assert "Turn 1: examine window" in synthesis_prompt
        assert "Turn 2: open window" in synthesis_prompt
        assert "Turn 3: enter window" in synthesis_prompt

        # Prompt should include reasoning history section
        assert "AGENT'S REASONING" in synthesis_prompt
        assert "examine the window to understand" in synthesis_prompt
        assert "try opening it before attempting entry" in synthesis_prompt

        # Prompt should include multi-step guidance
        assert "MULTI-STEP PROCEDURE DETECTION" in synthesis_prompt
        assert "Prerequisites" in synthesis_prompt

        # Verify memory was stored
        memories_path = Path(tmp_path) / "Memories.md"
        assert memories_path.exists()

        content = memories_path.read_text(encoding="utf-8")

        # Memory should be stored at location 79 (Behind House)
        assert "## Location 79: Behind House" in content

        # Memory should capture multi-step procedure
        assert "[SUCCESS] Window entry to kitchen" in content
        assert "(1) examine window" in content
        assert "(2) open window" in content
        assert "(3) enter window" in content
        assert "requires opening before entry" in content

        # Metadata should show turn 3 and score delta
        assert "(Ep" in content  # Episode ID
        assert "T3" in content  # Turn 3
        assert "+5)" in content  # Score delta

        # Memory should NOT be stored at location 203 (Kitchen)
        assert "## Location 203: Kitchen" not in content

    def test_window_sequence_without_history_misses_procedure(
        self,
        mock_logger,
        mock_config,
        mock_game_state,
        create_memories_file,
        tmp_path
    ):
        """
        Test that WITHOUT history context, system would miss multi-step nature.

        This is a control test showing the value of Phase 3's history integration.
        Without history, the LLM only sees "enter window" → success, missing that
        opening was a prerequisite.
        """
        # Create empty memories file
        create_memories_file(tmp_path)

        # Mock LLM client that returns single-step memory (no multi-step awareness)
        mock_llm_no_context = Mock()
        mock_response = Mock()
        mock_response.content = """
        {
            "should_remember": true,
            "category": "SUCCESS",
            "memory_title": "Enter window works",
            "memory_text": "Entering the window leads to the kitchen.",
            "reasoning": "Action succeeded and moved to new location."
        }
        """
        mock_llm_no_context.chat.completions.create.return_value = mock_response

        # Remove history from game state to simulate old system
        mock_game_state.action_history = []
        mock_game_state.action_reasoning_history = []

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_no_context  # Inject mock
        )

        z_machine_context = {
            "location_id": 203,
            "location_name": "Kitchen",
            "inventory": [],
            "score": 5,
            "moves": 3,
            "visible_objects": [],
            "score_before": 0,
            "score_after": 5,
            "score_delta": 5,
            "location_before": 79,
            "location_after": 203,
            "location_changed": True,
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,
            "response_length": 45,
            "first_visit": False
        }

        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="enter window",
            response="You carefully climb through the window.",
            z_machine_context=z_machine_context
        )

        # Verify synthesis prompt did NOT include actual history data
        call_args = mock_llm_no_context.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        synthesis_prompt = messages[0]["content"]

        # Should not include actual turn data from history
        # (The multi-step guidance has examples like "open window" and "enter window",
        # so we check for the specific formatted turn data)
        assert "Turn 1: examine window" not in synthesis_prompt
        assert "Turn 2: open window" not in synthesis_prompt
        assert "Turn 3: enter window" not in synthesis_prompt
        assert "window is slightly ajar" not in synthesis_prompt  # Specific response from Turn 1
        assert "With great effort" not in synthesis_prompt  # Specific response from Turn 2

        # Should only see the current action (in ACTION ANALYSIS section)
        assert "Action: enter window" in synthesis_prompt  # Current action

        # Memory stored, but misses the multi-step nature
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        assert "## Location 79: Behind House" in content
        assert "[SUCCESS] Enter window works" in content

        # Should NOT mention multi-step procedure
        assert "(1)" not in content
        assert "(2)" not in content
        assert "open window" not in content  # Missing prerequisite knowledge

    def test_window_sequence_history_formatting(
        self,
        mock_logger,
        mock_config,
        mock_game_state,
        mock_llm_client_multi_step_synthesis,
        create_memories_file,
        tmp_path
    ):
        """
        Test that history is formatted correctly in synthesis prompt.

        Validates:
        - Turn numbers are sequential (1, 2, 3)
        - Actions are included with responses
        - Reasoning is included with correct turn attribution
        - Format matches ContextManager's format (for consistency)
        """
        create_memories_file(tmp_path)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_client_multi_step_synthesis  # Inject mock
        )

        z_machine_context = {
            "location_id": 203,
            "location_name": "Kitchen",
            "inventory": [],
            "score": 5,
            "moves": 3,
            "visible_objects": [],
            "score_before": 0,
            "score_after": 5,
            "score_delta": 5,
            "location_before": 79,
            "location_after": 203,
            "location_changed": True,
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,
            "response_length": 45,
            "first_visit": False
        }

        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="enter window",
            response="You carefully climb through the window.",
            z_machine_context=z_machine_context
        )

        # Extract synthesis prompt
        call_args = mock_llm_client_multi_step_synthesis.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        synthesis_prompt = messages[0]["content"]

        # Verify action history formatting
        # Format should be: "Turn N: action\nResponse: response\n"
        assert "Turn 1: examine window" in synthesis_prompt
        assert "Response: The window is slightly ajar, but not open enough to allow entry." in synthesis_prompt

        assert "Turn 2: open window" in synthesis_prompt
        assert "Response: With great effort, you open the window far enough to allow entry." in synthesis_prompt

        assert "Turn 3: enter window" in synthesis_prompt
        assert "Response: You carefully climb through the window." in synthesis_prompt

        # Verify reasoning history formatting
        # Format should include: Turn, Reasoning, Action, Response
        assert "Turn 1:" in synthesis_prompt
        assert "Reasoning: I should examine the window" in synthesis_prompt
        assert "Action: examine window" in synthesis_prompt

        assert "Turn 2:" in synthesis_prompt
        assert "Reasoning: Window is ajar but needs to be opened more" in synthesis_prompt
        assert "Action: open window" in synthesis_prompt

        assert "Turn 3:" in synthesis_prompt
        assert "Reasoning: Window is now open. I can enter it" in synthesis_prompt
        assert "Action: enter window" in synthesis_prompt

    def test_window_sequence_with_partial_history(
        self,
        mock_logger,
        mock_config,
        mock_game_state,
        mock_llm_client_multi_step_synthesis,
        create_memories_file,
        tmp_path
    ):
        """
        Test window sequence with only 2 actions in history (smaller window).

        Scenario: Window size is 2, so only "open window" and "enter window" are visible.
        The LLM should still synthesize useful memory, though less complete.
        """
        create_memories_file(tmp_path)

        # Reduce window size to 2
        mock_config.get_memory_history_window = Mock(return_value=2)

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_client_multi_step_synthesis  # Inject mock
        )

        z_machine_context = {
            "location_id": 203,
            "location_name": "Kitchen",
            "inventory": [],
            "score": 5,
            "moves": 3,
            "visible_objects": [],
            "score_before": 0,
            "score_after": 5,
            "score_delta": 5,
            "location_before": 79,
            "location_after": 203,
            "location_changed": True,
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,
            "response_length": 45,
            "first_visit": False
        }

        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="enter window",
            response="You carefully climb through the window.",
            z_machine_context=z_machine_context
        )

        # Extract synthesis prompt
        call_args = mock_llm_client_multi_step_synthesis.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        synthesis_prompt = messages[0]["content"]

        # Should include last 2 actions only (Turn 2 and Turn 3)
        assert "Turn 2: open window" in synthesis_prompt
        assert "Turn 3: enter window" in synthesis_prompt

        # Should NOT include Turn 1 (outside window)
        assert "Turn 1: examine window" not in synthesis_prompt

        # Reasoning should also be limited to last 2 entries
        reasoning_section_start = synthesis_prompt.find("AGENT'S REASONING")
        reasoning_section = synthesis_prompt[reasoning_section_start:]

        # Turn 1 reasoning should not appear
        assert "Turn 1:" not in reasoning_section
        assert "I should examine the window" not in reasoning_section


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestWindowSequenceEdgeCases:
    """Test edge cases for window sequence scenario."""

    def test_window_sequence_first_turn_no_history(
        self,
        mock_logger,
        mock_config,
        create_memories_file,
        tmp_path
    ):
        """
        Test window sequence at first turn (no history available).

        System should gracefully handle empty history and still synthesize memory.
        """
        create_memories_file(tmp_path)

        # Create game state at turn 1 (no history)
        state = Mock(spec=GameState)
        state.episode_id = "ep_first_turn"
        state.turn_count = 1
        state.current_room_id = 79
        state.current_room_name_for_map = "Behind House"
        state.current_inventory = []
        state.previous_zork_score = 0
        state.action_history = []  # Empty
        state.action_reasoning_history = []  # Empty
        state.memory_log_history = []
        state.action_counts = {}
        state.prev_room_for_prompt_context = ""
        state.action_leading_to_current_room_for_prompt_context = ""

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
        {
            "should_remember": false,
            "category": "NOTE",
            "memory_title": "",
            "memory_text": "",
            "reasoning": "First turn, no context yet."
        }
        """
        mock_llm.chat.completions.create.return_value = mock_response

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=state,
            llm_client=mock_llm  # Inject mock
        )

        z_machine_context = {
            "location_id": 79,
            "location_name": "Behind House",
            "inventory": [],
            "score": 0,
            "moves": 1,
            "visible_objects": [],
            "score_before": 0,
            "score_after": 0,
            "score_delta": 0,
            "location_before": 79,
            "location_after": 79,
            "location_changed": False,
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,
            "response_length": 35,  # Length of "You are behind the white house."
            "first_visit": True  # Trigger synthesis even with no movement
        }

        # Should not crash with empty history
        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="look",
            response="You are behind the white house.",
            z_machine_context=z_machine_context
        )

        # Verify synthesis was attempted
        assert mock_llm.chat.completions.create.called

        # Verify prompt handled empty history gracefully
        call_args = mock_llm.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        synthesis_prompt = messages[0]["content"]

        # Should not include actual turn data (empty history)
        # The multi-step guidance section will reference "RECENT ACTION SEQUENCE"
        # but there should be no actual turn data (Turn 1:, Turn 2:, etc.)
        assert "Turn 1:" not in synthesis_prompt
        assert "Turn 2:" not in synthesis_prompt

    def test_window_sequence_score_delta_calculation(
        self,
        mock_logger,
        mock_config,
        mock_game_state,
        mock_llm_client_multi_step_synthesis,
        create_memories_file,
        tmp_path
    ):
        """
        Test that score delta is calculated correctly for window entry.

        Entering window should award +5 points.
        """
        create_memories_file(tmp_path)

        # Set previous score to 0
        mock_game_state.previous_zork_score = 0

        manager = SimpleMemoryManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            llm_client=mock_llm_client_multi_step_synthesis  # Inject mock
        )

        z_machine_context = {
            "location_id": 203,
            "location_name": "Kitchen",
            "inventory": [],
            "score": 5,  # New score
            "moves": 3,
            "visible_objects": [],
            "score_before": 0,
            "score_after": 5,
            "score_delta": 5,
            "location_before": 79,
            "location_after": 203,
            "location_changed": True,
            "inventory_before": [],
            "inventory_after": [],
            "inventory_changed": False,
            "died": False,
            "response_length": 45,
            "first_visit": False
        }

        manager.record_action_outcome(
            location_id=79,
            location_name="Behind House",
            action="enter window",
            response="You carefully climb through the window.",
            z_machine_context=z_machine_context
        )

        # Check memory metadata includes correct score delta
        memories_path = Path(tmp_path) / "Memories.md"
        content = memories_path.read_text(encoding="utf-8")

        # Metadata format: *(EpID, TN, +DELTA)*
        assert "+5)" in content  # Score delta of +5

        # Extract and verify the full metadata
        # Should match: *(Ep<id>, T3, +5)*
        import re
        metadata_pattern = r'\(Ep[^,]+, T\d+, \+5\)'
        assert re.search(metadata_pattern, content), \
            f"Expected metadata with +5 score delta, got:\n{content}"
