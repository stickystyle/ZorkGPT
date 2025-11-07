"""
ABOUTME: Tests for HistoryFormatter component used by SimpleMemoryManager.
ABOUTME: Covers format_recent_actions() and format_recent_reasoning() methods.
"""

import pytest
from typing import List, Tuple, Dict, Any
from unittest.mock import Mock

from managers.memory.formatting import HistoryFormatter


# ============================================================================
# Tests for format_recent_actions()
# ============================================================================

def test_format_empty_actions():
    """Test format_recent_actions() with empty list returns empty string."""
    formatter = HistoryFormatter()

    result = formatter.format_recent_actions([], start_turn=47)

    assert result == ""


def test_format_single_action():
    """Test format_recent_actions() with single action formats correctly."""
    formatter = HistoryFormatter()

    actions = [("go north", "You are in a forest clearing.")]
    result = formatter.format_recent_actions(actions, start_turn=47)

    expected = "Turn 47: go north\nResponse: You are in a forest clearing."
    assert result == expected


def test_format_multiple_actions():
    """Test format_recent_actions() with multiple actions formats correctly."""
    formatter = HistoryFormatter()

    actions = [
        ("go north", "You are in a forest clearing."),
        ("examine trees", "The trees are ordinary pine trees."),
        ("go east", "You are in a meadow.")
    ]
    result = formatter.format_recent_actions(actions, start_turn=47)

    expected = """Turn 47: go north
Response: You are in a forest clearing.

Turn 48: examine trees
Response: The trees are ordinary pine trees.

Turn 49: go east
Response: You are in a meadow."""

    assert result == expected


def test_format_actions_turn_numbering():
    """Test _format_recent_actions() turn numbers increment correctly from start_turn."""
    formatter = HistoryFormatter()

    actions = [
        ("action1", "response1"),
        ("action2", "response2"),
        ("action3", "response3")
    ]
    result = formatter.format_recent_actions(actions, start_turn=100)

    # Check that turn numbers are correct
    assert "Turn 100:" in result
    assert "Turn 101:" in result
    assert "Turn 102:" in result
    # Ensure no incorrect turn numbers
    assert "Turn 99:" not in result
    assert "Turn 103:" not in result


def test_format_actions_with_long_responses():
    """Test _format_recent_actions() handles long responses correctly."""
    formatter = HistoryFormatter()

    long_response = "This is a very long response " * 10
    actions = [
        ("examine room", long_response),
        ("take item", "Taken.")
    ]
    result = formatter.format_recent_actions(actions, start_turn=50)

    # Check formatting is intact
    assert "Turn 50: examine room" in result
    assert f"Response: {long_response}" in result
    assert "Turn 51: take item" in result
    assert "Response: Taken." in result


# ============================================================================
# Tests for _format_recent_reasoning()
# ============================================================================

def test_format_empty_reasoning():
    """Test _format_recent_reasoning() with empty list returns empty string."""
    formatter = HistoryFormatter()

    result = formatter.format_recent_reasoning([])

    assert result == ""


def test_format_single_reasoning():
    """Test _format_recent_reasoning() with single entry formats correctly."""
    formatter = HistoryFormatter()

    entries = [
        {
            "turn": 47,
            "reasoning": "I need to explore north systematically.",
            "action": "go north",
            "timestamp": "2025-11-03T10:00:00"
        }
    ]
    result = formatter.format_recent_reasoning(entries)

    expected = """Turn 47:
Reasoning: I need to explore north systematically.
Action: go north
Response: (Response not recorded)"""

    assert result == expected


def test_format_multiple_reasoning():
    """Test _format_recent_reasoning() with multiple entries formats correctly."""
    formatter = HistoryFormatter()

    entries = [
        {
            "turn": 47,
            "reasoning": "I need to explore north systematically.",
            "action": "go north",
            "timestamp": "2025-11-03T10:00:00"
        },
        {
            "turn": 48,
            "reasoning": "Will examine objects before moving on.",
            "action": "examine trees",
            "timestamp": "2025-11-03T10:01:00"
        },
        {
            "turn": 49,
            "reasoning": "Nothing interesting here. Moving east.",
            "action": "go east",
            "timestamp": "2025-11-03T10:02:00"
        }
    ]
    result = formatter.format_recent_reasoning(entries)

    expected = """Turn 47:
Reasoning: I need to explore north systematically.
Action: go north
Response: (Response not recorded)

Turn 48:
Reasoning: Will examine objects before moving on.
Action: examine trees
Response: (Response not recorded)

Turn 49:
Reasoning: Nothing interesting here. Moving east.
Action: go east
Response: (Response not recorded)"""

    assert result == expected


def test_format_reasoning_missing_fields():
    """Test _format_recent_reasoning() handles missing fields with fallbacks."""
    formatter = HistoryFormatter()

    entries = [
        {
            # Missing all optional fields
            "turn": 47
        },
        {
            "turn": 48,
            "action": "go north"
            # Missing reasoning
        },
        {
            "turn": 49,
            "reasoning": "Some reasoning"
            # Missing action
        }
    ]
    result = formatter.format_recent_reasoning(entries)

    # Check fallbacks are used
    assert "Turn 47:" in result
    assert "Reasoning: (No reasoning recorded)" in result
    assert "Action: (No action recorded)" in result
    assert "Response: (Response not recorded)" in result

    assert "Turn 48:" in result
    assert "Action: go north" in result

    assert "Turn 49:" in result
    assert "Reasoning: Some reasoning" in result


def test_format_reasoning_partial_data():
    """Test _format_recent_reasoning() with some fields present, some missing."""
    formatter = HistoryFormatter()

    entries = [
        {
            "turn": 47,
            "reasoning": "Valid reasoning",
            "action": "valid action"
        },
        {
            "turn": 48
            # Missing reasoning and action
        },
        {
            "turn": 49,
            "reasoning": "Another valid reasoning",
            "action": "another action"
        }
    ]
    result = formatter.format_recent_reasoning(entries)

    # Check first entry is normal
    assert "Turn 47:" in result
    assert "Reasoning: Valid reasoning" in result
    assert "Action: valid action" in result

    # Check second entry uses fallbacks
    assert "Turn 48:" in result
    assert result.count("(No reasoning recorded)") == 1
    assert result.count("(No action recorded)") == 1

    # Check third entry is normal
    assert "Turn 49:" in result
    assert "Reasoning: Another valid reasoning" in result
    assert "Action: another action" in result


def test_format_reasoning_non_dict_entries():
    """Test _format_recent_reasoning() skips non-dict entries gracefully."""
    formatter = HistoryFormatter()

    entries = [
        {
            "turn": 47,
            "reasoning": "Valid reasoning",
            "action": "valid action"
        },
        "not a dict",  # Invalid entry
        None,  # Invalid entry
        {
            "turn": 48,
            "reasoning": "Another valid reasoning",
            "action": "another action"
        }
    ]
    result = formatter.format_recent_reasoning(entries)

    # Check only valid entries are formatted
    assert "Turn 47:" in result
    assert "Turn 48:" in result
    # Check invalid entries are skipped
    assert "not a dict" not in result

    # Note: HistoryFormatter silently skips non-dict entries
    # No logging is performed at the formatter level


def test_format_reasoning_missing_turn_field():
    """Test _format_recent_reasoning() uses '?' for missing turn field."""
    formatter = HistoryFormatter()

    entries = [
        {
            # Missing turn field
            "reasoning": "Some reasoning",
            "action": "some action"
        }
    ]
    result = formatter.format_recent_reasoning(entries)

    # Check fallback for turn
    assert "Turn ?:" in result
    assert "Reasoning: Some reasoning" in result
    assert "Action: some action" in result


def test_format_reasoning_empty_strings():
    """Test _format_recent_reasoning() handles empty string fields correctly."""
    formatter = HistoryFormatter()

    entries = [
        {
            "turn": 47,
            "reasoning": "",  # Empty string (not missing)
            "action": ""  # Empty string (not missing)
        }
    ]
    result = formatter.format_recent_reasoning(entries)

    # Empty strings should be preserved (not replaced with fallback)
    assert "Turn 47:" in result
    assert "Reasoning: " in result
    assert "Action: " in result
    # But fallback should NOT appear
    assert "(No reasoning recorded)" not in result
    assert "(No action recorded)" not in result


def test_format_reasoning_blank_lines_between_entries():
    """Test _format_recent_reasoning() adds blank lines between entries but not after last."""
    formatter = HistoryFormatter()

    entries = [
        {"turn": 47, "reasoning": "First", "action": "action1"},
        {"turn": 48, "reasoning": "Second", "action": "action2"},
        {"turn": 49, "reasoning": "Third", "action": "action3"}
    ]
    result = formatter.format_recent_reasoning(entries)

    # Split by double newline to check blank line spacing
    sections = result.split("\n\n")

    # Should have 3 sections (one per entry, each with 4 lines now: Turn/Reasoning/Action/Response)
    assert len(sections) == 3

    # Result should not end with blank line
    assert not result.endswith("\n\n")

    # Verify each section has 4 lines
    for section in sections:
        lines = section.split("\n")
        assert len(lines) == 4, f"Expected 4 lines per entry, got {len(lines)}"


def test_format_reasoning_includes_response():
    """Test that responses are included when action_history provided."""
    formatter = HistoryFormatter()

    entries = [
        {
            "turn": 47,
            "reasoning": "I need to explore north systematically.",
            "action": "go north",
            "timestamp": "2025-11-03T10:00:00"
        },
        {
            "turn": 48,
            "reasoning": "Will examine objects before moving on.",
            "action": "examine trees",
            "timestamp": "2025-11-03T10:01:00"
        },
        {
            "turn": 49,
            "reasoning": "Nothing interesting here. Moving east.",
            "action": "go east",
            "timestamp": "2025-11-03T10:02:00"
        }
    ]

    action_history = [
        ("go north", "You are in a forest clearing."),
        ("examine trees", "The trees are ordinary pine trees."),
        ("go east", "You are in a meadow.")
    ]

    result = formatter.format_recent_reasoning(entries, action_history=action_history)

    # Verify responses are included from action_history
    assert "Response: You are in a forest clearing." in result
    assert "Response: The trees are ordinary pine trees." in result
    assert "Response: You are in a meadow." in result

    # Verify structure is complete
    expected = """Turn 47:
Reasoning: I need to explore north systematically.
Action: go north
Response: You are in a forest clearing.

Turn 48:
Reasoning: Will examine objects before moving on.
Action: examine trees
Response: The trees are ordinary pine trees.

Turn 49:
Reasoning: Nothing interesting here. Moving east.
Action: go east
Response: You are in a meadow."""

    assert result == expected


def test_format_reasoning_response_lookup_reverse_iteration():
    """Test that response lookup matches most recent occurrence when action repeats."""
    formatter = HistoryFormatter()

    entries = [
        {
            "turn": 47,
            "reasoning": "Going north to explore.",
            "action": "go north",
            "timestamp": "2025-11-03T10:00:00"
        }
    ]

    # Action history has duplicate "go north" actions
    action_history = [
        ("go north", "First response."),
        ("examine room", "You see nothing special."),
        ("go north", "Second response (most recent).")
    ]

    result = formatter.format_recent_reasoning(entries, action_history=action_history)

    # Should match the most recent "go north" response
    assert "Response: Second response (most recent)." in result
    assert "Response: First response." not in result


def test_format_reasoning_no_matching_response():
    """Test that fallback is used when no matching response in action_history."""
    formatter = HistoryFormatter()

    entries = [
        {
            "turn": 47,
            "reasoning": "Going north to explore.",
            "action": "go north",
            "timestamp": "2025-11-03T10:00:00"
        }
    ]

    # Action history doesn't contain "go north"
    action_history = [
        ("go south", "Response 1."),
        ("examine room", "Response 2.")
    ]

    result = formatter.format_recent_reasoning(entries, action_history=action_history)

    # Should use fallback
    assert "Response: (Response not recorded)" in result


def test_format_actions_blank_lines_between_entries():
    """Test _format_recent_actions() adds blank lines between entries but not after last."""
    formatter = HistoryFormatter()

    actions = [
        ("action1", "response1"),
        ("action2", "response2"),
        ("action3", "response3")
    ]
    result = formatter.format_recent_actions(actions, start_turn=47)

    # Split by double newline to check blank line spacing
    sections = result.split("\n\n")

    # Should have 3 sections (one per action)
    assert len(sections) == 3

    # Result should not end with blank line
    assert not result.endswith("\n\n")
