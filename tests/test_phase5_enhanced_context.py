"""
Tests for Phase 5.2: Enhanced ContextManager with Structured World Snapshot.

Verifies that the ContextManager properly integrates with JerichoInterface
to provide structured object data to the Agent.
"""

import pytest
from unittest.mock import Mock, MagicMock
from managers.context_manager import ContextManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return Mock()


@pytest.fixture
def game_config():
    """Create a test game configuration."""
    return GameConfiguration(
        max_turns_per_episode=1000,
        turn_delay_seconds=0.0,
        game_file_path="test_game.z5",
        critic_rejection_threshold=0.5,
        episode_log_file="test.log",
        json_log_file="test.jsonl",
        state_export_file="test_state.json",
        zork_game_workdir="test_workdir",
        client_base_url="http://localhost:1234",
        client_api_key="test_key",
        agent_model="test-agent",
        critic_model="test-critic",
        info_ext_model="test-extractor",
        analysis_model="test-analysis",
        memory_model="test-memory",
        condensation_model="test-condensation",
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
        map_state_file="test_map_state.json",
        knowledge_file="test_knowledgebase.md",
        # Sampling parameters
        agent_sampling={},
        critic_sampling={},
        extractor_sampling={},
        analysis_sampling={},
        memory_sampling={},
        condensation_sampling={},
    )


@pytest.fixture
def game_state():
    """Create a test game state."""
    state = GameState()
    state.episode_id = "test_episode_001"
    state.turn_count = 10
    state.current_room_name_for_map = "West of House"
    state.current_room_id = 1
    state.previous_zork_score = 0
    state.current_inventory = []
    return state


@pytest.fixture
def context_manager(mock_logger, game_config, game_state):
    """Create a ContextManager instance."""
    return ContextManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )


@pytest.fixture
def mock_jericho_interface():
    """Create a mock JerichoInterface with structured data."""
    interface = Mock()

    # Mock inventory objects
    mock_leaflet = Mock()
    mock_leaflet.num = 18
    mock_leaflet.name = "leaflet"

    mock_lamp = Mock()
    mock_lamp.num = 23
    mock_lamp.name = "brass lantern"

    interface.get_inventory_structured.return_value = [mock_leaflet, mock_lamp]

    # Mock visible objects
    mock_mailbox = Mock()
    mock_mailbox.num = 33
    mock_mailbox.name = "small mailbox"

    interface.get_visible_objects_in_location.return_value = [mock_mailbox]

    # Mock object attributes
    def mock_get_attributes(obj):
        if obj.num == 18:  # leaflet
            return {
                'touched': True,
                'container': False,
                'openable': False,
                'takeable': True,
                'transparent': False,
                'portable': True,
                'readable': True,
            }
        elif obj.num == 23:  # lamp
            return {
                'touched': True,
                'container': False,
                'openable': False,
                'takeable': True,
                'transparent': False,
                'portable': True,
                'readable': False,
            }
        elif obj.num == 33:  # mailbox
            return {
                'touched': False,
                'container': True,
                'openable': True,
                'takeable': False,
                'transparent': False,
                'portable': False,
                'readable': False,
            }
        return {}

    interface.get_object_attributes.side_effect = mock_get_attributes

    # Mock valid verbs
    interface.get_valid_verbs.return_value = [
        'go', 'north', 'south', 'east', 'west', 'up', 'down',
        'take', 'drop', 'open', 'close', 'examine', 'look', 'read',
        'inventory', 'attack', 'wait', 'save', 'restore', 'quit'
    ]

    return interface


class TestEnhancedContextWithJericho:
    """Test enhanced context functionality with Jericho integration."""

    def test_enhanced_context_includes_structured_inventory(
        self, context_manager, game_state, mock_jericho_interface
    ):
        """Test that enhanced context includes structured inventory data."""
        context = context_manager.get_agent_context(
            current_state="You are west of a house.",
            inventory=["leaflet", "brass lantern"],
            location="West of House",
            location_id=1,
            jericho_interface=mock_jericho_interface,
        )

        # Verify inventory_objects is present
        assert "inventory_objects" in context
        assert len(context["inventory_objects"]) == 2

        # Verify structure of inventory objects
        leaflet = context["inventory_objects"][0]
        assert leaflet["id"] == 18
        assert leaflet["name"] == "leaflet"
        assert "attributes" in leaflet
        assert leaflet["attributes"]["takeable"] is True
        assert leaflet["attributes"]["portable"] is True
        assert leaflet["attributes"]["readable"] is True

        lamp = context["inventory_objects"][1]
        assert lamp["id"] == 23
        assert lamp["name"] == "brass lantern"
        assert lamp["attributes"]["takeable"] is True
        assert lamp["attributes"]["portable"] is True
        assert lamp["attributes"]["readable"] is False

    def test_enhanced_context_includes_visible_objects(
        self, context_manager, game_state, mock_jericho_interface
    ):
        """Test that enhanced context includes visible objects with attributes."""
        context = context_manager.get_agent_context(
            current_state="You are west of a house. There is a mailbox here.",
            inventory=[],
            location="West of House",
            location_id=1,
            jericho_interface=mock_jericho_interface,
        )

        # Verify visible_objects is present
        assert "visible_objects" in context
        assert len(context["visible_objects"]) == 1

        # Verify structure of visible objects
        mailbox = context["visible_objects"][0]
        assert mailbox["id"] == 33
        assert mailbox["name"] == "small mailbox"
        assert "attributes" in mailbox
        assert mailbox["attributes"]["container"] is True
        assert mailbox["attributes"]["openable"] is True
        assert mailbox["attributes"]["takeable"] is False

    def test_enhanced_context_includes_action_vocabulary(
        self, context_manager, game_state, mock_jericho_interface
    ):
        """Test that enhanced context includes action vocabulary."""
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=[],
            location="Test Room",
            location_id=1,
            jericho_interface=mock_jericho_interface,
        )

        # Verify action_vocabulary is present
        assert "action_vocabulary" in context
        assert len(context["action_vocabulary"]) > 0
        assert "take" in context["action_vocabulary"]
        assert "open" in context["action_vocabulary"]
        assert "examine" in context["action_vocabulary"]


class TestBackwardCompatibility:
    """Test backward compatibility when jericho_interface is not provided."""

    def test_context_works_without_jericho_interface(
        self, context_manager, game_state
    ):
        """Test that context still works when jericho_interface=None."""
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=["sword"],
            location="Test Room",
            location_id=1,
            jericho_interface=None,  # Explicit None
        )

        # Verify basic context structure
        assert context["game_state"] == "You are in a room."
        assert context["inventory"] == ["sword"]
        assert context["current_location"] == "Test Room"

        # Verify Jericho-specific fields are NOT present
        assert "inventory_objects" not in context
        assert "visible_objects" not in context
        assert "action_vocabulary" not in context

    def test_context_works_when_jericho_interface_omitted(
        self, context_manager, game_state
    ):
        """Test that context works when jericho_interface parameter is omitted."""
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=["sword"],
            location="Test Room",
        )

        # Should still work
        assert context["game_state"] == "You are in a room."
        assert "inventory_objects" not in context

    def test_graceful_degradation_on_jericho_failure(
        self, context_manager, game_state
    ):
        """Test graceful degradation when Jericho methods fail."""
        # Create a mock that raises exceptions
        broken_jericho = Mock()
        broken_jericho.get_inventory_structured.side_effect = Exception("Jericho error")
        broken_jericho.get_visible_objects_in_location.side_effect = Exception("Jericho error")
        broken_jericho.get_valid_verbs.side_effect = Exception("Jericho error")

        # Should not crash
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=["sword"],
            location="Test Room",
            jericho_interface=broken_jericho,
        )

        # Should still have basic context
        assert context["game_state"] == "You are in a room."
        assert context["inventory"] == ["sword"]

        # Jericho fields should not be present (graceful degradation)
        assert "inventory_objects" not in context
        assert "visible_objects" not in context
        assert "action_vocabulary" not in context


class TestFormattedPromptWithStructuredData:
    """Test that formatted prompts include new structured data."""

    def test_formatted_prompt_includes_inventory_details(
        self, context_manager, game_state, mock_jericho_interface
    ):
        """Test that formatted prompt includes inventory details."""
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=["leaflet", "brass lantern"],
            location="Test Room",
            location_id=1,
            jericho_interface=mock_jericho_interface,
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Verify inventory details section is present
        assert "INVENTORY DETAILS:" in formatted
        assert "leaflet (ID:18," in formatted
        assert "brass lantern (ID:23," in formatted
        assert "takeable" in formatted
        assert "portable" in formatted
        assert "readable" in formatted

    def test_formatted_prompt_includes_visible_objects(
        self, context_manager, game_state, mock_jericho_interface
    ):
        """Test that formatted prompt includes visible objects section."""
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=[],
            location="Test Room",
            location_id=1,
            jericho_interface=mock_jericho_interface,
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Verify visible objects section is present
        assert "VISIBLE OBJECTS:" in formatted
        assert "small mailbox (ID:33," in formatted
        assert "container" in formatted
        assert "openable" in formatted

    def test_formatted_prompt_includes_action_count(
        self, context_manager, game_state, mock_jericho_interface
    ):
        """Test that formatted prompt mentions action vocabulary count."""
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=[],
            location="Test Room",
            location_id=1,
            jericho_interface=mock_jericho_interface,
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Verify action vocabulary count is mentioned (not full list)
        assert "VALID ACTIONS:" in formatted
        assert "verbs available" in formatted
        # Should show count, not full verb list
        assert "20 verbs available" in formatted

    def test_formatted_prompt_without_jericho_data(
        self, context_manager, game_state
    ):
        """Test that formatted prompt works without Jericho data."""
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=["sword"],
            location="Test Room",
            jericho_interface=None,
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Should have basic sections
        assert "CURRENT LOCATION:" in formatted
        assert "INVENTORY:" in formatted

        # Should NOT have Jericho sections
        assert "INVENTORY DETAILS:" not in formatted
        assert "VISIBLE OBJECTS:" not in formatted
        assert "VALID ACTIONS:" not in formatted


class TestAttributeFormatting:
    """Test proper formatting of object attributes."""

    def test_only_true_attributes_shown(
        self, context_manager, game_state, mock_jericho_interface
    ):
        """Test that only attributes set to True are shown in formatted output."""
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=["leaflet"],
            location="Test Room",
            location_id=1,
            jericho_interface=mock_jericho_interface,
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Leaflet has: touched, takeable, portable, readable = True
        # Should show these attributes
        assert "touched" in formatted
        assert "takeable" in formatted
        assert "portable" in formatted
        assert "readable" in formatted

        # Should NOT show False attributes like 'container', 'openable'
        # (They might appear if other objects have them, so we check the leaflet line)
        leaflet_line = [line for line in formatted.split('\n') if 'leaflet' in line][0]
        assert "container" not in leaflet_line
        assert "openable" not in leaflet_line

    def test_empty_attributes_handled_gracefully(
        self, context_manager, game_state
    ):
        """Test handling of objects with no attributes set."""
        # Create mock with no attributes
        mock_jericho = Mock()
        mock_obj = Mock()
        mock_obj.num = 1
        mock_obj.name = "test object"

        mock_jericho.get_inventory_structured.return_value = [mock_obj]
        mock_jericho.get_visible_objects_in_location.return_value = []
        mock_jericho.get_object_attributes.return_value = {}  # No attributes
        mock_jericho.get_valid_verbs.return_value = []

        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=["test object"],
            location="Test Room",
            jericho_interface=mock_jericho,
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Should still format properly with empty attribute string
        assert "test object (ID:1," in formatted


class TestReasoningHistoryFormatting:
    """Test formatting of reasoning history for agent context."""

    def test_get_recent_reasoning_formatted_empty_history(
        self, context_manager, game_state
    ):
        """Test formatting with empty reasoning history."""
        # Clear any existing reasoning history
        game_state.action_reasoning_history.clear()

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Should return empty string
        assert formatted == ""

    def test_get_recent_reasoning_formatted_single_entry(
        self, context_manager, game_state
    ):
        """Test formatting with single reasoning entry."""
        # Add one reasoning entry
        game_state.action_reasoning_history.append({
            "turn": 1,
            "reasoning": "I should explore north to find new areas.",
            "action": "go north",
            "timestamp": "2025-11-02T10:00:00"
        })

        # Add matching action history
        game_state.action_history.append(
            ("go north", "You are in a forest clearing.")
        )

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Verify formatting
        assert "Turn 1:" in formatted
        assert "Reasoning: I should explore north to find new areas." in formatted
        assert "Action: go north" in formatted
        assert "Response: You are in a forest clearing." in formatted

    def test_get_recent_reasoning_formatted_multiple_entries(
        self, context_manager, game_state
    ):
        """Test formatting with multiple reasoning entries."""
        # Add three reasoning entries
        entries = [
            {
                "turn": 47,
                "reasoning": "I need to explore north systematically. Plan: (1) go north, (2) search area, (3) return if nothing found.",
                "action": "go north",
                "timestamp": "2025-11-02T10:00:00"
            },
            {
                "turn": 48,
                "reasoning": "Continuing systematic exploration. Will examine objects before moving on.",
                "action": "examine trees",
                "timestamp": "2025-11-02T10:01:00"
            },
            {
                "turn": 49,
                "reasoning": "Nothing interesting here. Moving east to continue exploration.",
                "action": "go east",
                "timestamp": "2025-11-02T10:02:00"
            }
        ]

        for entry in entries:
            game_state.action_reasoning_history.append(entry)

        # Add matching action history
        game_state.action_history.extend([
            ("go north", "You are in a forest clearing. Trees surround you."),
            ("examine trees", "The trees are ordinary pine trees."),
            ("go east", "You are in a meadow.")
        ])

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Verify all three turns are present
        assert "Turn 47:" in formatted
        assert "Turn 48:" in formatted
        assert "Turn 49:" in formatted

        # Verify reasoning is present
        assert "I need to explore north systematically" in formatted
        assert "Continuing systematic exploration" in formatted
        assert "Nothing interesting here" in formatted

        # Verify actions are present
        assert "Action: go north" in formatted
        assert "Action: examine trees" in formatted
        assert "Action: go east" in formatted

        # Verify responses are present
        assert "Response: You are in a forest clearing." in formatted
        assert "Response: The trees are ordinary pine trees." in formatted
        assert "Response: You are in a meadow." in formatted

    def test_get_recent_reasoning_formatted_limits_turns(
        self, context_manager, game_state
    ):
        """Test that formatting limits to num_turns entries."""
        # Add 5 reasoning entries
        for i in range(1, 6):
            game_state.action_reasoning_history.append({
                "turn": i,
                "reasoning": f"Reasoning for turn {i}",
                "action": f"action {i}",
                "timestamp": "2025-11-02T10:00:00"
            })
            game_state.action_history.append((f"action {i}", f"response {i}"))

        # Request only last 3
        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Should only include turns 3, 4, 5
        assert "Turn 1:" not in formatted
        assert "Turn 2:" not in formatted
        assert "Turn 3:" in formatted
        assert "Turn 4:" in formatted
        assert "Turn 5:" in formatted

    def test_get_recent_reasoning_formatted_handles_missing_reasoning(
        self, context_manager, game_state
    ):
        """Test handling of entry with missing reasoning field."""
        game_state.action_reasoning_history.append({
            "turn": 1,
            "action": "go north",
            "timestamp": "2025-11-02T10:00:00"
            # Missing "reasoning" field
        })

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Should use fallback text
        assert "Turn 1:" in formatted
        assert "(No reasoning recorded)" in formatted
        assert "Action: go north" in formatted

    def test_get_recent_reasoning_formatted_handles_missing_action(
        self, context_manager, game_state
    ):
        """Test handling of entry with missing action field."""
        game_state.action_reasoning_history.append({
            "turn": 1,
            "reasoning": "Some reasoning",
            "timestamp": "2025-11-02T10:00:00"
            # Missing "action" field
        })

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Should use fallback text
        assert "Turn 1:" in formatted
        assert "Reasoning: Some reasoning" in formatted
        assert "(No action recorded)" in formatted

    def test_get_recent_reasoning_formatted_handles_missing_turn(
        self, context_manager, game_state
    ):
        """Test handling of entry with missing turn field."""
        game_state.action_reasoning_history.append({
            "reasoning": "Some reasoning",
            "action": "go north",
            "timestamp": "2025-11-02T10:00:00"
            # Missing "turn" field
        })

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Should skip entry without turn number
        assert formatted == ""

    def test_get_recent_reasoning_formatted_handles_non_dict_entries(
        self, context_manager, game_state
    ):
        """Test graceful handling of non-dict entries."""
        # Add invalid entries
        game_state.action_reasoning_history.extend([
            "invalid string entry",
            None,
            123,
            ["list", "entry"]
        ])

        # Add one valid entry
        game_state.action_reasoning_history.append({
            "turn": 1,
            "reasoning": "Valid reasoning",
            "action": "go north",
            "timestamp": "2025-11-02T10:00:00"
        })

        game_state.action_history.append(("go north", "You go north."))

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=10)

        # Should skip invalid entries and format the valid one
        assert "Turn 1:" in formatted
        assert "Valid reasoning" in formatted

    def test_get_recent_reasoning_formatted_handles_missing_response(
        self, context_manager, game_state
    ):
        """Test handling when response is not found in action history."""
        game_state.action_reasoning_history.append({
            "turn": 1,
            "reasoning": "Some reasoning",
            "action": "go north",
            "timestamp": "2025-11-02T10:00:00"
        })

        # Don't add matching action history

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Should use fallback text for response
        assert "Turn 1:" in formatted
        assert "Response: (Response not recorded)" in formatted

    def test_get_recent_reasoning_formatted_preserves_long_reasoning(
        self, context_manager, game_state
    ):
        """Test that long reasoning is preserved in full without truncation."""
        # Create reasoning > 500 characters
        long_reasoning = "A" * 600

        game_state.action_reasoning_history.append({
            "turn": 1,
            "reasoning": long_reasoning,
            "action": "go north",
            "timestamp": "2025-11-02T10:00:00"
        })

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Should preserve full reasoning without truncation
        assert "Turn 1:" in formatted
        assert long_reasoning in formatted  # Full reasoning preserved
        # Count the reasoning portion (not including "Reasoning: " prefix)
        reasoning_line = [line for line in formatted.split('\n') if 'Reasoning: ' in line][0]
        reasoning_text = reasoning_line.replace("Reasoning: ", "")
        assert len(reasoning_text) == 600  # Full length preserved

    def test_get_recent_reasoning_formatted_blank_lines_between_turns(
        self, context_manager, game_state
    ):
        """Test that blank lines separate turns for readability."""
        # Add two reasoning entries
        for i in range(1, 3):
            game_state.action_reasoning_history.append({
                "turn": i,
                "reasoning": f"Reasoning {i}",
                "action": f"action {i}",
                "timestamp": "2025-11-02T10:00:00"
            })
            game_state.action_history.append((f"action {i}", f"response {i}"))

        formatted = context_manager.get_recent_reasoning_formatted(num_turns=3)

        # Split into lines
        lines = formatted.split('\n')

        # Find turn separators (blank lines)
        # Format: Turn X:\nReasoning:\nAction:\nResponse:\n<blank>\nTurn Y:...
        blank_lines = [i for i, line in enumerate(lines) if line == ""]

        # Should have at least one blank line separating turns
        assert len(blank_lines) >= 1

    def test_duplicate_actions_match_most_recent_response(
        self, context_manager, game_state
    ):
        """Test that duplicate actions match the most recent response, not the first."""
        # Add same action multiple times with different responses
        game_state.action_history.extend([
            ("go north", "You enter a forest."),
            ("examine tree", "It's a pine tree."),
            ("go north", "You enter a clearing."),
            ("examine leaf", "It's green."),
            ("go north", "You enter a cave."),
        ])

        # Add reasoning for the LAST "go north" (the cave one)
        game_state.action_reasoning_history.append({
            "turn": 5,
            "reasoning": "Exploring the cave entrance",
            "action": "go north",
            "timestamp": "2025-01-01T00:05:00",
        })

        formatted = context_manager.get_recent_reasoning_formatted()

        # Should match the MOST RECENT "go north" response (the cave)
        assert "You enter a cave." in formatted, \
            f"Expected most recent 'go north' response, got: {formatted}"

        # Should NOT match earlier responses
        assert "You enter a forest." not in formatted
        assert "You enter a clearing." not in formatted

        # Verify the full formatting
        assert "Turn 5:" in formatted
        assert "Reasoning: Exploring the cave entrance" in formatted
        assert "Action: go north" in formatted


class TestFormattedPromptWithReasoning:
    """Test that formatted prompt includes reasoning section."""

    def test_formatted_prompt_includes_reasoning_section(
        self, context_manager, game_state
    ):
        """Test that formatted prompt includes reasoning section when history exists."""
        # Add reasoning history
        game_state.action_reasoning_history.append({
            "turn": 1,
            "reasoning": "I should explore north.",
            "action": "go north",
            "timestamp": "2025-11-02T10:00:00"
        })
        game_state.action_history.append(("go north", "You go north."))

        # Get context and format it
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=[],
            location="Test Room",
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Verify reasoning section is present
        assert "## Previous Reasoning and Actions" in formatted
        assert "Turn 1:" in formatted
        assert "I should explore north." in formatted

    def test_formatted_prompt_without_reasoning_history(
        self, context_manager, game_state
    ):
        """Test that formatted prompt works without reasoning history."""
        # Clear reasoning history
        game_state.action_reasoning_history.clear()

        # Get context and format it
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=[],
            location="Test Room",
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Verify reasoning section is NOT present
        assert "## Previous Reasoning and Actions" not in formatted

        # But other sections should still be present
        assert "CURRENT LOCATION:" in formatted

    def test_formatted_prompt_reasoning_section_placement(
        self, context_manager, game_state
    ):
        """Test that reasoning section appears after game state and before objectives."""
        # Add reasoning history
        game_state.action_reasoning_history.append({
            "turn": 1,
            "reasoning": "Test reasoning",
            "action": "test action",
            "timestamp": "2025-11-02T10:00:00"
        })
        game_state.action_history.append(("test action", "test response"))

        # Add objectives
        game_state.discovered_objectives.append("Find treasure")

        # Get context and format it
        context = context_manager.get_agent_context(
            current_state="You are in a room.",
            inventory=[],
            location="Test Room",
            discovered_objectives=["Find treasure"]
        )

        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Find positions of key sections
        score_pos = formatted.find("SCORE:")
        reasoning_pos = formatted.find("## Previous Reasoning and Actions")
        objectives_pos = formatted.find("CURRENT OBJECTIVES:")

        # Verify ordering: score < reasoning < objectives
        assert score_pos < reasoning_pos < objectives_pos


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
