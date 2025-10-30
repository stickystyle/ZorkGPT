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
        map_update_interval=50,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
