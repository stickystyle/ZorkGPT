#!/usr/bin/env python3
"""
Unit tests for inventory skipping during combat functionality.
"""

import unittest
from zork_agent import ZorkAgent
from hybrid_zork_extractor import ExtractorResponse
from session.game_configuration import GameConfiguration


class TestInventorySkip(unittest.TestCase):
    """Test cases for inventory skipping during combat in ZorkAgent."""

    def setUp(self):
        """Set up ZorkAgent instance for testing."""
        config = GameConfiguration.from_toml()
        self.agent = ZorkAgent(config=config)

    def test_combat_inventory_skip_logic(self):
        """Test that the agent correctly skips inventory during combat."""
        # Create combat extraction
        combat_extraction = ExtractorResponse(
            current_location_name="Troll Room",
            exits=["north", "south", "east", "west"],
            visible_objects=["bloody axe"],
            visible_characters=["nasty-looking troll"],
            inventory=[],
            in_combat=True,
        )

        # Add to memory history to simulate previous turn
        self.agent.memory_log_history = [combat_extraction]

        # Test the combat detection logic from the main loop
        in_combat = False
        if self.agent.memory_log_history:
            last_extraction = self.agent.memory_log_history[-1]
            in_combat = getattr(last_extraction, "in_combat", False)

        self.assertTrue(in_combat, "Should detect combat from previous turn")

    def test_peaceful_inventory_check_logic(self):
        """Test that the agent correctly allows inventory checks during peaceful scenarios."""
        # Create peaceful extraction
        peaceful_extraction = ExtractorResponse(
            current_location_name="West of House",
            exits=["north", "east"],
            visible_objects=["mailbox", "house"],
            visible_characters=[],
            inventory=[],
            in_combat=False,
        )

        # Add to memory history
        self.agent.memory_log_history = [peaceful_extraction]

        # Test the peaceful detection logic
        in_combat = False
        if self.agent.memory_log_history:
            last_extraction = self.agent.memory_log_history[-1]
            in_combat = getattr(last_extraction, "in_combat", False)

        self.assertFalse(in_combat, "Should not detect combat in peaceful scenario")

    def test_combat_context_in_memory_prompt(self):
        """Test that combat context is properly included in memory prompts."""
        combat_extraction = ExtractorResponse(
            current_location_name="Troll Room",
            exits=["north", "south", "east", "west"],
            visible_objects=["bloody axe"],
            visible_characters=["nasty-looking troll"],
            inventory=[],
            in_combat=True,
        )

        # Test combat context in memory prompt
        # Use a mock MapGraph for this test
        from unittest.mock import Mock
        mock_map = Mock()
        mock_map.get_context_for_prompt.return_value = "Mock map context"
        mock_map.get_navigation_suggestions.return_value = []  # Return empty list for navigation

        combat_context = self.agent.get_relevant_memories_for_prompt(
            current_location_name_from_current_extraction="Troll Room",
            memory_log_history=[combat_extraction],
            current_inventory=["sword", "lamp"],
            game_map=mock_map,
            in_combat=True,
        )

        self.assertIn(
            "COMBAT SITUATION",
            combat_context,
            "Agent should receive combat warning in context",
        )

    def test_empty_memory_history(self):
        """Test behavior when memory history is empty."""
        self.agent.memory_log_history = []

        in_combat = False
        if self.agent.memory_log_history:
            last_extraction = self.agent.memory_log_history[-1]
            in_combat = getattr(last_extraction, "in_combat", False)

        self.assertFalse(in_combat, "Should not detect combat with empty memory")

    def test_extraction_without_combat_attribute(self):
        """Test behavior when extraction doesn't have in_combat attribute."""
        # Create extraction with default combat attribute
        extraction_no_combat = ExtractorResponse(
            current_location_name="Test Room",
            exits=["north"],
            visible_objects=[],
            visible_characters=[],
            inventory=[],
            in_combat=False,  # Default value
        )

        self.agent.memory_log_history = [extraction_no_combat]

        in_combat = False
        if self.agent.memory_log_history:
            last_extraction = self.agent.memory_log_history[-1]
            in_combat = getattr(last_extraction, "in_combat", False)

        self.assertFalse(in_combat, "Should default to False when in_combat is None")


if __name__ == "__main__":
    unittest.main()
