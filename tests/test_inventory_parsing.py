import unittest
import sys
import os
import pytest

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.test_utils import game_client, skip_if_server_unavailable, run_test_commands
from hybrid_zork_extractor import HybridZorkExtractor


class TestInventoryParsing(unittest.TestCase):
    """Test inventory parsing with real Zork game output."""

    def setUp(self):
        """Set up test fixtures."""
        skip_if_server_unavailable()
        self.extractor = HybridZorkExtractor()

    def test_empty_handed_with_status_line(self):
        """Test parsing empty inventory with game status line."""
        raw_text = (
            "> West of House                                    Score: 0        Moves: 1\n\n"
            "You are empty-handed."
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        self.assertEqual(result, [], "Should return empty list for empty-handed inventory")

    def test_single_item_with_status_line(self):
        """Test parsing inventory with one item and status line."""
        raw_text = (
            "> West of House                                    Score: 0        Moves: 7\n\n"
            "You are carrying:\n"
            "  A leaflet"
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        self.assertEqual(result, ["A leaflet"], "Should return list with single item")

    def test_multiple_items_with_status_line(self):
        """Test parsing inventory with multiple items and status line."""
        raw_text = (
            "> Forest Path                                      Score: 5        Moves: 15\n\n"
            "You are carrying:\n"
            "  A leaflet\n"
            "  A sword\n"
            "  A lantern"
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        expected = ["A leaflet", "A sword", "A lantern"]
        self.assertEqual(result, expected, "Should return list with all items")

    def test_inventory_with_atmospheric_text(self):
        """Test parsing inventory with additional atmospheric game text."""
        raw_text = (
            "> Forest Path                                      Score: 0        Moves: 16\n\n"
            "You are carrying:\n"
            "  A leaflet\n"
            "You hear in the distance the chirping of a song bird."
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        # The atmospheric text should be included as it's not a status line
        expected = ["A leaflet", "You hear in the distance the chirping of a song bird"]
        self.assertEqual(result, expected, "Should include atmospheric text but filter status line")

    def test_container_items_with_status_line(self):
        """Test parsing inventory with items in containers and status line."""
        raw_text = (
            "> Kitchen                                          Score: 10       Moves: 25\n\n"
            "You are carrying:\n"
            "  A brown sack\n"
            "The brown sack contains:\n"
            "  A lunch\n"
            "  A clove of garlic"
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        # Should handle container relationships
        expected = ["A brown sack: Containing A lunch"]
        self.assertEqual(result, expected, "Should parse container relationships correctly")

    def test_status_line_variations(self):
        """Test different status line formats are properly filtered."""
        # Test different location name lengths and score formats
        raw_text1 = (
            "> North of House                                   Score: 0        Moves: 9\n\n"
            "You are carrying:\n"
            "  A leaflet"
        )
        result1 = self.zork._parse_inventory(raw_text1)
        self.assertEqual(result1, ["A leaflet"], "Should handle different location names")

        raw_text2 = (
            "> Clearing                                         Score: 15       Moves: 123\n\n"
            "You are empty-handed."
        )
        result2 = self.zork._parse_inventory(raw_text2)
        self.assertEqual(result2, [], "Should handle different score/move values")

    def test_no_status_line(self):
        """Test parsing inventory without status line (edge case)."""
        raw_text = (
            "You are carrying:\n"
            "  A leaflet\n"
            "  A sword"
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        expected = ["A leaflet", "A sword"]
        self.assertEqual(result, expected, "Should work without status line")

    def test_items_with_periods(self):
        """Test parsing items that end with periods."""
        raw_text = (
            "> West of House                                    Score: 0        Moves: 7\n\n"
            "You are carrying:\n"
            "  A leaflet.\n"
            "  A brass lantern."
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        expected = ["A leaflet", "A brass lantern"]
        self.assertEqual(result, expected, "Should remove trailing periods from items")

    def test_empty_inventory_alternative_format(self):
        """Test alternative empty inventory formats."""
        # Test lowercase variation
        raw_text = (
            "> West of House                                    Score: 0        Moves: 1\n\n"
            "you are empty handed."
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        self.assertEqual(result, [], "Should handle lowercase 'empty handed'")

    def test_status_line_filtering_robustness(self):
        """Test that status line filtering doesn't accidentally filter valid content."""
        # Make sure we don't filter lines that happen to contain "Score:" or "Moves:" but aren't status lines
        raw_text = (
            "> Library                                          Score: 20       Moves: 50\n\n"
            "You are carrying:\n"
            "  A book titled 'High Score: Gaming Adventures'\n"
            "  A manual about 'Chess Moves: Advanced Tactics'"
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        expected = [
            "A book titled 'High Score: Gaming Adventures'",
            "A manual about 'Chess Moves: Advanced Tactics'"
        ]
        self.assertEqual(result, expected, "Should not filter valid items containing Score: or Moves:")

    def test_complex_container_scenario(self):
        """Test complex scenario with multiple containers and status line."""
        raw_text = (
            "> Kitchen                                          Score: 25       Moves: 75\n\n"
            "You are carrying:\n"
            "  A brown sack\n"
            "The brown sack contains:\n"
            "  A lunch\n"
            "  A bottle\n"
            "The bottle contains:\n"
            "  Water\n"
            "  A small key"
        )
        parsed = self.extractor.extract(raw_text)
        result = parsed.inventory
        expected = [
            "A brown sack: Containing A lunch", 
            "A bottle: Containing Water"
        ]
        self.assertEqual(result, expected, "Should handle complex container relationships")


    def test_real_game_inventory_sequence(self, game_client):
        """Test inventory parsing with actual game commands."""
        # Execute test sequence
        commands = [
            "south",
            "east", 
            "open window",
            "enter window",
            "take sack",
            "take garlic",
            "inventory"
        ]
        
        responses = run_test_commands(game_client, commands)
        
        # Parse the final inventory response
        inventory_response = responses[-1]['raw_response']
        parsed = self.extractor.extract(inventory_response)
        
        # Should have both sack and garlic
        self.assertIn("brown sack", [item.lower() for item in parsed.inventory])
        self.assertIn("clove of garlic", [item.lower() for item in parsed.inventory])
        
        # Check that we got score increase from taking items
        self.assertGreater(parsed.score, 0)


if __name__ == '__main__':
    unittest.main() 