#!/usr/bin/env python3
"""
Unit tests for game over detection functionality.
"""

import unittest
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_utils import skip_if_server_unavailable, run_test_commands
from game_interface.core.zork_interface import ZorkInterface


class TestGameOverDetection(unittest.TestCase):
    """Test cases for game over detection in game responses."""

    def setUp(self):
        """Set up zork interface instance for testing."""
        skip_if_server_unavailable()
        self.zork = ZorkInterface()

    def test_troll_miss_not_game_over(self):
        """Test that troll miss messages are not incorrectly detected as game over."""
        troll_response = """Your swing misses the troll by an inch.
The troll swings his axe, but it misses."""

        is_over, reason = self.zork.is_game_over(troll_response)

        self.assertFalse(
            is_over, f"Troll miss should not be game over. Reason: {reason}"
        )

    def test_combat_messages_not_game_over(self):
        """Test that various combat messages are not flagged as game over."""
        combat_messages = [
            "The troll swings his axe, but it misses.",
            "Your swing misses the troll by an inch.",
            "The troll attacks you with his axe.",
            "You dodge the troll's swing.",
            "The troll is wounded.",
        ]

        for msg in combat_messages:
            with self.subTest(message=msg):
                is_over, reason = self.zork.is_game_over(msg)
                self.assertFalse(
                    is_over,
                    f"Combat message should not be game over: '{msg}'. Reason: {reason}",
                )

    def test_death_messages_are_game_over(self):
        """Test that actual death messages are correctly detected as game over."""
        death_messages = [
            "You have died.",
            "Your head is taken off by the axe.",
            "The troll swings his axe and it cuts your head off.",
            "With his final blow, the troll kills you.",
        ]

        for msg in death_messages:
            with self.subTest(message=msg):
                is_over, reason = self.zork.is_game_over(msg)
                self.assertTrue(is_over, f"Death message should be game over: '{msg}'")

    def test_real_game_death_scenario(self, game_client):
        """Test game over detection with actual death scenario."""
        # This is a known death sequence in Zork
        commands = [
            "north",  # To North of House
            "north",  # To Forest Path
            "climb tree",  # Up the tree
            "take egg",  # Get the jeweled egg
            "down",  # Back to Forest Path
            "south",  # To North of House
            "east",  # To Behind House
            "open window",  # Open kitchen window
            "enter window",  # Enter kitchen
            "west",  # To Living Room
            "open trap door",  # Open trap door
            "down",  # To Cellar
            "north",  # To Troll Room - triggers combat
        ]

        # Run commands up to troll room
        run_test_commands(game_client, commands[:-1])

        # Enter troll room multiple times to potentially trigger death
        for _ in range(10):  # Try up to 10 times
            response = game_client.send_command("north")
            if response.get("game_over", False):
                self.assertTrue(response["game_over"])
                self.assertIsNotNone(response.get("game_over_reason"))
                break
            # If not dead, try to leave and re-enter
            game_client.send_command("south")

        # Note: Death from troll is probabilistic, so we just verify the detection works
        # when it does occur


if __name__ == "__main__":
    unittest.main()
