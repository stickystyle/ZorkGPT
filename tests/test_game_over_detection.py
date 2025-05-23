#!/usr/bin/env python3
"""
Unit tests for game over detection functionality.
"""

import unittest
from zork_api import ZorkInterface


class TestGameOverDetection(unittest.TestCase):
    """Test cases for game over detection in ZorkInterface."""

    def setUp(self):
        """Set up ZorkInterface instance for testing."""
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


if __name__ == "__main__":
    unittest.main()
