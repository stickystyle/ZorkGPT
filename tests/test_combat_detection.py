#!/usr/bin/env python3
"""
Unit tests for combat detection functionality.
"""

import unittest
from main import ZorkAgent


class TestCombatDetection(unittest.TestCase):
    """Test cases for combat detection in ZorkAgent."""

    def setUp(self):
        """Set up ZorkAgent instance for testing."""
        self.agent = ZorkAgent()

    def test_combat_scenario_detection(self):
        """Test that the extractor correctly identifies a combat situation."""
        combat_text = """You are in a small room with passages off in all directions.
Bloodstains and deep scratches (perhaps made by an axe) mar the walls.
A nasty-looking troll, brandishing a bloody axe, blocks all passages
out of the room.
The troll's swing almost knocks you over as you barely parry in time.
Your sword has begun to glow very brightly."""

        extraction = self.agent.get_extracted_info(combat_text)

        self.assertIsNotNone(extraction, "Combat extraction should not be None")
        self.assertTrue(extraction.in_combat, "Should detect combat situation")
        self.assertIsNotNone(extraction.visible_characters, "Should detect characters")
        # Check if troll is mentioned in visible characters
        characters_str = str(extraction.visible_characters).lower()
        self.assertIn("troll", characters_str, "Should detect troll character")

    def test_peaceful_scenario_detection(self):
        """Test that the extractor correctly identifies a peaceful situation."""
        peaceful_text = """You are in an open field west of a big white house with a boarded
front door.
There is a small mailbox here."""

        extraction = self.agent.get_extracted_info(peaceful_text)

        self.assertIsNotNone(extraction, "Peaceful extraction should not be None")
        self.assertFalse(
            extraction.in_combat, "Should not detect combat in peaceful scenario"
        )

    def test_combat_message_variations(self):
        """Test that various combat messages are correctly identified."""
        combat_variations = [
            ("The troll swings his axe, but it misses.", True),
            ("You attack the troll with your sword.", True),
            ("The creature blocks your path, snarling.", True),
            ("A dragon appears and breathes fire at you!", True),
        ]

        for message, expected_combat in combat_variations:
            with self.subTest(message=message):
                extraction = self.agent.get_extracted_info(message)
                self.assertIsNotNone(
                    extraction, f"Extraction should not be None for: {message}"
                )
                self.assertEqual(
                    extraction.in_combat,
                    expected_combat,
                    f"Combat detection mismatch for: {message}",
                )

    def test_non_combat_variations(self):
        """Test that non-combat messages are correctly identified."""
        non_combat_variations = [
            "You pick up the lamp.",
            "The door is locked.",
            "A beautiful garden stretches before you.",
            "You examine the mailbox.",
        ]

        for message in non_combat_variations:
            with self.subTest(message=message):
                extraction = self.agent.get_extracted_info(message)
                self.assertIsNotNone(
                    extraction, f"Extraction should not be None for: {message}"
                )
                self.assertFalse(
                    extraction.in_combat, f"Should not detect combat for: {message}"
                )


if __name__ == "__main__":
    unittest.main()
