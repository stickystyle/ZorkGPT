# ABOUTME: Tests for critic object tree validation with multi-object commands
# ABOUTME: Validates that comma-separated "take X, Y, Z" commands are properly handled

import pytest
from zork_critic import ZorkCritic, ValidationResult
from game_interface.core.jericho_interface import JerichoInterface


class TestCriticMultiObjectValidation:
    """Test critic validation of multi-object take commands."""

    @pytest.fixture
    def jericho_interface(self):
        """Create a JerichoInterface for testing."""
        interface = JerichoInterface(game_file_path="infrastructure/zork.z5")
        interface.start()
        yield interface
        interface.close()

    @pytest.fixture
    def critic(self):
        """Create a ZorkCritic for testing."""
        return ZorkCritic()

    def test_single_object_take_valid(self, critic, jericho_interface):
        """Test validation of single object take command (valid object)."""
        # Start at West of House - mailbox is here, open it to access leaflet
        jericho_interface.send_command("open mailbox")

        result = critic._validate_against_object_tree(
            "take leaflet",
            jericho_interface
        )

        # Should be valid (leaflet is visible in open mailbox)
        assert result.valid, f"Expected valid but got: {result.reason}"

    def test_single_object_take_invalid(self, critic, jericho_interface):
        """Test validation of single object take command (invalid object)."""
        # Start at West of House
        jericho_interface.send_command("west")

        result = critic._validate_against_object_tree(
            "take dragon",  # Dragon doesn't exist
            jericho_interface
        )

        # Should be invalid
        assert not result.valid
        assert "not visible" in result.reason.lower()

    def test_multi_object_take_all_valid(self, critic, jericho_interface):
        """Test validation of multi-object take command (all objects valid)."""
        # Navigate to Kitchen using walkthrough sequence
        jericho_interface.send_command("north")
        jericho_interface.send_command("north")
        jericho_interface.send_command("up")
        jericho_interface.send_command("get egg")
        jericho_interface.send_command("down")
        jericho_interface.send_command("south")
        jericho_interface.send_command("east")
        jericho_interface.send_command("open window")
        jericho_interface.send_command("west")
        jericho_interface.send_command("open sack")

        # Now lunch, garlic, and bottle should be visible and takeable
        result = critic._validate_against_object_tree(
            "take lunch, garlic, bottle",
            jericho_interface
        )

        # Should be valid (all objects exist and are takeable)
        assert result.valid, f"Expected valid but got: {result.reason}"

    def test_multi_object_take_one_invalid(self, critic, jericho_interface):
        """Test validation of multi-object take command (one object invalid)."""
        # Navigate to Kitchen using walkthrough sequence
        jericho_interface.send_command("north")
        jericho_interface.send_command("north")
        jericho_interface.send_command("up")
        jericho_interface.send_command("get egg")
        jericho_interface.send_command("down")
        jericho_interface.send_command("south")
        jericho_interface.send_command("east")
        jericho_interface.send_command("open window")
        jericho_interface.send_command("west")
        jericho_interface.send_command("open sack")

        # Try to take valid objects plus an invalid one
        result = critic._validate_against_object_tree(
            "take lunch, garlic, dragon",  # Dragon doesn't exist
            jericho_interface
        )

        # Should be invalid (dragon doesn't exist)
        assert not result.valid
        assert "dragon" in result.reason.lower()

    def test_multi_object_take_not_takeable(self, critic, jericho_interface):
        """Test validation of multi-object take command (object tree validation is lenient)."""
        # Navigate to Kitchen where there's a table (not takeable)
        jericho_interface.send_command("north")
        jericho_interface.send_command("north")
        jericho_interface.send_command("up")
        jericho_interface.send_command("get egg")
        jericho_interface.send_command("down")
        jericho_interface.send_command("south")
        jericho_interface.send_command("east")
        jericho_interface.send_command("open window")
        jericho_interface.send_command("west")
        jericho_interface.send_command("open sack")

        # Try to take takeable objects plus the table
        result = critic._validate_against_object_tree(
            "take lunch, table",  # Table is visible but not actually takeable
            jericho_interface
        )

        # Object tree validation only checks visibility, not takeability
        # (because Jericho's takeable attribute is unreliable - e.g., lunch has
        # takeable=False but can actually be taken). The game itself will reject
        # trying to take the table, so validation passes it through to the LLM.
        assert result.valid, "Object tree validation should only check visibility, not takeability"

    def test_multi_object_with_spaces(self, critic, jericho_interface):
        """Test validation of multi-object command with extra spaces."""
        # Navigate to Kitchen
        jericho_interface.send_command("north")
        jericho_interface.send_command("north")
        jericho_interface.send_command("up")
        jericho_interface.send_command("get egg")
        jericho_interface.send_command("down")
        jericho_interface.send_command("south")
        jericho_interface.send_command("east")
        jericho_interface.send_command("open window")
        jericho_interface.send_command("west")
        jericho_interface.send_command("open sack")

        # Try with extra spaces around commas
        result = critic._validate_against_object_tree(
            "take lunch , garlic , bottle",  # Extra spaces
            jericho_interface
        )

        # Should be valid (spaces should be stripped)
        assert result.valid, f"Expected valid but got: {result.reason}"
