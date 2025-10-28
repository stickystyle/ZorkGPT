# ABOUTME: Comprehensive production test suite for JerichoInterface
# ABOUTME: Tests inventory, location, state management, and edge cases

import pytest
from game_interface.core.jericho_interface import JerichoInterface


# Constants
GAME_FILE_PATH = "/Volumes/workingfolder/ZorkGPT/infrastructure/zork.z5"


# Test fixture for game interface
@pytest.fixture
def game_interface():
    """Create a fresh game interface for each test."""
    with JerichoInterface(GAME_FILE_PATH) as interface:
        interface.start()
        yield interface
    # close() called automatically by context manager


class TestInventory:
    """Test inventory extraction functionality."""

    def test_empty_inventory(self, game_interface):
        """Test that inventory is empty at game start."""
        inventory = game_interface.get_inventory_structured()
        assert inventory == [], "Inventory should be empty at start"

        inventory_text = game_interface.get_inventory_text()
        assert inventory_text == [], "Text inventory should be empty at start"

    def test_single_item_inventory(self, game_interface):
        """Test inventory after picking up leaflet."""
        # Open mailbox and take leaflet
        game_interface.send_command("open mailbox")
        game_interface.send_command("take leaflet")

        # Check structured inventory
        inventory = game_interface.get_inventory_structured()
        assert len(inventory) == 1, "Should have exactly one item"
        assert hasattr(inventory[0], 'num'), "Should have ZObject properties"
        assert "leaflet" in inventory[0].name.lower(), "Item should be leaflet"

        # Check text inventory
        inventory_text = game_interface.get_inventory_text()
        assert len(inventory_text) == 1, "Text inventory should have one item"
        assert "leaflet" in inventory_text[0].lower(), "Text item should be leaflet"

    def test_multiple_items(self, game_interface):
        """Test inventory with multiple items."""
        # Get multiple items
        game_interface.send_command("open mailbox")
        game_interface.send_command("take all from mailbox")

        inventory = game_interface.get_inventory_structured()
        assert len(inventory) > 0, "Should have at least one item"

        # Verify all have ZObject properties
        for item in inventory:
            assert hasattr(item, 'num'), "Should have num property"
            assert item.num > 0, "Should have valid object ID"
            assert item.name, "Should have a name"

    def test_item_in_container(self, game_interface):
        """Test handling of items in containers."""
        # Take the mailbox itself (if possible) or another container
        game_interface.send_command("open mailbox")
        game_interface.send_command("take leaflet")

        # Get structured inventory
        inventory = game_interface.get_inventory_structured()
        assert len(inventory) > 0, "Should have items"

        # Verify structure
        for item in inventory:
            assert hasattr(item, 'child'), "Should have child attribute"
            assert hasattr(item, 'parent'), "Should have parent attribute"


class TestLocation:
    """Test location extraction functionality."""

    def test_starting_location(self, game_interface):
        """Test that starting location is 'West of House'."""
        location = game_interface.get_location_structured()
        assert location is not None, "Location should not be None"
        assert hasattr(location, 'num'), "Should have ZObject properties"
        # Jericho shortens "West of House" to "West House"
        assert "west" in location.name.lower() and "house" in location.name.lower(), \
            f"Should start at West House, got: {location.name}"

        location_text = game_interface.get_location_text()
        assert "west" in location_text.lower() and "house" in location_text.lower(), \
            "Text location should contain 'west' and 'house'"

    def test_movement(self, game_interface):
        """Test that location changes after movement."""
        # Get initial location
        initial_location = game_interface.get_location_text()

        # Move south
        game_interface.send_command("go south")
        new_location = game_interface.get_location_text()

        assert new_location != initial_location, "Location should change after movement"
        # Jericho shortens "South of House" to "South House"
        assert "south" in new_location.lower() and "house" in new_location.lower(), \
            f"Should be at South House, got: {new_location}"

    def test_location_object_properties(self, game_interface):
        """Test that location ZObject has expected properties."""
        location = game_interface.get_location_structured()

        assert location.num > 0, "Should have valid object ID"
        assert location.name, "Should have a name"
        assert hasattr(location, 'attr'), "Should have attributes"
        assert hasattr(location, 'parent'), "Should have parent"
        assert hasattr(location, 'child'), "Should have child"
        assert hasattr(location, 'sibling'), "Should have sibling"


class TestStateManagement:
    """Test save/restore functionality."""

    def test_save_restore(self, game_interface):
        """Test save and restore functionality."""
        # Execute some commands
        game_interface.send_command("open mailbox")
        game_interface.send_command("take leaflet")

        # Save state
        saved_state = game_interface.save_state()
        assert saved_state is not None, "Should return a state object"

        # Record current state
        saved_location = game_interface.get_location_text()
        saved_inventory = game_interface.get_inventory_text()
        saved_score = game_interface.get_score()

        # Make changes
        game_interface.send_command("go south")
        changed_location = game_interface.get_location_text()
        assert changed_location != saved_location, "Location should have changed"

        # Restore
        game_interface.restore_state(saved_state)

        # Verify restoration
        restored_location = game_interface.get_location_text()
        restored_inventory = game_interface.get_inventory_text()
        restored_score = game_interface.get_score()

        assert restored_location == saved_location, "Location should be restored"
        assert restored_inventory == saved_inventory, "Inventory should be restored"
        assert restored_score == saved_score, "Score should be restored"

    def test_score_tracking(self, game_interface):
        """Test score extraction works."""
        score, max_score = game_interface.get_score()

        assert isinstance(score, int), "Score should be integer"
        assert isinstance(max_score, int), "Max score should be integer"
        assert score >= 0, "Score should be non-negative"
        assert max_score > 0, "Max score should be positive"
        assert score <= max_score, "Current score should not exceed max"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_death_handling(self, game_interface):
        """Test that interface handles death gracefully."""
        # Try to trigger death (going into dark area without lamp)
        game_interface.send_command("go south")
        game_interface.send_command("open window")
        game_interface.send_command("go west")  # Into darkness

        # Should still be able to get state (even if dead)
        try:
            location = game_interface.get_location_structured()
            inventory = game_interface.get_inventory_structured()
            # If we get here without exception, that's good
            assert True
        except Exception as e:
            pytest.fail(f"Should handle death gracefully, got: {e}")

    def test_invalid_command(self, game_interface):
        """Test that invalid commands don't break state extraction."""
        # Send invalid command
        response = game_interface.send_command("xyzzy")

        # Should still be able to extract state
        location = game_interface.get_location_text()
        inventory = game_interface.get_inventory_text()

        assert location, "Should still have location after invalid command"
        assert isinstance(inventory, list), "Should still have inventory list"


class TestStructuredVsText:
    """Test consistency between structured and text methods."""

    def test_inventory_structured_vs_text(self, game_interface):
        """Test that structured and text inventory are consistent."""
        # Get some items
        game_interface.send_command("open mailbox")
        game_interface.send_command("take all from mailbox")

        structured = game_interface.get_inventory_structured()
        text = game_interface.get_inventory_text()

        # Should have same number of items
        assert len(structured) == len(text), \
            "Structured and text inventory should have same count"

        # Each structured item should have corresponding text
        for i, obj in enumerate(structured):
            assert obj.name == text[i], \
                f"Item {i} should match: {obj.name} vs {text[i]}"

    def test_location_structured_vs_text(self, game_interface):
        """Test that structured and text location are consistent."""
        structured = game_interface.get_location_structured()
        text = game_interface.get_location_text()

        assert structured is not None, "Structured location should not be None"
        assert structured.name == text, \
            f"Location should match: {structured.name} vs {text}"


class TestObjectTree:
    """Test object tree functionality."""

    def test_get_all_objects(self, game_interface):
        """Test that get_all_objects returns valid objects."""
        all_objects = game_interface.get_all_objects()

        assert len(all_objects) > 0, "Should have objects"
        assert all(hasattr(obj, 'num') for obj in all_objects), \
            "All should have ZObject properties"

    def test_object_relationships(self, game_interface):
        """Test that object parent/child relationships make sense."""
        # Get an item
        game_interface.send_command("open mailbox")
        game_interface.send_command("take leaflet")

        # Get inventory
        inventory = game_interface.get_inventory_structured()
        assert len(inventory) > 0, "Should have items"

        # Get player object (which is itself a ZObject)
        player_obj = game_interface.get_player_object()
        player_id = player_obj.num

        # All inventory items should have player object ID as parent
        for item in inventory:
            assert item.parent == player_id, \
                f"Inventory item should have player ID {player_id} as parent, got {item.parent}"

    def test_visible_objects_exclude_player(self, game_interface):
        """Test that visible objects do not include the player object."""
        # Get player object
        player_obj = game_interface.get_player_object()
        player_id = player_obj.num
        player_name = player_obj.name.lower()

        # Get visible objects in starting location
        visible = game_interface.get_visible_objects_in_location()

        # Player object should not be in visible objects
        visible_ids = [obj.num for obj in visible]
        assert player_id not in visible_ids, \
            f"Player object (ID {player_id}) should not be in visible objects"

        # Player name should not appear in visible objects
        visible_names = [obj.name.lower() for obj in visible]
        assert player_name not in visible_names, \
            f"Player name '{player_name}' should not be in visible objects"

        # But mailbox should be visible
        assert any("mailbox" in name for name in visible_names), \
            "Mailbox should be visible at starting location"


class TestInitialization:
    """Test initialization and cleanup."""

    def test_interface_before_start(self):
        """Test that methods fail before start() is called."""
        interface = JerichoInterface(GAME_FILE_PATH)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError):
            interface.send_command("look")

        with pytest.raises(RuntimeError):
            interface.get_inventory_structured()

        interface.close()

    def test_cleanup(self, game_interface):
        """Test that cleanup works properly."""
        # Use the interface
        game_interface.send_command("look")

        # Close it
        game_interface.close()

        # Environment should be None
        assert game_interface.env is None, "Environment should be None after close"

    def test_invalid_game_file(self):
        """Test that initialization fails gracefully with invalid file."""
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            interface = JerichoInterface("/nonexistent/zork.z5")
            interface.start()

    def test_double_close(self, game_interface):
        """Test that calling close() twice doesn't fail."""
        game_interface.close()
        game_interface.close()  # Should not raise
        assert game_interface.env is None

    def test_command_after_close(self, game_interface):
        """Test that commands fail gracefully after close()."""
        game_interface.close()
        with pytest.raises(RuntimeError):
            game_interface.send_command("look")

    def test_empty_command(self, game_interface):
        """Test sending empty command."""
        response = game_interface.send_command("")
        assert isinstance(response, str)

    def test_context_manager(self):
        """Test that context manager protocol works."""
        with JerichoInterface(GAME_FILE_PATH) as interface:
            interface.start()
            response = interface.send_command("look")
            assert isinstance(response, str)
            assert interface.env is not None
        # After exiting context, should be closed
        assert interface.env is None


class TestCompatibilityAliases:
    """Test convenience method aliases for backward compatibility."""

    def test_score_alias(self, game_interface):
        """Test that score() alias works."""
        result1 = game_interface.score()
        result2 = game_interface.get_score()
        assert result1 == result2
        assert isinstance(result1, tuple)
        assert len(result1) == 2

    def test_inventory_alias(self, game_interface):
        """Test that inventory() alias works."""
        result1 = game_interface.inventory()
        result2 = game_interface.get_inventory_text()
        assert result1 == result2
        assert isinstance(result1, list)
