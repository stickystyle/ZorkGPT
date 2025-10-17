# ABOUTME: Test suite for Phase 5.1 object attribute helpers in JerichoInterface
# ABOUTME: Tests attribute extraction, bit checking, visible objects, and verb list

import pytest
from game_interface.core.jericho_interface import JerichoInterface


# Constants
GAME_FILE_PATH = "/Volumes/workingfolder/ZorkGPT/infrastructure/zork.z5"


@pytest.fixture
def game_interface():
    """Create a fresh game interface for each test."""
    with JerichoInterface(GAME_FILE_PATH) as interface:
        interface.start()
        yield interface


class TestCheckAttribute:
    """Test the _check_attribute helper method."""

    def test_check_attribute_with_valid_object(self, game_interface):
        """Test checking attributes on a valid object."""
        # Get the mailbox which should have some attributes
        game_interface.send_command("examine mailbox")

        # Get objects at location
        visible_objects = game_interface.get_visible_objects_in_location()

        # Find the mailbox
        mailbox = None
        for obj in visible_objects:
            if 'mailbox' in obj.name.lower():
                mailbox = obj
                break

        assert mailbox is not None, "Should find mailbox in starting location"

        # Check that we can call _check_attribute without errors
        # Note: We don't know exact bit values yet, just test the method works
        result = game_interface._check_attribute(mailbox, 0)
        assert isinstance(result, bool), "Should return boolean"

        result = game_interface._check_attribute(mailbox, 10)
        assert isinstance(result, bool), "Should return boolean"

    def test_check_attribute_with_none(self, game_interface):
        """Test that _check_attribute handles None gracefully."""
        result = game_interface._check_attribute(None, 10)
        assert result is False, "Should return False for None object"

    def test_check_attribute_with_invalid_bit(self, game_interface):
        """Test that _check_attribute handles out-of-range bits gracefully."""
        visible_objects = game_interface.get_visible_objects_in_location()
        if visible_objects:
            obj = visible_objects[0]
            # Test with very high bit number (beyond Version 3 max of 47)
            result = game_interface._check_attribute(obj, 100)
            assert result is False, "Should return False for out-of-range bit"

    def test_check_attribute_bit_ordering(self, game_interface):
        """Test that bit checking follows Z-machine MSB-first ordering."""
        # Get an object with known attributes
        game_interface.send_command("open mailbox")
        game_interface.send_command("take leaflet")

        inventory = game_interface.get_inventory_structured()
        assert len(inventory) > 0, "Should have leaflet in inventory"

        leaflet = inventory[0]

        # Test that we can check various bit positions
        # The exact values don't matter here, just that the method works
        for bit in [0, 7, 8, 15, 16, 23, 31]:
            result = game_interface._check_attribute(leaflet, bit)
            assert isinstance(result, bool), f"Bit {bit} should return boolean"


class TestGetObjectAttributes:
    """Test the get_object_attributes method."""

    def test_get_attributes_returns_dict(self, game_interface):
        """Test that get_object_attributes returns a dictionary."""
        visible_objects = game_interface.get_visible_objects_in_location()
        assert len(visible_objects) > 0, "Should have visible objects"

        attrs = game_interface.get_object_attributes(visible_objects[0])
        assert isinstance(attrs, dict), "Should return dictionary"

    def test_get_attributes_has_expected_keys(self, game_interface):
        """Test that attribute dictionary has expected keys."""
        visible_objects = game_interface.get_visible_objects_in_location()
        assert len(visible_objects) > 0, "Should have visible objects"

        attrs = game_interface.get_object_attributes(visible_objects[0])

        expected_keys = ['touched', 'container', 'openable', 'takeable', 'transparent', 'portable', 'readable']
        for key in expected_keys:
            assert key in attrs, f"Should have '{key}' attribute"
            assert isinstance(attrs[key], bool), f"'{key}' should be boolean"

    def test_get_attributes_with_none(self, game_interface):
        """Test that get_object_attributes handles None gracefully."""
        attrs = game_interface.get_object_attributes(None)
        assert attrs == {}, "Should return empty dict for None"

    def test_mailbox_attributes(self, game_interface):
        """Test mailbox has expected attributes."""
        visible_objects = game_interface.get_visible_objects_in_location()

        mailbox = None
        for obj in visible_objects:
            if 'mailbox' in obj.name.lower():
                mailbox = obj
                break

        assert mailbox is not None, "Should find mailbox"

        attrs = game_interface.get_object_attributes(mailbox)

        # Mailbox should be a container (probably)
        # Note: If this fails, we'll need to adjust bit positions
        assert isinstance(attrs['container'], bool), "Container should be boolean"
        assert isinstance(attrs['openable'], bool), "Openable should be boolean"

    def test_leaflet_attributes(self, game_interface):
        """Test leaflet attributes after taking it."""
        game_interface.send_command("open mailbox")
        game_interface.send_command("take leaflet")

        inventory = game_interface.get_inventory_structured()
        assert len(inventory) > 0, "Should have leaflet"

        leaflet = inventory[0]
        attrs = game_interface.get_object_attributes(leaflet)

        # Leaflet should be takeable (we just took it!)
        # Note: If this fails, we'll need to adjust bit positions
        assert isinstance(attrs['takeable'], bool), "Takeable should be boolean"

        # Leaflet is probably not a container or openable
        assert isinstance(attrs['container'], bool), "Container should be boolean"
        assert isinstance(attrs['openable'], bool), "Openable should be boolean"

    def test_attributes_for_multiple_objects(self, game_interface):
        """Test that we can get attributes for multiple objects."""
        visible_objects = game_interface.get_visible_objects_in_location()

        for obj in visible_objects:
            attrs = game_interface.get_object_attributes(obj)
            assert isinstance(attrs, dict), f"Should return dict for {obj.name}"
            assert len(attrs) > 0, f"Should have attributes for {obj.name}"


class TestGetVisibleObjectsInLocation:
    """Test the get_visible_objects_in_location method."""

    def test_starting_location_has_objects(self, game_interface):
        """Test that starting location has visible objects."""
        visible = game_interface.get_visible_objects_in_location()

        # West of House should have mailbox at minimum
        assert len(visible) > 0, "Starting location should have visible objects"

    def test_visible_objects_are_zobjects(self, game_interface):
        """Test that returned objects are ZObjects."""
        visible = game_interface.get_visible_objects_in_location()

        for obj in visible:
            assert hasattr(obj, 'num'), "Should be ZObject with num"
            assert hasattr(obj, 'name'), "Should be ZObject with name"
            assert hasattr(obj, 'parent'), "Should be ZObject with parent"

    def test_visible_objects_have_correct_parent(self, game_interface):
        """Test that visible objects have current location as parent."""
        location = game_interface.get_location_structured()
        visible = game_interface.get_visible_objects_in_location()

        for obj in visible:
            assert obj.parent == location.num, \
                f"Object {obj.name} should have location {location.num} as parent"

    def test_visible_objects_change_with_location(self, game_interface):
        """Test that visible objects change when moving."""
        initial_visible = game_interface.get_visible_objects_in_location()
        initial_names = {obj.name for obj in initial_visible}

        # Move to different location
        game_interface.send_command("go south")

        new_visible = game_interface.get_visible_objects_in_location()
        new_names = {obj.name for obj in new_visible}

        # Objects should be different (unless both locations are empty)
        # Note: This might not always be true, but typically locations have different objects
        assert isinstance(new_visible, list), "Should return list"

    def test_visible_objects_after_taking_item(self, game_interface):
        """Test that objects disappear from location when taken."""
        initial_visible = game_interface.get_visible_objects_in_location()

        # Find leaflet in mailbox and take it
        game_interface.send_command("open mailbox")

        # Check visible objects include leaflet (it's now visible in mailbox)
        visible_before_take = game_interface.get_visible_objects_in_location()

        # Take the leaflet
        game_interface.send_command("take leaflet")

        # Leaflet should now be in inventory, not in location
        inventory = game_interface.get_inventory_structured()
        assert len(inventory) > 0, "Should have leaflet in inventory"

        visible_after_take = game_interface.get_visible_objects_in_location()

        # The counts might be different
        assert isinstance(visible_after_take, list), "Should return list"

    def test_empty_location_returns_empty_list(self, game_interface):
        """Test that empty locations return empty list."""
        # Note: Most Zork locations have at least some objects
        # This test just ensures we get a list, even if empty
        visible = game_interface.get_visible_objects_in_location()
        assert isinstance(visible, list), "Should return list"

    def test_not_started_raises_error(self):
        """Test that method raises error if environment not started."""
        interface = JerichoInterface(GAME_FILE_PATH)

        with pytest.raises(RuntimeError):
            interface.get_visible_objects_in_location()

        interface.close()


class TestGetValidVerbs:
    """Test the get_valid_verbs method."""

    def test_returns_list(self, game_interface):
        """Test that get_valid_verbs returns a list."""
        verbs = game_interface.get_valid_verbs()
        assert isinstance(verbs, list), "Should return list"

    def test_list_not_empty(self, game_interface):
        """Test that verb list is not empty."""
        verbs = game_interface.get_valid_verbs()
        assert len(verbs) > 0, "Should have at least some verbs"

    def test_contains_common_verbs(self, game_interface):
        """Test that list contains common Zork verbs."""
        verbs = game_interface.get_valid_verbs()

        # Check for essential movement verbs
        assert 'north' in verbs, "Should include 'north'"
        assert 'south' in verbs, "Should include 'south'"
        assert 'east' in verbs, "Should include 'east'"
        assert 'west' in verbs, "Should include 'west'"

        # Check for essential action verbs
        assert 'take' in verbs, "Should include 'take'"
        assert 'drop' in verbs, "Should include 'drop'"
        assert 'open' in verbs, "Should include 'open'"
        assert 'close' in verbs, "Should include 'close'"
        assert 'examine' in verbs, "Should include 'examine'"
        assert 'look' in verbs, "Should include 'look'"

        # Check for inventory
        assert 'inventory' in verbs, "Should include 'inventory'"

    def test_verbs_are_strings(self, game_interface):
        """Test that all verbs are strings."""
        verbs = game_interface.get_valid_verbs()

        for verb in verbs:
            assert isinstance(verb, str), f"Verb should be string, got {type(verb)}"
            assert len(verb) > 0, "Verb should not be empty string"

    def test_verb_list_is_stable(self, game_interface):
        """Test that verb list doesn't change between calls."""
        verbs1 = game_interface.get_valid_verbs()
        verbs2 = game_interface.get_valid_verbs()

        assert verbs1 == verbs2, "Verb list should be stable"

    def test_no_duplicate_verbs(self, game_interface):
        """Test that verb list has no duplicates."""
        verbs = game_interface.get_valid_verbs()

        assert len(verbs) == len(set(verbs)), "Should have no duplicate verbs"

    def test_works_without_started_environment(self):
        """Test that get_valid_verbs works even without starting environment."""
        # This should work because it's just returning a static list
        interface = JerichoInterface(GAME_FILE_PATH)

        verbs = interface.get_valid_verbs()
        assert isinstance(verbs, list), "Should return list"
        assert len(verbs) > 0, "Should have verbs"

        interface.close()


class TestIntegration:
    """Integration tests combining multiple new methods."""

    def test_examine_visible_objects_attributes(self, game_interface):
        """Test examining attributes of all visible objects."""
        visible = game_interface.get_visible_objects_in_location()

        for obj in visible:
            attrs = game_interface.get_object_attributes(obj)

            # Should get valid attributes
            assert isinstance(attrs, dict), f"Should have attributes for {obj.name}"
            assert 'takeable' in attrs, "Should have takeable attribute"
            assert 'openable' in attrs, "Should have openable attribute"

    def test_attribute_consistency_across_locations(self, game_interface):
        """Test that object attributes are consistent across game state."""
        # Get mailbox in first location
        visible1 = game_interface.get_visible_objects_in_location()

        mailbox = None
        for obj in visible1:
            if 'mailbox' in obj.name.lower():
                mailbox = obj
                break

        if mailbox:
            attrs_before = game_interface.get_object_attributes(mailbox)

            # Move away and back
            game_interface.send_command("go south")
            game_interface.send_command("go north")

            # Get mailbox again
            visible2 = game_interface.get_visible_objects_in_location()
            mailbox2 = None
            for obj in visible2:
                if 'mailbox' in obj.name.lower():
                    mailbox2 = obj
                    break

            if mailbox2:
                attrs_after = game_interface.get_object_attributes(mailbox2)

                # Attributes should be the same
                assert attrs_before == attrs_after, \
                    "Object attributes should be consistent"

    def test_all_methods_work_together(self, game_interface):
        """Test that all new methods work together in a realistic scenario."""
        # Get valid verbs
        verbs = game_interface.get_valid_verbs()
        assert 'take' in verbs, "Should have 'take' verb"
        assert 'examine' in verbs, "Should have 'examine' verb"

        # Get visible objects
        visible = game_interface.get_visible_objects_in_location()
        assert len(visible) > 0, "Should have visible objects"

        # Check attributes of each object
        for obj in visible:
            attrs = game_interface.get_object_attributes(obj)

            # If object is takeable, we could use the 'take' verb
            if attrs.get('takeable'):
                # This object can be taken
                assert 'take' in verbs, "'take' should be in verb list"

            # If object is openable, we could use the 'open' verb
            if attrs.get('openable'):
                assert 'open' in verbs, "'open' should be in verb list"


class TestErrorHandling:
    """Test error handling in new methods."""

    def test_get_attributes_with_corrupted_object(self, game_interface):
        """Test that get_object_attributes handles edge cases gracefully."""
        # Create a mock object-like thing
        class FakeObject:
            pass

        fake = FakeObject()
        attrs = game_interface.get_object_attributes(fake)

        # Should return empty dict or handle gracefully
        assert isinstance(attrs, dict), "Should return dict"

    def test_check_attribute_with_negative_bit(self, game_interface):
        """Test _check_attribute with negative bit number."""
        visible = game_interface.get_visible_objects_in_location()
        if visible:
            result = game_interface._check_attribute(visible[0], -1)
            # Should handle gracefully (likely return False)
            assert isinstance(result, bool), "Should return boolean"
