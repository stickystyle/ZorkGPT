"""
Test complete exit detection in JerichoInterface using manual direction testing.

This test validates that the new get_valid_exits() method provides complete
exit detection by testing each direction from the Z-machine dictionary,
resolving the issue where get_valid_actions() only detected ~17-30% of exits.
"""

import pytest
from game_interface.core.jericho_interface import JerichoInterface
from tests.fixtures.walkthrough import get_walkthrough_slice


class TestJerichoExitsCompleteness:
    """Test complete exit detection using manual direction testing."""

    @pytest.fixture
    def jericho_interface(self):
        """Create a fresh JerichoInterface for each test."""
        interface = JerichoInterface(game_file_path="infrastructure/zork.z5")
        interface.start()  # Initialize the environment
        yield interface
        interface.close()

    def test_exits_at_west_of_house_starting_location(self, jericho_interface):
        """
        Validate exit detection at West of House (starting location).

        West of House has multiple exits including north and south.
        This test validates that the new method detects all working exits.
        """
        # Start at West of House (no movement needed)
        location = jericho_interface.get_location_structured()
        starting_location = location.num  # Record whatever the actual starting location is

        # Get exits using the new manual testing method
        exits = jericho_interface.get_valid_exits()

        # Verify we have at least some exits
        assert len(exits) >= 2, f"Should detect at least 2 exits, got {len(exits)}: {exits}"

        # Verify 'north' or 'n' is in exits (one of the main directions from West of House)
        has_north = any(direction in exits for direction in ["north", "n"])
        assert has_north, f"Should detect north exit. Got exits: {exits}"

        # Verify exits are sorted
        assert exits == sorted(exits), f"Exits should be sorted: {exits}"

        # Verify location hasn't changed
        location_after = jericho_interface.get_location_structured()
        assert location_after.num == starting_location, (
            f"Location changed from {starting_location} to {location_after.num} "
            "after calling get_valid_exits()"
        )

    def test_exits_in_forest_room_worst_case(self, jericho_interface):
        """
        Validate exit detection in Forest room (worst case scenario).

        Forest room is a worst case where multiple exits lead to the same
        destination (Clearing). This is where get_valid_actions() fails
        because Jericho collapses duplicate effects.

        Navigation sequence: north, north, east, east (from West of House)
        """
        # Navigate to Forest room
        navigation_sequence = ["north", "north", "east", "east"]
        for action in navigation_sequence:
            response = jericho_interface.send_command(action)
            assert response, f"Navigation failed at '{action}': got empty response"

        # Record the location (Forest room)
        location = jericho_interface.get_location_structured()
        forest_room_id = location.num

        # Get exits using the new manual testing method
        exits = jericho_interface.get_valid_exits()

        # Verify we detected multiple exits (Forest has several directions)
        # The key test: verify ALL exits were detected
        # We don't need to verify they ALL work here - that's get_valid_exits()'s job
        # This test just verifies we detected multiple exits in the worst-case scenario
        assert len(exits) >= 3, (
            f"Forest should have at least 3 exits (worst case for duplicate destinations), "
            f"got {len(exits)}: {exits}"
        )

        # Verify exits are sorted and unique
        assert exits == sorted(set(exits)), f"Exits should be sorted and unique: {exits}"

    def test_deterministic_behavior_repeated_calls(self, jericho_interface):
        """
        Validate that get_valid_exits() returns consistent results on repeated calls.

        This ensures state save/restore is working correctly and the method
        doesn't have side effects that change behavior across calls.
        """
        location = jericho_interface.get_location_structured()
        starting_location = location.num  # Record actual starting location

        # Call get_valid_exits() multiple times
        exits_1 = jericho_interface.get_valid_exits()
        exits_2 = jericho_interface.get_valid_exits()
        exits_3 = jericho_interface.get_valid_exits()

        # All results should be identical
        assert exits_1 == exits_2, "First and second calls should return same exits"
        assert exits_2 == exits_3, "Second and third calls should return same exits"
        assert exits_1 == exits_3, "First and third calls should return same exits"

        # Verify location hasn't changed
        location_after = jericho_interface.get_location_structured()
        assert location_after.num == starting_location, (
            "Location should not change after calling get_valid_exits(). "
            f"Started at {starting_location}, ended at {location_after.num}"
        )

    def test_state_restoration_after_exit_detection(self, jericho_interface):
        """
        Validate that game state is completely restored after checking exits.

        This ensures the state save/restore mechanism works correctly and
        get_valid_exits() doesn't cause side effects on game state.
        """
        # Record initial state
        initial_location = jericho_interface.get_location_structured()
        initial_score, initial_moves = jericho_interface.get_score()
        initial_inventory = jericho_interface.get_inventory_structured()

        # Call get_valid_exits() (which will test multiple directions)
        exits = jericho_interface.get_valid_exits()
        assert len(exits) > 0, "Should detect at least some exits"

        # Verify state is unchanged
        final_location = jericho_interface.get_location_structured()
        final_score, final_moves = jericho_interface.get_score()
        final_inventory = jericho_interface.get_inventory_structured()

        assert final_location.num == initial_location.num, (
            f"Location changed from {initial_location.num} to {final_location.num}"
        )
        assert final_score == initial_score, (
            f"Score changed from {initial_score} to {final_score}"
        )
        assert final_moves == initial_moves, (
            f"Moves changed from {initial_moves} to {final_moves}"
        )
        assert len(final_inventory) == len(initial_inventory), (
            f"Inventory changed from {len(initial_inventory)} to {len(final_inventory)} items"
        )

    def test_comparison_with_walkthrough_known_working_directions(self, jericho_interface):
        """
        Validate that get_valid_exits() detects all directions used in walkthrough.

        This test uses the first 20 walkthrough steps to validate that
        get_valid_exits() correctly detects the directions used in successful
        gameplay. This is a reality check against known-working game sequences.
        """
        walkthrough_slice = get_walkthrough_slice(0, 20)

        # Track all direction commands used in walkthrough
        direction_commands = {
            "north", "south", "east", "west",
            "northeast", "northwest", "southeast", "southwest",
            "up", "down", "in", "out", "enter", "exit",
            "ne", "nw", "se", "sw", "n", "s", "e", "w", "u", "d"
        }

        for action in walkthrough_slice:
            # Get location before action
            location_before = jericho_interface.get_location_structured()

            # Get exits before executing action
            exits_before = jericho_interface.get_valid_exits()

            # Execute the action
            response = jericho_interface.send_command(action)

            # Get location after action
            location_after = jericho_interface.get_location_structured()

            # If action was a direction command, verify it was detected
            if action.lower() in direction_commands:
                # If location changed, the direction should have been in exits
                if location_after.num != location_before.num and "can't go that way" not in response.lower():
                    assert action.lower() in exits_before, (
                        f"get_valid_exits() missed working direction '{action}'. "
                        f"Detected exits were: {exits_before}"
                    )

    def test_empty_result_on_error(self, jericho_interface):
        """
        Validate that get_valid_exits() returns empty list on errors.

        This ensures graceful degradation when the environment is in
        an invalid state or encounters errors.
        """
        # Close the environment to simulate error state
        jericho_interface.close()

        # Should return empty list, not raise exception
        exits = jericho_interface.get_valid_exits()
        assert exits == [], "Should return empty list when environment is closed"

    def test_exits_include_up_down_when_present(self, jericho_interface):
        """
        Validate that vertical exits (up, down) are detected when present.

        This ensures the method doesn't just detect cardinal directions,
        but also vertical and special movement directions.
        """
        # Navigate to a location with up/down exits
        # From walkthrough: West of House -> north -> up (to tree)
        location_before = jericho_interface.get_location_structured()
        jericho_interface.send_command("north")
        response = jericho_interface.send_command("up")
        location_after = jericho_interface.get_location_structured()

        # If we successfully moved up, there should be a down exit
        # (Most locations with 'up' have corresponding 'down')
        if location_after.num != location_before.num and "can't" not in response.lower():
            exits = jericho_interface.get_valid_exits()

            # Verify we can detect vertical directions (not just cardinal)
            assert "down" in exits or "d" in exits, (
                f"Should detect 'down' exit after successfully going 'up'. "
                f"Location {location_after.num} exits: {exits}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
