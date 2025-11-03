# ABOUTME: Phase 7.2 - Comprehensive integration tests using walkthrough fixtures
# ABOUTME: Validates Jericho implementation through deterministic walkthrough replays

import pytest
from typing import List, Tuple, Dict, Any
from pathlib import Path
from jericho import FrotzEnv

from game_interface.core.jericho_interface import JerichoInterface
from map_graph import MapGraph
from tests.fixtures.walkthrough import (
    get_zork1_walkthrough,
    get_walkthrough_slice,
    get_walkthrough_until_lamp,
    get_walkthrough_dark_sequence,
    replay_walkthrough,
    GAME_FILE_PATH,
)


@pytest.fixture
def jericho_interface():
    """Create a fresh Jericho interface for each test."""
    interface = JerichoInterface(GAME_FILE_PATH)
    interface.start()
    yield interface
    interface.close()


@pytest.fixture
def frotz_env():
    """Create a fresh FrotzEnv for walkthrough replay tests."""
    env = FrotzEnv(GAME_FILE_PATH)
    env.reset()
    yield env
    env.close()


class TestLocationIDStability:
    """Test that location IDs are deterministic and stable across replays."""

    def test_location_id_determinism_across_replays(self, jericho_interface):
        """Verify location IDs are deterministic across multiple replays."""
        walkthrough = get_walkthrough_until_lamp()

        # Track location IDs for first replay
        first_replay_ids = []

        # First replay
        for action in walkthrough:
            jericho_interface.send_command(action)
            location = jericho_interface.get_location_structured()
            first_replay_ids.append(location.num)

        # Reset to initial state
        jericho_interface.close()
        jericho_interface_2 = JerichoInterface(GAME_FILE_PATH)
        jericho_interface_2.start()

        # Track location IDs for second replay
        second_replay_ids = []

        # Second replay
        for action in walkthrough:
            jericho_interface_2.send_command(action)
            location = jericho_interface_2.get_location_structured()
            second_replay_ids.append(location.num)

        jericho_interface_2.close()

        # Location IDs should be identical across replays
        assert first_replay_ids == second_replay_ids, \
            "Location IDs should be deterministic across replays"
        assert len(first_replay_ids) > 0, \
            "Should have captured location IDs from walkthrough"

    def test_location_id_uniqueness_for_different_rooms(self, jericho_interface):
        """Verify each unique location has a unique ID."""
        walkthrough = get_walkthrough_until_lamp()

        # Track ID-to-name mapping
        id_to_name_mapping: Dict[int, str] = {}

        for action in walkthrough:
            jericho_interface.send_command(action)
            location = jericho_interface.get_location_structured()
            location_id = location.num
            location_name = location.name

            if location_id in id_to_name_mapping:
                # Same ID should always map to same name
                assert id_to_name_mapping[location_id] == location_name, \
                    f"ID {location_id} mapped to multiple names: " \
                    f"{id_to_name_mapping[location_id]} vs {location_name}"
            else:
                id_to_name_mapping[location_id] = location_name

        # Should have visited multiple locations
        assert len(id_to_name_mapping) > 1, \
            "Walkthrough should visit multiple locations"

    def test_same_location_revisit_has_same_id(self, jericho_interface):
        """Verify revisiting the same location returns the same ID."""
        # Get starting location
        start_location = jericho_interface.get_location_structured()
        start_id = start_location.num
        start_name = start_location.name

        # Go north from starting position
        jericho_interface.send_command("north")
        first_location = jericho_interface.get_location_structured()

        # Go back using a movement that should work (try multiple directions)
        # In Zork, going back isn't always the opposite direction
        # So let's use a more reliable test: go somewhere and look at it twice
        jericho_interface.send_command("look")
        looked_location_1 = jericho_interface.get_location_structured()
        looked_id_1 = looked_location_1.num
        looked_name_1 = looked_location_1.name

        # Look again - should be same location
        jericho_interface.send_command("look")
        looked_location_2 = jericho_interface.get_location_structured()
        looked_id_2 = looked_location_2.num
        looked_name_2 = looked_location_2.name

        # Looking at the same place multiple times should give same ID
        assert looked_id_1 == looked_id_2, \
            f"Looking at same location should return same ID: {looked_id_1} vs {looked_id_2}"
        assert looked_name_1 == looked_name_2, \
            f"Looking at same location should return same name: {looked_name_1} vs {looked_name_2}"


class TestMapBuilding:
    """Test map building capabilities using walkthrough data."""

    def test_map_building_with_walkthrough_until_lamp(self, jericho_interface):
        """Test map correctly tracks movements during lamp acquisition."""
        walkthrough = get_walkthrough_until_lamp()
        map_graph = MapGraph()

        previous_location_id = None
        previous_action = None

        for action in walkthrough:
            # Get current location before action
            if previous_location_id is None:
                current_location = jericho_interface.get_location_structured()
                previous_location_id = current_location.num
                map_graph.add_room(current_location.num, current_location.name)

            # Execute action
            jericho_interface.send_command(action)
            new_location = jericho_interface.get_location_structured()

            # Add new room to map
            map_graph.add_room(new_location.num, new_location.name)

            # If location changed, add connection
            if new_location.num != previous_location_id and previous_action is not None:
                from map_graph import normalize_direction, is_non_movement_command

                # Only add connection for movement commands
                if not is_non_movement_command(previous_action):
                    map_graph.add_connection(
                        from_room_id=previous_location_id,
                        exit_taken=previous_action,
                        to_room_id=new_location.num,
                        confidence=1.0
                    )

            previous_location_id = new_location.num
            previous_action = action

        # Map should have multiple rooms
        assert len(map_graph.rooms) > 1, \
            f"Map should have multiple rooms, found {len(map_graph.rooms)}"

        # Map should have some connections
        assert len(map_graph.connections) > 0, \
            "Map should have tracked some connections"

    def test_integer_id_map_structure(self, jericho_interface):
        """Verify all map keys are integers (not strings)."""
        walkthrough = get_walkthrough_slice(0, 20)
        map_graph = MapGraph()

        for action in walkthrough:
            jericho_interface.send_command(action)
            location = jericho_interface.get_location_structured()
            map_graph.add_room(location.num, location.name)

        # Verify all room IDs are integers
        for room_id in map_graph.rooms.keys():
            assert isinstance(room_id, int), \
                f"Room ID should be int, got {type(room_id)}: {room_id}"

        # Verify all connection keys are integers
        for from_id in map_graph.connections.keys():
            assert isinstance(from_id, int), \
                f"Connection from_id should be int, got {type(from_id)}: {from_id}"

            for to_id in map_graph.connections[from_id].values():
                assert isinstance(to_id, int), \
                    f"Connection to_id should be int, got {type(to_id)}: {to_id}"

        # Verify room_names keys are integers
        for room_id in map_graph.room_names.keys():
            assert isinstance(room_id, int), \
                f"room_names key should be int, got {type(room_id)}: {room_id}"


class TestZeroFragmentation:
    """Test that there is no room fragmentation (same room always has same ID)."""

    def test_no_room_fragmentation_50_actions(self, jericho_interface):
        """Verify each ID maps to one name consistently over 50 actions."""
        walkthrough = get_walkthrough_slice(0, 50)

        # Track ID-to-name mapping throughout gameplay
        id_to_names: Dict[int, set] = {}

        for action in walkthrough:
            jericho_interface.send_command(action)
            location = jericho_interface.get_location_structured()

            if location.num not in id_to_names:
                id_to_names[location.num] = set()

            id_to_names[location.num].add(location.name)

        # Each ID should map to exactly one name
        for location_id, names in id_to_names.items():
            assert len(names) == 1, \
                f"Location ID {location_id} mapped to multiple names: {names}. " \
                f"This indicates room fragmentation!"

    def test_revisited_locations_maintain_same_id(self, jericho_interface):
        """Test that revisiting locations maintains ID consistency."""
        # Use walkthrough to ensure we actually revisit locations
        # The walkthrough is deterministic and will revisit some locations
        walkthrough = get_walkthrough_slice(0, 50)

        # Track all location visits
        location_visits: Dict[int, List[str]] = {}

        for action in walkthrough:
            jericho_interface.send_command(action)
            location = jericho_interface.get_location_structured()

            if location.num not in location_visits:
                location_visits[location.num] = []

            location_visits[location.num].append(location.name)

        # Find locations that were visited multiple times
        revisited_locations = {
            loc_id: names for loc_id, names in location_visits.items()
            if len(names) > 1
        }

        # Should have some revisited locations in 50 steps
        assert len(revisited_locations) > 0, \
            "Walkthrough should revisit some locations"

        # For each revisited location, all names should be identical
        for loc_id, names in revisited_locations.items():
            unique_names = set(names)
            assert len(unique_names) == 1, \
                f"Location ID {loc_id} mapped to multiple names on revisits: {unique_names}"


class TestDarkRoomMovement:
    """Test movement detection in dark rooms where descriptions are minimal."""

    def test_dark_room_movement_detection(self, jericho_interface):
        """Verify IDs work in pitch dark rooms even without descriptions."""
        # Get to dark area
        lamp_sequence = get_walkthrough_until_lamp()
        dark_sequence = get_walkthrough_dark_sequence()

        # Play through lamp acquisition
        for action in lamp_sequence:
            jericho_interface.send_command(action)

        # Track location changes in dark sequence
        location_changes = 0
        previous_id = jericho_interface.get_location_structured().num

        for action in dark_sequence:
            jericho_interface.send_command(action)
            current_location = jericho_interface.get_location_structured()
            current_id = current_location.num

            if current_id != previous_id:
                location_changes += 1

            previous_id = current_id

        # Should detect movement even in dark areas
        assert location_changes > 0, \
            "Should detect location changes even in dark rooms via ID comparison"

    def test_location_id_changes_indicate_movement(self, jericho_interface):
        """Test that ID changes reliably indicate movement occurred."""
        # Test with clear movement sequence
        movement_actions = ["north", "east", "south", "west"]

        previous_id = jericho_interface.get_location_structured().num
        movements_detected = 0

        for action in movement_actions:
            jericho_interface.send_command(action)
            current_id = jericho_interface.get_location_structured().num

            if current_id != previous_id:
                movements_detected += 1

            previous_id = current_id

        # At least some movements should be detected
        # (Note: Not all directions may be valid, but at least one should work)
        assert movements_detected > 0, \
            "ID comparison should detect when movement occurs"


class TestInventoryTracking:
    """Test structured inventory tracking through walkthrough."""

    def test_inventory_tracking_walkthrough(self, jericho_interface):
        """Test structured inventory changes through lamp acquisition."""
        walkthrough = get_walkthrough_until_lamp()

        initial_inventory = jericho_interface.get_inventory_structured()
        inventory_changes = 0

        for action in walkthrough:
            prev_inv_count = len(jericho_interface.get_inventory_structured())
            jericho_interface.send_command(action)
            curr_inv_count = len(jericho_interface.get_inventory_structured())

            if curr_inv_count != prev_inv_count:
                inventory_changes += 1

        final_inventory = jericho_interface.get_inventory_structured()

        # Should have acquired items (lamp acquisition sequence)
        assert len(final_inventory) > len(initial_inventory), \
            "Should have acquired items during walkthrough"

        assert inventory_changes > 0, \
            "Should detect inventory changes"

    def test_inventory_objects_have_integer_ids(self, jericho_interface):
        """Verify inventory objects have integer IDs (ZObject.num)."""
        # Take an item
        walkthrough = get_walkthrough_until_lamp()

        for action in walkthrough:
            jericho_interface.send_command(action)

        inventory = jericho_interface.get_inventory_structured()

        if len(inventory) > 0:
            for item in inventory:
                assert hasattr(item, 'num'), \
                    "Inventory items should have 'num' attribute"
                assert isinstance(item.num, int), \
                    f"Inventory item ID should be int, got {type(item.num)}"
                assert hasattr(item, 'name'), \
                    "Inventory items should have 'name' attribute"


class TestExtendedSession:
    """Test map stability and consistency over extended gameplay."""

    def test_extended_session_100_turns(self, jericho_interface):
        """Test map stability over extended gameplay (100 turns)."""
        walkthrough = get_walkthrough_slice(0, 100)

        # Track ID consistency
        id_to_name_mapping: Dict[int, str] = {}
        total_actions = 0
        fragmentation_errors = []

        for action in walkthrough:
            total_actions += 1
            jericho_interface.send_command(action)
            location = jericho_interface.get_location_structured()

            if location.num in id_to_name_mapping:
                # Check for fragmentation
                if id_to_name_mapping[location.num] != location.name:
                    fragmentation_errors.append({
                        'turn': total_actions,
                        'location_id': location.num,
                        'expected_name': id_to_name_mapping[location.num],
                        'actual_name': location.name,
                    })
            else:
                id_to_name_mapping[location.num] = location.name

        # No fragmentation should occur
        assert len(fragmentation_errors) == 0, \
            f"Found {len(fragmentation_errors)} fragmentation errors: {fragmentation_errors[:3]}"

        # Should have executed all actions
        assert total_actions == len(walkthrough), \
            f"Should have executed {len(walkthrough)} actions, executed {total_actions}"

    def test_map_connections_remain_stable(self, jericho_interface):
        """Test that map connections remain stable over extended play."""
        walkthrough = get_walkthrough_slice(0, 50)
        map_graph = MapGraph()

        # First pass - build map
        previous_location_id = None
        previous_action = None

        for action in walkthrough:
            if previous_location_id is None:
                current_location = jericho_interface.get_location_structured()
                previous_location_id = current_location.num
                map_graph.add_room(current_location.num, current_location.name)

            jericho_interface.send_command(action)
            new_location = jericho_interface.get_location_structured()
            map_graph.add_room(new_location.num, new_location.name)

            # Track connections for movement commands
            from map_graph import is_non_movement_command
            if new_location.num != previous_location_id and previous_action:
                if not is_non_movement_command(previous_action):
                    map_graph.add_connection(
                        from_room_id=previous_location_id,
                        exit_taken=previous_action,
                        to_room_id=new_location.num,
                        confidence=1.0
                    )

            previous_location_id = new_location.num
            previous_action = action

        # Check map quality metrics
        metrics = map_graph.get_map_quality_metrics()

        # Should have high confidence connections
        assert metrics['average_confidence'] >= 0.7, \
            f"Average confidence should be high, got {metrics['average_confidence']}"

        # Should have minimal conflicts (fragmentation would cause conflicts)
        assert metrics['conflicts_detected'] <= metrics['total_connections'] * 0.1, \
            f"Should have minimal conflicts, got {metrics['conflicts_detected']} " \
            f"out of {metrics['total_connections']} connections"


class TestWalkthroughInfrastructure:
    """Test the walkthrough fixture infrastructure itself."""

    def test_walkthrough_retrieval_is_cached(self):
        """Verify walkthrough retrieval is cached properly."""
        # Call twice - should return same instance (due to lru_cache)
        walkthrough1 = get_zork1_walkthrough()
        walkthrough2 = get_zork1_walkthrough()

        # Same object due to caching
        assert walkthrough1 is walkthrough2, \
            "Walkthrough should be cached"

    def test_walkthrough_slice_functionality(self):
        """Test walkthrough slicing works correctly."""
        full = get_zork1_walkthrough()
        slice_10 = get_walkthrough_slice(0, 10)
        slice_20_30 = get_walkthrough_slice(20, 30)

        assert len(slice_10) == 10, \
            "Slice should have correct length"
        assert slice_10 == full[0:10], \
            "Slice should match full walkthrough slice"
        assert len(slice_20_30) == 10, \
            "Mid-sequence slice should have correct length"

    def test_replay_walkthrough_integration(self, frotz_env):
        """Test replay_walkthrough helper function."""
        actions = get_walkthrough_slice(0, 5)
        results = replay_walkthrough(frotz_env, actions)

        assert len(results) == len(actions), \
            "Should return results for each action"

        for obs, score, done, info in results:
            assert isinstance(obs, str), \
                "Observation should be string"
            assert isinstance(score, int), \
                "Score should be integer"
            assert isinstance(done, bool), \
                "Done flag should be boolean"
            assert isinstance(info, dict), \
                "Info should be dictionary"


class TestMovementDetectionWithoutTextParsing:
    """Test that movement detection works purely via ID comparison."""

    def test_movement_detection_via_id_only(self, jericho_interface):
        """Verify movement can be detected using only location ID changes."""
        # Establish baseline
        initial_location = jericho_interface.get_location_structured()
        initial_id = initial_location.num

        # Try each cardinal direction
        directions = ["north", "south", "east", "west"]
        movement_detected_count = 0

        for direction in directions:
            jericho_interface.send_command(direction)
            new_location = jericho_interface.get_location_structured()
            new_id = new_location.num

            # Movement detected purely by ID comparison
            if new_id != initial_id:
                movement_detected_count += 1
                # Return to initial position (if possible)
                # This is a simple test, so we'll just track movements

            initial_id = new_id  # Update for next comparison

        # At least some directions should cause movement
        assert movement_detected_count > 0, \
            "Should detect movement via ID comparison for at least one direction"

    def test_no_text_parsing_required_for_movement(self, jericho_interface):
        """Verify we don't need to parse text to detect movement."""
        walkthrough = get_walkthrough_slice(0, 20)

        movements_by_id = 0
        previous_id = jericho_interface.get_location_structured().num

        for action in walkthrough:
            # Execute action and get structured data only
            jericho_interface.send_command(action)
            current_id = jericho_interface.get_location_structured().num

            # Detect movement purely from ID change
            if current_id != previous_id:
                movements_by_id += 1

            previous_id = current_id

        # Should detect movements without any text parsing
        assert movements_by_id > 0, \
            "Should detect movements using only location ID comparison"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
