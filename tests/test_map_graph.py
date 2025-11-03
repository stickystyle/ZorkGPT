import unittest
from map_graph import (
    MapGraph,
    Room,
    normalize_direction,
)


class TestNormalizeDirection(unittest.TestCase):
    def test_normalize_simple_directions(self):
        self.assertEqual(normalize_direction("n"), "north")
        self.assertEqual(normalize_direction("go south"), "south")
        self.assertEqual(normalize_direction("E"), "east")
        self.assertEqual(normalize_direction("westward"), "west")
        self.assertEqual(normalize_direction("U"), "up")
        self.assertEqual(normalize_direction("climb down"), "down")
        self.assertEqual(normalize_direction("NE"), "northeast")
        self.assertEqual(normalize_direction("NorthWest"), "northwest")
        self.assertEqual(normalize_direction("go se"), "southeast")
        self.assertEqual(normalize_direction("sW"), "southwest")
        self.assertEqual(normalize_direction("in"), "in")
        self.assertEqual(normalize_direction("out"), "out")

    def test_normalize_invalid_directions(self):
        self.assertIsNone(normalize_direction("take lamp"))
        self.assertIsNone(normalize_direction("look"))
        self.assertIsNone(normalize_direction("go to the house"))
        self.assertIsNone(normalize_direction("examine door"))
        self.assertIsNone(normalize_direction(""))
        self.assertIsNone(normalize_direction("  "))

    def test_normalize_edge_cases(self):
        self.assertEqual(normalize_direction("  Go  Up  "), "up")
        self.assertEqual(normalize_direction("go up"), "up")


class TestMapGraph(unittest.TestCase):
    def setUp(self):
        self.map = MapGraph()

    def test_add_room(self):
        # Room ID mapping for test
        WEST_OF_HOUSE_ID = 1

        self.map.add_room(WEST_OF_HOUSE_ID, "West of House")
        self.assertIn(WEST_OF_HOUSE_ID, self.map.rooms)
        self.assertIsInstance(self.map.rooms[WEST_OF_HOUSE_ID], Room)
        self.assertEqual(self.map.rooms[WEST_OF_HOUSE_ID].name, "West of House")

    def test_update_room_exits_normalization(self):
        # Room ID mapping for test
        CAVE_ID = 1

        self.map.add_room(CAVE_ID, "Cave")
        self.map.update_room_exits(CAVE_ID, ["N", "go East", "downward", "slide"])
        cave_room = self.map.rooms[CAVE_ID]
        self.assertIn("north", cave_room.exits)
        self.assertIn("east", cave_room.exits)
        self.assertIn("down", cave_room.exits)
        self.assertIn("slide", cave_room.exits)

    def test_add_connection_simple_and_bidirectional(self):
        # Room ID mapping for test
        WEST_ID = 1
        NORTH_ID = 2

        self.map.add_room(WEST_ID, "West of House")
        self.map.add_room(NORTH_ID, "North of House")
        self.map.add_connection(WEST_ID, "north", NORTH_ID)
        self.assertIn(WEST_ID, self.map.connections)
        self.assertEqual(self.map.connections[WEST_ID]["north"], NORTH_ID)
        self.assertIn(NORTH_ID, self.map.connections)
        self.assertEqual(self.map.connections[NORTH_ID]["south"], WEST_ID)
        self.assertIn("north", self.map.rooms[WEST_ID].exits)
        self.assertIn("south", self.map.rooms[NORTH_ID].exits)

    def test_add_connection_with_normalization_in_add_connection(self):
        # Room ID mapping for test
        LIVING_ROOM_ID = 1
        KITCHEN_ID = 2

        self.map.add_room(LIVING_ROOM_ID, "Living Room")
        self.map.add_room(KITCHEN_ID, "Kitchen")
        self.map.add_connection(LIVING_ROOM_ID, "west", KITCHEN_ID)
        self.assertEqual(self.map.connections[LIVING_ROOM_ID]["west"], KITCHEN_ID)
        self.assertEqual(self.map.connections[KITCHEN_ID]["east"], LIVING_ROOM_ID)

    def test_add_connection_non_directional(self):
        # Room ID mapping for test
        CELLAR_ID = 1
        DARK_TUNNEL_ID = 2

        self.map.add_room(CELLAR_ID, "Cellar")
        self.map.add_room(DARK_TUNNEL_ID, "Dark Tunnel")
        self.map.add_connection(CELLAR_ID, "open trap door", DARK_TUNNEL_ID)
        self.assertEqual(
            self.map.connections[CELLAR_ID]["open trap door"], DARK_TUNNEL_ID
        )
        # No reverse connection for non-directional actions
        self.assertNotIn("open trap door", self.map.connections.get(DARK_TUNNEL_ID, {}))

    def test_add_connection_one_way_if_opposite_exists(self):
        # Room ID mapping for test
        ROOM_A_ID = 1
        ROOM_B_ID = 2
        ROOM_C_ID = 3

        self.map.add_room(ROOM_A_ID, "RoomA")
        self.map.add_room(ROOM_B_ID, "RoomB")
        self.map.add_room(ROOM_C_ID, "RoomC")
        self.map.add_connection(ROOM_A_ID, "north", ROOM_B_ID)
        self.map.add_connection(ROOM_C_ID, "south", ROOM_B_ID)
        self.assertEqual(self.map.connections[ROOM_B_ID]["south"], ROOM_A_ID)
        self.assertEqual(self.map.connections[ROOM_B_ID]["north"], ROOM_C_ID)

    def test_get_room_info(self):
        # Room ID mapping for test
        ATTIC_ID = 1
        LANDING_ID = 2

        self.map.add_room(ATTIC_ID, "Attic")
        self.map.add_room(LANDING_ID, "Landing")
        self.map.add_connection(ATTIC_ID, "down", LANDING_ID)
        self.map.update_room_exits(ATTIC_ID, ["down", "window (closed)"])

        info = self.map.get_room_info(ATTIC_ID)
        self.assertIn("Current room: Attic.", info)
        self.assertIn("down (leads to Landing)", info)
        self.assertIn("window (closed)", info)

    def test_connection_overwrite_preserves_symmetry_if_possible(self):
        # Room ID mapping for test
        ROOM_A_ID = 1
        ROOM_B_ID = 2
        ROOM_C_ID = 3
        ROOM_D_ID = 4

        self.map.add_room(ROOM_A_ID, "A")
        self.map.add_room(ROOM_B_ID, "B")
        self.map.add_room(ROOM_C_ID, "C")
        self.map.add_room(ROOM_D_ID, "D")
        self.map.add_connection(ROOM_B_ID, "north", ROOM_C_ID)
        self.map.add_connection(ROOM_B_ID, "south", ROOM_A_ID)
        self.assertEqual(self.map.connections[ROOM_B_ID]["north"], ROOM_C_ID)
        self.assertEqual(self.map.connections[ROOM_C_ID]["south"], ROOM_B_ID)
        self.assertEqual(self.map.connections[ROOM_B_ID]["south"], ROOM_A_ID)
        self.assertEqual(self.map.connections[ROOM_A_ID]["north"], ROOM_B_ID)
        self.map.add_connection(ROOM_D_ID, "north", ROOM_B_ID)
        self.assertEqual(self.map.connections[ROOM_D_ID]["north"], ROOM_B_ID)
        self.assertEqual(self.map.connections[ROOM_B_ID]["south"], ROOM_A_ID)  # Should remain A
        self.assertEqual(self.map.connections[ROOM_A_ID]["north"], ROOM_B_ID)

    def test_get_context_for_prompt_empty_current_room_name(self):
        context = self.map.get_context_for_prompt(None, "Some Room")
        self.assertIn("Map: Current location ID is missing.", context)

    def test_get_context_for_prompt_newly_added_room(self):
        # Room ID mapping for test
        VOID_ID = 1

        self.map.add_room(VOID_ID, "Void")
        context = self.map.get_context_for_prompt(VOID_ID, "Void")
        self.assertIn("Current location: Void (according to map).", context)

    def test_get_context_for_prompt_with_arrival_info(self):
        # Room ID mapping for test
        OUTSIDE_ID = 1
        ENTRANCE_HALL_ID = 2

        self.map.add_room(OUTSIDE_ID, "Outside")
        self.map.add_room(ENTRANCE_HALL_ID, "Entrance Hall")
        context = self.map.get_context_for_prompt(
            ENTRANCE_HALL_ID, "Entrance Hall", OUTSIDE_ID, "Outside", "enter building"
        )
        self.assertIn("Current location: Entrance Hall (according to map).", context)
        self.assertIn("You arrived from 'Outside' by going enter building.", context)

    def test_get_context_for_prompt_unknown_room_name(self):
        # Room ID that doesn't exist in the map
        UNKNOWN_ID = 999

        context = self.map.get_context_for_prompt(UNKNOWN_ID, "Deep Dungeon")
        self.assertIn("Map: Location 'Deep Dungeon' is new or not yet mapped.", context)

    def test_confidence_tracking(self):
        """Test the enhanced confidence tracking features."""
        # Room ID mapping for test
        ROOM_A_ID = 1
        ROOM_B_ID = 2

        # Add a connection with default confidence
        self.map.add_room(ROOM_A_ID, "Room A")
        self.map.add_room(ROOM_B_ID, "Room B")
        self.map.add_connection(ROOM_A_ID, "north", ROOM_B_ID)

        # Check initial confidence
        confidence = self.map.get_connection_confidence(ROOM_A_ID, "north")
        self.assertEqual(confidence, 1.0)

        # Add the same connection again to increase confidence
        self.map.add_connection(ROOM_A_ID, "north", ROOM_B_ID)

        # Check that verifications increased
        key = (ROOM_A_ID, "north")
        self.assertGreater(self.map.connection_verifications.get(key, 0), 1)

    def test_map_quality_metrics(self):
        """Test the map quality metrics functionality."""
        # Room ID mapping for test
        START_ID = 1
        MIDDLE_ID = 2
        END_ID = 3

        # Add some connections
        self.map.add_room(START_ID, "Start")
        self.map.add_room(MIDDLE_ID, "Middle")
        self.map.add_room(END_ID, "End")
        self.map.add_connection(START_ID, "north", MIDDLE_ID)
        self.map.add_connection(MIDDLE_ID, "east", END_ID)

        # Get quality metrics
        metrics = self.map.get_map_quality_metrics()

        # Check that metrics are returned
        self.assertIn("total_connections", metrics)
        self.assertIn("average_confidence", metrics)
        self.assertIn("high_confidence_ratio", metrics)
        self.assertIn("verified_connections", metrics)

        # Check reasonable values
        self.assertGreaterEqual(metrics["average_confidence"], 0.0)
        self.assertLessEqual(metrics["average_confidence"], 1.0)

    def test_non_directional_actions_preserved(self):
        """Test that non-directional actions are preserved separately."""
        # Room ID mapping for test
        ROOM_C_ID = 1
        ROOM_D_ID = 2
        ROOM_E_ID = 3

        self.map.add_room(ROOM_C_ID, "Room C")
        self.map.add_room(ROOM_D_ID, "Room D")
        self.map.add_room(ROOM_E_ID, "Room E")
        self.map.add_connection(ROOM_C_ID, "open door", ROOM_D_ID)
        self.map.add_connection(ROOM_C_ID, "north", ROOM_E_ID)

        # Should have two separate connections
        self.assertEqual(len(self.map.connections[ROOM_C_ID]), 2)
        self.assertIn("open door", self.map.connections[ROOM_C_ID])
        self.assertIn("north", self.map.connections[ROOM_C_ID])


if __name__ == "__main__":
    unittest.main()
