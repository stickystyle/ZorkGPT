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
        self.map.add_room("West of House")
        self.assertIn("West Of House", self.map.rooms)  # Normalized to title case
        self.assertIsInstance(self.map.rooms["West Of House"], Room)
        self.assertEqual(self.map.rooms["West Of House"].name, "West Of House")

    def test_update_room_exits_normalization(self):
        self.map.add_room("Cave")
        self.map.update_room_exits("Cave", ["N", "go East", "downward", "slide"])
        cave_room = self.map.rooms["Cave"]
        self.assertIn("north", cave_room.exits)
        self.assertIn("east", cave_room.exits)
        self.assertIn("down", cave_room.exits)
        self.assertIn("slide", cave_room.exits)

    def test_add_connection_simple_and_bidirectional(self):
        self.map.add_connection("West of House", "north", "North of House")
        self.assertIn("West Of House", self.map.connections)  # Normalized
        self.assertEqual(
            self.map.connections["West Of House"]["north"], "North Of House"
        )
        self.assertIn("North Of House", self.map.connections)  # Normalized
        self.assertEqual(
            self.map.connections["North Of House"]["south"], "West Of House"
        )
        self.assertIn("north", self.map.rooms["West Of House"].exits)  # Normalized
        self.assertIn("south", self.map.rooms["North Of House"].exits)  # Normalized

    def test_add_connection_with_normalization_in_add_connection(self):
        self.map.add_connection("Living Room", "west", "Kitchen")
        self.assertEqual(self.map.connections["Living Room"]["west"], "Kitchen")
        self.assertEqual(self.map.connections["Kitchen"]["east"], "Living Room")

    def test_add_connection_non_directional(self):
        self.map.add_connection("Cellar", "open trap door", "Dark Tunnel")
        self.assertEqual(
            self.map.connections["Cellar"]["open trap door"], "Dark Tunnel"
        )
        self.assertNotIn("Dark Tunnel", self.map.connections)

    def test_add_connection_one_way_if_opposite_exists(self):
        self.map.add_connection("RoomA", "north", "RoomB")
        self.map.add_connection("RoomC", "south", "RoomB")
        self.assertEqual(self.map.connections["Roomb"]["south"], "Rooma")  # Normalized
        self.assertEqual(self.map.connections["Roomb"]["north"], "Roomc")  # Normalized

    def test_get_room_info(self):
        self.map.add_connection("Attic", "down", "Landing")
        self.map.update_room_exits("Attic", ["down", "window (closed)"])

        info = self.map.get_room_info("Attic")
        self.assertIn("Current room: Attic.", info)
        self.assertIn("down (leads to Landing)", info)
        self.assertIn("window (closed)", info)

    def test_connection_overwrite_preserves_symmetry_if_possible(self):
        self.map.add_connection("B", "north", "C")
        self.map.add_connection("B", "south", "A")
        self.assertEqual(self.map.connections["B"]["north"], "C")
        self.assertEqual(self.map.connections["C"]["south"], "B")
        self.assertEqual(self.map.connections["B"]["south"], "A")
        self.assertEqual(self.map.connections["A"]["north"], "B")
        self.map.add_connection("D", "north", "B")
        self.assertEqual(self.map.connections["D"]["north"], "B")
        self.assertEqual(self.map.connections["B"]["south"], "A")  # Should remain A
        self.assertEqual(self.map.connections["A"]["north"], "B")

    def test_get_context_for_prompt_empty_current_room_name(self):
        context = self.map.get_context_for_prompt(None)
        self.assertIn("Map: Current location name is missing.", context)
        context = self.map.get_context_for_prompt("")
        self.assertIn("Map: Current location name is missing.", context)

    def test_get_context_for_prompt_newly_added_room(self):
        self.map.add_room("Void")
        context = self.map.get_context_for_prompt("Void")
        self.assertIn("Current location: Void (according to map).", context)

    def test_get_context_for_prompt_with_arrival_info(self):
        self.map.add_room("Entrance Hall")
        context = self.map.get_context_for_prompt(
            "Entrance Hall", "Outside", "enter building"
        )
        self.assertIn("Current location: Entrance Hall (according to map).", context)
        self.assertIn("You arrived from 'Outside' by going enter building.", context)

    def test_get_context_for_prompt_unknown_room_name(self):
        context = self.map.get_context_for_prompt("Deep Dungeon")
        self.assertIn("Map: Location 'Deep Dungeon' is new or not yet mapped.", context)

    def test_confidence_tracking(self):
        """Test the enhanced confidence tracking features."""
        # Add a connection with default confidence
        self.map.add_connection("Room A", "north", "Room B")

        # Check initial confidence
        confidence = self.map.get_connection_confidence("Room A", "north")
        self.assertEqual(confidence, 1.0)

        # Add the same connection again to increase confidence
        self.map.add_connection("Room A", "north", "Room B")

        # Check that verifications increased
        key = ("Room A", "north")
        self.assertGreater(self.map.connection_verifications.get(key, 0), 1)

    def test_map_quality_metrics(self):
        """Test the map quality metrics functionality."""
        # Add some connections
        self.map.add_connection("Start", "north", "Middle")
        self.map.add_connection("Middle", "east", "End")

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

    def test_navigation_suggestions(self):
        """Test the navigation suggestions functionality."""
        # Add some connections
        self.map.add_connection("Hub", "north", "North Room")
        self.map.add_connection("Hub", "south", "South Room")

        # Get navigation suggestions
        suggestions = self.map.get_navigation_suggestions("Hub")

        # Check that suggestions are returned
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

        # Check suggestion structure
        for suggestion in suggestions:
            self.assertIn("exit", suggestion)
            self.assertIn("destination", suggestion)
            self.assertIn("confidence", suggestion)

    def test_non_directional_actions_preserved(self):
        """Test that non-directional actions are preserved separately."""
        self.map.add_connection("Room C", "open door", "Room D")
        self.map.add_connection("Room C", "north", "Room E")

        # Should have two separate connections
        self.assertEqual(len(self.map.connections["Room C"]), 2)
        self.assertIn("open door", self.map.connections["Room C"])
        self.assertIn("north", self.map.connections["Room C"])


if __name__ == "__main__":
    unittest.main()
