"""
Comprehensive tests for MapGraph JSON serialization and cross-episode persistence.

Tests cover:
- Serialization/deserialization roundtrip
- Data structure validation
- Cross-episode persistence via MapManager
- Error handling (corrupted JSON, missing files)
- Edge cases (empty maps, complex structures)
"""

import pytest
import json
import os
import tempfile
from map_graph import MapGraph, Room
from managers.map_manager import MapManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class TestMapGraphSerialization:
    """Test MapGraph to_dict() and from_dict() methods."""

    def test_empty_map_serialization(self):
        """Test serialization of an empty map."""
        map1 = MapGraph()
        data = map1.to_dict()

        assert "rooms" in data
        assert "connections" in data
        assert "metadata" in data
        assert data["metadata"]["total_rooms"] == 0
        assert data["metadata"]["total_connections"] == 0
        assert data["metadata"]["version"] == "1.0"

    def test_basic_map_serialization_roundtrip(self):
        """Test basic serialization and deserialization preserve data."""
        # Create map with basic data
        map1 = MapGraph()
        map1.add_room(15, "West of House")
        map1.add_room(180, "North of House")
        map1.add_connection(15, "north", 180, confidence=0.9)
        map1.update_room_exits(15, ["north", "south", "west"])

        # Serialize
        data = map1.to_dict()

        # Deserialize
        map2 = MapGraph.from_dict(data)

        # Verify
        assert len(map2.rooms) == 2
        assert 15 in map2.rooms
        assert 180 in map2.rooms
        assert map2.rooms[15].name == "West of House"
        assert map2.rooms[180].name == "North of House"
        assert "north" in map2.rooms[15].exits
        assert 15 in map2.connections
        assert map2.connections[15]["north"] == 180

    def test_confidence_and_verifications_preserved(self):
        """Test that confidence scores and verification counts are preserved."""
        map1 = MapGraph()
        map1.add_room(15, "West of House")
        map1.add_room(180, "North of House")

        # Add connection with specific confidence
        map1.add_connection(15, "north", 180, confidence=0.85)

        # Manually set verification count
        map1.connection_verifications[(15, "north")] = 3

        # Roundtrip
        data = map1.to_dict()
        map2 = MapGraph.from_dict(data)

        # Verify confidence
        assert (15, "north") in map2.connection_confidence
        assert abs(map2.connection_confidence[(15, "north")] - 0.85) < 0.01

        # Verify verification count
        assert (15, "north") in map2.connection_verifications
        assert map2.connection_verifications[(15, "north")] == 3

    def test_exit_failures_and_pruned_exits_preserved(self):
        """Test that exit failure tracking and pruned exits are preserved."""
        map1 = MapGraph()
        map1.add_room(15, "West of House")

        # Track exit failures
        map1.track_exit_failure(15, "west")
        map1.track_exit_failure(15, "west")
        map1.track_exit_failure(15, "west")

        # Manually add pruned exit
        map1.pruned_exits[15] = {"east", "down"}

        # Roundtrip
        data = map1.to_dict()
        map2 = MapGraph.from_dict(data)

        # Verify exit failures
        assert (15, "west") in map2.exit_failure_counts
        assert map2.exit_failure_counts[(15, "west")] == 3

        # Verify pruned exits
        assert 15 in map2.pruned_exits
        assert map2.pruned_exits[15] == {"east", "down"}

    def test_exits_with_underscores_handled_correctly(self):
        """Test that exit names with underscores are handled correctly."""
        map1 = MapGraph()
        map1.add_room(15, "Test Room")
        map1.add_room(20, "Destination")

        # Exit name with underscore
        map1.add_connection(15, "go_through_door", 20, confidence=0.9)

        # Roundtrip
        data = map1.to_dict()
        map2 = MapGraph.from_dict(data)

        # Verify - should use split("_", 1) so only first underscore splits
        assert (15, "go_through_door") in map2.connection_confidence
        assert map2.connections[15]["go_through_door"] == 20

    def test_complex_map_structure(self):
        """Test serialization of a complex map with many rooms and connections."""
        map1 = MapGraph()

        # Add 10 rooms
        for i in range(10):
            map1.add_room(i, f"Room {i}")

        # Add various connections
        map1.add_connection(0, "north", 1, confidence=0.9)
        map1.add_connection(1, "east", 2, confidence=0.8)
        map1.add_connection(2, "south", 3, confidence=0.95)
        map1.add_connection(3, "west", 0, confidence=0.7)
        map1.add_connection(4, "up", 5, confidence=1.0)
        map1.add_connection(5, "down", 4, confidence=1.0)

        # Add exit failures
        map1.track_exit_failure(0, "south")
        map1.track_exit_failure(1, "west")

        # Roundtrip
        data = map1.to_dict()
        map2 = MapGraph.from_dict(data)

        # Verify structure
        assert len(map2.rooms) == 10
        assert len(map2.connections) > 0
        assert map2.connections[0]["north"] == 1
        assert map2.connections[4]["up"] == 5
        assert (0, "south") in map2.exit_failure_counts


class TestMapGraphFileOperations:
    """Test MapGraph save_to_json() and load_from_json() methods."""

    def test_save_and_load_json(self):
        """Test saving and loading map from JSON file."""
        map1 = MapGraph()
        map1.add_room(15, "West of House")
        map1.add_room(180, "North of House")
        map1.add_connection(15, "north", 180, confidence=0.9)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save
            success = map1.save_to_json(temp_path)
            assert success
            assert os.path.exists(temp_path)

            # Load
            map2 = MapGraph.load_from_json(temp_path)
            assert map2 is not None
            assert len(map2.rooms) == 2
            assert map2.connections[15]["north"] == 180

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_nonexistent_file_returns_none(self):
        """Test that loading a nonexistent file returns None gracefully."""
        map_loaded = MapGraph.load_from_json("nonexistent_file_12345.json")
        assert map_loaded is None

    def test_load_corrupted_json_returns_none(self):
        """Test that loading corrupted JSON returns None gracefully."""
        # Create corrupted JSON file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("{invalid json content")
            temp_path = f.name

        try:
            map_loaded = MapGraph.load_from_json(temp_path)
            assert map_loaded is None
        finally:
            os.remove(temp_path)

    def test_load_invalid_structure_returns_none(self):
        """Test that loading JSON with invalid structure returns None."""
        # Create JSON with missing required fields
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump({"rooms": {}, "invalid_field": "test"}, f)
            temp_path = f.name

        try:
            map_loaded = MapGraph.load_from_json(temp_path)
            assert map_loaded is None
        finally:
            os.remove(temp_path)

    def test_json_format_is_readable(self):
        """Test that saved JSON is human-readable with proper formatting."""
        map1 = MapGraph()
        map1.add_room(15, "West of House")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            map1.save_to_json(temp_path)

            # Read and verify formatting
            with open(temp_path, 'r') as f:
                content = f.read()
                # Should have indentation (indent=2)
                assert "  " in content
                # Should be valid JSON
                data = json.loads(content)
                assert "rooms" in data
                assert "metadata" in data

        finally:
            os.remove(temp_path)


class TestMapManagerIntegration:
    """Test MapManager integration with map persistence."""

    def test_map_manager_loads_existing_map(self):
        """Test that MapManager loads existing map state on initialization."""
        # Create a map and save it
        map1 = MapGraph()
        map1.add_room(15, "West of House")
        map1.add_room(180, "North of House")
        map1.add_connection(15, "north", 180, confidence=0.9)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            map1.save_to_json(temp_path)

            # Create config pointing to temp file
            config = GameConfiguration.from_toml()
            config.map_state_file = temp_path

            # Create MapManager - should load the map
            game_state = GameState()
            map_manager = MapManager(logger=None, config=config, game_state=game_state)

            # Verify map was loaded
            assert len(map_manager.game_map.rooms) == 2
            assert 15 in map_manager.game_map.rooms
            assert map_manager.game_map.rooms[15].name == "West of House"

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_map_manager_starts_fresh_if_no_file(self):
        """Test that MapManager starts with empty map if no state file exists."""
        config = GameConfiguration.from_toml()
        config.map_state_file = "nonexistent_map_file_12345.json"

        game_state = GameState()
        map_manager = MapManager(logger=None, config=config, game_state=game_state)

        # Should have empty map
        assert len(map_manager.game_map.rooms) == 0

    def test_map_manager_save_map_state(self):
        """Test that MapManager.save_map_state() works correctly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Create MapManager with temp file
            config = GameConfiguration.from_toml()
            config.map_state_file = temp_path
            game_state = GameState()
            game_state.episode_id = "test_episode"
            map_manager = MapManager(logger=None, config=config, game_state=game_state)

            # Add some data
            map_manager.game_map.add_room(15, "West of House")
            map_manager.game_map.add_room(180, "North of House")

            # Save
            success = map_manager.save_map_state()
            assert success
            assert os.path.exists(temp_path)

            # Verify saved content
            with open(temp_path, 'r') as f:
                data = json.load(f)
                assert len(data["rooms"]) == 2

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_cross_episode_persistence_simulation(self):
        """Test that map state persists across simulated episodes."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Episode 1: Create MapManager, add rooms, save
            config1 = GameConfiguration.from_toml()
            config1.map_state_file = temp_path
            game_state1 = GameState()
            game_state1.episode_id = "episode_1"
            map_manager1 = MapManager(logger=None, config=config1, game_state=game_state1)

            map_manager1.game_map.add_room(15, "West of House")
            map_manager1.game_map.add_room(180, "North of House")
            map_manager1.game_map.add_connection(15, "north", 180, confidence=0.9)

            map_manager1.save_map_state()

            # Episode 2: Create new MapManager, should load previous state
            config2 = GameConfiguration.from_toml()
            config2.map_state_file = temp_path
            game_state2 = GameState()
            game_state2.episode_id = "episode_2"
            map_manager2 = MapManager(logger=None, config=config2, game_state=game_state2)

            # Verify map was loaded from episode 1
            assert len(map_manager2.game_map.rooms) == 2
            assert 15 in map_manager2.game_map.rooms
            assert map_manager2.game_map.rooms[15].name == "West of House"
            assert map_manager2.game_map.connections[15]["north"] == 180

            # Add more rooms in episode 2
            map_manager2.game_map.add_room(181, "South of House")
            map_manager2.save_map_state()

            # Episode 3: Should have all rooms from episodes 1 and 2
            config3 = GameConfiguration.from_toml()
            config3.map_state_file = temp_path
            game_state3 = GameState()
            game_state3.episode_id = "episode_3"
            map_manager3 = MapManager(logger=None, config=config3, game_state=game_state3)

            assert len(map_manager3.game_map.rooms) == 3
            assert 15 in map_manager3.game_map.rooms
            assert 180 in map_manager3.game_map.rooms
            assert 181 in map_manager3.game_map.rooms

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_room_with_empty_exits(self):
        """Test room with no exits."""
        map1 = MapGraph()
        map1.add_room(15, "Isolated Room")

        data = map1.to_dict()
        map2 = MapGraph.from_dict(data)

        assert 15 in map2.rooms
        assert len(map2.rooms[15].exits) == 0

    def test_room_with_special_characters_in_name(self):
        """Test room names with special characters."""
        map1 = MapGraph()
        map1.add_room(15, "Room with \"quotes\" and 'apostrophes'")
        map1.add_room(20, "Room with [brackets] and {braces}")

        data = map1.to_dict()
        map2 = MapGraph.from_dict(data)

        assert map2.rooms[15].name == "Room with \"quotes\" and 'apostrophes'"
        assert map2.rooms[20].name == "Room with [brackets] and {braces}"

    def test_metadata_includes_timestamp(self):
        """Test that metadata includes a timestamp."""
        map1 = MapGraph()
        map1.add_room(15, "Test Room")

        data = map1.to_dict()

        assert "metadata" in data
        assert "timestamp" in data["metadata"]
        assert "version" in data["metadata"]
        assert data["metadata"]["version"] == "1.0"

    def test_backward_compatibility_with_missing_optional_fields(self):
        """Test that from_dict works with minimal required fields."""
        # Create minimal data structure
        minimal_data = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": []}
            },
            "connections": {},
            "metadata": {"version": "1.0", "total_rooms": 1, "total_connections": 0}
        }

        # Should not raise exception
        map1 = MapGraph.from_dict(minimal_data)
        assert len(map1.rooms) == 1
        assert 15 in map1.rooms


class TestSecurityValidation:
    """Test security validations added after code review."""

    def test_connection_to_nonexistent_room_rejected(self):
        """Test that connections to non-existent rooms are rejected."""
        invalid_data = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": ["north"]}
            },
            "connections": {
                "15": {"north": 999}  # Room 999 doesn't exist
            },
            "metadata": {"version": "1.0", "total_rooms": 1, "total_connections": 1}
        }

        with pytest.raises(ValueError, match="does not exist"):
            MapGraph.from_dict(invalid_data)

    def test_connection_with_non_integer_destination_rejected(self):
        """Test that connections with non-integer destinations are rejected."""
        invalid_data = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": []},
                "20": {"id": 20, "name": "Dest Room", "exits": []}
            },
            "connections": {
                "15": {"north": "not_an_int"}  # Should be int
            },
            "metadata": {"version": "1.0", "total_rooms": 2, "total_connections": 1}
        }

        with pytest.raises(ValueError, match="Invalid connection destination type"):
            MapGraph.from_dict(invalid_data)

    def test_confidence_with_non_numeric_value_rejected(self):
        """Test that non-numeric confidence scores are rejected."""
        invalid_data = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": []}
            },
            "connections": {},
            "connection_confidence": {
                "15_north": "high"  # Should be float
            },
            "metadata": {"version": "1.0", "total_rooms": 1, "total_connections": 0}
        }

        with pytest.raises(ValueError, match="Invalid confidence type"):
            MapGraph.from_dict(invalid_data)

    def test_confidence_out_of_range_rejected(self):
        """Test that confidence scores outside [0.0, 1.0] are rejected."""
        # Test value > 1.0
        invalid_data_high = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": []}
            },
            "connections": {},
            "connection_confidence": {
                "15_north": 1.5  # Out of range
            },
            "metadata": {"version": "1.0", "total_rooms": 1, "total_connections": 0}
        }

        with pytest.raises(ValueError, match="Confidence out of range"):
            MapGraph.from_dict(invalid_data_high)

        # Test value < 0.0
        invalid_data_low = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": []}
            },
            "connections": {},
            "connection_confidence": {
                "15_north": -0.5  # Out of range
            },
            "metadata": {"version": "1.0", "total_rooms": 1, "total_connections": 0}
        }

        with pytest.raises(ValueError, match="Confidence out of range"):
            MapGraph.from_dict(invalid_data_low)

    def test_negative_verification_count_rejected(self):
        """Test that negative verification counts are rejected."""
        invalid_data = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": []}
            },
            "connections": {},
            "connection_verifications": {
                "15_north": -5  # Negative count
            },
            "metadata": {"version": "1.0", "total_rooms": 1, "total_connections": 0}
        }

        with pytest.raises(ValueError, match="cannot be negative"):
            MapGraph.from_dict(invalid_data)

    def test_negative_failure_count_rejected(self):
        """Test that negative failure counts are rejected."""
        invalid_data = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": []}
            },
            "connections": {},
            "exit_failure_counts": {
                "15_west": -3  # Negative count
            },
            "metadata": {"version": "1.0", "total_rooms": 1, "total_connections": 0}
        }

        with pytest.raises(ValueError, match="cannot be negative"):
            MapGraph.from_dict(invalid_data)

    def test_non_string_exits_rejected(self):
        """Test that non-string exit names are rejected."""
        invalid_data = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": ["north", 123, "south"]}  # 123 is not a string
            },
            "connections": {},
            "metadata": {"version": "1.0", "total_rooms": 1, "total_connections": 0}
        }

        with pytest.raises(ValueError, match="non-string exits"):
            MapGraph.from_dict(invalid_data)

    def test_connection_from_nonexistent_source_room_rejected(self):
        """Test that connections from non-existent source rooms are rejected."""
        invalid_data = {
            "rooms": {
                "15": {"id": 15, "name": "Test Room", "exits": []},
                "20": {"id": 20, "name": "Dest Room", "exits": []}
            },
            "connections": {
                "999": {"north": 20}  # Source room 999 doesn't exist
            },
            "metadata": {"version": "1.0", "total_rooms": 2, "total_connections": 1}
        }

        with pytest.raises(ValueError, match="source room.*does not exist"):
            MapGraph.from_dict(invalid_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
