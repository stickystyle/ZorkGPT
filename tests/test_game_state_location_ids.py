"""
ABOUTME: Unit tests for GameState location ID fields migration.
ABOUTME: Verifies Phase 3.1 changes for Jericho integer location IDs.
"""
import pytest
from session.game_state import GameState


class TestGameStateLocationIDs:
    """Test the new location ID fields in GameState."""

    def test_initial_state_has_location_fields(self):
        """Test that GameState initializes with both location fields."""
        state = GameState()

        # New fields should exist and be initialized to zero/empty
        assert hasattr(state, "current_room_id")
        assert hasattr(state, "current_room_name")
        assert state.current_room_id == 0
        assert state.current_room_name == ""

        # Deprecated field should still exist for backward compatibility
        assert hasattr(state, "current_room_name_for_map")
        assert state.current_room_name_for_map == ""

    def test_location_fields_are_correct_types(self):
        """Test that location fields have correct types."""
        state = GameState()
        state.current_room_id = 42
        state.current_room_name = "West of House"

        assert isinstance(state.current_room_id, int)
        assert isinstance(state.current_room_name, str)

    def test_reset_episode_clears_location_fields(self):
        """Test that reset_episode() properly resets both location fields."""
        state = GameState()

        # Set some values
        state.current_room_id = 42
        state.current_room_name = "West of House"
        state.current_room_name_for_map = "West of House"

        # Reset episode
        state.reset_episode("test_episode")

        # Both fields should be reset
        assert state.current_room_id == 0
        assert state.current_room_name == ""
        assert state.current_room_name_for_map == ""

    def test_export_data_includes_location_fields(self):
        """Test that get_export_data() includes both location fields."""
        state = GameState()
        state.episode_id = "test_episode"
        state.current_room_id = 42
        state.current_room_name = "West of House"

        export = state.get_export_data()

        # Verify export structure includes both fields
        assert "game_state" in export
        assert "current_room_id" in export["game_state"]
        assert "current_room" in export["game_state"]  # backward compatible key name

        # Verify values
        assert export["game_state"]["current_room_id"] == 42
        assert export["game_state"]["current_room"] == "West of House"

    def test_location_id_is_primary_key(self):
        """Test that location ID can be used as integer primary key."""
        state = GameState()

        # Location IDs should be usable as dictionary keys, graph node IDs, etc.
        location_map = {}

        # Set location
        state.current_room_id = 1
        state.current_room_name = "West of House"
        location_map[state.current_room_id] = state.current_room_name

        # Change location
        state.current_room_id = 2
        state.current_room_name = "Behind House"
        location_map[state.current_room_id] = state.current_room_name

        # Verify we can look up by ID
        assert location_map[1] == "West of House"
        assert location_map[2] == "Behind House"
        assert len(location_map) == 2

    def test_backward_compatibility_field_exists(self):
        """Test that deprecated field still exists for migration period."""
        state = GameState()

        # Old field should still be present
        assert hasattr(state, "current_room_name_for_map")

        # It can be set independently (during migration)
        state.current_room_id = 42
        state.current_room_name = "West of House"
        state.current_room_name_for_map = "West of House"

        # All three can coexist
        assert state.current_room_id == 42
        assert state.current_room_name == "West of House"
        assert state.current_room_name_for_map == "West of House"
