"""
ABOUTME: Unit tests for ObjectiveManager Phase 1 enhancement (helper methods).
ABOUTME: Tests the 7 new helper methods added to ObjectiveManager without modifying existing behavior.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, mock_open, patch
from dataclasses import dataclass

from managers.objective_manager import ObjectiveManager
from managers.simple_memory_manager import Memory
from session.game_state import GameState
from session.game_configuration import GameConfiguration


@dataclass
class MockLocation:
    """Mock Z-machine location structure."""
    num: int
    name: str


class TestObjectiveManagerEnhanced:
    """Test suite for Phase 1 ObjectiveManager enhancement - helper methods only."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock configuration."""
        config = Mock(spec=GameConfiguration)
        config.zork_game_workdir = str(tmp_path)
        config.knowledge_file = "knowledgebase.md"
        return config

    @pytest.fixture
    def mock_game_state(self):
        """Create mock game state."""
        state = Mock(spec=GameState)
        state.current_room_id = 180
        state.current_room_name = "West of House"
        state.turn_count = 10
        state.action_history = []
        return state

    @pytest.fixture
    def mock_adaptive_knowledge_manager(self):
        """Create mock adaptive knowledge manager."""
        return Mock()

    @pytest.fixture
    def mock_map_manager(self):
        """Create mock map manager with game_map."""
        manager = Mock()
        manager.game_map = Mock()
        manager.game_map.rooms = {180: Mock(), 79: Mock(), 62: Mock()}
        manager.game_map.room_names = {
            180: "West of House",
            79: "Behind House",
            62: "Kitchen"
        }
        manager.game_map.connections = {
            180: {"north": 79},
            79: {"south": 180, "enter window": 62},
            62: {}
        }
        manager.game_map.render_mermaid = Mock(return_value="```mermaid\ngraph TD\nL180-->L79\n```")
        return manager

    @pytest.fixture
    def mock_simple_memory(self):
        """Create mock simple memory manager."""
        manager = Mock()
        manager.memory_cache = {}
        return manager

    @pytest.fixture
    def objective_manager(self, mock_logger, mock_config, mock_game_state,
                          mock_adaptive_knowledge_manager, mock_map_manager,
                          mock_simple_memory):
        """Create ObjectiveManager with all dependencies."""
        return ObjectiveManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            adaptive_knowledge_manager=mock_adaptive_knowledge_manager,
            map_manager=mock_map_manager,
            simple_memory=mock_simple_memory
        )

    # Test 1: _get_full_knowledge - loads file
    def test_get_full_knowledge_loads_file(self, objective_manager, tmp_path):
        """Should load knowledge base from file."""
        kb_path = tmp_path / "knowledgebase.md"
        kb_content = "# Strategic Knowledge\n\nAvoid the troll at Location 152."
        kb_path.write_text(kb_content, encoding="utf-8")

        result = objective_manager._get_full_knowledge()

        assert result == kb_content
        assert "Strategic Knowledge" in result
        assert "troll" in result

    # Test 2: _get_full_knowledge - handles missing file
    def test_get_full_knowledge_handles_missing_file(self, objective_manager, tmp_path):
        """Should return fallback message if file missing."""
        # File doesn't exist
        result = objective_manager._get_full_knowledge()

        assert result == "No strategic knowledge available."

    # Test 3: _calculate_distance_bfs - same location
    def test_calculate_distance_bfs_same_location(self, objective_manager):
        """Should return 0 for same location."""
        distance = objective_manager._calculate_distance_bfs(180, 180)
        assert distance == 0

    # Test 4: _calculate_distance_bfs - adjacent locations
    def test_calculate_distance_bfs_adjacent(self, objective_manager):
        """Should return 1 for directly connected rooms."""
        distance = objective_manager._calculate_distance_bfs(180, 79)
        assert distance == 1

    # Test 5: _calculate_distance_bfs - multi-hop path
    def test_calculate_distance_bfs_multi_hop(self, objective_manager):
        """Should calculate correct distance for multi-hop paths."""
        # 180 → 79 → 62 (2 hops)
        distance = objective_manager._calculate_distance_bfs(180, 62)
        assert distance == 2

    # Test 6: _calculate_distance_bfs - unreachable location
    def test_calculate_distance_bfs_unreachable(self, objective_manager, mock_map_manager):
        """Should return float('inf') for unreachable locations."""
        # Add isolated location 999 with no connections
        mock_map_manager.game_map.rooms[999] = Mock()
        mock_map_manager.game_map.connections[999] = {}

        distance = objective_manager._calculate_distance_bfs(180, 999)
        assert distance == float('inf')

    # Test 7: _calculate_distance_bfs - no map_manager
    def test_calculate_distance_bfs_no_map_manager(self, objective_manager):
        """Should return float('inf') when map_manager is None."""
        objective_manager.map_manager = None
        distance = objective_manager._calculate_distance_bfs(180, 79)
        assert distance == float('inf')

    # Test 8: _get_all_memories_by_distance - filtering ACTIVE only
    def test_get_all_memories_by_distance_filtering(self, objective_manager, mock_simple_memory):
        """Should filter to ACTIVE memories only."""
        mock_simple_memory.memory_cache = {
            180: [
                Memory("SUCCESS", "Active memory", 1, "10", 0, "This is active.", status="ACTIVE", persistence="permanent"),
                Memory("DANGER", "Tentative memory", 1, "11", 0, "This is tentative.", status="TENTATIVE", persistence="permanent"),
                Memory("NOTE", "Superseded memory", 1, "12", 0, "This is superseded.", status="SUPERSEDED", persistence="permanent"),
            ]
        }

        result = objective_manager._get_all_memories_by_distance(180)

        assert "Active memory" in result
        assert "Tentative memory" not in result
        assert "Superseded memory" not in result

    # Test 9: _get_all_memories_by_distance - sorting by distance
    def test_get_all_memories_by_distance_sorting(self, objective_manager, mock_simple_memory):
        """Should sort locations by distance (closest first)."""
        mock_simple_memory.memory_cache = {
            62: [Memory("NOTE", "Far memory", 1, "5", 0, "Kitchen memory.", status="ACTIVE", persistence="permanent")],
            79: [Memory("NOTE", "Mid memory", 1, "6", 0, "Behind house memory.", status="ACTIVE", persistence="permanent")],
            180: [Memory("NOTE", "Near memory", 1, "7", 0, "West house memory.", status="ACTIVE", persistence="permanent")],
        }

        result = objective_manager._get_all_memories_by_distance(180)

        # Check order: 180 (0 hops), 79 (1 hop), 62 (2 hops)
        lines = result.split('\n')
        location_lines = [line for line in lines if line.startswith('**Location')]

        assert "Location 180" in location_lines[0]
        assert "0 hops away" in location_lines[0]
        assert "Location 79" in location_lines[1]
        assert "1 hops away" in location_lines[1]
        assert "Location 62" in location_lines[2]
        assert "2 hops away" in location_lines[2]

    # Test 10: _get_all_memories_by_distance - no cutoffs
    def test_get_all_memories_by_distance_no_cutoffs(self, objective_manager, mock_simple_memory, mock_map_manager):
        """Should include all locations with strategic memories (no arbitrary limits)."""
        # Create 10 locations with memories
        for i in range(1, 11):
            loc_id = 100 + i
            mock_map_manager.game_map.rooms[loc_id] = Mock()
            mock_map_manager.game_map.room_names[loc_id] = f"Room {i}"
            mock_simple_memory.memory_cache[loc_id] = [
                Memory("NOTE", f"Memory {i}", 1, str(i), 0, f"Text {i}", status="ACTIVE", persistence="permanent")
            ]
            # Chain connections: 180 → 101 → 102 → ... → 110
            if i == 1:
                mock_map_manager.game_map.connections[180]["east"] = loc_id
            else:
                prev_loc_id = 100 + i - 1
                if prev_loc_id not in mock_map_manager.game_map.connections:
                    mock_map_manager.game_map.connections[prev_loc_id] = {}
                mock_map_manager.game_map.connections[prev_loc_id]["east"] = loc_id
            mock_map_manager.game_map.connections[loc_id] = {}

        result = objective_manager._get_all_memories_by_distance(180)

        # Should include all 10 locations
        for i in range(1, 11):
            assert f"Memory {i}" in result

    # Test 11: _get_all_memories_by_distance - no dependencies
    def test_get_all_memories_by_distance_no_dependencies(self, objective_manager):
        """Should return fallback when dependencies missing."""
        objective_manager.simple_memory = None
        result = objective_manager._get_all_memories_by_distance(180)
        assert result == "No memory data available"

        # Reset and test with no map_manager
        objective_manager.simple_memory = Mock()
        objective_manager.map_manager = None
        result = objective_manager._get_all_memories_by_distance(180)
        assert result == "No memory data available"

    # Test 12: _format_memories - shows status markers
    def test_format_memories_shows_status(self, objective_manager):
        """Should show status markers for non-ACTIVE memories."""
        memories = [
            Memory("SUCCESS", "Active memory", 1, "10", 0, "Active text.", status="ACTIVE", persistence="permanent"),
            Memory("DANGER", "Tentative memory", 1, "11", 0, "Tentative text.", status="TENTATIVE", persistence="permanent"),
        ]

        result = objective_manager._format_memories(memories)

        # ACTIVE should not have status marker
        assert "[SUCCESS] Active memory" in result
        assert "[SUCCESS] Active memory [ACTIVE]" not in result

        # TENTATIVE should have status marker
        assert "[DANGER] Tentative memory [TENTATIVE]" in result

    # Test 13: _format_memories - limits to 5 memories
    def test_format_memories_limits_to_five(self, objective_manager):
        """Should show only top 5 memories per location."""
        memories = [
            Memory("NOTE", f"Memory {i}", 1, str(i), 0, f"Text {i}", status="ACTIVE", persistence="permanent")
            for i in range(10)
        ]

        result = objective_manager._format_memories(memories)

        # Should include first 5
        for i in range(5):
            assert f"Memory {i}" in result

        # Should not include last 5
        for i in range(5, 10):
            assert f"Memory {i}" not in result

    # Test 14: _format_memories - handles empty list
    def test_format_memories_handles_empty_list(self, objective_manager):
        """Should return fallback for empty memory list."""
        result = objective_manager._format_memories([])
        assert result == "  (No memories)"

    # Test 15: _get_map_context - includes Mermaid diagram
    def test_get_map_context_includes_mermaid(self, objective_manager):
        """Should include Mermaid diagram in context."""
        result = objective_manager._get_map_context()

        assert "## Map Visualization (Mermaid Format)" in result
        assert "```mermaid" in result
        assert "graph TD" in result

    # Test 16: _get_map_context - includes current location
    def test_get_map_context_includes_current_location(self, objective_manager):
        """Should show current location in map context."""
        result = objective_manager._get_map_context()

        assert "**Current location**: L180" in result
        assert "Current location: West of House (ID: 180)" in result

    # Test 17: _get_map_context - includes exploration stats
    def test_get_map_context_includes_exploration_stats(self, objective_manager):
        """Should include exploration statistics."""
        result = objective_manager._get_map_context()

        assert "## Exploration Statistics" in result
        assert "Rooms discovered: 3" in result

    # Test 18: _get_map_context - no map_manager
    def test_get_map_context_no_map_manager(self, objective_manager):
        """Should return fallback when map_manager is None."""
        objective_manager.map_manager = None
        result = objective_manager._get_map_context()
        assert result == "No map data available"

    # Test 19: _get_routing_summary - shows current location exits
    def test_get_routing_summary_current_location(self, objective_manager):
        """Should show exits from current location only."""
        result = objective_manager._get_routing_summary(180)

        assert "## Current Location: 180 (West of House)" in result
        assert "**Available Exits:**" in result
        assert "north → Location 79 (Behind House)" in result

    # Test 20: _get_routing_summary - no exits
    def test_get_routing_summary_no_exits(self, objective_manager):
        """Should handle locations with no mapped exits."""
        result = objective_manager._get_routing_summary(62)

        assert "## Current Location: 62 (Kitchen)" in result
        assert "- No mapped exits" in result

    # Test 21: _get_routing_summary - no map_manager
    def test_get_routing_summary_no_map_manager(self, objective_manager):
        """Should return fallback when map_manager is None."""
        objective_manager.map_manager = None
        result = objective_manager._get_routing_summary(180)
        assert result == "No routing data available"

    # Test 22: _get_gameplay_context - formats recent actions
    def test_get_gameplay_context_formats_recent_actions(self, objective_manager, mock_game_state):
        """Should format recent action/response pairs."""
        from session.game_state import ActionHistoryEntry
        mock_game_state.action_history = [
            ActionHistoryEntry(action="go north", response="You are in a forest.", location_id=10, location_name="Test"),
            ActionHistoryEntry(action="examine tree", response="The tree is an oak tree.", location_id=11, location_name="Test 2"),
        ]
        mock_game_state.turn_count = 10

        result = objective_manager._get_gameplay_context()

        assert "## Recent Actions (Last 10 Turns)" in result
        assert "Turn 9:" in result
        assert "Action: go north" in result
        assert "Response: You are in a forest." in result
        assert "Turn 10:" in result
        assert "Action: examine tree" in result

    # Test 23: _get_gameplay_context - truncates long responses
    def test_get_gameplay_context_truncates_long_responses(self, objective_manager, mock_game_state):
        """Should truncate responses longer than 200 characters."""
        from session.game_state import ActionHistoryEntry
        long_response = "A" * 250
        mock_game_state.action_history = [ActionHistoryEntry(action="look", response=long_response, location_id=10, location_name="Test")]
        mock_game_state.turn_count = 10

        result = objective_manager._get_gameplay_context()

        assert "A" * 200 in result
        assert "..." in result
        assert len(long_response) > 200  # Original was longer
        # Response in result should be truncated
        response_line = [line for line in result.split('\n') if line.strip().startswith("Response:")][0]
        assert len(response_line) < len("Response: ") + len(long_response)

    # Test 24: _get_gameplay_context - handles empty history
    def test_get_gameplay_context_handles_empty_history(self, objective_manager, mock_game_state):
        """Should handle empty action history gracefully."""
        mock_game_state.action_history = []

        result = objective_manager._get_gameplay_context()

        assert "## Recent Actions" in result
        assert "(No actions yet - start of episode)" in result

    # Test 25: _get_gameplay_context - limits to 10 actions
    def test_get_gameplay_context_limits_to_ten_actions(self, objective_manager, mock_game_state):
        """Should include only last 10 actions."""
        from session.game_state import ActionHistoryEntry
        mock_game_state.action_history = [
            ActionHistoryEntry(action=f"step_{i:03d}", response=f"result_{i:03d}", location_id=i, location_name=f"Loc {i}") for i in range(20)
        ]
        mock_game_state.turn_count = 20

        result = objective_manager._get_gameplay_context()

        # Should include last 10 (actions 10-19)
        for i in range(10, 20):
            assert f"step_{i:03d}" in result

        # Should not include first 10 (actions 0-9)
        for i in range(10):
            assert f"step_{i:03d}" not in result

    # Test 26: Integration - all helpers work together
    def test_integration_all_helpers(self, objective_manager, mock_simple_memory, tmp_path):
        """Test that all helper methods work together without errors."""
        # Setup knowledge file
        kb_path = tmp_path / "knowledgebase.md"
        kb_path.write_text("# Knowledge\nAvoid troll.", encoding="utf-8")

        # Setup memories
        mock_simple_memory.memory_cache = {
            180: [Memory("NOTE", "Test memory", 1, "5", 0, "Test text.", status="ACTIVE", persistence="permanent")]
        }

        # Call all helpers
        knowledge = objective_manager._get_full_knowledge()
        memories = objective_manager._get_all_memories_by_distance(180)
        map_context = objective_manager._get_map_context()
        gameplay = objective_manager._get_gameplay_context()

        # All should return strings without errors
        assert isinstance(knowledge, str)
        assert isinstance(memories, str)
        assert isinstance(map_context, str)
        assert isinstance(gameplay, str)

        # Verify content
        assert "Avoid troll" in knowledge
        assert "Test memory" in memories
        assert "Mermaid" in map_context
        assert "Recent Actions" in gameplay

    # Test 27: ObjectiveManager initialization with new dependencies
    def test_initialization_with_dependencies(self, mock_logger, mock_config,
                                               mock_game_state, mock_adaptive_knowledge_manager,
                                               mock_map_manager, mock_simple_memory):
        """Should initialize with new optional dependencies."""
        manager = ObjectiveManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            adaptive_knowledge_manager=mock_adaptive_knowledge_manager,
            map_manager=mock_map_manager,
            simple_memory=mock_simple_memory
        )

        assert manager.map_manager is mock_map_manager
        assert manager.simple_memory is mock_simple_memory
        assert manager.adaptive_knowledge_manager is mock_adaptive_knowledge_manager

    # Test 28: ObjectiveManager initialization without optional dependencies
    def test_initialization_without_optional_dependencies(self, mock_logger, mock_config,
                                                          mock_game_state,
                                                          mock_adaptive_knowledge_manager):
        """Should initialize without optional dependencies (backwards compatible)."""
        manager = ObjectiveManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            adaptive_knowledge_manager=mock_adaptive_knowledge_manager
        )

        assert manager.map_manager is None
        assert manager.simple_memory is None
