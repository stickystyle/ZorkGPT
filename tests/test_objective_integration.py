"""
ABOUTME: Integration tests for ObjectiveManager Phase 2 - orchestrator dependency injection.
ABOUTME: Tests that ObjectiveManager works with MapManager and SimpleMemoryManager dependencies.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
import tempfile
import shutil

from managers.objective_manager import ObjectiveManager
from managers.simple_memory_manager import SimpleMemoryManager, Memory
from managers.map_manager import MapManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class TestObjectiveManagerOrchestrationIntegration:
    """Integration tests for ObjectiveManager with real dependencies."""

    @pytest.fixture
    def temp_workdir(self):
        """Create temporary working directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def mock_config(self, temp_workdir):
        """Create mock configuration."""
        config = Mock(spec=GameConfiguration)
        config.zork_game_workdir = str(temp_workdir)
        config.knowledge_file = "knowledgebase.md"
        config.map_state_file = "map_state.json"
        config.objective_check_interval = 10
        config.objective_update_interval = 10  # Missing attribute causing test failure
        return config

    @pytest.fixture
    def game_state(self):
        """Create real GameState instance."""
        state = GameState()
        state.current_room_id = 180
        state.current_room_name = "West of House"
        state.turn_count = 10
        state.action_history = []
        return state

    @pytest.fixture
    def map_manager(self, mock_logger, mock_config, game_state):
        """Create real MapManager instance."""
        return MapManager(
            logger=mock_logger,
            config=mock_config,
            game_state=game_state
        )

    @pytest.fixture
    def simple_memory(self, mock_logger, mock_config, game_state):
        """Create real SimpleMemoryManager instance (without LLM client)."""
        # Create without LLM client for testing
        memory_mgr = SimpleMemoryManager.__new__(SimpleMemoryManager)
        memory_mgr.logger = mock_logger
        memory_mgr.config = mock_config
        memory_mgr.game_state = game_state
        memory_mgr._llm_client = None  # Use private attribute to bypass property
        memory_mgr.memory_cache = {}
        memory_mgr.pending_actions = []
        memory_mgr.synthesis_cooldown = 0
        return memory_mgr

    @pytest.fixture
    def objective_manager(self, mock_logger, mock_config, game_state,
                          map_manager, simple_memory):
        """Create ObjectiveManager with real dependencies."""
        return ObjectiveManager(
            logger=mock_logger,
            config=mock_config,
            game_state=game_state,
            adaptive_knowledge_manager=Mock(),
            map_manager=map_manager,
            simple_memory=simple_memory
        )

    # Test 1: ObjectiveManager receives dependencies
    def test_objective_manager_receives_dependencies(self, objective_manager, map_manager, simple_memory):
        """ObjectiveManager should receive map_manager and simple_memory dependencies."""
        # Verify dependencies were passed
        assert objective_manager.map_manager is not None
        assert objective_manager.simple_memory is not None

        # Verify they're the same instances we created
        assert objective_manager.map_manager is map_manager
        assert objective_manager.simple_memory is simple_memory

    # Test 2: Verify dependency types
    def test_dependency_types(self, objective_manager):
        """Dependencies should be correct types."""
        assert isinstance(objective_manager.map_manager, MapManager)
        assert isinstance(objective_manager.simple_memory, SimpleMemoryManager)

    # Test 3: Helper methods work with real dependencies
    def test_helper_methods_with_real_dependencies(self, objective_manager, map_manager, simple_memory):
        """Helper methods should work with actual MapManager and SimpleMemoryManager instances."""
        # Add rooms to the map
        map_manager.add_initial_room(180, "West of House")
        map_manager.update_from_movement(
            action_taken="north",
            new_room_id=79,
            new_room_name="Behind House",
            previous_room_id=180,
            previous_room_name="West of House",
            game_response="Behind House\nYou are behind the white house."
        )

        # Add a memory
        simple_memory.memory_cache[180] = [
            Memory(
                category="SUCCESS",
                title="Lamp acquired",
                episode=1,
                turns="5",
                score_change=5,
                text="Successfully picked up the lamp at West of House.",
                status="ACTIVE",
                persistence="permanent"
            )
        ]

        # Test _get_map_context() - should return map data
        map_context = objective_manager._get_map_context()
        assert isinstance(map_context, str)
        assert len(map_context) > 0
        assert "West of House" in map_context or "180" in map_context

        # Test _get_all_memories_by_distance() - should return memories
        memories = objective_manager._get_all_memories_by_distance(180)
        assert isinstance(memories, str)
        assert len(memories) > 0
        # Should contain the memory we added (filtered to ACTIVE only)
        assert "Lamp acquired" in memories or "SUCCESS" in memories

        # Test _get_routing_summary() - should return routing info
        routing = objective_manager._get_routing_summary(180)
        assert isinstance(routing, str)
        assert len(routing) > 0
        assert "180" in routing or "West of House" in routing

    # Test 5: Helper methods gracefully handle missing dependencies
    def test_graceful_degradation_without_dependencies(self):
        """ObjectiveManager should work even if dependencies are None (backward compatibility)."""
        # Create ObjectiveManager directly without dependencies
        mock_logger = Mock()
        mock_config = Mock()
        mock_config.zork_game_workdir = "/tmp/test"
        mock_config.knowledge_file = "knowledgebase.md"
        mock_config.objective_check_interval = 10

        mock_game_state = Mock()
        mock_game_state.current_room_id = 180
        mock_game_state.turn_count = 10
        mock_game_state.action_history = []

        mock_adaptive_knowledge = Mock()

        # Initialize WITHOUT map_manager and simple_memory
        objective_mgr = ObjectiveManager(
            logger=mock_logger,
            config=mock_config,
            game_state=mock_game_state,
            adaptive_knowledge_manager=mock_adaptive_knowledge,
            map_manager=None,  # Explicitly None
            simple_memory=None  # Explicitly None
        )

        # Verify dependencies are None
        assert objective_mgr.map_manager is None
        assert objective_mgr.simple_memory is None

        # Verify helper methods return fallback messages
        assert objective_mgr._get_map_context() == "No map data available"
        assert objective_mgr._get_all_memories_by_distance(180) == "No memory data available"
        assert objective_mgr._get_routing_summary(180) == "No routing data available"

    # Test 4: Integration - real map data flows through helper methods
    def test_real_map_data_integration(self, objective_manager, map_manager):
        """Test that real map data from MapManager flows through ObjectiveManager helpers."""
        # Build a small map
        map_manager.add_initial_room(180, "West of House")
        map_manager.update_from_movement(
            action_taken="north",
            new_room_id=79,
            new_room_name="Behind House",
            previous_room_id=180,
            previous_room_name="West of House",
            game_response="You are behind the house."
        )
        map_manager.update_from_movement(
            action_taken="enter window",
            new_room_id=62,
            new_room_name="Kitchen",
            previous_room_id=79,
            previous_room_name="Behind House",
            game_response="You are in a kitchen."
        )

        # Test distance calculation - adjacent rooms
        distance_adjacent = objective_manager._calculate_distance_bfs(180, 79)
        assert distance_adjacent >= 0, "Should find a path between adjacent rooms"

        # Test map context includes rooms
        map_context = objective_manager._get_map_context()
        assert "West of House" in map_context or "180" in map_context
        # Map context should include some rooms
        assert len(map_context) > 100, "Map context should have substantive content"

    # Test 5: Integration - real memory data flows through helper methods
    def test_real_memory_data_integration(self, objective_manager, map_manager, simple_memory):
        """Test that real memory data from SimpleMemoryManager flows through ObjectiveManager helpers."""
        # Add memories at different locations
        simple_memory.memory_cache[180] = [
            Memory("SUCCESS", "Opened mailbox", 1, "1", 0, "Mailbox contains leaflet.", status="ACTIVE", persistence="permanent")
        ]
        simple_memory.memory_cache[79] = [
            Memory("DANGER", "Window warning", 1, "2", 0, "Window is tricky.", status="ACTIVE", persistence="permanent")
        ]

        # Setup minimal map for distance calculation
        map_manager.add_initial_room(180, "West of House")
        map_manager.update_from_movement(
            action_taken="north",
            new_room_id=79,
            new_room_name="Behind House",
            previous_room_id=180,
            previous_room_name="West of House",
            game_response="Behind House"
        )

        # Test memory retrieval
        memories = objective_manager._get_all_memories_by_distance(180)

        # Should contain both memories, sorted by distance
        assert "Opened mailbox" in memories
        assert "Window warning" in memories

        # Should show location info
        assert "180" in memories or "West of House" in memories
        assert "79" in memories or "Behind House" in memories

    # Test 6: Verify no existing tests were broken
    def test_backward_compatibility(self, objective_manager):
        """Verify that adding new dependencies doesn't break existing functionality."""
        # Test that all original attributes still exist
        assert hasattr(objective_manager, 'logger')
        assert hasattr(objective_manager, 'config')
        assert hasattr(objective_manager, 'game_state')
        assert hasattr(objective_manager, 'adaptive_knowledge_manager')

        # Test that original methods still work
        assert hasattr(objective_manager, 'should_process_turn')
        assert hasattr(objective_manager, 'check_objective_completion')
        assert hasattr(objective_manager, 'process_periodic_updates')
        assert hasattr(objective_manager, 'reset_episode')
        assert hasattr(objective_manager, 'get_status')

        # Test a simple operation
        should_process = objective_manager.should_process_turn()
        assert isinstance(should_process, bool)


class TestObjectiveManagerPhase3EnhancedContext:
    """Phase 3 tests: Enhanced prompt with knowledge, memories, and map context."""

    @pytest.fixture
    def temp_workdir(self):
        """Create temporary working directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def mock_config(self, temp_workdir):
        """Create mock configuration with knowledge file."""
        config = Mock(spec=GameConfiguration)
        config.zork_game_workdir = str(temp_workdir)
        config.knowledge_file = "knowledgebase.md"
        config.map_state_file = "map_state.json"
        config.objective_update_interval = 20
        return config

    @pytest.fixture
    def game_state(self):
        """Create real GameState instance with action history."""
        state = GameState()
        state.current_room_id = 180
        state.current_room_name = "West of House"
        state.turn_count = 20
        state.objective_update_turn = 0
        state.action_history = [
            ("north", "Behind House\nYou are behind the white house."),
            ("enter window", "You can't reach the window."),
            ("open window", "With some effort, you open the window."),
        ]
        return state

    @pytest.fixture
    def map_manager(self, mock_logger, mock_config, game_state):
        """Create MapManager with test data."""
        mgr = MapManager(logger=mock_logger, config=mock_config, game_state=game_state)
        # Add test rooms
        mgr.add_initial_room(180, "West of House")
        mgr.update_from_movement(
            action_taken="north",
            new_room_id=79,
            new_room_name="Behind House",
            previous_room_id=180,
            previous_room_name="West of House",
            game_response="Behind House"
        )
        mgr.update_from_movement(
            action_taken="enter window",
            new_room_id=62,
            new_room_name="Kitchen",
            previous_room_id=79,
            previous_room_name="Behind House",
            game_response="Kitchen"
        )
        return mgr

    @pytest.fixture
    def simple_memory(self, mock_logger, mock_config, game_state):
        """Create SimpleMemoryManager with test memories."""
        memory_mgr = SimpleMemoryManager.__new__(SimpleMemoryManager)
        memory_mgr.logger = mock_logger
        memory_mgr.config = mock_config
        memory_mgr.game_state = game_state
        memory_mgr._llm_client = None
        memory_mgr.memory_cache = {}
        memory_mgr.pending_actions = []
        memory_mgr.synthesis_cooldown = 0

        # Add test memories at different locations
        memory_mgr.memory_cache[79] = [
            Memory(
                category="SUCCESS",
                title="Window entry procedure",
                episode=1,
                turns="47-49",
                score_change=0,
                text="To enter kitchen: (1) open window, (2) enter window. Window must be opened first.",
                status="ACTIVE",
                persistence="permanent"
            )
        ]
        memory_mgr.memory_cache[62] = [
            Memory(
                category="DISCOVERY",
                title="Kitchen has food",
                episode=1,
                turns="50",
                score_change=0,
                text="Kitchen contains sack of lunch. Could be useful for troll.",
                status="ACTIVE",
                persistence="permanent"
            )
        ]
        return memory_mgr

    @pytest.fixture
    def knowledge_file(self, temp_workdir):
        """Create test knowledge file."""
        kb_path = Path(temp_workdir) / "knowledgebase.md"
        kb_path.write_text(
            """# Strategic Knowledge

## Dangers
- **Troll at Location 152**: Blocks passage. Requires sword or lunch offering.

## Priorities
- Light source (lantern) is critical for dark areas
- Treasures increase score

## Procedures
- Window entry: open before enter
"""
        )
        return kb_path

    @pytest.fixture
    def objective_manager(self, mock_logger, mock_config, game_state,
                          map_manager, simple_memory, knowledge_file):
        """Create ObjectiveManager with all dependencies and test data."""
        return ObjectiveManager(
            logger=mock_logger,
            config=mock_config,
            game_state=game_state,
            adaptive_knowledge_manager=Mock(),
            map_manager=map_manager,
            simple_memory=simple_memory
        )

    # Test 1: Context gathering includes all sections
    def test_context_gathering_includes_all_sections(self, objective_manager):
        """Verify that context gathering returns all expected sections."""
        knowledge = objective_manager._get_full_knowledge()
        memories = objective_manager._get_all_memories_by_distance(180)
        map_context = objective_manager._get_map_context()
        gameplay = objective_manager._get_gameplay_context()

        # All should return non-empty strings
        assert isinstance(knowledge, str) and len(knowledge) > 0
        assert isinstance(memories, str) and len(memories) > 0
        assert isinstance(map_context, str) and len(map_context) > 0
        assert isinstance(gameplay, str) and len(gameplay) > 0

    # Test 2: Knowledge content is loaded
    def test_knowledge_content_loaded(self, objective_manager):
        """Knowledge content should include file contents."""
        knowledge = objective_manager._get_full_knowledge()

        # Should contain strategic knowledge from file
        assert "Troll at Location 152" in knowledge
        assert "lantern" in knowledge or "light source" in knowledge
        assert "Priorities" in knowledge or "Dangers" in knowledge

    # Test 3: Memories include location IDs
    def test_memories_include_location_ids(self, objective_manager):
        """Memories should be formatted with location IDs."""
        memories = objective_manager._get_all_memories_by_distance(180)

        # Should include location IDs (79, 62 from test data)
        assert "79" in memories or "Behind House" in memories
        assert "62" in memories or "Kitchen" in memories

    # Test 4: Map context includes current location
    def test_map_context_includes_current_location(self, objective_manager, game_state):
        """Map context should highlight current location."""
        map_context = objective_manager._get_map_context()

        # Should mention current location
        assert str(game_state.current_room_id) in map_context
        assert "West of House" in map_context or str(game_state.current_room_id) in map_context

    # Test 5: Gameplay context includes recent actions
    def test_gameplay_context_includes_recent_actions(self, objective_manager, game_state):
        """Gameplay context should show recent action history."""
        gameplay = objective_manager._get_gameplay_context()

        # Should include recent actions from game_state
        assert "north" in gameplay or "open window" in gameplay
        assert "Behind House" in gameplay or "window" in gameplay

    # Test 6: Token counting works
    def test_token_counting_works(self, objective_manager):
        """Verify token counting for context sections."""
        from shared_utils import estimate_tokens

        knowledge = objective_manager._get_full_knowledge()
        memories = objective_manager._get_all_memories_by_distance(180)
        map_context = objective_manager._get_map_context()
        gameplay = objective_manager._get_gameplay_context()

        # All should have positive token counts
        assert estimate_tokens(knowledge) > 0
        assert estimate_tokens(memories) > 0
        assert estimate_tokens(map_context) > 0
        assert estimate_tokens(gameplay) > 0

    # Test 7: Memory filtering (ACTIVE only)
    def test_memory_filtering_active_only(self, simple_memory, objective_manager):
        """Only ACTIVE memories should be included in context."""
        # Add TENTATIVE and SUPERSEDED memories
        simple_memory.memory_cache[180] = [
            Memory("NOTE", "Active memory", 1, "1", 0, "This is active.", status="ACTIVE", persistence="permanent"),
            Memory("NOTE", "Tentative memory", 1, "2", 0, "This is tentative.", status="TENTATIVE", persistence="permanent"),
            Memory("NOTE", "Superseded memory", 1, "3", 0, "This is superseded.", status="SUPERSEDED", persistence="permanent"),
        ]

        memories = objective_manager._get_all_memories_by_distance(180)

        # Should only include ACTIVE
        assert "Active memory" in memories
        assert "Tentative memory" not in memories
        assert "Superseded memory" not in memories
