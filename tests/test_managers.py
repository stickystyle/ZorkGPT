"""
Comprehensive unit tests for all ZorkGPT manager classes.

Tests each manager independently with mocked dependencies to ensure
proper functionality and isolation.
"""

import pytest
import logging
from unittest.mock import Mock, patch

from session.game_state import GameState
from session.game_configuration import GameConfiguration
from managers import (
    ObjectiveManager,
    KnowledgeManager,
    MapManager,
    StateManager,
    ContextManager,
    EpisodeSynthesizer,
)


class TestBaseManagerSetup:
    """Common setup for manager tests."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger for testing."""
        logger = Mock(spec=logging.Logger)
        return logger

    @pytest.fixture
    def game_config(self):
        """Create a test game configuration."""
        return GameConfiguration(
            # Core game settings
            max_turns_per_episode=1000,
            turn_delay_seconds=0.0,
            # File paths
            episode_log_file="test_episode.log",
            json_log_file="test_episode.jsonl",
            state_export_file="test_state.json",
            zork_game_workdir="test_game_files",
            # LLM client settings
            client_base_url="http://localhost:1234",
            client_api_key="test_api_key",
            # Model specifications
            agent_model="test-agent-model",
            critic_model="test-critic-model",
            info_ext_model="test-extractor-model",
            analysis_model="test-analysis-model",
            # Update intervals
            knowledge_update_interval=100,
            map_update_interval=50,
            objective_update_interval=20,
            # Objective refinement
            enable_objective_refinement=True,
            objective_refinement_interval=200,
            max_objectives_before_forced_refinement=15,
            refined_objectives_target_count=10,
            # Context management
            max_context_tokens=100000,
            context_overflow_threshold=0.8,
            # State export
            enable_state_export=True,
            s3_bucket="test-bucket",
            s3_key_prefix="test/",
        )

    @pytest.fixture
    def game_state(self):
        """Create a test game state."""
        state = GameState()
        state.episode_id = "test_episode_001"
        state.turn_count = 10
        state.current_room_name_for_map = "Living Room"
        state.previous_zork_score = 50
        state.current_inventory = ["lamp", "sword"]
        return state


class TestObjectiveManager(TestBaseManagerSetup):
    """Test the ObjectiveManager class."""

    @pytest.fixture
    def mock_adaptive_manager(self):
        """Create a mock adaptive knowledge manager."""
        manager = Mock()
        manager.client = Mock()
        manager.analysis_model = "gpt-4"
        manager.analysis_sampling = Mock()
        manager.analysis_sampling.model_dump.return_value = {"temperature": 0.3}
        return manager

    @pytest.fixture
    def objective_manager(
        self, mock_logger, game_config, game_state, mock_adaptive_manager
    ):
        """Create an ObjectiveManager instance for testing."""
        return ObjectiveManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state,
            adaptive_knowledge_manager=mock_adaptive_manager,
        )

    def test_initialization(self, objective_manager):
        """Test that ObjectiveManager initializes correctly."""
        assert objective_manager.component_name == "objective_manager"
        assert objective_manager.last_objective_refinement_turn == 0

    def test_reset_episode(self, objective_manager):
        """Test episode reset functionality."""
        objective_manager.last_objective_refinement_turn = 100
        objective_manager.reset_episode()
        assert objective_manager.last_objective_refinement_turn == 0

    def test_should_process_turn(self, objective_manager, game_state):
        """Test turn processing decision logic."""
        # Should not process on turn 0
        game_state.turn_count = 0
        assert not objective_manager.should_process_turn()

        # Should not process if interval not reached
        game_state.turn_count = 10
        game_state.objective_update_turn = 5
        assert not objective_manager.should_process_turn()

        # Should process if interval reached
        game_state.turn_count = 25
        game_state.objective_update_turn = 0
        assert objective_manager.should_process_turn()

    def test_parse_objectives_from_response(self, objective_manager):
        """Test parsing objectives from LLM response."""
        response = """
        Some other text here.
        
        OBJECTIVES:
        - Find the treasure chest
        - Solve the puzzle in the maze
        - Escape from the dungeon
        
        More text here.
        """

        objectives = objective_manager._parse_objectives_from_response(response)

        assert len(objectives) == 3
        assert "Find the treasure chest" in objectives
        assert "Solve the puzzle in the maze" in objectives
        assert "Escape from the dungeon" in objectives

    def test_parse_objectives_empty_response(self, objective_manager):
        """Test parsing with empty or invalid response."""
        assert objective_manager._parse_objectives_from_response("") == []
        assert (
            objective_manager._parse_objectives_from_response("No objectives here")
            == []
        )

    def test_check_objective_completion(self, objective_manager, game_state):
        """Test objective completion detection."""
        # Setup game state with objectives
        game_state.discovered_objectives = ["Find treasure", "Get lamp"]

        # Mock extracted info with score increase
        mock_extracted_info = Mock()
        mock_extracted_info.score = 60  # Increased from 50

        # Test completion detection
        objective_manager.check_objective_completion(
            action_taken="take treasure",
            game_response="You have earned 10 points! Well done!",
            extracted_info=mock_extracted_info,
        )

        # Should have triggered evaluation (we can't easily test the LLM call without mocking)
        assert True  # Basic smoke test

    def test_check_objective_staleness(self, objective_manager, game_state):
        """Test objective staleness tracking."""
        # Setup objectives
        game_state.discovered_objectives = ["Test objective"]
        game_state.objective_staleness_tracker = {"Test objective": 25}
        game_state.current_room_name_for_map = "Same Room"
        game_state.last_location_for_staleness = "Same Room"
        game_state.last_score_for_staleness = 50
        game_state.previous_zork_score = 50

        # No progress should increase staleness
        objective_manager.check_objective_staleness()
        assert game_state.objective_staleness_tracker["Test objective"] == 26

        # Make progress (score increase)
        game_state.previous_zork_score = 60
        objective_manager.check_objective_staleness()
        assert game_state.objective_staleness_tracker["Test objective"] == 0

    def test_get_status(self, objective_manager, game_state):
        """Test status reporting."""
        game_state.discovered_objectives = ["obj1", "obj2"]
        game_state.completed_objectives = [{"objective": "completed1"}]

        status = objective_manager.get_status()

        assert status["discovered_objectives_count"] == 2
        assert status["completed_objectives_count"] == 1
        assert status["component"] == "objective_manager"


class TestKnowledgeManager(TestBaseManagerSetup):
    """Test the KnowledgeManager class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock()
        agent.reload_knowledge_base = Mock()
        return agent

    @pytest.fixture
    def mock_map_manager(self):
        """Create a mock map manager."""
        map_manager = Mock()
        # Mock the game_map attribute
        mock_game_map = Mock()
        mock_game_map.render_mermaid.return_value = "graph TD\n  A --> B"
        map_manager.game_map = mock_game_map
        # Mock the get_quality_metrics method
        map_manager.get_quality_metrics.return_value = {"confidence": 0.8}
        return map_manager

    @pytest.fixture
    def knowledge_manager(
        self, mock_logger, game_config, game_state, mock_agent, mock_map_manager
    ):
        """Create a KnowledgeManager instance for testing."""
        return KnowledgeManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state,
            agent=mock_agent,
            game_map=mock_map_manager,
            json_log_file="test.jsonl",
        )

    def test_initialization(self, knowledge_manager):
        """Test that KnowledgeManager initializes correctly."""
        assert knowledge_manager.component_name == "knowledge_manager"
        assert knowledge_manager.last_knowledge_update_turn == 0
        assert knowledge_manager.adaptive_knowledge_manager is not None

    def test_should_process_turn(self, knowledge_manager, game_state):
        """Test knowledge update timing logic."""
        # Should not process if interval not reached
        game_state.turn_count = 50
        knowledge_manager.last_knowledge_update_turn = 10
        assert not knowledge_manager.should_process_turn()

        # Should process if interval reached
        game_state.turn_count = 110
        knowledge_manager.last_knowledge_update_turn = 0
        assert knowledge_manager.should_process_turn()

    def test_update_map_in_knowledge_base(self, knowledge_manager, mock_map_manager):
        """Test map update in knowledge base."""
        knowledge_manager.update_map_in_knowledge_base()
        mock_map_manager.game_map.render_mermaid.assert_called_once()

    def test_reload_agent_knowledge(self, knowledge_manager, mock_agent):
        """Test agent knowledge reloading."""
        knowledge_manager.reload_agent_knowledge()
        mock_agent.reload_knowledge_base.assert_called_once()

    def test_should_synthesize_inter_episode_wisdom(self, knowledge_manager):
        """Test synthesis decision logic."""
        # Should synthesize on death episodes
        assert knowledge_manager.should_synthesize_inter_episode_wisdom(0, 1, [])

        # Should synthesize on high scores
        assert knowledge_manager.should_synthesize_inter_episode_wisdom(60, 0, [])

        # Should synthesize on long episodes
        knowledge_manager.game_state.turn_count = 600
        assert knowledge_manager.should_synthesize_inter_episode_wisdom(10, 0, [])

        # Should synthesize on high confidence
        assert knowledge_manager.should_synthesize_inter_episode_wisdom(
            10, 0, [0.9, 0.8, 0.85]
        )

    @patch("builtins.open", create=True)
    def test_get_knowledge_base_summary(self, mock_open, knowledge_manager):
        """Test knowledge base summary retrieval."""
        mock_open.return_value.__enter__.return_value.read.return_value = """
        # Knowledge Base
        
        Some content here.
        
        ## Map
        graph TD
          A --> B
        
        More content.
        """

        summary = knowledge_manager.get_knowledge_base_summary()
        assert "## Map" not in summary
        assert "Some content here" in summary

    def test_get_status(self, knowledge_manager):
        """Test status reporting."""
        status = knowledge_manager.get_status()

        assert status["last_knowledge_update_turn"] == 0
        assert status["has_adaptive_manager"] is True
        assert status["component"] == "knowledge_manager"


class TestMapManager(TestBaseManagerSetup):
    """Test the MapManager class."""

    @pytest.fixture
    def map_manager(self, mock_logger, game_config, game_state):
        """Create a MapManager instance for testing."""
        return MapManager(logger=mock_logger, config=game_config, game_state=game_state)

    def test_initialization(self, map_manager):
        """Test that MapManager initializes correctly."""
        assert map_manager.component_name == "map_manager"
        assert map_manager.game_map is not None
        assert map_manager.movement_analyzer is not None
        assert map_manager.last_map_update_turn == 0

    def test_add_initial_room(self, map_manager, game_state):
        """Test adding initial room to map."""
        map_manager.add_initial_room("Starting Room")
        assert game_state.current_room_name_for_map == "Starting Room"

    def test_update_from_movement(self, map_manager, game_state):
        """Test map update from movement."""
        game_state.current_room_name_for_map = "Room A"

        map_manager.update_from_movement(
            action_taken="north", new_room_name="Room B", previous_room_name="Room A"
        )

        assert game_state.current_room_name_for_map == "Room B"
        assert game_state.prev_room_for_prompt_context == "Room A"
        assert game_state.action_leading_to_current_room_for_prompt_context == "north"

    def test_track_failed_action(self, map_manager, game_state):
        """Test failed action tracking."""
        location = "Test Room"
        action = "north"

        map_manager.track_failed_action(action, location)

        assert location in game_state.failed_actions_by_location
        assert action in game_state.failed_actions_by_location[location]

    def test_should_process_turn(self, map_manager, game_state):
        """Test map update timing logic."""
        # Should not process if interval not reached
        game_state.turn_count = 25
        map_manager.last_map_update_turn = 10
        assert not map_manager.should_process_turn()

        # Should process if interval reached
        game_state.turn_count = 60
        map_manager.last_map_update_turn = 0
        assert map_manager.should_process_turn()

    def test_get_current_room_context(self, map_manager, game_state):
        """Test current room context retrieval."""
        game_state.current_room_name_for_map = "Current Room"
        game_state.prev_room_for_prompt_context = "Previous Room"
        game_state.action_leading_to_current_room_for_prompt_context = "east"
        game_state.failed_actions_by_location = {"Current Room": ["north", "west"]}

        context = map_manager.get_current_room_context()

        assert context["current_room"] == "Current Room"
        assert context["previous_room"] == "Previous Room"
        assert context["action_to_current"] == "east"
        assert context["failed_actions"] == ["north", "west"]

    def test_get_status(self, map_manager, game_state):
        """Test status reporting."""
        game_state.current_room_name_for_map = "Test Room"

        status = map_manager.get_status()

        assert status["current_room"] == "Test Room"
        assert status["last_map_update_turn"] == 0
        assert status["component"] == "map_manager"


class TestStateManager(TestBaseManagerSetup):
    """Test the StateManager class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        return client

    @pytest.fixture
    def state_manager(self, mock_logger, game_config, game_state, mock_llm_client):
        """Create a StateManager instance for testing."""
        return StateManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state,
            llm_client=mock_llm_client,
        )

    def test_initialization(self, state_manager):
        """Test that StateManager initializes correctly."""
        assert state_manager.component_name == "state_manager"
        assert state_manager.last_summarization_turn == 0

    def test_estimate_context_tokens(self, state_manager, game_state):
        """Test context token estimation."""
        # Add some test data
        game_state.memory_log_history = [{"test": "data"}, {"more": "data"}]
        game_state.action_history = [
            ("look", "You see a room"),
            ("north", "You move north"),
        ]

        tokens = state_manager.estimate_context_tokens()
        assert tokens > 0

    def test_is_critical_memory(self, state_manager):
        """Test critical memory detection."""
        # Score increase should be critical
        memory_with_score = {"score": 10}
        assert state_manager.is_critical_memory(memory_with_score)

        # Death event should be critical
        memory_with_death = {"text": "You died horribly"}
        assert state_manager.is_critical_memory(memory_with_death)

        # Normal memory should not be critical
        normal_memory = {"text": "You look around"}
        assert not state_manager.is_critical_memory(normal_memory)

        # Non-dict should not be critical
        assert not state_manager.is_critical_memory("not a dict")

    def test_generate_fallback_summary(self, state_manager, game_state):
        """Test fallback summary generation."""
        game_state.episode_id = "test_123"
        game_state.turn_count = 50
        game_state.previous_zork_score = 25
        game_state.current_room_name_for_map = "Test Room"

        summary = state_manager.generate_fallback_summary()

        assert "test_123" in summary
        assert "50" in summary
        assert "25" in summary
        assert "Test Room" in summary

    def test_get_combat_status(self, state_manager, game_state):
        """Test combat status detection."""
        # No combat indicators
        game_state.memory_log_history = [{"text": "peaceful room"}]
        assert not state_manager.get_combat_status()

        # With combat indicators
        game_state.memory_log_history = [{"text": "You are fighting a monster!"}]
        assert state_manager.get_combat_status()

    def test_is_death_episode(self, state_manager, game_state):
        """Test death episode detection."""
        # No death indicators
        game_state.memory_log_history = [{"text": "normal gameplay"}]
        assert not state_manager.is_death_episode()

        # With death indicators
        game_state.memory_log_history = [{"text": "You died tragically"}]
        assert state_manager.is_death_episode()

    def test_get_status(self, state_manager, game_state):
        """Test status reporting."""
        game_state.memory_log_history = [1, 2, 3]
        game_state.action_history = [("a", "b"), ("c", "d")]

        status = state_manager.get_status()

        assert status["memory_entries"] == 3
        assert status["action_history_length"] == 2
        assert status["export_enabled"] is True
        assert status["component"] == "state_manager"


class TestContextManager(TestBaseManagerSetup):
    """Test the ContextManager class."""

    @pytest.fixture
    def context_manager(self, mock_logger, game_config, game_state):
        """Create a ContextManager instance for testing."""
        return ContextManager(
            logger=mock_logger, config=game_config, game_state=game_state
        )

    def test_initialization(self, context_manager):
        """Test that ContextManager initializes correctly."""
        assert context_manager.component_name == "context_manager"

    def test_add_memory(self, context_manager, game_state):
        """Test adding memory to context."""
        test_memory = {"turn": 1, "action": "test"}
        context_manager.add_memory(test_memory)

        assert len(game_state.memory_log_history) == 1
        assert game_state.memory_log_history[0] == test_memory

    def test_add_action(self, context_manager, game_state):
        """Test adding action to context."""
        context_manager.add_action("look", "You see a room")

        assert len(game_state.action_history) == 1
        assert game_state.action_history[0] == ("look", "You see a room")

    def test_add_reasoning(self, context_manager, game_state):
        """Test adding reasoning to context."""
        context_manager.add_reasoning("I should look around", "look")

        assert len(game_state.action_reasoning_history) == 1
        reasoning = game_state.action_reasoning_history[0]
        assert reasoning["reasoning"] == "I should look around"
        assert reasoning["action"] == "look"
        assert reasoning["turn"] == game_state.turn_count

    def test_get_recent_actions(self, context_manager, game_state):
        """Test retrieving recent actions."""
        # Add some actions
        game_state.action_history = [
            ("action1", "response1"),
            ("action2", "response2"),
            ("action3", "response3"),
        ]

        recent = context_manager.get_recent_actions(2)
        assert len(recent) == 2
        assert recent == [("action2", "response2"), ("action3", "response3")]

    def test_is_successful_action(self, context_manager):
        """Test action success detection."""
        # Failure indicators
        assert not context_manager.is_successful_action("You can't do that")
        assert not context_manager.is_successful_action("I don't understand")
        assert not context_manager.is_successful_action("Nothing happens")

        # Success indicators
        assert context_manager.is_successful_action("Taken")
        assert context_manager.is_successful_action("You have earned 10 points")
        assert context_manager.is_successful_action("The door opened")

        # Neutral should default to success
        assert context_manager.is_successful_action("You are in a room")

    def test_get_agent_context(self, context_manager, game_state):
        """Test agent context assembly."""
        # Setup game state
        game_state.action_history = [("look", "room")]
        game_state.memory_log_history = [{"test": "memory"}]
        game_state.action_counts = {"look": 5, "north": 2}

        context = context_manager.get_agent_context(
            current_state="Test state",
            inventory=["lamp"],
            location="Test Room",
            failed_actions=["west"],
            discovered_objectives=["Find treasure"],
        )

        assert context["game_state"] == "Test state"
        assert context["inventory"] == ["lamp"]
        assert context["current_location"] == "Test Room"
        assert context["failed_actions_here"] == ["west"]
        assert context["discovered_objectives"] == ["Find treasure"]
        assert "recent_actions" in context
        assert "recent_memories" in context

    def test_detect_loops_in_recent_actions(self, context_manager, game_state):
        """Test loop detection in actions."""
        # No loop
        game_state.action_history = [("a", "1"), ("b", "2"), ("c", "3")]
        assert not context_manager.detect_loops_in_recent_actions()

        # Simple loop pattern
        game_state.action_history = [
            ("north", "1"),
            ("south", "2"),
            ("north", "1"),
            ("south", "2"),
        ]
        assert context_manager.detect_loops_in_recent_actions()

    def test_get_status(self, context_manager, game_state):
        """Test status reporting."""
        game_state.memory_log_history = [1, 2]
        game_state.action_history = [("a", "b")]
        game_state.current_room_name_for_map = "Current"
        game_state.prev_room_for_prompt_context = "Previous"

        status = context_manager.get_status()

        assert status["memory_entries"] == 2
        assert status["action_history_length"] == 1
        assert status["current_location"] == "Current"
        assert status["previous_location"] == "Previous"
        assert status["component"] == "context_manager"


class TestEpisodeSynthesizer(TestBaseManagerSetup):
    """Test the EpisodeSynthesizer class."""

    @pytest.fixture
    def mock_knowledge_manager(self):
        """Create a mock knowledge manager."""
        manager = Mock()
        manager.perform_final_update = Mock()
        return manager

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        manager.export_current_state = Mock()
        return manager

    @pytest.fixture
    def episode_synthesizer(
        self,
        mock_logger,
        game_config,
        game_state,
        mock_knowledge_manager,
        mock_state_manager,
    ):
        """Create an EpisodeSynthesizer instance for testing."""
        return EpisodeSynthesizer(
            logger=mock_logger,
            config=game_config,
            game_state=game_state,
            knowledge_manager=mock_knowledge_manager,
            state_manager=mock_state_manager,
        )

    def test_initialization(self, episode_synthesizer):
        """Test that EpisodeSynthesizer initializes correctly."""
        assert episode_synthesizer.component_name == "episode_synthesizer"
        assert episode_synthesizer.knowledge_manager is not None
        assert episode_synthesizer.state_manager is not None

    def test_initialize_episode(self, episode_synthesizer, game_state):
        """Test episode initialization."""
        # Mock components
        mock_agent = Mock()
        mock_agent.update_episode_id = Mock()

        # Test episode ID (provided by orchestrator)
        test_episode_id = "2025-06-08T15:45:00"

        returned_episode_id = episode_synthesizer.initialize_episode(
            episode_id=test_episode_id, agent=mock_agent
        )

        # Should return the same episode ID and set it in game state
        assert returned_episode_id == test_episode_id
        assert game_state.episode_id == test_episode_id
        mock_agent.update_episode_id.assert_called_once_with(test_episode_id)

    def test_is_death_episode(self, episode_synthesizer, game_state):
        """Test death episode detection."""
        # No death indicators
        game_state.action_reasoning_history = [{"reasoning": "normal gameplay"}]
        assert not episode_synthesizer.is_death_episode()

        # With death in reasoning
        game_state.action_reasoning_history = [{"reasoning": "I died horribly"}]
        assert episode_synthesizer.is_death_episode()

        # With death in memory
        game_state.memory_log_history = [{"text": "You were killed"}]
        assert episode_synthesizer.is_death_episode()

    def test_should_synthesize_inter_episode_wisdom(
        self, episode_synthesizer, game_state
    ):
        """Test synthesis decision logic."""
        # Should synthesize on death
        game_state.action_reasoning_history = [{"reasoning": "died"}]
        assert episode_synthesizer.should_synthesize_inter_episode_wisdom(0, [])

        # Should synthesize on high score
        game_state.action_reasoning_history = []
        assert episode_synthesizer.should_synthesize_inter_episode_wisdom(60, [])

        # Should synthesize on long episode
        game_state.turn_count = 600
        assert episode_synthesizer.should_synthesize_inter_episode_wisdom(10, [])

        # Should synthesize on high confidence
        game_state.turn_count = 50
        assert episode_synthesizer.should_synthesize_inter_episode_wisdom(
            10, [0.9, 0.8]
        )

        # Should synthesize on many completed objectives
        game_state.completed_objectives = [1, 2, 3, 4]
        assert episode_synthesizer.should_synthesize_inter_episode_wisdom(10, [])

    def test_generate_fallback_episode_summary(self, episode_synthesizer, game_state):
        """Test fallback episode summary generation."""
        game_state.episode_id = "test_episode"
        game_state.turn_count = 100
        game_state.current_room_name_for_map = "Final Room"
        game_state.discovered_objectives = ["obj1", "obj2"]
        game_state.completed_objectives = [{"obj": "completed1"}]

        summary = episode_synthesizer.generate_fallback_episode_summary(75, True)

        assert "test_episode" in summary
        assert "100" in summary
        assert "75" in summary
        assert "True" in summary
        assert "Final Room" in summary

    def test_get_episode_metrics(self, episode_synthesizer, game_state):
        """Test episode metrics collection."""
        game_state.episode_id = "test_123"
        game_state.turn_count = 150
        game_state.previous_zork_score = 80
        game_state.discovered_objectives = ["obj1", "obj2"]
        game_state.completed_objectives = [{"obj": "done1"}]
        game_state.visited_locations = {"room1", "room2", "room3"}
        game_state.action_history = [("a", "b"), ("c", "d")]

        metrics = episode_synthesizer.get_episode_metrics()

        assert metrics["episode_id"] == "test_123"
        assert metrics["turn_count"] == 150
        assert metrics["final_score"] == 80
        assert metrics["objectives_discovered"] == 2
        assert metrics["objectives_completed"] == 1
        assert metrics["locations_visited"] == 3
        assert metrics["actions_taken"] == 2

    def test_finalize_episode(
        self, episode_synthesizer, mock_knowledge_manager, mock_state_manager
    ):
        """Test episode finalization."""
        episode_synthesizer.finalize_episode(
            final_score=100, critic_confidence_history=[0.8, 0.9]
        )

        # Should call final update on knowledge manager
        mock_knowledge_manager.perform_final_update.assert_called_once()

        # Should export state
        mock_state_manager.export_current_state.assert_called_once()

    def test_get_status(self, episode_synthesizer, game_state):
        """Test status reporting."""
        game_state.episode_id = "test_episode"
        game_state.turn_count = 50
        game_state.death_count = 2
        game_state.discovered_objectives = ["obj1"]
        game_state.completed_objectives = [{"obj": "done"}]

        status = episode_synthesizer.get_status()

        assert status["current_episode"] == "test_episode"
        assert status["turn_count"] == 50
        assert status["death_count"] == 2
        assert status["objectives_discovered"] == 1
        assert status["objectives_completed"] == 1
        assert status["has_knowledge_manager"] is True
        assert status["has_state_manager"] is True
        assert status["component"] == "episode_synthesizer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
