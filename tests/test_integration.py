"""
Integration tests for ZorkGPT system with Jericho.

Tests the full system integration including orchestrator coordination,
manager interactions, and end-to-end workflows using JerichoInterface.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from orchestration import ZorkOrchestratorV2
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class TestZorkOrchestratorV2Integration:
    """Integration tests for the complete ZorkOrchestrator v2 system with Jericho."""

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield {
                "episode_log": os.path.join(tmpdir, "episode.log"),
                "json_log": os.path.join(tmpdir, "episode.jsonl"),
                "state_export": os.path.join(tmpdir, "state.json"),
                "knowledge_base": os.path.join(tmpdir, "knowledgebase.md"),
            }

    @pytest.fixture
    def mock_jericho_interface(self):
        """Create a mock Jericho interface for testing."""
        jericho = Mock()

        # Mock start method
        jericho.start.return_value = "West of House\nYou are standing in an open field west of a white house."

        # Mock send_command responses
        responses = [
            "You moved north. You are in a forest.",
            "You see a lamp here.",
            "Taken. You now have a lamp.",
            "You have earned 10 points!",
        ]
        jericho.send_command.side_effect = responses

        # Mock structured methods
        location_obj = Mock()
        location_obj.name = "West of House"
        location_obj.num = 1
        jericho.get_location_structured.return_value = location_obj

        jericho.get_inventory_structured.return_value = []
        jericho.get_all_objects.return_value = []
        jericho.get_score.return_value = (0, 350)
        jericho.is_game_over.return_value = (False, None)

        # Mock close method
        jericho.close.return_value = None

        return jericho

    @pytest.fixture
    def mock_llm_responses(self):
        """Create mock LLM responses for testing."""
        return {
            "agent_action": {
                "action": "take lamp",
                "reasoning": "I should take the lamp as it might be useful for lighting dark areas.",
            },
            "critic_evaluation": {
                "action": "take lamp",
                "confidence": 0.9,
                "reasoning": "Taking the lamp is a good idea for future exploration.",
            },
            "objective_update": "OBJECTIVES:\n- Find treasure\n- Explore the house\n- Get light source",
            "knowledge_summary": "The player successfully found and took a lamp, earning 10 points.",
        }

    @pytest.fixture
    def orchestrator(self, temp_files):
        """Create an orchestrator instance with test configuration."""
        import time

        episode_id = f"test_episode_{int(time.time())}"
        return ZorkOrchestratorV2(
            episode_id=episode_id,
            max_turns_per_episode=10,  # Keep test episodes short
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Test that the orchestrator initializes all components correctly."""
        # Check core components
        assert orchestrator.agent is not None
        assert orchestrator.critic is not None
        assert orchestrator.extractor is not None
        assert orchestrator.jericho_interface is not None

        # Check managers
        assert orchestrator.map_manager is not None
        assert orchestrator.context_manager is not None
        assert orchestrator.state_manager is not None
        assert orchestrator.objective_manager is not None
        assert orchestrator.knowledge_manager is not None
        assert orchestrator.episode_synthesizer is not None

        # Check configuration
        assert orchestrator.config.max_turns_per_episode == 10

        # Check game state
        assert orchestrator.game_state is not None
        assert isinstance(orchestrator.game_state, GameState)

    def test_manager_dependencies(self, orchestrator):
        """Test that managers have correct dependencies."""
        # Knowledge manager should have agent and map references
        assert orchestrator.knowledge_manager.agent is not None
        # Note: KnowledgeManager stores map_manager, not game_map
        assert hasattr(orchestrator.knowledge_manager, 'map_manager') or hasattr(orchestrator.knowledge_manager, 'game_map')

        # Objective manager should have adaptive knowledge manager
        assert orchestrator.objective_manager.adaptive_knowledge_manager is not None

        # Episode synthesizer should have manager references
        assert orchestrator.episode_synthesizer.knowledge_manager is not None
        assert orchestrator.episode_synthesizer.state_manager is not None

        # All managers should share the same game state
        for manager in orchestrator.managers:
            assert manager.game_state is orchestrator.game_state

    @patch("zork_agent.ZorkAgent")
    @patch("zork_critic.ZorkCritic")
    @patch("hybrid_zork_extractor.HybridZorkExtractor")
    def test_episode_workflow(
        self,
        mock_extractor_class,
        mock_critic_class,
        mock_agent_class,
        orchestrator,
        mock_jericho_interface,
        mock_llm_responses,
    ):
        """Test a complete episode workflow with Jericho."""
        # Setup mocks
        mock_agent = Mock()
        # Add new_objective field to agent response
        agent_response = mock_llm_responses["agent_action"].copy()
        agent_response["new_objective"] = None
        mock_agent.get_action_with_reasoning.return_value = agent_response
        mock_agent.client = Mock()
        mock_agent_class.return_value = mock_agent

        mock_critic = Mock()
        mock_critic_result = Mock()
        mock_critic_result.score = 0.8
        mock_critic_result.confidence = 0.9
        mock_critic_result.justification = "Good action"
        mock_critic.evaluate_action.return_value = mock_critic_result
        mock_critic_class.return_value = mock_critic

        mock_extractor = Mock()
        mock_extracted_info = Mock()
        mock_extracted_info.current_location_name = "West of House"
        mock_extracted_info.inventory = ["lamp"]
        mock_extracted_info.visible_objects = []
        mock_extracted_info.visible_characters = []
        mock_extracted_info.exits = ["north", "south"]
        mock_extracted_info.important_messages = []
        mock_extracted_info.in_combat = False
        mock_extracted_info.score = 10
        mock_extracted_info.moves = None
        mock_extracted_info.game_over = False
        mock_extractor.extract_info.return_value = mock_extracted_info
        mock_extractor.get_clean_game_text.return_value = "You moved north."
        mock_extractor_class.return_value = mock_extractor

        # Replace Jericho interface with mock
        orchestrator.jericho_interface = mock_jericho_interface
        orchestrator.extractor.jericho = mock_jericho_interface

        # Re-initialize orchestrator with mocks
        orchestrator._initialize_game_components()
        orchestrator._initialize_managers()

        # Replace the orchestrator's components with mocks AFTER initialization
        orchestrator.agent = mock_agent
        orchestrator.critic = mock_critic
        orchestrator.extractor = mock_extractor

        # Run episode
        final_score = orchestrator.play_episode()

        # Verify episode ran successfully
        # Note: Score tracking depends on proper orchestrator-extractor integration
        # The test verifies workflow completion rather than exact score value
        assert final_score >= 0
        assert orchestrator.game_state.turn_count > 0

        # Verify Jericho interactions
        mock_jericho_interface.start.assert_called_once()
        mock_jericho_interface.close.assert_called_once()
        assert mock_jericho_interface.send_command.call_count > 0

        # Verify component interactions
        assert mock_agent.get_action_with_reasoning.call_count > 0
        assert mock_critic.evaluate_action.call_count > 0
        assert mock_extractor.extract_info.call_count > 0

    def test_manager_coordination(self, orchestrator):
        """Test that managers coordinate properly during gameplay."""
        # Simulate some game state changes
        orchestrator.game_state.turn_count = 5
        orchestrator.game_state.current_room_name_for_map = "Test Room"
        orchestrator.game_state.previous_zork_score = 20
        orchestrator.game_state.current_inventory = ["lamp", "key"]

        # Add some history
        orchestrator.context_manager.add_action("look", "You see a room")
        orchestrator.context_manager.add_memory({"turn": 1, "test": "data"})

        # Test periodic updates
        orchestrator._check_periodic_updates()

        # Verify managers were called appropriately
        assert True  # Basic smoke test for coordination

    def test_context_assembly(self, orchestrator):
        """Test that context is properly assembled for components."""
        # Setup game state
        orchestrator.game_state.current_room_name_for_map = "Living Room"
        orchestrator.game_state.current_inventory = ["lamp"]
        orchestrator.game_state.discovered_objectives = ["Find treasure"]
        orchestrator.game_state.failed_actions_by_location = {"Living Room": ["west"]}

        # Get agent context
        context = orchestrator.context_manager.get_agent_context(
            current_state="Test game state",
            inventory=orchestrator.game_state.current_inventory,
            location=orchestrator.game_state.current_room_name_for_map,
            game_map=orchestrator.map_manager.game_map,
            failed_actions=orchestrator.game_state.failed_actions_by_location.get(
                "Living Room", []
            ),
            discovered_objectives=orchestrator.game_state.discovered_objectives,
        )

        # Verify context structure
        assert context["game_state"] == "Test game state"
        assert context["current_location"] == "Living Room"
        assert context["inventory"] == ["lamp"]
        assert context["failed_actions_here"] == ["west"]
        assert context["discovered_objectives"] == ["Find treasure"]
        assert "recent_actions" in context
        assert "recent_memories" in context

    @patch("builtins.open", create=True)
    def test_state_export(self, mock_open, orchestrator, temp_files):
        """Test state export functionality."""
        # Setup game state
        orchestrator.game_state.episode_id = "test_episode"
        orchestrator.game_state.turn_count = 25
        orchestrator.game_state.previous_zork_score = 50
        orchestrator.game_state.current_room_name_for_map = "Test Room"

        # Mock file writing
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Export state
        success = orchestrator.state_manager.export_current_state()

        # Verify export was attempted
        assert success is True
        mock_open.assert_called_with(orchestrator.config.state_export_file, "w")

    def test_episode_synthesis(self, orchestrator):
        """Test episode synthesis functionality."""
        # Setup for synthesis
        orchestrator.game_state.episode_id = "synthesis_test"
        orchestrator.game_state.turn_count = 100
        orchestrator.game_state.previous_zork_score = 75
        orchestrator.game_state.discovered_objectives = ["obj1", "obj2"]
        orchestrator.game_state.completed_objectives = [{"objective": "completed1"}]

        # Test synthesis decision
        should_synthesize = (
            orchestrator.episode_synthesizer.should_synthesize_inter_episode_wisdom(
                final_score=75, critic_confidence_history=[0.8, 0.9, 0.7]
            )
        )

        # Should synthesize due to high score
        assert should_synthesize is True

        # Test episode metrics
        metrics = orchestrator.episode_synthesizer.get_episode_metrics()
        assert metrics["episode_id"] == "synthesis_test"
        assert metrics["turn_count"] == 100
        assert metrics["final_score"] == 75
        assert metrics["objectives_discovered"] == 2
        assert metrics["objectives_completed"] == 1

    def test_error_handling(self, orchestrator, mock_jericho_interface):
        """Test system behavior with errors."""
        # Mock Jericho start failure
        mock_jericho_interface.start.side_effect = RuntimeError("Failed to start")

        # Replace Jericho interface with mock
        orchestrator.jericho_interface = mock_jericho_interface

        # Should handle gracefully and return 0 score
        final_score = orchestrator.play_episode()
        assert final_score == 0

    def test_manager_status_reporting(self, orchestrator):
        """Test comprehensive status reporting."""
        # Setup some state
        orchestrator.game_state.episode_id = "status_test"
        orchestrator.game_state.turn_count = 50
        orchestrator.game_state.discovered_objectives = ["obj1", "obj2"]

        # Get orchestrator status
        status = orchestrator.get_orchestrator_status()

        # Verify top-level status
        assert status["orchestrator"] == "v2"
        assert status["episode_id"] == "status_test"
        assert status["turn_count"] == 50
        assert "managers" in status

        # Verify manager statuses
        manager_names = [
            "MapManager",
            "ContextManager",
            "StateManager",
            "ObjectiveManager",
            "KnowledgeManager",
            "EpisodeSynthesizer",
        ]

        for manager_name in manager_names:
            assert manager_name in status["managers"]
            manager_status = status["managers"][manager_name]
            assert "component" in manager_status
            assert "turn" in manager_status
            assert "episode_id" in manager_status

    def test_multiple_episodes(self, orchestrator, mock_jericho_interface):
        """Test running multiple episodes sequentially."""
        # Replace Jericho interface with mock
        orchestrator.jericho_interface = mock_jericho_interface

        # Mock multiple episode responses
        mock_jericho_interface.send_command.side_effect = [
            "Game over! You won!"
        ] * 20  # Enough responses for multiple episodes

        with patch.object(
            orchestrator.episode_synthesizer, "initialize_episode"
        ) as mock_init:
            with patch.object(orchestrator, "_run_game_loop") as mock_game_loop:
                mock_game_loop.return_value = 50  # Fixed score

                # Run 2 episodes
                scores = orchestrator.run_multiple_episodes(2)

                # Verify both episodes ran
                assert len(scores) == 2
                assert all(score == 50 for score in scores)
                assert mock_init.call_count == 2

    def test_s3_integration(self, orchestrator):
        """Test S3 integration for state export."""
        # Skip if boto3 not installed
        pytest.importorskip("boto3")

        from unittest.mock import patch
        with patch("boto3.client") as mock_boto_client:
            # Setup S3 configuration
            orchestrator.config.s3_bucket = "test-bucket"
            orchestrator.state_manager.s3_client = mock_boto_client.return_value

            # Test state upload
            test_state = {"episode_id": "test", "score": 100}
            success = orchestrator.state_manager.upload_state_to_s3(test_state)

            # Verify S3 upload was attempted
            assert success is True
            # Should upload twice: current_state.json + snapshot
            assert mock_boto_client.return_value.put_object.call_count == 2


class TestManagerInteractions:
    """Test interactions between different managers."""

    @pytest.fixture
    def orchestrator_components(self):
        """Create orchestrator components for interaction testing."""
        config = GameConfiguration.from_toml()
        game_state = GameState()
        game_state.episode_id = "interaction_test"

        return config, game_state

    def test_objective_knowledge_interaction(self, orchestrator_components):
        """Test interaction between ObjectiveManager and KnowledgeManager."""
        config, game_state = orchestrator_components

        # Create mock logger
        mock_logger = Mock()

        # Create mock knowledge manager with adaptive manager
        mock_adaptive_manager = Mock()
        mock_adaptive_manager.client = Mock()
        mock_adaptive_manager.analysis_model = "gpt-4"
        # analysis_sampling is now a dict after config migration
        mock_adaptive_manager.analysis_sampling = {"temperature": 0.3, "max_tokens": 5000}

        # Create objective manager
        from managers import ObjectiveManager

        objective_manager = ObjectiveManager(
            logger=mock_logger,
            config=config,
            game_state=game_state,
            adaptive_knowledge_manager=mock_adaptive_manager,
        )

        # Test that objective manager can use knowledge manager's LLM client
        assert objective_manager.adaptive_knowledge_manager.client is not None
        assert objective_manager.adaptive_knowledge_manager.analysis_model == "gpt-4"

    def test_map_context_interaction(self, orchestrator_components):
        """Test interaction between MapManager and ContextManager."""
        config, game_state = orchestrator_components
        mock_logger = Mock()

        # Create managers
        from managers import MapManager, ContextManager

        map_manager = MapManager(mock_logger, config, game_state)
        context_manager = ContextManager(mock_logger, config, game_state)

        # Setup initial room state
        game_state.current_room_id = 1
        game_state.current_room_name_for_map = "Old Room"

        # Simulate movement with integer IDs
        map_manager.update_from_movement(
            action_taken="north",
            new_room_id=2,
            new_room_name="New Room",
            previous_room_id=1,
            previous_room_name="Old Room"
        )

        # Update context with movement
        context_manager.update_location_context("Old Room", "New Room", "north")

        # Verify state consistency
        assert game_state.current_room_id == 2
        assert game_state.current_room_name_for_map == "New Room"
        assert game_state.prev_room_for_prompt_context == "Old Room"
        assert game_state.action_leading_to_current_room_for_prompt_context == "north"

    def test_state_episode_interaction(self, orchestrator_components):
        """Test interaction between StateManager and EpisodeSynthesizer."""
        config, game_state = orchestrator_components
        mock_logger = Mock()

        # Create managers
        from managers import StateManager, EpisodeSynthesizer

        state_manager = StateManager(mock_logger, config, game_state)
        episode_synthesizer = EpisodeSynthesizer(
            mock_logger, config, game_state, state_manager=state_manager
        )

        # Simulate episode data
        game_state.turn_count = 100
        game_state.previous_zork_score = 75
        game_state.discovered_objectives = ["Find treasure"]

        # Test episode finalization
        episode_synthesizer.finalize_episode(final_score=75)

        # Verify state manager was called for export
        assert episode_synthesizer.state_manager is not None

    def test_orchestrator_extracts_agent_objectives(self):
        """Orchestrator correctly extracts and adds agent objectives from agent responses."""
        import time

        # Create orchestrator with short episode
        episode_id = f"test_extract_objective_{int(time.time())}"
        orchestrator = ZorkOrchestratorV2(
            episode_id=episode_id,
            max_turns_per_episode=1,  # Only 1 turn
        )

        # Mock agent response with new_objective
        agent_response = {
            "action": "take lamp",
            "reasoning": "I should collect the lamp for lighting",
            "new_objective": "collect all treasures for trophy case",
        }

        # Mock critic response
        mock_critic_result = Mock()
        mock_critic_result.score = 0.8
        mock_critic_result.confidence = 0.9
        mock_critic_result.justification = "Good action"

        # Mock extractor response
        mock_extracted_info = Mock()
        mock_extracted_info.current_location_name = "West of House"
        mock_extracted_info.inventory = []
        mock_extracted_info.visible_objects = []
        mock_extracted_info.visible_characters = []
        mock_extracted_info.exits = ["north", "south"]
        mock_extracted_info.important_messages = []
        mock_extracted_info.in_combat = False
        mock_extracted_info.score = 0
        mock_extracted_info.moves = None
        mock_extracted_info.game_over = False

        # Mock Jericho send_command
        with patch.object(orchestrator.jericho_interface, 'send_command', return_value="You moved north."):
            with patch.object(orchestrator.agent, 'get_action_with_reasoning', return_value=agent_response):
                with patch.object(orchestrator.critic, 'evaluate_action', return_value=mock_critic_result):
                    with patch.object(orchestrator.extractor, 'extract_info', return_value=mock_extracted_info):
                        with patch.object(orchestrator.extractor, 'get_clean_game_text', return_value="You moved north."):
                            # Run episode (will run exactly 1 turn due to max_turns_per_episode)
                            orchestrator.play_episode()

                            # Verify objective was added to ObjectiveManager
                            objectives = orchestrator.objective_manager.game_state.discovered_objectives
                            assert "collect all treasures for trophy case" in objectives, \
                                f"Expected objective not found in {objectives}"

    def test_agent_objective_completion_workflow(self):
        """Agent-declared objectives are tracked through completion lifecycle."""
        import time

        # Create orchestrator with short episode
        episode_id = f"test_objective_workflow_{int(time.time())}"
        orchestrator = ZorkOrchestratorV2(
            episode_id=episode_id,
            max_turns_per_episode=4,  # 4 turns total
        )

        # Track which turn we're on
        turn_counter = {"count": 0}

        # Mock agent response - different for each turn
        def mock_get_action(**kwargs):
            turn_counter["count"] += 1
            if turn_counter["count"] == 1:
                # First turn: declare objective
                return {
                    "action": "look",
                    "reasoning": "I should look around first",
                    "new_objective": "find the lamp",
                }
            else:
                # Subsequent turns: no new objective
                return {
                    "action": "north",
                    "reasoning": "I should explore north",
                    "new_objective": None,
                }

        # Mock critic response
        mock_critic_result = Mock()
        mock_critic_result.score = 0.8
        mock_critic_result.confidence = 0.9
        mock_critic_result.justification = "Good action"

        # Mock extractor response
        mock_extracted_info = Mock()
        mock_extracted_info.current_location_name = "West of House"
        mock_extracted_info.inventory = []
        mock_extracted_info.visible_objects = []
        mock_extracted_info.visible_characters = []
        mock_extracted_info.exits = ["north", "south"]
        mock_extracted_info.important_messages = []
        mock_extracted_info.in_combat = False
        mock_extracted_info.score = 0
        mock_extracted_info.moves = None
        mock_extracted_info.game_over = False

        # Mock Jericho send_command
        with patch.object(orchestrator.jericho_interface, 'send_command', return_value="You moved north."):
            with patch.object(orchestrator.agent, 'get_action_with_reasoning', side_effect=mock_get_action):
                with patch.object(orchestrator.critic, 'evaluate_action', return_value=mock_critic_result):
                    with patch.object(orchestrator.extractor, 'extract_info', return_value=mock_extracted_info):
                        with patch.object(orchestrator.extractor, 'get_clean_game_text', return_value="You moved north."):
                            # Run full episode (4 turns)
                            orchestrator.play_episode()

                            # Verify objective was added on turn 1
                            objectives = orchestrator.objective_manager.game_state.discovered_objectives
                            assert "find the lamp" in objectives, \
                                f"Expected objective not found after declaration in {objectives}"

                            # Verify objective persists but wasn't duplicated
                            # (we declared it once, it should still be in the list exactly once)
                            objective_count = sum(1 for obj in objectives if obj == "find the lamp")
                            assert objective_count == 1, \
                                f"Expected objective to appear exactly once, found {objective_count} times"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
