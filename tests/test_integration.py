"""
Integration tests for ZorkGPT system.

Tests the full system integration including orchestrator coordination,
manager interactions, and end-to-end workflows.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from orchestration import ZorkOrchestratorV2
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class TestZorkOrchestratorV2Integration:
    """Integration tests for the complete ZorkOrchestrator v2 system."""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield {
                "episode_log": os.path.join(tmpdir, "episode.log"),
                "json_log": os.path.join(tmpdir, "episode.jsonl"),
                "state_export": os.path.join(tmpdir, "state.json"),
                "knowledge_base": os.path.join(tmpdir, "knowledgebase.md")
            }
    
    @pytest.fixture
    def mock_game_server_client(self):
        """Create a mock game server client for testing."""
        client = Mock()
        
        # Mock successful session start
        client.start_session.return_value = {"success": True, "session_id": "test_session"}
        client.stop_session.return_value = {"success": True}
        
        # Mock game responses
        responses = [
            {"success": True, "response": "You are in a white house. There is a mailbox here."},
            {"success": True, "response": "You moved north. You are in a forest."},
            {"success": True, "response": "You see a lamp here."},
            {"success": True, "response": "Taken. You now have a lamp."},
            {"success": True, "response": "You have earned 10 points!"},
        ]
        client.send_command.side_effect = responses
        
        return client
    
    @pytest.fixture
    def mock_llm_responses(self):
        """Create mock LLM responses for testing."""
        return {
            "agent_action": {
                "action": "take lamp",
                "reasoning": "I should take the lamp as it might be useful for lighting dark areas."
            },
            "critic_evaluation": {
                "action": "take lamp",
                "confidence": 0.9,
                "reasoning": "Taking the lamp is a good idea for future exploration."
            },
            "objective_update": "OBJECTIVES:\n- Find treasure\n- Explore the house\n- Get light source",
            "knowledge_summary": "The player successfully found and took a lamp, earning 10 points."
        }
    
    @pytest.fixture
    def orchestrator(self, temp_files):
        """Create an orchestrator instance with test configuration."""
        return ZorkOrchestratorV2(
            episode_log_file=temp_files["episode_log"],
            json_log_file=temp_files["json_log"],
            state_export_file=temp_files["state_export"],
            max_turns_per_episode=10,  # Keep test episodes short
            knowledge_update_interval=5,
            map_update_interval=3,
            objective_update_interval=2,
            enable_state_export=True,
            turn_delay_seconds=0.0,  # No delay in tests
            game_server_url="http://localhost:8000"
        )
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test that the orchestrator initializes all components correctly."""
        # Check core components
        assert orchestrator.agent is not None
        assert orchestrator.critic is not None
        assert orchestrator.extractor is not None
        
        # Check managers
        assert orchestrator.map_manager is not None
        assert orchestrator.context_manager is not None
        assert orchestrator.state_manager is not None
        assert orchestrator.objective_manager is not None
        assert orchestrator.knowledge_manager is not None
        assert orchestrator.episode_synthesizer is not None
        
        # Check configuration
        assert orchestrator.config.max_turns_per_episode == 10
        assert orchestrator.config.enable_state_export is True
        
        # Check game state
        assert orchestrator.game_state is not None
        assert isinstance(orchestrator.game_state, GameState)
    
    def test_manager_dependencies(self, orchestrator):
        """Test that managers have correct dependencies."""
        # Knowledge manager should have agent and map references
        assert orchestrator.knowledge_manager.agent is not None
        assert orchestrator.knowledge_manager.game_map is not None
        
        # Objective manager should have adaptive knowledge manager
        assert orchestrator.objective_manager.adaptive_knowledge_manager is not None
        
        # Episode synthesizer should have manager references
        assert orchestrator.episode_synthesizer.knowledge_manager is not None
        assert orchestrator.episode_synthesizer.state_manager is not None
        
        # All managers should share the same game state
        for manager in orchestrator.managers:
            assert manager.game_state is orchestrator.game_state
    
    @patch('zork_agent.ZorkAgent')
    @patch('zork_critic.ZorkCritic')
    @patch('hybrid_zork_extractor.HybridZorkExtractor')
    def test_episode_workflow(self, mock_extractor_class, mock_critic_class, mock_agent_class, 
                             orchestrator, mock_game_server_client, mock_llm_responses):
        """Test a complete episode workflow."""
        # Setup mocks
        mock_agent = Mock()
        mock_agent.get_action.return_value = mock_llm_responses["agent_action"]
        mock_agent.client = Mock()
        mock_agent_class.return_value = mock_agent
        
        mock_critic = Mock()
        mock_critic.evaluate_action.return_value = mock_llm_responses["critic_evaluation"]
        mock_critic_class.return_value = mock_critic
        
        mock_extractor = Mock()
        mock_extracted_info = Mock()
        mock_extracted_info.current_location_name = "White House"
        mock_extracted_info.inventory = ["lamp"]
        mock_extracted_info.score = 10
        mock_extracted_info.game_over = False
        mock_extractor.extract_info.return_value = mock_extracted_info
        mock_extractor_class.return_value = mock_extractor
        
        # Re-initialize orchestrator with mocks
        orchestrator._initialize_game_components()
        orchestrator._initialize_managers()
        
        # Run episode
        final_score = orchestrator.play_episode(mock_game_server_client)
        
        # Verify episode ran
        assert final_score >= 0
        assert orchestrator.game_state.turn_count > 0
        # Episode ID should be in ISO8601 format (YYYY-MM-DDTHH:MM:SS)
        import re
        assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', orchestrator.game_state.episode_id)
        
        # Verify game server interactions
        mock_game_server_client.start_session.assert_called_once()
        mock_game_server_client.stop_session.assert_called_once()
        assert mock_game_server_client.send_command.call_count > 0
        
        # Verify component interactions
        assert mock_agent.get_action.call_count > 0
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
        # (Most testing here is ensuring no exceptions are raised)
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
            failed_actions=orchestrator.game_state.failed_actions_by_location.get("Living Room", []),
            discovered_objectives=orchestrator.game_state.discovered_objectives
        )
        
        # Verify context structure
        assert context["game_state"] == "Test game state"
        assert context["current_location"] == "Living Room"
        assert context["inventory"] == ["lamp"]
        assert context["failed_actions_here"] == ["west"]
        assert context["discovered_objectives"] == ["Find treasure"]
        assert "recent_actions" in context
        assert "recent_memories" in context
    
    @patch('builtins.open', create=True)
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
        mock_open.assert_called_with(orchestrator.config.state_export_file, 'w')
    
    def test_episode_synthesis(self, orchestrator):
        """Test episode synthesis functionality."""
        # Setup for synthesis
        orchestrator.game_state.episode_id = "synthesis_test"
        orchestrator.game_state.turn_count = 100
        orchestrator.game_state.previous_zork_score = 75
        orchestrator.game_state.discovered_objectives = ["obj1", "obj2"]
        orchestrator.game_state.completed_objectives = [{"objective": "completed1"}]
        
        # Test synthesis decision
        should_synthesize = orchestrator.episode_synthesizer.should_synthesize_inter_episode_wisdom(
            final_score=75,
            critic_confidence_history=[0.8, 0.9, 0.7]
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
    
    def test_error_handling(self, orchestrator, mock_game_server_client):
        """Test system behavior with errors."""
        # Mock game server failure
        mock_game_server_client.start_session.return_value = {"success": False, "error": "Connection failed"}
        
        # Should handle gracefully and return 0 score
        final_score = orchestrator.play_episode(mock_game_server_client)
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
            "MapManager", "ContextManager", "StateManager", 
            "ObjectiveManager", "KnowledgeManager", "EpisodeSynthesizer"
        ]
        
        for manager_name in manager_names:
            assert manager_name in status["managers"]
            manager_status = status["managers"][manager_name]
            assert "component" in manager_status
            assert "turn" in manager_status
            assert "episode_id" in manager_status
    
    def test_multiple_episodes(self, orchestrator, mock_game_server_client):
        """Test running multiple episodes sequentially."""
        # Mock successful short episodes
        mock_game_server_client.send_command.side_effect = [
            {"success": True, "response": "Game over! You won!"}
        ] * 10  # Enough responses for multiple episodes
        
        with patch.object(orchestrator.episode_synthesizer, 'initialize_episode') as mock_init:
            mock_init.side_effect = ["episode_1", "episode_2"]
            
            with patch.object(orchestrator, '_run_game_loop') as mock_game_loop:
                mock_game_loop.return_value = 50  # Fixed score
                
                # Run 2 episodes
                scores = orchestrator.run_multiple_episodes(2)
                
                # Verify both episodes ran
                assert len(scores) == 2
                assert all(score == 50 for score in scores)
                assert mock_init.call_count == 2
    
    @patch('boto3.client')
    def test_s3_integration(self, mock_boto_client, orchestrator):
        """Test S3 integration for state export."""
        # Setup S3 configuration
        orchestrator.config.s3_bucket = "test-bucket"
        orchestrator.state_manager.s3_client = mock_boto_client.return_value
        
        # Test state upload
        test_state = {"episode_id": "test", "score": 100}
        success = orchestrator.state_manager.upload_state_to_s3(test_state)
        
        # Verify S3 upload was attempted
        assert success is True
        mock_boto_client.return_value.put_object.assert_called_once()


class TestManagerInteractions:
    """Test interactions between different managers."""
    
    @pytest.fixture
    def orchestrator_components(self):
        """Create orchestrator components for interaction testing."""
        config = GameConfiguration(
            knowledge_update_interval=5,
            map_update_interval=3,
            objective_update_interval=2
        )
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
        mock_adaptive_manager.analysis_sampling = Mock()
        mock_adaptive_manager.analysis_sampling.model_dump.return_value = {}
        
        # Create objective manager
        from managers import ObjectiveManager
        objective_manager = ObjectiveManager(
            logger=mock_logger,
            config=config,
            game_state=game_state,
            adaptive_knowledge_manager=mock_adaptive_manager
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
        
        # Simulate movement
        map_manager.update_from_movement("north", "New Room", "Old Room")
        
        # Update context with movement
        context_manager.update_location_context("Old Room", "New Room", "north")
        
        # Verify state consistency
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
            mock_logger, config, game_state, 
            state_manager=state_manager
        )
        
        # Simulate episode data
        game_state.turn_count = 100
        game_state.previous_zork_score = 75
        game_state.discovered_objectives = ["Find treasure"]
        
        # Test episode finalization
        episode_synthesizer.finalize_episode(final_score=75)
        
        # Verify state manager was called for export
        # (In real scenario, state_manager.export_current_state would be called)
        assert episode_synthesizer.state_manager is not None


class TestRealDfrotzIntegration:
    """Test with real dfrotz process for score parsing validation."""
    
    def test_real_score_parsing_with_dfrotz(self):
        """Test that score parsing works correctly with real dfrotz output.
        
        Executes the sequence: south, east, open window, enter window, take sack
        which should result in a score of 10 points.
        """
        import requests
        import time
        from hybrid_zork_extractor import HybridZorkExtractor
        from llm_client import LLMClient
        from session.game_configuration import GameConfiguration
        
        # Check if game server is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Game server not running - start with 'docker-compose up -d'")
        except requests.exceptions.RequestException:
            pytest.skip("Game server not running - start with 'docker-compose up -d'")
        
        # Start a new game session
        session_response = requests.post("http://localhost:8000/sessions/score_test_session")
        assert session_response.status_code == 200
        session_data = session_response.json()
        assert "session_id" in session_data
        
        try:
            # Initialize extractor for score parsing
            import logging
            logger = logging.getLogger("test_extractor")
            config = GameConfiguration()
            extractor = HybridZorkExtractor(logger=logger)
            
            # Execute the scoring sequence
            commands = ["south", "east", "open window", "enter window", "take sack"]
            
            for command in commands:
                # Send command
                command_response = requests.post("http://localhost:8000/sessions/score_test_session/command",
                                               json={"command": command})
                assert command_response.status_code == 200
                
                response_data = command_response.json()
                assert "raw_response" in response_data
                
                # Brief pause between commands
                time.sleep(0.1)
            
            # Get final game state and check score
            final_response = requests.post("http://localhost:8000/sessions/score_test_session/command",
                                         json={"command": "score"})
            assert final_response.status_code == 200
            
            final_data = final_response.json()
            final_output = final_data["raw_response"]
            
            # Extract score using the real extractor
            extracted_info = extractor.extract_info(final_output)
            
            # Verify score is 10
            assert extracted_info.score == 10, f"Expected score 10, got {extracted_info.score}. Output: {final_output}"
            
            # Verify score parsing worked by checking the output contains score info
            assert "10" in final_output.lower() or "ten" in final_output.lower()
            
        finally:
            # Clean up session
            requests.delete("http://localhost:8000/sessions/score_test_session")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])