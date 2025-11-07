# ABOUTME: Integration tests for Phase 6 state and object tracking
# ABOUTME: Validates end-to-end integration with orchestrator and game loop

import pytest
from unittest.mock import Mock, MagicMock, patch
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2


class TestPhase6Integration:
    """Integration tests for Phase 6 state and object tracking."""

    @pytest.fixture
    def mock_jericho_interface(self):
        """Create a comprehensive mock Jericho interface."""
        mock = Mock()
        mock.start.return_value = "Welcome to Zork I"
        mock.send_command.return_value = "You are in a forest."
        mock.get_location_structured.return_value = Mock(num=1, name="Forest")
        mock.get_inventory_structured.return_value = []
        mock.get_score.return_value = (0, 350)
        mock.is_game_over.return_value = (False, None)
        mock.save_state.return_value = (1, 2, 3)  # State tuple
        mock.close.return_value = None
        return mock

    def test_state_loop_detection_integration(self, mock_jericho_interface):
        """Test that state loop detection works in orchestrator."""
        with patch(
            "orchestration.zork_orchestrator_v2.JerichoInterface",
            return_value=mock_jericho_interface,
        ):
            # Create orchestrator
            from session.game_configuration import GameConfiguration

            config = GameConfiguration(
                max_turns_per_episode=5,
                turn_delay_seconds=0.0,
                game_file_path="test_game.z5",
                critic_rejection_threshold=0.5,
                episode_log_file="test_episode.log",
                json_log_file="test_episode.jsonl",
                state_export_file="test_state.json",
                zork_game_workdir="test_game_files",
                client_base_url="http://localhost:1234",
                client_api_key="test_api_key",
                agent_model="test-agent-model",
                critic_model="test-critic-model",
                info_ext_model="test-extractor-model",
                analysis_model="test-analysis-model",
                memory_model="test-memory-model",
                knowledge_update_interval=100,
                objective_update_interval=20,
                enable_objective_refinement=True,
                objective_refinement_interval=200,
                max_objectives_before_forced_refinement=15,
                refined_objectives_target_count=10,
                max_context_tokens=100000,
                context_overflow_threshold=0.8,
                enable_state_export=False,
                s3_bucket="test-bucket",
                s3_key_prefix="test/",
                simple_memory_file="Memories.md",
                simple_memory_max_shown=10,
                map_state_file="test_map_state.json",
                knowledge_file="test_knowledgebase.md",
                # Sampling parameters
                agent_sampling={},
                critic_sampling={},
                extractor_sampling={},
                analysis_sampling={},
                memory_sampling={},
            )

            # Mock other components
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-integration")

                # StateManager should have loop detection enabled
                assert orchestrator.state_manager.max_state_history_size == 1000
                assert orchestrator.state_manager.state_history == []

    def test_object_event_tracking_integration(self):
        """Test that object events are tracked during gameplay."""
        # This test would require more complex mocking of the full game loop
        # For now, test that the knowledge manager is properly initialized
        with patch("orchestration.zork_orchestrator_v2.JerichoInterface"):
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-obj-tracking")

                # KnowledgeManager should have object events list
                assert orchestrator.knowledge_manager.object_events == []

    def test_state_and_object_tracking_reset_on_episode_reset(self):
        """Test that both state and object tracking reset on new episode."""
        with patch("orchestration.zork_orchestrator_v2.JerichoInterface"):
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-reset")

                # Add some data
                orchestrator.state_manager.state_history = [123, 456]
                orchestrator.knowledge_manager.object_events = [
                    {"event_type": "acquired", "object_name": "lamp"}
                ]

                # Reset episodes
                orchestrator.state_manager.reset_episode()
                orchestrator.knowledge_manager.reset_episode()

                # Should be cleared
                assert orchestrator.state_manager.state_history == []
                assert orchestrator.knowledge_manager.object_events == []

    def test_state_export_includes_phase6_data(self):
        """Test that state export includes Phase 6 tracking data."""
        with patch("orchestration.zork_orchestrator_v2.JerichoInterface"):
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-export")

                # Add tracking data
                orchestrator.state_manager.state_history = [111, 222]
                orchestrator.knowledge_manager.object_events = [
                    {"event_type": "acquired", "object_name": "lantern", "turn": 5}
                ]

                # Get status
                state_status = orchestrator.state_manager.get_status()
                knowledge_status = orchestrator.knowledge_manager.get_status()

                # Verify Phase 6 data is present
                assert state_status["state_history_size"] == 2
                assert state_status["loop_detection_enabled"] is True
                assert knowledge_status["object_events_tracked"] == 1

    def test_managers_status_reporting(self):
        """Test that all managers report Phase 6 status correctly."""
        with patch("orchestration.zork_orchestrator_v2.JerichoInterface"):
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-status")

                # Get orchestrator status (includes all managers)
                status = orchestrator.get_orchestrator_status()

                # Should include manager statuses
                assert "managers" in status
                assert "StateManager" in status["managers"]
                assert "KnowledgeManager" in status["managers"]

    def test_state_loop_detection_with_jericho_state_changes(self):
        """Test state loop detection with actual Jericho state changes."""
        mock_jericho = Mock()

        # Simulate state changes
        state_sequence = [
            (1, 2, 3),  # State 1 - West of House
            (4, 5, 6),  # State 2 - North of House
            (7, 8, 9),  # State 3 - Behind House
            (4, 5, 6),  # State 2 again - Loop back to North of House
        ]

        mock_jericho.save_state.side_effect = state_sequence
        mock_jericho.start.return_value = "Welcome"
        mock_jericho.get_location_structured.return_value = Mock(num=1, name="Start")
        mock_jericho.get_inventory_structured.return_value = []
        mock_jericho.get_score.return_value = (0, 350)
        mock_jericho.is_game_over.return_value = (False, None)
        mock_jericho.send_command.return_value = "Ok"
        mock_jericho.close.return_value = None

        with patch(
            "orchestration.zork_orchestrator_v2.JerichoInterface",
            return_value=mock_jericho,
        ):
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-loop-detection")

                # Track states manually to simulate gameplay
                for i in range(3):
                    loop_detected = orchestrator.state_manager.track_state_hash(
                        mock_jericho
                    )
                    assert loop_detected is False

                # Fourth state should detect loop
                loop_detected = orchestrator.state_manager.track_state_hash(mock_jericho)
                assert loop_detected is True

    def test_object_event_tracking_during_gameplay_simulation(self):
        """Test object event tracking during simulated gameplay."""
        mock_jericho = Mock()
        mock_jericho.start.return_value = "Welcome"

        # Create properly configured location mock
        mock_location = Mock()
        mock_location.num = 1
        mock_location.name = "Start"
        mock_jericho.get_location_structured.return_value = mock_location

        mock_jericho.get_score.return_value = (0, 350)
        mock_jericho.is_game_over.return_value = (False, None)
        mock_jericho.save_state.return_value = (1, 2, 3)
        mock_jericho.close.return_value = None

        # Simulate inventory changes - properly configure Mock object
        mock_obj = Mock()
        mock_obj.num = 42
        mock_obj.name = "brass lantern"
        mock_jericho.get_inventory_structured.return_value = [mock_obj]

        with patch(
            "orchestration.zork_orchestrator_v2.JerichoInterface",
            return_value=mock_jericho,
        ):
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-obj-events")

                # Simulate inventory change
                prev_inv = []
                curr_inv = ["brass lantern"]

                orchestrator.knowledge_manager.detect_object_events(
                    prev_inventory=prev_inv,
                    current_inventory=curr_inv,
                    jericho_interface=mock_jericho,
                    action="take lantern",
                    turn=1,
                )

                # Should have tracked event
                assert len(orchestrator.knowledge_manager.object_events) == 1
                event = orchestrator.knowledge_manager.object_events[0]
                assert event["event_type"] == "acquired"
                assert event["object_name"] == "brass lantern"

    def test_phase6_data_persistence_across_turns(self):
        """Test that Phase 6 data persists correctly across turns."""
        mock_jericho = Mock()
        mock_jericho.start.return_value = "Welcome"
        mock_jericho.get_location_structured.return_value = Mock(num=1, name="Start")
        mock_jericho.get_inventory_structured.return_value = []
        mock_jericho.get_score.return_value = (0, 350)
        mock_jericho.is_game_over.return_value = (False, None)
        mock_jericho.close.return_value = None

        # Different states for each turn
        mock_jericho.save_state.side_effect = [
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
        ]

        with patch(
            "orchestration.zork_orchestrator_v2.JerichoInterface",
            return_value=mock_jericho,
        ):
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-persistence")

                # Track states across multiple turns
                for turn in range(3):
                    orchestrator.state_manager.track_state_hash(mock_jericho)

                # All states should be tracked
                assert len(orchestrator.state_manager.state_history) == 3

                # Track some object events
                for i in range(5):
                    orchestrator.knowledge_manager.track_object_event(
                        event_type="acquired",
                        obj_id=i,
                        obj_name=f"item_{i}",
                        turn=i,
                    )

                # All events should be tracked
                assert len(orchestrator.knowledge_manager.object_events) == 5

    def test_phase6_export_data_format(self):
        """Test that Phase 6 data is exported in correct format."""
        with patch("orchestration.zork_orchestrator_v2.JerichoInterface"):
            with patch("orchestration.zork_orchestrator_v2.ZorkAgent"), patch(
                "orchestration.zork_orchestrator_v2.ZorkCritic"
            ), patch("orchestration.zork_orchestrator_v2.HybridZorkExtractor"):

                orchestrator = ZorkOrchestratorV2(episode_id="test-export-format")

                # Add Phase 6 data
                orchestrator.state_manager.state_history = [111, 222, 333]
                orchestrator.knowledge_manager.track_object_event(
                    event_type="opened",
                    obj_id=50,
                    obj_name="mailbox",
                    turn=1,
                    additional_context={"location": "West of House"},
                )

                # Get export data
                knowledge_export = orchestrator.knowledge_manager.get_export_data()

                # Verify format
                assert "object_events" in knowledge_export
                assert "total_object_events" in knowledge_export
                assert isinstance(knowledge_export["object_events"], list)
                assert knowledge_export["total_object_events"] == 1

                # Verify event structure
                event = knowledge_export["object_events"][0]
                assert "event_type" in event
                assert "object_id" in event
                assert "object_name" in event
                assert "turn" in event
                assert "location" in event


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
