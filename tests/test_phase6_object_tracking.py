# ABOUTME: Tests for Phase 6 object event tracking in KnowledgeManager
# ABOUTME: Validates object lifecycle event detection and tracking

import pytest
from unittest.mock import Mock, MagicMock, patch
from managers.knowledge_manager import KnowledgeManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class TestPhase6ObjectTracking:
    """Test object event tracking functionality (Phase 6)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return GameConfiguration(
            # Core game settings
            max_turns_per_episode=1000,
            turn_delay_seconds=0.0,
            game_file_path="test_game.z5",
            critic_rejection_threshold=0.5,
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
            memory_model="test-memory-model",
            condensation_model="test-condensation-model",
            # Update intervals
            knowledge_update_interval=100,
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
            enable_state_export=False,
            s3_bucket="test-bucket",
            s3_key_prefix="test/",
            # Simple Memory
            simple_memory_file="Memories.md",
            simple_memory_max_shown=10,
            map_state_file="test_map_state.json",
            # Sampling parameters
            agent_sampling={},
            critic_sampling={},
            extractor_sampling={},
            analysis_sampling={},
            memory_sampling={},
            condensation_sampling={},
        )

    @pytest.fixture
    def game_state(self):
        """Create test game state."""
        state = GameState()
        state.episode_id = "test-episode"
        state.turn_count = 10
        return state

    @pytest.fixture
    def logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        return Mock()

    @pytest.fixture
    def mock_map_manager(self):
        """Create mock map manager."""
        mock_map = Mock()
        # Mock the game_map attribute
        mock_game_map = Mock()
        mock_game_map.render_mermaid.return_value = "graph TD\n  A --> B"
        mock_map.game_map = mock_game_map
        # Mock quality metrics
        mock_map.get_quality_metrics.return_value = {"confidence": 0.8}
        return mock_map

    @pytest.fixture
    def knowledge_manager(self, logger, config, game_state, mock_agent, mock_map_manager):
        """Create KnowledgeManager instance."""
        return KnowledgeManager(
            logger, config, game_state, mock_agent, mock_map_manager
        )

    def test_object_events_initialization(self, knowledge_manager):
        """Test that object events list is initialized empty."""
        assert knowledge_manager.object_events == []

    def test_object_events_reset_on_episode_reset(self, knowledge_manager):
        """Test that object events are cleared on episode reset."""
        # Add some events
        knowledge_manager.object_events = [
            {"event_type": "acquired", "object_name": "lamp"},
            {"event_type": "opened", "object_name": "mailbox"},
        ]

        # Reset episode
        knowledge_manager.reset_episode()

        # Events should be cleared
        assert knowledge_manager.object_events == []

    def test_track_object_event_acquired(self, knowledge_manager, game_state):
        """Test tracking an acquired object event."""
        knowledge_manager.track_object_event(
            event_type="acquired",
            obj_id=42,
            obj_name="brass lantern",
            turn=5,
            additional_context={"action": "take lantern"},
        )

        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["event_type"] == "acquired"
        assert event["object_id"] == 42
        assert event["object_name"] == "brass lantern"
        assert event["turn"] == 5
        assert event["action"] == "take lantern"
        assert event["episode_id"] == "test-episode"
        assert "timestamp" in event

    def test_track_object_event_dropped(self, knowledge_manager):
        """Test tracking a dropped object event."""
        knowledge_manager.track_object_event(
            event_type="dropped", obj_id=-1, obj_name="sword", turn=15  # Placeholder ID
        )

        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["event_type"] == "dropped"
        assert event["object_id"] == -1
        assert event["object_name"] == "sword"

    def test_track_object_event_opened(self, knowledge_manager):
        """Test tracking an opened object event."""
        knowledge_manager.track_object_event(
            event_type="opened", obj_id=100, obj_name="mailbox", turn=2
        )

        assert len(knowledge_manager.object_events) == 1
        assert knowledge_manager.object_events[0]["event_type"] == "opened"

    def test_track_object_event_validates_event_type(self, knowledge_manager, logger):
        """Test that invalid event types trigger a warning."""
        # Should log warning but still track the event
        knowledge_manager.track_object_event(
            event_type="invalid_type", obj_id=1, obj_name="test", turn=1
        )

        # Event should still be tracked
        assert len(knowledge_manager.object_events) == 1
        # Verify warning was logged
        logger.warning.assert_called()

    def test_detect_object_events_acquired_items(self, knowledge_manager):
        """Test detecting acquired items from inventory changes."""
        prev_inventory = ["sword"]
        current_inventory = ["sword", "brass lantern"]

        # Mock Jericho interface
        mock_jericho = Mock()
        mock_obj = Mock()
        mock_obj.num = 42
        mock_obj.name = "brass lantern"
        mock_jericho.get_inventory_structured.return_value = [mock_obj]

        knowledge_manager.detect_object_events(
            prev_inventory=prev_inventory,
            current_inventory=current_inventory,
            jericho_interface=mock_jericho,
            action="take lantern",
            turn=5,
        )

        # Should have tracked one "acquired" event
        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["event_type"] == "acquired"
        assert event["object_id"] == 42
        assert event["object_name"] == "brass lantern"

    def test_detect_object_events_dropped_items(self, knowledge_manager):
        """Test detecting dropped items from inventory changes."""
        prev_inventory = ["sword", "lantern"]
        current_inventory = ["lantern"]

        mock_jericho = Mock()

        knowledge_manager.detect_object_events(
            prev_inventory=prev_inventory,
            current_inventory=current_inventory,
            jericho_interface=mock_jericho,
            action="drop sword",
            turn=10,
        )

        # Should have tracked one "dropped" event
        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["event_type"] == "dropped"
        assert event["object_name"] == "sword"
        assert event["object_id"] == -1  # Placeholder for dropped items

    def test_detect_object_events_open_action(self, knowledge_manager):
        """Test detecting open actions from action text."""
        mock_jericho = Mock()

        knowledge_manager.detect_object_events(
            prev_inventory=[],
            current_inventory=[],
            jericho_interface=mock_jericho,
            action="open mailbox",
            turn=2,
        )

        # Should have tracked one "opened" event
        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["event_type"] == "opened"
        assert event["object_name"] == "mailbox"

    def test_detect_object_events_close_action(self, knowledge_manager):
        """Test detecting close actions from action text."""
        mock_jericho = Mock()

        knowledge_manager.detect_object_events(
            prev_inventory=[],
            current_inventory=[],
            jericho_interface=mock_jericho,
            action="close door",
            turn=8,
        )

        # Should have tracked one "closed" event
        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["event_type"] == "closed"
        assert event["object_name"] == "door"

    def test_get_export_data_includes_object_events(self, knowledge_manager):
        """Test that export data includes recent object events."""
        # Add some events
        for i in range(100):
            knowledge_manager.track_object_event(
                event_type="acquired", obj_id=i, obj_name=f"item_{i}", turn=i
            )

        export_data = knowledge_manager.get_export_data()

        # Should include only recent 50 events
        assert "object_events" in export_data
        assert len(export_data["object_events"]) == 50
        assert "total_object_events" in export_data
        assert export_data["total_object_events"] == 100

    def test_knowledge_manager_status_includes_object_events(self, knowledge_manager):
        """Test that status reporting includes object event count."""
        # Add some events
        knowledge_manager.object_events = [
            {"event_type": "acquired", "object_name": "lamp"},
            {"event_type": "opened", "object_name": "mailbox"},
        ]

        status = knowledge_manager.get_status()

        assert "object_events_tracked" in status
        assert status["object_events_tracked"] == 2

    def test_track_object_event_with_additional_context(self, knowledge_manager):
        """Test tracking events with additional context fields."""
        knowledge_manager.track_object_event(
            event_type="acquired",
            obj_id=50,
            obj_name="rusty key",
            turn=20,
            additional_context={
                "action": "take key",
                "location": "West of House",
                "score_change": 5,
            },
        )

        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["action"] == "take key"
        assert event["location"] == "West of House"
        assert event["score_change"] == 5

    def test_detect_object_events_multiple_acquired_items(self, knowledge_manager):
        """Test detecting multiple acquired items in one turn."""
        prev_inventory = []
        current_inventory = ["sword", "shield", "helmet"]

        # Mock Jericho interface with multiple objects
        # The implementation calls get_inventory_structured() for each acquired item
        mock_jericho = Mock()

        # Create properly configured Mock objects with name attributes
        mock_sword = Mock()
        mock_sword.num = 1
        mock_sword.name = "sword"

        mock_shield = Mock()
        mock_shield.num = 2
        mock_shield.name = "shield"

        mock_helmet = Mock()
        mock_helmet.num = 3
        mock_helmet.name = "helmet"

        mock_objects = [mock_sword, mock_shield, mock_helmet]
        # Return the same list each time get_inventory_structured is called
        mock_jericho.get_inventory_structured.return_value = mock_objects

        knowledge_manager.detect_object_events(
            prev_inventory=prev_inventory,
            current_inventory=current_inventory,
            jericho_interface=mock_jericho,
            action="take all",
            turn=1,
        )

        # Should have tracked three "acquired" events
        # Each item should be found when the implementation iterates through acquired items
        assert len(knowledge_manager.object_events) == 3
        event_names = {event["object_name"] for event in knowledge_manager.object_events}
        assert event_names == {"sword", "shield", "helmet"}
        # Verify all events have valid object IDs
        for event in knowledge_manager.object_events:
            assert event["object_id"] in [1, 2, 3]

    def test_detect_object_events_no_inventory_change(self, knowledge_manager):
        """Test that no events are tracked when inventory doesn't change."""
        prev_inventory = ["sword"]
        current_inventory = ["sword"]

        mock_jericho = Mock()

        knowledge_manager.detect_object_events(
            prev_inventory=prev_inventory,
            current_inventory=current_inventory,
            jericho_interface=mock_jericho,
            action="look",
            turn=5,
        )

        # Should not have tracked any events
        assert len(knowledge_manager.object_events) == 0

    def test_detect_object_events_case_insensitive_actions(self, knowledge_manager):
        """Test that action detection is case-insensitive."""
        mock_jericho = Mock()

        # Test uppercase
        knowledge_manager.detect_object_events(
            prev_inventory=[],
            current_inventory=[],
            jericho_interface=mock_jericho,
            action="OPEN DOOR",
            turn=1,
        )

        # Test mixed case
        knowledge_manager.detect_object_events(
            prev_inventory=[],
            current_inventory=[],
            jericho_interface=mock_jericho,
            action="ClOsE wInDoW",
            turn=2,
        )

        # Should have tracked both events
        assert len(knowledge_manager.object_events) == 2
        assert knowledge_manager.object_events[0]["event_type"] == "opened"
        assert knowledge_manager.object_events[1]["event_type"] == "closed"

    def test_track_object_event_closed(self, knowledge_manager):
        """Test tracking a closed object event."""
        knowledge_manager.track_object_event(
            event_type="closed", obj_id=75, obj_name="chest", turn=12
        )

        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["event_type"] == "closed"
        assert event["object_id"] == 75
        assert event["object_name"] == "chest"

    def test_track_object_event_examined(self, knowledge_manager):
        """Test tracking an examined object event."""
        knowledge_manager.track_object_event(
            event_type="examined", obj_id=88, obj_name="painting", turn=25
        )

        assert len(knowledge_manager.object_events) == 1
        event = knowledge_manager.object_events[0]
        assert event["event_type"] == "examined"
        assert event["object_id"] == 88
        assert event["object_name"] == "painting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
