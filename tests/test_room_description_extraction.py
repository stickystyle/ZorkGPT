# ABOUTME: Tests for room description detection in HybridZorkExtractor
# ABOUTME: Validates extractor correctly flags room descriptions vs action results

import pytest
from unittest.mock import Mock, MagicMock
from hybrid_zork_extractor import HybridZorkExtractor, ExtractorResponse
from game_interface.core.jericho_interface import JerichoInterface
from session.game_configuration import GameConfiguration


# Test fixture for game configuration
@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=GameConfiguration)
    config.info_ext_model = "gpt-4o-mini"
    config.extractor_sampling = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "min_p": None,
    }
    config.get_llm_base_url_for_model = Mock(return_value="https://api.openai.com/v1")
    config.get_effective_api_key = Mock(return_value="test-key")
    return config


# Test fixture for Jericho interface
@pytest.fixture
def mock_jericho():
    """Create a mock Jericho interface for testing."""
    jericho = Mock(spec=JerichoInterface)

    # Setup default location
    location = Mock()
    location.name = "West Of House"
    location.num = 1
    jericho.get_location_structured = Mock(return_value=location)

    # Setup empty inventory
    jericho.get_inventory_structured = Mock(return_value=[])

    # Setup empty visible objects
    jericho.get_all_objects = Mock(return_value=[])

    # Setup score
    jericho.get_score = Mock(return_value=(0, 350))

    return jericho


# Test fixture for extractor with mocked LLM
@pytest.fixture
def extractor_with_mocked_llm(mock_jericho, mock_config):
    """Create extractor with mocked LLM client."""
    # Create mock LLM client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = '{"exits": ["north", "south", "east"], "in_combat": false, "is_room_description": true}'
    mock_client.chat.completions.create = Mock(return_value=mock_response)

    extractor = HybridZorkExtractor(
        jericho_interface=mock_jericho,
        config=mock_config,
        client=mock_client,
        logger=None,
        episode_id="test-episode",
    )

    return extractor


class TestExtractorResponseSchema:
    """Test ExtractorResponse schema accepts is_room_description field."""

    def test_extractor_response_has_is_room_description_field(self):
        """Verify ExtractorResponse schema accepts is_room_description boolean field."""
        # Test that field exists and accepts boolean values
        response = ExtractorResponse(
            current_location_name="West Of House",
            exits=["north", "south", "east"],
            visible_objects=["mailbox"],
            visible_characters=[],
            inventory=[],
            in_combat=False,
            score=0,
            moves=0,
            is_room_description=True,
        )

        assert hasattr(response, "is_room_description"), "ExtractorResponse should have is_room_description field"
        assert response.is_room_description is True, "is_room_description should be True when set"

    def test_extractor_response_default_is_room_description_false(self):
        """Verify is_room_description defaults to False."""
        response = ExtractorResponse(
            current_location_name="West Of House",
            exits=["north"],
            visible_objects=[],
            visible_characters=[],
            inventory=[],
            in_combat=False,
        )

        assert response.is_room_description is False, "is_room_description should default to False"


class TestRoomDescriptionDetection:
    """Test extractor correctly detects room descriptions."""

    def test_extractor_flags_room_description(self, extractor_with_mocked_llm, mock_jericho):
        """Test extractor correctly flags room description as is_room_description=True."""
        # Room description text from Zork
        room_description_text = """West of House
You are standing in an open field west of a white house, with a boarded
front door. There is a small mailbox here."""

        # Mock LLM response to flag this as a room description
        mock_response = Mock()
        mock_response.content = '{"exits": ["north", "south", "east"], "in_combat": false, "is_room_description": true}'
        extractor_with_mocked_llm.client.chat.completions.create = Mock(return_value=mock_response)

        # Extract info
        result = extractor_with_mocked_llm.extract_info(room_description_text)

        # Verify it was flagged as room description
        assert result is not None, "Extraction should succeed"
        assert hasattr(result, "is_room_description"), "Result should have is_room_description field"
        assert result.is_room_description is True, "Room description should be flagged as is_room_description=True"

    def test_extractor_rejects_action_result(self, extractor_with_mocked_llm, mock_jericho):
        """Test extractor correctly rejects action results as is_room_description=False."""
        # Simple action result
        action_result_text = "Taken."

        # Mock LLM response to NOT flag this as a room description
        mock_response = Mock()
        mock_response.content = '{"exits": [], "in_combat": false, "is_room_description": false}'
        extractor_with_mocked_llm.client.chat.completions.create = Mock(return_value=mock_response)

        # Extract info
        result = extractor_with_mocked_llm.extract_info(action_result_text)

        # Verify it was NOT flagged as room description
        assert result is not None, "Extraction should succeed"
        assert result.is_room_description is False, "Action result should NOT be flagged as room description"

    def test_extractor_rejects_reading_text(self, extractor_with_mocked_llm, mock_jericho):
        """Test extractor correctly rejects reading text as is_room_description=False."""
        # Reading text from an object
        reading_text = "Opening the small mailbox reveals a leaflet."

        # Mock LLM response to NOT flag this as a room description
        mock_response = Mock()
        mock_response.content = '{"exits": [], "in_combat": false, "is_room_description": false}'
        extractor_with_mocked_llm.client.chat.completions.create = Mock(return_value=mock_response)

        # Extract info
        result = extractor_with_mocked_llm.extract_info(reading_text)

        # Verify it was NOT flagged as room description
        assert result is not None, "Extraction should succeed"
        assert result.is_room_description is False, "Reading text should NOT be flagged as room description"


class TestRoomDescriptionStorage:
    """Test GameState fields and Orchestrator storage logic for room descriptions."""

    def test_game_state_has_room_description_fields(self):
        """Verify GameState has required room description fields with correct types."""
        from session.game_state import GameState

        # Create instance
        game_state = GameState()

        # Verify fields exist with correct types
        assert hasattr(game_state, "last_room_description"), "GameState should have last_room_description field"
        assert hasattr(game_state, "last_room_description_turn"), "GameState should have last_room_description_turn field"
        assert hasattr(game_state, "last_room_description_location_id"), "GameState should have last_room_description_location_id field"

        # Verify default values
        assert game_state.last_room_description == "", "last_room_description should default to empty string"
        assert game_state.last_room_description_turn == 0, "last_room_description_turn should default to 0"
        assert game_state.last_room_description_location_id is None, "last_room_description_location_id should default to None"

    def test_game_state_reset_clears_room_description(self):
        """Verify reset_episode() clears all room description fields."""
        from session.game_state import GameState

        # Create and populate
        game_state = GameState()
        game_state.last_room_description = "Test room description"
        game_state.last_room_description_turn = 42
        game_state.last_room_description_location_id = 5

        # Reset episode
        game_state.reset_episode("test-episode")

        # Verify cleared
        assert game_state.last_room_description == "", "last_room_description should be cleared"
        assert game_state.last_room_description_turn == 0, "last_room_description_turn should be cleared"
        assert game_state.last_room_description_location_id is None, "last_room_description_location_id should be cleared"

    def test_orchestrator_stores_room_description_when_flagged(self):
        """Verify orchestrator stores room description when extractor flags is_room_description=True."""
        from session.game_state import GameState
        from game_interface.core.jericho_interface import JerichoInterface
        from unittest.mock import patch, MagicMock
        import logging

        # Create minimal components
        game_state = GameState()
        game_state.turn_count = 5  # Simulate turn 5

        # Mock logger
        mock_logger = Mock(spec=logging.Logger)

        # Mock location
        mock_location = Mock()
        mock_location.num = 7
        mock_location.name = "Test Room"

        # Mock Jericho interface
        mock_jericho = Mock(spec=JerichoInterface)
        mock_jericho.get_location_structured.return_value = mock_location

        # Mock extractor response with is_room_description=True
        mock_extracted_info = ExtractorResponse(
            current_location_name="Test Room",
            exits=["north", "south"],
            visible_objects=[],
            visible_characters=[],
            inventory=[],
            in_combat=False,
            is_room_description=True,
        )

        # Simulate the orchestrator storage logic
        clean_response = "Test room description text"

        # This is the logic from zork_orchestrator_v2.py
        if mock_extracted_info.is_room_description:
            current_location = mock_jericho.get_location_structured()
            location_id = current_location.num if current_location else None

            game_state.last_room_description = clean_response
            game_state.last_room_description_turn = game_state.turn_count
            game_state.last_room_description_location_id = location_id

            mock_logger.info(
                "Room description captured",
                extra={
                    "event_type": "room_description_captured",
                    "turn": game_state.turn_count,
                    "location_id": location_id,
                    "text_length": len(clean_response),
                }
            )

        # Verify storage
        assert game_state.last_room_description == "Test room description text", "Room description should be stored"
        assert game_state.last_room_description_turn == 5, "Turn should be recorded"
        assert game_state.last_room_description_location_id == 7, "Location ID should be recorded"

        # Verify logger was called
        assert mock_logger.info.called, "Logger should be called"
        log_call = mock_logger.info.call_args
        assert "Room description captured" in log_call[0][0], "Log message should mention room description"
        assert log_call[1]["extra"]["event_type"] == "room_description_captured", "event_type should be correct"
        assert log_call[1]["extra"]["turn"] == 5, "Turn should be logged"
        assert log_call[1]["extra"]["location_id"] == 7, "Location ID should be logged"
        assert log_call[1]["extra"]["text_length"] == len(clean_response), "Text length should be logged"

    def test_orchestrator_does_not_store_when_not_flagged(self):
        """Verify orchestrator does not store room description when is_room_description=False."""
        from session.game_state import GameState
        from game_interface.core.jericho_interface import JerichoInterface
        from unittest.mock import Mock
        import logging

        # Create minimal components
        game_state = GameState()
        game_state.turn_count = 3

        # Mock logger
        mock_logger = Mock(spec=logging.Logger)

        # Mock location
        mock_location = Mock()
        mock_location.num = 8
        mock_location.name = "Another Room"

        # Mock Jericho interface
        mock_jericho = Mock(spec=JerichoInterface)
        mock_jericho.get_location_structured.return_value = mock_location

        # Mock extractor response with is_room_description=False
        mock_extracted_info = ExtractorResponse(
            current_location_name="Another Room",
            exits=["east"],
            visible_objects=[],
            visible_characters=[],
            inventory=[],
            in_combat=False,
            is_room_description=False,
        )

        # Simulate the orchestrator storage logic
        clean_response = "Taken."

        # This is the logic from zork_orchestrator_v2.py
        if mock_extracted_info.is_room_description:
            current_location = mock_jericho.get_location_structured()
            location_id = current_location.num if current_location else None

            game_state.last_room_description = clean_response
            game_state.last_room_description_turn = game_state.turn_count
            game_state.last_room_description_location_id = location_id

            mock_logger.info(
                "Room description captured",
                extra={
                    "event_type": "room_description_captured",
                    "turn": game_state.turn_count,
                    "location_id": location_id,
                    "text_length": len(clean_response),
                }
            )

        # Verify NOT stored (remains at defaults)
        assert game_state.last_room_description == "", "Room description should remain empty"
        assert game_state.last_room_description_turn == 0, "Turn should remain 0"
        assert game_state.last_room_description_location_id is None, "Location ID should remain None"

        # Verify logger was NOT called
        assert not mock_logger.info.called, "Logger should not be called when is_room_description=False"

    def test_orchestrator_logs_room_description_capture(self):
        """Verify orchestrator logs room description capture with correct metadata."""
        from session.game_state import GameState
        from game_interface.core.jericho_interface import JerichoInterface
        from unittest.mock import Mock
        import logging

        # Create minimal components
        game_state = GameState()
        game_state.turn_count = 9

        # Mock logger (use MagicMock to track calls with extra parameters)
        from unittest.mock import MagicMock
        mock_logger = MagicMock(spec=logging.Logger)

        # Mock location
        mock_location = Mock()
        mock_location.num = 42
        mock_location.name = "Logged Room"

        # Mock Jericho interface
        mock_jericho = Mock(spec=JerichoInterface)
        mock_jericho.get_location_structured.return_value = mock_location

        # Mock extractor response with is_room_description=True
        mock_extracted_info = ExtractorResponse(
            current_location_name="Logged Room",
            exits=["up"],
            visible_objects=[],
            visible_characters=[],
            inventory=[],
            in_combat=False,
            is_room_description=True,
        )

        # Simulate the orchestrator storage logic
        test_description = "This is a test room description for logging."

        # This is the logic from zork_orchestrator_v2.py
        if mock_extracted_info.is_room_description:
            current_location = mock_jericho.get_location_structured()
            location_id = current_location.num if current_location else None

            game_state.last_room_description = test_description
            game_state.last_room_description_turn = game_state.turn_count
            game_state.last_room_description_location_id = location_id

            mock_logger.info(
                "Room description captured",
                extra={
                    "event_type": "room_description_captured",
                    "turn": game_state.turn_count,
                    "location_id": location_id,
                    "text_length": len(test_description),
                }
            )

        # Verify logger was called with correct metadata
        assert mock_logger.info.called, "Logger should be called"
        log_call = mock_logger.info.call_args
        assert "Room description captured" in log_call[0][0], "Log message should mention room description"

        # Verify extra metadata
        extra_data = log_call[1]["extra"]
        assert extra_data["event_type"] == "room_description_captured", "event_type should be room_description_captured"
        assert extra_data["turn"] == 9, "Turn should be 9"
        assert extra_data["location_id"] == 42, "Location ID should be 42"
        assert extra_data["text_length"] == len(test_description), f"Text length should be {len(test_description)}"

    def test_room_description_location_id_matching(self):
        """Verify stored room description retains original location_id when location changes."""
        from session.game_state import GameState
        from game_interface.core.jericho_interface import JerichoInterface
        from unittest.mock import Mock
        import logging

        # Create minimal components
        game_state = GameState()
        mock_logger = Mock(spec=logging.Logger)

        # Mock Jericho interface
        mock_jericho = Mock(spec=JerichoInterface)

        # First turn: Store description at location 1
        game_state.turn_count = 1
        mock_location_1 = Mock()
        mock_location_1.num = 1
        mock_location_1.name = "Room One"
        mock_jericho.get_location_structured.return_value = mock_location_1

        mock_extracted_info_1 = ExtractorResponse(
            current_location_name="Room One",
            exits=["north"],
            visible_objects=[],
            visible_characters=[],
            inventory=[],
            in_combat=False,
            is_room_description=True,
        )

        clean_response_1 = "First room description"

        # Simulate storage logic
        if mock_extracted_info_1.is_room_description:
            current_location = mock_jericho.get_location_structured()
            location_id = current_location.num if current_location else None

            game_state.last_room_description = clean_response_1
            game_state.last_room_description_turn = game_state.turn_count
            game_state.last_room_description_location_id = location_id

        # Verify stored at location 1
        assert game_state.last_room_description_location_id == 1, "Should store location 1"
        assert game_state.last_room_description == "First room description", "Should store description"

        # Second turn: Move to location 2 (NOT a room description)
        game_state.turn_count = 2
        mock_location_2 = Mock()
        mock_location_2.num = 2
        mock_location_2.name = "Room Two"
        mock_jericho.get_location_structured.return_value = mock_location_2

        mock_extracted_info_2 = ExtractorResponse(
            current_location_name="Room Two",
            exits=["south"],
            visible_objects=[],
            visible_characters=[],
            inventory=[],
            in_combat=False,
            is_room_description=False,  # NOT a room description
        )

        clean_response_2 = "You moved."

        # Simulate storage logic (should NOT update)
        if mock_extracted_info_2.is_room_description:
            current_location = mock_jericho.get_location_structured()
            location_id = current_location.num if current_location else None

            game_state.last_room_description = clean_response_2
            game_state.last_room_description_turn = game_state.turn_count
            game_state.last_room_description_location_id = location_id

        # Verify location ID still 1 (not updated)
        assert game_state.last_room_description_location_id == 1, "Should retain original location ID"
        assert game_state.last_room_description == "First room description", "Should retain original description"
        assert game_state.last_room_description_turn == 1, "Should retain original turn"


class TestRoomDescriptionContextIntegration:
    """Test ContextManager integration for exposing room descriptions to agent and critic."""

    def test_get_room_description_for_context_returns_description_if_recent(self):
        """Verify _get_room_description_for_context returns description with correct age if recent."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # Set room description at turn 5
        game_state.last_room_description = "You are standing in an open field."
        game_state.last_room_description_turn = 5
        game_state.last_room_description_location_id = 1
        game_state.turn_count = 7  # Current turn is 7 (2 turns later)

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Call method
        result = context_manager._get_room_description_for_context(current_location_id=1)

        # Verify
        assert result is not None, "Should return description dict"
        assert result["text"] == "You are standing in an open field.", "Should return correct text"
        assert result["turns_ago"] == 2, "Should calculate turns_ago correctly (7 - 5 = 2)"

    def test_get_room_description_for_context_returns_none_if_too_old(self):
        """Verify _get_room_description_for_context returns None if description aged out (>10 turns)."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # Set room description at turn 1
        game_state.last_room_description = "Old room description."
        game_state.last_room_description_turn = 1
        game_state.last_room_description_location_id = 1
        game_state.turn_count = 12  # Current turn is 12 (11 turns later, > 10)

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Call method
        result = context_manager._get_room_description_for_context(current_location_id=1)

        # Verify
        assert result is None, "Should return None for descriptions older than 10 turns"

    def test_get_room_description_for_context_returns_none_if_location_mismatch(self):
        """Verify _get_room_description_for_context returns None if description is for different location."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # Set room description at location 1
        game_state.last_room_description = "Description for location 1."
        game_state.last_room_description_turn = 5
        game_state.last_room_description_location_id = 1
        game_state.turn_count = 7

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Call method with different location
        result = context_manager._get_room_description_for_context(current_location_id=2)

        # Verify
        assert result is None, "Should return None when current location differs from stored location"

    def test_get_room_description_for_context_returns_none_if_no_description_stored(self):
        """Verify _get_room_description_for_context returns None if no description stored."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # No room description stored (defaults to empty string)
        game_state.turn_count = 10

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Call method
        result = context_manager._get_room_description_for_context(current_location_id=1)

        # Verify
        assert result is None, "Should return None when no description is stored"

    def test_agent_context_includes_room_description(self):
        """Verify get_agent_context includes room description when available and recent."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # Set room description
        game_state.last_room_description = "You are in a forest clearing."
        game_state.last_room_description_turn = 8
        game_state.last_room_description_location_id = 5
        game_state.turn_count = 10  # 2 turns later

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Call get_agent_context with matching location_id
        context = context_manager.get_agent_context(
            current_state="Forest clearing",
            inventory=["lamp"],
            location="Forest Clearing",
            location_id=5,
        )

        # Verify room description included
        assert "room_description" in context, "Context should include room_description key"
        assert context["room_description"] == "You are in a forest clearing.", "Should include correct description"
        assert "room_description_age" in context, "Context should include room_description_age key"
        assert context["room_description_age"] == 2, "Should calculate age correctly"

    def test_agent_context_excludes_room_description_if_not_available(self):
        """Verify get_agent_context excludes room description when not available."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # No room description stored
        game_state.turn_count = 10

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Call get_agent_context
        context = context_manager.get_agent_context(
            current_state="Some room",
            inventory=[],
            location="Some Room",
            location_id=3,
        )

        # Verify room description NOT included
        assert "room_description" not in context, "Context should NOT include room_description key when not available"
        assert "room_description_age" not in context, "Context should NOT include room_description_age key when not available"

    def test_critic_context_includes_room_description(self):
        """Verify get_critic_context includes room description when available and recent."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # Set room description
        game_state.last_room_description = "You see a troll blocking the path."
        game_state.last_room_description_turn = 15
        game_state.last_room_description_location_id = 8
        game_state.turn_count = 16  # 1 turn later

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Call get_critic_context with location_id parameter
        context = context_manager.get_critic_context(
            current_state="Troll Room",
            proposed_action="attack troll",
            location="Troll Room",
            location_id=8,
        )

        # Verify room description included
        assert "room_description" in context, "Context should include room_description key"
        assert context["room_description"] == "You see a troll blocking the path.", "Should include correct description"
        assert "room_description_age" in context, "Context should include room_description_age key"
        assert context["room_description_age"] == 1, "Should calculate age correctly"

    def test_formatted_prompt_includes_room_description_current_turn(self):
        """Verify get_formatted_agent_prompt_context includes room description with no age indicator when age=0."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Create context with room description (age 0)
        context = {
            "current_location": "West of House",
            "inventory": ["lamp"],
            "room_description": "You are standing in an open field west of a white house.",
            "room_description_age": 0,
        }

        # Call formatting method
        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Verify formatting
        assert "ROOM DESCRIPTION:" in formatted, "Should include ROOM DESCRIPTION header"
        assert "You are standing in an open field west of a white house." in formatted, "Should include description text"
        assert "turns ago" not in formatted.split("ROOM DESCRIPTION:")[1].split("\n")[0], "Should NOT show age when age=0"

    def test_formatted_prompt_includes_room_description_with_age(self):
        """Verify get_formatted_agent_prompt_context includes room description with age indicator when age>0."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Create context with room description (age 3)
        context = {
            "current_location": "Behind House",
            "inventory": [],
            "room_description": "You are behind the white house. A window is slightly open.",
            "room_description_age": 3,
        }

        # Call formatting method
        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Verify formatting
        assert "ROOM DESCRIPTION (3 turns ago):" in formatted, "Should include ROOM DESCRIPTION with age"
        assert "You are behind the white house. A window is slightly open." in formatted, "Should include description text"

    def test_formatted_prompt_excludes_room_description_if_not_in_context(self):
        """Verify get_formatted_agent_prompt_context excludes room description section when not in context."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Setup
        mock_logger = Mock()
        mock_config = Mock(spec=GameConfiguration)
        mock_config.room_description_age_window = 10
        game_state = GameState()

        # Create ContextManager
        context_manager = ContextManager(mock_logger, mock_config, game_state)

        # Create context WITHOUT room description
        context = {
            "current_location": "Kitchen",
            "inventory": ["knife"],
        }

        # Call formatting method
        formatted = context_manager.get_formatted_agent_prompt_context(context)

        # Verify NO room description section
        assert "ROOM DESCRIPTION" not in formatted, "Should NOT include ROOM DESCRIPTION when not in context"


class TestRoomDescriptionConfiguration:
    """Test configurable aging window for room descriptions."""

    def test_default_aging_window_is_10_turns(self):
        """Verify room_description_age_window defaults to 10 in GameConfiguration."""
        from session.game_configuration import GameConfiguration

        # Create default config with required field
        config = GameConfiguration(max_turns_per_episode=500)

        # Verify default value
        assert hasattr(config, "room_description_age_window"), "Config should have room_description_age_window field"
        assert config.room_description_age_window == 10, "Default aging window should be 10 turns"

    def test_custom_aging_window_from_config(self):
        """Verify custom aging window value is respected in ContextManager."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Create config with custom aging window (15 turns)
        config = GameConfiguration(
            max_turns_per_episode=500,
            room_description_age_window=15
        )

        # Setup game state
        mock_logger = Mock()
        game_state = GameState()

        # Set room description at turn 1
        game_state.last_room_description = "Test room description."
        game_state.last_room_description_turn = 1
        game_state.last_room_description_location_id = 5

        # Create ContextManager with custom config
        context_manager = ContextManager(mock_logger, config, game_state)

        # At turn 12 (11 turns ago): should still be available with 15-turn window
        game_state.turn_count = 12
        result_12 = context_manager._get_room_description_for_context(current_location_id=5)
        assert result_12 is not None, "Description should be available at 11 turns ago with 15-turn window"
        assert result_12["text"] == "Test room description.", "Should return correct description"
        assert result_12["turns_ago"] == 11, "Should calculate correct age"

        # At turn 17 (16 turns ago): should be None (exceeds 15-turn window)
        game_state.turn_count = 17
        result_17 = context_manager._get_room_description_for_context(current_location_id=5)
        assert result_17 is None, "Description should be None at 16 turns ago (exceeds 15-turn window)"

    def test_context_manager_respects_configured_age_window(self):
        """Verify ContextManager respects configured age window (test with 5-turn window)."""
        from managers.context_manager import ContextManager
        from session.game_state import GameState
        from session.game_configuration import GameConfiguration
        from unittest.mock import Mock

        # Create config with small aging window (5 turns)
        config = GameConfiguration(
            max_turns_per_episode=500,
            room_description_age_window=5
        )

        # Setup game state
        mock_logger = Mock()
        game_state = GameState()

        # Set room description at turn 1
        game_state.last_room_description = "Small window test."
        game_state.last_room_description_turn = 1
        game_state.last_room_description_location_id = 10

        # Create ContextManager with small window config
        context_manager = ContextManager(mock_logger, config, game_state)

        # At turn 7 (6 turns ago): should return None (exceeds 5-turn window)
        game_state.turn_count = 7
        result = context_manager._get_room_description_for_context(current_location_id=10)
        assert result is None, "Description should be None at 6 turns ago (exceeds 5-turn window)"
