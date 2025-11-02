"""
ABOUTME: Unit tests for SimpleMemoryManager LLM synthesis pipeline.
ABOUTME: Tests Pydantic schema, synthesis method, and action outcome recording.
"""

import pytest
from unittest.mock import Mock
import json

from tests.simple_memory.conftest import SAMPLE_MEMORIES_FULL


class TestMemorySynthesisResponseSchema:
    """Test the MemorySynthesisResponse Pydantic model."""

    def test_valid_response_parsing(self, mock_logger, game_config, game_state):
        """Test valid response parsing with all fields present."""
        from managers.simple_memory_manager import MemorySynthesisResponse
        import json

        # Valid JSON with all fields
        valid_json = {
            "should_remember": True,
            "category": "SUCCESS",
            "memory_title": "Test Title",
            "memory_text": "Test memory text.",
            "reasoning": "Test reasoning"
        }

        # Parse using Pydantic
        response = MemorySynthesisResponse.model_validate(valid_json)

        # Verify all fields
        assert response.should_remember is True
        assert response.category == "SUCCESS"
        assert response.memory_title == "Test Title"
        assert response.memory_text == "Test memory text."
        assert response.reasoning == "Test reasoning"

    def test_missing_optional_field_reasoning(self, mock_logger, game_config, game_state):
        """Test missing optional field (reasoning) defaults to empty string."""
        from managers.simple_memory_manager import MemorySynthesisResponse

        # JSON without reasoning field
        json_data = {
            "should_remember": True,
            "category": "FAILURE",
            "memory_title": "Test",
            "memory_text": "Test text."
        }

        # Parse
        response = MemorySynthesisResponse.model_validate(json_data)

        # Reasoning should default to empty string
        assert response.reasoning == ""

    def test_invalid_category_still_parses(self, mock_logger, game_config, game_state):
        """Test invalid category still parses (validation happens later)."""
        from managers.simple_memory_manager import MemorySynthesisResponse

        # JSON with invalid category
        json_data = {
            "should_remember": True,
            "category": "INVALID_CATEGORY",  # Not in [SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE]
            "memory_title": "Test",
            "memory_text": "Test text.",
            "reasoning": "Test"
        }

        # Should still parse (we don't enforce enum validation at Pydantic level)
        response = MemorySynthesisResponse.model_validate(json_data)
        assert response.category == "INVALID_CATEGORY"

    def test_field_type_validation(self, mock_logger, game_config, game_state):
        """Test field types are validated correctly."""
        from managers.simple_memory_manager import MemorySynthesisResponse
        from pydantic import ValidationError

        # should_remember must be bool
        with pytest.raises(ValidationError):
            MemorySynthesisResponse.model_validate({
                "should_remember": "yes",  # Wrong type
                "category": "SUCCESS",
                "memory_title": "Test",
                "memory_text": "Test."
            })

        # Other fields must be strings
        with pytest.raises(ValidationError):
            MemorySynthesisResponse.model_validate({
                "should_remember": True,
                "category": 123,  # Wrong type
                "memory_title": "Test",
                "memory_text": "Test."
            })

    def test_model_validate_json_from_string(self, mock_logger, game_config, game_state):
        """Test parsing from JSON string directly."""
        from managers.simple_memory_manager import MemorySynthesisResponse
        import json

        # JSON string
        json_string = json.dumps({
            "should_remember": True,
            "category": "DISCOVERY",
            "memory_title": "Found key",
            "memory_text": "Key unlocks door.",
            "reasoning": "Important discovery"
        })

        # Parse from string
        response = MemorySynthesisResponse.model_validate_json(json_string)

        # Verify
        assert response.should_remember is True
        assert response.category == "DISCOVERY"
        assert response.memory_title == "Found key"


# ============================================================================
# Part B: LLM Synthesis Method Tests (10 tests)
# ============================================================================


class TestSynthesizeMemoryMethod:
    """Test _synthesize_memory() LLM synthesis method."""

    def test_successful_synthesis_should_remember(self, mock_logger, game_config, game_state,
                                                   mock_llm_client_synthesis, sample_z_machine_context):
        """Test successful synthesis when LLM returns should_remember=True."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Patch LLMClientWrapper to return our mock
        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Call synthesis
            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the brass lantern.",
                z_machine_context=sample_z_machine_context
            )

            # Should return MemorySynthesisResponse object
            if result is None:
                # Print logger calls to debug
                print("\n=== Logger calls ===")
                for call in mock_logger.method_calls:
                    print(f"{call}")
            assert result is not None, f"Synthesis returned None. Logger calls: {mock_logger.method_calls}"
            assert result.should_remember is True
            assert result.category == "SUCCESS"
            assert result.memory_title == "Acquired lamp"

            # Verify LLM was called
            assert mock_llm_client_synthesis.chat.completions.create.called

    def test_synthesis_decides_not_to_remember(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test synthesis returns None when LLM says should_remember=False."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch
        import json

        # Mock LLM client that returns should_remember=False
        client = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "should_remember": False,
            "category": "NOTE",
            "memory_title": "Trivial action",
            "memory_text": "Not worth remembering.",
            "reasoning": "Duplicate of existing memory"
        })
        client.chat.completions.create.return_value = mock_response

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="look",
                response="You see nothing special.",
                z_machine_context=sample_z_machine_context
            )

            # Should return None (don't store)
            assert result is None

    def test_existing_memories_passed_to_llm(self, mock_logger, game_config, game_state,
                                            mock_llm_client_synthesis, sample_z_machine_context, create_memories_file):
        """Test existing memories are passed to LLM for deduplication."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Create file with existing memories
        create_memories_file(SAMPLE_MEMORIES_FULL)

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Call synthesis for location that has existing memories
            result = manager._synthesize_memory(
                location_id=23,  # Has existing memories
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Verify LLM was called
            assert mock_llm_client_synthesis.chat.completions.create.called

            # Get the call arguments
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args

            # Verify messages passed to LLM
            messages = call_args[1]['messages']
            prompt = messages[0]['content']

            # Should include existing memories
            assert "Existing Memories" in prompt or "existing memories" in prompt.lower()

    def test_z_machine_context_in_prompt(self, mock_logger, game_config, game_state,
                                        mock_llm_client_synthesis, sample_z_machine_context):
        """Test Z-machine context is included in LLM prompt."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Get call arguments
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            messages = call_args[1]['messages']
            prompt = messages[0]['content']

            # Should include Z-machine context data
            assert "score" in prompt.lower() or "Score" in prompt
            assert "inventory" in prompt.lower() or "Inventory" in prompt

    def test_action_and_response_in_prompt(self, mock_logger, game_config, game_state,
                                          mock_llm_client_synthesis, sample_z_machine_context):
        """Test action text and game response are included in prompt."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            test_action = "take brass lantern"
            test_response = "You pick up the heavy brass lantern."

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action=test_action,
                response=test_response,
                z_machine_context=sample_z_machine_context
            )

            # Get call arguments
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            messages = call_args[1]['messages']
            prompt = messages[0]['content']

            # Should include action and response
            assert test_action in prompt
            assert test_response in prompt

    def test_uses_info_ext_model(self, mock_logger, game_config, game_state,
                                 mock_llm_client_synthesis, sample_z_machine_context):
        """Test synthesis uses config.memory_model."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Set a specific model in config
        game_config.memory_model = "test-memory-model-v2"

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Verify correct model was used
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            assert call_args[1]['model'] == "test-memory-model-v2"

    def test_structured_output_with_json_schema(self, mock_logger, game_config, game_state,
                                               mock_llm_client_synthesis, sample_z_machine_context):
        """Test uses response_format with Pydantic schema for structured output."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Verify response_format was used
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            assert 'response_format' in call_args[1]

            # Should be JSON schema format
            response_format = call_args[1]['response_format']
            assert 'type' in response_format
            assert response_format['type'] == 'json_schema'

    def test_handles_llm_error_gracefully(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test handles LLM error gracefully without crashing."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch

        # Mock LLM client that raises exception
        client = Mock()
        client.chat.completions.create.side_effect = Exception("LLM API error")

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should return None (error handled)
            assert result is None

            # Should log error
            assert mock_logger.error.called

    def test_handles_invalid_json_gracefully(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test handles invalid JSON response gracefully."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch

        # Mock LLM client that returns malformed JSON
        client = Mock()
        mock_response = Mock()
        mock_response.content = "{ invalid json here"  # Malformed
        client.chat.completions.create.return_value = mock_response

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should return None (error handled)
            assert result is None

            # Should log error
            assert mock_logger.error.called

    def test_reasoning_field_captured(self, mock_logger, game_config, game_state,
                                     mock_llm_client_synthesis, sample_z_machine_context):
        """Test reasoning field is captured and logged."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            result = manager._synthesize_memory(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should have reasoning field
            assert result.reasoning == "Significant item acquisition"

            # Should be logged for debugging
            assert mock_logger.debug.called


# ============================================================================
# Part C: record_action_outcome Method Tests (12 tests)
# ============================================================================


class TestRecordActionOutcomeMethod:
    """Test record_action_outcome() complete flow."""

    def test_complete_flow_trigger_synthesize_write_cache(self, mock_logger, game_config, game_state,
                                                          mock_llm_client_synthesis, sample_z_machine_context):
        """Test complete flow: trigger → synthesize → write → cache."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch
        from pathlib import Path

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Record action outcome (should trigger synthesis)
            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the brass lantern.",
                z_machine_context=sample_z_machine_context
            )

            # Should have called LLM
            assert mock_llm_client_synthesis.chat.completions.create.called

            # Should have written to file
            memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
            assert memories_path.exists()
            content = memories_path.read_text()
            assert "Location 23" in content
            assert "Acquired lamp" in content

            # Should have updated cache
            assert 23 in manager.memory_cache
            assert len(manager.memory_cache[23]) == 1
            assert manager.memory_cache[23][0].title == "Acquired lamp"

    def test_skip_when_no_trigger(self, mock_logger, game_config, game_state, mock_llm_client_synthesis):
        """Test skips synthesis when no trigger fires."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Context with no triggers
            no_trigger_context = {
                'score_before': 50,
                'score_after': 50,
                'score_delta': 0,
                'location_before': 15,
                'location_after': 15,
                'location_changed': False,
                'inventory_before': ['lamp'],
                'inventory_after': ['lamp'],
                'inventory_changed': False,
                'died': False,
                'response_length': 30,
                'first_visit': False
            }

            manager.record_action_outcome(
                location_id=15,
                location_name="West of House",
                action="look",
                response="Nothing special.",
                z_machine_context=no_trigger_context
            )

            # Should NOT have called LLM
            assert not mock_llm_client_synthesis.chat.completions.create.called

    def test_skip_when_llm_says_dont_remember(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test skips write when LLM says should_remember=False."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch
        from pathlib import Path
        import json

        # Mock LLM that says don't remember
        client = Mock()
        mock_response = Mock()
        mock_response.content = json.dumps({
            "should_remember": False,
            "category": "NOTE",
            "memory_title": "Trivial",
            "memory_text": "Not worth it.",
            "reasoning": "Duplicate"
        })
        client.chat.completions.create.return_value = mock_response

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="look",
                response="Nothing new.",
                z_machine_context=sample_z_machine_context
            )

            # Should have called LLM
            assert client.chat.completions.create.called

            # Should NOT have written to file
            memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
            assert not memories_path.exists()

            # Cache should be empty
            assert 23 not in manager.memory_cache

    def test_memory_formatted_correctly_before_write(self, mock_logger, game_config, game_state,
                                                    mock_llm_client_synthesis, sample_z_machine_context):
        """Test MemorySynthesisResponse converted to Memory dataclass correctly."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Set game state for metadata
        game_state.episode_id = "ep_001"
        game_state.turn_count = 45

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Check cache has correct Memory format
            memory = manager.memory_cache[23][0]
            assert memory.category == "SUCCESS"
            assert memory.title == "Acquired lamp"
            assert memory.episode == 1  # Extracted from episode_id
            assert memory.turns == "45"  # From turn_count
            assert memory.score_change == 5  # From z_machine_context
            assert memory.text == "Brass lantern provides light for dark areas."

    def test_add_memory_called_with_correct_args(self, mock_logger, game_config, game_state,
                                                 mock_llm_client_synthesis, sample_z_machine_context):
        """Test add_memory is called with correct arguments."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch, Mock

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Mock add_memory method
            manager.add_memory = Mock(return_value=True)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Verify add_memory was called
            assert manager.add_memory.called

            # Check arguments
            call_args = manager.add_memory.call_args
            assert call_args[0][0] == 23  # location_id
            assert call_args[0][1] == "Living Room"  # location_name
            assert hasattr(call_args[0][2], 'category')  # Memory object

    def test_cache_updated_immediately(self, mock_logger, game_config, game_state,
                                      mock_llm_client_synthesis, sample_z_machine_context):
        """Test cache is updated immediately after write."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Cache should be empty initially
            assert 23 not in manager.memory_cache

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Cache should be updated immediately
            assert 23 in manager.memory_cache
            assert len(manager.memory_cache[23]) == 1

            # Should be able to retrieve immediately
            memory_text = manager.get_location_memory(23)
            assert "Acquired lamp" in memory_text

    def test_handles_synthesis_failure_gracefully(self, mock_logger, game_config, game_state, sample_z_machine_context):
        """Test handles synthesis failure gracefully (returns None)."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import Mock, patch

        # Mock LLM that fails
        client = Mock()
        client.chat.completions.create.side_effect = Exception("LLM error")

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=client):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Should not crash
            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should log error
            assert mock_logger.error.called

            # Cache should be empty (no write)
            assert 23 not in manager.memory_cache

    def test_handles_write_failure_gracefully(self, mock_logger, game_config, game_state,
                                              mock_llm_client_synthesis, sample_z_machine_context):
        """Test handles write failure gracefully (add_memory returns False)."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch, Mock

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Mock add_memory to fail
            manager.add_memory = Mock(return_value=False)

            # Should not crash
            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should log warning/error
            assert mock_logger.warning.called or mock_logger.error.called

    def test_existing_memories_retrieved_for_deduplication(self, mock_logger, game_config, game_state,
                                                          mock_llm_client_synthesis, sample_z_machine_context,
                                                          create_memories_file):
        """Test existing memories are retrieved from cache for deduplication."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Create file with existing memories
        create_memories_file(SAMPLE_MEMORIES_FULL)

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            # Verify cache has existing memories
            assert 23 in manager.memory_cache
            initial_count = len(manager.memory_cache[23])

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # LLM should have received existing memories in prompt
            call_args = mock_llm_client_synthesis.chat.completions.create.call_args
            messages = call_args[1]['messages']
            prompt = messages[0]['content']

            # Should mention existing memories
            assert "existing" in prompt.lower() or "Existing" in prompt

    def test_logging_at_each_step(self, mock_logger, game_config, game_state,
                                  mock_llm_client_synthesis, sample_z_machine_context):
        """Test logging happens at each step of the pipeline."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Should have debug logs for trigger
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("trigger" in call.lower() or "synthesis" in call.lower() for call in debug_calls)

            # Should have info log for storage
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("memory" in call.lower() or "stored" in call.lower() for call in info_calls)

    def test_metadata_extracted_from_z_machine_context(self, mock_logger, game_config, game_state,
                                                       mock_llm_client_synthesis, sample_z_machine_context):
        """Test metadata is correctly extracted from z_machine_context."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch

        # Set game state
        game_state.episode_id = "ep_003"
        game_state.turn_count = 127

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="You pick up the lamp.",
                z_machine_context=sample_z_machine_context
            )

            # Check memory has correct metadata
            memory = manager.memory_cache[23][0]
            assert memory.episode == 3  # From episode_id
            assert memory.turns == "127"  # From turn_count
            assert memory.score_change == 5  # From z_machine_context

    def test_end_to_end_with_realistic_context(self, mock_logger, game_config, game_state,
                                               mock_llm_client_synthesis):
        """Test end-to-end with realistic Z-machine context."""
        from managers.simple_memory_manager import SimpleMemoryManager
        from unittest.mock import patch
        from pathlib import Path

        # Realistic context for acquiring an item
        realistic_context = {
            'score_before': 0,
            'score_after': 5,
            'score_delta': 5,
            'location_before': 23,
            'location_after': 23,
            'location_changed': False,
            'inventory_before': [],
            'inventory_after': ['brass lantern'],
            'inventory_changed': True,
            'died': False,
            'response_length': 87,
            'first_visit': False
        }

        game_state.episode_id = "ep_001"
        game_state.turn_count = 45

        with patch('managers.simple_memory_manager.LLMClientWrapper', return_value=mock_llm_client_synthesis):
            manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

            manager.record_action_outcome(
                location_id=23,
                location_name="Living Room",
                action="take lamp",
                response="Taken. The brass lantern is now in your possession and could prove useful for lighting your way.",
                z_machine_context=realistic_context
            )

            # Should complete successfully
            assert 23 in manager.memory_cache
            assert len(manager.memory_cache[23]) == 1

            # File should exist
            memories_path = Path(game_config.zork_game_workdir) / "Memories.md"
            assert memories_path.exists()

            # Content should be correct
            content = memories_path.read_text()
            assert "Location 23: Living Room" in content
            assert "[SUCCESS]" in content
            assert "Acquired lamp" in content
            assert "*(Ep1, T45, +5)*" in content


class TestTriggerDetectionMultipleTriggers:
    """Test behavior when multiple triggers fire simultaneously."""

    def test_multiple_triggers_score_and_location(self, mock_logger, game_config, game_state):
        """Test multiple triggers fire together (score + location)."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 55,  # Score changed
            'score_delta': 5,
            'location_before': 15,
            'location_after': 23,  # Location changed
            'location_changed': True,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should trigger (multiple conditions met)
        assert result is True

        # Logger should log at least one trigger reason
        mock_logger.debug.assert_called()

    def test_multiple_triggers_inventory_and_death(self, mock_logger, game_config, game_state):
        """Test multiple triggers fire together (inventory + death)."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp', 'sword'],
            'inventory_after': ['lamp'],  # Lost sword
            'inventory_changed': True,
            'died': True,  # Also died
            'response_length': 50,
            'first_visit': False
        }

        result = manager._should_synthesize_memory(context)

        # Should trigger (multiple conditions met)
        assert result is True

        # Logger should log at least one trigger reason
        mock_logger.debug.assert_called()

    def test_multiple_triggers_all_conditions(self, mock_logger, game_config, game_state):
        """Test when ALL trigger conditions are met."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 45,  # Score changed (negative)
            'score_delta': -5,
            'location_before': 15,
            'location_after': 23,  # Location changed
            'location_changed': True,
            'inventory_before': ['lamp'],
            'inventory_after': [],  # Lost item
            'inventory_changed': True,
            'died': True,  # Died
            'response_length': 200,  # Substantial response
            'first_visit': True  # First visit
        }

        result = manager._should_synthesize_memory(context)

        # Should trigger (all conditions met)
        assert result is True

        # Logger should log trigger reason
        mock_logger.debug.assert_called()


class TestTriggerDetectionPerformance:
    """Test that trigger detection is fast (no LLM calls)."""

    def test_trigger_detection_is_fast(self, mock_logger, game_config, game_state):
        """Test that trigger detection completes quickly without LLM calls."""
        from managers.simple_memory_manager import SimpleMemoryManager
        import time

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        context = {
            'score_before': 50,
            'score_after': 55,
            'score_delta': 5,
            'location_before': 15,
            'location_after': 23,
            'location_changed': True,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp', 'key'],
            'inventory_changed': True,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        # Measure execution time
        start_time = time.time()
        result = manager._should_synthesize_memory(context)
        elapsed_time = time.time() - start_time

        # Should complete in under 1ms (boolean logic only)
        assert elapsed_time < 0.001, f"Trigger detection took {elapsed_time*1000:.2f}ms (should be <1ms)"

        # Should still return correct result
        assert result is True

    def test_no_llm_client_called_during_trigger_detection(self, mock_logger, game_config, game_state):
        """Test that LLM client is not called during trigger detection."""
        from managers.simple_memory_manager import SimpleMemoryManager

        manager = SimpleMemoryManager(logger=mock_logger, config=game_config, game_state=game_state)

        # Verify manager doesn't have llm_client attribute (not needed for triggers)
        # Trigger detection should be pure logic - no external calls

        context = {
            'score_before': 50,
            'score_after': 55,
            'score_delta': 5,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        # Should work without any external dependencies
        result = manager._should_synthesize_memory(context)

        # Should trigger on score change
        assert result is True
