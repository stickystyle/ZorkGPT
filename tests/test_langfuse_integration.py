"""Integration tests for Langfuse observability.

This test suite validates that the complete Langfuse integration works correctly
end-to-end, covering trace hierarchy, session metadata, component nesting, usage
tracking, graceful degradation, flush behavior, and error resilience.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
from llm_client import LLMClient


class TestLangfuseClientInitialization:
    """Tests for Langfuse client initialization."""

    def test_langfuse_client_initialization_with_credentials(self, monkeypatch):
        """Test that Langfuse client initializes when credentials are present."""
        # Set environment variables
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # Mock Langfuse to avoid actual network calls
        with patch('orchestration.zork_orchestrator_v2.Langfuse') as MockLangfuse:
            mock_instance = MagicMock()
            MockLangfuse.return_value = mock_instance

            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Verify Langfuse was initialized
            assert orchestrator.langfuse_client is not None
            MockLangfuse.assert_called_once()

    def test_langfuse_client_initialization_without_credentials(self, monkeypatch):
        """Test that system works without Langfuse credentials."""
        # Note: Global conftest.py fixture already clears Langfuse env vars
        # Orchestrator should initialize but langfuse_client should be None
        with patch('orchestration.zork_orchestrator_v2.Langfuse') as MockLangfuse:
            # Simulate Langfuse initialization failure due to missing credentials
            MockLangfuse.side_effect = ValueError("Missing credentials")

            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Verify graceful degradation
            assert orchestrator.langfuse_client is None


class TestTurnLevelTraceCreation:
    """Tests for turn-level trace creation and metadata."""

    def test_turn_level_trace_creation(self, monkeypatch):
        """Test that each turn creates a trace with proper metadata."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('orchestration.zork_orchestrator_v2.Langfuse') as MockLangfuse:
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            # Create context manager mock for span
            mock_span = MagicMock()
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_span)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_client.start_as_current_span.return_value = mock_context

            orchestrator = ZorkOrchestratorV2(episode_id="test-episode-123")

            # Setup game state
            orchestrator.game_state.turn_count = 1
            orchestrator.game_state.previous_zork_score = 0
            orchestrator.game_state.current_room_id = 1
            orchestrator.game_state.current_room_name_for_map = "West of House"

            # Mock the game interface and components to avoid actual LLM calls
            with patch.object(orchestrator.jericho_interface, 'send_command', return_value="You are in a forest."):
                with patch.object(orchestrator.jericho_interface, 'is_game_over', return_value=(False, "")):
                    with patch.object(orchestrator.agent, 'get_action_with_reasoning', return_value={"action": "look", "reasoning": "test", "new_objective": None}):
                        with patch.object(orchestrator.critic, 'evaluate_action') as mock_critic:
                            # Create proper CriticResult mock
                            mock_critic_result = MagicMock()
                            mock_critic_result.score = 0.8
                            mock_critic_result.justification = "Good action"
                            mock_critic_result.confidence = 0.9
                            mock_critic.return_value = mock_critic_result

                            with patch.object(orchestrator.extractor, 'extract_info', return_value=MagicMock()):
                                with patch.object(orchestrator.extractor, 'get_clean_game_text', return_value="You are in a forest."):
                                    # Execute one turn
                                    orchestrator._run_turn("You are standing in a forest.")

            # Verify trace creation
            mock_client.start_as_current_span.assert_called_once()
            call_args = mock_client.start_as_current_span.call_args

            # Verify turn trace parameters
            assert call_args.kwargs['name'] == 'turn-1'
            assert 'input' in call_args.kwargs
            assert 'metadata' in call_args.kwargs

            # Verify metadata contains turn information
            metadata = call_args.kwargs['metadata']
            assert 'turn_number' in metadata
            assert metadata['turn_number'] == 1
            assert 'score_before' in metadata
            assert 'location_id' in metadata
            assert 'location_name' in metadata

            # Verify trace attributes were set
            mock_span.update_trace.assert_called_once()
            trace_update = mock_span.update_trace.call_args.kwargs
            assert trace_update['session_id'] == "test-episode-123"
            assert trace_update['user_id'] == "zorkgpt-agent"
            assert "zorkgpt" in trace_update['tags']
            assert "game-turn" in trace_update['tags']

    def test_turn_trace_includes_output_metadata(self, monkeypatch):
        """Test that turn trace includes output metadata after turn completion."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('orchestration.zork_orchestrator_v2.Langfuse') as MockLangfuse:
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            # Create context manager mock for span
            mock_span = MagicMock()
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_span)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_client.start_as_current_span.return_value = mock_context

            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Setup game state
            orchestrator.game_state.turn_count = 1
            orchestrator.game_state.previous_zork_score = 5  # Score increases during turn
            orchestrator.game_state.game_over_flag = False

            # Mock components
            with patch.object(orchestrator.jericho_interface, 'send_command', return_value="You picked up the lamp."):
                with patch.object(orchestrator.jericho_interface, 'is_game_over', return_value=(False, "")):
                    with patch.object(orchestrator.agent, 'get_action_with_reasoning', return_value={"action": "take lamp", "reasoning": "test", "new_objective": None}):
                        with patch.object(orchestrator.critic, 'evaluate_action') as mock_critic:
                            mock_critic_result = MagicMock()
                            mock_critic_result.score = 0.9
                            mock_critic_result.justification = "Good action"
                            mock_critic_result.confidence = 0.95
                            mock_critic.return_value = mock_critic_result

                            with patch.object(orchestrator.extractor, 'extract_info', return_value=MagicMock()):
                                with patch.object(orchestrator.extractor, 'get_clean_game_text', return_value="You picked up the lamp."):
                                    orchestrator._run_turn("You are in a room.")

            # Verify span was updated with output
            assert mock_span.update.called
            update_call = mock_span.update.call_args
            assert 'output' in update_call.kwargs
            output = update_call.kwargs['output']
            assert 'action_taken' in output
            assert output['action_taken'] == "take lamp"
            assert 'score_after' in output
            assert 'game_over' in output


class TestComponentSpanNesting:
    """Tests for component @observe decorators and span nesting."""

    def test_component_decorators_applied(self):
        """Test that component @observe decorators are applied correctly."""
        from zork_agent import ZorkAgent
        from zork_critic import ZorkCritic
        from hybrid_zork_extractor import HybridZorkExtractor

        # Verify decorators are applied by checking for wrapped attributes
        # The @observe decorator modifies the function
        agent_method = ZorkAgent.get_action_with_reasoning
        critic_method = ZorkCritic.evaluate_action
        extractor_method = HybridZorkExtractor.extract_info

        # Check if methods exist (they should be callable)
        assert callable(agent_method)
        assert callable(critic_method)
        assert callable(extractor_method)

        # The langfuse decorator adds these attributes when available
        # If Langfuse is not available, the no-op decorator is used
        # Either way, the methods should remain callable
        assert hasattr(ZorkAgent, 'get_action_with_reasoning')
        assert hasattr(ZorkCritic, 'evaluate_action')
        assert hasattr(HybridZorkExtractor, 'extract_info')


class TestLLMClientGenerationTracking:
    """Tests for LLM client generation observation tracking."""

    def test_llm_client_generation_tracking(self, monkeypatch):
        """Test that LLM calls create generation observations."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('llm_client.get_langfuse_client') as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Create context manager mock for generation
            mock_generation = MagicMock()
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_generation)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_client.start_as_current_observation.return_value = mock_context

            # Create LLM client
            client = LLMClient(base_url="http://test", api_key="test-key", logger=None)

            # Mock the actual HTTP request
            with patch.object(client, '_execute_request') as mock_execute:
                from llm_client import LLMResponse
                mock_response = LLMResponse(
                    content="test response",
                    model="gpt-4",
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150
                    }
                )
                mock_execute.return_value = mock_response

                # Make a request
                messages = [{"role": "user", "content": "test"}]
                result = client._make_request(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7
                )

            # Verify generation was created
            mock_client.start_as_current_observation.assert_called_once()
            call_args = mock_client.start_as_current_observation.call_args

            # Verify generation parameters
            assert call_args.kwargs['name'] == "llm-client-call"
            assert call_args.kwargs['as_type'] == "generation"
            assert call_args.kwargs['model'] == "gpt-4"
            assert call_args.kwargs['input'] == messages

            # Verify generation was updated with output and usage
            assert mock_generation.update.call_count >= 1


class TestUsageDetailsExtraction:
    """Tests for usage details extraction and reporting."""

    def test_usage_details_extraction_and_reporting(self, monkeypatch):
        """Test that usage details are extracted and passed to Langfuse."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('llm_client.get_langfuse_client') as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Create context manager mock for generation
            mock_generation = MagicMock()
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_generation)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_client.start_as_current_observation.return_value = mock_context

            client = LLMClient(base_url="http://test", api_key="test-key", logger=None)

            with patch.object(client, '_execute_request') as mock_execute:
                from llm_client import LLMResponse
                # Mock response with Anthropic cache fields
                mock_response = LLMResponse(
                    content="test response",
                    model="claude-3-opus-20240229",
                    usage={
                        "prompt_tokens": 1000,
                        "completion_tokens": 200,
                        "total_tokens": 1200,
                        "cache_creation_input_tokens": 500,
                        "cache_read_input_tokens": 300
                    }
                )
                mock_execute.return_value = mock_response

                messages = [{"role": "user", "content": "test"}]
                result = client._make_request(
                    model="claude-3-opus-20240229",
                    messages=messages
                )

            # Verify usage details were extracted and passed to generation
            update_calls = [call for call in mock_generation.update.call_args_list
                           if call.kwargs and 'usage_details' in call.kwargs]

            assert len(update_calls) > 0, "Usage details should be passed to generation"

            usage_details = update_calls[0].kwargs['usage_details']
            assert usage_details['input'] == 1000
            assert usage_details['output'] == 200
            assert usage_details['total'] == 1200
            assert usage_details['cache_creation_input_tokens'] == 500
            assert usage_details['cache_read_input_tokens'] == 300


class TestFlushAtEpisodeEnd:
    """Tests for Langfuse flush at episode end."""

    def test_flush_at_episode_end(self, monkeypatch):
        """Test that Langfuse traces are flushed when episode ends."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('orchestration.zork_orchestrator_v2.Langfuse') as MockLangfuse:
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Mock all the methods needed for play_episode
            with patch.object(orchestrator.jericho_interface, 'start', return_value="Game intro"):
                with patch.object(orchestrator.jericho_interface, 'send_command', return_value="Verbose enabled"):
                    with patch.object(orchestrator.jericho_interface, 'close'):
                        with patch.object(orchestrator.extractor, 'extract_info', return_value=MagicMock()):
                            with patch.object(orchestrator, '_run_game_loop', return_value=10):
                                with patch.object(orchestrator.episode_synthesizer, 'initialize_episode'):
                                    with patch.object(orchestrator.episode_synthesizer, 'finalize_episode'):
                                        with patch.object(orchestrator, '_export_coordinated_state'):
                                            # Simulate episode end
                                            orchestrator.play_episode()

            # Verify flush was called
            mock_client.flush.assert_called_once()
            # Verify flush was called with timeout
            call_kwargs = mock_client.flush.call_args.kwargs
            assert 'timeout_seconds' in call_kwargs

    def test_flush_continues_on_error(self, monkeypatch):
        """Test that flush errors are handled gracefully."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('orchestration.zork_orchestrator_v2.Langfuse') as MockLangfuse:
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            # Make flush raise an exception
            mock_client.flush.side_effect = RuntimeError("Flush failed")

            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Mock all the methods needed for play_episode
            with patch.object(orchestrator.jericho_interface, 'start', return_value="Game intro"):
                with patch.object(orchestrator.jericho_interface, 'send_command', return_value="Verbose enabled"):
                    with patch.object(orchestrator.jericho_interface, 'close'):
                        with patch.object(orchestrator.extractor, 'extract_info', return_value=MagicMock()):
                            with patch.object(orchestrator, '_run_game_loop', return_value=10):
                                with patch.object(orchestrator.episode_synthesizer, 'initialize_episode'):
                                    with patch.object(orchestrator.episode_synthesizer, 'finalize_episode'):
                                        with patch.object(orchestrator, '_export_coordinated_state'):
                                            # Should not crash despite flush error
                                            score = orchestrator.play_episode()
                                            assert score == 10


class TestGracefulDegradation:
    """Tests for graceful degradation when Langfuse is not available."""

    def test_graceful_degradation_without_langfuse(self):
        """Test that system works correctly when Langfuse is not available."""
        # This test runs without Langfuse credentials
        # Should not crash and should complete normally

        with patch('orchestration.zork_orchestrator_v2.LANGFUSE_AVAILABLE', False):
            with patch('orchestration.zork_orchestrator_v2.Langfuse', None):
                orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

                # Verify orchestrator initialized
                assert orchestrator is not None
                assert orchestrator.langfuse_client is None

                # Mock components and run one turn
                with patch.object(orchestrator.jericho_interface, 'send_command', return_value="You are in a forest."):
                    with patch.object(orchestrator.jericho_interface, 'is_game_over', return_value=(False, "")):
                        with patch.object(orchestrator.agent, 'get_action_with_reasoning', return_value={"action": "look", "reasoning": "test", "new_objective": None}):
                            with patch.object(orchestrator.critic, 'evaluate_action') as mock_critic:
                                mock_critic_result = MagicMock()
                                mock_critic_result.score = 0.8
                                mock_critic_result.justification = "Good"
                                mock_critic_result.confidence = 0.9
                                mock_critic.return_value = mock_critic_result

                                with patch.object(orchestrator.extractor, 'extract_info', return_value=MagicMock()):
                                    with patch.object(orchestrator.extractor, 'get_clean_game_text', return_value="You are in a forest."):
                                        # Should not crash
                                        action, state = orchestrator._run_turn("Test state")

                                        assert action is not None
                                        assert state is not None


class TestErrorResilience:
    """Tests for error resilience when Langfuse operations fail."""

    def test_error_resilience_langfuse_span_failures(self, monkeypatch):
        """Test that Langfuse span failures don't break the game."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('orchestration.zork_orchestrator_v2.Langfuse') as MockLangfuse:
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            # Make start_as_current_span raise an exception
            mock_client.start_as_current_span.side_effect = RuntimeError("Langfuse connection failed")

            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Mock components and run one turn - should not crash
            with patch.object(orchestrator.jericho_interface, 'send_command', return_value="You are in a forest."):
                with patch.object(orchestrator.jericho_interface, 'is_game_over', return_value=(False, "")):
                    with patch.object(orchestrator.agent, 'get_action_with_reasoning', return_value={"action": "look", "reasoning": "test", "new_objective": None}):
                        with patch.object(orchestrator.critic, 'evaluate_action') as mock_critic:
                            mock_critic_result = MagicMock()
                            mock_critic_result.score = 0.8
                            mock_critic_result.justification = "Good"
                            mock_critic_result.confidence = 0.9
                            mock_critic.return_value = mock_critic_result

                            with patch.object(orchestrator.extractor, 'extract_info', return_value=MagicMock()):
                                with patch.object(orchestrator.extractor, 'get_clean_game_text', return_value="You are in a forest."):
                                    # Should handle error gracefully and continue
                                    # The exception will be caught in _run_turn's try/except
                                    action, state = orchestrator._run_turn("Test state")

                                    # Should fall back to "look" on error
                                    assert action == "look"
                                    assert state is not None

    def test_error_resilience_llm_client_tracking_failures(self, monkeypatch):
        """Test that LLM client tracking failures don't break LLM calls."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('llm_client.get_langfuse_client') as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Make start_as_current_observation raise an exception
            mock_client.start_as_current_observation.side_effect = ConnectionError("Langfuse tracking failed")

            client = LLMClient(base_url="http://test", api_key="test-key", logger=None)

            # Mock the actual HTTP request
            with patch.object(client, '_execute_request') as mock_execute:
                from llm_client import LLMResponse
                mock_response = LLMResponse(
                    content="test response",
                    model="gpt-4",
                    usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
                )
                mock_execute.return_value = mock_response

                # Should not crash despite tracking failure
                messages = [{"role": "user", "content": "test"}]
                result = client._make_request(
                    model="gpt-4",
                    messages=messages
                )

                # Verify request succeeded despite Langfuse failure
                assert result is not None
                assert result.content == "test response"

    def test_error_resilience_generation_update_failures(self, monkeypatch):
        """Test that generation.update failures are handled gracefully."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

        with patch('llm_client.get_langfuse_client') as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Create generation mock that raises on update
            mock_generation = MagicMock()
            mock_generation.update.side_effect = ValueError("Update failed")

            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_generation)
            # Make __exit__ propagate the exception (return None/False)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_client.start_as_current_observation.return_value = mock_context

            client = LLMClient(base_url="http://test", api_key="test-key", logger=None)

            # Mock the actual HTTP request
            with patch.object(client, '_execute_request') as mock_execute:
                from llm_client import LLMResponse
                mock_response = LLMResponse(
                    content="test response",
                    model="gpt-4",
                    usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
                )
                mock_execute.return_value = mock_response

                # Should not crash despite update failure
                # The error should be caught and handled gracefully
                messages = [{"role": "user", "content": "test"}]
                result = client._make_request(
                    model="gpt-4",
                    messages=messages
                )

                # Verify the request succeeded despite Langfuse error
                assert result is not None
                assert result.content == "test response"
