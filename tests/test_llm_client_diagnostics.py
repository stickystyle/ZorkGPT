# ABOUTME: Tests for enhanced diagnostic logging in llm_client
# ABOUTME: Verifies comprehensive logging for empty responses and retry behavior

import pytest
from unittest.mock import Mock, patch, call
from llm_client import LLMClient


class TestDiagnosticLoggingForEmptyResponses:
    """Test diagnostic logging when empty responses occur."""

    def test_empty_response_logs_model_name(self, test_config):
        """Verify empty response logging includes model name."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

            try:
                client._execute_request(url, headers, payload)
            except:
                pass  # Expected to raise EmptyResponseError

            # Verify model name was logged
            assert mock_logger.warning.called
            warning_call = mock_logger.warning.call_args
            assert "gpt-4" in str(warning_call)

    def test_empty_response_logs_response_structure(self, test_config):
        """Verify empty response logging includes response structure (keys)."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "deepseek-r1",
            "choices": [{
                "message": {"content": "", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {"model": "deepseek-r1", "messages": [{"role": "user", "content": "test"}]}

            try:
                client._execute_request(url, headers, payload)
            except:
                pass

            # Verify response keys were logged
            assert mock_logger.warning.called
            warning_extras = mock_logger.warning.call_args[1]['extra']['extras']
            assert 'response_keys' in warning_extras
            assert isinstance(warning_extras['response_keys'], list)

    def test_empty_response_logs_token_usage(self, test_config):
        """Verify empty response logging includes token usage."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 150, "completion_tokens": 0, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

            try:
                client._execute_request(url, headers, payload)
            except:
                pass

            # Verify usage was logged
            assert mock_logger.warning.called
            warning_extras = mock_logger.warning.call_args[1]['extra']['extras']
            assert 'usage' in warning_extras
            assert warning_extras['usage']['prompt_tokens'] == 150

    def test_empty_response_logs_event_type(self, test_config):
        """Verify empty response has proper event_type."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "   ", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

            try:
                client._execute_request(url, headers, payload)
            except:
                pass

            # Verify event_type
            assert mock_logger.warning.called
            warning_extras = mock_logger.warning.call_args[1]['extra']['extras']
            assert warning_extras['event_type'] == 'empty_response_detected'


class TestRetryDiagnosticLogging:
    """Test diagnostic logging during retry attempts."""

    def test_retry_logs_include_empty_response_flag(self, test_config):
        """Verify retry logs include empty_response flag when applicable."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)
        client.retry_config["max_retries"] = 2

        mock_empty = Mock()
        mock_empty.ok = True
        mock_empty.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
        }

        mock_valid = Mock()
        mock_valid.ok = True
        mock_valid.json.return_value = {
            "id": "chatcmpl-456",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "Valid response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', side_effect=[mock_empty, mock_valid]):
            client.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}]
            )

            # Find the retry warning log
            warning_calls = [call for call in mock_logger.warning.call_args_list
                           if 'retry' in str(call).lower() or 'attempt' in str(call).lower()]

            assert len(warning_calls) > 0, "Should have retry warning logs"

    def test_retry_logs_include_max_tokens_for_reasoning_models(self, test_config):
        """Verify retry logs include max_tokens info for reasoning models."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)
        client.retry_config["max_retries"] = 2

        mock_empty = Mock()
        mock_empty.ok = True
        mock_empty.json.return_value = {
            "id": "chatcmpl-123",
            "model": "deepseek-r1",
            "choices": [{
                "message": {"content": "", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
        }

        mock_valid = Mock()
        mock_valid.ok = True
        mock_valid.json.return_value = {
            "id": "chatcmpl-456",
            "model": "deepseek-r1",
            "choices": [{
                "message": {"content": "Valid response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', side_effect=[mock_empty, mock_valid]):
            # Don't specify max_tokens - should use default 8000
            client.chat_completions_create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": "test"}]
            )

            # Verify max_tokens was logged during setup
            debug_calls = [call for call in mock_logger.debug.call_args_list
                          if 'max_tokens' in str(call).lower()]
            assert len(debug_calls) > 0, "Should log max_tokens for reasoning models"

    def test_successful_retry_logs_recovery(self, test_config):
        """Verify successful retry logs recovery message."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)
        client.retry_config["max_retries"] = 3

        mock_empty = Mock()
        mock_empty.ok = True
        mock_empty.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
        }

        mock_valid = Mock()
        mock_valid.ok = True
        mock_valid.json.return_value = {
            "id": "chatcmpl-456",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "Valid response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', side_effect=[mock_empty, mock_valid]):
            result = client.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}]
            )

            # Should succeed and log warnings about retry
            assert result.content == "Valid response"
            assert mock_logger.warning.called

    def test_exhausted_retries_logs_cumulative_diagnostics(self, test_config):
        """Verify exhausted retries logs cumulative diagnostic info."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)
        client.retry_config["max_retries"] = 1

        mock_empty = Mock()
        mock_empty.ok = True
        mock_empty.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}
        }

        with patch('requests.post', return_value=mock_empty):
            with pytest.raises(Exception):
                client.chat_completions_create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "test"}]
                )

            # Should have logged warnings for each attempt
            assert mock_logger.warning.call_count >= 2  # At least 2 retries
