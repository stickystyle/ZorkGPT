# ABOUTME: Tests for EmptyResponseError exception and retry behavior in llm_client
# ABOUTME: Verifies empty/whitespace responses trigger retries with proper diagnostics

import pytest
from unittest.mock import Mock, patch
from llm_client import LLMClient, EmptyResponseError, RetryableError, LLMResponse


class TestEmptyResponseError:
    """Test EmptyResponseError exception class."""

    def test_empty_response_error_inherits_from_retryable_error(self, test_config):
        """Verify EmptyResponseError inherits from RetryableError."""
        error = EmptyResponseError("Test message")
        assert isinstance(error, RetryableError)
        assert isinstance(error, Exception)

    def test_empty_response_error_message(self, test_config):
        """Verify EmptyResponseError stores message correctly."""
        message = "Empty response from model gpt-4"
        error = EmptyResponseError(message)
        assert str(error) == message


class TestEmptyResponseDetection:
    """Test detection of empty responses in _execute_request."""

    @pytest.fixture
    def mock_http_response_empty(self):
        """Create mock HTTP response with empty content (OpenAI format)."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "content": "",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 0,
                "total_tokens": 100
            }
        }
        return mock_response

    @pytest.fixture
    def mock_http_response_whitespace(self):
        """Create mock HTTP response with whitespace content."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-456",
            "model": "deepseek-r1",
            "choices": [{
                "message": {
                    "content": "   \n\t  ",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105
            }
        }
        return mock_response

    @pytest.fixture
    def mock_http_response_valid(self):
        """Create mock HTTP response with valid content."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-789",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "content": "Valid response content",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        return mock_response

    def test_empty_string_triggers_error(self, test_config, mock_http_response_empty):
        """Verify empty string response triggers EmptyResponseError."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        url = "http://test.com/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

        with patch('requests.post', return_value=mock_http_response_empty):
            with pytest.raises(EmptyResponseError) as exc_info:
                client._execute_request(url, headers, payload)

            assert "Empty response" in str(exc_info.value)
            assert "gpt-4" in str(exc_info.value)

    def test_whitespace_only_triggers_error(self, test_config, mock_http_response_whitespace):
        """Verify whitespace-only response triggers EmptyResponseError."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        url = "http://test.com/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {"model": "deepseek-r1", "messages": [{"role": "user", "content": "test"}]}

        with patch('requests.post', return_value=mock_http_response_whitespace):
            with pytest.raises(EmptyResponseError) as exc_info:
                client._execute_request(url, headers, payload)

            assert "Empty response" in str(exc_info.value)
            assert "deepseek-r1" in str(exc_info.value)

    def test_valid_response_does_not_trigger_error(self, test_config, mock_http_response_valid):
        """Verify valid response does not trigger EmptyResponseError."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        url = "http://test.com/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

        with patch('requests.post', return_value=mock_http_response_valid):
            # Should not raise EmptyResponseError
            result = client._execute_request(url, headers, payload)

            assert result.content == "Valid response content"
            assert result.model == "gpt-4"

    def test_error_message_includes_diagnostic_info(self, test_config, mock_http_response_empty):
        """Verify error message includes diagnostic info (model, response keys)."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        url = "http://test.com/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

        with patch('requests.post', return_value=mock_http_response_empty):
            with pytest.raises(EmptyResponseError) as exc_info:
                client._execute_request(url, headers, payload)

            error_msg = str(exc_info.value)
            assert "gpt-4" in error_msg
            # Should include some diagnostic info
            assert any(key in error_msg.lower() for key in ["response", "empty", "usage"])


class TestEmptyResponseRetryBehavior:
    """Test retry behavior for empty responses."""

    @pytest.fixture
    def mock_http_empty_then_valid(self):
        """Create mock HTTP responses: empty first, then valid."""
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

        return [mock_empty, mock_valid]

    def test_retry_counter_increments_for_empty_responses(self, test_config, mock_http_empty_then_valid):
        """Verify retry counter increments when empty response is retried."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        # Override retry config to ensure retries are enabled
        client.retry_config["max_retries"] = 3

        with patch('requests.post', side_effect=mock_http_empty_then_valid):
            # Should succeed on second attempt
            result = client.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}]
            )

            assert result.content == "Valid response"

    def test_max_retries_exhausted_raises_final_exception(self, test_config):
        """Verify max retries exhausted raises final EmptyResponseError."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        # Override retry config to limit retries
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

        with patch('requests.post', return_value=mock_empty):
            # Should raise after exhausting retries
            with pytest.raises(Exception) as exc_info:
                client.chat_completions_create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "test"}]
                )

            # The final exception wraps the EmptyResponseError
            assert "failed after" in str(exc_info.value).lower()

    def test_successful_retry_recovers_from_empty_response(self, test_config, mock_http_empty_then_valid):
        """Verify successful retry recovers from empty response."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        # Override retry config to ensure retries are enabled
        client.retry_config["max_retries"] = 3

        with patch('requests.post', side_effect=mock_http_empty_then_valid):
            result = client.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}]
            )

            assert result.content == "Valid response"
