# ABOUTME: Tests for reasoning model detection and max_tokens handling in llm_client
# ABOUTME: Verifies reasoning models (DeepSeek R1, QwQ) get increased max_tokens by default

import pytest
from unittest.mock import Mock, patch
from llm_client import LLMClient


class TestReasoningModelDetection:
    """Test detection of reasoning models."""

    def test_deepseek_r1_detected_as_reasoning_model(self, test_config):
        """Verify deepseek-r1 is detected as reasoning model."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "deepseek-r1",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": "test"}]
            )

            # Verify request was made
            assert mock_post.called
            payload = mock_post.call_args[1]['json']
            # All messages should be "user" role for reasoning models
            assert all(msg["role"] == "user" for msg in payload["messages"])

    def test_deepseek_reasoner_detected_as_reasoning_model(self, test_config):
        """Verify deepseek-reasoner is detected as reasoning model."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "deepseek-reasoner",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": "test"}]
            )

            # Verify request was made
            assert mock_post.called
            payload = mock_post.call_args[1]['json']
            # All messages should be "user" role for reasoning models
            assert all(msg["role"] == "user" for msg in payload["messages"])

    def test_qwq_detected_as_reasoning_model(self, test_config):
        """Verify qwq is detected as reasoning model."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "qwq-32b-preview",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="qwq-32b-preview",
                messages=[{"role": "user", "content": "test"}]
            )

            # Verify request was made
            assert mock_post.called
            payload = mock_post.call_args[1]['json']
            # All messages should be "user" role for reasoning models
            assert all(msg["role"] == "user" for msg in payload["messages"])

    def test_o1_detected_as_reasoning_model(self, test_config):
        """Verify o1 models are detected as reasoning models."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "o1-preview",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="o1-preview",
                messages=[{"role": "user", "content": "test"}]
            )

            # Verify request was made
            assert mock_post.called
            payload = mock_post.call_args[1]['json']
            # All messages should be "user" role for reasoning models
            assert all(msg["role"] == "user" for msg in payload["messages"])

    def test_non_reasoning_model_not_detected(self, test_config):
        """Verify non-reasoning models are not detected as reasoning models."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "previous response"}
                ]
            )

            # Verify request was made
            assert mock_post.called
            payload = mock_post.call_args[1]['json']
            # Should preserve original roles for non-reasoning models
            assert payload["messages"][0]["role"] == "user"
            assert payload["messages"][1]["role"] == "assistant"


class TestReasoningModelMaxTokens:
    """Test max_tokens handling for reasoning models."""

    def test_reasoning_model_with_no_max_tokens_gets_default(self, test_config):
        """Verify reasoning model with no max_tokens gets 8000 default."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "deepseek-r1",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            # Don't specify max_tokens
            client.chat_completions_create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": "test"}]
            )

            # Verify max_tokens was set to 8000
            payload = mock_post.call_args[1]['json']
            assert payload["max_tokens"] == 8000

    def test_reasoning_model_with_explicit_max_tokens_keeps_it(self, test_config):
        """Verify reasoning model with explicit max_tokens keeps it."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "qwq-32b-preview",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            # Explicitly specify max_tokens
            client.chat_completions_create(
                model="qwq-32b-preview",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=4000
            )

            # Verify explicit max_tokens was preserved
            payload = mock_post.call_args[1]['json']
            assert payload["max_tokens"] == 4000

    def test_non_reasoning_model_uses_original_max_tokens(self, test_config):
        """Verify non-reasoning model uses original max_tokens (or None)."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            # Don't specify max_tokens for non-reasoning model
            client.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}]
            )

            # Verify max_tokens was NOT added to payload
            payload = mock_post.call_args[1]['json']
            assert "max_tokens" not in payload

    def test_max_tokens_increase_is_logged(self, test_config):
        """Verify max_tokens increase is logged for debugging."""
        mock_logger = Mock()
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key", logger=mock_logger)

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "deepseek-r1",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        with patch('requests.post', return_value=mock_response):
            # Don't specify max_tokens
            client.chat_completions_create(
                model="deepseek-r1",
                messages=[{"role": "user", "content": "test"}]
            )

            # Verify logging occurred
            assert mock_logger.debug.called
            # Check if any debug call mentions max_tokens
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("max_tokens" in call.lower() or "8000" in call for call in debug_calls)

    def test_all_reasoning_models_get_default_max_tokens(self, test_config):
        """Verify all reasoning model variants get default max_tokens."""
        client = LLMClient(config=test_config, base_url="http://test.com", api_key="test-key")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "test-model",
            "choices": [{
                "message": {"content": "Test response", "role": "assistant"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        reasoning_models = [
            "deepseek-r1",
            "deepseek-reasoner",
            "qwq-32b-preview",
            "o1-preview",
            "o1-mini",
            "o3-mini"
        ]

        for model in reasoning_models:
            with patch('requests.post', return_value=mock_response) as mock_post:
                client.chat_completions_create(
                    model=model,
                    messages=[{"role": "user", "content": "test"}]
                )

                # Verify max_tokens was set to 8000
                payload = mock_post.call_args[1]['json']
                assert payload["max_tokens"] == 8000, f"Model {model} should get max_tokens=8000"
