# ABOUTME: Tests for LLMClient tool calling support
# ABOUTME: Validates tool call dataclasses, parsing, and OpenAI format compatibility

import pytest
from unittest.mock import Mock, patch
from llm_client import (
    LLMClient,
    FunctionCall,
    ToolCall,
    ToolCallResult,
    LLMResponse
)


class TestDataclasses:
    """Test tool call dataclasses."""

    def test_function_call_creation(self):
        """Test FunctionCall dataclass creation."""
        func_call = FunctionCall(
            name="thoughtbox.think",
            arguments='{"query": "test"}'
        )
        assert func_call.name == "thoughtbox.think"
        assert func_call.arguments == '{"query": "test"}'

    def test_tool_call_creation(self):
        """Test ToolCall dataclass creation."""
        func_call = FunctionCall(
            name="thoughtbox.think",
            arguments='{"query": "test"}'
        )
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=func_call
        )
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "thoughtbox.think"

    def test_tool_call_result_success(self):
        """Test ToolCallResult for successful execution."""
        result = ToolCallResult(
            content={"answer": "test result"},
            is_error=False
        )
        assert result.content == {"answer": "test result"}
        assert result.is_error is False
        assert result.error_message is None

    def test_tool_call_result_error(self):
        """Test ToolCallResult for failed execution."""
        result = ToolCallResult(
            content=None,
            is_error=True,
            error_message="Tool execution failed"
        )
        assert result.content is None
        assert result.is_error is True
        assert result.error_message == "Tool execution failed"

    def test_tool_call_result_to_dict_success(self):
        """Test ToolCallResult.to_dict() for success case."""
        result = ToolCallResult(
            content={"answer": "test"},
            is_error=False
        )
        result_dict = result.to_dict()
        assert result_dict == {"content": {"answer": "test"}}

    def test_tool_call_result_to_dict_error(self):
        """Test ToolCallResult.to_dict() for error case."""
        result = ToolCallResult(
            content=None,
            is_error=True,
            error_message="Failed"
        )
        result_dict = result.to_dict()
        assert result_dict == {"error": "Failed", "content": None}

    def test_tool_call_result_to_dict_with_string_content(self):
        """Test ToolCallResult.to_dict() with string content."""
        result = ToolCallResult(
            content="Simple string result",
            is_error=False
        )
        result_dict = result.to_dict()
        assert result_dict == {"content": "Simple string result"}

    def test_tool_call_result_to_dict_with_list_content(self):
        """Test ToolCallResult.to_dict() with list content."""
        result = ToolCallResult(
            content=[1, 2, 3],
            is_error=False
        )
        result_dict = result.to_dict()
        assert result_dict == {"content": [1, 2, 3]}


class TestLLMResponseExtension:
    """Test extended LLMResponse with tool_calls and finish_reason."""

    def test_llm_response_with_content(self):
        """Test LLMResponse with standard content response."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            tool_calls=None,
            finish_reason="stop"
        )
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.tool_calls is None
        assert response.finish_reason == "stop"

    def test_llm_response_with_tool_calls(self):
        """Test LLMResponse with tool calls (content should be None)."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test_tool", arguments='{}')
        )
        response = LLMResponse(
            content=None,
            model="gpt-4",
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        assert response.content is None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_123"
        assert response.finish_reason == "tool_calls"

    def test_llm_response_backward_compatibility(self):
        """Test LLMResponse works with old code (content only)."""
        response = LLMResponse(
            content="Old style response",
            model="gpt-4"
        )
        assert response.content == "Old style response"
        assert response.tool_calls is None
        assert response.finish_reason is None


class TestToolCallParsing:
    """Test tool call parsing from LLM responses."""

    def test_parse_empty_tool_calls_list(self, test_config):
        """Test parsing response with empty tool_calls list."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I decided not to use tools",
                    "tool_calls": []  # Empty list
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}]
            }

            result = client._execute_request(url, headers, payload)

            # Empty list should be treated as "no tool calls"
            assert result.content == "I decided not to use tools"
            assert result.tool_calls is None or len(result.tool_calls) == 0
            assert result.finish_reason == "stop"

    def test_parse_malformed_tool_call_in_batch(self, test_config):
        """Test parsing response with one malformed tool call in batch."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "tool_a",
                                "arguments": '{"param": "a"}'  # Valid
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "tool_b"
                                # Missing "arguments" key - malformed
                            }
                        },
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": {
                                "name": "tool_c",
                                "arguments": '{"param": "c"}'  # Valid
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}]
            }

            result = client._execute_request(url, headers, payload)

            # Should parse valid tool calls, skip malformed ones
            assert result.content is None
            assert result.tool_calls is not None
            assert len(result.tool_calls) == 2  # Only call_1 and call_3
            assert result.tool_calls[0].function.name == "tool_a"
            assert result.tool_calls[1].function.name == "tool_c"

            # Verify warning was logged
            mock_logger.warning.assert_called()

    def test_parse_invalid_json_arguments(self, test_config):
        """Test parsing response with invalid JSON in tool call arguments."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "tool_a",
                                "arguments": '{invalid json}'  # Invalid JSON
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "tool_b",
                                "arguments": '{"param": "b"}'  # Valid
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}]
            }

            result = client._execute_request(url, headers, payload)

            # Should skip invalid JSON, parse valid tool call
            assert result.content is None
            assert result.tool_calls is not None
            assert len(result.tool_calls) == 1  # Only call_2
            assert result.tool_calls[0].function.name == "tool_b"

            # Verify warning was logged
            mock_logger.warning.assert_called()

    def test_parse_tool_calls_from_response(self, test_config):
        """Test parsing tool_calls from LLM response."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "thoughtbox.think",
                                "arguments": '{"query": "test"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}],
                "tools": [{"type": "function", "function": {"name": "thoughtbox.think"}}]
            }

            result = client._execute_request(url, headers, payload)

            assert result.content is None
            assert result.tool_calls is not None
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "call_abc123"
            assert result.tool_calls[0].type == "function"
            assert result.tool_calls[0].function.name == "thoughtbox.think"
            assert result.tool_calls[0].function.arguments == '{"query": "test"}'
            assert result.finish_reason == "tool_calls"

    def test_parse_multiple_tool_calls(self, test_config):
        """Test parsing multiple tool_calls from single response."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "tool_a",
                                "arguments": '{"param": "a"}'
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "tool_b",
                                "arguments": '{"param": "b"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}]
            }

            result = client._execute_request(url, headers, payload)

            assert result.content is None
            assert len(result.tool_calls) == 2
            assert result.tool_calls[0].function.name == "tool_a"
            assert result.tool_calls[1].function.name == "tool_b"

    def test_parse_content_response_no_tool_calls(self, test_config):
        """Test parsing standard content response (no tool_calls)."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "This is a normal response"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}
        }

        with patch('requests.post', return_value=mock_response):
            url = "http://test.com/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}]
            }

            result = client._execute_request(url, headers, payload)

            assert result.content == "This is a normal response"
            assert result.tool_calls is None
            assert result.finish_reason == "stop"

    def test_tool_calls_exclusive_with_content(self, test_config):
        """Test that tool_calls and content are mutually exclusive."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        # Test tool_calls response
        mock_tool_response = Mock()
        mock_tool_response.ok = True
        mock_tool_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "test", "arguments": "{}"}
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        }

        with patch('requests.post', return_value=mock_tool_response):
            result = client._execute_request(
                "http://test.com/chat/completions",
                {"Content-Type": "application/json"},
                {"model": "gpt-4", "messages": []}
            )
            # When tool_calls present, content is None
            assert result.content is None
            assert result.tool_calls is not None

        # Test content response
        mock_content_response = Mock()
        mock_content_response.ok = True
        mock_content_response.json.return_value = {
            "id": "chatcmpl-456",
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Test content"
                },
                "finish_reason": "stop"
            }]
        }

        with patch('requests.post', return_value=mock_content_response):
            result = client._execute_request(
                "http://test.com/chat/completions",
                {"Content-Type": "application/json"},
                {"model": "gpt-4", "messages": []}
            )
            # When content present, tool_calls is None
            assert result.content == "Test content"
            assert result.tool_calls is None


class TestToolsParameterSupport:
    """Test tools and tool_choice parameters in chat_completions_create."""

    def test_chat_completions_create_accepts_tools_parameter(self, test_config):
        """Test that chat_completions_create accepts tools parameter."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop"
            }]
        }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "thoughtbox.think",
                    "description": "Think about a query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        with patch('requests.post', return_value=mock_response) as mock_post:
            # Should not raise error
            result = client.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                tools=tools
            )
            assert result.content == "test"

            # Verify tools were included in request payload
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert 'tools' in payload
            assert payload['tools'] == tools

    def test_chat_completions_create_accepts_tool_choice_parameter(self, test_config):
        """Test that chat_completions_create accepts tool_choice parameter."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop"
            }]
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            # Test with tool_choice="auto"
            client.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                tools=[{"type": "function", "function": {"name": "test"}}],
                tool_choice="auto"
            )

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert 'tool_choice' in payload
            assert payload['tool_choice'] == "auto"

    def test_tools_parameter_not_included_when_none(self, test_config):
        """Test that tools parameter is not included in payload when None."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop"
            }]
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                tools=None
            )

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert 'tools' not in payload


class TestCacheControl:
    """Test cache_control metadata on messages."""

    def test_cache_control_added_to_system_messages(self, test_config):
        """Test that cache_control is added to system role messages."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop"
            }]
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello"}
                ]
            )

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            messages = payload['messages']

            # System message should have cache_control
            assert messages[0]['role'] == 'system'
            assert 'cache_control' in messages[0]
            assert messages[0]['cache_control'] == {"type": "ephemeral"}

    def test_cache_control_added_to_user_messages(self, test_config):
        """Test that cache_control is added to user role messages."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop"
            }]
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "Hello"}
                ]
            )

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            messages = payload['messages']

            # User message should have cache_control
            assert messages[0]['role'] == 'user'
            assert 'cache_control' in messages[0]
            assert messages[0]['cache_control'] == {"type": "ephemeral"}

    def test_cache_control_not_added_to_assistant_messages(self, test_config):
        """Test that cache_control is NOT added to assistant role messages."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop"
            }]
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"}
                ]
            )

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            messages = payload['messages']

            # Assistant message should NOT have cache_control
            assert messages[1]['role'] == 'assistant'
            assert 'cache_control' not in messages[1]

    def test_cache_control_preserves_existing_cache_control(self, test_config):
        """Test that existing cache_control in messages is preserved."""
        mock_logger = Mock()
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key",
            logger=mock_logger
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop"
            }]
        }

        with patch('requests.post', return_value=mock_response) as mock_post:
            client.chat_completions_create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "Test",
                        "cache_control": {"type": "persistent"}
                    }
                ]
            )

            call_args = mock_post.call_args
            payload = call_args[1]['json']
            messages = payload['messages']

            # Should keep existing cache_control
            assert messages[0]['cache_control'] == {"type": "persistent"}


class TestModelCompatibility:
    """Test model compatibility checking for tool support."""

    def test_supports_tool_calling_gpt_models(self, test_config):
        """Test that GPT models support tool calling."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        assert client._supports_tool_calling("gpt-4") is True
        assert client._supports_tool_calling("gpt-4-turbo") is True
        assert client._supports_tool_calling("gpt-3.5-turbo") is True

    def test_supports_tool_calling_claude_models(self, test_config):
        """Test that Claude models support tool calling."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        assert client._supports_tool_calling("claude-3-opus") is True
        assert client._supports_tool_calling("claude-3-sonnet") is True

    def test_does_not_support_o1_models(self, test_config):
        """Test that o1 models do NOT support tool calling."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        assert client._supports_tool_calling("o1-preview") is False
        assert client._supports_tool_calling("o1-mini") is False

    def test_does_not_support_o3_models(self, test_config):
        """Test that o3 models do NOT support tool calling."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        assert client._supports_tool_calling("o3-mini") is False
        assert client._supports_tool_calling("o3-preview") is False

    def test_does_not_support_deepseek_r1(self, test_config):
        """Test that DeepSeek R1 models do NOT support tool calling."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        assert client._supports_tool_calling("deepseek-r1") is False
        assert client._supports_tool_calling("deepseek-reasoner") is False

    def test_does_not_support_qwq(self, test_config):
        """Test that QwQ models do NOT support tool calling."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        assert client._supports_tool_calling("qwq-32b") is False
        assert client._supports_tool_calling("qwq") is False

    def test_permissive_default_for_unknown_models(self, test_config):
        """Test that unknown models are assumed to support tools (permissive)."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        # Unknown models should return True (permissive default)
        assert client._supports_tool_calling("new-model-2025") is True
        assert client._supports_tool_calling("custom-model") is True

    def test_force_tool_support_override(self, test_config):
        """Test that force_tool_support config overrides detection."""
        # Set force_tool_support in config
        test_config.mcp_force_tool_support = True

        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        # Even o1 models should return True when forced
        assert client._supports_tool_calling("o1-preview") is True
        assert client._supports_tool_calling("deepseek-r1") is True
        assert client._supports_tool_calling("qwq") is True
