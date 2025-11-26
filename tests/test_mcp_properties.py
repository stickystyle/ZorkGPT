# ABOUTME: Property-based tests for MCP integration using Hypothesis
# ABOUTME: Validates configuration defaults and LLM client tool calling invariants

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock

from session.game_configuration import GameConfiguration
from llm_client import (
    LLMClient,
    FunctionCall,
    ToolCall,
    ToolCallResult,
    LLMResponse
)


class TestMCPConfigurationProperties:
    """Property-based tests for MCP configuration (Task 1.2).

    These tests verify that configuration properties hold across many
    randomly generated inputs, providing stronger guarantees than example-based tests.
    """

    @settings(max_examples=100)
    @given(
        max_turns=st.integers(min_value=1, max_value=10000),
    )
    def test_property_45_mcp_enabled_always_defaults_false(self, max_turns: int):
        """Property 45: mcp_enabled defaults to False for any valid config.

        Validates: Requirement - MCP is disabled by default.
        """
        config = GameConfiguration(
            max_turns_per_episode=max_turns,
            game_file_path="test.z5",
        )
        assert config.mcp_enabled is False

    @settings(max_examples=100)
    @given(
        max_turns=st.integers(min_value=1, max_value=10000),
    )
    def test_property_45_mcp_config_file_always_defaults_to_json(self, max_turns: int):
        """Property 45: mcp_config_file defaults to 'mcp_config.json' (Req 13.4).

        Validates: Requirements 13.4
        """
        config = GameConfiguration(
            max_turns_per_episode=max_turns,
            game_file_path="test.z5",
        )
        assert config.mcp_config_file == "mcp_config.json"

    @settings(max_examples=100)
    @given(
        max_turns=st.integers(min_value=1, max_value=10000),
    )
    def test_property_45_mcp_max_tool_iterations_always_defaults_20(
        self, max_turns: int
    ):
        """Property 45: mcp_max_tool_iterations defaults to 20 (Req 13.1).

        Validates: Requirements 13.1
        """
        config = GameConfiguration(
            max_turns_per_episode=max_turns,
            game_file_path="test.z5",
        )
        assert config.mcp_max_tool_iterations == 20

    @settings(max_examples=100)
    @given(
        max_turns=st.integers(min_value=1, max_value=10000),
    )
    def test_property_45_mcp_tool_call_timeout_always_defaults_30(
        self, max_turns: int
    ):
        """Property 45: mcp_tool_call_timeout_seconds defaults to 30 (Req 13.2).

        Validates: Requirements 13.2
        """
        config = GameConfiguration(
            max_turns_per_episode=max_turns,
            game_file_path="test.z5",
        )
        assert config.mcp_tool_call_timeout_seconds == 30

    @settings(max_examples=100)
    @given(
        max_turns=st.integers(min_value=1, max_value=10000),
    )
    def test_property_45_mcp_server_startup_timeout_always_defaults_10(
        self, max_turns: int
    ):
        """Property 45: mcp_server_startup_timeout_seconds defaults to 10 (Req 13.3).

        Validates: Requirements 13.3
        """
        config = GameConfiguration(
            max_turns_per_episode=max_turns,
            game_file_path="test.z5",
        )
        assert config.mcp_server_startup_timeout_seconds == 10

    @settings(max_examples=100)
    @given(
        max_turns=st.integers(min_value=1, max_value=10000),
    )
    def test_property_45_mcp_force_tool_support_always_defaults_false(
        self, max_turns: int
    ):
        """Property 45: mcp_force_tool_support defaults to False (Req 13.5).

        Validates: Requirements 13.5
        """
        config = GameConfiguration(
            max_turns_per_episode=max_turns,
            game_file_path="test.z5",
        )
        assert config.mcp_force_tool_support is False

    @settings(max_examples=100)
    @given(
        max_iterations=st.integers(min_value=1, max_value=100),
        timeout=st.integers(min_value=1, max_value=300),
        startup_timeout=st.integers(min_value=1, max_value=60),
    )
    def test_property_all_mcp_values_within_bounds(
        self,
        max_iterations: int,
        timeout: int,
        startup_timeout: int,
    ):
        """Property: MCP values must always be within documented bounds.

        Validates: Configuration constraints are enforced for any valid inputs.
        """
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_max_tool_iterations=max_iterations,
            mcp_tool_call_timeout_seconds=timeout,
            mcp_server_startup_timeout_seconds=startup_timeout,
        )

        # Values should be stored as provided when within bounds
        assert config.mcp_max_tool_iterations == max_iterations
        assert config.mcp_tool_call_timeout_seconds == timeout
        assert config.mcp_server_startup_timeout_seconds == startup_timeout

        # Additional bound checks
        assert 1 <= config.mcp_max_tool_iterations <= 100
        assert 1 <= config.mcp_tool_call_timeout_seconds <= 300
        assert 1 <= config.mcp_server_startup_timeout_seconds <= 60

    @settings(max_examples=50)
    @given(
        enabled=st.booleans(),
        force_support=st.booleans(),
    )
    def test_property_boolean_fields_preserve_value(
        self,
        enabled: bool,
        force_support: bool,
    ):
        """Property: Boolean MCP fields preserve their assigned values.

        Validates: Boolean fields correctly store True/False values.
        """
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=enabled,
            mcp_force_tool_support=force_support,
        )

        assert config.mcp_enabled is enabled
        assert config.mcp_force_tool_support is force_support

    @settings(max_examples=50)
    @given(
        config_file=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters="_-./",
            ),
            min_size=1,
            max_size=100,
        ).filter(lambda x: x.strip() and "/" not in x[:1]),  # Not empty, doesn't start with /
    )
    def test_property_config_file_path_preserved(self, config_file: str):
        """Property: mcp_config_file preserves arbitrary valid path strings.

        Validates: Configuration file paths are stored without modification.
        """
        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_config_file=config_file,
        )

        assert config.mcp_config_file == config_file


class TestProperty12ResponseContentExclusivity:
    """
    Property 12: Response Content Exclusivity (Req 4.3, 4.4)

    Invariant: An LLMResponse must have either content OR tool_calls, never both.
    """

    @settings(max_examples=50)
    @given(
        content=st.one_of(st.none(), st.text(min_size=1)),
        has_tool_calls=st.booleans()
    )
    def test_content_and_tool_calls_are_mutually_exclusive(self, content, has_tool_calls):
        """Test that content and tool_calls are mutually exclusive in responses."""
        tool_calls = None
        if has_tool_calls:
            # If we have tool_calls, content must be None
            tool_calls = [
                ToolCall(
                    id="call_123",
                    type="function",
                    function=FunctionCall(name="test", arguments="{}")
                )
            ]
            content = None

        response = LLMResponse(
            content=content,
            model="gpt-4",
            tool_calls=tool_calls
        )

        # Verify mutual exclusivity
        if response.tool_calls is not None and len(response.tool_calls) > 0:
            assert response.content is None, "When tool_calls present, content must be None"
        elif response.content is not None:
            assert response.tool_calls is None or len(response.tool_calls) == 0, \
                "When content present, tool_calls must be None or empty"

    @settings(max_examples=30)
    @given(
        num_tool_calls=st.integers(min_value=1, max_value=5)
    )
    def test_tool_calls_response_has_no_content(self, num_tool_calls):
        """Test that responses with tool_calls always have content=None."""
        tool_calls = [
            ToolCall(
                id=f"call_{i}",
                type="function",
                function=FunctionCall(name=f"tool_{i}", arguments="{}")
            )
            for i in range(num_tool_calls)
        ]

        response = LLMResponse(
            content=None,
            model="gpt-4",
            tool_calls=tool_calls,
            finish_reason="tool_calls"
        )

        assert response.content is None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == num_tool_calls

    @settings(max_examples=50)
    @given(
        content_text=st.text(min_size=1, max_size=1000)
    )
    def test_content_response_has_no_tool_calls(self, content_text):
        """Test that responses with content have no tool_calls."""
        response = LLMResponse(
            content=content_text,
            model="gpt-4",
            tool_calls=None,
            finish_reason="stop"
        )

        assert response.content == content_text
        assert response.tool_calls is None


class TestProperty13ModelCompatibilityDetection:
    """
    Property 13: Model Compatibility Detection (Req 4.5, 12.1-12.3)

    Invariant: Model compatibility detection is deterministic and consistent.
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        model_name=st.sampled_from([
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            "claude-3-opus", "claude-3-sonnet",
            "gemini-pro", "mistral-large",
            "custom-model-2025", "unknown-model"
        ])
    )
    def test_non_reasoning_models_support_tools(self, model_name, test_config):
        """Test that non-reasoning models consistently support tools."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        # Non-reasoning models should always support tools
        assert client._supports_tool_calling(model_name) is True

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        reasoning_model=st.sampled_from([
            "o1-preview", "o1-mini", "o1-pro",
            "o3-mini", "o3-preview",
            "deepseek-r1", "deepseek-reasoner",
            "qwq-32b", "qwq"
        ])
    )
    def test_reasoning_models_do_not_support_tools(self, reasoning_model, test_config):
        """Test that reasoning models consistently don't support tools."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        # Reasoning models should never support tools (unless forced)
        assert client._supports_tool_calling(reasoning_model) is False

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        model_name=st.text(min_size=1, max_size=100)
    )
    def test_model_compatibility_is_deterministic(self, model_name, test_config):
        """Test that calling _supports_tool_calling multiple times gives same result."""
        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        # Call multiple times
        result1 = client._supports_tool_calling(model_name)
        result2 = client._supports_tool_calling(model_name)
        result3 = client._supports_tool_calling(model_name)

        # Should always get the same result
        assert result1 == result2 == result3

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        reasoning_model=st.sampled_from([
            "o1-preview", "deepseek-r1", "qwq"
        ])
    )
    def test_force_tool_support_overrides_detection(self, reasoning_model, test_config):
        """Test that force_tool_support config overrides auto-detection."""
        # Set force_tool_support flag
        test_config.mcp_force_tool_support = True

        client = LLMClient(
            config=test_config,
            base_url="http://test.com",
            api_key="test-key"
        )

        # Even reasoning models should return True when forced
        assert client._supports_tool_calling(reasoning_model) is True


class TestProperty43ToolCallResultSerialization:
    """
    Property 43: ToolCallResult Serialization (Req 11.3)

    Invariant: ToolCallResult.to_dict() is lossless and idempotent for JSON serialization.
    """

    @settings(max_examples=50)
    @given(
        content=st.one_of(
            st.text(),
            st.integers(),
            st.floats(allow_nan=False),
            st.booleans(),
            st.lists(st.text()),
            st.dictionaries(st.text(min_size=1), st.text())
        )
    )
    def test_success_result_serialization_preserves_content(self, content):
        """Test that successful ToolCallResult serialization preserves content."""
        result = ToolCallResult(content=content, is_error=False)
        serialized = result.to_dict()

        assert "content" in serialized
        assert serialized["content"] == content
        assert "error" not in serialized

    @settings(max_examples=30)
    @given(
        error_message=st.text(min_size=1, max_size=500)
    )
    def test_error_result_serialization_includes_error_field(self, error_message):
        """Test that error ToolCallResult serialization includes error field."""
        result = ToolCallResult(
            content=None,
            is_error=True,
            error_message=error_message
        )
        serialized = result.to_dict()

        assert "error" in serialized
        assert serialized["error"] == error_message
        assert "content" in serialized
        assert serialized["content"] is None

    @settings(max_examples=30)
    @given(
        content=st.one_of(
            st.text(),
            st.lists(st.integers()),
            st.dictionaries(st.text(min_size=1), st.integers())
        )
    )
    def test_serialization_is_idempotent(self, content):
        """Test that calling to_dict() multiple times gives same result."""
        result = ToolCallResult(content=content, is_error=False)

        # Call multiple times
        dict1 = result.to_dict()
        dict2 = result.to_dict()
        dict3 = result.to_dict()

        # Should always get the same result
        assert dict1 == dict2 == dict3

    @settings(max_examples=50)
    @given(
        content=st.one_of(st.none(), st.text(), st.integers()),
        is_error=st.booleans()
    )
    def test_serialization_round_trip_structure(self, content, is_error):
        """Test that serialized result has correct structure."""
        error_message = "Error occurred" if is_error else None
        result = ToolCallResult(
            content=content,
            is_error=is_error,
            error_message=error_message
        )
        serialized = result.to_dict()

        # Verify structure
        assert isinstance(serialized, dict)
        assert "content" in serialized

        if is_error:
            assert "error" in serialized
            assert serialized["error"] == error_message
        else:
            assert "error" not in serialized


class TestProperty44ToolCallArgumentParsing:
    """
    Property 44: Tool Call Arguments are JSON Strings

    Invariant: ToolCall.function.arguments is always a JSON string, never a dict.
    """

    @settings(max_examples=50)
    @given(
        tool_name=st.text(min_size=1, max_size=50),
        arguments_str=st.text(min_size=2, max_size=200)
    )
    def test_tool_call_arguments_are_strings(self, tool_name, arguments_str):
        """Test that FunctionCall.arguments is always a string."""
        func_call = FunctionCall(name=tool_name, arguments=arguments_str)

        assert isinstance(func_call.arguments, str)
        assert func_call.arguments == arguments_str

    @settings(max_examples=30)
    @given(
        num_args=st.integers(min_value=0, max_value=5)
    )
    def test_tool_call_preserves_json_string_format(self, num_args):
        """Test that tool calls preserve JSON string format for arguments."""
        # Build a valid JSON string
        args_dict = {f"param_{i}": f"value_{i}" for i in range(num_args)}
        import json
        args_str = json.dumps(args_dict)

        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test_tool", arguments=args_str)
        )

        # Arguments should still be a string
        assert isinstance(tool_call.function.arguments, str)
        # Should be parseable as JSON
        parsed = json.loads(tool_call.function.arguments)
        assert parsed == args_dict
