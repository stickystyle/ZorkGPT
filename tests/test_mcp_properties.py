# ABOUTME: Property-based tests for MCP integration using Hypothesis
# ABOUTME: Validates configuration defaults, LLM client tool calling, and MCPManager invariants

import asyncio
import json
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock, AsyncMock, MagicMock, patch

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


# =============================================================================
# MCPManager Property Tests (Tasks 3.2, 3.3)
# =============================================================================


# Strategy for generating valid MCP server config data
@st.composite
def valid_mcp_config_strategy(draw):
    """Generate valid MCP configuration data."""
    server_name = draw(st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-"),
        min_size=1,
        max_size=20,
    ))
    command = draw(st.sampled_from(["echo", "node", "python", "npx"]))
    args = draw(st.lists(st.text(min_size=1, max_size=20), max_size=5))
    env_vars = draw(st.dictionaries(
        st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ_", min_size=1, max_size=20),
        st.text(min_size=0, max_size=50),
        max_size=5
    ))

    return {
        "mcpServers": {
            server_name or "default": {
                "command": command,
                "args": args,
                "env": env_vars if env_vars else None
            }
        }
    }


class TestProperty6SessionLifecycleCoupling:
    """
    Property 6: Session Lifecycle Coupling (Req 3.1, 3.5)

    Invariant: Session connect and disconnect are always properly paired.
    After connect, session is non-None. After disconnect, session is None.
    """

    @pytest.fixture
    def mock_mcp_dependencies(self):
        """Mock MCP SDK dependencies for session lifecycle tests."""
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        return mock_context, mock_session

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config_data=valid_mcp_config_strategy())
    @pytest.mark.asyncio
    async def test_property_6_session_connected_after_connect(
        self, config_data, tmp_path, mock_mcp_dependencies
    ):
        """Property 6: After connect_session(), session is non-None."""
        from managers.mcp_manager import MCPManager

        mock_context, mock_session = mock_mcp_dependencies

        # Create config file
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=MagicMock())

                await manager.connect_session()

                # Property: session must be non-None after connect
                assert manager._session is not None

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config_data=valid_mcp_config_strategy())
    @pytest.mark.asyncio
    async def test_property_6_session_none_after_disconnect(
        self, config_data, tmp_path, mock_mcp_dependencies
    ):
        """Property 6: After disconnect_session(), session is None."""
        from managers.mcp_manager import MCPManager

        mock_context, mock_session = mock_mcp_dependencies

        # Create config file
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=MagicMock())

                await manager.connect_session()
                await manager.disconnect_session()

                # Property: session must be None after disconnect
                assert manager._session is None
                assert manager._stdio_context is None

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        config_data=valid_mcp_config_strategy(),
        num_cycles=st.integers(min_value=1, max_value=5)
    )
    @pytest.mark.asyncio
    async def test_property_6_connect_disconnect_cycles(
        self, config_data, num_cycles, tmp_path, mock_mcp_dependencies
    ):
        """Property 6: Multiple connect/disconnect cycles maintain consistent state."""
        from managers.mcp_manager import MCPManager

        mock_context, mock_session = mock_mcp_dependencies

        # Create config file
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=MagicMock())

                for _ in range(num_cycles):
                    await manager.connect_session()
                    assert manager._session is not None, "Session must be non-None after connect"

                    await manager.disconnect_session()
                    assert manager._session is None, "Session must be None after disconnect"


class TestProperty9SubprocessTerminationOnDisconnect:
    """
    Property 9: Subprocess Termination on Disconnect (Req 3.7)

    Invariant: When session disconnects, the subprocess context is properly cleaned up.
    """

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config_data=valid_mcp_config_strategy())
    @pytest.mark.asyncio
    async def test_property_9_context_exit_called_on_disconnect(
        self, config_data, tmp_path
    ):
        """Property 9: __aexit__ is called on disconnect to terminate subprocess."""
        from managers.mcp_manager import MCPManager

        # Create tracking mock context fresh for each example
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        exit_called = {"value": False}

        async def track_exit(*args):
            exit_called["value"] = True
            return None

        mock_context.__aexit__ = track_exit

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        # Create config file
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=MagicMock())

                await manager.connect_session()
                assert not exit_called["value"], "Exit should not be called yet"

                await manager.disconnect_session()
                assert exit_called["value"], "Exit must be called on disconnect"

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(config_data=valid_mcp_config_strategy())
    @pytest.mark.asyncio
    async def test_property_9_context_cleared_after_disconnect(
        self, config_data, tmp_path
    ):
        """Property 9: stdio_context is set to None after disconnect."""
        from managers.mcp_manager import MCPManager

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        # Create config file
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=MagicMock())

                await manager.connect_session()
                assert manager._stdio_context is not None

                await manager.disconnect_session()
                # Property: context must be None after disconnect
                assert manager._stdio_context is None


# =============================================================================
# MCP Schema Translation Property Tests (Tasks 4.2, 4.3)
# =============================================================================


class TestProperty34ToolNameFormat:
    """
    Property 34: Tool Name Format (Req 8.2)

    Invariant: Translated tool names follow {server_name}.{tool_name} format.
    Parsing the result should roundtrip correctly.
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        server_name=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-")
        ).filter(lambda x: x and not x.startswith("-") and not x.endswith("-")),
        tool_name=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_")
        ).filter(lambda x: x and x[0].isalpha())  # Tool names typically start with letter
    )
    def test_property_tool_name_format_roundtrip(self, server_name, tool_name, tmp_path):
        """Property 34: Tool names follow {server_name}.{tool_name} format.

        Validates: Requirements 8.2

        For any server name and tool name, the translated tool name should:
        1. Follow format {server_name}.{tool_name}
        2. Parsing the result should roundtrip correctly
        """
        from managers.mcp_manager import MCPManager

        # Create minimal MCP config
        config_data = {
            "mcpServers": {
                server_name: {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        manager = MCPManager(config=config, logger=MagicMock())

        # Create mock tool with given name
        mock_tool = MagicMock()
        mock_tool.name = tool_name
        mock_tool.description = "Test tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        # Translate tool schema
        translated = manager._translate_tool_schema(tool=mock_tool, server_name=server_name)

        # Property 1: Result name matches format {server_name}.{tool_name}
        expected_name = f"{server_name}.{tool_name}"
        assert translated["function"]["name"] == expected_name, \
            f"Expected {expected_name}, got {translated['function']['name']}"

        # Property 2: Parsing the result returns original server_name and tool_name
        parsed_server, parsed_tool = manager._parse_tool_name(translated["function"]["name"])
        assert parsed_server == server_name, \
            f"Server name roundtrip failed: {server_name} != {parsed_server}"
        assert parsed_tool == tool_name, \
            f"Tool name roundtrip failed: {tool_name} != {parsed_tool}"

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        server_name=st.text(
            min_size=1,
            max_size=15,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-")
        ).filter(lambda x: x and not x.startswith("-")),
        tool_name=st.text(
            min_size=1,
            max_size=15,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_")
        ).filter(lambda x: x and x[0].isalpha())
    )
    def test_property_tool_name_has_single_dot_separator(self, server_name, tool_name, tmp_path):
        """Property 34: Tool names have exactly one dot separator.

        Validates: Requirements 8.2

        The format {server_name}.{tool_name} implies exactly one dot.
        """
        from managers.mcp_manager import MCPManager

        # Create minimal MCP config
        config_data = {
            "mcpServers": {
                server_name: {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        manager = MCPManager(config=config, logger=MagicMock())

        # Create mock tool
        mock_tool = MagicMock()
        mock_tool.name = tool_name
        mock_tool.description = "Test"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        # Translate
        translated = manager._translate_tool_schema(tool=mock_tool, server_name=server_name)
        result_name = translated["function"]["name"]

        # Property: Exactly one dot separator
        assert result_name.count(".") == 1, \
            f"Tool name must have exactly 1 dot, found {result_name.count('.')}: {result_name}"


class TestProperty37CompleteToolTranslation:
    """
    Property 37: Complete Tool Translation (Req 8.5)

    Invariant: All tools from server are translated and available.
    Each translated tool has required OpenAI format fields.
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(num_tools=st.integers(min_value=1, max_value=10))
    @pytest.mark.asyncio
    async def test_property_all_tools_translated(self, num_tools, tmp_path):
        """Property 37: All tools from server are translated and available.

        Validates: Requirements 8.5

        For any number of tools on a server:
        1. All tools should be translated
        2. Each translated tool should have required OpenAI format fields
        """
        from managers.mcp_manager import MCPManager

        # Create config
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        # Generate num_tools mock MCP tools
        mock_tools = []
        for i in range(num_tools):
            mock_tool = MagicMock()
            mock_tool.name = f"tool_{i}"
            mock_tool.description = f"Test tool {i}"
            mock_tool.inputSchema = {
                "type": "object",
                "properties": {
                    f"param_{i}": {"type": "string", "description": f"Parameter {i}"}
                },
                "required": [f"param_{i}"]
            }
            mock_tools.append(mock_tool)

        # Mock list_tools result
        mock_list_tools_result = MagicMock()
        mock_list_tools_result.tools = mock_tools

        # Mock session
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_list_tools_result)

        # Mock stdio client
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=MagicMock())

                # Connect session
                await manager.connect_session()

                # Call get_tool_schemas() (mocked)
                result = await manager.get_tool_schemas()

                # Property 1: All tools translated (count matches)
                assert len(result) == num_tools, \
                    f"Expected {num_tools} tools, got {len(result)}"

                # Property 2: Each has required OpenAI format fields
                for tool_schema in result:
                    assert "type" in tool_schema, "Missing 'type' field"
                    assert tool_schema["type"] == "function", \
                        f"Expected type='function', got {tool_schema['type']}"

                    assert "function" in tool_schema, "Missing 'function' field"

                    function_def = tool_schema["function"]
                    assert "name" in function_def, "Missing 'function.name' field"
                    assert "description" in function_def, "Missing 'function.description' field"
                    assert "parameters" in function_def, "Missing 'function.parameters' field"

                    # Verify name format (server.tool)
                    assert "." in function_def["name"], \
                        f"Tool name missing server prefix: {function_def['name']}"

                    # Verify parameters is object schema
                    parameters = function_def["parameters"]
                    assert "type" in parameters, "Missing 'parameters.type' field"
                    assert parameters["type"] == "object", \
                        f"Expected parameters.type='object', got {parameters['type']}"

                # Disconnect
                await manager.disconnect_session()

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_tools=st.integers(min_value=1, max_value=5),
        server_name=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-")
        ).filter(lambda x: x and not x.startswith("-"))
    )
    @pytest.mark.asyncio
    async def test_property_tool_names_unique_after_translation(
        self, num_tools, server_name, tmp_path
    ):
        """Property 37: Translated tool names are unique.

        Validates: Requirements 8.5

        After translation with server prefix, all tool names should be unique.
        """
        from managers.mcp_manager import MCPManager

        # Create config with given server name
        config_data = {
            "mcpServers": {
                server_name: {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        # Generate tools with unique names
        mock_tools = []
        for i in range(num_tools):
            mock_tool = MagicMock()
            mock_tool.name = f"unique_tool_{i}"
            mock_tool.description = f"Tool {i}"
            mock_tool.inputSchema = {"type": "object", "properties": {}}
            mock_tools.append(mock_tool)

        # Mock session
        mock_list_tools_result = MagicMock()
        mock_list_tools_result.tools = mock_tools

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_list_tools_result)

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=MagicMock())

                await manager.connect_session()
                result = await manager.get_tool_schemas()

                # Extract all translated names
                translated_names = [tool["function"]["name"] for tool in result]

                # Property: All names are unique
                assert len(translated_names) == len(set(translated_names)), \
                    f"Duplicate tool names found: {translated_names}"

                # Property: All names have server prefix
                for name in translated_names:
                    assert name.startswith(f"{server_name}."), \
                        f"Tool name missing server prefix: {name}"

                await manager.disconnect_session()


# =============================================================================
# Tool Execution Property Tests (Task 5.2, 5.3)
# =============================================================================


class TestProperty28ToolCallLogging:
    """
    Property 28: Tool Call Logging (Req 7.1)

    Invariant: For any tool call, the system logs the tool_name and arguments.
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        tool_name=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_")
        ).filter(lambda x: x and x[0].isalpha()),
        server_name=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-")
        ).filter(lambda x: x and not x.startswith("-")),
    )
    @pytest.mark.asyncio
    async def test_property_28_tool_call_logs_name_and_arguments(
        self, tool_name, server_name, tmp_path
    ):
        """Property 28: Tool call logging includes tool_name and arguments.

        Validates: Requirements 7.1

        For any tool call, the log entry must contain:
        - tool_name (the full prefixed name)
        - arguments (the arguments dict)
        """
        from managers.mcp_manager import MCPManager

        # Create config
        config_data = {
            "mcpServers": {
                server_name: {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
            mcp_tool_call_timeout_seconds=30,
        )

        # Setup mocks
        mock_logger = MagicMock()

        mock_call_result = MagicMock()
        mock_call_result.content = {"result": "success"}
        mock_call_result.isError = False

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=mock_logger)

                await manager.connect_session()

                # Generate test arguments
                test_args = {"param1": "value1", "param2": 42}
                full_tool_name = f"{server_name}.{tool_name}"

                # Call tool
                await manager.call_tool(full_tool_name, test_args)

                # Property: Logger must have been called with tool_name and arguments
                log_calls = mock_logger.info.call_args_list

                # Find the start log call
                start_log_found = False
                for call in log_calls:
                    if call.kwargs.get("extra"):
                        extra = call.kwargs["extra"]
                        if extra.get("event_type") == "mcp_tool_call_start":
                            # Verify tool_name
                            assert extra.get("tool_name") == full_tool_name, \
                                f"Expected tool_name={full_tool_name}, got {extra.get('tool_name')}"
                            # Verify arguments
                            assert extra.get("arguments") == test_args, \
                                f"Expected arguments={test_args}, got {extra.get('arguments')}"
                            start_log_found = True
                            break

                assert start_log_found, "Tool call start log not found"

                await manager.disconnect_session()

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        arguments=st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"))),
            st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
            max_size=5
        )
    )
    @pytest.mark.asyncio
    async def test_property_28_arbitrary_arguments_logged(self, arguments, tmp_path):
        """Property 28: Arbitrary argument dictionaries are logged correctly.

        Validates: Requirements 7.1

        For any valid arguments dict, the log entry preserves the arguments.
        """
        from managers.mcp_manager import MCPManager

        # Create config
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
            mcp_tool_call_timeout_seconds=30,
        )

        # Setup mocks
        mock_logger = MagicMock()

        mock_call_result = MagicMock()
        mock_call_result.content = {"result": "success"}
        mock_call_result.isError = False

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=mock_logger)

                await manager.connect_session()

                # Call tool with generated arguments
                await manager.call_tool("test-server.test_tool", arguments)

                # Property: Arguments are preserved in log
                log_calls = mock_logger.info.call_args_list
                for call in log_calls:
                    if call.kwargs.get("extra"):
                        extra = call.kwargs["extra"]
                        if extra.get("event_type") == "mcp_tool_call_start":
                            assert extra.get("arguments") == arguments, \
                                "Arguments not preserved in log"
                            break

                await manager.disconnect_session()


class TestProperty29ToolResultLogging:
    """
    Property 29: Tool Result Logging (Req 7.2)

    Invariant: For any tool call completion, the system logs result type, length, and duration.
    """

    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        result_content=st.one_of(
            st.text(max_size=100),
            st.dictionaries(st.text(min_size=1, max_size=10), st.text(max_size=20), max_size=3),
            st.lists(st.text(max_size=20), max_size=5),
            st.integers(),
        )
    )
    @pytest.mark.asyncio
    async def test_property_29_result_logging_includes_type_length_duration(
        self, result_content, tmp_path
    ):
        """Property 29: Tool result logging includes type, length, and duration.

        Validates: Requirements 7.2

        For any tool call completion:
        - result_type is logged
        - result_length is logged (as string length)
        - duration_ms is logged
        """
        from managers.mcp_manager import MCPManager

        # Create config
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
            mcp_tool_call_timeout_seconds=30,
        )

        # Setup mocks
        mock_logger = MagicMock()

        mock_call_result = MagicMock()
        mock_call_result.content = result_content
        mock_call_result.isError = False

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=mock_logger)

                await manager.connect_session()

                # Call tool
                await manager.call_tool("test-server.test_tool", {"arg": "value"})

                # Property: Find success log with required fields
                log_calls = mock_logger.info.call_args_list
                success_log_found = False

                for call in log_calls:
                    if call.kwargs.get("extra"):
                        extra = call.kwargs["extra"]
                        if extra.get("event_type") == "mcp_tool_call_success":
                            # Verify result_type is logged
                            assert "result_type" in extra, "result_type not logged"
                            assert extra["result_type"] == type(result_content).__name__, \
                                f"Expected result_type={type(result_content).__name__}"

                            # Verify result_length is logged
                            assert "result_length" in extra, "result_length not logged"
                            expected_length = len(str(result_content)) if result_content else 0
                            assert extra["result_length"] == expected_length, \
                                f"Expected result_length={expected_length}, got {extra['result_length']}"

                            # Verify duration_ms is logged
                            assert "duration_ms" in extra, "duration_ms not logged"
                            assert isinstance(extra["duration_ms"], (int, float)), \
                                "duration_ms must be a number"
                            assert extra["duration_ms"] >= 0, "duration_ms must be non-negative"

                            success_log_found = True
                            break

                assert success_log_found, "Tool call success log not found"

                await manager.disconnect_session()

    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        error_message=st.text(min_size=1, max_size=200)
    )
    @pytest.mark.asyncio
    async def test_property_29_error_logging_includes_duration(
        self, error_message, tmp_path
    ):
        """Property 29: Error logging also includes duration.

        Validates: Requirements 7.2

        Even on error, duration_ms should be logged.
        """
        from managers.mcp_manager import MCPManager

        # Create config
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
            mcp_tool_call_timeout_seconds=30,
        )

        # Setup mocks with error
        mock_logger = MagicMock()

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=Exception(error_message))

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                manager = MCPManager(config=config, logger=mock_logger)

                await manager.connect_session()

                # Call tool (will fail)
                result = await manager.call_tool("test-server.test_tool", {"arg": "value"})

                # Property: Error log should have duration_ms
                log_calls = mock_logger.error.call_args_list
                error_log_found = False

                for call in log_calls:
                    if call.kwargs.get("extra"):
                        extra = call.kwargs["extra"]
                        if extra.get("event_type") == "mcp_tool_call_error":
                            # Verify duration_ms is logged even on error
                            assert "duration_ms" in extra, "duration_ms not logged on error"
                            assert isinstance(extra["duration_ms"], (int, float)), \
                                "duration_ms must be a number"
                            assert extra["duration_ms"] >= 0, "duration_ms must be non-negative"

                            # Verify error is logged
                            assert "error" in extra, "error not logged"

                            error_log_found = True
                            break

                assert error_log_found, "Tool call error log not found"

                await manager.disconnect_session()


# =============================================================================
# MCP Graceful Degradation Property Tests (Tasks 6.2, 6.3)
# =============================================================================


class TestMCPGracefulDegradationProperties:
    """
    Property tests for MCP graceful degradation and retry logic.

    These tests verify that connection retry and graceful degradation
    behavior holds across many randomly generated inputs.
    """

    @pytest.mark.asyncio
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        successful_connections=st.integers(min_value=1, max_value=100)
    )
    async def test_property_26_retry_exactly_once_on_subsequent_turn(
        self, tmp_path, successful_connections
    ):
        """Property 26: Subsequent turn failures retry exactly once (Req 6.7).

        For any _successful_connections > 0 (subsequent turn),
        when connection fails, exactly one retry should be attempted.

        Validates: Requirements 6.7
        """
        from managers.mcp_manager import MCPManager

        # Create config
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        manager = MCPManager(config=config, logger=MagicMock())

        # Set successful_connections to simulate subsequent turn
        manager._successful_connections = successful_connections

        # Mock stdio_client: first call fails, second call succeeds
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        call_count = {"value": 0}

        def track_calls(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise Exception("First connection failed")
            return mock_context

        with patch("managers.mcp_manager.stdio_client", side_effect=track_calls):
            with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
                # Should succeed after retry
                await manager.connect_session()

        # Property: Exactly one retry (2 total calls to stdio_client)
        assert call_count["value"] == 2, \
            f"Expected exactly 2 connection attempts (initial + retry), got {call_count['value']}"

        # Property: Connection succeeded after retry
        assert manager._session is not None, "Session should be non-None after successful retry"

        # Property: Retry flag was set
        assert manager._retry_attempted is False, \
            "Retry flag should be reset after successful connection"

        # Property: Counter incremented
        assert manager._successful_connections == successful_connections + 1, \
            "Successful connections counter should increment after retry success"

    @pytest.mark.asyncio
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        successful_connections=st.integers(min_value=1, max_value=100)
    )
    async def test_property_27_graceful_degradation_after_retry_failure(
        self, tmp_path, successful_connections
    ):
        """Property 27: Retry failure disables MCP (Req 6.8, 6.9).

        For any _successful_connections > 0 (subsequent turn),
        when both initial and retry connections fail,
        MCP should be disabled for remainder of episode.

        Validates: Requirements 6.8, 6.9
        """
        from managers.mcp_manager import MCPManager

        # Create config
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        manager = MCPManager(config=config, logger=MagicMock())

        # Set successful_connections to simulate subsequent turn
        manager._successful_connections = successful_connections

        # Mock stdio_client: both calls fail
        call_count = {"value": 0}

        def track_failed_calls(*args, **kwargs):
            call_count["value"] += 1
            raise Exception(f"Connection failed (attempt {call_count['value']})")

        with patch("managers.mcp_manager.stdio_client", side_effect=track_failed_calls):
            # Should NOT raise exception (graceful degradation)
            await manager.connect_session()

        # Property: Manager is disabled
        assert manager._disabled is True, "Manager should be disabled after retry failure"
        assert manager.is_disabled is True, "is_disabled property should reflect disabled state"

        # Property: Session was not established
        assert manager._session is None, "Session should remain None after failed retry"

        # Property: Counter did NOT increment
        assert manager._successful_connections == successful_connections, \
            "Successful connections counter should not increment after failed retry"

        # Property: Future connect_session() calls return immediately
        with patch("managers.mcp_manager.stdio_client") as mock_stdio:
            await manager.connect_session()
            mock_stdio.assert_not_called()

    @pytest.mark.asyncio
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        any_value=st.integers(min_value=0, max_value=100)  # Just for variation
    )
    async def test_property_26_first_turn_never_retries(
        self, tmp_path, any_value
    ):
        """Property 26 (complement): First turn never retries (Req 6.1).

        For _successful_connections == 0 (first turn),
        connection failure should raise immediately without retry.

        Validates: Requirements 6.1
        """
        from managers.mcp_manager import MCPManager
        from managers.mcp_config import MCPServerStartupError

        # Create config
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        manager = MCPManager(config=config, logger=MagicMock())

        # Verify initial state
        assert manager._successful_connections == 0, "Should start with 0 successful connections"

        # Mock stdio_client: connection fails
        call_count = {"value": 0}

        def track_calls(*args, **kwargs):
            call_count["value"] += 1
            raise Exception("First turn connection failed")

        with patch("managers.mcp_manager.stdio_client", side_effect=track_calls):
            # Should raise MCPServerStartupError
            with pytest.raises(MCPServerStartupError):
                await manager.connect_session()

        # Property: No retry attempted (only 1 call to stdio_client)
        assert call_count["value"] == 1, \
            f"Expected exactly 1 connection attempt (no retry), got {call_count['value']}"

        # Property: Retry flag not set
        assert manager._retry_attempted is False, "Retry flag should not be set on first turn"

        # Property: Manager not disabled
        assert manager._disabled is False, "Manager should not be disabled on first turn failure"

        # Property: Counter did not increment
        assert manager._successful_connections == 0, \
            "Counter should remain 0 after first turn failure"

    @pytest.mark.asyncio
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        any_value=st.integers(min_value=0, max_value=100)
    )
    async def test_property_27_disabled_flag_prevents_all_connections(
        self, tmp_path, any_value
    ):
        """Property 27: Disabled flag prevents all future connection attempts.

        Once _disabled is True, connect_session() should return
        immediately without any connection attempt.

        Validates: Requirements 6.9
        """
        from managers.mcp_manager import MCPManager

        # Create config
        config_data = {
            "mcpServers": {
                "test-server": {
                    "command": "echo",
                    "args": ["test"]
                }
            }
        }
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))

        config = GameConfiguration(
            max_turns_per_episode=100,
            game_file_path="test.z5",
            mcp_enabled=True,
            mcp_config_file=str(config_file),
        )

        manager = MCPManager(config=config, logger=MagicMock())

        # Set disabled flag
        manager._disabled = True

        # Track if stdio_client is called
        with patch("managers.mcp_manager.stdio_client") as mock_stdio:
            # Call connect_session multiple times
            await manager.connect_session()
            await manager.connect_session()
            await manager.connect_session()

            # Property: stdio_client was never called
            mock_stdio.assert_not_called()

        # Property: disabled remains True
        assert manager._disabled is True, "Disabled flag should remain True"

        # Property: session remains None
        assert manager._session is None, "Session should remain None when disabled"


# =============================================================================
# Batch Error Handling Property Tests (Task #9)
# =============================================================================


class TestProperty22_23_24_BatchErrorHandling:
    """
    Property tests for batch error handling in the tool-calling loop.

    These tests verify that error handling behavior holds across
    many randomly generated inputs.
    """

    @pytest.mark.asyncio
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        num_tool_calls=st.integers(min_value=1, max_value=5)
    )
    async def test_property_22_batch_processing(
        self, num_tool_calls, test_config, tmp_path
    ):
        """Property 22: Batch Error Handling (Req 6.3).

        For any number of tool calls in a single LLM response,
        all tool calls should be treated as part of the same batch
        and processed sequentially.

        Validates: Requirements 6.3
        """
        import asyncio
        from zork_agent import ZorkAgent, MCPContext

        # Setup agent with mocked MCP manager
        mock_mcp_manager = MagicMock()
        mock_mcp_manager.call_tool = AsyncMock(return_value=ToolCallResult(
            content={"result": "success"},
            is_error=False
        ))
        mock_mcp_manager.is_disabled = False

        mock_llm_client = MagicMock()
        mock_llm_client.client = MagicMock()
        mock_llm_client.client._supports_tool_calling = MagicMock(return_value=True)

        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=test_config,
                model="gpt-4",
                client=mock_llm_client,
                mcp_manager=mock_mcp_manager,
            )
            agent.system_prompt = "Test agent"

        # Generate num_tool_calls tool calls in one response
        tool_calls = [
            ToolCall(
                id=f"call_{i}",
                type="function",
                function=FunctionCall(name=f"test.tool_{i}", arguments="{}")
            )
            for i in range(num_tool_calls)
        ]

        # First response: all tool calls in one batch
        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=tool_calls
        )

        # Second response: content (exit loop)
        second_response = LLMResponse(
            content='{"thinking": "Done", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None
        )

        mock_llm_client.client.chat_completions_create = MagicMock(
            side_effect=[first_response, second_response]
        )

        # Create context
        mcp_context = MCPContext(
            messages=[{"role": "system", "content": "Test"}],
            tool_schemas=[{"type": "function", "function": {"name": "test.tool", "description": "Test"}}],
            mcp_connected=True
        )

        # Run loop
        result = await agent._run_tool_calling_loop(mcp_context)

        # Property 1: All tools executed (count matches)
        assert mock_mcp_manager.call_tool.call_count == num_tool_calls, \
            f"Expected {num_tool_calls} tool executions, got {mock_mcp_manager.call_tool.call_count}"

        # Property 2: Sequential execution within batch (not parallel)
        # Verify call_tool was awaited num_tool_calls times
        assert len(mock_mcp_manager.call_tool.call_args_list) == num_tool_calls, \
            "All tool calls should be executed sequentially"

        # Property 3: Loop exited with content
        assert result is not None
        assert "action" in result

    @pytest.mark.asyncio
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        error_position=st.integers(min_value=0, max_value=3),
        total_tools=st.integers(min_value=2, max_value=5)
    )
    async def test_property_23_non_timeout_error_recovery(
        self, error_position, total_tools, test_config
    ):
        """Property 23: Non-Timeout Error Recovery (Req 6.4).

        For any tool that returns ToolCallResult(is_error=True),
        the next tool in the batch should still execute.

        Validates: Requirements 6.4
        """
        import asyncio
        from zork_agent import ZorkAgent, MCPContext

        # Ensure error_position is within bounds
        if error_position >= total_tools:
            error_position = total_tools - 1

        # Setup agent
        mock_mcp_manager = MagicMock()

        # Create side_effect list: error at error_position, success elsewhere
        call_results = []
        for i in range(total_tools):
            if i == error_position:
                call_results.append(ToolCallResult(
                    content=None,
                    is_error=True,
                    error_message="Tool execution failed"
                ))
            else:
                call_results.append(ToolCallResult(
                    content={"result": f"success_{i}"},
                    is_error=False
                ))

        mock_mcp_manager.call_tool = AsyncMock(side_effect=call_results)
        mock_mcp_manager.is_disabled = False

        mock_llm_client = MagicMock()
        mock_llm_client.client = MagicMock()
        mock_llm_client.client._supports_tool_calling = MagicMock(return_value=True)

        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=test_config,
                model="gpt-4",
                client=mock_llm_client,
                mcp_manager=mock_mcp_manager,
            )
            agent.system_prompt = "Test agent"

        # Generate tool calls
        tool_calls = [
            ToolCall(
                id=f"call_{i}",
                type="function",
                function=FunctionCall(name=f"test.tool_{i}", arguments="{}")
            )
            for i in range(total_tools)
        ]

        # Mock LLM responses
        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=tool_calls
        )

        second_response = LLMResponse(
            content='{"thinking": "Done", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None
        )

        mock_llm_client.client.chat_completions_create = MagicMock(
            side_effect=[first_response, second_response]
        )

        # Create context
        mcp_context = MCPContext(
            messages=[{"role": "system", "content": "Test"}],
            tool_schemas=[{"type": "function", "function": {"name": "test.tool", "description": "Test"}}],
            mcp_connected=True
        )

        # Run loop
        result = await agent._run_tool_calling_loop(mcp_context)

        # Property: All tools executed despite error at error_position
        assert mock_mcp_manager.call_tool.call_count == total_tools, \
            f"All {total_tools} tools should execute despite error at position {error_position}"

        # Property: Result was returned (loop didn't abort)
        assert result is not None
        assert "action" in result

    @pytest.mark.asyncio
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        timeout_position=st.integers(min_value=0, max_value=3),
        total_tools=st.integers(min_value=2, max_value=5)
    )
    async def test_property_24_timeout_batch_abort(
        self, timeout_position, total_tools, test_config
    ):
        """Property 24: Timeout Batch Abort (Req 6.5).

        For any tool that raises asyncio.TimeoutError,
        remaining tools in the batch should NOT be executed.

        Validates: Requirements 6.5
        """
        import asyncio
        from zork_agent import ZorkAgent, MCPContext

        # Ensure timeout_position is within bounds
        if timeout_position >= total_tools:
            timeout_position = total_tools - 1

        # Setup agent
        mock_mcp_manager = MagicMock()

        # Create side_effect list: timeout at timeout_position, success before
        call_results = []
        for i in range(total_tools):
            if i < timeout_position:
                call_results.append(ToolCallResult(
                    content={"result": f"success_{i}"},
                    is_error=False
                ))
            elif i == timeout_position:
                # Timeout error
                call_results.append(asyncio.TimeoutError("Tool call timeout"))
            else:
                # Should never be called
                call_results.append(ToolCallResult(
                    content={"result": f"success_{i}"},
                    is_error=False
                ))

        mock_mcp_manager.call_tool = AsyncMock(side_effect=call_results)
        mock_mcp_manager.is_disabled = False

        mock_llm_client = MagicMock()
        mock_llm_client.client = MagicMock()
        mock_llm_client.client._supports_tool_calling = MagicMock(return_value=True)

        with patch.object(ZorkAgent, "_load_system_prompt"):
            agent = ZorkAgent(
                config=test_config,
                model="gpt-4",
                client=mock_llm_client,
                mcp_manager=mock_mcp_manager,
            )
            agent.system_prompt = "Test agent"

        # Generate tool calls
        tool_calls = [
            ToolCall(
                id=f"call_{i}",
                type="function",
                function=FunctionCall(name=f"test.tool_{i}", arguments="{}")
            )
            for i in range(total_tools)
        ]

        # Mock LLM responses
        first_response = LLMResponse(
            content=None,
            model="gpt-4",
            finish_reason="tool_calls",
            tool_calls=tool_calls
        )

        second_response = LLMResponse(
            content='{"thinking": "Done", "action": "wait"}',
            model="gpt-4",
            finish_reason="stop",
            tool_calls=None
        )

        mock_llm_client.client.chat_completions_create = MagicMock(
            side_effect=[first_response, second_response]
        )

        # Create context
        mcp_context = MCPContext(
            messages=[{"role": "system", "content": "Test"}],
            tool_schemas=[{"type": "function", "function": {"name": "test.tool", "description": "Test"}}],
            mcp_connected=True
        )

        # Run loop
        result = await agent._run_tool_calling_loop(mcp_context)

        # Property: Only tools up to and including timeout_position executed
        expected_calls = timeout_position + 1
        assert mock_mcp_manager.call_tool.call_count == expected_calls, \
            f"Expected {expected_calls} tool calls (up to timeout), got {mock_mcp_manager.call_tool.call_count}"

        # Property: Remaining tools (after timeout) were NOT executed
        remaining_tools = total_tools - expected_calls
        if remaining_tools > 0:
            # Verify we didn't call more than expected
            assert mock_mcp_manager.call_tool.call_count < total_tools, \
                f"Timeout should abort batch, but all {total_tools} tools were executed"

        # Property: Result was returned (loop didn't crash)
        assert result is not None
        assert "action" in result
