# ABOUTME: Property-based tests for MCP integration using Hypothesis
# ABOUTME: Validates configuration defaults, LLM client tool calling, and MCPManager invariants

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
