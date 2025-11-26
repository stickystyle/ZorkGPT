# ABOUTME: Property-based tests for MCP configuration using Hypothesis.
# ABOUTME: Validates configuration defaults hold across all valid inputs (Property 45).

import pytest
from hypothesis import given, strategies as st, settings

from session.game_configuration import GameConfiguration


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
