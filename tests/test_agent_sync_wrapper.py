# ABOUTME: Tests for ZorkAgent sync wrapper implementation (Task #13.1)
# ABOUTME: Validates asyncio.run boundary, MCP enabled/disabled paths, backward compatibility

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, Any

from zork_agent import ZorkAgent, AgentResponse
from session.game_configuration import GameConfiguration


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_mcp_manager():
    """Mock MCPManager for testing."""
    manager = MagicMock()
    manager.connect_session = AsyncMock()
    manager.disconnect_session = AsyncMock()
    manager.get_tool_schemas = AsyncMock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "test.tool",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    manager.is_disabled = False
    manager._server_name = "test-server"
    return manager


@pytest.fixture
def mock_mcp_manager_disabled():
    """Mock disabled MCPManager for testing."""
    manager = MagicMock()
    manager.is_disabled = True
    manager._server_name = "test-server"
    return manager


@pytest.fixture
def mock_llm_client():
    """Mock LLM client wrapper for testing."""
    wrapper = MagicMock()
    wrapper.client = MagicMock()
    wrapper.client._supports_tool_calling = MagicMock(return_value=True)
    wrapper.client.chat_completions_create = MagicMock()
    return wrapper


@pytest.fixture
def test_config():
    """GameConfiguration for testing - loads from pyproject.toml."""
    return GameConfiguration.from_toml()


@pytest.fixture
def agent_with_mcp(test_config, mock_llm_client, mock_mcp_manager):
    """Agent instance with MCP manager enabled."""
    with patch.object(ZorkAgent, "_load_system_prompt"):
        agent = ZorkAgent(
            config=test_config,
            model="gpt-4",
            client=mock_llm_client,
            mcp_manager=mock_mcp_manager,
        )
        agent.system_prompt = "You are a Zork agent."
    return agent


@pytest.fixture
def agent_without_mcp(test_config, mock_llm_client):
    """Agent instance without MCP manager."""
    with patch.object(ZorkAgent, "_load_system_prompt"):
        agent = ZorkAgent(
            config=test_config,
            model="gpt-4",
            client=mock_llm_client,
            mcp_manager=None,
        )
        agent.system_prompt = "You are a Zork agent."
    return agent


@pytest.fixture
def agent_with_mcp_disabled(test_config, mock_llm_client, mock_mcp_manager_disabled):
    """Agent instance with MCP manager but disabled."""
    with patch.object(ZorkAgent, "_load_system_prompt"):
        agent = ZorkAgent(
            config=test_config,
            model="gpt-4",
            client=mock_llm_client,
            mcp_manager=mock_mcp_manager_disabled,
        )
        agent.system_prompt = "You are a Zork agent."
    return agent


@pytest.fixture
def sample_game_state():
    """Sample game state for testing."""
    return "West of House\nYou are standing in an open field west of a white house."


@pytest.fixture
def sample_memories():
    """Sample memory text for testing."""
    return "Previous actions: moved north, opened mailbox"


# ============================================================================
# Test Classes
# ============================================================================


class TestSyncWrapperMCPEnabled:
    """Test sync wrapper with MCP enabled."""

    def test_calls_generate_action_async(self, agent_with_mcp, sample_game_state):
        """
        Verify get_action_with_reasoning calls _generate_action_async via asyncio.run.

        Test approach:
        1. Mock _generate_action_async to return known dict
        2. Call get_action_with_reasoning
        3. Verify _generate_action_async was called with correct args
        4. Verify return value matches async result
        """
        # Arrange
        expected_async_result = {
            "action": "look",
            "reasoning": "Looking around",
            "new_objective": None,
            "raw_response": '{"thinking": "Looking around", "action": "look"}',
        }

        # Mock _generate_action_async to return expected result
        async def mock_generate_async(*args, **kwargs):
            return expected_async_result

        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ) as mock_async:
            # Act
            result = agent_with_mcp.get_action_with_reasoning(sample_game_state)

            # Assert
            mock_async.assert_called_once_with(sample_game_state, None)
            assert result == expected_async_result, "Result should match async implementation"

    def test_calls_generate_action_async_with_memories(
        self, agent_with_mcp, sample_game_state, sample_memories
    ):
        """
        Verify get_action_with_reasoning passes memories to _generate_action_async.

        Test approach:
        1. Mock _generate_action_async
        2. Call get_action_with_reasoning with memories
        3. Verify memories were passed to async method
        """
        # Arrange
        expected_result = {
            "action": "north",
            "reasoning": "Going north",
            "new_objective": None,
            "raw_response": "{}",
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ) as mock_async:
            # Act
            result = agent_with_mcp.get_action_with_reasoning(
                sample_game_state, sample_memories
            )

            # Assert
            mock_async.assert_called_once_with(sample_game_state, sample_memories)
            assert result == expected_result

    def test_mcp_session_connected_during_action_generation(
        self, agent_with_mcp, sample_game_state, mock_mcp_manager
    ):
        """
        Verify MCP session is connected when calling _generate_action_async.

        Test approach:
        1. Mock LLM client to return valid response
        2. Call get_action_with_reasoning
        3. Verify connect_session was called on mcp_manager
        """
        # Arrange - Mock LLM response
        agent_with_mcp.client.client.chat_completions_create.return_value = MagicMock(
            content='{"thinking": "test reasoning", "action": "look", "new_objective": null}',
            tool_calls=None
        )

        # Act
        result = agent_with_mcp.get_action_with_reasoning(sample_game_state)

        # Assert - Verify session was connected
        mock_mcp_manager.connect_session.assert_called_once()
        mock_mcp_manager.disconnect_session.assert_called_once()
        assert result["action"] == "look"
        assert result["reasoning"] == "test reasoning"

    def test_returns_cleaned_action(self, agent_with_mcp, sample_game_state):
        """
        Verify action is cleaned via _clean_action when returned.

        Test approach:
        1. Mock _generate_action_async to return action with whitespace/formatting
        2. Mock _clean_action to return cleaned version
        3. Verify result contains cleaned action
        """
        # Arrange
        dirty_action = "  LOOK  \n"
        clean_action = "look"

        async def mock_generate_async(*args, **kwargs):
            return {
                "action": dirty_action,
                "reasoning": "test",
                "new_objective": None,
                "raw_response": "{}",
            }

        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ), patch.object(
            agent_with_mcp, "_clean_action", return_value=clean_action
        ) as mock_clean:
            # Act
            result = agent_with_mcp.get_action_with_reasoning(sample_game_state)

            # Assert
            # Verify _clean_action was called and the result contains the cleaned action
            mock_clean.assert_called_once_with(dirty_action)
            assert result["action"] == clean_action, "Action should be cleaned"


class TestSyncWrapperMCPDisabled:
    """Test sync wrapper with MCP disabled."""

    def test_works_without_mcp_manager(self, agent_without_mcp, sample_game_state):
        """
        Verify agent with mcp_manager=None works correctly.

        Test approach:
        1. Create agent without MCP manager
        2. Mock _generate_action_async to return valid result
        3. Call get_action_with_reasoning
        4. Verify no errors and valid result returned
        """
        # Arrange
        expected_result = {
            "action": "look",
            "reasoning": "test reasoning",
            "new_objective": None,
            "raw_response": "{}",
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        with patch.object(
            agent_without_mcp, "_generate_action_async", side_effect=mock_generate_async
        ) as mock_async:
            # Act
            result = agent_without_mcp.get_action_with_reasoning(sample_game_state)

            # Assert
            mock_async.assert_called_once_with(sample_game_state, None)
            assert result == expected_result

    def test_works_with_disabled_mcp_manager(
        self, agent_with_mcp_disabled, sample_game_state
    ):
        """
        Verify agent with mcp_manager.is_disabled=True works correctly.

        Test approach:
        1. Create agent with disabled MCP manager
        2. Mock _generate_action_async
        3. Verify async method is called
        4. Verify result is valid
        """
        # Arrange
        expected_result = {
            "action": "north",
            "reasoning": "going north",
            "new_objective": None,
            "raw_response": "{}",
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        with patch.object(
            agent_with_mcp_disabled,
            "_generate_action_async",
            side_effect=mock_generate_async,
        ) as mock_async:
            # Act
            result = agent_with_mcp_disabled.get_action_with_reasoning(sample_game_state)

            # Assert
            mock_async.assert_called_once()
            assert result == expected_result

    def test_uses_same_async_path(
        self, agent_with_mcp, agent_without_mcp, sample_game_state
    ):
        """
        Verify both MCP enabled and disabled use _generate_action_async.

        Test approach:
        1. Mock _generate_action_async for both agents
        2. Call get_action_with_reasoning on both
        3. Verify both called _generate_action_async
        """
        # Arrange
        expected_result = {
            "action": "look",
            "reasoning": "test",
            "new_objective": None,
            "raw_response": "{}",
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        # Act & Assert - MCP enabled
        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ) as mock_async_enabled:
            result_enabled = agent_with_mcp.get_action_with_reasoning(sample_game_state)
            mock_async_enabled.assert_called_once()
            assert result_enabled == expected_result

        # Act & Assert - MCP disabled
        with patch.object(
            agent_without_mcp, "_generate_action_async", side_effect=mock_generate_async
        ) as mock_async_disabled:
            result_disabled = agent_without_mcp.get_action_with_reasoning(
                sample_game_state
            )
            mock_async_disabled.assert_called_once()
            assert result_disabled == expected_result


class TestAsyncBoundary:
    """Test asyncio.run boundary and event loop handling."""

    @pytest.mark.asyncio
    async def test_raises_error_when_called_from_async_context(
        self, agent_with_mcp, sample_game_state
    ):
        """
        Verify get_action_with_reasoning raises RuntimeError when called from async context.

        Test approach:
        1. Call get_action_with_reasoning from within an async function
        2. Verify RuntimeError is raised with appropriate message
        3. Verify error message points to _generate_action_async as alternative
        """
        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            agent_with_mcp.get_action_with_reasoning(sample_game_state)

        error_message = str(exc_info.value)
        assert "cannot be called from within an async context" in error_message, (
            "Error should explain the async context limitation"
        )
        assert "_generate_action_async" in error_message, (
            "Error should point to the async alternative"
        )

    def test_single_asyncio_run(self, agent_with_mcp, sample_game_state):
        """
        Verify only one asyncio.run call occurs.

        Test approach:
        1. Mock asyncio.run to count calls
        2. Call get_action_with_reasoning
        3. Verify asyncio.run was called exactly once
        """
        # Arrange
        run_count = {"count": 0}

        original_asyncio_run = asyncio.run

        def mock_asyncio_run(coro):
            run_count["count"] += 1
            return original_asyncio_run(coro)

        expected_result = {
            "action": "look",
            "reasoning": "test",
            "new_objective": None,
            "raw_response": "{}",
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        with patch("asyncio.run", side_effect=mock_asyncio_run) as mock_run, patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ):
            # Act
            result = agent_with_mcp.get_action_with_reasoning(sample_game_state)

            # Assert
            assert (
                run_count["count"] == 1
            ), "asyncio.run should be called exactly once (Req 10.1)"
            assert mock_run.call_count == 1, "asyncio.run should be called exactly once"
            assert result == expected_result

    def test_no_nested_event_loops(self, agent_with_mcp, sample_game_state):
        """
        Verify calling get_action_with_reasoning doesn't raise RuntimeError.

        Test approach:
        1. Call get_action_with_reasoning from sync context
        2. Verify no RuntimeError about event loop already running
        3. Verify result is valid
        """
        # Arrange
        expected_result = {
            "action": "look",
            "reasoning": "test",
            "new_objective": None,
            "raw_response": "{}",
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ):
            # Act - should not raise RuntimeError
            try:
                result = agent_with_mcp.get_action_with_reasoning(sample_game_state)
                error_raised = None
            except RuntimeError as e:
                error_raised = e

            # Assert
            assert (
                error_raised is None
            ), f"Should not raise RuntimeError about nested loops (Req 10.5): {error_raised}"
            assert result == expected_result


class TestBackwardCompatibility:
    """Test backward compatibility of return format."""

    def test_returns_expected_keys(self, agent_with_mcp, sample_game_state):
        """
        Verify return dict has all expected keys.

        Test approach:
        1. Mock _generate_action_async to return complete dict
        2. Call get_action_with_reasoning
        3. Verify all keys present: action, reasoning, new_objective, raw_response
        """
        # Arrange
        expected_result = {
            "action": "look",
            "reasoning": "test reasoning",
            "new_objective": "explore the house",
            "raw_response": '{"thinking": "test", "action": "look"}',
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ):
            # Act
            result = agent_with_mcp.get_action_with_reasoning(sample_game_state)

            # Assert
            assert "action" in result, "Result must contain 'action' key (Req 15.5)"
            assert "reasoning" in result, "Result must contain 'reasoning' key (Req 15.5)"
            assert (
                "new_objective" in result
            ), "Result must contain 'new_objective' key (Req 15.5)"
            assert (
                "raw_response" in result
            ), "Result must contain 'raw_response' key (Req 15.5)"

    def test_action_is_string(self, agent_with_mcp, sample_game_state):
        """
        Verify action value is always a string.

        Test approach:
        1. Mock _generate_action_async with string action
        2. Call get_action_with_reasoning
        3. Verify action is string type
        """
        # Arrange
        expected_result = {
            "action": "look",
            "reasoning": "test",
            "new_objective": None,
            "raw_response": "{}",
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ):
            # Act
            result = agent_with_mcp.get_action_with_reasoning(sample_game_state)

            # Assert
            assert isinstance(
                result["action"], str
            ), "Action must be string type (Req 15.5)"
            assert len(result["action"]) > 0, "Action must not be empty"

    def test_error_returns_safe_defaults(self, agent_with_mcp, sample_game_state):
        """
        Verify that on exception, get_action_with_reasoning returns safe defaults.

        Test approach:
        1. Mock _generate_action_async to raise exception
        2. Call get_action_with_reasoning
        3. Verify return dict has safe default values
        """
        # Arrange
        async def mock_generate_async(*args, **kwargs):
            raise Exception("Simulated async error")

        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ):
            # Act
            result = agent_with_mcp.get_action_with_reasoning(sample_game_state)

            # Assert
            assert result["action"] == "look", "Error should return 'look' as safe default"
            assert (
                result["reasoning"] is None
            ), "Error should return None for reasoning"
            assert (
                result["new_objective"] is None
            ), "Error should return None for new_objective"
            assert (
                result["raw_response"] is None
            ), "Error should return None for raw_response"

    def test_null_objective_handled(self, agent_with_mcp, sample_game_state):
        """
        Verify new_objective can be None (backward compatibility).

        Test approach:
        1. Mock _generate_action_async with None objective
        2. Call get_action_with_reasoning
        3. Verify new_objective is None
        """
        # Arrange
        expected_result = {
            "action": "look",
            "reasoning": "test",
            "new_objective": None,
            "raw_response": "{}",
        }

        async def mock_generate_async(*args, **kwargs):
            return expected_result

        with patch.object(
            agent_with_mcp, "_generate_action_async", side_effect=mock_generate_async
        ):
            # Act
            result = agent_with_mcp.get_action_with_reasoning(sample_game_state)

            # Assert
            assert result["new_objective"] is None, "new_objective can be None"
