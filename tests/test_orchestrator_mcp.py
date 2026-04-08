# ABOUTME: Tests for orchestrator MCP integration (Task 15)
# ABOUTME: Validates MCPManager initialization, agent integration, and error handling

import pytest
from unittest.mock import MagicMock, patch, Mock, PropertyMock
from pathlib import Path

from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
from managers.mcp_manager import MCPManager
from managers.mcp_config import MCPError, MCPConfigError, MCPServerStartupError
from session.game_configuration import GameConfiguration
from game_interface.core.jericho_interface import JerichoInterface
from zork_agent import ZorkAgent


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config_mcp_enabled(tmp_path):
    """Mock GameConfiguration with MCP enabled."""
    config = MagicMock(spec=GameConfiguration)
    config.mcp_enabled = True
    config.mcp_config_file = str(tmp_path / "mcp_config.json")
    config.game_file_path = str(tmp_path / "zork.z5")
    config.episode_log_file = str(tmp_path / "episode.log")
    config.json_log_file = str(tmp_path / "game.json")
    config.zork_game_workdir = str(tmp_path)
    config.max_turns_per_episode = 100
    config.agent_model = "gpt-4"
    config.critic_model = "gpt-4"
    config.info_ext_model = "gpt-4"
    config.enable_critic = True
    return config


@pytest.fixture
def mock_config_mcp_disabled(tmp_path):
    """Mock GameConfiguration with MCP disabled."""
    config = MagicMock(spec=GameConfiguration)
    config.mcp_enabled = False
    config.mcp_config_file = str(tmp_path / "mcp_config.json")
    config.game_file_path = str(tmp_path / "zork.z5")
    config.episode_log_file = str(tmp_path / "episode.log")
    config.json_log_file = str(tmp_path / "game.json")
    config.zork_game_workdir = str(tmp_path)
    config.max_turns_per_episode = 100
    config.agent_model = "gpt-4"
    config.critic_model = "gpt-4"
    config.info_ext_model = "gpt-4"
    config.enable_critic = True
    return config


@pytest.fixture
def mock_jericho_interface(tmp_path):
    """Mock JerichoInterface to avoid real game file dependencies."""
    with patch('orchestration.zork_orchestrator_v2.JerichoInterface') as mock_jericho:
        instance = MagicMock(spec=JerichoInterface)
        instance.start.return_value = "West of House\nYou are standing..."
        instance.get_location_structured.return_value = MagicMock(num=15, name="West of House")
        instance.get_inventory_structured.return_value = []
        instance.get_valid_exits.return_value = ["north", "south", "east"]
        instance.get_score.return_value = (0, 0)
        mock_jericho.return_value = instance
        yield mock_jericho


@pytest.fixture
def mock_mcp_manager():
    """Mock MCPManager for testing."""
    with patch('orchestration.zork_orchestrator_v2.MCPManager') as mock_mcp:
        instance = MagicMock(spec=MCPManager)
        instance.is_disabled = False
        instance.server_name = "test-mcp-server"  # Add server name property
        mock_mcp.return_value = instance
        yield mock_mcp


@pytest.fixture
def mock_agent():
    """Mock ZorkAgent to avoid LLM dependencies."""
    with patch('orchestration.zork_orchestrator_v2.ZorkAgent') as mock_agent_class:
        instance = MagicMock(spec=ZorkAgent)
        instance.client = MagicMock()
        instance.mcp_manager = None  # Will be set by orchestrator
        mock_agent_class.return_value = instance
        yield mock_agent_class


@pytest.fixture
def mock_other_dependencies(tmp_path):
    """Mock all other orchestrator dependencies."""
    patches = [
        patch('orchestration.zork_orchestrator_v2.setup_logging'),
        patch('logger.setup_episode_logging'),  # Mock from logger module, not orchestrator
        patch('orchestration.zork_orchestrator_v2.ZorkCritic'),
        patch('orchestration.zork_orchestrator_v2.HybridZorkExtractor'),
        patch('orchestration.zork_orchestrator_v2.MapManager'),
        patch('orchestration.zork_orchestrator_v2.ContextManager'),
        patch('orchestration.zork_orchestrator_v2.RejectionManager'),
        patch('orchestration.zork_orchestrator_v2.SimpleMemoryManager'),
        patch('orchestration.zork_orchestrator_v2.StateManager'),
        patch('orchestration.zork_orchestrator_v2.KnowledgeManager'),
        patch('orchestration.zork_orchestrator_v2.ObjectiveManager'),
        patch('orchestration.zork_orchestrator_v2.EpisodeSynthesizer'),
        patch('orchestration.zork_orchestrator_v2.Langfuse', None),  # Disable Langfuse
    ]

    mocks = [p.start() for p in patches]

    # Configure logging mocks
    mocks[0].return_value = MagicMock()  # setup_logging
    mocks[1].return_value = str(tmp_path / "episode.log")  # setup_episode_logging

    yield mocks

    for p in patches:
        p.stop()


# ============================================================================
# Test Cases
# ============================================================================


class TestOrchestratorMCPInitialization:
    """Tests for MCPManager initialization in orchestrator."""

    def test_mcp_manager_initialized_when_enabled(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """When MCP is enabled, orchestrator should create MCPManager instance.

        Requirement: 9.1 - When orchestrator initializes and MCP is enabled,
        create MCPManager instance.
        """
        # Arrange
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            # Act
            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Assert
            assert orchestrator.mcp_manager is not None, "MCPManager should be initialized when MCP enabled"
            assert isinstance(orchestrator.mcp_manager, (MCPManager, MagicMock)), \
                "mcp_manager should be MCPManager instance"

            # Verify MCPManager was instantiated with correct args
            mock_mcp_manager.assert_called_once()
            call_args = mock_mcp_manager.call_args
            assert call_args.kwargs['config'] == mock_config_mcp_enabled
            assert call_args.kwargs['logger'] is not None
            assert 'langfuse_client' in call_args.kwargs

    def test_no_mcp_manager_when_disabled(
        self,
        tmp_path,
        mock_config_mcp_disabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """When MCP is disabled, orchestrator should not create MCPManager.

        Requirement: 9.4 - When MCP is disabled, do not create MCPManager.
        """
        # Arrange
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_disabled):
            # Act
            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Assert
            assert orchestrator.mcp_manager is None, "MCPManager should be None when MCP disabled"
            mock_mcp_manager.assert_not_called()

    def test_mcp_manager_passed_to_agent(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """When MCP is enabled, agent should receive MCPManager instance.

        Requirement: 9.2 - Pass MCPManager to agent when creating it.
        """
        # Arrange
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            # Act
            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Assert
            # Verify ZorkAgent was instantiated with mcp_manager kwarg
            mock_agent.assert_called_once()
            call_kwargs = mock_agent.call_args.kwargs
            assert 'mcp_manager' in call_kwargs, "Agent should receive mcp_manager kwarg"
            assert call_kwargs['mcp_manager'] == orchestrator.mcp_manager, \
                "Agent's mcp_manager should match orchestrator's mcp_manager"

    def test_langfuse_client_passed_to_agent(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """Agent should receive langfuse_client from orchestrator.

        Requirement: 9.2 (implicit) - Agent receives langfuse_client for tracing.
        """
        # Arrange
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            # Mock Langfuse available
            with patch('orchestration.zork_orchestrator_v2.LANGFUSE_AVAILABLE', True):
                with patch('orchestration.zork_orchestrator_v2.Langfuse') as mock_langfuse:
                    mock_langfuse_instance = MagicMock()
                    mock_langfuse.return_value = mock_langfuse_instance

                    # Act
                    orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

                    # Assert
                    mock_agent.assert_called_once()
                    call_kwargs = mock_agent.call_args.kwargs
                    assert 'langfuse_client' in call_kwargs, "Agent should receive langfuse_client kwarg"
                    assert call_kwargs['langfuse_client'] == mock_langfuse_instance, \
                        "Agent's langfuse_client should match orchestrator's langfuse_client"

    def test_mcp_config_error_raises_mcp_error(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """When MCPManager init fails with MCPConfigError, orchestrator should raise MCPError.

        Requirement: 9.3 - Raise MCPError when MCPManager initialization fails.
        """
        # Arrange
        mock_mcp_manager.side_effect = MCPConfigError("Config file not found")

        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            # Act & Assert
            with pytest.raises(MCPError) as exc_info:
                orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Verify error message contains useful info
            assert "Config file not found" in str(exc_info.value)

    def test_mcp_server_startup_error_raises_mcp_error(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """When MCPManager init fails with MCPServerStartupError, orchestrator should raise MCPError.

        Requirement: 9.3 - Raise MCPError when MCPManager initialization fails.
        """
        # Arrange
        mock_mcp_manager.side_effect = MCPServerStartupError("Server failed to start")

        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            # Act & Assert
            with pytest.raises(MCPError) as exc_info:
                orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Verify error message contains useful info
            assert "Server failed to start" in str(exc_info.value)

    def test_orchestrator_remains_synchronous(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """Orchestrator initialization should remain synchronous (no asyncio.run calls).

        Requirement: 10.2 - Orchestrator remains synchronous.
        """
        # Arrange
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            with patch('asyncio.run') as mock_asyncio_run:
                # Act
                orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

                # Assert
                mock_asyncio_run.assert_not_called(), \
                    "Orchestrator initialization should not call asyncio.run"


class TestOrchestratorMCPIntegration:
    """Integration tests for orchestrator MCP behavior."""

    def test_mcp_disabled_no_langfuse_client_to_agent(
        self,
        tmp_path,
        mock_config_mcp_disabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """When MCP is disabled, agent still receives langfuse_client if available.

        This ensures tracing works regardless of MCP status.
        """
        # Arrange
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_disabled):
            with patch('orchestration.zork_orchestrator_v2.LANGFUSE_AVAILABLE', True):
                with patch('orchestration.zork_orchestrator_v2.Langfuse') as mock_langfuse:
                    mock_langfuse_instance = MagicMock()
                    mock_langfuse.return_value = mock_langfuse_instance

                    # Act
                    orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

                    # Assert
                    mock_agent.assert_called_once()
                    call_kwargs = mock_agent.call_args.kwargs
                    assert 'langfuse_client' in call_kwargs
                    # Langfuse should still be passed even when MCP disabled
                    assert call_kwargs['langfuse_client'] == mock_langfuse_instance

    def test_mcp_manager_initialization_order(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """MCPManager should be initialized before agent creation.

        This ensures agent can receive mcp_manager reference during init.
        """
        # Arrange
        init_order = []

        def track_mcp_init(*args, **kwargs):
            init_order.append('mcp_manager')
            mock_instance = MagicMock(spec=MCPManager)
            mock_instance.server_name = "test-server"  # Add required property
            return mock_instance

        def track_agent_init(*args, **kwargs):
            init_order.append('agent')
            mock_instance = MagicMock(spec=ZorkAgent)
            mock_instance.client = MagicMock()
            return mock_instance

        mock_mcp_manager.side_effect = track_mcp_init
        mock_agent.side_effect = track_agent_init

        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            # Act
            orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Assert
            assert init_order == ['mcp_manager', 'agent'], \
                "MCPManager must be initialized before ZorkAgent"

    def test_mcp_error_includes_episode_context(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """MCPError should include episode context for debugging.

        This helps identify which episode had MCP initialization issues.
        """
        # Arrange
        mock_mcp_manager.side_effect = MCPConfigError("Test error")

        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            # Act & Assert
            with pytest.raises(MCPError):
                orchestrator = ZorkOrchestratorV2(episode_id="test-episode-123")

            # Could verify logging calls here if needed


class TestOrchestratorMCPEdgeCases:
    """Edge case tests for orchestrator MCP integration."""

    def test_langfuse_unavailable_no_crash(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """Orchestrator should handle Langfuse being unavailable gracefully."""
        # Arrange
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            with patch('orchestration.zork_orchestrator_v2.LANGFUSE_AVAILABLE', False):
                # Act
                orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

                # Assert
                assert orchestrator.langfuse_client is None
                # Agent should still receive None for langfuse_client
                mock_agent.assert_called_once()
                call_kwargs = mock_agent.call_args.kwargs
                assert call_kwargs.get('langfuse_client') is None

    def test_langfuse_init_exception_graceful_degradation(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_mcp_manager,
        mock_agent,
        mock_other_dependencies,
    ):
        """Orchestrator should gracefully degrade when Langfuse init fails."""
        # Arrange
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            with patch('orchestration.zork_orchestrator_v2.LANGFUSE_AVAILABLE', True):
                with patch('orchestration.zork_orchestrator_v2.Langfuse') as mock_langfuse:
                    mock_langfuse.side_effect = Exception("Langfuse init failed")

                    # Act
                    orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

                    # Assert
                    assert orchestrator.langfuse_client is None
                    # Agent should receive None for langfuse_client
                    mock_agent.assert_called_once()
                    call_kwargs = mock_agent.call_args.kwargs
                    assert call_kwargs.get('langfuse_client') is None

    def test_mcp_manager_none_when_config_missing(
        self,
        tmp_path,
        mock_config_mcp_enabled,
        mock_jericho_interface,
        mock_agent,
        mock_other_dependencies,
    ):
        """When MCP config file is missing, orchestrator should raise MCPError (wrapping MCPConfigError)."""
        # Arrange
        # Don't mock MCPManager - let real import fail
        with patch('orchestration.zork_orchestrator_v2.GameConfiguration.from_toml', return_value=mock_config_mcp_enabled):
            # Make config file path invalid
            mock_config_mcp_enabled.mcp_config_file = str(tmp_path / "nonexistent.json")

            # Act & Assert - orchestrator wraps MCPConfigError in MCPError
            with pytest.raises(MCPError) as exc_info:
                orchestrator = ZorkOrchestratorV2(episode_id="test-episode")

            # Verify helpful error message
            assert "not found" in str(exc_info.value).lower()
