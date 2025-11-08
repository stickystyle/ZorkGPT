# ABOUTME: Unit tests for agent self-directed objective declaration
# ABOUTME: Tests the add_agent_objective() method in ObjectiveManager

import pytest
from unittest.mock import Mock

from managers.objective_manager import ObjectiveManager
from session.game_state import GameState


class TestAgentObjectives:
    """Unit tests for agent-declared objectives in ObjectiveManager."""

    @pytest.fixture
    def game_state(self):
        """Create a GameState instance for testing."""
        return GameState()

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock configuration."""
        config = Mock()
        config.zork_game_workdir = str(tmp_path)
        config.knowledge_file = "knowledgebase.md"
        return config

    @pytest.fixture
    def mock_adaptive_knowledge_manager(self):
        """Create mock adaptive knowledge manager."""
        return Mock()

    @pytest.fixture
    def objective_manager(self, game_state, mock_config, mock_adaptive_knowledge_manager):
        """Create an ObjectiveManager instance for testing."""
        mock_logger = Mock()
        manager = ObjectiveManager(
            logger=mock_logger,
            config=mock_config,
            game_state=game_state,
            adaptive_knowledge_manager=mock_adaptive_knowledge_manager
        )
        return manager

    def test_add_agent_objective(self, objective_manager):
        """Agent can declare objectives via add_agent_objective()."""
        # Add a valid objective
        objective_manager.add_agent_objective("collect all treasures")

        # Verify it was added
        assert "collect all treasures" in objective_manager.game_state.discovered_objectives
        assert len(objective_manager.game_state.discovered_objectives) == 1

    def test_add_agent_objective_deduplication(self, objective_manager):
        """Duplicate agent objectives are not added (case-insensitive)."""
        # Add same objective twice with different casing
        objective_manager.add_agent_objective("find key")
        objective_manager.add_agent_objective("Find Key")
        objective_manager.add_agent_objective("FIND KEY")

        # Verify only one exists
        assert len(objective_manager.game_state.discovered_objectives) == 1
        assert "find key" in objective_manager.game_state.discovered_objectives

    def test_add_agent_objective_empty_string(self, objective_manager):
        """Empty objectives are ignored."""
        # Try to add empty and whitespace-only strings
        objective_manager.add_agent_objective("")
        objective_manager.add_agent_objective("   ")
        objective_manager.add_agent_objective("\n\t")

        # Verify nothing was added
        assert len(objective_manager.game_state.discovered_objectives) == 0
