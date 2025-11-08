"""
ABOUTME: Comprehensive tests for every-turn LLM-based objective completion checking.
ABOUTME: Tests configuration, early exits, enhanced context, and completion detection.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from managers.objective_manager import ObjectiveManager, ObjectiveCompletionResponse
from managers.simple_memory_manager import SimpleMemoryManager, Memory
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from hybrid_zork_extractor import ExtractorResponse


class TestObjectiveCompletionEveryTurn:
    """Test suite for every-turn LLM-based objective completion checking."""

    @pytest.fixture
    def temp_workdir(self):
        """Create temporary working directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def game_config(self, temp_workdir):
        """Create real GameConfiguration with completion checking enabled."""
        config = GameConfiguration(
            max_turns_per_episode=500,
            zork_game_workdir=str(temp_workdir),
            knowledge_file="knowledgebase.md",
            map_state_file="map_state.json",
            # Objective completion checking config
            enable_objective_completion_llm_check=True,
            completion_check_interval=1,  # Check every turn
            completion_history_window=3,
            completion_include_memories=True,
        )
        return config

    @pytest.fixture
    def game_state(self):
        """Create GameState with test data."""
        state = GameState()
        state.episode_id = "test-episode"
        state.turn_count = 10
        state.current_room_id = 180
        state.current_room_name = "West of House"
        state.current_room_name_for_map = "West of House"
        state.current_inventory = ["brass lantern"]
        state.previous_zork_score = 5
        state.discovered_objectives = [
            "Acquire the brass lantern",
            "Visit the Kitchen",
            "Open the trap door"
        ]
        state.completed_objectives = []
        # Add action history
        from session.game_state import ActionHistoryEntry
        state.action_history = [
            ActionHistoryEntry(
                action="go north",
                response="You are at a forest clearing.",
                location_id=10,
                location_name="Forest Path"
            ),
            ActionHistoryEntry(
                action="examine trees",
                response="The trees are ordinary pine trees.",
                location_id=10,
                location_name="Forest Path"
            ),
            ActionHistoryEntry(
                action="take lantern",
                response="Taken.",
                location_id=180,
                location_name="West of House"
            )
        ]
        return state

    @pytest.fixture
    def mock_adaptive_knowledge_manager(self):
        """Create mock AdaptiveKnowledgeManager with LLM client."""
        mock_akm = Mock()
        mock_akm.client = Mock()
        mock_akm.analysis_model = "test-model"
        mock_akm.analysis_sampling = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 50,
            "min_p": 0.0,
            "max_tokens": 1000
        }
        return mock_akm

    @pytest.fixture
    def simple_memory(self, mock_logger, game_config, game_state):
        """Create SimpleMemoryManager with test memories."""
        from managers.memory import MemoryCacheManager

        memory_mgr = SimpleMemoryManager.__new__(SimpleMemoryManager)
        memory_mgr.logger = mock_logger
        memory_mgr.config = game_config
        memory_mgr.game_state = game_state
        memory_mgr._llm_client = None

        # Initialize cache_manager BEFORE setting caches
        memory_mgr.cache_manager = MemoryCacheManager()

        memory_mgr.memory_cache = {
            180: [
                Memory(
                    category="SUCCESS",
                    title="Lantern location",
                    episode=1,
                    turns="5",
                    score_change=5,
                    text="The brass lantern is here.",
                    status="ACTIVE",
                    persistence="permanent"
                )
            ]
        }
        memory_mgr.ephemeral_cache = {}
        memory_mgr.pending_actions = []
        memory_mgr.synthesis_cooldown = 0
        return memory_mgr

    @pytest.fixture
    def objective_manager(
        self,
        mock_logger,
        game_config,
        game_state,
        mock_adaptive_knowledge_manager,
        simple_memory
    ):
        """Create ObjectiveManager with mocked dependencies."""
        manager = ObjectiveManager(
            logger=mock_logger,
            config=game_config,
            game_state=game_state,
            adaptive_knowledge_manager=mock_adaptive_knowledge_manager,
            map_manager=Mock(),
            simple_memory=simple_memory
        )
        return manager

    # Test 1: Every turn checking when objectives exist
    def test_every_turn_check_when_objectives_exist(
        self, objective_manager, mock_adaptive_knowledge_manager
    ):
        """
        LLM should be called every turn when objectives exist and checking is enabled.

        Test approach:
        1. Setup: objectives exist, config enabled
        2. Execute: call check_objective_completion
        3. Verify: LLM client was called
        """
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '{"completed_objectives": [], "reasoning": "No completion yet"}'
        mock_adaptive_knowledge_manager.client.chat.completions.create.return_value = mock_response

        # Call completion check
        objective_manager.check_objective_completion(
            action_taken="go north",
            game_response="You are at a clearing.",
            extracted_info=None
        )

        # Verify LLM was called
        assert mock_adaptive_knowledge_manager.client.chat.completions.create.called, \
            "LLM should be called when objectives exist"

    # Test 2: Skip check when no objectives exist
    def test_skip_check_when_no_objectives(
        self, objective_manager, mock_adaptive_knowledge_manager
    ):
        """
        LLM should NOT be called when no objectives exist.

        Test approach:
        1. Setup: empty objectives list
        2. Execute: call check_objective_completion
        3. Verify: LLM client was NOT called
        """
        # Clear objectives
        objective_manager.game_state.discovered_objectives = []

        # Call completion check
        objective_manager.check_objective_completion(
            action_taken="go north",
            game_response="You are at a clearing.",
            extracted_info=None
        )

        # Verify LLM was NOT called
        assert not mock_adaptive_knowledge_manager.client.chat.completions.create.called, \
            "LLM should not be called when no objectives exist"

    # Test 3: Skip check when disabled in config
    def test_skip_check_when_disabled_in_config(
        self, objective_manager, mock_adaptive_knowledge_manager, game_config
    ):
        """
        LLM should NOT be called when completion checking is disabled in config.

        Test approach:
        1. Setup: disable completion checking in config
        2. Execute: call check_objective_completion
        3. Verify: LLM client was NOT called
        """
        # Disable completion checking
        game_config.enable_objective_completion_llm_check = False

        # Call completion check
        objective_manager.check_objective_completion(
            action_taken="go north",
            game_response="You are at a clearing.",
            extracted_info=None
        )

        # Verify LLM was NOT called
        assert not mock_adaptive_knowledge_manager.client.chat.completions.create.called, \
            "LLM should not be called when completion checking is disabled"

    # Test 4: Enhanced context includes memories
    def test_enhanced_context_includes_memories(
        self, objective_manager, mock_adaptive_knowledge_manager, simple_memory
    ):
        """
        LLM prompt should include location-specific memories when available.

        Test approach:
        1. Setup: add memories for current location
        2. Execute: call check_objective_completion
        3. Verify: prompt contains memory section
        """
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '{"completed_objectives": [], "reasoning": "No completion yet"}'
        mock_adaptive_knowledge_manager.client.chat.completions.create.return_value = mock_response

        # Call completion check
        objective_manager.check_objective_completion(
            action_taken="take lantern",
            game_response="Taken.",
            extracted_info=None
        )

        # Get the prompt that was sent to LLM
        call_args = mock_adaptive_knowledge_manager.client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]

        # Verify prompt includes memory section
        assert "Location-Specific Memories" in prompt, \
            "Prompt should include location-specific memories section"
        assert "brass lantern" in prompt.lower(), \
            "Prompt should include memory about lantern"

    # Test 5: Enhanced context includes action history
    def test_enhanced_context_includes_action_history(
        self, objective_manager, mock_adaptive_knowledge_manager, game_state
    ):
        """
        LLM prompt should include recent action history.

        Test approach:
        1. Setup: populate action history
        2. Execute: call check_objective_completion
        3. Verify: prompt contains recent actions
        """
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '{"completed_objectives": [], "reasoning": "No completion yet"}'
        mock_adaptive_knowledge_manager.client.chat.completions.create.return_value = mock_response

        # Call completion check
        objective_manager.check_objective_completion(
            action_taken="go north",
            game_response="You are at a clearing.",
            extracted_info=None
        )

        # Get the prompt that was sent to LLM
        call_args = mock_adaptive_knowledge_manager.client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]

        # Verify prompt includes action history section
        assert "Recent Action History" in prompt, \
            "Prompt should include recent action history section"
        assert "Turn" in prompt, \
            "Prompt should include turn numbers"
        assert "examine trees" in prompt, \
            "Prompt should include actions from history"

    # Test 6: Completion removes objective from list
    def test_completion_removes_objective_from_list(
        self, objective_manager, mock_adaptive_knowledge_manager, game_state
    ):
        """
        Completed objectives should be removed from discovered_objectives list.

        Test approach:
        1. Setup: objectives list with target objective
        2. Execute: LLM marks objective as complete
        3. Verify: objective removed from list, added to completed
        """
        # Mock LLM response indicating completion
        mock_response = Mock()
        mock_response.content = '''{
            "completed_objectives": ["Acquire the brass lantern"],
            "reasoning": "Successfully acquired lantern with take command"
        }'''
        mock_adaptive_knowledge_manager.client.chat.completions.create.return_value = mock_response

        initial_objectives_count = len(game_state.discovered_objectives)

        # Call completion check
        objective_manager.check_objective_completion(
            action_taken="take lantern",
            game_response="Taken.",
            extracted_info=None
        )

        # Verify objective removed from discovered list
        assert "Acquire the brass lantern" not in game_state.discovered_objectives, \
            "Completed objective should be removed from discovered list"
        assert len(game_state.discovered_objectives) == initial_objectives_count - 1, \
            "Discovered objectives count should decrease by 1"

        # Verify objective added to completed list
        assert len(game_state.completed_objectives) == 1, \
            "Completed objectives count should increase by 1"
        assert game_state.completed_objectives[0]["objective"] == "Acquire the brass lantern", \
            "Completed objective should be added to completed list"

    # Test 7: Multi-objective completion in single turn
    def test_multi_objective_completion_single_turn(
        self, objective_manager, mock_adaptive_knowledge_manager, game_state
    ):
        """
        Multiple objectives can be completed in a single turn.

        Test approach:
        1. Setup: multiple objectives in list
        2. Execute: LLM marks multiple objectives as complete
        3. Verify: all marked objectives removed and tracked
        """
        # Mock LLM response indicating multiple completions
        mock_response = Mock()
        mock_response.content = '''{
            "completed_objectives": [
                "Acquire the brass lantern",
                "Visit the Kitchen"
            ],
            "reasoning": "Took lantern and entered kitchen in same turn"
        }'''
        mock_adaptive_knowledge_manager.client.chat.completions.create.return_value = mock_response

        initial_objectives_count = len(game_state.discovered_objectives)

        # Call completion check
        objective_manager.check_objective_completion(
            action_taken="enter kitchen",
            game_response="You are in a small kitchen. There is a brass lantern here. Taken.",
            extracted_info=None
        )

        # Verify both objectives removed
        assert "Acquire the brass lantern" not in game_state.discovered_objectives
        assert "Visit the Kitchen" not in game_state.discovered_objectives
        assert len(game_state.discovered_objectives) == initial_objectives_count - 2

        # Verify both objectives added to completed list
        assert len(game_state.completed_objectives) == 2
        completed_obj_texts = [obj["objective"] for obj in game_state.completed_objectives]
        assert "Acquire the brass lantern" in completed_obj_texts
        assert "Visit the Kitchen" in completed_obj_texts

    # Test 8: Check interval respects config
    def test_check_interval_respects_config(
        self, objective_manager, mock_adaptive_knowledge_manager, game_config, game_state
    ):
        """
        Completion checking should respect check_interval configuration.

        Test approach:
        1. Setup: set check_interval to 5 turns
        2. Execute: call check at turns that should be skipped
        3. Verify: LLM only called on correct interval turns
        """
        # Set check interval to 5 turns
        game_config.completion_check_interval = 5

        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '{"completed_objectives": [], "reasoning": "No completion"}'
        mock_adaptive_knowledge_manager.client.chat.completions.create.return_value = mock_response

        # Test turn 11 (not divisible by 5) - should skip
        game_state.turn_count = 11
        objective_manager.check_objective_completion(
            action_taken="go north",
            game_response="You are at a clearing.",
            extracted_info=None
        )
        assert not mock_adaptive_knowledge_manager.client.chat.completions.create.called, \
            "LLM should not be called on turn 11 with interval=5"

        # Test turn 15 (divisible by 5) - should call
        game_state.turn_count = 15
        objective_manager.check_objective_completion(
            action_taken="go south",
            game_response="You are back at the house.",
            extracted_info=None
        )
        assert mock_adaptive_knowledge_manager.client.chat.completions.create.called, \
            "LLM should be called on turn 15 with interval=5"
