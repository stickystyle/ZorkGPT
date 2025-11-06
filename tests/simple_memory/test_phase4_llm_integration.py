"""
ABOUTME: Tests for Phase 4: LLM Integration with standalone invalidation support.
ABOUTME: Verifies record_action_outcome() processes invalidations and synthesis prompt includes examples.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from managers.simple_memory_manager import (
    SimpleMemoryManager,
    MemorySynthesisResponse,
    MemoryStatus,
    Memory,
    INVALIDATION_MARKER
)
from session.game_state import GameState
from session.game_configuration import GameConfiguration


@pytest.fixture
def temp_memories_file(tmp_path):
    """Create temporary memories file for testing."""
    workdir = tmp_path / "game_files"
    workdir.mkdir(exist_ok=True)
    memories_file = workdir / "Memories.md"

    # Write initial content with existing memories
    memories_file.write_text("""# Location Memories

## Location 152: Troll Room
**Visits:** 2 | **Episodes:** 1

### Memories

**[NOTE - TENTATIVE] Troll might be friendly** *(Ep1, T10)*
The troll hasn't attacked yet. Might be approachable.

**[NOTE] Troll accepts lunch gift** *(Ep1, T12)*
Troll graciously accepts the lunch offering.

---

""")
    return memories_file


@pytest.fixture
def game_state():
    """Create GameState instance."""
    state = GameState()
    state.turn_count = 15
    state.episode_id = "ep_001"
    return state


@pytest.fixture
def config(tmp_path):
    """Create GameConfiguration with temp directory."""
    return GameConfiguration(
        zork_game_workdir=str(tmp_path / "game_files"),
        max_turns_per_episode=100
    )


@pytest.fixture
def memory_manager(config, game_state, temp_memories_file):
    """Create SimpleMemoryManager instance with mocked LLM client."""
    logger = Mock()
    manager = SimpleMemoryManager(logger, config, game_state)

    # Mock LLM client
    manager._llm_client = Mock()
    manager._llm_client_initialized = True

    return manager


class TestPhase4StandaloneInvalidation:
    """Test LLM integration for standalone invalidation (should_remember=False + invalidate)."""

    def test_invalidate_without_creating_new_memory(self, memory_manager):
        """Test that LLM can invalidate memories without creating a replacement."""
        # Count initial memories
        initial_count = len(memory_manager.memory_cache[152])

        # Mock synthesis response directly with MemorySynthesisResponse object
        from managers.simple_memory_manager import MemorySynthesisResponse

        mock_synthesis = MemorySynthesisResponse(
            should_remember=False,
            invalidate_memory_titles={"Troll might be friendly", "Troll accepts lunch gift"},
            invalidation_reason="Both proven false by troll attack resulting in death",
            reasoning="Death proves both assumptions were wrong"
        )

        # Patch _synthesize_memory to return mock response
        with patch.object(memory_manager, '_synthesize_memory', return_value=mock_synthesis):
            # Call record_action_outcome with death trigger
            z_machine_context = {
                'score_delta': 0,
                'location_changed': False,
                'inventory_changed': False,
                'died': True,  # Death trigger
                'first_visit': False,
                'response_length': 50
            }

            memory_manager.record_action_outcome(
                location_id=152,
                location_name="Troll Room",
                action="attack troll",
                response="The troll swings his axe and cleaves you in twain!",
                z_machine_context=z_machine_context
            )

        # Verify no new memory created (count unchanged)
        memories = memory_manager.memory_cache[152]
        assert len(memories) == initial_count, "Should not create new memory"

        # Verify both memories invalidated
        invalidated_count = sum(
            1 for m in memories
            if m.status == MemoryStatus.SUPERSEDED and m.superseded_by == INVALIDATION_MARKER
        )
        assert invalidated_count == 2, f"Should invalidate 2 memories, got {invalidated_count}"

        # Verify invalidation reason stored
        for memory in memories:
            if memory.status == MemoryStatus.SUPERSEDED and memory.superseded_by == INVALIDATION_MARKER:
                assert memory.invalidation_reason == "Both proven false by troll attack resulting in death"

    def test_create_new_memory_and_invalidate_others(self, memory_manager):
        """Test that LLM can create new memory AND invalidate unrelated memories."""
        from managers.simple_memory_manager import MemorySynthesisResponse

        mock_synthesis = MemorySynthesisResponse(
            should_remember=True,
            category="DANGER",
            memory_title="Troll attacks after accepting gift",
            memory_text="Troll accepts gift but then attacks immediately. Gift strategy fails.",
            persistence="permanent",
            status="ACTIVE",
            supersedes_memory_titles={"Troll accepts lunch gift"},
            invalidate_memory_titles={"Troll might be friendly"},
            invalidation_reason="Proven false by attack",
            reasoning="Superseding direct memory, invalidating unrelated assumption"
        )

        with patch.object(memory_manager, '_synthesize_memory', return_value=mock_synthesis):
            z_machine_context = {
                'score_delta': 0,
                'location_changed': False,
                'inventory_changed': False,
                'died': True,
                'first_visit': False,
                'response_length': 50
            }

            memory_manager.record_action_outcome(
                location_id=152,
                location_name="Troll Room",
                action="go north",
                response="The troll attacks you!",
                z_machine_context=z_machine_context
            )

        # Verify new memory created
        memories = memory_manager.memory_cache[152]
        new_memory = next((m for m in memories if m.title == "Troll attacks after accepting gift"), None)
        assert new_memory is not None, "Should create new memory"
        assert new_memory.status == MemoryStatus.ACTIVE

        # Verify supersession (direct replacement)
        lunch_memory = next((m for m in memories if "lunch gift" in m.title), None)
        assert lunch_memory.status == MemoryStatus.SUPERSEDED
        assert lunch_memory.superseded_by == "Troll attacks after accepting gift"

        # Verify invalidation (unrelated wrong assumption)
        friendly_memory = next((m for m in memories if "friendly" in m.title), None)
        assert friendly_memory.status == MemoryStatus.SUPERSEDED
        assert friendly_memory.superseded_by == INVALIDATION_MARKER
        assert friendly_memory.invalidation_reason == "Proven false by attack"

    def test_logging_for_invalidations(self, memory_manager, caplog):
        """Test that invalidations are logged properly."""
        from managers.simple_memory_manager import MemorySynthesisResponse

        mock_synthesis = MemorySynthesisResponse(
            should_remember=False,
            invalidate_memory_titles={"Troll might be friendly"},
            invalidation_reason="Proven false by death",
            reasoning="Death invalidates speculation"
        )

        with patch.object(memory_manager, '_synthesize_memory', return_value=mock_synthesis):
            z_machine_context = {'died': True, 'score_delta': 0, 'location_changed': False, 'inventory_changed': False}

            memory_manager.record_action_outcome(
                location_id=152,
                location_name="Troll Room",
                action="attack troll",
                response="You die!",
                z_machine_context=z_machine_context
            )

        # Verify logging calls (checking mock logger)
        log_calls = [str(call) for call in memory_manager.logger.info.call_args_list]

        # Should log invalidation summary
        invalidation_logs = [c for c in log_calls if "Invalidating" in c]
        assert len(invalidation_logs) >= 1, "Should log invalidation operation"


class TestPhase4PromptExamples:
    """Test that synthesis prompt includes invalidation examples and guidance."""

    def test_synthesis_prompt_includes_invalidation_section(self, memory_manager):
        """Test that _synthesize_memory prompt includes INVALIDATION CHECK section."""
        # Mock LLM to capture prompt
        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs['messages'][0]['content']

            # Return valid response
            response = Mock()
            response.content = '{"should_remember": false, "reasoning": "test"}'
            return response

        memory_manager.llm_client.chat.completions.create.side_effect = capture_prompt

        # Call synthesis
        z_machine_context = {'died': True, 'score_delta': 0, 'location_changed': False, 'inventory_changed': False}

        memory_manager.record_action_outcome(
            location_id=152,
            location_name="Troll Room",
            action="test",
            response="test",
            z_machine_context=z_machine_context
        )

        # Verify prompt includes INVALIDATION CHECK section
        assert "INVALIDATION CHECK (without replacement)" in captured_prompt
        assert "invalidate_memory_titles" in captured_prompt
        assert "invalidation_reason" in captured_prompt

    def test_synthesis_prompt_includes_examples(self, memory_manager):
        """Test that prompt includes clear examples of invalidation vs supersession."""
        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs['messages'][0]['content']
            response = Mock()
            response.content = '{"should_remember": false, "reasoning": "test"}'
            return response

        memory_manager.llm_client.chat.completions.create.side_effect = capture_prompt

        z_machine_context = {'died': True, 'score_delta': 0, 'location_changed': False, 'inventory_changed': False}
        memory_manager.record_action_outcome(152, "Troll Room", "test", "test", z_machine_context)

        # Verify examples present
        assert "Death invalidates TENTATIVE assumptions" in captured_prompt
        assert "Core assumption proven false" in captured_prompt
        assert "Multiple related memories wrong" in captured_prompt

        # Verify guidance on when to use each
        assert "When to use invalidate_memory_titles vs supersedes_memory_titles" in captured_prompt
        assert "INVALIDATE (standalone)" in captured_prompt
        assert "SUPERSEDE (with replacement)" in captured_prompt

    def test_synthesis_prompt_includes_json_examples(self, memory_manager):
        """Test that prompt includes JSON response examples for invalidation."""
        captured_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs['messages'][0]['content']
            response = Mock()
            response.content = '{"should_remember": false, "reasoning": "test"}'
            return response

        memory_manager.llm_client.chat.completions.create.side_effect = capture_prompt

        z_machine_context = {'died': True, 'score_delta': 0, 'location_changed': False, 'inventory_changed': False}
        memory_manager.record_action_outcome(152, "Troll Room", "test", "test", z_machine_context)

        # Verify JSON examples
        assert "Example valid response for invalidating without new memory" in captured_prompt
        assert "Example valid response for creating new memory AND invalidating others" in captured_prompt
        assert '"invalidate_memory_titles"' in captured_prompt


class TestPhase4BackwardCompatibility:
    """Test that existing synthesis workflow still works (backward compatibility)."""

    def test_traditional_supersession_still_works(self, memory_manager):
        """Test that traditional supersession (without invalidation) still works."""
        # Mock LLM response: traditional supersession only
        mock_response = Mock()
        mock_response.content = """{
            "should_remember": true,
            "category": "DANGER",
            "memory_title": "Troll attacks after gift",
            "memory_text": "Troll attacks despite accepting gift.",
            "persistence": "permanent",
            "status": "ACTIVE",
            "supersedes_memory_titles": ["Troll accepts lunch gift"],
            "reasoning": "Contradicts previous memory"
        }"""

        memory_manager.llm_client.chat.completions.create.return_value = mock_response

        z_machine_context = {'died': True, 'score_delta': 0, 'location_changed': False, 'inventory_changed': False}

        memory_manager.record_action_outcome(
            location_id=152,
            location_name="Troll Room",
            action="go north",
            response="Troll attacks!",
            z_machine_context=z_machine_context
        )

        # Verify new memory created
        memories = memory_manager.memory_cache[152]
        new_memory = next((m for m in memories if m.title == "Troll attacks after gift"), None)
        assert new_memory is not None
        assert new_memory.status == MemoryStatus.ACTIVE

        # Verify old memory superseded (traditional way, not invalidated)
        old_memory = next((m for m in memories if "lunch gift" in m.title), None)
        assert old_memory.status == MemoryStatus.SUPERSEDED
        assert old_memory.superseded_by == "Troll attacks after gift"  # Not INVALIDATION_MARKER
        assert old_memory.invalidation_reason is None  # No invalidation reason

    def test_no_supersession_or_invalidation_still_works(self, memory_manager):
        """Test creating memory without any supersession or invalidation."""
        mock_response = Mock()
        mock_response.content = """{
            "should_remember": true,
            "category": "DISCOVERY",
            "memory_title": "Sword found on floor",
            "memory_text": "Elvish sword lying on the floor, can be taken.",
            "persistence": "permanent",
            "status": "ACTIVE",
            "reasoning": "New discovery"
        }"""

        memory_manager.llm_client.chat.completions.create.return_value = mock_response

        z_machine_context = {'first_visit': True, 'score_delta': 0, 'location_changed': False, 'inventory_changed': False}

        memory_manager.record_action_outcome(
            location_id=100,
            location_name="Living Room",
            action="examine floor",
            response="You see a sword.",
            z_machine_context=z_machine_context
        )

        # Verify memory created
        assert 100 in memory_manager.memory_cache
        memories = memory_manager.memory_cache[100]
        assert len(memories) == 1
        assert memories[0].title == "Sword found on floor"
        assert memories[0].status == MemoryStatus.ACTIVE
