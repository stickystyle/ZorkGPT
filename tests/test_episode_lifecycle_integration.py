"""
ABOUTME: Comprehensive episode lifecycle integration test for Ephemeral Memory System.
ABOUTME: Validates dual caches, routing, supersession, episode reset, and cross-episode persistence.

This is Phase 7.1 - Integration test covering the complete workflow:
- Dual cache routing (ephemeral vs persistent)
- Supersession with cache migration
- Episode reset clearing ephemeral cache
- Cross-episode persistence of CORE/PERMANENT
- File format with persistence markers
- Cache isolation and combining

Test approach:
- Episode 1: Add memories of all persistence levels, test supersession
- Episode Reset: Clear ephemeral cache, verify persistent remains
- Episode 2: Add new ephemeral memories, verify cross-episode state
- Final Validation: Check file format, breakdowns, isolation
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
import logging

from managers.simple_memory_manager import SimpleMemoryManager, Memory, MemoryStatus
from session.game_state import GameState
from session.game_configuration import GameConfiguration


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    return logger


@pytest.fixture
def game_config(tmp_path):
    """Create a test game configuration with temporary work directory."""
    return GameConfiguration(
        max_turns_per_episode=1000,
        turn_delay_seconds=0.0,
        game_file_path="test_game.z5",
        critic_rejection_threshold=0.5,
        episode_log_file="test_episode.log",
        json_log_file="test_episode.jsonl",
        state_export_file="test_state.json",
        map_state_file="test_map_state.json",
        zork_game_workdir=str(tmp_path),
        client_base_url="http://localhost:1234",
        client_api_key="test_api_key",
        agent_model="test-agent-model",
        critic_model="test-critic-model",
        info_ext_model="test-extractor-model",
        analysis_model="test-analysis-model",
        memory_model="test-memory-model",
        knowledge_update_interval=100,
        objective_update_interval=20,
        enable_objective_refinement=True,
        objective_refinement_interval=200,
        max_objectives_before_forced_refinement=15,
        refined_objectives_target_count=10,
        enable_state_export=True,
        s3_bucket="test-bucket",
        s3_key_prefix="test/",
        simple_memory_file="Memories.md",
        simple_memory_max_shown=10,
        knowledge_file="test_knowledgebase.md",
        agent_sampling={},
        critic_sampling={},
        extractor_sampling={},
        analysis_sampling={},
        memory_sampling={'temperature': 0.3, 'max_tokens': 1000},
    )


@pytest.fixture
def game_state():
    """Create a test game state."""
    state = GameState()
    state.episode_id = "test_episode_001"
    state.turn_count = 10
    state.current_room_name_for_map = "Living Room"
    state.previous_zork_score = 50
    state.current_inventory = ["lamp", "sword"]
    return state


def test_full_episode_lifecycle_with_ephemeral_system(
    mock_logger, game_config, game_state, tmp_path
):
    """
    Comprehensive integration test for ephemeral memory system.

    Tests full episode lifecycle including:
    - Dual cache routing (ephemeral vs persistent)
    - Supersession with cache migration
    - Episode reset clearing ephemeral cache
    - Cross-episode persistence of CORE/PERMANENT
    - File format with persistence markers
    - Cache isolation and combining
    """
    # Setup
    game_config.zork_game_workdir = str(tmp_path)
    game_state.episode_id = "ep_001"
    game_state.episode_number = 1
    game_state.turn_count = 10

    manager = SimpleMemoryManager(
        logger=mock_logger,
        config=game_config,
        game_state=game_state
    )

    memories_path = tmp_path / "Memories.md"

    # ========================================================================
    # EPISODE 1 - Initial exploration
    # ========================================================================

    # 1. Add CORE memory (first visit room description) to location 10
    core_memory = Memory(
        category="NOTE",
        title="West of House room layout",
        episode=1,
        turns="5",
        score_change=None,
        text="Standard starting location with mailbox and window visible.",
        persistence="core",
        status=MemoryStatus.ACTIVE
    )
    success = manager.add_memory(10, "West of House", core_memory)
    assert success, "CORE memory addition should succeed"

    # 2. Add PERMANENT memory (game mechanic) to location 10
    permanent_memory = Memory(
        category="SUCCESS",
        title="Mailbox contains leaflet",
        episode=1,
        turns="6",
        score_change=0,
        text="Opening mailbox reveals leaflet with basic instructions.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    success = manager.add_memory(10, "West of House", permanent_memory)
    assert success, "PERMANENT memory addition should succeed"

    # 3. Add EPHEMERAL memory (agent action: "dropped sword") to location 10
    ephemeral_memory = Memory(
        category="NOTE",
        title="Dropped sword at West of House",
        episode=1,
        turns="7",
        score_change=None,
        text="Temporarily dropped sword here to free inventory space.",
        persistence="ephemeral",
        status=MemoryStatus.ACTIVE
    )
    success = manager.add_memory(10, "West of House", ephemeral_memory)
    assert success, "EPHEMERAL memory addition should succeed"

    # 4. Verify all three appear in get_location_memory(10)
    location_10_memories = manager.get_location_memory(10)
    assert "West of House room layout" in location_10_memories, "CORE memory should be retrievable"
    assert "Mailbox contains leaflet" in location_10_memories, "PERMANENT memory should be retrievable"
    assert "Dropped sword at West of House" in location_10_memories, "EPHEMERAL memory should be retrievable"

    # 5. Verify CORE and PERMANENT in memory_cache
    assert 10 in manager.memory_cache, "Location 10 should be in memory_cache"
    cached_memories = manager.memory_cache[10]
    assert len(cached_memories) == 2, "memory_cache should have 2 persistent memories"
    cached_titles = {m.title for m in cached_memories}
    assert "West of House room layout" in cached_titles, "CORE memory should be in memory_cache"
    assert "Mailbox contains leaflet" in cached_titles, "PERMANENT memory should be in memory_cache"

    # 6. Verify EPHEMERAL in ephemeral_cache only
    assert 10 in manager.ephemeral_cache, "Location 10 should be in ephemeral_cache"
    ephemeral_cached = manager.ephemeral_cache[10]
    assert len(ephemeral_cached) == 1, "ephemeral_cache should have 1 ephemeral memory"
    assert ephemeral_cached[0].title == "Dropped sword at West of House", "EPHEMERAL memory should be in ephemeral_cache"

    # 7. Verify file contains CORE and PERMANENT with correct markers, no EPHEMERAL
    assert memories_path.exists(), "Memories.md should be created"
    file_content = memories_path.read_text()
    assert "**[NOTE - CORE] West of House room layout**" in file_content, "CORE memory should have CORE marker"
    assert "**[SUCCESS - PERMANENT] Mailbox contains leaflet**" in file_content, "PERMANENT memory should have PERMANENT marker"
    assert "Dropped sword at West of House" not in file_content, "EPHEMERAL memory should NOT be in file"

    # ========================================================================
    # Mid-episode - Supersession (migration case)
    # ========================================================================

    game_state.turn_count = 15

    # 8. Add EPHEMERAL memory "sword on ground" to location 15
    ephemeral_sword = Memory(
        category="NOTE",
        title="Sword on ground at location 15",
        episode=1,
        turns="12",
        score_change=None,
        text="Sword temporarily placed here, status unclear.",
        persistence="ephemeral",
        status=MemoryStatus.TENTATIVE
    )
    success = manager.add_memory(15, "Behind House", ephemeral_sword)
    assert success, "Ephemeral sword memory should be added"

    # 9. Supersede with PERMANENT "sword is quest item" (migration case)
    permanent_sword = Memory(
        category="DISCOVERY",
        title="Sword is quest item",
        episode=1,
        turns="15",
        score_change=5,
        text="Sword is important quest item needed for dungeon puzzles.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    success = manager.supersede_memory(
        location_id=15,
        location_name="Behind House",
        old_memory_title="Sword on ground at location 15",
        new_memory=permanent_sword
    )
    assert success, "Supersession should succeed"

    # 10. Verify old EPHEMERAL marked SUPERSEDED in ephemeral_cache
    ephemeral_at_15 = manager.ephemeral_cache.get(15, [])
    assert len(ephemeral_at_15) == 1, "Should have 1 ephemeral memory at location 15"
    old_ephemeral = ephemeral_at_15[0]
    assert old_ephemeral.status == MemoryStatus.SUPERSEDED, "Old ephemeral should be marked SUPERSEDED"
    assert old_ephemeral.superseded_by == "Sword is quest item", "Should reference new memory title"
    assert old_ephemeral.superseded_at_turn == 15, "Should record turn number"

    # 11. Verify new PERMANENT in memory_cache AND file
    persistent_at_15 = manager.memory_cache.get(15, [])
    assert len(persistent_at_15) == 1, "Should have 1 persistent memory at location 15"
    new_permanent = persistent_at_15[0]
    assert new_permanent.title == "Sword is quest item", "New memory should be in persistent cache"
    assert new_permanent.persistence == "permanent", "New memory should be permanent"

    file_content = memories_path.read_text()
    assert "**[DISCOVERY - PERMANENT] Sword is quest item**" in file_content, "New PERMANENT memory should be in file"
    assert "Sword on ground at location 15" not in file_content, "Old ephemeral should NOT be in file"

    # 12. Verify get_location_memory(15) shows both with correct statuses
    location_15_memories = manager.get_location_memory(15)
    assert "Sword is quest item" in location_15_memories, "New ACTIVE memory should appear"
    # SUPERSEDED memories are NOT shown to agent (proven wrong)
    assert "Sword on ground at location 15" not in location_15_memories, "SUPERSEDED memory should NOT appear"

    # ========================================================================
    # End of Episode 1 - State capture
    # ========================================================================

    # 13. Get final cache counts (ephemeral and persistent)
    ephemeral_count_ep1 = manager.get_ephemeral_count()
    persistent_count_ep1 = manager.get_persistent_count()

    assert ephemeral_count_ep1 == 2, "Should have 2 ephemeral memories (1 active + 1 superseded)"
    assert persistent_count_ep1 == 3, "Should have 3 persistent memories (2 at loc 10 + 1 at loc 15)"

    # 14. Read Memories.md and verify format/markers
    file_content = memories_path.read_text()
    assert "## Location 10: West of House" in file_content, "Location 10 section should exist"
    assert "## Location 15: Behind House" in file_content, "Location 15 section should exist"
    assert file_content.count("- CORE]") == 1, "Should have 1 CORE marker"
    assert file_content.count("- PERMANENT]") == 2, "Should have 2 PERMANENT markers"

    # 15. Verify get_location_memory() shows all active memories
    all_memories_loc_10 = manager.get_location_memory(10)
    assert "West of House room layout" in all_memories_loc_10
    assert "Mailbox contains leaflet" in all_memories_loc_10
    assert "Dropped sword at West of House" in all_memories_loc_10

    # ========================================================================
    # Episode Reset
    # ========================================================================

    game_state.episode_number = 2
    game_state.episode_id = "ep_002"
    game_state.turn_count = 1

    # 16. Call manager.reset_episode()
    manager.reset_episode()

    # 17. Verify ephemeral_cache is empty
    assert len(manager.ephemeral_cache) == 0, "ephemeral_cache should be completely empty after reset"
    assert manager.get_ephemeral_count() == 0, "Total ephemeral count should be 0"

    # 18. Verify memory_cache unchanged (CORE and PERMANENT still there)
    assert len(manager.memory_cache) == 2, "memory_cache should still have 2 locations"
    assert 10 in manager.memory_cache, "Location 10 should still be in memory_cache"
    assert 15 in manager.memory_cache, "Location 15 should still be in memory_cache"
    assert len(manager.memory_cache[10]) == 2, "Location 10 should have 2 persistent memories"
    assert len(manager.memory_cache[15]) == 1, "Location 15 should have 1 persistent memory"

    # 19. Verify get_location_memory(10) no longer shows EPHEMERAL
    location_10_after_reset = manager.get_location_memory(10)
    assert "West of House room layout" in location_10_after_reset, "CORE memory should persist"
    assert "Mailbox contains leaflet" in location_10_after_reset, "PERMANENT memory should persist"
    assert "Dropped sword at West of House" not in location_10_after_reset, "EPHEMERAL memory should be gone"

    # 20. Verify get_location_memory(15) only shows PERMANENT (old EPHEMERAL gone)
    location_15_after_reset = manager.get_location_memory(15)
    assert "Sword is quest item" in location_15_after_reset, "PERMANENT memory should persist"
    assert "Sword on ground at location 15" not in location_15_after_reset, "Old ephemeral should be gone"

    # ========================================================================
    # EPISODE 2 - Fresh start with persistent knowledge
    # ========================================================================

    game_state.turn_count = 5

    # 21. Add new EPHEMERAL memory "picked up key" to location 10
    new_ephemeral = Memory(
        category="NOTE",
        title="Picked up brass key",
        episode=2,
        turns="5",
        score_change=None,
        text="Brass key acquired from mailbox in Episode 2.",
        persistence="ephemeral",
        status=MemoryStatus.ACTIVE
    )
    success = manager.add_memory(10, "West of House", new_ephemeral)
    assert success, "New ephemeral memory should be added in Episode 2"

    # 22. Verify get_location_memory(10) shows: CORE, PERMANENT (from Ep1), new EPHEMERAL
    location_10_ep2 = manager.get_location_memory(10)
    assert "West of House room layout" in location_10_ep2, "CORE from Episode 1 should be visible"
    assert "Mailbox contains leaflet" in location_10_ep2, "PERMANENT from Episode 1 should be visible"
    assert "Picked up brass key" in location_10_ep2, "New EPHEMERAL from Episode 2 should be visible"
    assert "Dropped sword at West of House" not in location_10_ep2, "Old Episode 1 ephemeral should NOT be visible"

    # 23. Verify CORE and PERMANENT have episode=1 (from Episode 1)
    core_mem = [m for m in manager.memory_cache[10] if m.title == "West of House room layout"][0]
    permanent_mem = [m for m in manager.memory_cache[10] if m.title == "Mailbox contains leaflet"][0]
    assert core_mem.episode == 1, "CORE memory should have episode=1"
    assert permanent_mem.episode == 1, "PERMANENT memory should have episode=1"

    # 24. Verify new EPHEMERAL has episode=2 (current episode)
    new_ephemeral_cached = manager.ephemeral_cache[10][0]
    assert new_ephemeral_cached.episode == 2, "New EPHEMERAL should have episode=2"

    # 25. Add PERMANENT memory to location 20
    game_state.turn_count = 10
    location_20_memory = Memory(
        category="DISCOVERY",
        title="Troll room discovered",
        episode=2,
        turns="10",
        score_change=0,
        text="Troll guards bridge, requires strategy to pass.",
        persistence="permanent",
        status=MemoryStatus.ACTIVE
    )
    success = manager.add_memory(20, "Troll Room", location_20_memory)
    assert success, "Location 20 memory should be added"

    # 26. Verify cross-location isolation works
    location_10_final = manager.get_location_memory(10)
    location_20_final = manager.get_location_memory(20)

    # Location 10 should NOT have location 20 memories
    assert "Troll room discovered" not in location_10_final, "Location 10 should not show location 20 memories"

    # Location 20 should only have its own memory
    assert "Troll room discovered" in location_20_final, "Location 20 should show its own memory"
    assert "West of House room layout" not in location_20_final, "Location 20 should not show location 10 memories"

    # ========================================================================
    # File Consistency
    # ========================================================================

    # 27. Read final Memories.md
    final_file_content = memories_path.read_text()

    # 28. Verify all PERMANENT and CORE memories from both episodes
    assert "**[NOTE - CORE] West of House room layout**" in final_file_content, "CORE from Ep1 should be in file"
    assert "**[SUCCESS - PERMANENT] Mailbox contains leaflet**" in final_file_content, "PERMANENT from Ep1 should be in file"
    assert "**[DISCOVERY - PERMANENT] Sword is quest item**" in final_file_content, "PERMANENT from Ep1 (supersession) should be in file"
    assert "**[DISCOVERY - PERMANENT] Troll room discovered**" in final_file_content, "PERMANENT from Ep2 should be in file"

    # 29. Verify correct persistence markers ("- CORE", "- PERMANENT")
    assert final_file_content.count("- CORE]") == 1, "Should have exactly 1 CORE marker"
    assert final_file_content.count("- PERMANENT]") == 3, "Should have exactly 3 PERMANENT markers"

    # 30. Verify NO ephemeral memories in file (from either episode)
    assert "Dropped sword at West of House" not in final_file_content, "Ep1 ephemeral should NOT be in file"
    assert "Sword on ground at location 15" not in final_file_content, "Ep1 superseded ephemeral should NOT be in file"
    assert "Picked up brass key" not in final_file_content, "Ep2 ephemeral should NOT be in file"

    # 31. Verify supersession metadata preserved
    # Note: The old ephemeral "Sword on ground" is NOT in file because it was ephemeral
    # The new permanent "Sword is quest item" IS in file
    assert "[Superseded at T15 by" not in final_file_content or "Sword on ground" not in final_file_content, \
        "Superseded ephemeral should not appear in file"

    # ========================================================================
    # Breakdown Validation
    # ========================================================================

    # 32. Check memory breakdown for each location
    breakdown_10 = manager.get_memory_breakdown(10)
    assert breakdown_10["core"] == 1, "Location 10 should have 1 CORE memory"
    assert breakdown_10["permanent"] == 1, "Location 10 should have 1 PERMANENT memory"
    assert breakdown_10["ephemeral"] == 1, "Location 10 should have 1 EPHEMERAL memory (Episode 2)"

    breakdown_15 = manager.get_memory_breakdown(15)
    assert breakdown_15["core"] == 0, "Location 15 should have 0 CORE memories"
    assert breakdown_15["permanent"] == 1, "Location 15 should have 1 PERMANENT memory"
    assert breakdown_15["ephemeral"] == 0, "Location 15 should have 0 EPHEMERAL memories (superseded one cleared on reset)"

    breakdown_20 = manager.get_memory_breakdown(20)
    assert breakdown_20["core"] == 0, "Location 20 should have 0 CORE memories"
    assert breakdown_20["permanent"] == 1, "Location 20 should have 1 PERMANENT memory"
    assert breakdown_20["ephemeral"] == 0, "Location 20 should have 0 EPHEMERAL memories"

    # 33. Verify counts match expected (core, permanent, ephemeral)
    total_core = sum(bd["core"] for bd in [breakdown_10, breakdown_15, breakdown_20])
    total_permanent = sum(bd["permanent"] for bd in [breakdown_10, breakdown_15, breakdown_20])
    total_ephemeral = sum(bd["ephemeral"] for bd in [breakdown_10, breakdown_15, breakdown_20])

    assert total_core == 1, "Should have 1 total CORE memory"
    assert total_permanent == 3, "Should have 3 total PERMANENT memories"
    assert total_ephemeral == 1, "Should have 1 total EPHEMERAL memory (Episode 2 only)"

    # Final validation: Persistent count should match file entries
    assert manager.get_persistent_count() == 4, "Should have 4 persistent memories total"
    assert manager.get_ephemeral_count() == 1, "Should have 1 ephemeral memory total (Episode 2)"
