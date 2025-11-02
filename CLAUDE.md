# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: Do not make any changes until you have 95% confidence in the change you need to make. Ask me questions until you reach that confidence

## Project Overview

ZorkGPT is an AI agent system that plays the classic text adventure game "Zork" using Large Language Models. The system uses a modular architecture with specialized LLM-driven components for action generation, information extraction, action evaluation, and adaptive learning.

**Key Principle**: All game reasoning must originate from LLMs - no hardcoded solutions or predetermined game mechanics are allowed.

### Game Interface Architecture
- **JerichoInterface** (`game_interface/core/jericho_interface.py`): Direct Z-machine access via Jericho library
- **Orchestrator** (`orchestration/zork_orchestrator_v2.py`): Streamlined coordination layer using JerichoInterface

This architecture enables:
- Direct Z-machine memory access (no text parsing for inventory, location, score)
- Stable integer-based location IDs (eliminates room fragmentation)
- Perfect movement detection via ID comparison
- Built-in save/restore functionality via Z-machine
- Clean separation of concerns
- ~40% reduction in LLM calls

## Jericho Integration (Phases 1-7 Complete)

### Architecture Overview

The system uses the Jericho library for direct Z-machine memory access, eliminating brittle text parsing and providing structured game state data.

**Key Components:**
- **JerichoInterface** (`game_interface/core/jericho_interface.py`): Direct Z-machine access layer
- **Integer-based Maps**: Location IDs from Z-machine (`location.num`) used as primary keys
- **Object Tree Integration**: Structured access to game objects with attributes and valid verbs
- **Hybrid Extraction**: Z-machine for core data, LLM for complex text parsing only

### Key Principles for Development

**Use Z-machine Data Directly:**
- Use `location.num` for room IDs (stable integers, never fragment)
- Use `get_inventory_structured()` for inventory (returns list of ZObjects)
- Use `get_location_structured()` for current location (returns ZObject with .num and .name)
- Use `get_visible_objects_in_location()` for objects in current room
- Use `get_score()` for score/moves without text parsing

**Movement Detection:**
- Compare location IDs before/after action: `before_id != after_id` means movement occurred
- NO text parsing needed - works perfectly in dark rooms, teleportation, etc.
- NO heuristics needed - IDs are ground truth from Z-machine

**Object Tree Validation:**
- Use `get_object_attributes(obj)` to check if object is takeable, openable, readable, etc.
- Use `get_valid_verbs()` to get valid action vocabulary from Z-machine
- Critic uses object tree validation BEFORE expensive LLM calls
- Fast rejection (microseconds) vs slow LLM evaluation (~800ms)

**Map Management:**
- MapGraph uses `Dict[int, Room]` with integer keys (location IDs)
- Room class has `id: int` as primary key, `name: str` for display only
- NO consolidation needed - Z-machine IDs guarantee uniqueness
- Multiple rooms can have same name but different IDs (both are distinct)

### Testing Guidelines

**Use Walkthrough Fixtures:**
- Import from `tests.fixtures.walkthrough` for deterministic tests
- `get_zork1_walkthrough()` - Full walkthrough from Jericho
- `get_walkthrough_slice(start, end)` - Subset of actions
- `get_walkthrough_until_lamp()` - First ~15 actions
- `get_walkthrough_dark_sequence()` - Dark room navigation
- `replay_walkthrough(env, actions)` - Execute sequence and collect results

**Example Test Pattern:**
```python
from tests.fixtures.walkthrough import get_walkthrough_slice, replay_walkthrough
from game_interface.core.jericho_interface import JerichoInterface

def test_location_id_stability():
    """Verify location IDs are deterministic across replays."""
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    walkthrough = get_walkthrough_slice(0, 20)

    # First run
    location_ids = []
    for action in walkthrough:
        interface.send_command(action)
        loc = interface.get_location_structured()
        location_ids.append(loc.num)

    # Second run should match exactly
    interface2 = JerichoInterface(rom_path="infrastructure/zork.z5")
    replay_ids = []
    for action in walkthrough:
        interface2.send_command(action)
        loc = interface2.get_location_structured()
        replay_ids.append(loc.num)

    assert location_ids == replay_ids, "Location IDs must be deterministic"
```

**Run Benchmarks:**
```bash
# Performance validation
uv run python benchmarks/comparison_report.py

# Individual benchmarks
uv run python benchmarks/performance_metrics.py
```

### Performance Metrics (Validated)

**Code Reduction:**
- 739 lines deleted (11-12% of codebase)
- Regex parsing eliminated: ~100 lines
- Consolidation methods eliminated: 512 lines
- Exit compatibility logic eliminated: 77 lines
- Movement heuristics eliminated: 150 lines

**LLM Call Reduction:**
- 40% per-turn reduction (5 calls → 3 calls)
- Inventory: LLM → instant Z-machine
- Location: LLM → instant Z-machine
- Score: LLM → instant Z-machine
- Visible objects: Text parsing → object tree
- Phase 5 bonus: 83.3% reduction for invalid actions via object tree validation

**Quality Improvements:**
- Room fragmentation: 0 (guaranteed by Z-machine IDs)
- Movement detection: 100% accuracy (ID comparison)
- Dark room handling: Perfect (IDs work regardless of visibility)
- Walkthrough completion: 350/350 score
- Turn processing: 15,000+ actions/second

### Common Patterns

**Extracting Current State:**
```python
# Get structured location
location = jericho_interface.get_location_structured()
room_id = location.num       # Integer ID (primary key)
room_name = location.name    # Display name (can duplicate)

# Get inventory
inventory = jericho_interface.get_inventory_structured()
for item in inventory:
    print(f"Item: {item.name} (ID: {item.num})")
    attrs = jericho_interface.get_object_attributes(item)
    if attrs.get("takeable"):
        print("  -> Can be taken")

# Get visible objects
visible = jericho_interface.get_visible_objects_in_location()
for obj in visible:
    print(f"Object: {obj.name} (ID: {obj.num})")

# Get score
score, moves = jericho_interface.get_score()
```

**Detecting Movement:**
```python
# Before action
before_location = jericho_interface.get_location_structured()
before_id = before_location.num

# Execute action
response = jericho_interface.send_command("north")

# After action
after_location = jericho_interface.get_location_structured()
after_id = after_location.num

# Movement detection (perfect accuracy)
if before_id != after_id:
    print(f"Moved from room {before_id} to room {after_id}")
    # Update map with connection
    map_graph.add_connection(before_id, "north", after_id)
```

**Object Tree Validation (Critic):**
```python
# Fast validation before LLM call
def _validate_against_object_tree(action: str, jericho_interface):
    if "take " in action.lower():
        object_name = action.lower().replace("take ", "").strip()

        # Get visible objects
        visible = jericho_interface.get_visible_objects_in_location()

        # Check if object exists and is takeable
        for obj in visible:
            if object_name in obj.name.lower():
                attrs = jericho_interface.get_object_attributes(obj)
                if not attrs.get("takeable"):
                    return ValidationResult(
                        valid=False,
                        reason=f"{obj.name} is not takeable",
                        confidence=0.9  # High confidence in Z-machine data
                    )
                return ValidationResult(valid=True)

        return ValidationResult(
            valid=False,
            reason=f"{object_name} not visible",
            confidence=0.9
        )

    return ValidationResult(valid=True)  # Allow action by default
```

### What NOT to Do

**Don't:**
- Parse text for inventory, location, or score (use Z-machine methods)
- Use room names as primary keys (use location IDs)
- Implement consolidation logic (Z-machine IDs prevent fragmentation)
- Use heuristics for movement detection (use ID comparison)
- Call LLM before object tree validation (validate first, LLM second)
- Assume room names are unique (multiple rooms can share names with different IDs)
- Try to normalize or canonicalize location IDs (they're already canonical from Z-machine)

**Do:**
- Trust Z-machine data as ground truth
- Use IDs for all map operations
- Use names for display/logging only
- Validate with object tree before expensive LLM calls
- Test with walkthrough fixtures for determinism
- Measure performance with benchmarks

### Migration Notes

If you're working with code that pre-dates Jericho integration:

**Old Pattern (DEPRECATED):**
```python
# DON'T DO THIS
room_name = extract_location_from_text(game_text)
map_graph.add_room(room_name)
```

**New Pattern (CORRECT):**
```python
# DO THIS
location = jericho_interface.get_location_structured()
room_id = location.num
room_name = location.name
map_graph.add_room(room_id, room_name)
```

### Phase Summary

- **Phase 1**: JerichoInterface foundation - dfrotz eliminated
- **Phase 2**: Direct Z-machine access - 40% LLM reduction
- **Phase 3**: Integer-based maps - 512 lines deleted, zero fragmentation
- **Phase 4**: Movement detection - ID comparison (100% accuracy)
- **Phase 5**: Object tree integration - 83.3% LLM reduction for invalid actions
- **Phase 6**: State loop detection - exact state tracking
- **Phase 7**: Testing & validation - 74 Phase 5 tests, walkthrough-based testing

All phases complete and validated. See `refactor.md` for detailed implementation notes.

### Cross-Episode Map Persistence

The map automatically persists across episodes via `map_state.json`, enabling spatial knowledge accumulation similar to how `knowledgebase.md` accumulates strategic wisdom.

**How It Works:**
- Map state saved at episode end via `MapManager.save_map_state()` (called from orchestrator)
- Map state loaded at MapManager initialization (if file exists)
- First episode starts with empty map (file doesn't exist yet)
- Subsequent episodes seed from previous spatial knowledge

**What Gets Persisted:**
- All rooms discovered (ID, name, exits)
- All connections with confidence scores
- Connection verification counts (tracks reliability)
- Exit failure tracking and pruned exits
- Map quality metadata (version, totals, timestamp)

**JSON Structure Example:**
```json
{
  "rooms": {
    "15": {
      "id": 15,
      "name": "West of House",
      "exits": ["north", "south", "west"]
    }
  },
  "connections": {
    "15": {"north": 180}
  },
  "connection_confidence": {
    "15_north": 0.95
  },
  "connection_verifications": {
    "15_north": 3
  },
  "exit_failure_counts": {
    "15_west": 2
  },
  "pruned_exits": {
    "15": ["east"]
  },
  "metadata": {
    "version": "1.0",
    "total_rooms": 42,
    "total_connections": 67,
    "timestamp": "2025-10-31T12:00:00Z"
  }
}
```

**Configuration:**
- File path: `map_state_file` in `pyproject.toml` (default: `map_state.json`)
- Automatically excluded from version control via `.gitignore`

**When to Reset:**
Delete `map_state.json` to start fresh spatial learning (useful for testing or after game mechanics changes).

**Implementation Details:**
- Serialization: `MapGraph.to_dict()` and `MapGraph.from_dict()`
- File operations: `MapGraph.save_to_json()` and `MapGraph.load_from_json()`
- Graceful handling of missing/corrupted files (returns None, starts fresh)
- Comprehensive test coverage: `tests/test_map_persistence.py` (19 tests)

### Manager-Based Architecture

The refactored system follows a **manager pattern** where specialized managers handle distinct responsibilities:

#### Core Session Management
- **GameState** (`session/game_state.py`): Centralized shared state using dataclass pattern
- **GameConfiguration** (`session/game_configuration.py`): Configuration management with proper precedence

#### Specialized Managers
- **ObjectiveManager** (`managers/objective_manager.py`): Objective discovery, tracking, completion, and refinement
- **KnowledgeManager** (`managers/knowledge_manager.py`): Knowledge updates, synthesis, and learning integration
- **MapManager** (`managers/map_manager.py`): Map building, navigation, and spatial intelligence
- **StateManager** (`managers/state_manager.py`): State export, context management, and memory tracking
- **ContextManager** (`managers/context_manager.py`): Context assembly and prompt preparation
- **EpisodeSynthesizer** (`managers/episode_synthesizer.py`): Episode lifecycle and synthesis coordination

#### LLM-Powered Components
- **Agent** (`zork_agent.py`): Generates game actions based on current state and context
- **Extractor** (`hybrid_zork_extractor.py`): Parses raw game text into structured information
- **Critic** (`zork_critic.py`): Evaluates proposed actions before execution with confidence scoring
- **Strategy Generator** (`knowledge/adaptive_manager.py`): Manages adaptive knowledge and continuous learning

#### Supporting Systems
- **Map Graph** (`map_graph.py`): Builds and maintains spatial understanding with confidence tracking
- **Movement Analyzer** (`movement_analyzer.py`): Analyzes movement patterns and spatial relationships
- **Logger** (`logger.py`): Comprehensive logging for analysis and debugging
- **LLM Client** (`llm_client.py`): Custom LLM client with advanced sampling parameters and standardized response handling

### Manager Lifecycle and Dependencies

Managers follow a standardized lifecycle:
1. **Initialization**: Dependency injection with logger, config, and game state
2. **Reset**: Episode-specific state cleanup for new episodes
3. **Processing**: Turn-based and periodic update processing
4. **Status**: Comprehensive status reporting for monitoring

Dependency flow:
- MapManager → no dependencies
- ContextManager → no dependencies
- StateManager → needs LLM client
- KnowledgeManager → needs agent and map references
- ObjectiveManager → needs knowledge manager reference
- EpisodeSynthesizer → needs knowledge and state managers

### Knowledge Base Structure

The ZorkGPT knowledge base (`knowledgebase.md`) is a consolidated document containing:

#### Strategic Sections (Updated during gameplay)
- **DANGERS & THREATS**: Specific dangers and recognition patterns
- **PUZZLE SOLUTIONS**: Puzzle mechanics and solutions
- **STRATEGIC PATTERNS**: Successful/failed approaches and patterns
- **DEATH & DANGER ANALYSIS**: Death event analysis and prevention
- **COMMAND SYNTAX**: Exact commands that worked
- **LESSONS LEARNED**: Session-specific insights

#### CROSS-EPISODE INSIGHTS (Updated at episode completion)
This section synthesizes persistent wisdom that carries across multiple episodes:
- **Death Patterns Across Episodes**: Consistent death causes and prevention strategies
- **Environmental Knowledge**: Persistent facts about game world (dangerous locations, item behaviors, puzzle mechanics)
- **Strategic Meta-Patterns**: Approaches that prove consistently effective/ineffective across different situations
- **Major Discoveries**: Game mechanics, hidden areas, puzzle solutions discovered

This section is updated via `synthesize_inter_episode_wisdom()` at episode end when:
- Episode ended in death (critical learning event), OR
- Final score >= 50, OR
- Turn count >= 100, OR
- Average critic score >= 0.3

**Note on Map Storage**: The spatial map is NOT stored in `knowledgebase.md`. It is persisted in `map_state.json` (see "Cross-Episode Map Persistence" section above) and passed to the agent dynamically via context at each turn. The knowledge base focuses solely on strategic wisdom, not spatial/factual data.

**Historical Note**: Cross-episode wisdom was previously stored in a separate `persistent_wisdom.md` file but has been consolidated into the CROSS-EPISODE INSIGHTS section of `knowledgebase.md`.

## Memories and Principles

- The state should be exported at the end of every turn.

## Development Commands

[Rest of the file remains the same as in the original content]