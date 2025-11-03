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

## Jericho Integration ()

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

### Reasoning History and Strategic Continuity

**Problem Solved**: Without access to previous reasoning, the agent experiences "strategic amnesia" - it must re-derive strategies each turn, breaking continuity for multi-step plans.

**Solution**: The system now captures and includes agent reasoning in turn context, enabling strategic continuity across turns.

#### Architecture

**Data Storage** (`session/game_state.py`):
- `action_reasoning_history: List[Dict[str, Any]]` - Stores recent reasoning entries
- Each entry: `{"turn": int, "reasoning": str, "action": str, "timestamp": str}`
- Automatically cleared on episode reset
- History accumulates during episode (unbounded; consider size limit for very long episodes)

**Capture** (`orchestration/zork_orchestrator_v2.py`):
- After `agent.get_action_with_reasoning()` returns, reasoning is captured
- `ContextManager.add_reasoning()` stores entry in game state
- Reasoning extracted from `<think>`, `<thinking>`, or `<reflection>` tags

**Context Inclusion** (`managers/context_manager.py`):
- `get_recent_reasoning_formatted()` method formats last 3 turns
- Includes: turn number, reasoning, action, game response
- Section added to agent prompt: "## Previous Reasoning and Actions"
- Positioned after game state, before objectives (strategic context flow)
- Conditionally included (only when history exists)

**Agent Guidance** (`agent.md`):
- Explicit instructions on using previous reasoning
- Three scenarios: continuing plans, revising plans, starting fresh
- Encourages building on prior thinking rather than restarting

#### Example Context Format

```markdown
## Previous Reasoning and Actions

Turn 47:
Reasoning: I need to explore north systematically. Plan: (1) go north, (2) search area, (3) return if nothing found.
Action: go north
Response: You are in a forest clearing. Trees surround you.

Turn 48:
Reasoning: Continuing systematic exploration. Will examine objects before moving on.
Action: examine trees
Response: The trees are ordinary pine trees.

Turn 49:
Reasoning: Nothing interesting here. Moving east to continue exploration.
Action: go east
Response: You are in a meadow.
```

#### Benefits

1. **Strategic Continuity**: Agent maintains multi-step plans across turns
2. **Reduced Redundancy**: No need to re-explain same strategy each turn
3. **Progress Tracking**: Agent explicitly references previous steps
4. **Adaptive Learning**: Agent must explicitly justify plan changes
5. **Reasoning Efficiency**: Shorter reasoning blocks that build on prior thinking

#### Edge Case Handling

- **Empty history** (first turns): Section not included in context
- **Missing fields**: Graceful fallbacks ("(No reasoning recorded)")
- **Duplicate actions**: Matches most recent response (reverse iteration)
- **Non-dict entries**: Skipped gracefully with logging

#### Configuration

- **Window size**: Last 3 turns included in agent context
  - Location: `managers/context_manager.py:505` (`num_turns=3` parameter)
- **Storage**: Unbounded during episode (cleared on episode reset)

#### Testing

- Unit tests: `tests/test_phase5_enhanced_context.py` (27 tests)
- Integration tests: `tests/test_phase3_reasoning_guidance.py` (9 tests)
- Coverage: Data structures, formatting, edge cases, prompt integration

### Multi-Step Memory Synthesis

**Problem Solved**: Original memory synthesis was turn-atomic, only seeing single action/response pairs. This prevented capturing procedural knowledge that spans multiple turns (e.g., "open window, then enter window" or "troll accepts gift → attacks later").

**Solution**: SimpleMemoryManager now includes recent action and reasoning history in synthesis context, enabling LLM to detect multi-step procedures, delayed consequences, and progressive discoveries.

#### Architecture

**Configuration** (`pyproject.toml`):
```toml
[tool.zorkgpt.memory_sampling]
temperature = 0.3
max_tokens = 1000
memory_history_window = 3  # Number of recent turns for multi-step detection
```

**Data Retrieval** (`managers/simple_memory_manager.py:1134-1162`):
- Retrieves last N actions from `game_state.action_history`
- Retrieves last N reasoning entries from `game_state.action_reasoning_history`
- Window size controlled by `config.get_memory_history_window()` (default: 3)
- Gracefully handles empty history (early turns)

**Formatting Helpers** (`managers/simple_memory_manager.py:1552-1671`):
- `_format_recent_actions()`: Formats action/response pairs into markdown
- `_format_recent_reasoning()`: Formats reasoning entries with response matching
- Both match ContextManager conventions for consistency

**Synthesis Prompt Enhancement** (`managers/simple_memory_manager.py:1164-1294`):
- Adds "RECENT ACTION SEQUENCE" section before current action analysis
- Includes formatted actions and agent reasoning
- Provides explicit multi-step procedure detection guidance
- Positioned to give LLM full temporal context

#### Multi-Step Patterns Detected

**1. Prerequisites** (action B requires action A first):
```
Turn 47: examine window
Response: The window is slightly ajar. Perhaps you could open it further.

Turn 48: open window
Response: With some effort, you open the window wide enough to pass through.

Turn 49: enter window
Response: You climb through the window. Kitchen: You are in a small kitchen.

→ Memory: "To enter kitchen from behind house: (1) examine window to confirm it's usable, (2) open window to make it passable, (3) enter window to reach kitchen. Window requires opening before entry is possible."
→ Status: ACTIVE
→ Location: 79 (Behind House - where sequence started)
```

**2. Delayed Consequences** (action seemed successful but had delayed effect):
```
Turn 12: give lunch to troll
Response: The troll graciously accepts the lunch and eats it hungrily.

→ Memory (Turn 12): "Troll accepts lunch gift"
→ Status: TENTATIVE (outcome unclear)
→ Location: 152 (Troll Room)

Turn 13: go north
Response: The troll frowns and blocks your path, snarling. The troll swings his axe and cleaves you in twain!

→ Memory (Turn 13): "Troll attacks after accepting gift - gift strategy fails"
→ Status: ACTIVE (confirmed by death)
→ Supersedes: ["Troll accepts lunch gift"]
→ Location: 152 (Troll Room)
```

**3. Progressive Discovery** (understanding deepens over multiple turns):
```
Turn 20: examine door
Response: The door is locked.

Turn 21: unlock door with rusty key
Response: The key doesn't fit this lock.

Turn 22: unlock door with brass key
Response: The brass key turns smoothly in the lock. *Click*

Turn 23: open door
Response: The door swings open, revealing a dimly lit chamber.

→ Memory: "Stone door in hallway requires brass key (not rusty key) to unlock before opening. Multi-step: unlock with brass key, then open door."
→ Status: ACTIVE
→ Location: 134 (Hallway)
```

#### Memory Status Types

**ACTIVE**: Confirmed knowledge, high confidence
- Used for verified procedures, confirmed facts
- Replaces previous TENTATIVE memories when confirmed
- Example: "To enter kitchen: (1) open window, (2) enter window"

**TENTATIVE**: Uncertain outcome, awaiting confirmation
- Used when action succeeds but consequences unclear
- Should be superseded or confirmed by later evidence
- Example: "Troll accepts lunch gift" (before seeing if it works)

**SUPERSEDED**: Contradicted by later evidence
- Old memory marked when new evidence contradicts it
- Preserved for learning but not used for decision-making
- Metadata includes: `[Superseded at T<turn> by "<new_memory_title>"]`
- Example: TENTATIVE memory superseded when troll attacks

#### Source Location Storage Principle

**Critical Rule**: Memories are ALWAYS stored at the SOURCE location (where action was taken), NOT the destination.

**Rationale**:
- Destination storage: "At Kitchen, I know window entry works" → Useless (already there)
- Source storage: "At Behind House, I know 'enter window' leads to Kitchen" → Useful for next visit

**Implementation** (`orchestration/zork_orchestrator_v2.py:889-893, 963-964`):
```python
# Capture location BEFORE action
location_name_before = location_before.name if location_before else "Unknown"

# Store memory at SOURCE location (where action was taken)
self.simple_memory.record_action_outcome(
    location_id=location_id_before,      # SOURCE, not destination
    location_name=location_name_before,  # SOURCE, not destination
    action=action_to_take,
    response=clean_response,
    z_machine_context=z_machine_context
)
```

**Impact**: Episode 2 agent benefits from Episode 1 discoveries when returning to same locations.

#### Supersession Workflow

When LLM detects contradiction in synthesis:

1. **LLM Response**:
```json
{
  "should_remember": true,
  "category": "DANGER",
  "memory_title": "Troll attacks after accepting gift",
  "memory_text": "Troll accepts lunch gift but then becomes hostile...",
  "status": "ACTIVE",
  "supersedes_memory_titles": ["Troll accepts lunch gift"],
  "reasoning": "This contradicts the previous TENTATIVE memory..."
}
```

2. **SimpleMemoryManager Processing** (`managers/simple_memory_manager.py:1440-1475`):
   - Creates new ACTIVE memory
   - Searches memory file for titles in `supersedes_memory_titles`
   - Marks old memories as SUPERSEDED with metadata
   - Atomic file write with backup creation (thread-safe)

3. **Memory File Result**:
```markdown
**[DANGER] Troll attacks after accepting gift** *(Epep_01, T13, +0)*
The troll accepts lunch gift but becomes hostile and attacks. Gift strategy fails.

**[NOTE - SUPERSEDED] Troll accepts lunch gift** *(Epep_01, T12, +0)*
[Superseded at T13 by "Troll attacks after accepting gift"]
Troll accepts lunch offering graciously. Reaction to gift unclear.
```

#### Configuration Details

**Window Size** (`pyproject.toml`):
- Default: 3 actions (recommended range: 3-8)
- Minimum: 1 (validated by `GameConfiguration.get_memory_history_window()`)
- Warning if > 10 (excessive context usage)

**Validation** (`session/game_configuration.py:get_memory_history_window()`):
```python
def get_memory_history_window(self) -> int:
    """Get number of recent turns for multi-step detection."""
    window = self.memory_sampling.get("memory_history_window", 3)

    if window < 1:
        warnings.warn("memory_history_window must be >= 1, using default: 3")
        return 3

    if window > 10:
        warnings.warn("memory_history_window = {window} may use excessive context. Recommended: 3-8")

    return window
```

#### Testing Coverage

**Integration Tests**:
- `tests/test_multi_step_window_sequence.py` (6 tests): Window entry prerequisite scenario
- `tests/test_multi_step_delayed_consequence.py` (5 tests): Troll attack delayed consequence + supersession
- `tests/simple_memory/test_movement_memory_location.py` (7 tests): Source location storage verification
- `tests/simple_memory/test_simple_memory_formatting.py` (18 tests): History formatting helpers

**Test Scenarios**:
1. **Window sequence captures full procedure**: Validates multi-step memory includes all prerequisite actions
2. **Without history misses procedure**: Control test proving Phase 3's value
3. **History formatting matches ContextManager**: Ensures consistency across systems
4. **Partial history (window size = 2)**: Tests sliding window behavior
5. **First turn no history**: Edge case with empty history
6. **Score delta calculation**: Metadata validation
7. **TENTATIVE → SUPERSEDED flow**: Validates status transitions
8. **Supersession with death**: Death event triggers supersession
9. **No supersession without history**: Control test for history requirement
10. **Response field lookup**: Validates reverse iteration matching

**Test Results**: 36 new multi-step memory tests passing across 4 test files

#### Example Synthesis Prompts

**Turn 49 (Multi-step window entry)**:
```markdown
RECENT ACTION SEQUENCE:
═══════════════════════════════════════════════════════════════

Turn 47: examine window
Response: The window is slightly ajar. Perhaps you could open it further.

Turn 48: open window
Response: With some effort, you open the window wide enough to pass through.

Turn 49: enter window
Response: You climb through the window.

AGENT'S REASONING:
Turn 47:
Reasoning: Need to find entry to kitchen. Will examine window first.
Action: examine window
Response: The window is slightly ajar. Perhaps you could open it further.

Turn 48:
Reasoning: Window seems openable. Will try opening before entering.
Action: open window
Response: With some effort, you open the window wide enough to pass through.

Turn 49:
Reasoning: Window is now open. Attempting entry.
Action: enter window
Response: You climb through the window.

═══════════════════════════════════════════════════════════════

MULTI-STEP PROCEDURE DETECTION:
Review the RECENT ACTION SEQUENCE above. Does the current outcome depend on previous actions?

**Look for these patterns:**

1. **Prerequisites** (action B requires action A first):
   Example: "open window" (turn N) → "enter window" (turn N+1) → success
   Memory: "To enter kitchen: (1) open window, (2) enter window"
```

**Turn 13 (Delayed consequence with death)**:
```markdown
RECENT ACTION SEQUENCE:
═══════════════════════════════════════════════════════════════

Turn 12: give lunch to troll
Response: The troll graciously accepts the lunch and eats it hungrily.

Turn 13: go north
Response: The troll frowns and blocks your path, snarling. The troll swings his axe and cleaves you in twain!

AGENT'S REASONING:
Turn 12:
Reasoning: Maybe I can pacify the troll with food. Worth trying.
Action: give lunch to troll
Response: The troll graciously accepts the lunch and eats it hungrily.

Turn 13:
Reasoning: Troll accepted gift, seems calm. Will try moving north.
Action: go north
Response: The troll frowns and blocks your path, snarling. The troll swings his axe and cleaves you in twain!

═══════════════════════════════════════════════════════════════

MULTI-STEP PROCEDURE DETECTION:

2. **Delayed Consequences** (action seemed successful but had delayed effect):
   Example: "give lunch to troll" (turn N, seemed ok) → troll attacks (turn N+1)
   Memory: "Troll attacks after accepting gift - gift strategy fails"
   Action: Mark previous TENTATIVE memory as SUPERSEDED
```

#### Benefits

1. **Cross-Turn Learning**: Captures procedures spanning 2-5 actions
2. **Delayed Feedback**: Recognizes when initial success leads to later failure
3. **Adaptive Memory**: TENTATIVE → ACTIVE or SUPERSEDED based on evidence
4. **Location-Specific**: Memories stored at source for contextual relevance
5. **Cross-Episode Transfer**: Episode 2 benefits from Episode 1 multi-step discoveries

#### Common Pitfalls

**Don't:**
- Store memories at destination location (breaks cross-episode learning)
- Use window size > 10 (excessive context, diminishing returns)
- Ignore status field in Memory creation (critical bug - all memories default to ACTIVE)
- Parse action history manually (use `_format_recent_actions()` helper)

**Do:**
- Always capture `location_id_before` and `location_name_before` in orchestrator
- Pass `status=synthesis.status` when creating Memory objects
- Use dedicated formatting helpers for consistency
- Test with multi-step integration scenarios (not just unit tests)
- Monitor TENTATIVE memory accumulation (should be superseded or confirmed)

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
