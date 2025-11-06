# Managers - Architecture and Implementation Guide

This guide covers the manager-based architecture, lifecycle patterns, and specialized manager implementations including the memory synthesis system.

## Manager-Based Architecture

The refactored system follows a **manager pattern** where specialized managers handle distinct responsibilities:

### Core Session Management
- **GameState** (`session/game_state.py`): Centralized shared state using dataclass pattern
- **GameConfiguration** (`session/game_configuration.py`): Configuration management with proper precedence

### Specialized Managers
- **ObjectiveManager** (`objective_manager.py`): Objective discovery, tracking, completion, and refinement
- **KnowledgeManager** (`knowledge_manager.py`): Knowledge updates, synthesis, and learning integration
- **MapManager** (`map_manager.py`): Map building, navigation, and spatial intelligence
- **StateManager** (`state_manager.py`): State export, context management, and memory tracking
- **ContextManager** (`context_manager.py`): Context assembly and prompt preparation
- **EpisodeSynthesizer** (`episode_synthesizer.py`): Episode lifecycle and synthesis coordination
- **SimpleMemoryManager** (`simple_memory_manager.py`): Location-specific memory synthesis with multi-step detection

### LLM-Powered Components (Root Directory)
- **Agent** (`../zork_agent.py`): Generates game actions based on current state and context
- **Extractor** (`../hybrid_zork_extractor.py`): Parses raw game text into structured information
- **Critic** (`../zork_critic.py`): Evaluates proposed actions before execution with confidence scoring

### Supporting Systems
- **Map Graph** (`../map_graph.py`): Builds and maintains spatial understanding with confidence tracking
- **Movement Analyzer** (`../movement_analyzer.py`): Analyzes movement patterns and spatial relationships
- **Logger** (`../logger.py`): Comprehensive logging for analysis and debugging
- **LLM Client** (`../llm_client.py`): Custom LLM client with advanced sampling parameters

## Manager Lifecycle and Dependencies

Managers follow a standardized lifecycle:

### 1. Initialization
Dependency injection with logger, config, and game state:

```python
def __init__(
    self,
    logger: logging.Logger,
    config: GameConfiguration,
    game_state: GameState,
    # Additional dependencies as needed
):
    self.logger = logger
    self.config = config
    self.game_state = game_state
```

### 2. Reset
Episode-specific state cleanup for new episodes:

```python
def reset(self) -> None:
    """Reset manager state for new episode."""
    # Clear episode-specific data
    # Preserve cross-episode learned data if applicable
```

### 3. Processing
Turn-based and periodic update processing:

```python
def process_turn(self, ...) -> None:
    """Process turn-specific updates."""

def process_periodic_update(self, ...) -> None:
    """Process periodic updates (every N turns)."""
```

### 4. Status
Comprehensive status reporting for monitoring:

```python
def get_status(self) -> Dict[str, Any]:
    """Return current manager status for monitoring."""
    return {
        "type": "manager_type",
        "state": {...},
        "metrics": {...}
    }
```

## Manager Dependency Flow

**Important:** Initialize managers in dependency order:

```
MapManager → no dependencies
ContextManager → no dependencies
StateManager → needs LLM client
KnowledgeManager → needs agent and map references
ObjectiveManager → needs knowledge manager reference
EpisodeSynthesizer → needs knowledge and state managers
SimpleMemoryManager → needs game_state, config, logger
```

## Multi-Step Memory Synthesis

**Problem Solved**: Original memory synthesis was turn-atomic, only seeing single action/response pairs. This prevented capturing procedural knowledge that spans multiple turns.

**Solution**: SimpleMemoryManager includes recent action and reasoning history in synthesis context, enabling detection of multi-step procedures, delayed consequences, and progressive discoveries.

### Configuration

In `pyproject.toml`:
```toml
[tool.zorkgpt.memory_sampling]
temperature = 0.3
max_tokens = 1000
memory_history_window = 3  # Number of recent turns for multi-step detection
```

**Recommended window size:** 3-8 turns
- Minimum: 1 (validated by `GameConfiguration.get_memory_history_window()`)
- Warning if > 10 (excessive context usage)

### Multi-Step Patterns Detected

#### 1. Prerequisites (action B requires action A first)

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

#### 2. Delayed Consequences (action seemed successful but had delayed effect)

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

#### 3. Progressive Discovery (understanding deepens over multiple turns)

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

### Memory Status Types

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

### Source Location Storage Principle

**CRITICAL RULE**: Memories are ALWAYS stored at the SOURCE location (where action was taken), NOT the destination.

**Rationale:**
- Destination storage: "At Kitchen, I know window entry works" → Useless (already there)
- Source storage: "At Behind House, I know 'enter window' leads to Kitchen" → Useful for next visit

**Implementation in orchestrator:**
```python
# Capture location BEFORE action (in orchestration/zork_orchestrator_v2.py)
location_before = jericho_interface.get_location_structured()
location_id_before = location_before.num if location_before else None
location_name_before = location_before.name if location_before else "Unknown"

# Execute action
response = jericho_interface.send_command(action)

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

## Standalone Memory Invalidation

**Problem Solved**: Original memory system could only mark memories as wrong by creating replacement memories (supersession). This required always having a "better" version even when the key insight is simply "this was false."

**Solution**: SimpleMemoryManager supports standalone invalidation - marking memories as SUPERSEDED without creating a replacement, using INVALIDATION_MARKER as a sentinel value.

### When to Use Invalidation vs Supersession

**Use INVALIDATION when:**
- Memory proven false without specific replacement needed
- Multiple unrelated memories all wrong due to core false assumption
- Death invalidates speculative/TENTATIVE memories
- Evidence shows memory is incorrect but correct approach isn't yet known

**Use SUPERSESSION when:**
- Old memory was close but needs refinement
- Better understanding of same situation
- Specific correction or update to existing memory

### Public API Methods

#### Single Invalidation
```python
success = manager.invalidate_memory(
    location_id=152,
    memory_title="Troll is friendly",
    reason="Proven false by death",
    turn=25  # Optional, defaults to current turn
)
```

#### Batch Invalidation
```python
results = manager.invalidate_memories(
    location_id=152,
    memory_titles=["Troll is friendly", "Troll accepts gifts"],
    reason="Both proven false by troll attack"
)
# Returns: {'Troll is friendly': True, 'Troll accepts gifts': True}
```

### LLM-Driven Invalidation

The LLM synthesis workflow supports standalone invalidation via MemorySynthesisResponse:

```python
# Invalidate without creating new memory
{
  "should_remember": false,
  "invalidate_memory_titles": ["Old assumption"],
  "invalidation_reason": "Proven false by evidence",
  "reasoning": "Death proves speculation wrong, no new memory needed"
}

# Create new memory AND invalidate unrelated wrong assumptions
{
  "should_remember": true,
  "memory_title": "Correct approach",
  "memory_text": "Detailed correct information",
  "category": "NOTE",
  "supersedes_memory_titles": ["Close but incomplete"],
  "invalidate_memory_titles": ["Unrelated wrong assumption"],
  "invalidation_reason": "Proven false separately",
  "reasoning": "Superseding one memory while invalidating another"
}
```

### File Format

**Superseded Memory (Traditional):**
```markdown
**[DANGER - SUPERSEDED] Old approach** *(Ep01, T23, +0)*
[Superseded at T50 by "Better approach"]
~~Troll accepts lunch gift graciously.~~
```

**Invalidated Memory (Standalone):**
```markdown
**[DANGER - SUPERSEDED] Troll is friendly** *(Ep01, T23, +0)*
[Invalidated at T25: "Proven false by death"]
~~Troll seems friendly when approached slowly.~~
```

### Memory Status Lifecycle

Both supersession and standalone invalidation use the same `SUPERSEDED` status:

1. **ACTIVE**: Confirmed reliable memory (default)
2. **TENTATIVE**: Appears true but may be invalidated by future evidence
3. **SUPERSEDED**: Proven wrong, either:
   - Replaced by better memory (`superseded_by` = new memory title)
   - Standalone invalidation (`superseded_by` = INVALIDATION_MARKER, `invalidation_reason` = explanation)

### Implementation Details

**Data Model:**
- `Memory.invalidation_reason: Optional[str]` - Reason for standalone invalidation
- `INVALIDATION_MARKER: str = "INVALIDATED"` - Sentinel value for `superseded_by` field
- `MemorySynthesisResponse.invalidate_memory_titles: Set[str]` - LLM can request invalidation
- `MemorySynthesisResponse.invalidation_reason: Optional[str]` - Shared reason for all invalidations

**Core Logic:**
- `_update_memory_status()` handles both supersession and invalidation
- Conditional reference line generation based on `invalidation_reason` presence
- File format: `[Invalidated at T{turn}: "{reason}"]` vs `[Superseded at T{turn} by "{title}"]`

**Processing Flow:**
1. LLM synthesis detects contradiction or false memory
2. Decides: supersession (replacement) or invalidation (standalone)
3. `record_action_outcome()` processes both supersessions and invalidations
4. File and cache updated atomically with file locking

### Example Scenarios

**Scenario 1: Death Invalidates Speculation**
```
Turn 20: Agent assumes "Troll might be friendly" (TENTATIVE)
Turn 25: Agent dies from troll attack
→ Invalidate "Troll might be friendly", reason: "Proven false by death"
→ Create DANGER memory: "Troll attacks unprovoked"
```

**Scenario 2: Multiple Related Assumptions Wrong**
```
Turn 10: Agent believes "Troll accepts gifts", "Troll is pacified by food"
Turn 15: Troll attacks after accepting food
→ Invalidate both memories, reason: "Troll hostile regardless of gifts"
→ Create DANGER memory: "Troll attacks immediately after accepting food"
```

**Scenario 3: Core Assumption Disproven**
```
Turn 30: Agent believes "Door is unlocked", "Safe to enter"
Turn 35: Door was locked, entering triggers trap
→ Invalidate both memories, reason: "Door was locked, not unlocked"
→ Create DANGER memory: "Door locked, entering triggers trap"
```

### Configuration

No additional configuration needed. The feature uses existing memory system configuration:
- `memory_model`: Model for synthesis (default: configured in pyproject.toml)
- `memory_sampling`: Temperature and max_tokens for synthesis
- `memory_history_window`: Number of turns for multi-step detection (default: 3)

### Testing

**Test Coverage:**
- `test_phase3_public_invalidation_api.py`: 11 tests for public API
- `test_phase4_llm_integration.py`: 8 tests for LLM synthesis integration
- `test_validation_mutual_exclusivity.py`: 12 tests for Pydantic validation
- Total: 168 simple_memory tests passing

**Key Test Patterns:**
```python
# Test standalone invalidation
def test_invalidate_without_replacement(manager):
    manager.invalidate_memory(
        location_id=152,
        memory_title="Wrong assumption",
        reason="Proven false",
        turn=25
    )
    # Verify memory marked SUPERSEDED with INVALIDATION_MARKER

# Test LLM-driven invalidation
def test_llm_invalidation_workflow(manager):
    synthesis = MemorySynthesisResponse(
        should_remember=False,
        invalidate_memory_titles={"Old memory"},
        invalidation_reason="Proven false"
    )
    # Verify record_action_outcome processes correctly
```

### Common Pitfalls for Invalidation

**Don't:**
- Use invalidation when you have a specific replacement (use supersession instead)
- Create redundant memories after invalidation (e.g., "I died" is already known)
- Invalidate memories at destination location (use source location where action was taken)

**Do:**
- Invalidate TENTATIVE memories when proven false
- Batch invalidate related memories with shared reason
- Create new DANGER/NOTE memories after invalidation if new information discovered
- Use clear invalidation reasons explaining why memory was wrong

## Ephemeral Memory System

**Problem Solved**: Agent-caused state changes (e.g., "I dropped the sword here") were persisting across episode boundaries, causing false expectations when the agent returned to locations in new episodes where objects had been reset to spawn state.

**Solution**: Three-tier persistence classification system with dual cache architecture that automatically clears temporary agent state on episode reset while preserving permanent game knowledge.

### Three-Tier Persistence Classification

**CORE** (`persistence="core"`):
- Spawn state from first-visit room descriptions
- Resets each episode (rooms return to spawn state)
- Example: "Room has mailbox at spawn" (true every episode)
- Storage: File + memory_cache (persistent across code runs)

**PERMANENT** (`persistence="permanent"`):
- Game mechanics, reusable procedural knowledge
- Persists across all episodes indefinitely
- Example: "Open window → enter window leads to Kitchen"
- Storage: File + memory_cache (persistent across code runs)

**EPHEMERAL** (`persistence="ephemeral"`):
- Agent-caused temporary state changes
- Cleared on episode reset (agent state doesn't persist)
- Example: "Dropped sword at West of House" (not true next episode)
- Storage: ephemeral_cache only (in-memory, never written to file)

### Dual Cache Architecture

SimpleMemoryManager maintains two separate caches:

```python
# Persistent: Loaded from Memories.md (CORE + PERMANENT)
self.memory_cache: Dict[int, List[Memory]] = {}

# Ephemeral: In-memory only, cleared on episode reset
self.ephemeral_cache: Dict[int, List[Memory]] = {}
```

**Why Dual Caches?**
- Performance: Ephemeral memories never touch disk (fast operations)
- Isolation: Episode reset clears ephemeral_cache, preserves memory_cache
- Simplicity: add_memory() routes by persistence field automatically

### Persistence Routing

`add_memory()` automatically routes memories based on `persistence` field:

```python
# EPHEMERAL: In-memory only
memory = Memory(
    category="NOTE",
    title="Dropped sword here",
    episode=1,
    turns="25",
    score_change=0,
    text="I dropped the sword at this location for tactical reasons.",
    persistence="ephemeral"  # → ephemeral_cache only
)
manager.add_memory(location_id, location_name, memory)

# CORE or PERMANENT: File + cache
memory = Memory(
    category="SUCCESS",
    title="Window leads to Kitchen",
    episode=1,
    turns="48",
    score_change=5,
    text="Entering the window leads to the Kitchen.",
    persistence="permanent"  # → memory_cache + Memories.md
)
manager.add_memory(location_id, location_name, memory)
```

**Implementation in add_memory() (lines 653-742):**
```python
if memory.persistence == "ephemeral":
    # Route to ephemeral_cache (no file write)
    if location_id not in self.ephemeral_cache:
        self.ephemeral_cache[location_id] = []
    self.ephemeral_cache[location_id].append(memory)
    return True
else:
    # Route to file + memory_cache
    success = self._write_memory_to_file(location_id, location_name, memory)
    if success:
        if location_id not in self.memory_cache:
            self.memory_cache[location_id] = []
        self.memory_cache[location_id].append(memory)
    return success
```

### Cache Migration During Supersession

`supersede_memory()` handles migration from ephemeral to permanent:

```python
# Upgrade ephemeral to permanent (migration case)
old_ephemeral = Memory(
    title="Sword on ground",
    persistence="ephemeral",
    # ...
)

new_permanent = Memory(
    title="Sword is quest item, always here",
    persistence="permanent",
    # ...
)

manager.supersede_memory(
    location_id=15,
    location_name="Living Room",
    old_memory_title="Sword on ground",
    new_memory=new_permanent
)

# Result:
# - Old ephemeral marked SUPERSEDED in ephemeral_cache
# - New permanent added to memory_cache + file (migrated!)
# - Episode reset will clear old, keep new
```

**Implementation in supersede_memory() (lines 744-835):**
1. Searches both caches for old memory by title
2. Marks old memory as SUPERSEDED (stays in original cache)
3. Calls `add_memory()` for new memory (routing automatic)
4. Migration happens transparently via routing logic

#### Persistence Level Validation

Supersession enforces persistence compatibility to prevent data loss after episode resets.

**Validation Logic** (`supersede_memory()` lines 826-836):
```python
if old_memory.persistence in ["core", "permanent"] and new_memory.persistence == "ephemeral":
    self.log_warning("Cannot downgrade ... - would cause data loss after episode reset")
    return False
```

**Allowed Transitions**:
- Ephemeral → Ephemeral (state updates)
- Ephemeral → Permanent (upgrades to game mechanics)
- Permanent → Permanent (refinements)
- Core → Core (rare: correcting spawn state)
- Core → Permanent (spawn confirmations)

**Rejected Transitions** (return False, log warning):
- Permanent → Ephemeral (would lose game mechanic after reset)
- Core → Ephemeral (would lose spawn state after reset)

**Why This Matters**:

Without validation, this bug occurs:
1. Agent learns "Troll attacks on sight" (PERMANENT, in file)
2. Agent drops sword near troll (creates EPHEMERAL memory)
3. LLM incorrectly supersedes PERMANENT with EPHEMERAL
4. Episode resets → EPHEMERAL cleared, PERMANENT marked SUPERSEDED
5. Agent loses knowledge that troll is dangerous!

**Error Handling**:
- `supersede_memory()` returns `False` when downgrade attempted
- Warning logged with both persistence levels and reason
- Original memory unchanged (not marked SUPERSEDED)
- New ephemeral memory NOT added to cache

**Testing**: See `tests/simple_memory/test_supersede_validation.py` for validation tests.

### Episode Reset Behavior

`reset_episode()` clears ephemeral cache while preserving persistent cache:

```python
def reset_episode(self) -> None:
    """
    Reset manager state for new episode.

    CRITICAL: Clears ephemeral_cache to prevent false memories.
    Persistent cache (memory_cache) remains unchanged.
    """
    ephemeral_count = sum(len(mems) for mems in self.ephemeral_cache.values())
    self.ephemeral_cache.clear()

    self.log_info(
        f"Episode reset: Cleared {ephemeral_count} ephemeral memories",
        ephemeral_count=ephemeral_count
    )
    # Note: memory_cache (persistent) is NOT cleared
```

**Episode Lifecycle:**
```
Episode 1:
- Agent adds CORE memory "Room has mailbox" (persistent)
- Agent adds PERMANENT memory "Mailbox has leaflet" (persistent)
- Agent adds EPHEMERAL memory "Dropped sword here" (ephemeral)
- get_location_memory() returns all 3 memories

Episode Reset:
- ephemeral_cache cleared (sword memory gone)
- memory_cache unchanged (room + mailbox remain)

Episode 2:
- Agent returns to location
- get_location_memory() returns CORE + PERMANENT only
- No false "sword on ground" memory
- Agent can add new EPHEMERAL memories for Episode 2
```

### Memory Retrieval

`get_location_memory()` combines both caches:

```python
def get_location_memory(self, location_id: int) -> str:
    """
    Get formatted memories for location from BOTH caches.

    Combines:
    - memory_cache (CORE + PERMANENT from file)
    - ephemeral_cache (EPHEMERAL from current episode)

    Filters out SUPERSEDED status (not shown to agent).
    """
    # Get memories from both caches
    persistent_memories = self.memory_cache.get(location_id, [])
    ephemeral_memories = self.ephemeral_cache.get(location_id, [])
    all_memories = persistent_memories + ephemeral_memories

    # Apply status filtering and format
    # ...
```

### File Format with Persistence Markers

Memories written to Memories.md include persistence markers in category field:

**CORE Memory:**
```markdown
**[SUCCESS - CORE] West of House has mailbox at spawn** *(Ep1, T1, +0)*
The mailbox is present at West of House in spawn state every episode.
```

**PERMANENT Memory:**
```markdown
**[SUCCESS - PERMANENT] Open window → enter window → Kitchen** *(Ep1, T48, +5)*
Multi-step procedure: (1) open window, (2) enter window leads to Kitchen.
```

**EPHEMERAL Memory:**
```
NOT WRITTEN TO FILE (ephemeral_cache only)
```

**Format with Status:**
```markdown
**[CATEGORY - PERSISTENCE - STATUS] Title** *(Episode, Turn, Score)*
**[NOTE - PERMANENT - TENTATIVE] Troll might accept gifts** *(Ep1, T12, +0)*
**[SUCCESS - CORE - SUPERSEDED] Old spawn state** *(Ep1, T1, +0)*
```

### Public API Methods

**Cache Inspection:**
```python
# Count ephemeral memories
ephemeral_count = manager.get_ephemeral_count(location_id)
total_ephemeral = manager.get_ephemeral_count()  # All locations

# Count persistent memories (CORE + PERMANENT)
persistent_count = manager.get_persistent_count(location_id)
total_persistent = manager.get_persistent_count()  # All locations

# Detailed breakdown
breakdown = manager.get_memory_breakdown(location_id)
# Returns: {"core": 1, "permanent": 2, "ephemeral": 3}
```

**Memory Operations:**
```python
# Add memory (automatic routing)
manager.add_memory(location_id, location_name, memory)

# Supersede with migration
manager.supersede_memory(
    location_id,
    location_name,
    old_memory_title,
    new_memory
)

# Retrieve combined view
memories_text = manager.get_location_memory(location_id)

# Episode reset
manager.reset_episode()  # Clears ephemeral_cache
```

### Configuration

No additional configuration required. The system uses existing memory system configuration:

```toml
[tool.zorkgpt.memory_sampling]
temperature = 0.3
max_tokens = 1000
memory_history_window = 3
```

### Testing

**Test Coverage:** 196 total tests in simple_memory suite

**Ephemeral-Specific Tests:**
- `test_memory_persistence.py` (4 tests): Memory dataclass persistence field validation
- `test_memory_synthesis_persistence.py` (5 tests): MemorySynthesisResponse validation
- `test_ephemeral_cache.py` (12 tests): Dual cache initialization and reset behavior
- `test_memory_routing.py` (6 tests): add_memory() persistence routing
- `test_cache_combining.py` (8 tests): get_location_memory() cache combining
- `test_supersede_cache_migration.py` (7 tests): supersede_memory() migration cases
- `test_persistence_markers.py` (10 tests): File format with persistence markers
- `test_episode_lifecycle_integration.py` (1 test): Full episode lifecycle validation

**Key Test Patterns:**
```python
# Test ephemeral routing
ephemeral_memory = Memory(..., persistence="ephemeral")
manager.add_memory(10, "Room", ephemeral_memory)
assert manager.get_ephemeral_count(10) == 1
assert manager.get_persistent_count(10) == 0
assert "ephemeral_title" not in (tmp_path / "Memories.md").read_text()

# Test cache migration
old_ephemeral = Memory(..., persistence="ephemeral")
new_permanent = Memory(..., persistence="permanent")
manager.add_memory(15, "Room", old_ephemeral)
manager.supersede_memory(15, "Room", old_ephemeral.title, new_permanent)
assert 15 in manager.memory_cache  # Migrated
assert new_permanent.title in (tmp_path / "Memories.md").read_text()

# Test episode reset
manager.add_memory(10, "Room", ephemeral_memory)
manager.reset_episode()
assert manager.get_ephemeral_count(10) == 0  # Cleared
assert manager.get_persistent_count(10) > 0  # Unchanged
```

### Common Pitfalls for Ephemeral Memories

**Don't:**
- Mark game mechanics as ephemeral (use PERMANENT instead)
- Mark spawn state as ephemeral (use CORE instead)
- Manually clear ephemeral_cache (use reset_episode())
- Write ephemeral memories to file (routing prevents this)
- Expect ephemeral memories to persist across episodes

**Do:**
- Use EPHEMERAL for agent actions: "dropped X", "opened Y", "killed Z"
- Use CORE for spawn state: "room has mailbox", "troll at north"
- Use PERMANENT for procedures: "open window → enter window", "give food to troll fails"
- Trust routing logic (add_memory() handles everything)
- Verify persistence field when creating Memory objects
- Test with episode reset scenarios

### Migration from Old System

**Backward Compatibility:**
- Old memories without persistence markers default to "permanent"
- Parser handles mixed old/new format files correctly
- No migration script needed (gradual transition)

**To Add Ephemeral Support:**
1. Set `persistence` field when creating Memory objects
2. No changes to file reading (parser handles both formats)
3. New memories automatically get persistence markers
4. Episode reset now clears ephemeral memories

### Architecture Benefits

**Performance:**
- Ephemeral operations never touch disk
- No file I/O overhead for temporary state
- Fast episode reset (just clear dict)

**Correctness:**
- False memories automatically prevented
- Agent can't be confused by stale temporary state
- Cross-episode learning works correctly

**Maintainability:**
- Single routing point (add_memory())
- Clear separation of concerns (dual caches)
- File format includes human-readable markers
- Comprehensive test coverage

### Supersession Workflow

When LLM detects contradiction in synthesis:

1. **LLM Response:**
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

2. **SimpleMemoryManager Processing:**
   - Creates new ACTIVE memory
   - Searches memory file for titles in `supersedes_memory_titles`
   - Marks old memories as SUPERSEDED with metadata
   - Atomic file write with backup creation (thread-safe)

3. **Memory File Result:**
```markdown
**[DANGER] Troll attacks after accepting gift** *(Ep01, T13, +0)*
The troll accepts lunch gift but becomes hostile and attacks. Gift strategy fails.

**[NOTE - SUPERSEDED] Troll accepts lunch gift** *(Ep01, T12, +0)*
[Superseded at T13 by "Troll attacks after accepting gift"]
Troll accepts lunch offering graciously. Reaction to gift unclear.
```

## Reasoning History and Strategic Continuity

**Problem Solved**: Without access to previous reasoning, the agent experiences "strategic amnesia" - it must re-derive strategies each turn, breaking continuity for multi-step plans.

**Solution**: The system captures and includes agent reasoning in turn context, enabling strategic continuity across turns.

### Data Storage

In `session/game_state.py`:
```python
action_reasoning_history: List[Dict[str, Any]]
```

Each entry: `{"turn": int, "reasoning": str, "action": str, "timestamp": str}`
- Automatically cleared on episode reset
- History accumulates during episode (unbounded; consider size limit for very long episodes)

### Capture (in orchestrator)

After `agent.get_action_with_reasoning()` returns:
```python
# Extract reasoning from response
reasoning = extract_reasoning_from_tags(response)  # From <think>, <thinking>, or <reflection> tags

# Store via ContextManager
self.context_manager.add_reasoning(
    turn=current_turn,
    reasoning=reasoning,
    action=action_taken
)
```

### Context Inclusion (ContextManager)

`get_recent_reasoning_formatted()` method formats last 3 turns:
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
```

**Configuration:**
- Window size: Last 3 turns included in agent context
- Location: `context_manager.py:505` (`num_turns=3` parameter)
- Positioned after game state, before objectives (strategic context flow)

## Room Description Extraction

**Problem Solved**: The "look" command consumes a precious game turn (~385 turn lantern lifespan in Zork), yet agents need room descriptions for spatial context and decision making.

**Solution**: LLM extractor flags game text as room descriptions (boolean field), orchestrator stores the original text, and ContextManager includes it in agent/critic prompts when recent and location-matched.

### How It Works

**1. Detection (Extractor)**
- LLM sets `is_room_description: true/false` in extraction response
- Boolean flag minimizes hallucination risk and token cost
- Example: Game outputs "West of House\nYou are standing in an open field..." → Extractor flags `is_room_description: true`

**2. Storage (Orchestrator)**
- Stores original game text (no LLM paraphrasing)
- Tracks location ID and turn number
- Stored in `GameState` (episode-scoped, cleared on reset)

**3. Context Integration (ContextManager)**
- Includes description in agent/critic contexts when:
  - Description is for current location (ID match)
  - Description is recent (within configured aging window)
- Prompt format: `ROOM DESCRIPTION (N turns ago):` with age indicator
- Age 0 shows as `ROOM DESCRIPTION:` (no age indicator)

### Configuration

**In `pyproject.toml` or `GameConfiguration`:**
```python
room_description_age_window = 10  # Number of turns before descriptions age out (default: 10)
```

**Recommended values:**
- Default: 10 turns (balances freshness with availability)
- Minimum: 1 turn (very conservative, descriptions age quickly)
- Maximum: 20 turns (liberal, may include stale descriptions)

**Custom configuration example:**
```python
config = GameConfiguration(
    max_turns_per_episode=500,
    room_description_age_window=15  # Custom 15-turn window
)
```

### Why This Matters

**Turn Economy:**
- "look" command costs 1 turn each use
- ~385 total turns available (lantern lifespan constraint)
- Room descriptions provide critical spatial context without turn cost

**Decision Making:**
- Agents need spatial context for navigation and object interaction
- Room descriptions reveal exits, objects, and environmental details
- Context enables better action selection without "looking" repeatedly

**Architecture Benefits:**
- Original game text preserved (no LLM interpretation layer)
- Boolean flag extraction (low hallucination risk)
- Location ID validation (prevents stale/wrong descriptions)
- Configurable aging window (balances freshness vs availability)

### Integration Points

**GameState Fields:**
```python
last_room_description: str = ""
last_room_description_turn: int = 0
last_room_description_location_id: Optional[int] = None
```

**Orchestrator Storage:**
```python
# After extracting info with is_room_description=True
if extracted_info.is_room_description:
    current_location = jericho_interface.get_location_structured()
    location_id = current_location.num if current_location else None

    game_state.last_room_description = clean_response
    game_state.last_room_description_turn = game_state.turn_count
    game_state.last_room_description_location_id = location_id
```

**ContextManager Integration:**
```python
# In get_agent_context() and get_critic_context()
room_desc = self._get_room_description_for_context(location_id)
if room_desc:
    context["room_description"] = room_desc["text"]
    context["room_description_age"] = room_desc["turns_ago"]
```

### Testing

**Test Coverage:** 24 tests in `tests/test_room_description_extraction.py`
- Schema validation (2 tests): ExtractorResponse field
- Detection (3 tests): LLM flagging room descriptions vs action results
- Storage (6 tests): GameState fields, orchestrator logic, logging
- Context integration (10 tests): ContextManager inclusion in agent/critic prompts
- Configuration (3 tests): Default window, custom window, boundary conditions

**Key test patterns:**
```python
# Test default aging window
config = GameConfiguration(max_turns_per_episode=500)
assert config.room_description_age_window == 10

# Test custom aging window
config = GameConfiguration(max_turns_per_episode=500, room_description_age_window=15)
context_manager = ContextManager(logger, config, game_state)
# Description available at 11 turns ago (within 15-turn window)
# Description None at 16 turns ago (exceeds 15-turn window)
```

## Common Pitfalls

**Don't:**
- Store memories at destination location (breaks cross-episode learning)
- Use window size > 10 (excessive context, diminishing returns)
- Ignore status field in Memory creation (critical bug - all memories default to ACTIVE)
- Parse action history manually (use `_format_recent_actions()` helper)
- Initialize managers out of dependency order

**Do:**
- Always capture `location_id_before` and `location_name_before` in orchestrator
- Pass `status=synthesis.status` when creating Memory objects
- Use dedicated formatting helpers for consistency
- Test with multi-step integration scenarios (not just unit tests)
- Monitor TENTATIVE memory accumulation (should be superseded or confirmed)
- Follow the standardized manager lifecycle pattern

## Testing

**Integration Tests:**
- `tests/test_multi_step_window_sequence.py` (6 tests): Window entry prerequisite scenario
- `tests/test_multi_step_delayed_consequence.py` (5 tests): Troll attack delayed consequence + supersession
- `tests/simple_memory/test_movement_memory_location.py` (7 tests): Source location storage verification
- `tests/simple_memory/test_simple_memory_formatting.py` (18 tests): History formatting helpers
- `tests/test_phase5_enhanced_context.py` (27 tests): Context management and reasoning history
