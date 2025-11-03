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
