# Simple Memories.md - Location Memory System Specification

## Problem Statement

### Current State

ZorkGPT currently maintains game knowledge in two markdown files:
- `knowledgebase.md` - General game mechanics and strategies (~7-15KB)
- `persistent_wisdom.md` - Critical learnings from failures (~3-5KB)

**Problems with Current Approach:**

1. **Context Bloat**: Entire knowledgebase (15-20K tokens) is injected into every agent prompt, regardless of relevance
   - Cost: ~$0.045 per turn at Claude Sonnet pricing
   - Latency: Additional processing time for irrelevant information

2. **No Location-Specific Memory**: Agent cannot remember what happened at specific locations
   - Repeatedly tries failed actions at the same location across episodes
   - "Try to take the window" at West of House fails → Episode 2 → tries again
   - No tracking of which items were found where

3. **No Cross-Episode Location Learning**: Each episode treats locations as fresh
   - Episode 1: "Enter dark room, die"
   - Episode 2: "Enter same dark room, die again"
   - No memory that "Location 23 requires lantern"

4. **Generic Knowledge Only**: Current system captures universal patterns ("always need light in dark areas") but not location-specific experiences ("the troll at Location 42 blocks the bridge")

### Core Problem

**The agent has no spatial memory.** It forgets what it learned about specific locations when episodes reset.

## Goal

### Primary Objective

Implement location-specific memory that allows the agent to:

1. **Remember location experiences** across episode resets
2. **Avoid repeated failed actions** at the same location
3. **Recall successful strategies** when returning to a location
4. **Reduce total turns** by not re-exploring known dead ends

### Success Metrics

1. **Repeated Action Rate**: Reduce from ~20-30% to <5%
   - Measure: % of actions identical to previous failures at same location

2. **Token Efficiency**: Reduce context from 15-20K to 200-500 tokens per turn
   - Savings: ~98% reduction = $0.04 per turn saved

3. **Cross-Episode Improvement**: Measurable reduction in turns to achieve milestones
   - Example: Episode 1: 50 turns to get lamp → Episode 2: 30 turns to get lamp

4. **Location Coverage**: Memory captured for 80%+ of visited locations by Episode 3

## Solution Overview

### Core Concept

A single markdown file (`Memories.md`) that maintains location-specific memories organized by Jericho's stable integer location IDs. The file is:
- **Human-readable**: Plain markdown, can be manually edited/reviewed
- **Version-controlled**: Tracked in git, can see memory evolution
- **Persistent**: Survives episode resets, accumulates cross-episode
- **Injected selectively**: Only current location's memories added to agent prompt

### Key Principles

#### 1. Use Jericho Location IDs

**CRITICAL**: Must use `location.num` (integer) as the canonical location identifier, NOT room names (strings).

**Why:**
- Jericho's Z-machine provides stable integer IDs (1, 2, 3, ...)
- IDs are guaranteed unique and never fragment
- Room names can duplicate ("Forest Path" appears multiple times)
- Phase 3 of Jericho integration deleted 512 lines of consolidation logic by using IDs

**Architecture Alignment:**
```python
# Current working architecture (Phase 1-7)
location = jericho_interface.get_location_structured()
room_id = location.num       # INTEGER - primary key
room_name = location.name    # STRING - display only

# Memory system MUST follow this pattern
memories[room_id] = {...}  # Keyed by integer ID
```

#### 2. LLM-First Memory Synthesis

**CRITICAL**: Memory decisions must be made by LLM reasoning, NOT hardcoded rules.

**Why:**
- Aligns with ZorkGPT's core principle: "All game reasoning must originate from LLMs"
- LLM can semantically understand action outcomes and synthesize meaningful insights
- Avoids brittle string matching and rule-based heuristics
- Enables natural language memories more useful to agent than raw action logs

**Architecture Approach:**
- **Z-machine Triggers**: Use ground truth data (score changes, location changes) to determine WHEN to invoke LLM
- **LLM Synthesis**: Use `info_ext_model` to determine WHAT to remember and HOW to categorize it
- **Hybrid System**: Fast deterministic triggers + intelligent LLM reasoning = efficient and principled

## Detailed Design

### File Structure

**File Location**: `{config.zork_game_workdir}/Memories.md`

**Format**:
```markdown
# Location Memories

## Location 15: West of House
**Visits:** 3 | **Episodes:** 1, 2, 3

### Memories

**[SUCCESS] Open and enter window** *(Ep1, T23-24, +0)*
Window can be opened with effort and used as alternative entrance to house. Must squeeze through opening.

**[FAILURE] Take or break window** *(Ep1, T25-26)*
Window is part of house structure - cannot be taken, moved, or broken. Violence not effective.

**[DISCOVERY] Mailbox location** *(Ep1, T20, +0)*
Small mailbox located here contains advertising leaflet. Likely tutorial document.

---

## Location 23: Living Room
**Visits:** 5 | **Episodes:** 1, 2, 3, 4

### Memories

**[SUCCESS] Acquire brass lantern** *(Ep1, T45, +5)*
Brass lantern is takeable and provides light source. CRITICAL item for dark areas - always take before exploring.

**[SUCCESS] Light lantern** *(Ep1, T46, +0)*
Lantern can be lit with simple command. Enables safe navigation of dark rooms.

**[FAILURE] Take sword** *(Ep1, T47)*
Ornamental sword is securely mounted and cannot be taken directly. Likely requires puzzle solution.

**[NOTE] Navigation options** *(Ep1, T50, +0)*
West exit leads to Kitchen. Room serves as central hub with multiple exits.

---
```

**Format Rationale:**
- **Location ID in header**: Canonical identifier for parsing
- **Room name in header**: Human-readable context
- **Visit count + episodes**: Track exploration frequency
- **Memory format**: **[CATEGORY] Title** *(metadata)* followed by synthesized learning
- **Categories**: SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE (LLM-determined)
- **Metadata**: Episode, Turn(s), Score change
- **Synthesis**: Natural language insights generated by LLM, not raw action logs

### Memory Storage Strategy

#### LLM-First Approach

**Principle**: Memory decisions are made by LLM reasoning, not hardcoded rules. The system uses Z-machine ground truth data as inputs to LLM synthesis.

#### Two-Phase Storage Process

**Phase 1: Significance Detection (Z-machine triggers)**

Fast checks using Z-machine ground truth determine if an action warrants LLM memory synthesis:

**Triggers** (any trigger fires → invoke LLM):
- Score changed (positive or negative)
- Location changed (movement occurred)
- Inventory changed (item acquired or dropped)
- Death occurred (critical danger event)
- First visit to location (initialization)
- Substantial response (>100 characters, may contain important information)

**Rationale**: These Z-machine signals indicate potential game state changes worth capturing, but the LLM decides what the memory actually means.

**Phase 2: LLM Memory Synthesis**

When triggered, the LLM analyzes the action outcome and decides:
1. **Should this be remembered?** (yes/no with reasoning)
2. **What category?** (SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE)
3. **What's the key insight?** (synthesized natural language memory, not raw logs)
4. **Does this supersede existing memories?** (e.g., new info contradicts old learning)

**LLM receives as input**:
- Current location ID and name
- Action taken and game response
- Z-machine context (score delta, location change, inventory change, death status)
- Existing memories for this location (to avoid duplicates and detect contradictions)

**LLM produces structured output**:
```json
{
  "should_remember": true,
  "category": "FAILURE",
  "memory_title": "Take or break window",
  "memory_text": "Window is part of house structure - cannot be taken, moved, or broken.",
  "reasoning": "Invalid actions worth remembering to prevent repetition"
}
```

#### Memory Categories (LLM-Determined)

Categories guide the LLM's classification but are not hardcoded rules:

- **SUCCESS**: Action achieved intended goal (puzzle solved, item acquired, progress made)
- **FAILURE**: Action was invalid or unsuccessful (prevents repetition of failed attempts)
- **DISCOVERY**: New information learned (object properties, room features, game mechanics)
- **DANGER**: Harmful action or location hazard (death events, score penalties, threats)
- **NOTE**: Strategic insights or context (navigation patterns, puzzle hints, item relationships)

#### When to Write to File

**Strategy: Immediate Write After Synthesis**

**Rationale:**
- Small markdown file writes are fast (<10ms)
- Immediate persistence enables real-time debugging
- No risk of memory loss if system crashes mid-episode
- Simpler architecture (no complex buffering logic)
- Agent sees memory immediately on next visit to location

**Process:**
1. **Z-machine trigger fires** → invoke LLM synthesis
2. **LLM decides to remember** → immediately append to Memories.md
3. **File cache updated** → in-memory representation stays current
4. **Next location visit** → agent sees the new memory

**File Write Implementation:**
- Parse existing Memories.md into memory cache (only on manager initialization)
- On synthesis decision to store: update cache + append to file
- Use file locking to prevent corruption if multiple processes (unlikely but safe)
- Log file writes for monitoring

**Performance:**
- File append operation: ~5-10ms (negligible compared to LLM synthesis latency of 200-400ms)
- Cache remains in memory for fast reads
- No episode-end flush needed

### Memory Creation Mechanism

**Type**: LLM-driven synthesis using `info_ext_model`

**Component**: `SimpleMemoryManager` (new manager following BaseManager pattern)

**Architecture**: Hybrid trigger system with LLM synthesis

**Process Flow**:

1. **Capture Z-machine Context**: After each action, extract ground truth data (score, location, inventory, death status)

2. **Significance Check**: Evaluate Z-machine triggers to determine if LLM synthesis is warranted

3. **LLM Synthesis**: If triggered, call `info_ext_model` with structured prompt containing:
   - Action and response
   - Z-machine context (deltas and state changes)
   - Existing memories for location from cache (to prevent duplicates)

4. **Store Decision**: If LLM determines memory is worth keeping, immediately write to file and update cache

5. **Duplicate Prevention**: LLM sees existing memories from cache and can determine if new action adds information or is redundant

**Example Flow**:
```
Turn 5: "take window" → response "can't do that"
  → Z-machine: no score change, no location change, no inventory change, >100 char response
  → Trigger: substantial response → invoke LLM
  → LLM synthesizes: [FAILURE] "Window is part of structure, cannot be taken"
  → Immediately written to Memories.md
  → Cache updated: memory_cache[15].append(new_memory)

Turn 10: "take window" → same response
  → Trigger: substantial response → invoke LLM
  → LLM sees existing memory about window from turn 5 (in cache)
  → LLM decides: should_remember = false (duplicate of existing memory)
  → Nothing written

Turn 15: Agent queries location memory
  → Retrieves from cache: memory_cache[15]
  → Returns failure from turn 5
  → Agent sees "window cannot be taken" and (hopefully) doesn't retry
```

**LLM Model Configuration**:
- Uses `info_ext_model` (already configured for structured extraction)
- Structured JSON output for reliable parsing
- Includes `reasoning` field for debugging LLM decisions

### Memory Retrieval Strategy

**Type**: Location-based file parsing + text formatting

**Retrieval Context**: When agent is about to act at a location

**Process**:

1. **Query**: Get current location ID from `game_state.current_room_id`
2. **Retrieve from Cache**: Lookup location in `memory_cache[location_id]`
3. **Format**: Convert synthesized memories to concise text block
4. **Inject**: Add to agent context under "Location Memory" section

**Cache Consistency**:
- Cache contains all memories (initialized from file on startup)
- Immediately updated when new memory synthesized and written
- Agent always sees most current state including memories from earlier in same episode
- **Example**: Turn 10 "take window" fails → immediately written to file and cache → Turn 20 agent sees this failure → doesn't retry

**Token Budget**: 200-300 tokens per location (vs 15-20K for full knowledgebase)

**Example Retrieved Context**:
```
Location Memory for Living Room (Location 23):

You've been here 5 times across 4 episodes.

[SUCCESS] Acquire brass lantern (Ep1, T45, +5)
Brass lantern is takeable and provides light source. CRITICAL item for dark areas - always take before exploring.

[SUCCESS] Light lantern (Ep1, T46, +0)
Lantern can be lit with simple command. Enables safe navigation of dark rooms.

[FAILURE] Take sword (Ep1, T47)
Ornamental sword is securely mounted and cannot be taken directly. Likely requires puzzle solution.

[NOTE] Navigation options (Ep1, T50, +0)
West exit leads to Kitchen. Room serves as central hub with multiple exits.
```

**Implementation**:
```python
def get_location_memory(self, location_id: int) -> str:
    """
    Retrieve formatted memory text for current location.

    Returns:
        Formatted string (200-300 tokens) from in-memory cache.
        Cache contains all memories including those from current episode.

        Returns "First visit - no prior experiences" if new location.
    """
    # Retrieve from cache (already contains all memories including current episode)
    memories = self.memory_cache.get(location_id, [])

    if not memories:
        return "First visit - no prior experiences"

    # Format for agent context
    return self._format_memory_context(memories, location_id)
```

### Integration Points

#### 1. New Manager: SimpleMemoryManager

**File**: `managers/simple_memory_manager.py`

**Responsibilities**:
- Parse and maintain Memories.md file with in-memory cache
- Invoke LLM synthesis for significant action outcomes (using `info_ext_model`)
- Write memories immediately to file after synthesis
- Retrieve location-specific context from cache
- Provide status for monitoring

**Interface** (follows BaseManager pattern):
```python
class SimpleMemoryManager(BaseManager):
    def __init__(self, logger, config, game_state, llm_client):
        """Initialize with reference to Memories.md file and LLM client."""

    def reset_episode(self):
        """Clear episode-specific state (file cache persists across resets)."""

    def record_action_outcome(self, location_id, location_name, action,
                             response, z_machine_context):
        """
        Evaluate action outcome and invoke LLM synthesis if significant.
        Writes immediately to Memories.md if LLM decides to store.

        z_machine_context contains: score_before, score_after,
        location_before, location_after, inventory_before,
        inventory_after, died
        """

    def get_location_memory(self, location_id: int) -> str:
        """Get formatted memory text for injection into agent context."""

    def get_status(self) -> Dict[str, Any]:
        """Return manager status for monitoring."""
```

**State Management**:
- `self.memory_file`: Path to Memories.md
- `self.llm_client`: Reference to LLM client for synthesis (uses `info_ext_model`)
- `self.memory_cache`: Dict[int, List[Memory]] (in-memory cache of entire Memories.md for fast reads)
- File writes happen immediately after synthesis, cache updated atomically

#### 2. Orchestrator Integration

**File**: `orchestration/zork_orchestrator_v2.py`

**Changes Required**:

**Initialization** (~5 lines):
```python
# In __init__
self.simple_memory = SimpleMemoryManager(
    logger=self.logger,
    config=self.config,
    game_state=self.game_state,
    llm_client=self.llm_client
)
```

**Turn Processing** (~15 lines):
```python
# In _process_turn, capture Z-machine context before and after action
z_context = {
    'score_before': score_before,
    'score_after': score_after,
    'location_before': location_id_before,
    'location_after': location_id_after,
    'inventory_before': inventory_before,
    'inventory_after': inventory_after,
    'died': self.game_state.is_dead
}

# Memory will be synthesized and written immediately if significant
self.simple_memory.record_action_outcome(
    location_id=self.game_state.current_room_id,
    location_name=self.game_state.current_room_name_for_map,
    action=action_to_take,
    response=game_response,
    z_machine_context=z_context
)
```

**Total Impact**: ~15 lines added to orchestrator (no episode-end flush needed)

#### 3. ContextManager Integration

**File**: `managers/context_manager.py`

**Changes Required**:

**Initialization** (~2 lines):
```python
# In __init__
self.simple_memory = None  # Injected by orchestrator
```

**Context Assembly** (~8 lines):
```python
# In get_agent_context, add new context section
location_memory = ""
if self.simple_memory:
    location_memory = self.simple_memory.get_location_memory(
        self.game_state.current_room_id
    )

context_parts.append(f"\n### Location Memory\n{location_memory}")
```

**Total Impact**: ~10 lines added to context manager

#### 4. Configuration

**File**: `pyproject.toml` or `GameConfiguration`

**New Settings**:
```toml
[tool.zorkgpt.simple_memory]
enabled = true
memory_file = "Memories.md"
max_successful_actions_shown = 5
max_failed_actions_shown = 5
flush_on_death = true
```

**Total Impact**: ~10 lines of configuration

### File Format Specification

#### Parsing Strategy

**Section Identification**:
- Regex: `^## Location (\d+): (.+)$` captures location ID and name
- Memory subsection: Lines starting with `### Memories`
- Visit metadata: `**Visits:** (\d+) \| **Episodes:** (.+)$`

**Memory Extraction**:
Each memory follows pattern:
```
**[CATEGORY] Title** *(Ep\d+, T\d+(-\d+)?, [+-]?\d+)*
Memory text (1-2 sentences of synthesized insight)
```

- **Category**: SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE (extracted from `[CATEGORY]`)
- **Title**: Short description (extracted from text between `]` and `*`)
- **Metadata**: Episode, turn(s), score change (extracted from parentheses)
- **Text**: Natural language synthesis (lines following header until next memory or section break)

**Writing Strategy**:
1. **Initialization**: Parse existing Memories.md into in-memory cache on manager startup
2. **Per-Memory Write**: When LLM synthesizes new memory:
   - Append formatted memory entry to appropriate location section in file
   - Update in-memory cache atomically
   - Use file locking to prevent concurrent write issues
3. **File Format**: Maintain consistent markdown structure with locations sorted by ID

**Memory Deduplication**:
LLM handles semantic deduplication during synthesis by seeing existing memories from cache. Optional text-based deduplication on file write as backup if LLM occasionally fails to detect redundancy.

#### File Growth Management

**Realistic Scale Estimate**:
- Zork has ~110 locations total
- Average 5-10 unique memories per location (LLM-synthesized, not raw actions)
- ~500-800 bytes per synthesized memory (title + 1-2 sentence text)
- ~5-8KB per location = ~550-880KB initial coverage
- LLM semantic deduplication prevents redundant storage

**Actual Growth Rate**: +3-7KB per episode (LLM filters semantic duplicates)

**At 100 Episodes**: ~100-200KB file size (highly manageable, faster than rule-based)

**Management Strategy**:
1. **LLM Semantic Deduplication** - Primary mechanism
   - LLM sees existing memories before deciding to store
   - Recognizes semantic duplicates ("take window" vs "get window" are same failure)
   - More effective than string matching

2. **File Writer Text Deduplication** - Backup mechanism
   - Compares memory titles and text for exact duplicates
   - Prevents storage if LLM occasionally misses redundancy
   - Fast text comparison on merged data before write

3. **Optional: Memory Consolidation** - Future enhancement
   - Could invoke LLM periodically to consolidate related memories
   - Example: Merge 3 separate "take X" failures into single "immovable objects" memory
   - Not implemented in initial version

**No Pruning of Old Episodes**: All memories valuable for cross-episode learning. LLM-synthesized memories are more concise than raw action logs, keeping file size manageable.

## Implementation Phases

### Phase 1: Core Infrastructure (Day 1, 4-6 hours)

**Deliverables**:
1. `SimpleMemoryManager` class skeleton with LLM client integration
2. Z-machine trigger detection logic
3. Memories.md file parsing and in-memory caching (new format)
4. Immediate file write after synthesis with cache update
5. File locking mechanism for safe concurrent writes

**Testing**:
- Unit tests for parsing new format
- Mock LLM responses for synthesis testing
- Test immediate write persistence (kill process mid-episode, verify memories saved)
- Manual: Run 1 episode, verify Memories.md updated in real-time with LLM-synthesized content

### Phase 2: LLM Synthesis Integration (Day 1-2, 3-5 hours)

**Deliverables**:
1. LLM prompt design for memory synthesis
2. Structured JSON output parsing
3. Integration with `info_ext_model`
4. Duplicate detection via LLM (sees existing memories)

**Testing**:
- Test LLM synthesis with various action outcomes
- Verify duplicate detection works (same action tried twice)
- Verify reasoning field captured for debugging

### Phase 3: System Integration (Day 2, 2-4 hours)

**Deliverables**:
1. Orchestrator integration (Z-machine context capture)
2. ContextManager integration (inject memories)
3. Configuration setup
4. Monitoring and status reporting

**Testing**:
- Integration test: Run 2 episodes, verify cross-episode memory
- Verify agent receives synthesized memories in prompt
- Verify LLM synthesis happens only when triggered

### Phase 4: Validation (Day 2, 1-2 hours)

**Deliverables**:
1. Metrics collection (repeated actions, tokens, turns, LLM calls)
2. Baseline comparison
3. Analysis of LLM synthesis quality
4. Documentation

**Testing**:
- Full playthrough (50+ turns per episode, 10 episodes)
- Compare metrics vs baseline without memory system
- Review synthesized memories for quality and relevance
- Measure LLM synthesis cost vs savings from reduced repetition

**Total Timeline**: 2-3 days (slightly longer due to LLM integration complexity)

## Success Criteria

### Must-Have (Phase 1-2)

1. ✅ Memories.md file created and populated
2. ✅ Location sections use integer IDs as keys
3. ✅ Cross-episode persistence works (Episode 2 sees Episode 1 memories)
4. ✅ Agent receives location memory in context
5. ✅ No crashes or data corruption

### Should-Have (Phase 3)

1. ✅ Repeated action rate < 10% (vs ~25% baseline)
2. ✅ Token usage < 2K per turn (vs 15-20K baseline)
3. ✅ 80%+ of visited locations have memory entries
4. ✅ File parsing < 10ms (negligible overhead)

### Nice-to-Have (Phase 4)

1. ✅ Repeated action rate < 5%
2. ✅ Token usage < 500 tokens per turn (200-300 for memory)
3. ✅ Measurable turn reduction in Episode 2+ vs Episode 1
4. ✅ Human-readable file can be manually edited for quality control

## Risks and Mitigations

### Risk 1: File Parsing Brittleness

**Risk**: Markdown parsing breaks if format changes or gets corrupted

**Mitigation**:
- Strict format validation on write
- Try/catch with graceful degradation (skip corrupted sections)
- Backup file before each write (Memories.md.backup)
- Version control tracks all changes

### Risk 2: Token Budget Overflow

**Risk**: Large memory sections exceed 500 token budget

**Mitigation**:
- Hard limit: Show only last 5 successes + 5 failures
- Truncate response text to 100 characters
- Monitor token usage, log warnings if > 300 tokens
- Provide config to reduce limits if needed

### Risk 3: Redundant Entries

**Risk**: Same action stored repeatedly, file bloats

**Mitigation**:
- **LLM Semantic Deduplication** - Primary defense
  - LLM receives existing memories in synthesis prompt
  - Can recognize semantic duplicates ("take window" vs "get window")
  - Decides `should_remember = false` if information already captured
  - More sophisticated than string matching
- **File Writer Backup** - Secondary defense
  - Text comparison on memory titles before writing
  - Prevents accidental duplicates if LLM fails
  - Fast and deterministic
- **Result**: Each unique insight stored once per location
- **File growth**: ~100-200KB total for 100 episodes (manageable, smaller than raw logs)

### Risk 4: Performance Degradation

**Risk**: LLM synthesis calls and file I/O slow down turn processing

**Mitigation**:
- **LLM Calls**: Only invoke when Z-machine triggers fire (not every turn)
  - ~30-40% of actions trigger synthesis (significant events only)
  - Use fast `info_ext_model` for synthesis
  - Structured output with minimal tokens (~200-300 per call)
  - Expected latency: 200-400ms per synthesis (acceptable)
- **File I/O**:
  - Cache parsed file in memory (only re-parse on manager initialization)
  - Immediate writes after synthesis (~5-10ms, negligible)
  - File locking prevents corruption
  - Benchmark: File write must be < 10ms, parsing < 10ms on init
- **Net Impact**: Slight per-turn increase (~200ms) offset by reduced agent context and fewer repeated explorations

### Risk 5: LLM Synthesis Quality

**Risk**: LLM produces low-quality or incorrect memory synthesis

**Mitigation**:
- **Structured Output**: JSON schema forces consistent format
- **Reasoning Field**: Captures LLM's decision-making for debugging
- **Z-machine Ground Truth**: Provides factual context (score changes, location changes)
- **Existing Memory Context**: LLM sees prior memories to maintain consistency
- **Monitoring**: Log all synthesis decisions for quality review
- **Fallback**: If LLM synthesis fails or returns invalid JSON, log error but don't crash (skip memory)
- **Human Review**: Memories.md is human-readable and can be manually corrected

### Risk 6: Location ID Mismatches

**Risk**: Using wrong location ID leads to memory fragmentation

**Mitigation**:
- Assert location_id is integer, not string
- Validate against MapManager's current room ID
- Log warnings if location_id not in MapGraph
- Unit tests verify ID consistency