# Ephemeral Memory System Specification

## Problem Statement

The current memory system suffers from **false persistence** of agent-caused state changes across episode boundaries:

**Scenario:**
```
Episode 1:
  Turn 2:  First visit to Living Room → "Sword here" (room description)
  Turn 10: Agent takes sword from Living Room
  Turn 45: Agent drops sword in Kitchen
  Turn 50: Memory created: "Sword in Kitchen"

Episode 2:
  Sword respawns in Living Room (game reset)
  Turn 5:  Agent goes to Kitchen expecting sword (from Episode 1 memory)
  Result:  Sword not there → WASTED ACTION, FALSE EXPECTATION
```

**Root Cause:** The system cannot distinguish between:
- **Permanent game state**: "Sword spawns in Living Room" (resets to this every episode)
- **Ephemeral agent state**: "Dropped sword in Kitchen" (only true in current episode)

## Solution: Three-Tier Memory Persistence

### Classification Types

#### 1. CORE (Spawn State)
**Definition:** Items, objects, and permanent fixtures discovered in room descriptions on **first visit only**.

**Criteria:**
- Can ONLY be created when `first_visit=true` (Z-machine flag)
- Describes items present in initial room description
- Represents spawn state that resets each episode

**Examples:**
```markdown
**[DISCOVERY - CORE] Sword spawns here** *(Ep1, T2, +0)*
Sword found in Living Room description on first visit.

**[DISCOVERY - CORE] Glass bottle holds water in kitchen** *(Ep1, T15, +0)*
Bottle discovered on kitchen counter during first visit.

**[DISCOVERY - CORE] Brass lantern in trophy case** *(Ep1, T29, +0)*
Lantern visible in trophy case on first entry to room.
```

**Persistence:** Written to `Memories.md` → Persists across all episodes

---

#### 2. PERMANENT (Game Mechanics)
**Definition:** Game rules, puzzle solutions, danger patterns, and mechanics learned through play.

**Criteria:**
- Can be created on first visit OR return visits
- Describes how the game works, not what state it's in
- Knowledge that remains true across episode resets

**Examples:**
```markdown
**[DANGER - PERMANENT] Troll attacks on sight** *(Ep1, T45, +0)*
Troll immediately hostile when entering Troll Room. Combat unavoidable.

**[DISCOVERY - PERMANENT] Window entry requires opening** *(Ep1, T20, +10)*
Window must be opened before entering. Sequence: examine → open → enter.

**[FAILURE - PERMANENT] Tree cannot be climbed** *(Ep1, T8, +0)*
Attempting to climb tree fails consistently. Not a valid action.

**[DISCOVERY - PERMANENT] Kitchen entry grants 10 points** *(Ep1, T21, +10)*
First entry to Kitchen awards 10 points. One-time score trigger.
```

**Persistence:** Written to `Memories.md` → Persists across all episodes

---

#### 3. EPHEMERAL (Agent State)
**Definition:** Agent-caused state changes that reset on episode boundaries.

**Criteria:**
- Can be created on ANY visit (first or return)
- Describes what the agent DID, not what the game IS
- State that becomes invalid after episode reset
- Classification based on ACTION TYPE, not visit timing

**Examples:**
```markdown
**[NOTE - EPHEMERAL] Dropped sword here** *(Ep1, T45, +0)*
Agent dropped sword in Kitchen for inventory management at Turn 45.

**[NOTE - EPHEMERAL] Placed nest in sack** *(Ep1, T27, +0)*
Agent stored fragile nest in brown sack for safe transport.

**[NOTE - EPHEMERAL] Opened window from outside** *(Ep1, T19, +0)*
Window opened by agent at Behind House. Allows entry to Kitchen.

**[NOTE - EPHEMERAL] Left lantern on table** *(Ep1, T50, +0)*
Brass lantern dropped on kitchen table to free inventory slot.
```

**Persistence:** **NEVER written to disk** → Stored in `ephemeral_cache` only → Auto-cleared on `reset_episode()`

---

### Memory Type Relationships

**IMPORTANT:** Memory types are **complementary, not mutually exclusive**. Multiple memory types can coexist for the same object/location:

**Example:**
```markdown
## Location 8: Kitchen

**[DISCOVERY - CORE] Glass bottle here** *(Ep1, T15, +0)*
Bottle discovered on kitchen counter during first visit.

**[DISCOVERY - PERMANENT] Bottle can be filled from stream** *(Ep1, T22, +0)*
Bottle can hold water when filled at stream location.

**[NOTE - EPHEMERAL] Left bottle on table** *(Ep1, T45, +0)*
Agent placed empty bottle on kitchen table for later retrieval.
```

**Relationship:**
- **CORE** describes **spawn state**: Where object initially appears (resets each episode)
- **PERMANENT** describes **mechanics**: How object behaves (stays true across episodes)
- **EPHEMERAL** describes **agent state**: What agent did with object (clears each episode)

All three can exist simultaneously - they describe different aspects:
- CORE = "Where is it at start?"
- PERMANENT = "How does it work?"
- EPHEMERAL = "What did I do with it?"

---

## Implementation Architecture

### Dual Cache System

```python
class SimpleMemoryManager:
    def __init__(self, logger, config, game_state, llm_client=None):
        # PERSISTENT: Loaded from Memories.md (core + permanent)
        self.memory_cache: Dict[int, List[Memory]] = {}

        # EPHEMERAL: In-memory only, cleared on episode reset
        self.ephemeral_cache: Dict[int, List[Memory]] = {}

        self._load_memories_from_file()
```

### Data Model Changes

#### Memory Dataclass
```python
@dataclass
class Memory:
    category: str              # SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE
    title: str
    episode: int
    turns: str
    score_change: Optional[int]
    text: str
    persistence: str           # "core" | "permanent" | "ephemeral" - REQUIRED, no default
    status: MemoryStatusType = MemoryStatus.ACTIVE

    # Existing supersession fields
    superseded_by: Optional[str] = None
    superseded_at_turn: Optional[int] = None
    invalidation_reason: Optional[str] = None
```

**IMPORTANT:** `persistence` is required with no default value to force explicit classification and prevent silent bugs during migration.

**Turn Number Usage:**
- `turns` field in metadata: **Mandatory** - appears in formatted header as `(Ep1, T45, +0)`
- Turn mention in `text` field: **Optional** - useful for EPHEMERAL memories where timing adds context
  - Example EPHEMERAL: "Agent dropped sword in Kitchen at Turn 45" (adds temporal narrative)
  - Example CORE: "Sword found in room description" (no turn mention needed - spawn state)
  - Example PERMANENT: "Troll attacks on sight" (no turn mention - always-true mechanic)

#### MemorySynthesisResponse Schema
```python
class MemorySynthesisResponse(BaseModel):
    should_remember: bool
    category: Optional[str] = None
    memory_title: Optional[str] = None
    memory_text: Optional[str] = None
    status: MemoryStatusType = Field(default=MemoryStatus.ACTIVE)

    # NEW: Persistence classification (Optional - only set when should_remember=True)
    persistence: Optional[str] = None  # "core" | "permanent" | "ephemeral"

    # Existing fields
    supersedes_memory_titles: Set[str] = Field(default_factory=set)
    invalidate_memory_titles: Set[str] = Field(default_factory=set)
    invalidation_reason: Optional[str] = None
    reasoning: str = ""

    @validator('persistence')
    def validate_persistence(cls, v, values):
        """Validate persistence field based on context."""
        if not values.get('should_remember'):
            return None  # Irrelevant if not creating memory
        if v is None and values.get('should_remember'):
            raise ValueError("persistence required when should_remember=True")
        if v not in ["core", "permanent", "ephemeral"]:
            raise ValueError(f"persistence must be core|permanent|ephemeral, got: {v}")
        return v
```

---

## Conditional Prompt Injection

### Orchestrator Context

```python
# In zork_orchestrator_v2.py
z_machine_context = {
    'first_visit': location_id not in self.game_state.visited_locations,
    'score_delta': score_after - score_before,
    'location_changed': location_before != location_after,
    'inventory_changed': inventory_before != inventory_after,
    'died': self.jericho_interface.is_dead(),
    'response_length': len(response),
}

self.simple_memory.record_action_outcome(
    location_id=location_id,  # SOURCE location where action occurred
    location_name=location_name,
    action=action,
    response=response,
    z_machine_context=z_machine_context
)

# IMPORTANT: After processing this action, add location to visited_locations
# This ensures first_visit=false for all subsequent actions at this location
if location_id not in self.game_state.visited_locations:
    self.game_state.visited_locations.add(location_id)
```

**`location_id` Semantics (SOURCE Location):**

- Represents the location WHERE THE ACTION OCCURRED, not where synthesis happens
- Memory is stored at the SOURCE location for cross-episode learning
- Example: Agent at Kitchen executes "drop sword" → location_id=8 (Kitchen)
- Even if synthesis happens over multiple turns, location_id remains the SOURCE
- This ensures memories are found when agent returns to the same location

```

**`first_visit` Semantics:**

- **True:** This is the first action/observation at this location (location not in visited_locations)
- **False:** Location has been visited before (location already in visited_locations)
- **Timing:** Flag becomes false after the first turn at a location completes
- **LOOK command:** If agent executes "look" after already being at a location, first_visit=false
- **Same-turn observations:** Within a single turn, multiple memory synthesis calls would see the same first_visit value (flag updates after turn completes)

**Example Timeline:**
```python
Turn 5: Enter Kitchen (location_id=8)
        first_visit = (8 not in visited_locations) = True
        Process action, synthesize memories
        Add 8 to visited_locations

Turn 6: Still in Kitchen, execute "take bottle"
        first_visit = (8 not in visited_locations) = False  # Already visited

Turn 7: Still in Kitchen, execute "look"
        first_visit = False  # Still a return visit
        Room shows new items → Cannot create CORE memories (first_visit=false)
```

**Implication for CORE Memories:**

CORE memories can ONLY be created during the first turn at a location. If the room description doesn't mention an item on first entry, but agent uses LOOK later and sees it, that observation must be PERMANENT (game mechanic: "Item visible on closer inspection") not CORE (spawn state).
```

### Synthesis Prompt Logic (CORRECTED - Unified Approach)

**CRITICAL CHANGE:** Persistence classification is now based on ACTION TYPE, not visit timing. All three types are always explained, with context signals guiding the LLM.

```python
def _synthesize_memory(
    self,
    location_id: int,
    location_name: str,
    action: str,
    response: str,
    z_machine_context: Dict
) -> Optional[MemorySynthesisResponse]:
    """Synthesize memory with context-aware persistence instructions."""

    # ... existing deduplication/contradiction sections ...

    # CONTEXT ANALYSIS: Build hints from z_machine_context
    first_visit = z_machine_context.get('first_visit', False)
    inventory_changed = z_machine_context.get('inventory_changed', False)
    location_changed = z_machine_context.get('location_changed', False)
    score_delta = z_machine_context.get('score_delta', 0)
    response_length = z_machine_context.get('response_length', 0)

    context_hints = []
    if first_visit and response_length > 100:
        context_hints.append("• Long response on first visit suggests room description (likely CORE)")
    if inventory_changed and not location_changed:
        context_hints.append("• Inventory changed without movement suggests item manipulation (likely EPHEMERAL)")
    if score_delta > 0:
        context_hints.append("• Score increase suggests game mechanic discovered (likely PERMANENT)")
    if any(phrase in response.lower() for phrase in ["you can't", "that doesn't", "impossible"]):
        context_hints.append("• Failed action suggests game constraint learned (likely PERMANENT)")
    if any(verb in action.lower() for verb in ['drop', 'put', 'place', 'insert', 'leave']):
        context_hints.append("• Placement action detected (likely EPHEMERAL)")

    context_section = "\n".join(context_hints) if context_hints else "• No strong signals from context"

    # UNIFIED PERSISTENCE SECTION: Always explain all three types
    persistence_section = f"""
MEMORY PERSISTENCE CLASSIFICATION:
═══════════════════════════════════════════════════════════════
Choose based on WHAT HAPPENED (action type), not WHEN (visit timing).

**CORE** - Spawn state from room description (FIRST VISIT ONLY):
  Definition: Items/objects/fixtures in room description on first visit
  When to use:
    ✓ ONLY on first visit to location (first_visit=true)
    ✓ ONLY for passive observations from room text
    ✓ NOT for agent actions or discoveries

  Examples:
    ✓ "Sword here" (from "Living Room. There is a sword here.") → CORE
    ✓ "Brass lantern in trophy case" (from room description) → CORE
    ✓ "Mailbox visible" (from "West of House" description) → CORE
    ✗ "Dropped sword here" (agent action, not room text) → NOT CORE
    ✗ "Sword was here" (return visit) → NOT CORE

**PERMANENT** - Game mechanics and reusable knowledge:
  Definition: How the game works; knowledge true across episodes
  When to use:
    ✓ ANY visit (first or return)
    ✓ Learning rules, mechanics, dangers, constraints
    ✓ Knowledge that stays true after episode reset

  Examples:
    ✓ "Troll attacks on sight" (danger behavior) → PERMANENT
    ✓ "Window can be opened" (game mechanic) → PERMANENT
    ✓ "Taking egg grants 5 points" (scoring rule) → PERMANENT
    ✓ "Door nailed shut" (permanent obstacle) → PERMANENT
    ✓ "Cannot climb tree from here" (constraint) → PERMANENT

**EPHEMERAL** - Agent-caused state changes:
  Definition: What agent DID that changes state temporarily
  When to use:
    ✓ ANY visit (first or return) ← CRITICAL FIX
    ✓ Agent performed action: drop, place, open, take, move
    ✓ State change that resets on episode boundary

  Examples:
    ✓ "Dropped sword here" (agent action) → EPHEMERAL
    ✓ "Placed nest in sack" (agent organization) → EPHEMERAL
    ✓ "Opened window from outside" (agent state change) → EPHEMERAL
    ✓ "Left lantern on table" (inventory management) → EPHEMERAL

CONTEXT SIGNALS FOR THIS ACTION:
{context_section}

DECISION CRITERIA:
1. CORE: Room description observation on FIRST VISIT only
2. EPHEMERAL: Agent action that changes state (ANY VISIT)
3. PERMANENT: Game mechanic/rule learned (ANY VISIT)

If agent DOES something → likely EPHEMERAL
If agent LEARNS something → likely PERMANENT
If agent SEES something in room description (first visit) → likely CORE

Current visit status: {"FIRST VISIT" if first_visit else "RETURN VISIT"}
⚠️  CORE only allowed on first visit (enforced by validation)

Response field: "persistence": "core" | "permanent" | "ephemeral"
═══════════════════════════════════════════════════════════════
"""

    # Build complete prompt
    prompt = f"""Location: {location_name} (ID: {location_id})
{existing_memories_section}
{deduplication_section}
{contradiction_section}
{persistence_section}
{action_analysis_section}
{output_format_section}
"""

    # Call LLM
    synthesis = self.llm_client.call(prompt)

    # POST-HOC VALIDATION: Enforce CORE constraint
    # NOTE: This is reactive (after LLM response) not preventative (schema-level)
    # because the Pydantic validator doesn't have access to z_machine_context.
    # This is intentional - serves as a guardrail for LLM mistakes.
    if synthesis.should_remember and synthesis.persistence == "core":
        if not first_visit:
            self.log_warning(
                "CORE memory rejected: not first visit, downgrading to PERMANENT",
                location_id=location_id,
                title=synthesis.memory_title,
                persistence_requested="core",
                first_visit=first_visit
            )
            synthesis.persistence = "permanent"

            # Track validation failures for monitoring
            # If this triggers frequently (>5%), prompt needs improvement
            self.metrics.increment("core_validation_downgrade")

    return synthesis
```

**Key Changes:**
1. **Unified prompt:** Always explain all three types (not conditional split)
2. **Context signals:** Use `z_machine_context` to provide concrete hints
3. **Action-based:** Classification driven by action type, not visit timing
4. **EPHEMERAL on first visits:** Now allowed (fixes CRITICAL FLAW #1)
5. **Post-hoc validation:** CORE constraint enforced after LLM response

---

## Memory Storage and Routing

### add_memory() - Route by Persistence

```python
def add_memory(
    self,
    location_id: int,
    location_name: str,
    memory: Memory
) -> bool:
    """Route memory to persistent file or ephemeral cache."""

    if memory.persistence == "ephemeral":
        # EPHEMERAL: In-memory only, never touch disk
        if location_id not in self.ephemeral_cache:
            self.ephemeral_cache[location_id] = []

        self.ephemeral_cache[location_id].append(memory)

        self.log_info(
            f"Added ephemeral memory (in-memory only): [{memory.category}] {memory.title}",
            location_id=location_id,
            episode=memory.episode,
            persistence="ephemeral"
        )

        return True

    else:
        # CORE or PERMANENT: Write to Memories.md
        success = self._write_memory_to_file(
            location_id,
            location_name,
            memory
        )

        if success:
            # Also add to persistent cache
            if location_id not in self.memory_cache:
                self.memory_cache[location_id] = []
            self.memory_cache[location_id].append(memory)

            self.log_info(
                f"Added {memory.persistence} memory to file: [{memory.category}] {memory.title}",
                location_id=location_id,
                persistence=memory.persistence
            )

        return success
```

### get_location_memory() - Combine Caches

```python
def get_location_memory(self, location_id: int) -> str:
    """
    Retrieve memories from both caches.

    Returns:
        Formatted string with:
        - Persistent memories (core + permanent from file)
        - Ephemeral memories (in-memory from current episode)
    """
    if location_id not in self.memory_cache and location_id not in self.ephemeral_cache:
        return ""

    # Get persistent memories (core + permanent)
    persistent = self.memory_cache.get(location_id, [])

    # Get ephemeral memories (current episode only)
    ephemeral = self.ephemeral_cache.get(location_id, [])

    # Combine and filter by status
    # ORDERING: Persistent first, then ephemeral
    # Rationale:
    #   - CORE memories (spawn state) are foundational - agent needs to know what's available
    #   - PERMANENT mechanics are reusable strategic knowledge
    #   - EPHEMERAL notes are tactical ("I dropped this here") - less critical
    #   - Agent reads foundation first, then current session context
    all_memories = []
    for mem in (persistent + ephemeral):
        if mem.status != MemoryStatus.SUPERSEDED:
            all_memories.append(mem)

    if not all_memories:
        return ""

    # Separate by status
    active = [m for m in all_memories if m.status == MemoryStatus.ACTIVE]
    tentative = [m for m in all_memories if m.status == MemoryStatus.TENTATIVE]

    # Format output
    lines = []

    if active:
        for mem in active:
            # Show persistence type for debugging
            persistence_marker = ""
            if mem.persistence == "ephemeral":
                persistence_marker = " [session]"
            elif mem.persistence == "core":
                persistence_marker = " [spawn]"

            lines.append(f"[{mem.category}] {mem.title}: {mem.text}{persistence_marker}")

    if tentative:
        if active:
            lines.append("")
        lines.append("⚠️  TENTATIVE MEMORIES (unconfirmed, may be invalidated):")
        for mem in tentative:
            lines.append(f"  [{mem.category}] {mem.title}: {mem.text}")

    return "\n".join(lines)
```

### reset_episode() - Clear Ephemeral Cache

```python
def reset_episode(self) -> None:
    """
    Reset manager state for new episode.

    CRITICAL: Clears ephemeral_cache to prevent false memories.
    Persistent cache (memory_cache) remains unchanged.
    """
    # Clear ephemeral memories
    ephemeral_count = sum(len(mems) for mems in self.ephemeral_cache.values())
    self.ephemeral_cache.clear()

    self.log_info(
        f"Episode reset: Cleared {ephemeral_count} ephemeral memories",
        ephemeral_count=ephemeral_count
    )

    # Note: memory_cache (persistent) is NOT cleared
```

### Public API for Testing and Introspection

**Problem:** Tests need to verify ephemeral cache state, but direct access couples implementation.

**Solution:** Provide public methods for cache introspection:

```python
def get_ephemeral_count(self, location_id: Optional[int] = None) -> int:
    """
    Get count of ephemeral memories.

    Args:
        location_id: Specific location, or None for total across all locations

    Returns:
        Count of ephemeral memories
    """
    if location_id is not None:
        return len(self.ephemeral_cache.get(location_id, []))
    else:
        return sum(len(mems) for mems in self.ephemeral_cache.values())

def get_persistent_count(self, location_id: Optional[int] = None) -> int:
    """
    Get count of persistent memories (CORE + PERMANENT).

    Args:
        location_id: Specific location, or None for total across all locations

    Returns:
        Count of persistent memories
    """
    if location_id is not None:
        return len(self.memory_cache.get(location_id, []))
    else:
        return sum(len(mems) for mems in self.memory_cache.values())

def get_memory_breakdown(self, location_id: int) -> Dict[str, int]:
    """
    Get breakdown of memory types at location.

    Returns:
        {"core": count, "permanent": count, "ephemeral": count}
    """
    breakdown = {"core": 0, "permanent": 0, "ephemeral": 0}

    for mem in self.memory_cache.get(location_id, []):
        if mem.status != MemoryStatus.SUPERSEDED:
            breakdown[mem.persistence] += 1

    for mem in self.ephemeral_cache.get(location_id, []):
        if mem.status != MemoryStatus.SUPERSEDED:
            breakdown[mem.persistence] += 1

    return breakdown
```

**Usage in Tests:**
```python
def test_ephemeral_not_persisted():
    manager = SimpleMemoryManager(...)

    # Use public API instead of accessing .ephemeral_cache directly
    assert manager.get_ephemeral_count(location_id=8) == 0

    manager.add_memory(8, "Kitchen", ephemeral_memory)

    assert manager.get_ephemeral_count(location_id=8) == 1
    assert manager.get_persistent_count(location_id=8) == 0
```

---

## File Format (Memories.md)

### Core and Permanent Only

**Ephemeral memories NEVER appear in Memories.md:**

```markdown
# Location Memories

## Location 5: Living Room
**Visits:** 3 | **Episodes:** 1, 2, 3

### Memories

**[DISCOVERY - CORE] Sword spawns here** *(Ep1, T2, +0)*
Sword found in room description on first visit to Living Room.

**[DISCOVERY - PERMANENT] Rug can be moved** *(Ep1, T10, +5)*
Moving rug reveals trap door. Grants 5 points on discovery.

---

## Location 8: Kitchen
**Visits:** 5 | **Episodes:** 1, 2

### Memories

**[DISCOVERY - CORE] Glass bottle holds water** *(Ep1, T15, +0)*
Bottle discovered on kitchen counter during first visit.

**[DISCOVERY - CORE] Brown sack contains food** *(Ep1, T21, +0)*
Sack found in kitchen with garlic and lunch inside.

**[DANGER - PERMANENT] Chimney too narrow to enter** *(Ep1, T30, +0)*
Attempting to enter chimney fails. Not a valid exit.

---
```

**Note:** No "Dropped sword here" or "Placed nest in sack" entries. These are ephemeral and live only in `ephemeral_cache`.

---

## Parsing Changes

### File Loading - Ignore persistence Field

```python
def _add_memory_to_cache(
    self,
    location_id: int,
    memory_header: tuple,
    text_lines: List[str],
    invalidation_info: Optional[tuple] = None
) -> None:
    """Add parsed memory to persistent cache."""

    category, status, title, metadata = memory_header

    # Parse metadata
    episode, turns, score_change = self._parse_metadata(metadata)

    # Determine persistence from category marker
    # Format: **[DISCOVERY - CORE]** or **[DANGER - PERMANENT]**
    persistence = "permanent"  # Default
    if " - CORE" in category:
        persistence = "core"
        category = category.replace(" - CORE", "")
    elif " - PERMANENT" in category:
        persistence = "permanent"
        category = category.replace(" - PERMANENT", "")

    # Create Memory object
    memory = Memory(
        category=category,
        title=title,
        episode=episode,
        turns=turns,
        score_change=score_change,
        text=" ".join(text_lines),
        status=status,
        persistence=persistence,
        # ... supersession fields
    )

    # Add to PERSISTENT cache only (from file)
    if location_id not in self.memory_cache:
        self.memory_cache[location_id] = []
    self.memory_cache[location_id].append(memory)
```

### File Writing - Include Persistence Marker

```python
def _format_memory_entry(self, memory: Memory) -> str:
    """Format memory entry with persistence marker."""

    # Build category string with persistence
    if memory.persistence == "core":
        category_str = f"{memory.category} - CORE"
    elif memory.persistence == "permanent":
        category_str = f"{memory.category} - PERMANENT"
    else:
        # GRACEFUL DEGRADATION: Log error instead of crashing
        self.log_error(
            f"Attempted to write EPHEMERAL memory to file (BUG): {memory.title}",
            persistence=memory.persistence,
            category=memory.category
        )
        return None  # Skip this memory, don't write to file

    # Format metadata
    metadata = f"Ep{memory.episode}, T{memory.turns}"
    if memory.score_change is not None:
        score_str = f"+{memory.score_change}" if memory.score_change >= 0 else str(memory.score_change)
        metadata += f", {score_str}"

    # Format header
    if memory.status == MemoryStatus.ACTIVE:
        header = f"**[{category_str}] {memory.title}** *({metadata})*"
    else:
        header = f"**[{category_str} - {memory.status}] {memory.title}** *({metadata})*"

    # Format text (strikethrough if superseded)
    text = f"~~{memory.text}~~" if memory.status == MemoryStatus.SUPERSEDED else memory.text

    # Build lines
    lines = [header]
    if memory.status == MemoryStatus.SUPERSEDED:
        if memory.invalidation_reason:
            lines.append(f'[Invalidated at T{memory.superseded_at_turn}: "{memory.invalidation_reason}"]')
        elif memory.superseded_by:
            lines.append(f'[Superseded at T{memory.superseded_at_turn} by "{memory.superseded_by}"]')
    lines.append(text)

    return "\n".join(lines)
```

---

## Edge Cases and Validation

### 1. CORE Memory on Return Visit
**Scenario:** LLM tries to create CORE memory when `first_visit=false`

**Solution:** Validation in synthesis prompt prevents this (conditional injection)

**Failsafe:** If LLM ignores instruction, log warning and downgrade to PERMANENT

```python
if synthesis.persistence == "core" and not z_machine_context.get('first_visit'):
    self.log_warning(
        "CORE memory rejected: not first visit, downgrading to PERMANENT",
        location_id=location_id,
        title=synthesis.memory_title
    )
    synthesis.persistence = "permanent"
```

### 2. EPHEMERAL Memory on First Visit (NOW SUPPORTED)
**Scenario:** Agent drops item immediately on first visit to location

**FIXED:** EPHEMERAL is now available on first visits (critical bug fix)

**Behavior:** LLM can correctly classify agent actions as EPHEMERAL regardless of visit status

**Example:**
```python
# Episode 1, Turn 5: First visit to Kitchen
z_machine_context = {'first_visit': True, 'inventory_changed': True}
action = "drop sword"
# LLM can now choose EPHEMERAL → Correct classification!
# Old spec would force PERMANENT → Bug (persists across episodes)
```

### 3. Superseding Ephemeral Memories
**Scenario:** Agent learns dropped item was actually part of puzzle

**Solution:** Ephemeral memories CAN be superseded within same episode

```python
# Episode 1, Turn 45: Dropped sword in Kitchen (ephemeral)
# Episode 1, Turn 50: Discover sword activates trap
# → Can supersede ephemeral memory with permanent lesson
```

**Implementation:** Supersession logic works across both caches

### 3a. Upgrading Persistence Level (NEW - CRITICAL ENHANCEMENT)
**Scenario:** Memory classified as EPHEMERAL but later discovered to be permanent game mechanic

**Example:**
```python
# Turn 45: Agent drops lamp in Kitchen → EPHEMERAL: "Dropped lamp here"
# Turn 50: Discover lamp placement triggers secret door
#          → Should upgrade to PERMANENT: "Lamp on table opens secret door"
```

**Problem:** Original EPHEMERAL memory in ephemeral_cache will be cleared on episode reset, losing puzzle solution

**Solution:** When superseding with different persistence level, migrate between caches:

```python
def supersede_memory(
    self,
    old_title: str,
    new_memory: Memory,
    location_id: int
) -> bool:
    """Supersede memory with potential persistence upgrade."""

    # Search both caches for old memory
    old_memory = None
    old_cache = None

    if location_id in self.memory_cache:
        for mem in self.memory_cache[location_id]:
            if mem.title == old_title:
                old_memory = mem
                old_cache = "persistent"
                break

    if old_memory is None and location_id in self.ephemeral_cache:
        for mem in self.ephemeral_cache[location_id]:
            if mem.title == old_title:
                old_memory = mem
                old_cache = "ephemeral"
                break

    if old_memory is None:
        return False  # Memory not found

    # Mark old memory as superseded
    old_memory.status = MemoryStatus.SUPERSEDED
    old_memory.superseded_by = new_memory.title
    old_memory.superseded_at_turn = new_memory.turns

    # PERSISTENCE UPGRADE: Migrate between caches if needed
    if old_cache == "ephemeral" and new_memory.persistence in ["core", "permanent"]:
        # Upgrade: ephemeral → persistent
        # Remove from ephemeral_cache (will be garbage collected)
        self.ephemeral_cache[location_id] = [
            m for m in self.ephemeral_cache[location_id] if m.title != old_title
        ]
        self.log_info(
            f"Upgraded memory from EPHEMERAL to {new_memory.persistence.upper()}",
            old_title=old_title,
            new_title=new_memory.title
        )

    elif old_cache == "persistent" and new_memory.persistence == "ephemeral":
        # Downgrade: persistent → ephemeral (rare but possible)
        self.log_warning(
            "Downgrading PERSISTENT memory to EPHEMERAL - verify correctness",
            old_title=old_title,
            new_title=new_memory.title
        )

    # Add new memory to appropriate cache
    return self.add_memory(location_id, location_name, new_memory)
```

**IMPORTANT:** This fixes CRITICAL FLAW #2 - allows agent to upgrade understanding from temporary state to permanent mechanic

### 4. Episode Reset Mid-Session
**Scenario:** Orchestrator calls `reset_episode()` without full restart

**Solution:** Ephemeral cache cleared automatically, no action needed

### 5. Reloading from File
**Scenario:** Process restarts mid-episode, ephemeral memories lost

**Behavior:** **Expected** - ephemeral memories are session-only

**Alternative:** If persistence needed, write ephemeral to separate session file (not Memories.md)

---

## Testing Strategy

### Unit Tests

```python
def test_ephemeral_memory_not_persisted():
    """Ephemeral memories should not be written to Memories.md."""
    manager = SimpleMemoryManager(...)

    memory = Memory(
        category="NOTE",
        title="Dropped sword here",
        episode=1,
        turns="45",
        score_change=0,
        text="Agent dropped sword in kitchen.",
        persistence="ephemeral"
    )

    manager.add_memory(location_id=8, location_name="Kitchen", memory=memory)

    # Check ephemeral cache populated
    assert len(manager.ephemeral_cache[8]) == 1

    # Check Memories.md NOT updated
    memories_file = Path(manager.config.zork_game_workdir) / "Memories.md"
    content = memories_file.read_text()
    assert "Dropped sword here" not in content

def test_core_memory_persisted():
    """Core memories should be written to Memories.md."""
    manager = SimpleMemoryManager(...)

    memory = Memory(
        category="DISCOVERY",
        title="Sword spawns here",
        episode=1,
        turns="2",
        score_change=0,
        text="Sword found in room description.",
        persistence="core"
    )

    manager.add_memory(location_id=5, location_name="Living Room", memory=memory)

    # Check persistent cache populated
    assert len(manager.memory_cache[5]) == 1

    # Check Memories.md updated
    memories_file = Path(manager.config.zork_game_workdir) / "Memories.md"
    content = memories_file.read_text()
    assert "Sword spawns here" in content
    assert "DISCOVERY - CORE" in content

def test_ephemeral_cleared_on_reset():
    """Ephemeral cache cleared on episode reset."""
    manager = SimpleMemoryManager(...)

    # Add ephemeral memory
    memory = Memory(
        category="NOTE",
        title="Dropped sword here",
        episode=1,
        turns="45",
        score_change=0,
        text="Agent dropped sword.",
        persistence="ephemeral"
    )
    manager.add_memory(location_id=8, location_name="Kitchen", memory=memory)

    # Verify populated
    assert len(manager.ephemeral_cache[8]) == 1

    # Reset episode
    manager.reset_episode()

    # Verify cleared
    assert len(manager.ephemeral_cache) == 0

def test_retrieval_combines_caches():
    """get_location_memory() should return persistent + ephemeral."""
    manager = SimpleMemoryManager(...)

    # Add core memory (persistent)
    core = Memory(
        category="DISCOVERY",
        title="Sword spawns here",
        episode=1,
        turns="2",
        score_change=0,
        text="Sword in room.",
        persistence="core"
    )
    manager.memory_cache[5] = [core]

    # Add ephemeral memory
    ephemeral = Memory(
        category="NOTE",
        title="Dropped lamp here",
        episode=1,
        turns="45",
        score_change=0,
        text="Agent dropped lamp.",
        persistence="ephemeral"
    )
    manager.ephemeral_cache[5] = [ephemeral]

    # Retrieve
    result = manager.get_location_memory(location_id=5)

    # Both present
    assert "Sword spawns here" in result
    assert "Dropped lamp here" in result

def test_conditional_prompt_first_visit():
    """Synthesis prompt should include CORE instructions on first visit."""
    manager = SimpleMemoryManager(...)

    z_machine_context = {'first_visit': True}

    # Mock LLM to capture prompt
    prompt_capture = []
    def mock_llm_call(prompt):
        prompt_capture.append(prompt)
        return Mock(...)

    manager.llm_client.call = mock_llm_call
    manager._synthesize_memory(..., z_machine_context=z_machine_context)

    prompt = prompt_capture[0]
    assert "**CORE**" in prompt
    assert "spawn state" in prompt.lower()

def test_conditional_prompt_return_visit():
    """Synthesis prompt should include EPHEMERAL instructions on return visit."""
    manager = SimpleMemoryManager(...)

    z_machine_context = {'first_visit': False}

    # Mock LLM
    prompt_capture = []
    def mock_llm_call(prompt):
        prompt_capture.append(prompt)
        return Mock(...)

    manager.llm_client.call = mock_llm_call
    manager._synthesize_memory(..., z_machine_context=z_machine_context)

    prompt = prompt_capture[0]
    assert "**EPHEMERAL**" in prompt
    assert "Agent-caused state changes" in prompt
```

### Integration Tests

```python
def test_full_episode_lifecycle():
    """Test complete episode with core, permanent, and ephemeral memories."""

    orchestrator = ZorkOrchestratorV2(...)

    # Episode 1, Turn 1: First visit to Living Room
    orchestrator.process_turn(action="look")
    # Should create CORE memory: "Sword spawns here"

    # Episode 1, Turn 5: Take sword
    orchestrator.process_turn(action="take sword")

    # Episode 1, Turn 10: Drop sword in Kitchen
    orchestrator.process_turn(action="drop sword")
    # Should create EPHEMERAL memory: "Dropped sword here"

    # Verify ephemeral in cache
    ephemeral = orchestrator.simple_memory.ephemeral_cache[kitchen_id]
    assert len(ephemeral) == 1

    # Episode 2: Reset
    orchestrator.reset_episode()

    # Verify ephemeral cleared
    assert len(orchestrator.simple_memory.ephemeral_cache) == 0

    # Episode 2, Turn 1: Visit Kitchen
    memories = orchestrator.simple_memory.get_location_memory(kitchen_id)

    # Should NOT see "Dropped sword here"
    assert "Dropped sword" not in memories
```

---

## Migration Strategy

### Phase 1: Code Implementation (Week 1)
1. Add `persistence` field to Memory dataclass
2. Add `ephemeral_cache: Dict[int, List[Memory]]` to SimpleMemoryManager
3. Update `reset_episode()` to clear ephemeral cache
4. Implement conditional prompt injection in `_synthesize_memory()`
5. Update `add_memory()` to route by persistence
6. Update `get_location_memory()` to combine caches
7. Update file format parsing to read persistence markers
8. Update file writing to include persistence markers

### Phase 2: Testing (Week 1-2)
1. Write unit tests for ephemeral cache operations
2. Write unit tests for conditional prompting
3. Write integration tests for episode lifecycle
4. Run existing test suite to ensure no regressions

### Phase 3: Existing Memory Classification (Week 2)
1. Review existing Memories.md entries (~100 memories)
2. Classify each as CORE or PERMANENT:
   - Room description items → CORE
   - Game mechanics learned → PERMANENT
3. Update file format with persistence markers
4. Validate classification correctness

### Phase 4: Monitoring and Refinement (Week 3+)
1. Run episodes and monitor memory classification
2. Check for misclassifications (core on return visit, etc.)
3. Refine synthesis prompts based on LLM behavior
4. Adjust persistence logic if edge cases discovered

---

## Configuration Options

### Add to pyproject.toml

```toml
[tool.zorkgpt.memory]
# Show persistence markers in agent context
show_persistence_markers = true  # "[session]", "[spawn]" suffixes

# Debug: Log all ephemeral memory operations
debug_ephemeral = false
```

**Note:** Ephemeral cache is always enabled - it's core functionality, not an optional feature.

### Add to GameConfiguration

```python
class GameConfiguration:
    def __init__(self, config_dict: dict):
        # ... existing config

        memory_config = config_dict.get("tool", {}).get("zorkgpt", {}).get("memory", {})
        self.show_persistence_markers = memory_config.get("show_persistence_markers", True)
        self.debug_ephemeral = memory_config.get("debug_ephemeral", False)
```

---

## Benefits

### 1. Eliminates False Memories
✅ Agent no longer expects dropped items in subsequent episodes
✅ Spawn locations correctly remembered across resets
✅ Game mechanics persist, agent state does not

### 2. No File Pollution
✅ Memories.md contains only permanent knowledge
✅ No filtering logic needed to hide ephemeral entries
✅ File remains human-readable and concise

### 3. Automatic Cleanup
✅ `reset_episode()` automatically clears ephemeral cache
✅ No manual deletion or invalidation needed
✅ Natural lifecycle tied to episode boundaries

### 4. Clear Separation of Concerns
✅ Disk = permanent knowledge (core + permanent)
✅ Memory = current session state (ephemeral)
✅ Easy to reason about what persists

### 5. Semi-Backward Compatible (Migration Required)
⚠️  Existing memories need classification review (manual or automated)
✅ File format extends existing structure (adds persistence markers)
✅ Parser defaults unmarked memories to "permanent"
⚠️  ~100 memories require manual CORE vs PERMANENT classification
✅ System functional immediately, migration improves accuracy over time

---

## Design Decisions

### Persistence Markers in Agent Context

**Decision:** Agent DOES see persistence markers (`[session]`, `[spawn]`) in memory text.

**Rationale:**
- **Benefit:** Agent understands memory lifecycle - knows that `[session]` memories are temporary to current episode
- **Benefit:** Helps agent distinguish spawn state `[spawn]` from mechanics (unmarked PERMANENT)
- **Cost:** Minimal token overhead (~10 characters per memory)
- **Configurable:** `show_persistence_markers` flag in config allows disabling if needed

**Example agent context:**
```
[DISCOVERY] Sword spawns here [spawn]
[PERMANENT] Window can be opened
[NOTE] Dropped lamp here [session]
```

Agent can reason: "The sword spawns here originally, but I dropped the lamp here this episode. After reset, sword will return but lamp won't be here."

---

## Open Questions

1. **Should ephemeral memories persist across process restarts?**
   - Current: No - ephemeral lost if process crashes mid-episode
   - Alternative: Write to `ephemeral_session.json` for recovery
   - Decision: Start with no persistence, add if needed

2. **Can permanent memories be created on first visit?**
   - Current: Yes - both CORE and PERMANENT allowed
   - Example: "Window can be opened" discovered on first visit
   - Decision: Allow both, LLM chooses based on content

3. **Should multi-step procedures be CORE or PERMANENT?**
   - Example: "To enter kitchen: (1) open window, (2) enter"
   - Behavior: Procedure learned across multiple turns (not first visit)
   - Decision: PERMANENT (procedural knowledge, not spawn state)

---

## Success Metrics

### Quantitative (Verifiable)
- **Ephemeral cache cleared on reset**: 100% (automated check: `get_ephemeral_count() == 0` after `reset_episode()`)
- **CORE only on first visit**: > 99% (log warning when CORE downgraded due to `first_visit=false`)
- **No EPHEMERAL in Memories.md**: 100% (grep file for persistence markers, verify no `[session]` or EPHEMERAL entries)
- **File size growth**: < 10% increase (compare file size before/after migration - no ephemeral bloat)
- **Cache consistency**: 100% (after restart, `memory_cache` matches file contents)

### Qualitative (Manual Review)
- Agent correctly returns to spawn locations for items
- Agent does not search for dropped items after episode reset
- Memories.md remains clean and human-readable
- No manual memory cleanup required after episodes
- CORE/PERMANENT distinction is semantically correct when sampled

---

## Future Enhancements

### 1. Ephemeral Session Persistence
If mid-episode crashes become common, add optional persistence:

```python
def _save_ephemeral_session(self):
    """Save ephemeral cache to session file for recovery."""
    session_file = Path(self.config.zork_game_workdir) / "ephemeral_session.json"
    data = {
        "episode_id": self.game_state.episode_id,
        "ephemeral_cache": {
            str(loc_id): [mem.to_dict() for mem in mems]
            for loc_id, mems in self.ephemeral_cache.items()
        }
    }
    session_file.write_text(json.dumps(data))

def _load_ephemeral_session(self):
    """Restore ephemeral cache from session file."""
    session_file = Path(self.config.zork_game_workdir) / "ephemeral_session.json"
    if not session_file.exists():
        return

    data = json.loads(session_file.read_text())
    if data["episode_id"] == self.game_state.episode_id:
        # Same episode - restore
        for loc_id, mems in data["ephemeral_cache"].items():
            self.ephemeral_cache[int(loc_id)] = [Memory.from_dict(m) for m in mems]
```

### 2. PROCEDURAL Persistence Type
Add fourth type for multi-step procedures that aren't spawn state:

```python
persistence: str = "permanent"  # "core" | "permanent" | "procedural" | "ephemeral"
```

Would always persist like PERMANENT, but formatted differently in context.

---

## Related Documents

- `managers/CLAUDE.md` - Memory manager patterns and lifecycle
- `game_files/Memories.md` - Existing memory database (to be migrated)
- `tests/test_simple_memory_manager.py` - Memory system tests
- `loop_break.md` - Loop detection system (separate concern)

---

## SPEC CORRECTIONS AND CRITICAL FIXES

**This specification has been corrected to fix 2 critical design flaws, 7 contradictions, 13 ambiguities, and 6 inaccuracies identified during systematic review.**

### Critical Flaws Fixed

#### FLAW #1: EPHEMERAL Prohibited on First Visits (FIXED)
**Original Problem:** Spec prohibited EPHEMERAL creation on first visits, forcing agent-caused state changes to be PERMANENT, recreating the exact bug the system aimed to fix.

**Scenario that would fail:**
```
Episode 1, Turn 5: Agent enters Kitchen for first time → first_visit=true
                    Agent drops sword
                    Original spec forces classification as PERMANENT
                    Memory persists across episodes → BUG!

Episode 2: Agent expects sword in Kitchen but it respawned in Living Room
```

**Fix Applied:**
- Lines 83-87: Updated EPHEMERAL criteria to allow creation on ANY visit
- Lines 203-340: Replaced conditional prompt split with unified approach
- Lines 275-286: Added EPHEMERAL examples with explicit "ANY visit" marker
- Lines 291-295: Decision criteria based on action type, not visit timing

**Key Change:** Classification now based on WHAT HAPPENED (action type), not WHEN (visit timing)

#### FLAW #2: No Persistence Upgrade Path (FIXED)
**Original Problem:** Once classified as EPHEMERAL, no mechanism to upgrade to PERMANENT when semantic meaning changes.

**Scenario that would fail:**
```
Turn 45: Agent drops lamp in Kitchen → EPHEMERAL
Turn 50: Discovers lamp placement triggers secret door → Should be PERMANENT
         But original EPHEMERAL gets cleared on episode reset
         Agent loses puzzle solution!
```

**Fix Applied:**
- Lines 661-735: Added new edge case "3a. Upgrading Persistence Level"
- Includes `supersede_memory()` implementation with cache migration
- Handles upgrade (ephemeral → persistent) and downgrade (persistent → ephemeral)

### Major Corrections

1. **Memory Dataclass** (line 136): Made `persistence` required field (no default) to force explicit classification
2. **MemorySynthesisResponse Schema** (lines 157-174): Added validator to enforce persistence when `should_remember=True`
3. **Error Handling** (lines 581-587): Changed from ValueError crash to graceful log + skip for ephemeral writes
4. **Public API** (lines 476-545): Added testing methods to avoid coupling to implementation details
5. **Backward Compatibility** (lines 1034-1039): Corrected claim from "No breaking changes" to "Semi-backward compatible (migration required)"

### Ambiguities Resolved

1. **CORE Definition** (lines 247-259): Clarified "ONLY for passive observations from room text, NOT for agent actions"
2. **Context Signals** (lines 220-239): Added explicit hints from z_machine_context to guide LLM
3. **Visit Status** (line 300): Made explicit that visit status is informational, not determinative
4. **Edge Case #2** (lines 632-646): Changed from problem to solution ("NOW SUPPORTED")

### Documentation Improvements

1. Added **Key Changes** summary (lines 335-340)
2. Added **Public API for Testing** section (lines 476-545)
3. Added **CRITICAL ENHANCEMENT** marker for persistence upgrades (line 735)
4. Clarified semi-backward compatibility requirements (lines 1034-1039)

---

**Status:** Specification CORRECTED and ready for implementation
**Last Updated:** 2025-01-05 (Systematic review and corrections)
**Estimated Effort:** 2-3 weeks (implementation + testing + migration)
**Priority:** CRITICAL (fixes major false memory bug + prevents persistence misclassification)
**Complexity:** Medium-High (dual cache system, unified prompting, cache migration)
