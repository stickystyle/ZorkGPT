# ObjectiveManager Enhancement Specification

## Overview

Enhance the ObjectiveManager to leverage strategic knowledge, location-specific memories, and map data when discovering and maintaining objectives. This will enable the agent to create objectives that align with learned strategic patterns, use procedural knowledge from previous episodes, and pursue exploration opportunities.

## Current State

### What ObjectiveManager Currently Uses

**Data Sources:**
- Recent memory log history (last 20 entries from `game_state.memory_log_history`)
- Recent action history (last 10 actions from `game_state.action_history`)
- Current agent reasoning (passed as parameter)
- Current game state (score, location, inventory)

**Dependencies:**
- `adaptive_knowledge_manager` - Only used for LLM client access, not for knowledge content

**Limitations:**
- ❌ No access to strategic knowledge base (`game_files/knowledgebase.md`)
- ❌ No access to location-specific memories (`game_files/Memories.md`)
- ❌ No access to map structure or exploration opportunities
- ❌ Cannot create objectives based on cross-episode learning
- ❌ Cannot leverage procedural knowledge from memories
- ❌ Cannot suggest exploration based on map topology

## Motivation

### Problem Statement

**Scenario 1: Missing Strategic Context**
```
Current: "Explore the troll area"
Better:  "Avoid troll encounter (knowledge warns: troll blocks passage, requires specific item)"
```

**Scenario 2: Ignoring Procedural Knowledge**
```
Current: "Enter the kitchen"
Better:  "Use window entry procedure from Location 79 memory (open window → enter window → reach Location 62 kitchen)"
```

**Scenario 3: No Exploration Guidance**
```
Current: "Explore more areas"
Better:  "Explore north to Location 81 (adjacent room with unexplored exits, no memories yet)"
```

### Expected Benefits

1. **Strategic Alignment**: Objectives informed by accumulated wisdom (dangers, puzzle solutions, resource priorities)
2. **Procedural Leverage**: Use multi-step procedures from memories instead of rediscovering them
3. **Exploration Guidance**: Target specific unexplored areas based on map topology
4. **Cross-Episode Learning**: Benefit from discoveries in previous episodes
5. **Location Awareness**: Understand spatial relationships and routing between objectives

## Proposed Architecture

### New Dependencies

```python
def __init__(
    self,
    logger,
    config: GameConfiguration,
    game_state: GameState,
    adaptive_knowledge_manager: AdaptiveKnowledgeManager,
    map_manager: "MapManager" = None,  # NEW
    memory_manager: "SimpleMemoryManager" = None,  # NEW
):
    super().__init__(logger, config, game_state, "objective_manager")
    self.adaptive_knowledge_manager = adaptive_knowledge_manager
    self.map_manager = map_manager
    self.memory_manager = memory_manager
```

### Data Integration Strategy

**Approach: Hybrid - Full Knowledge + Filtered Memories + Map Summary**

| Data Source | Strategy | Rationale |
|-------------|----------|-----------|
| **Knowledge** | Full file | Strategic wisdom (updated infrequently, ~2-5K tokens) |
| **Memories** | Current + Adjacent | Tactical procedures (filtered by location, ~800-1.5K tokens) |
| **Map** | Mermaid + Summary | Exploration context (visual + text, ~500-800 tokens) |

**Total Context Addition**: ~3-7K tokens per objective update (every 20 turns = acceptable)

## Implementation Details

### 1. Helper Methods

#### `_get_full_knowledge() -> str`

**Purpose**: Load complete strategic knowledge base

```python
def _get_full_knowledge(self) -> str:
    """Load full knowledge base - strategic wisdom."""
    kb_path = Path(self.config.zork_game_workdir) / self.config.knowledge_file
    if kb_path.exists():
        return kb_path.read_text(encoding="utf-8")
    return "No strategic knowledge available."
```

**Why full file?**
- Knowledge base is curated, high-value content
- Updates infrequently (only after episode synthesis)
- Objectives only update every 20 turns (not every turn)

---

#### `_get_memories_with_adjacent(location_id: int) -> str`

**Purpose**: Get memories for current location + adjacent locations (1 hop away)

```python
def _get_memories_with_adjacent(self, current_location_id: int) -> str:
    """
    Get memories for current location + adjacent locations.

    Returns formatted memory text for prompt inclusion.
    """
    if not self.memory_manager:
        return "No memory data available"

    lines = []

    # Current location memories
    current_memories = self.memory_manager.memory_cache.get(current_location_id, [])
    if current_memories:
        current_name = self.game_state.current_room_name_for_map
        lines.append(f"## Memories for Current Location {current_location_id} ({current_name}):")
        lines.append(self._format_memories(current_memories))

    # Adjacent location memories
    adjacent_ids = self._get_adjacent_room_ids(current_location_id)
    if adjacent_ids:
        lines.append("\n## Memories for Adjacent Locations:")
        for adj_id in sorted(adjacent_ids)[:5]:  # Limit to 5 adjacent rooms
            adj_memories = self.memory_manager.memory_cache.get(adj_id, [])
            if adj_memories:
                adj_name = self.map_manager.game_map.room_names.get(adj_id, f"Location #{adj_id}")
                lines.append(f"\n**Location {adj_id} ({adj_name}):**")
                lines.append(self._format_memories(adj_memories[:3]))  # Top 3 memories per location

    if not lines:
        return "No memories available for current or adjacent locations"

    return "\n".join(lines)
```

**Why adjacent locations?**
- Provides context for where agent could go next
- Enables objectives like "Go north to Location 81 to investigate lantern memory"
- Limits token usage (current + 5 adjacent = ~6 locations max)

---

#### `_get_adjacent_room_ids(location_id: int, max_depth: int = 1) -> List[int]`

**Purpose**: Get IDs of rooms adjacent to given location

```python
def _get_adjacent_room_ids(self, location_id: int, max_depth: int = 1) -> List[int]:
    """
    Get IDs of rooms adjacent to given location.

    Args:
        location_id: Current room ID
        max_depth: How many hops away (1 = immediate neighbors)

    Returns:
        List of adjacent room IDs
    """
    if not self.map_manager:
        return []

    adjacent_ids = set()
    map_data = self.map_manager.game_map

    # Get rooms we can reach FROM current location (outgoing connections)
    if location_id in map_data.connections:
        for exit_action, dest_id in map_data.connections[location_id].items():
            adjacent_ids.add(dest_id)

    # Get rooms that can reach US (incoming connections)
    for source_id, exits in map_data.connections.items():
        for exit_action, dest_id in exits.items():
            if dest_id == location_id:
                adjacent_ids.add(source_id)

    return list(adjacent_ids)
```

**Why bidirectional?**
- Outgoing: Where we can go from here
- Incoming: Where we came from (might have memories worth revisiting)

---

#### `_get_map_context() -> str`

**Purpose**: Generate map context including Mermaid diagram and exploration summary

```python
def _get_map_context(self) -> str:
    """
    Get map context including Mermaid diagram and exploration summary.

    Returns human-readable map information with location IDs.
    """
    if not self.map_manager:
        return "No map data available"

    lines = []

    # Mermaid diagram (uses L{location_id} format as of Phase X)
    mermaid = self.map_manager.game_map.render_mermaid()
    lines.append("## Map Visualization (Mermaid Format)")
    lines.append(mermaid)
    lines.append("\nNote: Node IDs match location IDs in memories (L180 = Location 180)")
    lines.append(f"**Current location**: L{self.game_state.current_room_id}\n")

    # Current location routing
    lines.append(self._get_routing_summary(self.game_state.current_room_id))

    # Exploration statistics
    total_rooms = len(self.map_manager.game_map.rooms)
    lines.append(f"\n## Exploration Statistics")
    lines.append(f"- Rooms discovered: {total_rooms}")
    lines.append(f"- Current location: {self.game_state.current_room_name_for_map} (ID: {self.game_state.current_room_id})")

    return "\n".join(lines)
```

**Why Mermaid + Text?**
- Mermaid: Visual understanding of spatial relationships
- Text: Actionable routing information
- Location IDs: Consistent with memories file format

---

#### `_get_routing_summary(current_location_id: int) -> str`

**Purpose**: Generate text-based routing summary for current location and adjacent rooms

```python
def _get_routing_summary(self, current_location_id: int) -> str:
    """
    Generate text-based routing summary for current location and adjacent rooms.

    Returns human-readable routing information with location IDs.
    """
    if not self.map_manager:
        return "No routing data available"

    map_graph = self.map_manager.game_map
    lines = []

    # Current location connections
    current_name = map_graph.room_names.get(current_location_id, f"Location #{current_location_id}")
    lines.append(f"## Current Location: {current_location_id} ({current_name})")

    if current_location_id in map_graph.connections:
        lines.append("**Available Exits:**")
        for exit_action, dest_id in sorted(map_graph.connections[current_location_id].items()):
            dest_name = map_graph.room_names.get(dest_id, f"Location #{dest_id}")
            lines.append(f"  - {exit_action} → Location {dest_id} ({dest_name})")
    else:
        lines.append("  - No mapped exits")

    # Adjacent locations (1 hop away)
    adjacent_ids = self._get_adjacent_room_ids(current_location_id)
    if adjacent_ids:
        lines.append("\n## Adjacent Locations (1 hop away):")
        for adj_id in sorted(adjacent_ids)[:5]:  # Limit to 5 to control token usage
            adj_name = map_graph.room_names.get(adj_id, f"Location #{adj_id}")
            lines.append(f"\n**Location {adj_id} ({adj_name}):**")

            if adj_id in map_graph.connections:
                for exit_action, dest_id in sorted(list(map_graph.connections[adj_id].items())[:3]):  # Limit exits
                    dest_name = map_graph.room_names.get(dest_id, f"Location #{dest_id}")
                    back_marker = " [back to current]" if dest_id == current_location_id else ""
                    lines.append(f"  - {exit_action} → Location {dest_id} ({dest_name}){back_marker}")

    return "\n".join(lines)
```

**Why routing summary?**
- LLM can understand: "To reach L62, from L180 go to L79, then 'enter window'"
- Location IDs match everywhere (memories, map, Z-machine)
- Enables specific, actionable objectives

---

#### `_format_memories(memories: List[Memory]) -> str`

**Purpose**: Format list of Memory objects for prompt

```python
def _format_memories(self, memories: List[Memory]) -> str:
    """Format list of Memory objects for prompt."""
    if not memories:
        return "  (No memories)"

    formatted = []
    for mem in memories[:5]:  # Top 5 memories
        status_marker = f" [{mem.status}]" if mem.status != "ACTIVE" else ""
        formatted.append(f"  - [{mem.category}] {mem.title}{status_marker}")
        formatted.append(f"    {mem.text}")

    return "\n".join(formatted)
```

**Why format like this?**
- Matches memory file format (category in brackets)
- Shows status for TENTATIVE/SUPERSEDED memories
- Concise but informative

---

### 2. Updated Prompt Structure

**In `_update_discovered_objectives()`, replace current prompt with:**

```python
# Get contextual data
knowledge_content = self._get_full_knowledge()
memories_content = self._get_memories_with_adjacent(self.game_state.current_room_id)
map_context = self._get_map_context()

prompt = f"""Analyze recent gameplay and available knowledge to discover objectives.

=== STRATEGIC KNOWLEDGE (General Wisdom) ===
{knowledge_content}

=== LOCATION-SPECIFIC MEMORIES ===
{memories_content}

=== MAP & ROUTING INFORMATION ===
{map_context}

=== RECENT GAMEPLAY ===
{gameplay_context}

CURRENT STATE:
- Score: {self.game_state.previous_zork_score}
- Location: {self.game_state.current_room_name_for_map} (ID: {self.game_state.current_room_id})
- Inventory: {self.game_state.current_inventory}

Based on ALL of this context, identify objectives that:

1. **Align with strategic knowledge**
   - Avoid known dangers mentioned in knowledge base
   - Pursue known high-value goals (treasures, puzzle solutions)
   - Follow resource priority guidance (lantern > axe > sack)
   - Apply learned command patterns and syntax

2. **Leverage location-specific memories**
   - Use known procedures from current or adjacent locations
   - Example: "Memory shows window entry sequence at Location 79 → create objective to use it"
   - Build on previous discoveries rather than rediscovering

3. **Address exploration opportunities**
   - Prioritize unexplored exits shown in map
   - Investigate adjacent rooms with interesting memories
   - Example: "Adjacent Location 81 has lantern memory → objective to go north and investigate"

4. **Build on recent gameplay patterns**
   - Continue successful strategies from recent actions
   - Learn from recent failures or obstacles

**Good Objective Examples:**
✅ "Use window entry procedure from Location 79 memory (open window → enter window) to access Location 62 (Kitchen)"
✅ "Avoid troll encounter at Location 152 (knowledge warns: requires specific item or combat)"
✅ "Explore north to Location 81 (adjacent room with unexplored exits, lantern mentioned in memory)"
✅ "Secure brass lantern before dark areas (knowledge priority: light source critical)"

**Bad Objective Examples:**
❌ "Explore the house" (too vague, no location IDs)
❌ "Get items" (no specifics, doesn't leverage context)
❌ "Try random actions" (ignores knowledge and memories)

**IMPORTANT**:
- Use location IDs when referencing locations (e.g., "Location 79", not just "behind house")
- Reference specific memories or knowledge when creating objectives
- Prioritize objectives that leverage cross-episode learning

**CRITICAL - OUTPUT FORMAT:**
YOU MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.

Required JSON format:
{{
  "objectives": ["objective 1", "objective 2", ...],
  "reasoning": "brief explanation of how objectives align with knowledge/memories/map"
}}

Example valid response:
{{
  "objectives": [
    "Use window entry procedure from Location 79 memory to access Location 62 (Kitchen)",
    "Avoid troll at Location 152 per knowledge base warning",
    "Explore north to Location 81 to investigate lantern memory"
  ],
  "reasoning": "Objectives leverage window procedure memory (cross-episode learning), avoid known danger (strategic knowledge), and pursue exploration opportunity with useful item (map + memory combination)."
}}"""
```

---

### 3. Orchestrator Updates

**In `orchestration/zork_orchestrator_v2.py` initialization:**

```python
# Initialize ObjectiveManager with new dependencies
self.objective_manager = ObjectiveManager(
    logger=self.logger,
    config=self.config,
    game_state=self.game_state,
    adaptive_knowledge_manager=self.adaptive_knowledge_manager,
    map_manager=self.map_manager,  # NEW: Pass MapManager reference
    memory_manager=self.simple_memory,  # NEW: Pass SimpleMemoryManager reference
)
```

**Manager Dependency Order** (from `managers/CLAUDE.md`):
```
MapManager → no dependencies ✅
SimpleMemoryManager → no dependencies ✅
ContextManager → no dependencies ✅
StateManager → needs LLM client ✅
KnowledgeManager → needs agent and map references ✅
ObjectiveManager → needs knowledge, map, memory managers ⬅️ NEW
EpisodeSynthesizer → needs knowledge and state managers ✅
```

## Example Outputs

### Scenario 1: Using Memory Procedure

**Context:**
- Current location: 79 (Behind House)
- Memory at Location 79: `[SUCCESS] Window entry sequence (Ep01, T47-49, +0)` - "To enter kitchen: (1) examine window, (2) open window, (3) enter window"
- Map shows: L79 connects to L62 (Kitchen) via "enter window"

**Expected Objective:**
```json
{
  "objectives": [
    "Use window entry procedure from Location 79 memory: examine window → open window → enter window to reach Location 62 (Kitchen)",
    "Once in Kitchen, search for items and score opportunities"
  ],
  "reasoning": "Memory at current location shows proven 3-step procedure for kitchen access. Leveraging cross-episode learning instead of rediscovering the sequence."
}
```

### Scenario 2: Avoiding Known Danger

**Context:**
- Current location: 150 (Maze Entrance)
- Knowledge base: "Troll at Location 152 blocks passage. Requires: (1) sword for combat OR (2) lunch offering. Combat risky without full health."
- Inventory: [brass lantern, rusty key]
- Map shows: north from 150 → 152 (Troll Room)

**Expected Objective:**
```json
{
  "objectives": [
    "Avoid north route to Location 152 (Troll Room) until acquiring sword or lunch per knowledge base",
    "Explore alternative routes: try east or west from current location",
    "Search adjacent areas for sword or lunch items before troll encounter"
  ],
  "reasoning": "Knowledge base warns about troll danger at Location 152. Agent lacks required items (sword/lunch). Prioritizing safer exploration and item acquisition before risky encounter."
}
```

### Scenario 3: Exploration Guidance

**Context:**
- Current location: 180 (West House)
- Map shows: L180 connects north to L81 (North House)
- Memory at L81: `[DISCOVERY] Brass lantern found (Ep01, T12, +0)` - "Living room contains brass lantern on table. Critical for dark areas."
- Adjacent location L81 has unexplored exits: east, up

**Expected Objective:**
```json
{
  "objectives": [
    "Go north to Location 81 (North House) - memory indicates brass lantern location",
    "Secure brass lantern from Location 81 if still available (knowledge: light source = top priority)",
    "Explore unexplored exits from Location 81: east and up directions"
  ],
  "reasoning": "Adjacent location memory shows brass lantern (top priority per knowledge). Map confirms unexplored exits from L81. Combining memory guidance with exploration opportunity."
}
```

## Token Budget Analysis

### Per Objective Update (Every 20 Turns)

| Component | Estimated Tokens | Notes |
|-----------|-----------------|-------|
| **Knowledge Base** | 2,000 - 5,000 | Full file, strategic wisdom |
| **Current Location Memories** | 300 - 500 | 3-5 memories, formatted |
| **Adjacent Memories** | 500 - 1,000 | 5 locations × 3 memories each |
| **Map Mermaid** | 200 - 400 | Visual diagram |
| **Routing Summary** | 200 - 400 | Current + 5 adjacent rooms |
| **Recent Gameplay** | 500 - 800 | Already in current prompt |
| **System Prompt** | 800 - 1,200 | Instructions and examples |
| **TOTAL ADDED** | **~3,000 - 7,000** | New context vs current |

**Total Objective Update Context**: ~10K - 15K tokens (acceptable for 20-turn interval)

### Trade-offs

**Pros:**
- Rich context enables high-quality objectives
- Only happens every 20 turns (not every turn)
- Leverages expensive knowledge curation work
- Enables cross-episode learning payoff

**Cons:**
- Larger LLM calls for objective updates
- Could hit token limits with very large knowledge bases

**Mitigation Strategies:**
1. **If knowledge base grows > 10K tokens:**
   - Add summarization step (LLM-generated summary for objectives)
   - Split knowledge into "objectives-relevant" section

2. **If memory filtering needed:**
   - Add recency filter (only memories from last N episodes)
   - Add category filter (only SUCCESS/DISCOVERY for objectives)

3. **If map is huge:**
   - Only include current region (3-hop radius)
   - Text summary instead of full Mermaid for very large maps

## Testing Strategy

### Unit Tests

**Test File:** `tests/test_objective_manager_enhanced.py`

```python
class TestObjectiveManagerEnhanced:
    def test_get_full_knowledge_loads_file(self):
        """Should load knowledge base from file"""

    def test_get_full_knowledge_handles_missing_file(self):
        """Should return fallback message if file missing"""

    def test_get_adjacent_room_ids_outgoing(self):
        """Should find rooms reachable from current location"""

    def test_get_adjacent_room_ids_incoming(self):
        """Should find rooms that can reach current location"""

    def test_get_adjacent_room_ids_no_map_manager(self):
        """Should handle missing map_manager gracefully"""

    def test_get_memories_with_adjacent_current_only(self):
        """Should format memories for current location"""

    def test_get_memories_with_adjacent_includes_neighbors(self):
        """Should include memories from adjacent locations"""

    def test_get_memories_with_adjacent_limits_count(self):
        """Should limit to 5 adjacent locations"""

    def test_get_map_context_includes_mermaid(self):
        """Should include Mermaid diagram in context"""

    def test_get_routing_summary_current_location(self):
        """Should show exits from current location"""

    def test_get_routing_summary_adjacent_locations(self):
        """Should show connections for adjacent rooms"""

    def test_format_memories_shows_status(self):
        """Should show status markers for non-ACTIVE memories"""
```

### Integration Tests

**Test File:** `tests/test_objective_integration.py`

```python
class TestObjectiveManagerIntegration:
    def test_objectives_use_knowledge_dangers(self):
        """Objectives should reference known dangers from knowledge"""

    def test_objectives_use_memory_procedures(self):
        """Objectives should leverage procedures from memories"""

    def test_objectives_reference_location_ids(self):
        """Objectives should use location IDs not just names"""

    def test_objectives_suggest_adjacent_exploration(self):
        """Objectives should suggest exploring adjacent rooms with memories"""

    def test_full_workflow_with_enhanced_context(self):
        """Full objective update with knowledge + memories + map"""
```

### Manual Testing Checklist

- [ ] Objective update includes knowledge base content
- [ ] Objectives reference specific memories by location ID
- [ ] Objectives suggest navigation using map structure
- [ ] Token count is within acceptable range (~10K-15K)
- [ ] LLM generates objectives that align with context
- [ ] Objectives avoid known dangers from knowledge
- [ ] Objectives leverage procedures from memories
- [ ] Location IDs in objectives match memories format

## Migration Path

### Phase 1: Add Helper Methods (Low Risk)
- Add `_get_full_knowledge()`
- Add `_get_adjacent_room_ids()`
- Add `_get_memories_with_adjacent()`
- Add `_get_map_context()`
- Add `_get_routing_summary()`
- Add `_format_memories()`
- **Risk**: Low (new methods, no existing code changes)

### Phase 2: Update __init__ (Medium Risk)
- Add `map_manager` parameter
- Add `memory_manager` parameter
- Update orchestrator initialization
- **Risk**: Medium (dependency injection changes)

### Phase 3: Update Prompt (High Impact)
- Integrate new context sections in prompt
- Update examples and instructions
- Test with real LLM calls
- **Risk**: High (changes objective generation behavior)

### Rollback Plan
- Keep old prompt in comments for quick revert
- Feature flag: `config.enable_enhanced_objectives` (default: False)
- Monitor objective quality metrics before full rollout

## Success Criteria

### Quantitative Metrics
- [ ] Objectives reference location IDs (>80% of objectives)
- [ ] Objectives cite knowledge or memories (>60% of objectives)
- [ ] Token usage per update: 10K-15K (within budget)
- [ ] Objective discovery time: < 30 seconds
- [ ] Test coverage: >85% for new methods

### Qualitative Metrics
- [ ] Objectives are more specific and actionable
- [ ] Objectives demonstrate cross-episode learning
- [ ] Objectives avoid repeating known failures
- [ ] Objectives leverage spatial understanding
- [ ] Human review: objectives are "better" than before

## Future Enhancements

### Short-term
1. **Objective Quality Scoring**: Rate objectives based on specificity, knowledge use
2. **Memory Recency Weighting**: Prioritize recent episode memories
3. **Category Filtering**: Only include SUCCESS/DISCOVERY memories for objectives

### Long-term
1. **Dynamic Context Sizing**: Adjust knowledge/memory inclusion based on token budget
2. **Objective Templates**: Pre-defined templates for common patterns (avoid danger, use procedure, explore area)
3. **Multi-Objective Planning**: Coordinate objectives with dependencies (get item X before area Y)
4. **Knowledge Indexing**: Semantic search for relevant knowledge sections instead of full file

## References

- **Architecture Doc**: `/Volumes/workingfolder/ZorkGPT/CLAUDE.md`
- **Manager Patterns**: `/Volumes/workingfolder/ZorkGPT/managers/CLAUDE.md`
- **Memory System**: `/Volumes/workingfolder/ZorkGPT/managers/simple_memory_manager.py`
- **Map Graph**: `/Volumes/workingfolder/ZorkGPT/map_graph.py`
- **Mermaid Update**: Location IDs now use `L{id}` format (consistent with memories)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-04
**Status**: Specification (Ready for Implementation)
