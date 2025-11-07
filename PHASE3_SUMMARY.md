# Phase 3 Implementation Summary: Enhanced Objective Discovery Context

## Overview
Phase 3 integrates rich context (knowledge, memories, map, gameplay) into the ObjectiveManager's `_update_discovered_objectives()` prompt, enabling the LLM to generate objectives that leverage cross-episode learning and strategic wisdom.

## Changes Made

### 1. Enhanced Prompt Structure (`managers/objective_manager.py`)

**Location**: Lines 206-287

**Changes**:
- Added context gathering calls before prompt creation:
  - `knowledge_content = self._get_full_knowledge()`
  - `memories_content = self._get_all_memories_by_distance(current_room_id)`
  - `map_context = self._get_map_context()`
  - `gameplay_context = self._get_gameplay_context()`

- Replaced old prompt with structured 4-section format:
  ```
  === STRATEGIC KNOWLEDGE (General Wisdom) ===
  === LOCATION-SPECIFIC MEMORIES ===
  === MAP & ROUTING INFORMATION ===
  === RECENT GAMEPLAY ===
  ```

- Enhanced guidance on objective types:
  1. Align with strategic knowledge (avoid dangers, pursue high-value goals)
  2. Leverage location-specific memories (use known procedures)
  3. Address exploration opportunities (unexplored exits)
  4. Build on recent gameplay patterns

- Added concrete examples:
  - ✅ Good: "Use window entry procedure from Location 79 memory (open window → enter window) to access Location 62 (Kitchen)"
  - ❌ Bad: "Explore the house" (too vague, no location IDs)

### 2. Token Count Logging (`managers/objective_manager.py`)

**Location**: Lines 212-238

**Added**:
- Token estimation for each context section (knowledge, memories, map, gameplay)
- Debug logging with breakdown
- Structured logger event: `objective_context_prepared` with all token counts
- Total context token tracking

**Example output**:
```
Objective context token breakdown: knowledge=2847, memories=1523, map=412, gameplay=658, total=5440
```

### 3. Enhanced Reasoning Logging (`managers/objective_manager.py`)

**Location**: Line 413

**Added**:
- LLM reasoning included in structured log event `objectives_updated`
- Helps understand how LLM aligned objectives with context

### 4. Import Addition (`managers/objective_manager.py`)

**Location**: Line 20

**Added**:
- `from shared_utils import ... estimate_tokens`

### 5. Integration Tests (`tests/test_objective_integration.py`)

**Added**: 7 new tests in `TestObjectiveManagerPhase3EnhancedContext` class

**Tests**:
1. `test_context_gathering_includes_all_sections`: Verifies all 4 context sections return data
2. `test_knowledge_content_loaded`: Knowledge file contents included
3. `test_memories_include_location_ids`: Memories formatted with location IDs
4. `test_map_context_includes_current_location`: Current location highlighted in map
5. `test_gameplay_context_includes_recent_actions`: Recent actions included
6. `test_token_counting_works`: Token estimation functional
7. `test_memory_filtering_active_only`: Only ACTIVE memories included (excludes TENTATIVE, SUPERSEDED)

**Test fixtures**:
- `knowledge_file`: Creates test knowledge file with strategic wisdom
- `map_manager`: Pre-populated with test rooms (180 → 79 → 62)
- `simple_memory`: Pre-populated with ACTIVE memories at locations 79 and 62
- `objective_manager`: Fully wired with all dependencies

## Test Results

### All Tests Pass ✅

```bash
# Phase 3 tests: 7/7 passed
uv run pytest tests/test_objective_integration.py::TestObjectiveManagerPhase3EnhancedContext -v
# Result: 7 passed in 0.47s

# All objective integration tests: 14/14 passed
uv run pytest tests/test_objective_integration.py -v
# Result: 14 passed in 0.44s

# All objective-related tests: 49/49 passed
uv run pytest tests/ -k "objective" -v
# Result: 49 passed, 725 deselected in 0.62s
```

**No regressions**: All existing tests continue to pass.

## Token Budget Analysis

### Estimated Token Usage Per Objective Update

Based on test data and specification estimates:

| Component | Estimated Tokens | Source |
|-----------|-----------------|--------|
| Knowledge Base | 2,000 - 5,000 | Full knowledgebase.md file |
| All Memories (Distance-Sorted) | 3,000 - 6,000 | ~25 locations × 3-5 ACTIVE memories each |
| Map Mermaid | 200 - 400 | Visual diagram with location IDs |
| Routing Summary | 200 - 400 | Current location connections |
| Recent Gameplay | 500 - 800 | Last 10 actions |
| System Prompt | 800 - 1,200 | Instructions and examples |
| **TOTAL CONTEXT** | **~12K - 18K tokens** | Per objective update (every 20 turns) |

**Why this is acceptable**:
- Updates only every 20 turns (not every turn)
- Strategic decisions require comprehensive context
- Filtering reduces noise (ACTIVE only)
- Distance sorting provides natural priority ordering
- Enables cross-episode learning payoff

## Example Context Sections

### Strategic Knowledge
```markdown
# Strategic Knowledge

## Dangers
- **Troll at Location 152**: Blocks passage. Requires sword or lunch offering.

## Priorities
- Light source (lantern) is critical for dark areas
- Treasures increase score

## Procedures
- Window entry: open before enter
```

### Location-Specific Memories
```markdown
## All Game Memories (Sorted by Distance from Current Location)

**Location 79 (Behind House) - 1 hops away:**
  - [SUCCESS] Window entry procedure
    To enter kitchen: (1) open window, (2) enter window. Window must be opened first.

**Location 62 (Kitchen) - 2 hops away:**
  - [DISCOVERY] Kitchen has food
    Kitchen contains sack of lunch. Could be useful for troll.
```

### Map & Routing Information
```markdown
## Map Visualization (Mermaid Format)
graph TD
  L180["180<br/>West of House"]
  L79["79<br/>Behind House"]
  L62["62<br/>Kitchen"]
  L180 -->|north| L79
  L79 -->|enter window| L62

Note: Node IDs match location IDs in memories (L180 = Location 180)
**Current location**: L180

## Current Location: 180 (West of House)
**Available Exits:**
  - north → Location 79 (Behind House)
```

### Recent Gameplay
```markdown
## Recent Actions (Last 10 Turns)

Turn 18:
  Action: north
  Response: Behind House\nYou are behind the white house.

Turn 19:
  Action: enter window
  Response: You can't reach the window.

Turn 20:
  Action: open window
  Response: With some effort, you open the window.
```

## Expected Objective Quality Improvements

### Before Phase 3 (Limited Context)
```json
{
  "objectives": [
    "Find a light source to explore dark areas safely",
    "Acquire the brass lantern from the Living Room",
    "Search for valuable treasures to increase score"
  ],
  "reasoning": "Agent has discovered the importance of light from game feedback."
}
```

**Issues**:
- No location IDs
- No reference to memories or knowledge
- Generic, could be generated without any context

### After Phase 3 (Rich Context)
```json
{
  "objectives": [
    "Use window entry procedure from Location 79 memory (open window → enter window) to access Location 62 (Kitchen)",
    "Retrieve sack of lunch from Location 62 (Kitchen) per memory - useful for troll encounter at Location 152",
    "Avoid troll at Location 152 until acquiring lunch (knowledge warns: requires sword or food offering)"
  ],
  "reasoning": "Objectives leverage window procedure memory (cross-episode learning), retrieve known item from adjacent location (map + memory combination), and avoid known danger per strategic knowledge. All objectives reference specific location IDs for clarity."
}
```

**Improvements**:
- ✅ Specific location IDs (79, 62, 152)
- ✅ References specific memories ("window entry procedure")
- ✅ References knowledge base ("troll warning")
- ✅ Builds multi-step plans (get lunch → use against troll)
- ✅ Leverages cross-episode learning

## Edge Cases Handled

1. **Empty knowledge file**: Returns "No strategic knowledge available."
2. **No memories available**: Returns "No memory data available"
3. **Map not available**: Returns "No map data available"
4. **Empty action history**: Shows "(No actions yet - start of episode)"
5. **JSON parsing errors**: Logs error and continues (existing error handling)

## Monitoring and Debugging

### Debug Logs
- Token breakdown per context section
- Total context size
- Reasoning from LLM response

### Structured Logs
**Event**: `objective_context_prepared`
```json
{
  "event_type": "objective_context_prepared",
  "episode_id": "ep_001",
  "turn": 20,
  "knowledge_tokens": 2847,
  "memories_tokens": 1523,
  "map_tokens": 412,
  "gameplay_tokens": 658,
  "total_context_tokens": 5440
}
```

**Event**: `objectives_updated`
```json
{
  "event_type": "objectives_updated",
  "episode_id": "ep_001",
  "turn": 20,
  "objective_count": 3,
  "objectives": ["...", "...", "..."],
  "reasoning": "Objectives leverage window procedure memory..."
}
```

## Next Steps (Not in Phase 3)

### Potential Future Enhancements
1. **Token budget monitoring**: Add alerts if context exceeds 20K tokens
2. **Dynamic context pruning**: If token budget exceeded, prune distant memories or summarize knowledge
3. **Objective quality metrics**: Track how often objectives reference location IDs, knowledge, or memories
4. **A/B testing**: Compare objective quality before/after Phase 3 in real episodes

## Success Criteria Met ✅

- [x] Prompt includes all 4 context sections (knowledge, memories, map, gameplay)
- [x] JSON response parsing works correctly (reasoning field included)
- [x] All integration tests pass (7 new Phase 3 tests)
- [x] Token usage tracked and logged
- [x] Objectives expected to be more specific and reference context
- [x] No regressions in existing functionality (49 objective tests pass)
- [x] Edge cases handled gracefully (empty files, missing dependencies)
- [x] Backward compatible (optional dependencies, graceful degradation)

## Files Modified

1. `/Volumes/workingfolder/ZorkGPT-objective_manager/managers/objective_manager.py`
   - Lines 20: Added `estimate_tokens` import
   - Lines 206-287: Replaced prompt with enhanced 4-section structure
   - Lines 212-238: Added token counting and logging
   - Line 413: Added reasoning to structured log

2. `/Volumes/workingfolder/ZorkGPT-objective_manager/tests/test_objective_integration.py`
   - Lines 277-493: Added `TestObjectiveManagerPhase3EnhancedContext` class with 7 tests
   - Added fixtures: `knowledge_file`, enhanced `game_state`, enhanced `map_manager`, enhanced `simple_memory`

## Conclusion

Phase 3 successfully integrates rich context into objective discovery, enabling the LLM to generate objectives that:
- Reference specific location IDs
- Leverage cross-episode knowledge from the knowledge base
- Use procedural memories from past successful actions
- Avoid known dangers
- Pursue high-value exploration opportunities

All tests pass with no regressions. The implementation is backward compatible and handles edge cases gracefully.
