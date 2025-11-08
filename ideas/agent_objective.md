# Agent Self-Directed Objectives via Structured Output

## Problem Statement

The agent currently cannot directly control its objective system. When it wants to perform multi-step actions, it relies on:
- Parsing previous turns' reasoning text (prone to confusion over multiple turns)
- ObjectiveManager's periodic LLM-driven discovery (every N turns, adds latency)
- No explicit mechanism for agent to say "I want to track this multi-step goal"

This creates a gap between agent intent and objective tracking.

## Proposed Solution

Enable the agent to directly declare objectives by migrating to **structured JSON output** with an optional `new_objective` field.

### Output Schema

```python
from pydantic import BaseModel, Field
from typing import Optional

class AgentResponse(BaseModel):
    thinking: str = Field(
        description="Your reasoning - what you observe, plan, and why"
    )
    action: str = Field(
        description="Single game command to execute"
    )
    new_objective: Optional[str] = Field(
        default=None,
        description="Optional multi-step objective to track. Only set when starting a new multi-turn plan. Should reference specific locations (e.g., 'get lamp from L124')"
    )
```

### Example JSON Output

```json
{
  "thinking": "I found a locked trophy case. The game emphasized it can hold valuables. This suggests I should collect treasures and place them here. Multi-step plan: explore house, collect treasures, return to trophy case.",
  "action": "west",
  "new_objective": "collect all treasures and bring to trophy case at L5"
}
```

## Design Constraints

### 1. Optional Field (Not Every Turn)
- `new_objective` defaults to `null`
- Agent should only set it when starting a new multi-turn plan
- Prompt must guide agent to use sparingly (not on every action)

### 2. One Active Objective at a Time
- Agent can only declare one objective per turn
- Prevents overwhelming the objective list
- ObjectiveManager still handles deduplication/merging

### 3. Existing Completion Detection Handles Cleanup
- ObjectiveManager's `check_objective_completion()` continues to work unchanged
- Agent doesn't need to manually mark objectives complete
- System handles lifecycle automatically

### 4. Format Validation
- Should reference specific locations when possible (e.g., `L124`)
- Should be actionable and specific
- Invalid objectives can be filtered/validated before adding

## Implementation Plan

### Phase 1: Pydantic Model & Schema (zork_agent.py)

**File:** `zork_agent.py`

**Reference the Pydantic model defined in Output Schema section above.**

**Create JSON schema:**
```python
from shared_utils import create_json_schema

# In get_action_with_reasoning() method
response_format = create_json_schema(AgentResponse)
```

**Update LLM call:**
```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=self.temperature,
    response_format=response_format,  # NEW: structured output
)
```

### Phase 2: Response Parsing (zork_agent.py)

**Replace current regex parsing in `get_action_with_reasoning()` method with:**

```python
def get_action_with_reasoning(self, game_state_text: str, relevant_memories: str) -> Dict[str, Any]:
    # ... build messages ...

    response = self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        temperature=self.temperature,
        response_format=create_json_schema(AgentResponse),
    )

    raw_response = response.choices[0].message.content

    # Parse structured JSON (guaranteed valid by response_format)
    try:
        agent_response = AgentResponse.model_validate_json(raw_response)
    except Exception as e:
        # Fallback to safe defaults
        logger.error(f"Failed to parse agent response: {e}")
        agent_response = AgentResponse(
            thinking="[Error parsing response]",
            action="look",
            new_objective=None
        )

    # Clean and validate action
    # Note: Extract existing inline cleaning logic (lines 307-337 in current zork_agent.py)
    # into _clean_action() method, or simplify to basic validation since JSON structure
    # already separates action from thinking (lowercase, strip, non-empty check)
    cleaned_action = self._clean_action(agent_response.action)

    return {
        "action": cleaned_action,
        "reasoning": agent_response.thinking,
        "new_objective": agent_response.new_objective,
        "raw_response": raw_response,
    }
```

### Phase 3: Orchestrator Integration (zork_orchestrator_v2.py)

**File:** `orchestration/zork_orchestrator_v2.py`

**Extract objective after agent call in main game loop:**

```python
# Current code
proposed_action = agent_result["action"]
agent_reasoning = agent_result.get("reasoning", "")

# Add reasoning to context
self.context_manager.add_reasoning(agent_reasoning, proposed_action)

# NEW: Extract and add agent-declared objective
new_objective = agent_result.get("new_objective")
if new_objective:
    self.objective_manager.add_agent_objective(new_objective)
    logger.info(f"Agent declared new objective: {new_objective}")
```

### Phase 4: ObjectiveManager Method (managers/objective_manager.py)

**File:** `managers/objective_manager.py`

**Add new method:**

```python
def add_agent_objective(self, objective_text: str) -> None:
    """
    Add an objective directly declared by the agent during reasoning.

    This provides a fast path for agent self-direction without waiting
    for periodic LLM-driven discovery.

    Args:
        objective_text: The objective description from agent response
    """
    # Basic validation
    if not objective_text or len(objective_text.strip()) == 0:
        self.log_warning("Agent provided empty objective, ignoring")
        return

    # Check if already exists (case-insensitive deduplication)
    for existing_obj in self.game_state.discovered_objectives:
        if existing_obj.lower() == objective_text.lower():
            self.log_info(f"Objective already exists: {objective_text}")
            return

    # Add to objectives list (simple string)
    self.game_state.discovered_objectives.append(objective_text)
    self.log_info(f"Added agent-declared objective: {objective_text}")
```

### Phase 5: Prompt Updates (game_files/agent.md)

**File:** `game_files/agent.md`

**Replace OUTPUT FORMAT section with:**

```markdown
**OUTPUT FORMAT (REQUIRED):**

You must respond with valid JSON containing three fields:

```json
{
  "thinking": "Your reasoning - what you observe, plan, and why",
  "action": "single_command_here",
  "new_objective": null
}
```

**Field Descriptions:**

- **thinking**: Your reasoning following the thinking guidelines above
  - Keep concise (2-3 sentences) for standard exploration/navigation
  - Expand to full paragraph for puzzles, dangerous situations, or strategic decisions
  - Avoid redundancy - don't repeat what's obvious from the action
- **action**: A single game command (one direction, or comma-separated non-movement actions)
- **new_objective**: (Optional) Set ONLY when you want to track a new multi-step goal
  - Example: "collect all treasures and bring to trophy case at L5"
  - Should reference specific locations when possible (e.g., "get lamp from L124")
  - Leave as `null` for most turns - only use when starting a multi-turn plan
  - Do NOT set this every turn - objectives persist until completed
```

**Add guidance section:**

```markdown
**WHEN TO SET new_objective:**

Set `new_objective` when:
1. You discover a multi-step puzzle or goal that will take several turns
2. You want to track progress toward a specific achievement
3. You're starting a collection/gathering task

Do NOT set `new_objective` when:
- Taking a single exploratory action
- Continuing an already-declared objective
- The action is self-contained (no follow-up needed)

**Concrete Examples:**

✅ GOOD - Multi-step procedure discovered:
   - Finding locked door → Set: "find key for brass door at L42"
   - Trophy case emphasized → Set: "collect treasures for trophy case at L5"
   - Multi-room puzzle → Set: "solve water puzzle spanning kitchen and basement"

✅ GOOD - Collection task:
   - Game mentions "treasures" → Set: "collect all treasures"
   - Multiple related items → Set: "gather tools for repair task"

❌ BAD - Single action:
   - Moving one room → Leave null
   - Picking up single item → Leave null (unless part of larger goal)
   - Examining object → Leave null

❌ BAD - Already tracking:
   - Continuing treasure collection → Leave null (objective already exists)
   - Working on existing puzzle → Leave null
```

## Testing Strategy

### Unit Tests

**File:** `tests/test_agent_objectives.py` (new file)

```python
def test_agent_response_parsing_with_objective():
    """Agent can declare objectives via JSON output."""
    # Mock LLM response with new_objective set
    # Verify parsing extracts all three fields

def test_agent_response_parsing_without_objective():
    """Agent responses without objectives parse correctly."""
    # Mock LLM response with new_objective=null
    # Verify parsing handles null gracefully

def test_invalid_agent_response_fallback():
    """Malformed responses fall back to safe defaults."""
    # Mock invalid JSON
    # Verify fallback to look command with empty objective
```

**File:** `tests/test_objective_manager.py` (add to existing)

```python
def test_add_agent_objective():
    """ObjectiveManager can add agent-declared objectives."""
    # Call add_agent_objective()
    # Verify objective added to discovered_objectives as string

def test_add_agent_objective_deduplication():
    """Duplicate agent objectives are not added."""
    # Add same objective twice
    # Verify only one exists in list

def test_add_agent_objective_empty_string():
    """Empty objectives are ignored."""
    # Call with empty/whitespace string
    # Verify nothing added
```

### Integration Tests

**File:** `tests/test_orchestrator_objectives.py` (add to existing)

```python
def test_orchestrator_extracts_agent_objectives():
    """Orchestrator correctly extracts and adds agent objectives."""
    # Mock agent returning response with new_objective
    # Run one turn
    # Verify objective added to ObjectiveManager

def test_agent_objective_completion_workflow():
    """Agent-declared objectives are tracked through completion."""
    # Agent declares objective
    # Subsequent turns work toward objective
    # Objective marked complete when achieved
    # Verify full lifecycle
```

### Test Updates Required

All existing tests that mock agent responses need updating:

**Pattern to find:**
```bash
grep -r "agent_result\|get_action_with_reasoning" tests/
```

**Update strategy:**
1. Mock responses now return `new_objective: None` in addition to action/reasoning
2. Tests checking response format need to expect JSON structure
3. Fixture data in `tests/fixtures/` may need updates

## Migration Checklist

- [ ] Add `AgentResponse` Pydantic model to `zork_agent.py`
- [ ] Update `get_action_with_reasoning()` to use structured output
- [ ] Replace regex parsing with Pydantic model parsing
- [ ] Update orchestrator to extract `new_objective` field
- [ ] Add `add_agent_objective()` method to ObjectiveManager
- [ ] Update `agent.md` OUTPUT FORMAT section
- [ ] Add guidance on when to use `new_objective`
- [ ] Write unit tests for agent response parsing
- [ ] Write unit tests for ObjectiveManager objective intake
- [ ] Write integration tests for full workflow
- [ ] Update all existing test mocks to include `new_objective` field
- [ ] Run full test suite and fix any failures
- [ ] Test with actual gameplay episode
- [ ] Monitor logs for agent objective declarations

## Benefits

1. **Immediate self-direction**: Agent can declare objectives without waiting for periodic discovery
2. **Explicit intent**: Clear signal when agent is starting multi-step plans
3. **Type safety**: Pydantic validation ensures schema compliance
4. **Consistent pattern**: Matches existing structured output usage (Critic, Extractor, ObjectiveManager)
5. **Simplified workflow**: Direct string-based objectives match existing system architecture

## Risks & Mitigations

### Risk: Agent overuses objectives
**Mitigation:**
- Clear prompt guidance on when to use
- Only allow one objective per turn
- Monitor logs and tune prompt if needed

### Risk: Breaking existing tests
**Mitigation:**
- Comprehensive test update checklist
- Run tests after each phase
- Keep backward compatibility during migration

### Risk: Agent declares vague objectives
**Mitigation:**
- Clear prompt guidance on specificity and location references
- Prompt emphasizes including location IDs when possible (e.g., "at L42")
- Simple deduplication via case-insensitive string matching prevents duplicate vague objectives

## Future Enhancements

### 1. Objective Priority Control
Allow agent to set priority:
```python
new_objective: Optional[Dict[str, str]] = Field(
    default=None,
    description="Optional objective with priority",
    example={"text": "get lamp", "priority": "high"}
)
```

### 2. Parent Objective Tracking
Allow agent to declare sub-objectives:
```python
new_objective: Optional[Dict[str, str]] = Field(
    default=None,
    example={"text": "unlock door", "parent": "explore house"}
)
```

### 3. Objective Completion Signals
Allow agent to signal objective completion:
```python
class AgentResponse(BaseModel):
    # ...
    complete_objective: Optional[str] = None  # ID of objective to mark complete
```

## References

- Current agent parsing: `zork_agent.py` in `get_action_with_reasoning()` method
- Current orchestrator integration: `zork_orchestrator_v2.py` in main game loop
- ObjectiveManager discovery: `managers/objective_manager.py` in `check_and_update_objectives()` and related methods
- Existing Pydantic patterns: `zork_critic.py`, `hybrid_zork_extractor.py`
- JSON schema utility: `shared_utils.py` in `create_json_schema()` function
