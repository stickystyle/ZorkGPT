# ZorkGPT Thoughtbox: Structured Cognitive Workspace for Puzzle Solving

**Status:** Proposal
**Created:** 2025-01-11
**Author:** Ryan Parrish & Claude
**Inspired by:** [Design Patterns in MCP: Thoughtboxes](https://medium.com/@glassBead) by glassBead

## Executive Summary

Implement a "thoughtbox" tool that provides structured, addressable workspace for the ZorkGPT agent's reasoning process. This moves beyond linear token-stream thinking to enable explicit iteration, revision, and branching during puzzle solving—similar to using a storyboard app vs verbal corrections.

**Key Benefit:** Enables the agent to explore alternatives, revise incorrect assumptions, and use backward chaining for puzzle solving with explicit addressability and observability—all BEFORE committing to actions.

## Core Concept: Thoughtboxes

A thoughtbox is structured workspace for LLM cognition with four defining characteristics:

### 1. Architectural Structure
Organized patterns for thought (numbered sequences, branches, hierarchies) rather than blank space. Thoughts have stable identifiers that can be referenced later.

### 2. Cognitive Affordances
Design choices that make specific reasoning modes more natural:
- **Branching** for hypothesis exploration
- **Revision** for iterative refinement
- **Backward chaining** for goal-driven planning
- **Forward chaining** for exploratory reasoning

### 3. Process Orientation
Scaffolds the **process** of cognition, not just outputs. The value is in how it shapes thinking, not what it produces. The agent works in a navigable workspace during reasoning.

### 4. Addressability
Provides stable references enabling non-linear navigation, revision, and composition. Thoughts can reference other thoughts by number, create branches, and revise earlier conclusions without linear verbal corrections.

## Why ZorkGPT Needs This

### Current State: Output-Oriented
ZorkGPT has thoughtbox-like elements but they're all POST-PROCESS:
- ✅ Memories have location-based addressability
- ✅ Memory supersession provides revision
- ✅ Reasoning history tracks action rationale across turns

**Gap:** The agent produces final results in one shot. No navigable workspace during thinking.

### Problem: Linear Token Stream Thinking
Current agent reasoning is like "thinking out loud to a friend":
```
Agent: "I should try giving the troll a gift. Wait, no, maybe I should fight it.
Actually, let me reconsider the gift approach because I have food. But wait,
what if the gift makes it hostile? Hmm, let me think about this differently..."
```

Expensive to iterate, hard to track alternatives, no way to reference earlier thoughts.

### Solution: Addressable Cognitive Space
With thoughtbox, same reasoning becomes:
```
Thought 1: Goal: Get past troll at Location 152
Thought 2: Option A - Fight (need weapon)
Thought 3: Option B - Bribe (have food in inventory)
Thought 4: Checking context... memory for Location 152 shows relevant info
Thought 5 (revises 3): Memory shows "Troll attacks after accepting gift"
Thought 6 (branch from 2): Checking inventory in context... no weapons available
Thought 7: Fighting not viable, gifting dangerous
Thought 8: Need alternative - search for puzzle clues
```

Structured iteration, alternatives are explicit, revision is addressable.

## Why Tool-Based vs Better Prompting?

**Alternative approach:** Instead of a thoughtbox tool, just prompt the agent to use structured reasoning:
```
"Before acting, consider your options:
- Option A: [analyze]
- Option B: [analyze]
- Option C: [analyze]

Select the best option and explain why."
```

**This is a valid alternative.** So why use a tool-based approach?

### 1. Addressability & Explicit Revision

**With prompting:**
```
"I think option A is best. Wait, actually option B is better because...
No, scratch that, option A handles the edge case..."
```
Revisions are verbal corrections layered on each other.

**With thoughtbox:**
```
Thought 1: Option A looks promising
Thought 2: Option B handles X better
Thought 3 (revises 1): Option A fails on edge case Y
Thought 4: Therefore choose option B
```
Each revision explicitly references what it's revising. Clear audit trail.

### 2. Observable Thinking Process

**With prompting:** Reasoning is embedded in the final response text. Hard to parse, visualize, or analyze patterns.

**With thoughtbox:** Each thought is a separate tool call with:
- Timestamp
- Thought number
- Branch ID (if branching)
- Revision metadata
- Structured logs → visualization, debugging, pattern analysis

### 3. Separation of Thinking from Action

**With prompting:** Agent must commit to action in same response as reasoning. If thinking reveals "I need more information," hard to recover.

**With thoughtbox:**
- Thinking phase (exploration, revision)
- Action phase (commit to decision)
Clear separation. Agent can think freely without immediately acting.

### 4. Token Efficiency for Extended Reasoning

**With prompting:** All reasoning must fit in single response (8K tokens max). Long reasoning chains are expensive or impossible.

**With thoughtbox:** Each thought is smaller context. Can chain 15+ thoughts without bloating single response. Better token distribution.

### 5. Backward Chaining Support

**With prompting:** "Start with goal, work backwards to current action" - but agent still generates linearly in text.

**With thoughtbox:** Agent can genuinely start with Thought 1 (goal), add prerequisites sequentially, end with current action. The structure matches the reasoning mode.

**Trade-off:** Tool approach adds complexity (tool-calling loop, iteration limits, session management). Only worth it if we value observability, addressability, and process separation over simplicity.

**Decision:** Implement as tool. If evaluation shows prompting achieves same results, we can deprecate the tool. But the observability and structured logging alone justify the experiment.

## Architecture

### Base Infrastructure (No Reasoning)

The thoughtbox provides structure but performs **no reasoning**:

```python
class ThinkingStep(BaseModel):
    """One step in the thinking process."""
    thought: str                        # The actual thinking content
    thoughtNumber: int                  # Addressable ID (1-N)
    totalThoughts: int                  # Current estimate (can adjust)
    nextThoughtNeeded: bool             # Continue or finalize?
    isRevision: bool = False            # Is this revising earlier thought?
    revisesThought: Optional[int] = None  # Which thought is being revised?
    branchFromThought: Optional[int] = None  # Branching point
    branchId: Optional[str] = None      # Branch identifier
    needsMoreThoughts: bool = False     # Expand total estimate?
```

**Critical:** The tool only provides scaffolding. All intelligence lives client-side (in the LLM).

**Context Preservation:** The tool-calling loop maintains the agent's full context (memories, inventory, map, objectives) throughout the thinking session. Each thought has access to the same rich context provided at turn start. This eliminates the need for separate query tools.

### Domain Affordances (Progressive Disclosure)

When agent invokes thoughtbox, provide **Zork-specific patterns cookbook**:

```markdown
# Zork Puzzle-Solving Patterns

## Backward Chaining (Goal-Driven)
State the goal in early thoughts, then reason backwards through prerequisites to current action.
Thoughts are numbered sequentially (1, 2, 3...), but reasoning flows from goal → prerequisites → current action.
Use when: Locked areas, puzzles with known goals, item dependencies
Pattern: Thought 1 (goal) → Thought 2 (what's needed) → ... → Thought N (current action)

## Branch Exploration (Hypothesis Testing)
Create parallel branches to explore alternatives before committing.
Use when: Uncertain NPC behavior, multiple puzzle solutions, risk assessment
Pattern: Branch A (hostile), Branch B (neutral), Branch C (helpful)

## Constraint Tracking
Maintain addressable slots for game constraints across thoughts.
Track: Lantern life, locked doors, required items, location dependencies

## Memory-Referenced Reasoning
Explicitly reference location memories by ID during thinking.
Check context for memories before making assumptions about locations/NPCs.
```

**Progressive Disclosure:** Agent can request cookbook via `includePatternsCookbook` parameter. Recommended: orchestrator sets this to `True` on first thoughtbox call of each session. Keeps agent context lean until needed.

### Tool Definition

```python
{
    "type": "function",
    "function": {
        "name": "zork_thoughtbox",
        "description": (
            "Structured workspace for complex puzzle solving. "
            "Use when facing locked areas, uncertain NPC behavior, "
            "multi-step puzzles, or need to explore alternatives. "
            "Supports forward thinking (exploration), backward thinking "
            "(goal-driven planning), and branching (hypothesis testing)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "thoughtNumber": {"type": "integer", "minimum": 1},
                "totalThoughts": {"type": "integer", "minimum": 1},
                "nextThoughtNeeded": {"type": "boolean"},
                "isRevision": {"type": "boolean"},
                "revisesThought": {"type": "integer", "minimum": 1},
                "branchFromThought": {"type": "integer", "minimum": 1},
                "branchId": {"type": "string"},
                "needsMoreThoughts": {"type": "boolean"},
                "includePatternsCookbook": {"type": "boolean"}
            },
            "required": ["thought", "thoughtNumber", "totalThoughts", "nextThoughtNeeded"]
        }
    }
}
```

### Context Access During Thinking

The agent uses the context already provided in its prompt (via ContextManager) when reasoning with the thoughtbox. This includes:
- Location memories (from SimpleMemoryManager)
- Current inventory (from Z-machine)
- Map knowledge (from MapManager)
- Active objectives (from ObjectiveManager)

**Critical Implementation Detail:** The tool-calling loop MUST preserve the original context in the messages list. Each LLM call during thinking includes:

```python
messages = [
    {"role": "system", "content": "You are ZorkGPT..."},
    {"role": "user", "content": full_context},  # Memories, inventory, map, objectives
    {"role": "assistant", "tool_calls": [...]},  # First thoughtbox call
    {"role": "tool", "content": tool_result_1},
    {"role": "assistant", "tool_calls": [...]},  # Second thoughtbox call
    {"role": "tool", "content": tool_result_2},
    # ... continues
]
```

Every subsequent LLM call sees the full original context plus all tool interactions. This ensures the agent can reference memories, inventory, and map throughout all thoughts without additional query tools.

**No additional query tools are needed** - the agent references this context in its thoughts. All context is assembled at turn start and remains stable throughout the thinking session.

## Integration with Existing Systems

### Relationship to Other Systems

| System | Role | Interaction with Thoughtbox |
|--------|------|----------------------------|
| **Memories** | Knowledge from past actions | Referenced during thinking for context |
| **Objectives** | Goals to achieve | Used as endpoints for backward chaining |
| **Map** | Spatial context | Referenced for navigation planning |
| **Reasoning History** | Turn-by-turn action rationale | Stores normal action reasoning (informed by thinking) |
| **Langfuse** | Observability | Logs full thinking chain for analysis |

### Tool Response & Storage Strategy

**What the tool returns (per Sequential Thinking MCP):**

```python
# Tool returns metadata after each thought
{
    "thoughtNumber": 3,
    "totalThoughts": 5,
    "nextThoughtNeeded": True,
    "branches": [],
    "thoughtHistoryLength": 3
}
```

The tool just echoes back session state. No "summary" is generated - the tool only provides scaffolding.

**How thinking integrates with normal action generation:**

```
1. Agent uses thoughtbox (multiple tool calls)
2. When nextThoughtNeeded=False, thinking session ends
3. Agent generates action + reasoning (normal flow, informed by thinking)
4. Orchestrator stores to action_reasoning_history
```

**What gets stored in action_reasoning_history:**

```python
# Normal action + reasoning (NOT a summary of thoughts)
{
    "turn": 47,
    "reasoning": "Door might be unlocked already, examine before searching for key",
    "action": "examine door",
    "timestamp": "2025-01-11T...",
    "used_thoughtbox": True,  # Metadata flag
    "thinking_steps": 8       # Count of thoughts
}
```

**Key point:** The "reasoning" field is the agent's normal action rationale, NOT a summary of all thoughts. The thinking informed the decision, but the reasoning field works the same as without thoughtbox.

**Full chain logged separately:**
```python
# In Langfuse (for deep analysis)
logger.info(
    "Agent used thoughtbox for complex reasoning",
    extra={
        "event_type": "thoughtbox_session",
        "turn": 47,
        "thinking_chain": [
            {"thought": "...", "number": 1, "type": "forward"},
            {"thought": "...", "number": 2, "type": "branch_A"},
            {"thought": "...", "number": 3, "type": "revision", "revises": 1},
            # ... full chain
        ],
        "final_action": "examine door"
    }
)
```

## Implementation Plan

### Phase 1: Base Infrastructure
1. Extend `LLMClient` to support `tools` parameter
2. Parse `tool_calls` and `finish_reason` in LLM response
3. Create `ToolExecutor` registry class
4. Implement basic `SequentialThinkingTool` (no domain affordances)
5. Add tool calling loop to `ZorkAgent.generate_action()` with:
   - Messages list that preserves original context (system + user context)
   - Append assistant tool_calls and tool results to messages
   - Each LLM call includes full history (context + all tool interactions)
   - Iteration counter (max: `max_thinking_iterations`)
   - Hard stop on limit exceeded
   - Session state tracking (for future tool gating)

**Critical:** The loop must maintain the original context from ContextManager in the messages list so the agent can reference memories, inventory, map, and objectives throughout all thoughts.

### Phase 2: Domain Affordances
1. Create Zork patterns cookbook markdown
2. Implement progressive disclosure (orchestrator sets `includePatternsCookbook=True` on first call)
3. Tool includes cookbook in response when parameter is True
4. Implement thinking modes (forward/backward/branching) in tool description

### Phase 3: Storage & Observability
1. Add `used_thoughtbox` and `thinking_steps` metadata to action_reasoning_history entries
2. Log full thinking chain to Langfuse with structured events
3. Create visualization for thinking chains in post-episode analysis
4. Ensure normal action + reasoning flow works after thoughtbox sessions

### Phase 4: Configuration & Testing

**Configuration:**
1. Add config flag `enable_thoughtbox_tool`
2. Add `max_thinking_iterations` safety limit

**Unit Tests:**
1. Tool schema validation (required fields, type checking)
2. Thought numbering (sequential, adjustable totalThoughts)
3. Branching logic (branchId, branchFromThought)
4. Revision tracking (isRevision, revisesThought)
5. Session termination (nextThoughtNeeded=False)
6. Iteration limit enforcement (MaxIterationsExceeded)

**Integration Tests:**
1. Full thinking sessions with mock LLM responses
2. Tool-calling loop state management
3. Progressive disclosure (orchestrator sets includePatternsCookbook=True on first call)
4. Cookbook inclusion in tool response when requested
5. Metadata storage (used_thoughtbox, thinking_steps)
6. Langfuse logging (full chain capture)

**End-to-End Tests:**
1. Complex puzzle scenarios:
   - Locked Door (backward chaining)
   - Troll Encounter (branching)
   - Multi-step Procedure (revision)
2. Token cost measurement
3. Performance comparison (post-hoc analysis)

## Configuration

```toml
[tool.zorkgpt.agent]
enable_thoughtbox_tool = true          # Enable/disable thoughtbox
max_thinking_iterations = 20           # Maximum tool calls per turn (hard stop)
                                       # Enforced by orchestrator's tool-calling loop
                                       # Prevents infinite thinking loops

[tool.zorkgpt.thoughtbox]
include_patterns_cookbook = true       # Progressive disclosure
patterns_cookbook_path = "game_files/thinking_patterns.md"
log_full_chain = true                  # Log to Langfuse
```

## Example Usage Scenarios

### Scenario 1: Locked Door Puzzle (Backward Chaining)

```
Agent faces locked door at Location 134

Thoughtbox Session:
  Thought 1 (Goal): Enter room behind locked door at Location 134
  Thought 2: Prerequisite - Door must be unlocked first
  Thought 3: Checking memories for Location 134... memory shows "brass key needed"
  Thought 4: Checking inventory... brass key not in inventory
  Thought 5 (Current action): Need to find brass key location

Final Action: "search for brass key"
```

### Scenario 2: Uncertain NPC Behavior (Branching)

```
Agent encounters troll at Location 152

Thoughtbox Session:
  Thought 1: Encountered troll blocking path at Location 152
  Thought 2: Checking memories for Location 152... relevant memory found
  Thought 3 (Branch A): If hostile → avoid or flee
  Thought 4 (Branch B): If neutral → try communication
  Thought 5 (Branch C): If puzzle → search for solution
  Thought 6: Memory shows "Troll attacks after accepting gift"
  Thought 7 (revises Branch B): Communication/gifting fails
  Thought 8: Need puzzle solution or alternate route

Final Action: "look for alternate path"
```

### Scenario 3: Multi-Step Procedure (Forward + Revision)

```
Agent trying to enter kitchen via window

Thoughtbox Session:
  Thought 1: Need to reach kitchen
  Thought 2: At Behind House (Location 79), window visible to north
  Thought 3: Try "enter window" directly
  Thought 4 (revision of 3): Wait, should check if open first
  Thought 5: Better approach: examine → open → enter
  Thought 6: Checking memories for Location 79... no memory of trying this approach
  Thought 7: Execute three-step sequence

Final Action: "examine window"
Memory: Stores multi-step procedure for future reference
```

## Benefits

### For Agent Performance
- ✅ Explore alternatives before committing to risky actions
- ✅ Use backward chaining for goal-oriented puzzle solving
- ✅ Revise incorrect assumptions without penalty
- ✅ Reference existing context explicitly during reasoning
- ✅ Explicit hypothesis testing reduces trial-and-error

### For Observability
- ✅ Full thinking chain logged for post-episode analysis
- ✅ Understand agent's reasoning process, not just final actions
- ✅ Identify where agent gets stuck or makes wrong assumptions
- ✅ Measure impact of different thinking patterns on success

### For Architecture
- ✅ Maintains "all reasoning from LLMs" principle
- ✅ Backward compatible (thoughtbox is optional tool)
- ✅ Separates thinking workspace from action execution
- ✅ Progressive disclosure keeps agent context lean

## Risks & Mitigations

### Risk: Token Cost Increase
**Impact:** Thoughtbox sessions add multiple LLM calls per turn
**Mitigation:**
- Make tool optional (agent decides when to use)
- Progressive disclosure (cookbook not in every context)
- Configuration to disable if cost becomes issue
- Monitor cost per episode and set budget alerts

### Risk: Analysis Paralysis
**Impact:** Agent gets stuck in infinite thinking loop
**Mitigation:**
- Hard limit: `max_thinking_iterations` (default: 20 tool calls per turn)
- Enforced by orchestrator's tool-calling loop
- Raises `MaxIterationsExceeded` exception if limit reached
- Monitor turns where thoughtbox used but no action generated

### Risk: Separation from Game State
**Impact:** Agent reasons about outdated game state during long thinking session
**Mitigation:**
- Context is assembled at turn start and remains stable during thinking
- Thinking session is within single turn (state doesn't change)
- No game actions possible during thinking (enforced separation)

### Risk: Complexity Overhead
**Impact:** Implementation and maintenance burden
**Mitigation:**
- Start with minimal base (Phase 1-2 only)
- No additional query tools needed (uses existing context)
- Keep thoughtbox stateless (no persistence needed)
- Comprehensive testing before domain affordances

## Success Metrics

### Quantitative (Post-Hoc Analysis Approach)

Run episodes with thoughtbox available, then segment results:

**Overall Performance:**
- Episode completion rate (thoughtbox available)
- Token cost per episode (compare to baseline)

**Thoughtbox Usage Patterns:**
- Frequency of use (% of turns where agent invoked thoughtbox)
- Average thinking session length (thoughts per session)
- Thinking mode distribution (forward/backward/branching)

**Effectiveness on Complex Puzzles:**
- Turns to solve known complex puzzles:
  - Loud Room puzzle
  - Troll encounter
  - Locked door sequences
- Success rate on turns where thoughtbox was used
- Compare complexity-matched scenarios (with vs without thoughtbox)

**Why post-hoc analysis:** Avoids selection bias by analyzing real usage patterns. Agent decides when to use thoughtbox, then we segment results by complexity tier for fair comparison.

### Qualitative
- Agent reasoning quality (from thinking chain logs)
- Reduction in trial-and-error actions
- Use of backward chaining for goal-driven planning
- Evidence of hypothesis testing before risky actions

## Open Questions

1. **Should backward thinking be a separate tool?** Could offer `zork_backward_chain` and `zork_forward_think` as distinct tools with different affordances.

2. **How to handle cross-turn thinking?** Current spec is single-turn. Could extend to "persistent thoughtbox" across multiple turns for long-term planning.

3. **Integration with objective system?** Objectives could automatically trigger backward chaining from goal to current state.

4. **Patterns cookbook: static or LLM-generated?** Could use LLM to dynamically suggest relevant patterns based on current situation.

## References

- [Design Patterns in MCP: Thoughtboxes](https://medium.com/@glassBead) by glassBead
- Sequential Thinking MCP Server: https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking
- Thoughtbox MCP Server: https://github.com/kastalien/thoughtbox
- ZorkGPT Architecture: `/Volumes/workingfolder/ZorkGPT/CLAUDE.md`
- Manager Documentation: `/Volumes/workingfolder/ZorkGPT/managers/CLAUDE.md`

## Next Steps

1. Review this spec with stakeholders
2. Prioritize phases (start with Phase 1-2 vs full implementation)
3. Estimate token cost impact via prototype
4. Create detailed implementation tickets
5. Build Phase 1 proof-of-concept

---

**Document Status:** Draft
**Last Updated:** 2025-01-11
**Next Review:** After Phase 1 implementation
