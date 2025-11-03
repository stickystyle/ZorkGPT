# Knowledge System

This guide covers the knowledge base structure, cross-episode learning, and strategic wisdom accumulation in ZorkGPT.

## Knowledge Base Structure

The ZorkGPT knowledge base (`game_files/knowledgebase.md`) is a consolidated document that accumulates strategic wisdom across episodes. It serves as the agent's long-term memory.

### Strategic Sections (Updated During Gameplay)

These sections are updated in real-time as the agent plays:

#### DANGERS & THREATS
Specific dangers and recognition patterns:
```markdown
**Grue Attacks**: In dark locations without light source, grue attacks after 2-3 turns
- Recognition: "It is pitch black. You are likely to be eaten by a grue."
- Prevention: Always carry lamp. Light lamp in dark areas immediately.
```

#### PUZZLE SOLUTIONS
Puzzle mechanics and solutions discovered:
```markdown
**Trap Door Puzzle**: Requires rope to descend safely
- Location: Living Room
- Items needed: rope (found in attic)
- Solution: tie rope to railing, then descend with rope
```

#### STRATEGIC PATTERNS
Successful/failed approaches and patterns:
```markdown
**Exploration Pattern**: Systematic room-by-room exploration more effective than random wandering
- Best practice: Map each room's exits before choosing direction
- Failed approach: Rushing through without examining objects
```

#### DEATH & DANGER ANALYSIS
Death event analysis and prevention strategies:
```markdown
**Death: Troll Attack** (Episode 1, Turn 45)
- Cause: Attempted to pass troll without proper weapon/strategy
- Prevention: Need sword or other combat item
- Alternative: Find way around troll
```

#### COMMAND SYNTAX
Exact commands that worked (important for parser):
```markdown
Successful: "open window", "enter window"
Failed: "go through window", "climb window"
→ Parser prefers "enter" verb for window entry
```

#### LESSONS LEARNED
Session-specific insights:
```markdown
**Episode 3 Learning**: Objects in dark rooms can't be examined until room is lit
- Impact: Must light lamp before taking inventory in dark areas
- Related: Some objects only appear in description after lighting
```

### CROSS-EPISODE INSIGHTS (Updated at Episode Completion)

This section synthesizes persistent wisdom that carries across multiple episodes. **This is the most valuable knowledge** because it represents validated patterns from multiple attempts.

Updated via `synthesize_inter_episode_wisdom()` at episode end when:
- Episode ended in death (critical learning event), OR
- Final score >= 50, OR
- Turn count >= 100, OR
- Average critic score >= 0.3

#### Death Patterns Across Episodes
Consistent death causes and prevention strategies validated across multiple episodes:
```markdown
**Grue Death Pattern** (Episodes 1, 2, 5):
- Consistent trigger: Being in dark location > 2 turns without light
- Prevention: 100% success rate when lamp carried and lit immediately
- Never attempt dark navigation without light source
```

#### Environmental Knowledge
Persistent facts about game world discovered and confirmed:
```markdown
**West of House → Kitchen Window Entry**:
- Confirmed in Episodes 1, 3, 4
- Reliable alternative to front door
- Procedure: (1) open window, (2) enter window
- Always works regardless of other game state
```

#### Strategic Meta-Patterns
Approaches that prove consistently effective/ineffective across different situations:
```markdown
**Meta-Pattern: Examine Before Take**:
- Success rate: 87% (23/26 episodes)
- Pattern: Examining objects first reveals takeable state, preventing wasted actions
- Failed attempts: Usually due to skipping examine step
- Conclusion: Always examine before attempting to take
```

#### Major Discoveries
Game mechanics, hidden areas, puzzle solutions discovered:
```markdown
**Discovery: Object Persistence** (Episode 7):
- Dropped objects remain in location indefinitely
- Can create "caches" of items for later retrieval
- Strategy: Drop excess inventory in safe, memorable locations
```

## Knowledge Manager System

Located in `managers/knowledge_manager.py`. Handles knowledge updates and synthesis.

### Key Responsibilities

1. **Real-time Updates**: Adds insights during gameplay to strategic sections
2. **Episode Synthesis**: Triggers cross-episode wisdom generation at episode end
3. **Knowledge Retrieval**: Provides relevant knowledge to agent based on context
4. **Deduplication**: Prevents redundant entries

### Integration with Other Systems

**SimpleMemoryManager** (`managers/simple_memory_manager.py`):
- Records location-specific memories (e.g., "At Behind House, enter window leads to Kitchen")
- Memories are tactical (what works at specific location)
- Feeds into Knowledge Manager for strategic synthesis

**KnowledgeManager** (`managers/knowledge_manager.py`):
- Synthesizes strategic patterns from multiple memories
- Updates knowledgebase.md with cross-episode insights
- Knowledge is strategic (what works generally)

**Flow:**
```
Game Action → Memory Recorded (location-specific) → Knowledge Synthesized (general pattern) → Cross-Episode Wisdom (validated pattern)
```

## Knowledge File Management

### Location
- Primary knowledgebase: `game_files/knowledgebase.md`
- Memory file (location-specific): `game_files/memories.md`
- Configuration in `pyproject.toml`:
  ```toml
  [tool.zorkgpt.paths]
  knowledgebase_file = "game_files/knowledgebase.md"
  memory_file = "game_files/memories.md"
  ```

### File Format

**knowledgebase.md** structure:
```markdown
# ZorkGPT Knowledge Base

## DANGERS & THREATS
[Dynamic content updated during gameplay]

## PUZZLE SOLUTIONS
[Dynamic content updated during gameplay]

## STRATEGIC PATTERNS
[Dynamic content updated during gameplay]

## DEATH & DANGER ANALYSIS
[Dynamic content updated during gameplay]

## COMMAND SYNTAX
[Dynamic content updated during gameplay]

## LESSONS LEARNED
[Dynamic content updated during gameplay]

## CROSS-EPISODE INSIGHTS
[Updated at episode completion when synthesis criteria met]

### Death Patterns Across Episodes
[Validated patterns from multiple episodes]

### Environmental Knowledge
[Confirmed facts about game world]

### Strategic Meta-Patterns
[Effective/ineffective approaches across situations]

### Major Discoveries
[Game mechanics, hidden areas, puzzle solutions]
```

### Synthesis Triggers

Cross-episode synthesis occurs when **any** of these conditions are met:

```python
should_synthesize = (
    episode_ended_in_death or
    final_score >= 50 or
    turn_count >= 100 or
    average_critic_score >= 0.3
)
```

**Rationale:**
- Death: Critical learning opportunity
- High score: Significant progress made
- Long episode: Enough data for pattern detection
- High critic scores: Quality actions suggest good strategies

## Working with Knowledge

### Adding Strategic Insights

Through KnowledgeManager:
```python
knowledge_manager.add_insight(
    category="DANGERS & THREATS",
    insight="Grue attacks in dark locations after 2-3 turns. Always carry lamp."
)
```

### Querying Knowledge

```python
# Get relevant knowledge for current context
relevant_knowledge = knowledge_manager.get_relevant_knowledge(
    current_location="Dark Cave",
    inventory=["lamp", "sword"],
    objectives=["Explore underground"]
)
```

### Cross-Episode Wisdom Synthesis

Automatically triggered at episode end:
```python
# In episode_synthesizer.py
if should_synthesize_wisdom:
    knowledge_manager.synthesize_inter_episode_wisdom(
        episode_summary=summary,
        death_occurred=died,
        final_score=score,
        turn_count=turns
    )
```

## Best Practices

### When Writing Knowledge

**Do:**
- Be specific and actionable: "Examine window, then open window, then enter window"
- Include location context: "At Behind House, ..."
- Note failure cases: "Tried X, failed because Y"
- Reference episode numbers: "Confirmed in Episodes 1, 3, 5"
- Use concrete examples: "Grue attacks after 2-3 turns" not "Grue might attack"

**Don't:**
- Write vague generalities: "Be careful in dark places"
- Duplicate information across sections
- Include unconfirmed theories in CROSS-EPISODE INSIGHTS
- Mix tactical (location-specific) with strategic (general) knowledge

### Knowledge vs Memory

**Use Memory** (`memories.md`) for:
- Location-specific procedures: "At Behind House, 'enter window' works"
- Single-episode tactical insights
- Specific item locations: "Lamp found in Living Room"
- Action-response pairs

**Use Knowledge** (`knowledgebase.md`) for:
- General strategies: "Always examine objects before taking"
- Game mechanics: "Dropped items persist in location"
- Parser patterns: "Parser prefers 'enter' over 'climb' for windows"
- Cross-episode validated patterns

## Debugging Knowledge Issues

### Common Problems

**Knowledge not being used:**
- Check if knowledge is in correct section
- Verify context manager includes knowledge in prompts
- Check if knowledge is too vague to be actionable

**Duplicate entries:**
- KnowledgeManager should deduplicate, check dedup logic
- May need to manually clean knowledge base between major refactors

**Stale knowledge:**
- After game mechanics changes, consider resetting knowledge base
- Use version markers in CROSS-EPISODE INSIGHTS to track validity

### Resetting Knowledge

```bash
# Reset everything (fresh start)
rm game_files/knowledgebase.md game_files/memories.md

# Keep strategic sections, reset cross-episode (after mechanics changes)
# Manually edit knowledgebase.md to remove CROSS-EPISODE INSIGHTS section
```

## Integration Points

### Orchestrator
- Calls `knowledge_manager.add_insight()` for death events
- Calls `knowledge_manager.process_turn()` each turn
- Triggers cross-episode synthesis at episode end

### Context Manager
- Includes relevant knowledge in agent prompts
- Formats knowledge for LLM consumption
- Filters knowledge by context (location, objectives, inventory)

### Episode Synthesizer
- Determines when to trigger cross-episode synthesis
- Provides episode summary for synthesis
- Coordinates with knowledge_manager

## Related Files

- `managers/knowledge_manager.py` - Knowledge management implementation
- `managers/simple_memory_manager.py` - Location-specific memory system
- `managers/episode_synthesizer.py` - Episode lifecycle and synthesis
- `game_files/knowledgebase.md` - The knowledge base file itself
- `game_files/memories.md` - Location-specific memory file
