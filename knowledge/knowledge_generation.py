# ABOUTME: LLM-based knowledge generation functions for ZorkGPT knowledge base.
# ABOUTME: Handles direct knowledge generation from turn data and formatting turn data for prompts.

"""
LLM-based knowledge generation module.

This module provides functions for generating knowledge base content using LLMs.
It handles formatting turn data into prompts and generating comprehensive knowledge
base updates that include strategic insights, danger patterns, puzzle solutions,
and lessons learned.
"""

from typing import Dict, Optional
from knowledge.section_utils import extract_cross_episode_section

# Langfuse observe decorator with graceful fallback
try:
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGFUSE_AVAILABLE = False


def format_turn_data_for_prompt(turn_data: Dict) -> str:
    """
    Format turn data for LLM prompt with clear structure.

    Converts structured turn data into a readable format for the LLM, including
    actions, responses, reasoning, critic scores, and events (deaths, score changes,
    location changes).

    Args:
        turn_data: Dictionary containing turn window data with keys:
            - episode_id: Episode identifier
            - start_turn: Starting turn number
            - end_turn: Ending turn number
            - actions_and_responses: List of action/response pairs
            - death_events: List of death events (optional)
            - score_changes: List of score changes (optional)
            - location_changes: List of location changes (optional)

    Returns:
        Formatted string suitable for LLM prompt
    """
    # Header information
    output = f"""EPISODE: {turn_data["episode_id"]}
TURNS: {turn_data["start_turn"]}-{turn_data["end_turn"]}
TOTAL ACTIONS: {len(turn_data["actions_and_responses"])}

"""

    # Gameplay log with truncation for very long responses
    output += "GAMEPLAY LOG:\n"

    for action in turn_data["actions_and_responses"]:
        response = action["response"]
        # Truncate very long responses but preserve key information
        if len(response) > 300:
            response = response[:250] + "... [truncated]"

        output += f"Turn {action['turn']}: {action['action']}\n"
        output += f"Response: {response}\n"
        output += f"Reasoning: {action.get('reasoning', 'N/A')}\n"
        output += f"Critic Score: {action.get('critic_score', 'N/A')}\n\n"

    # Events section
    output += "\nEVENTS:\n"

    # Death events with full details
    if turn_data.get("death_events"):
        output += f"Deaths: {len(turn_data['death_events'])}\n"
        for death in turn_data["death_events"]:
            output += f"  - Turn {death['turn']}: {death['reason']}\n"
            output += f"    Fatal action: {death.get('action_taken', 'Unknown')}\n"
            output += f"    Location: {death.get('death_location', 'Unknown')}\n"
            if death.get("death_messages"):
                output += f"    Messages: {', '.join(death['death_messages'])}\n"
    else:
        output += "Deaths: None\n"

    # Score changes
    if turn_data.get("score_changes"):
        output += f"\nScore Changes: {len(turn_data['score_changes'])}\n"
        for change in turn_data["score_changes"]:
            output += f"  - Turn {change['turn']}: {change['from_score']} → {change['to_score']}\n"

    # Location changes
    if turn_data.get("location_changes"):
        output += f"\nLocation Changes: {len(turn_data['location_changes'])}\n"
        for change in turn_data["location_changes"]:
            output += f"  - Turn {change['turn']}: {change['from_location']} → {change['to_location']}\n"

    return output


def format_death_analysis_section(turn_data: Dict) -> str:
    """
    Format death events for the knowledge base.

    Creates a formatted section describing death events with context, location,
    fatal actions, and key messages. This is used within the knowledge base
    to document death patterns and avoidance strategies.

    Args:
        turn_data: Dictionary containing death_events list

    Returns:
        Formatted death analysis section
    """
    if not turn_data.get("death_events"):
        return "No deaths occurred in this session."

    output = f"**{len(turn_data['death_events'])} death(s) occurred:**\n\n"

    for death in turn_data["death_events"]:
        output += f"**Death at Turn {death['turn']}**\n"
        output += f"- Cause: {death['reason']}\n"
        output += f"- Fatal Action: {death.get('action_taken', 'Unknown')}\n"
        output += f"- Location: {death.get('death_location', 'Unknown')}\n"
        output += f"- Final Score: {death.get('final_score', 'Unknown')}\n"

        if death.get("death_messages"):
            output += f"- Key Messages: {'; '.join(death['death_messages'])}\n"

        # Include contextual information
        if death.get("death_context"):
            output += f"- Context: {death['death_context']}\n"

        output += "\n"

    return output


@observe(name="strategy-generate-update")
def generate_knowledge_directly(
    turn_data: Dict,
    existing_knowledge: str,
    client,
    analysis_model: str,
    analysis_sampling: dict,
    logger=None
) -> str:
    """
    Generate knowledge base content in a single LLM call.

    This function takes turn data and existing knowledge and generates a complete
    updated knowledge base with all required sections: dangers & threats, puzzle
    solutions, strategic patterns, death & danger analysis, command syntax,
    lessons learned, and cross-episode insights.

    Args:
        turn_data: Extracted turn data with actions, responses, and events
        existing_knowledge: Current knowledge base content (if any)
        client: LLM client wrapper instance
        analysis_model: Model identifier for knowledge generation
        analysis_sampling: Sampling parameters (temperature, top_p, top_k, min_p, max_tokens)
        logger: Optional logger instance for logging

    Returns:
        Complete knowledge base content or existing knowledge on failure
    """
    # Format turn data
    formatted_data = format_turn_data_for_prompt(turn_data)

    # Extract cross-episode insights from existing knowledge base
    cross_episode_section = extract_cross_episode_section(existing_knowledge)

    # Construct comprehensive prompt
    prompt = f"""Analyze this Zork gameplay data and create/update the knowledge base.

NOTE: Item locations, room connections, and object properties are tracked in a separate structured memory system.
This knowledge base focuses on STRATEGIC insights: parser patterns, game mechanics, puzzle approaches, and meta-strategy.

{formatted_data}

EXISTING KNOWLEDGE BASE:
{"-" * 50}
{existing_knowledge if existing_knowledge else "No existing knowledge - this is the first update"}
{"-" * 50}

{cross_episode_section}

STRATEGIC vs SPATIAL BOUNDARIES:

✅ INCLUDE (Strategic patterns for knowledge base):
- Parser syntax: "Use 'enter X' not 'climb through X' for windows"
- Game mechanics: "Containers must be opened before accessing contents"
- Command patterns: "Parser accepts 'n/s/e/w' and 'north/south/east/west' equally"
- Puzzle approaches: "Multi-step sequences often require 'examine' before 'take'"
- Meta-strategy: "15+ turns without progress → evacuate to unmapped area"

❌ EXCLUDE (Spatial memory handles this):
- Location connections: "Behind House has window to Kitchen"
- Object locations: "Trophy case is in Living Room"
- Item positions: "Jeweled egg found in tree at Forest Path"
- Room adjacency: "West of House connects to North of House"

DEDUPLICATION REQUIREMENTS:
1. Before adding new entries, check if similar patterns already exist in the knowledge base
2. Consolidate location-specific variants into single general patterns with examples
3. If 2+ entries describe the same core mechanic, MERGE them into one entry
4. Maximum limits per section:
   - DANGERS & THREATS: 5 distinct death patterns (consolidate aggressively)
   - PUZZLE SOLUTIONS: 10 puzzle entries (merge similar puzzles)
   - STRATEGIC PATTERNS: 8 core patterns (avoid location-specific duplicates)
5. Format consolidated patterns as:
   **[Pattern Name]**: [Core insight]
   Examples/Instances: [specific case 1], [specific case 2], [specific case 3]

Example consolidation:
❌ DON'T: Separate entries for "Forest Death", "House Death", "Kitchen Cycling Death", "Mathematical Crisis Death"
✅ DO: "**Stuck-in-Area Death Pattern**: When 15+ turns pass without score progress in same area, immediate evacuation required. Recognition: repeated room descriptions, identical navigation loops. Prevention: Use unmapped directions to break pattern. High-risk areas: Forest paths, house perimeter (Kitchen-Living Room loops)"

LENGTH REQUIREMENTS:
- DANGERS & THREATS: Maximum 600 words total
- PUZZLE SOLUTIONS: Maximum 500 words total
- STRATEGIC PATTERNS: Maximum 600 words total
- COMMAND SYNTAX: Maximum 300 words total
- Total knowledge base: Target 2000-2500 words (consolidate to meet budget)
- Quality over quantity - remove verbose redundancy

INSTRUCTIONS:
Create a comprehensive knowledge base with ALL of the following sections. Consolidate existing content with new information.

## DANGERS & THREATS
Document specific dangers (MAX 5 DISTINCT PATTERNS - consolidate similar ones):
- **Death Patterns**: What actions/situations consistently lead to death? Merge location-specific variants
- **Warning Signs**: What game text signals danger? (specific quotes)
- **Safe Approaches**: How to safely navigate dangerous situations? (concrete protocols)
- **Environmental Hazards**: Special properties that pose risks (general mechanics, not locations)

Focus on PARSER and GAME MECHANICS that cause death, not just locations.

## PUZZLE SOLUTIONS
Document puzzle mechanics (MAX 10 PUZZLES - merge similar ones):
- **Solved Puzzles**: Complete solutions - but generalize the approach where possible
- **Puzzle Patterns**: Common puzzle types (containers, multi-step sequences, hidden reveals)
- **Failed Solutions**: What didn't work and why (parser rejections, wrong sequence)
- **Partial Progress**: Puzzles partially solved with specific next steps

Emphasize GENERAL APPROACHES over location-specific solutions.

## STRATEGIC PATTERNS
Identify actionable patterns (MAX 8 CORE PATTERNS):
- **Successful Actions**: What specific action types led to progress? (not locations)
- **Failed Approaches**: What didn't work and why? (focus on parser/mechanic failures)
- **Exploration Strategies**: Effective methods for discovering new areas
- **Resource Management**: How to use inventory, light, items effectively
- **Objective Recognition**: How to identify new goals from game responses

## DEATH & DANGER ANALYSIS
{format_death_analysis_section(turn_data) if turn_data.get("death_events") else "No deaths occurred in this session."}

Consolidate with existing death patterns above - don't duplicate information.

## COMMAND SYNTAX
Parser patterns discovered (MAX 300 WORDS - focus on non-obvious patterns):
- **Verb Synonyms**: What verbs are interchangeable? (take/get, examine/look at)
- **Verb Failures**: What verbs DON'T work when expected? (climb vs enter, disturb vs take)
- **Command Structure**: Multi-word commands, prepositions that work/fail
- **Special Syntax**: Unusual but effective command patterns discovered

THIS SECTION IS CRITICAL - LLMs struggle with parser quirks without explicit patterns.

## LESSONS LEARNED
Specific insights from this session (session-specific, will be consolidated in cross-episode later):
- **New Discoveries**: First-time strategic insights
- **Confirmed Patterns**: What previous knowledge was validated?
- **Updated Understanding**: What assumptions were corrected?
- **Future Strategies**: What should be tried next?

## CROSS-EPISODE INSIGHTS
Persistent strategic wisdom across episodes (updated at episode completion):
- **Death Patterns Across Episodes**: Validated death causes with episode references
- **Environmental Knowledge**: Persistent game mechanics discovered across multiple episodes
- **Strategic Meta-Patterns**: Approaches consistently effective/ineffective across situations
- **Major Discoveries**: Game mechanics, puzzle solutions validated by multiple episodes

NOTE: This section primarily updated during inter-episode synthesis. For now, include observations that relate to cross-episode patterns if they validate or contradict existing entries.

CRITICAL CONSOLIDATION RULES:
1. **Before adding ANY new entry, check if it duplicates existing content**
2. **Merge similar patterns** - don't create variants (e.g., "Forest Death" + "House Death" = "Stuck-in-Area Death")
3. **Use examples, not separate entries** - "Pattern X: [insight]. Examples: case1, case2, case3"
4. **Remove obsolete entries** - if new data contradicts old, remove the old
5. **Stay within word budgets** - if over budget, merge more aggressively
6. **Prioritize parser patterns** - command syntax knowledge is most valuable for LLM agents

OUTPUT REQUIREMENTS:
- Complete knowledge base with all sections
- Each section respects word/entry limits
- Consolidated patterns with examples, not duplicates
- Parser syntax patterns prominently documented
- Strategic patterns, not spatial facts
- Actionable insights for decision-making

Remember: The structured memory system handles factual data (locations, connections, inventory).
This knowledge base provides strategic intelligence: HOW to construct commands, WHAT mechanics govern puzzles, WHEN to change strategies."""

    try:
        messages = [
            {
                "role": "system",
                "content": """You are creating a strategic knowledge base for an AI agent playing Zork.

A separate memory system tracks factual data (room locations, item positions, connections).

Your role is to provide strategic intelligence:
1. Parser patterns - which verbs/commands work or fail
2. Game mechanics - how containers, inventory, light, puzzles actually function
3. Puzzle approaches - general strategies, not location-specific solutions
4. Meta-strategy - when to change tactics, how to recognize being stuck
5. Death prevention - core causes and recognition patterns

CRITICAL: Consolidate aggressively. If 2+ entries describe the same pattern, merge them. Use examples instead of creating variants. The agent needs concise, actionable patterns - not an encyclopedia of every instance.

Focus on WHY things happen and HOW to approach situations, not cataloging WHAT exists in which location.""",
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=analysis_model,
            messages=messages,
            temperature=analysis_sampling.get("temperature"),
            top_p=analysis_sampling.get("top_p"),
            top_k=analysis_sampling.get("top_k"),
            min_p=analysis_sampling.get("min_p"),
            max_tokens=analysis_sampling.get("max_tokens") or 3000,
            name="StrategyGenerator",
        )

        return response.content.strip()

    except Exception as e:
        if logger:
            logger.error(
                f"Knowledge generation failed: {e}",
                extra={"event_type": "knowledge_update", "error": str(e)},
            )
        # Return existing knowledge on failure
        return existing_knowledge


@observe(name="knowledge-consolidation-pass")
def consolidate_knowledge_base(
    existing_knowledge: str,
    client,
    analysis_model: str,
    analysis_sampling: dict,
    logger=None
) -> str:
    """
    Perform aggressive consolidation pass on knowledge base to eliminate redundancy.

    This is an optional cleanup function that can be run periodically (e.g., every 5-10 episodes)
    to keep the knowledge base concise and actionable.

    Args:
        existing_knowledge: Current knowledge base content
        client: LLM client wrapper instance
        analysis_model: Model identifier for consolidation task
        analysis_sampling: Sampling parameters (temperature, top_p, top_k, min_p, max_tokens)
        logger: Optional logger instance

    Returns:
        Consolidated knowledge base content
    """
    prompt = f"""Consolidate this Zork knowledge base to eliminate redundancy while preserving all unique insights.

**CURRENT KNOWLEDGE BASE:**
{existing_knowledge}

**CONSOLIDATION TASK:**

Your goal is to drastically reduce redundancy while preserving every unique insight. The knowledge base has grown through incremental updates and now contains many duplicate or near-duplicate entries.

CONSOLIDATION STRATEGY:

1. **Identify Duplicate Patterns**:
   - Look for entries that describe the same core mechanic/pattern
   - Group similar entries by underlying cause/pattern
   - Example: "Forest Death", "House Death", "Kitchen Loop Death" all describe "stuck without progress"

2. **Merge Duplicates**:
   - Create single entry for each unique pattern
   - Use format: "**Pattern Name**: [Core insight]. Examples/Instances: [case1, case2, case3]"
   - Preserve all unique details as examples within merged entry
   - Remove completely redundant information

3. **Enforce Section Limits**:
   - DANGERS & THREATS: Maximum 5 distinct death patterns
   - PUZZLE SOLUTIONS: Maximum 10 puzzle entries (merge similar puzzle types)
   - STRATEGIC PATTERNS: Maximum 8 core patterns (remove location-specific duplicates)
   - COMMAND SYNTAX: Maximum 300 words (group by command type)
   - CROSS-EPISODE INSIGHTS: Maximum 800 words total

4. **Prioritize Content**:
   - Parser syntax patterns (highest value for LLM agents)
   - Game mechanics (container behavior, inventory rules, light mechanics)
   - General puzzle approaches (not location-specific solutions)
   - Meta-strategy (when to change approach, how to recognize patterns)
   - Death patterns (consolidated to core causes, not location variants)

5. **Remove Spatial Duplicates**:
   These belong in spatial memory system, NOT knowledge base:
   - Room connections ("Behind House connects to Kitchen")
   - Object locations ("Trophy case is in Living Room")
   - Item positions ("Jeweled egg in tree at Forest Path")
   - Navigation paths ("West of House → North of House → Forest Path")

CONSOLIDATION EXAMPLES:

❌ BEFORE (redundant - 500 words across 5 entries):
- "Mathematical Crisis Death": 20+ turns without score progress creates unrecoverable deficits...
- "Navigation Cycling Death": Parser-limited connections create fatal turn consumption loops...
- "Forest Path Cycling Death": Persistent navigation loops in forest areas consume turns...
- "House Area Mathematical Death": Extended exploration within house perimeter during crises...
- "Extreme Mathematical Persistence Death": Continuing area-specific exploration during 30+ turn stagnation...

✅ AFTER (consolidated - 80 words, 1 entry):
- "**Stuck-in-Area Death**: When 15+ turns pass in same area without score progress, immediate evacuation required to prevent fatal turn deficit. Recognition: repeated identical room descriptions, navigation loops, parser-limited exits. Prevention: Use unmapped directions after 15 stagnant turns; during extreme crisis (30+ turns) complete area abandonment is only viable strategy. High-risk areas: Forest paths (limited exits), house perimeter (Kitchen-Living Room loops), any area with < 3 distinct exits."

❌ BEFORE (location-specific - belongs in spatial memory):
- "Window Entry Puzzle": "open window" followed by "enter window" grants 10 points at Kitchen entry from Behind House
- "Location-Specific Window Puzzle": Window entry scoring only works from Behind House location

✅ AFTER (general pattern):
- "**Window Entry Pattern**: Windows require two-step sequence: 'open [window]' then 'enter [window]'. Parser rejects 'climb through' or 'go through'. Entry is directional - verify correct side before attempting."

LENGTH TARGETS:
- Total knowledge base: 2000-2500 words (currently may be 3500+)
- Each section must fit within its budget
- Aim for 50% reduction in word count while preserving 100% of unique insights
- Quality over quantity: 5 excellent patterns > 20 redundant variants

CRITICAL SUCCESS CRITERIA:
1. Every unique insight from original is preserved (as entry or example)
2. No duplicate patterns remain (same cause with different names)
3. Total word count < 2500 words
4. Parser syntax patterns are prominent and complete
5. No spatial memory information remains (locations, connections, positions)
6. Each section respects its maximum entry/word limit

**OUTPUT:**
Provide the complete consolidated knowledge base with all sections.
"""

    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert at consolidating strategic knowledge while preserving information density. Your goal is aggressive deduplication without information loss - merge redundant patterns, remove spatial memory duplicates, and create concise actionable insights."
            },
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model=analysis_model,
            messages=messages,
            temperature=analysis_sampling.get("temperature"),
            top_p=analysis_sampling.get("top_p"),
            top_k=analysis_sampling.get("top_k"),
            min_p=analysis_sampling.get("min_p"),
            max_tokens=analysis_sampling.get("max_tokens") or 4000,
            name="KnowledgeConsolidator",
        )

        consolidated_content = response.content.strip()

        if logger:
            original_length = len(existing_knowledge)
            consolidated_length = len(consolidated_content)
            reduction_percent = round((1 - consolidated_length/original_length) * 100, 1) if original_length > 0 else 0

            logger.info(
                "Knowledge base consolidation completed",
                extra={
                    "event_type": "knowledge_consolidation",
                    "original_length": original_length,
                    "consolidated_length": consolidated_length,
                    "reduction_percent": reduction_percent
                }
            )

        return consolidated_content

    except Exception as e:
        if logger:
            logger.error(
                f"Knowledge consolidation failed: {e}",
                extra={"event_type": "knowledge_consolidation", "error": str(e)}
            )
        return existing_knowledge
