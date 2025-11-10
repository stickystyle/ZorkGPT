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
from knowledge.turn_extraction import episode_ended_in_loop_break

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
        episode_id = turn_data.get("episode_id", "unknown")
        # Filter Loop Break timeouts (system behavior, not game mechanic)
        if episode_ended_in_loop_break(episode_id, workdir="game_files"):
            output += "Deaths: Filtered (Loop Break timeout - system terminated stuck episode)\n"
        else:
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
    prompt = f"""Analyze this Zork gameplay data and create/update the strategic knowledge base.

ARCHITECTURE REMINDER:
- **Memory System** handles location-specific procedures (stored at SOURCE locations with multi-step detection)
- **Loop Break System** terminates episodes stuck 20+ turns without score (ignore these as "mathematical deaths")
- **Objective System** discovers and tracks goals automatically
- **Map System** manages spatial navigation and connections

THIS knowledge base provides UNIVERSAL strategic wisdom, not location-specific tactics.

{formatted_data}

EXISTING KNOWLEDGE BASE:
{"-" * 50}
{existing_knowledge if existing_knowledge else "No existing knowledge - this is the first update"}
{"-" * 50}

{cross_episode_section}

INSTRUCTIONS:
Create a strategic knowledge base with the following sections. Focus on UNIVERSAL patterns, not location-specific procedures.

## UNIVERSAL GAME MECHANICS
Document game rules and mechanics that apply everywhere:
- **Parser Patterns**: How the parser interprets commands (e.g., "EXAMINE reveals details", "Containers require OPEN before access")
- **Object Behaviors**: Universal object properties (e.g., "Dropped items persist in location", "Some actions are irreversible")
- **Scoring Mechanics**: How progress is measured (e.g., "Score increases indicate advancement")
- **Action Categories**: Types of actions and their properties (e.g., "Reversible: TAKE/DROP vs Irreversible: GIVE/THROW")

EXAMPLES (GOOD):
✅ "EXAMINE command reveals hidden information about objects"
✅ "Containers must be opened before contents can be accessed"
✅ "Some actions cannot be undone (GIVE, THROW, BREAK)"

EXAMPLES (BAD - belongs in Memory System):
❌ "Open window then enter window at Behind House leads to Kitchen"
❌ "Egg is found by going up from Forest Path"

## DANGER CATEGORIES
Document TYPES of dangers and recognition patterns (not specific instances):
- **Threat Awareness**: General categories of dangers that exist (e.g., "Combat enemies exist", "Dark areas have dangers")
- **Warning Signals**: Game text patterns that indicate danger (e.g., "pitch black" warns of visibility danger)
- **Death Mechanics**: Universal death causes (e.g., "Prolonged darkness", "Combat without weapons")
- **Safety Principles**: General danger avoidance strategies (e.g., "Light source required in dark areas")

EXAMPLES (GOOD):
✅ "Dark areas pose dangers - carry light source"
✅ "Combat enemies exist - some require weapons, others can be avoided"
✅ "Warning text 'pitch black' indicates visibility hazard"

EXAMPLES (BAD - too specific or system behavior):
❌ "Troll at Troll Room attacks on sight" (specific location → Memory System)
❌ "Forest exploration causes mathematical death after 20 turns" (Loop Break timeout, not game mechanic)
❌ "Kitchen-Living Room cycling wastes turns during score crisis" (Loop Break detection, not game danger)

## STRATEGIC PRINCIPLES
Universal decision-making strategies (not location-specific tactics):
- **Exploration Heuristics**: General approaches to discovering new areas (e.g., "Try all cardinal directions", "EXAMINE before interacting")
- **Resource Management**: Universal item/inventory strategies (e.g., "Drop non-essentials to manage weight", "Keep key items accessible")
- **Problem-Solving Patterns**: General puzzle-solving approaches (e.g., "Try alternative verbs when stuck", "Multi-step procedures may exist")
- **Progress Indicators**: How to recognize forward momentum (e.g., "Score increases", "New areas discovered")

EXAMPLES (GOOD):
✅ "When stuck, try alternative action verbs (SEARCH, MOVE, LOOK UNDER)"
✅ "Multi-step procedures exist - action A may enable action B"
✅ "Inventory management critical - drop non-essentials to maintain capacity"

EXAMPLES (BAD - too specific):
❌ "Execute time-sensitive window entry immediately" (specific objective → Objective System)
❌ "Evacuate house area during mathematical crisis" (Loop Break system behavior)
❌ "Collect egg from tree before exploring forest" (specific tactic → Memory System)

## DEATH & DANGER ANALYSIS
Analyze death events from CURRENT session for universal lessons:
{format_death_analysis_section(turn_data) if turn_data.get("death_events") else "No deaths occurred in this session."}

**CRITICAL**: If death occurred at 20+ turns without score progress, this is LOOP BREAK TIMEOUT (system behavior), NOT a game mechanic. Do not document as "mathematical death" - instead extract the ACTUAL game-related issue (e.g., "Stuck in navigation loop due to identical room descriptions").

## LESSONS LEARNED
Strategic insights from this session (universal principles only):
- **New Mechanics Discovered**: Universal game rules learned for the first time
- **Confirmed Patterns**: Previously known principles validated this session
- **Updated Understanding**: Corrections to previous assumptions about game mechanics
- **Meta-Strategies**: General approaches that proved effective/ineffective

FOCUS: Extract the PRINCIPLE, not the instance.
EXAMPLE: "Multi-step procedures revealed through examining objects" NOT "Window requires open then enter"

## CROSS-EPISODE INSIGHTS
Persistent strategic wisdom validated across multiple episodes:
- **Validated Game Mechanics**: Universal rules confirmed across multiple episodes
- **Danger Recognition Patterns**: Warning signals and threat categories proven reliable
- **Strategic Meta-Patterns**: Approaches consistently effective/ineffective across different situations
- **Critical Discoveries**: Game mechanics or parser behaviors discovered

NOTE: This section is primarily updated during inter-episode synthesis at episode end.

CRITICAL REQUIREMENTS:
1. **Universal Scope**: ONLY include knowledge that applies regardless of location
2. **Principle Over Instance**: Extract the pattern, not the specific example
3. **System Awareness**: Ignore Loop Break timeouts as "mathematical deaths" - they're system behavior
4. **No Location Coupling**: If it requires "at Location X", it belongs in Memory System
5. **Actionable Guidance**: Provide decision-making principles, not specific tactics

**BREVITY REQUIREMENT:**
Keep the knowledge base CONCISE and ACTION-ORIENTED:
- Each insight should be 1-3 sentences maximum
- Focus on the "what" and "why", omit verbose explanations
- Use bullet points for clarity, not paragraphs
- Prioritize high-value insights, discard low-signal observations
- NO repetition - if similar insight exists in another section, reference it
- Target output: 200-400 tokens per section (not per insight)
- When in doubt, OMIT rather than include marginal content

QUALITY CHECKS:
❌ If your entry mentions specific locations by name → Move to Memory System
❌ If your entry documents "mathematical crisis/death" → Ignore (Loop Break timeout)
❌ If your entry prescribes specific action sequences → Move to Memory System
❌ If your entry is only relevant at one location → Move to Memory System
✅ If your entry is a universal principle applicable anywhere → Keep in Knowledge Base
✅ If your entry is a game mechanic that always applies → Keep in Knowledge Base
✅ If your entry is a danger category (not specific instance) → Keep in Knowledge Base

Generate the complete knowledge base with all sections."""

    try:
        messages = [
            {
                "role": "system",
                "content": """You are creating a STRATEGIC knowledge base for an AI agent playing Zork.

CRITICAL CONTEXT - Other Systems Handle:
1. **Location-Specific Memory System**: Stores procedural knowledge at specific locations (e.g., "At Behind House, enter window leads to Kitchen"). Multi-step procedures are detected automatically.
2. **Loop Break System**: Programmatically terminates episodes stuck without progress (20+ turns without score). These are system timeouts, NOT game mechanics.
3. **Objective System**: Discovers and tracks goals automatically with LLM-based completion checking.
4. **Map System**: Manages spatial relationships, exits, and navigation.

YOUR ROLE:
Generate ONLY universal game mechanics and strategic principles that apply REGARDLESS of location.

SCOPE BOUNDARIES:
✅ INCLUDE: "EXAMINE reveals hidden information" (universal mechanic)
✅ INCLUDE: "Combat enemies exist in some locations" (danger category awareness)
✅ INCLUDE: "Multi-step procedures exist - action A may enable action B" (meta-pattern)
❌ EXCLUDE: "At Behind House, open window then enter window" (location-specific procedure → Memory System)
❌ EXCLUDE: "Extended forest exploration causes mathematical death" (loop break timeout → System behavior)
❌ EXCLUDE: "Prioritize time-sensitive objectives immediately" (objective prioritization → Objective System)

If your knowledge requires phrases like "at Location X", "in Room Y", or "from Behind House", it belongs in the Memory System instead.

Focus on WHY things happen and HOW to approach situations universally, not WHAT to do at specific locations.""",
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
            max_tokens=analysis_sampling.get("max_tokens") or 2000,  # Reduced from 3000
            name="StrategyGenerator",
        )

        return response.content.strip()

    except Exception as e:
        error_msg = f"Knowledge generation failed: {e}"
        if logger:
            logger.error(
                error_msg,
                extra={"event_type": "knowledge_update", "error": str(e)},
            )
        else:
            # Print error if no logger (useful for testing)
            print(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
        # Return existing knowledge on failure
        return existing_knowledge
