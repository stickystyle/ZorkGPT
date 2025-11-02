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

NOTE: Item locations, room connections, and object properties are now tracked in a separate structured memory system.
This knowledge base should focus on STRATEGIC insights, patterns, and lessons learned from gameplay.

{formatted_data}

EXISTING KNOWLEDGE BASE:
{"-" * 50}
{existing_knowledge if existing_knowledge else "No existing knowledge - this is the first update"}
{"-" * 50}

{cross_episode_section}

INSTRUCTIONS:
Create a comprehensive knowledge base with ALL of the following sections. If a section has no new information, keep the existing content for that section.

## DANGERS & THREATS
Document specific dangers and how to recognize them:
- **Death Patterns**: What actions/locations consistently lead to death? (e.g., "moving east from dark cellar without light causes grue death")
- **Warning Signs**: What game text signals danger? (e.g., "slavering fangs" indicates imminent grue attack)
- **Safe Approaches**: How to safely navigate dangerous areas?
- **Environmental Hazards**: Special location properties that pose risks

## PUZZLE SOLUTIONS
Document puzzle mechanics and solutions:
- **Solved Puzzles**: Complete solutions to puzzles encountered
- **Puzzle Patterns**: Common puzzle types and approaches that work
- **Failed Solutions**: What didn't work and why (avoid repeating mistakes)
- **Partial Progress**: Puzzles partially solved with notes on next steps

## STRATEGIC PATTERNS
Identify patterns from this gameplay session:
- **Successful Actions**: What specific actions led to progress?
- **Failed Approaches**: What didn't work and why?
- **Exploration Strategies**: Effective methods for discovering new areas
- **Resource Management**: How to use items effectively
- **Objective Recognition**: How to identify new goals from game responses

## DEATH & DANGER ANALYSIS
{format_death_analysis_section(turn_data) if turn_data.get("death_events") else "No deaths occurred in this session."}

## COMMAND SYNTAX
List exact commands that worked (focus on non-obvious or puzzle-specific commands):
- **Puzzle Commands**: Commands that solved specific puzzles
- **Special Interactions**: Unusual but effective command patterns
- **Combat**: Any combat-related commands
- **Syntax Discoveries**: Command formats that worked when standard approaches failed

## LESSONS LEARNED
Specific insights from this session:
- **New Discoveries**: What strategic insights were learned for the first time?
- **Confirmed Patterns**: What previous strategic knowledge was validated?
- **Updated Understanding**: What previous assumptions were corrected?
- **Future Strategies**: What should be tried next based on these learnings?

## CROSS-EPISODE INSIGHTS
Persistent strategic wisdom that carries across episodes (updated at episode completion):
- **Death Patterns Across Episodes**: Consistent death causes and prevention strategies
- **Environmental Knowledge**: Persistent facts about game world (dangerous locations, item behaviors, puzzle mechanics)
- **Strategic Meta-Patterns**: Approaches that prove consistently effective/ineffective across different situations
- **Major Discoveries**: Game mechanics, hidden areas, puzzle solutions discovered
- **NOTE**: This section is primarily updated during inter-episode synthesis at episode end, but may include observations from current session that relate to cross-episode patterns.

CRITICAL REQUIREMENTS:
1. **Strategic Focus**: Focus on WHY and HOW, not just WHAT (factual data is in memory system)
2. **Pattern Recognition**: Identify patterns across multiple situations
3. **Danger Prevention**: Emphasize death avoidance and danger recognition
4. **Actionable Insights**: Provide decision-making guidance, not just facts
5. **Complete Sections**: Include all sections even if some have minimal updates

Remember: The structured memory system handles factual data (locations, connections, inventory).
This knowledge base provides strategic intelligence to make better decisions."""

    try:
        messages = [
            {
                "role": "system",
                "content": "You are creating a strategic knowledge base for an AI agent playing Zork. A separate memory system already tracks factual data (room locations, item positions, connections). Your role is to identify strategic patterns, danger signals, puzzle solutions, and actionable decision-making insights. Focus on WHY things happen and HOW to approach situations, not just cataloging WHAT exists.",
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=analysis_model,
            messages=messages,
            temperature=analysis_sampling.temperature,
            top_p=analysis_sampling.top_p,
            top_k=analysis_sampling.top_k,
            min_p=analysis_sampling.min_p,
            max_tokens=analysis_sampling.max_tokens or 3000,
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
