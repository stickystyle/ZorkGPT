# ABOUTME: Cross-episode wisdom synthesis for persistent learning across game sessions.
# ABOUTME: Analyzes completed episodes to extract persistent strategic insights into CROSS-EPISODE INSIGHTS section.

"""
Cross-episode synthesis module.

This module provides functions for synthesizing persistent wisdom from completed
episodes into the CROSS-EPISODE INSIGHTS section of the knowledge base. It focuses
on death patterns, environmental knowledge, strategic meta-patterns, and major
discoveries that should persist across multiple game sessions.
"""

from typing import Dict, Optional
from knowledge.section_utils import extract_section_content, update_section_content
from knowledge.knowledge_generation import format_turn_data_for_prompt

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


@observe(name="knowledge-synthesize-strategic")
def synthesize_inter_episode_wisdom(
    episode_data: Dict,
    output_file: str,
    client,
    analysis_model: str,
    analysis_sampling: dict,
    logger=None
) -> bool:
    """
    Synthesize persistent wisdom from episode completion into the CROSS-EPISODE INSIGHTS
    section of knowledgebase.md. Focuses on deaths, major discoveries, and cross-episode patterns.

    This function is called at episode completion when significant learning opportunities
    exist (deaths, high scores, long episodes, good critic scores). It extracts persistent
    wisdom that should carry forward to future episodes.

    Args:
        episode_data: Dictionary containing episode summary information with keys:
            - episode_id: Episode identifier
            - turn_count: Total turns in episode
            - final_score: Final game score
            - death_count: Number of deaths in episode
            - episode_ended_in_death: Boolean indicating if episode ended in death
            - avg_critic_score: Average critic score across episode
            - discovered_objectives: List of discovered objectives (optional)
            - completed_objectives: List of completed objectives (optional)
            - extract_turn_data_callback: Callback function to extract turn data (episode_id, start, end) -> Dict
        output_file: Path to knowledge base file
        client: LLM client wrapper instance
        analysis_model: Model identifier for synthesis task
        analysis_sampling: Sampling parameters (temperature, top_p, top_k, min_p, max_tokens)
        logger: Optional logger instance for logging

    Returns:
        True if synthesis was performed and wisdom was updated, False if skipped
    """
    if logger:
        logger.info(
            f"Synthesizing cross-episode insights from episode {episode_data['episode_id']}",
            extra={"event_type": "knowledge_update"},
        )

    # Extract key episode data for synthesis
    episode_id = episode_data["episode_id"]
    turn_count = episode_data["turn_count"]
    final_score = episode_data["final_score"]
    death_count = episode_data["death_count"]
    episode_ended_in_death = episode_data["episode_ended_in_death"]
    avg_critic_score = episode_data["avg_critic_score"]

    # Always synthesize if episode ended in death (critical learning event)
    # or if significant progress was made (score > 50 or many turns)
    should_synthesize = (
        episode_ended_in_death
        or final_score >= 50
        or turn_count >= 100
        or avg_critic_score >= 0.3
    )

    if not should_synthesize:
        if logger:
            logger.info(
                "Episode not significant enough for cross-episode synthesis",
                extra={
                    "event_type": "knowledge_update",
                    "details": f"Death: {episode_ended_in_death}, Score: {final_score}, Turns: {turn_count}, Avg Critic: {avg_critic_score:.2f}",
                },
            )
        return False

    # Extract turn-by-turn data for death analysis and major discoveries
    # Use callback provided in episode_data to extract turn data
    extract_turn_data = episode_data.get("extract_turn_data_callback")
    if not extract_turn_data:
        if logger:
            logger.error(
                "No extract_turn_data_callback provided in episode_data",
                extra={"event_type": "knowledge_update"},
            )
        return False

    turn_data = extract_turn_data(episode_id, 1, turn_count)
    if not turn_data:
        if logger:
            logger.warning(
                "Could not extract turn data for cross-episode synthesis",
                extra={"event_type": "knowledge_update"},
            )
        return False

    # Load existing knowledge base
    existing_knowledge = ""
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_knowledge = f.read()
    except FileNotFoundError:
        # No existing knowledge base - create basic one
        existing_knowledge = "# Zork Game World Knowledge Base\n\n"
    except Exception as e:
        if logger:
            logger.warning(
                f"Could not load existing knowledge base: {e}",
                extra={"event_type": "knowledge_update"},
            )
        existing_knowledge = "# Zork Game World Knowledge Base\n\n"

    # Extract existing cross-episode insights section
    existing_cross_episode = extract_section_content(
        existing_knowledge, "CROSS-EPISODE INSIGHTS"
    )

    # Prepare death event analysis if applicable
    death_analysis = ""
    if episode_ended_in_death or turn_data.get("death_events"):
        death_analysis = "\n\nDEATH EVENT ANALYSIS:\n"
        for event in turn_data.get("death_events", []):
            death_analysis += (
                f"Episode {episode_id}, Turn {event['turn']}: {event['reason']}\n"
            )
            if event.get("death_context"):
                death_analysis += f"- Context: {event['death_context']}\n"
            if event.get("death_location"):
                death_analysis += f"- Location: {event['death_location']}\n"
            if event.get("action_taken"):
                death_analysis += f"- Fatal action: {event['action_taken']}\n"
            death_analysis += "\n"

    # Create synthesis prompt
    prompt = f"""Analyze this completed Zork episode and perform FULL KNOWLEDGE BASE CONSOLIDATION.

**CURRENT EPISODE SUMMARY:**
- Episode ID: {episode_id}
- Total turns: {turn_count}
- Final score: {final_score}
- Deaths this episode: {death_count}
- Episode ended in death: {episode_ended_in_death}
- Average critic score: {avg_critic_score:.2f}
- Discovered objectives: {len(episode_data.get("discovered_objectives", []))}
- Completed objectives: {len(episode_data.get("completed_objectives", []))}

**EPISODE ACTIONS SUMMARY:**
{format_turn_data_for_prompt(turn_data)[:2000] if turn_data.get("actions_and_responses") else "No action data available"}

{death_analysis}

**CURRENT FULL KNOWLEDGE BASE:**
{existing_knowledge if existing_knowledge else "No existing knowledge base."}

**CROSS-EPISODE CONSOLIDATION TASK:**

Your task is to update the CROSS-EPISODE INSIGHTS section AND consolidate the entire knowledge base to eliminate redundancy.

PHASE 1: Update Cross-Episode Insights
--------------------------------------
Add validated patterns from this episode to these subsections:

1. **Death Patterns Across Episodes**:
   - Add death patterns from this episode that represent NEW learnings
   - Merge with existing patterns if similar
   - Include episode reference (e.g., "Validated in Episodes 1, 3, 7")
   - Maximum 5 distinct death patterns total

2. **Environmental Knowledge**:
   - Persistent game mechanics discovered this episode
   - Parser syntax patterns confirmed
   - Puzzle mechanics validated
   - Merge with existing environmental knowledge

3. **Strategic Meta-Patterns**:
   - Approaches proven effective/ineffective this episode
   - Decision-making heuristics that worked
   - Maximum 6 distinct meta-patterns total

4. **Major Discoveries**:
   - Significant game mechanics or puzzle solutions discovered
   - Only include truly major discoveries
   - Maximum 5 entries total

PHASE 2: Consolidate Entire Knowledge Base
------------------------------------------
Review ALL sections for redundancy and consolidation:

**DANGERS & THREATS Section:**
- Consolidate duplicate death patterns (target: 5 distinct patterns max)
- Merge location-specific variants: "Pattern X (Examples: location1, location2)"
- Remove entries that are now in Cross-Episode Insights
- Ensure total section < 600 words

**PUZZLE SOLUTIONS Section:**
- Merge similar puzzle types
- Keep general approaches, remove location-specific duplicates
- Maximum 10 puzzle entries
- Focus on puzzle PATTERNS not individual instances

**STRATEGIC PATTERNS Section:**
- Consolidate similar patterns
- Remove spatial memory duplicates (location connections, item positions)
- Keep parser syntax patterns prominent
- Maximum 8 core patterns

**COMMAND SYNTAX Section:**
- Keep this section focused and concise (< 300 words)
- Group by verb type or command category
- Remove redundant examples

CONSOLIDATION RULES:
1. If 2+ entries describe the same core mechanic → MERGE into one entry
2. Location-specific variants → Convert to general pattern with examples
3. Obsolete information contradicted by new data → REMOVE
4. Cross-episode validated patterns → Move to CROSS-EPISODE INSIGHTS
5. Spatial facts (connections, locations) → Remove (handled by memory system)

Example consolidation:
❌ BEFORE (redundant):
- "Mathematical Crisis Death": 20+ turns without score progress...
- "Navigation Cycling Death": Parser-limited connections create loops...
- "Forest Path Cycling Death": Persistent navigation loops in forest...
- "House Area Mathematical Death": Extended house exploration...
[...15 more similar entries]

✅ AFTER (consolidated):
- "**Stuck-in-Area Death Pattern**": When 15+ turns pass without score progress, immediate area evacuation required. Recognition: repeated room descriptions, identical navigation loops, parser-limited connections. Prevention: Use unmapped directions immediately. High-risk areas include forest paths, house perimeter, any area with limited exits. Validated in Episodes 1, 3, 5, 7."

LENGTH TARGETS:
- Total knowledge base: 2000-2500 words
- DANGERS & THREATS: < 600 words
- PUZZLE SOLUTIONS: < 500 words
- STRATEGIC PATTERNS: < 600 words
- COMMAND SYNTAX: < 300 words
- CROSS-EPISODE INSIGHTS: < 800 words

QUALITY OVER QUANTITY:
- 5 well-written patterns > 20 redundant variants
- Actionable insights > verbose descriptions
- Parser patterns > location-specific tactics
- General mechanics > specific instances

**OUTPUT FORMAT:**
Provide the COMPLETE updated knowledge base with all sections, not just CROSS-EPISODE INSIGHTS.
Ensure all sections are consolidated and within word budgets.
Focus on strategic value and eliminate redundancy.

Structure:
# Zork Strategic Knowledge Base

## DANGERS & THREATS
[Consolidated content - max 5 patterns]

## PUZZLE SOLUTIONS
[Consolidated content - max 10 puzzles]

## STRATEGIC PATTERNS
[Consolidated content - max 8 patterns]

## DEATH & DANGER ANALYSIS
[Keep as-is, this is current episode specific]

## COMMAND SYNTAX
[Consolidated content - max 300 words]

## LESSONS LEARNED
[Update with current episode learnings]

## CROSS-EPISODE INSIGHTS
[Updated with validated patterns from this episode]

### Death Patterns Across Episodes
[Consolidated patterns with episode references]

### Environmental Knowledge
[Consolidated game mechanics and parser patterns]

### Strategic Meta-Patterns
[Consolidated decision-making heuristics]

### Major Discoveries
[Consolidated major findings]"""

    try:
        response = client.chat.completions.create(
            model=analysis_model,
            messages=[
                {
                    "role": "system",
                    "content": """You are performing full knowledge base consolidation for an AI agent playing Zork.

Your task has two parts:
1. Update CROSS-EPISODE INSIGHTS with validated patterns from the completed episode
2. Consolidate ALL sections to eliminate redundancy accumulated from incremental updates

Consolidation priorities:
- Merge duplicate patterns (same cause, different names)
- Convert location-specific variants to general patterns with examples
- Remove spatial memory duplicates (connections, locations, positions)
- Emphasize parser syntax patterns (highest value for LLM agents)
- Enforce strict entry/word limits per section

Success = Every unique insight preserved, 50% reduction in redundancy, actionable strategic intelligence.""",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=analysis_sampling.get("temperature"),
            top_p=analysis_sampling.get("top_p"),
            top_k=analysis_sampling.get("top_k"),
            min_p=analysis_sampling.get("min_p"),
            max_tokens=analysis_sampling.get("max_tokens") or 4000,
            name="StrategyGenerator",
        )

        # Get the full consolidated knowledge base (not just one section)
        consolidated_knowledge = response.content.strip()

        # Save the updated knowledge base directly
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(consolidated_knowledge)

            if logger:
                logger.info(
                    f"Cross-episode insights updated in {output_file}",
                    extra={
                        "event_type": "knowledge_update",
                        "details": f"Synthesized from episode with {turn_count} turns, score {final_score}",
                    },
                )

            return True

        except Exception as e:
            if logger:
                logger.error(
                    f"Failed to save updated knowledge base: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return False

    except Exception as e:
        if logger:
            logger.error(
                f"Cross-episode synthesis failed: {e}",
                extra={"event_type": "knowledge_update"},
            )
        return False
