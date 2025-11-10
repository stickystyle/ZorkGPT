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
    prompt = f"""Analyze this completed Zork episode and update the CROSS-EPISODE INSIGHTS section with UNIVERSAL strategic wisdom that persists across future episodes.

ARCHITECTURE REMINDER:
- **Memory System** handles location-specific procedures (e.g., "At Behind House, enter window")
- **Loop Break System** terminates stuck episodes (20+ turns no score) - NOT a game mechanic
- **Objective System** discovers and tracks goals automatically
- **Map System** manages spatial navigation and connections

CROSS-EPISODE INSIGHTS should contain ONLY universal strategic wisdom that applies regardless of location.

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

**EXISTING CROSS-EPISODE INSIGHTS:**
{existing_cross_episode if existing_cross_episode else "No previous cross-episode insights recorded."}

**SYNTHESIS TASK:**

Update the CROSS-EPISODE INSIGHTS section with universal strategic wisdom validated across episodes. Focus on:

1. **Validated Game Mechanics**: Universal rules confirmed across multiple episodes
   - Parser behaviors that always apply
   - Object interaction patterns
   - Scoring mechanics
   ✅ GOOD: "EXAMINE reveals hidden object properties"
   ❌ BAD: "Window at Behind House requires opening before entry" (location-specific → Memory System)

2. **Danger Recognition Patterns**: Warning signals and threat categories (NOT specific instances)
   - Universal warning text patterns
   - Danger category awareness
   - Death mechanic patterns
   ✅ GOOD: "Dark areas pose dangers - warning text includes 'pitch black'"
   ❌ BAD: "Troll Room is dangerous" (specific location → Memory System)
   ❌ BAD: "20 turns without score causes death" (Loop Break timeout, not game mechanic)

3. **Strategic Meta-Patterns**: Approaches consistently effective/ineffective across situations
   - Universal exploration strategies
   - Problem-solving heuristics
   - Resource management principles
   ✅ GOOD: "Multi-step procedures exist - verify prerequisites before attempting goals"
   ❌ BAD: "Always explore forest last" (location-specific tactic)

4. **Critical Discoveries**: Game mechanics or parser behaviors discovered
   - Universal game rules
   - Parser interpretation patterns
   - Action categories and properties
   ✅ GOOD: "Some actions are irreversible (GIVE, THROW, BREAK)"
   ❌ BAD: "Egg is retrieved by going up from Forest Path" (specific puzzle → Memory System)

**CRITICAL REQUIREMENTS:**
- **Universal Scope**: ONLY include knowledge applicable regardless of location
- **Principle Over Instance**: Extract patterns, not specific examples
- **System Awareness**: Ignore Loop Break timeouts as "mathematical deaths"
- **No Location Coupling**: If it requires "at Location X", it belongs in Memory System
- **Cross-Episode Validation**: Confirm patterns observed across multiple episodes

**SCOPE CHECKS:**
❌ If mentions specific locations by name → Memory System
❌ If documents "mathematical crisis/death" → Loop Break timeout
❌ If prescribes specific action sequences → Memory System
❌ If only relevant at one location → Memory System
✅ If universal principle applicable anywhere → Cross-Episode Insights
✅ If game mechanic confirmed across episodes → Cross-Episode Insights
✅ If danger category validated → Cross-Episode Insights

**OUTPUT FORMAT:**
Provide ONLY the updated CROSS-EPISODE INSIGHTS section content (without the ## header).
Structure with the four subsections above.
If no significant new universal insights emerged, return existing content unchanged."""

    try:
        response = client.chat.completions.create(
            model=analysis_model,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at extracting UNIVERSAL strategic wisdom from interactive fiction gameplay across multiple episodes.

CRITICAL CONTEXT - Other Systems Handle:
1. **Location-Specific Memory System**: Stores procedural knowledge at specific locations (e.g., "At Behind House, enter window leads to Kitchen")
2. **Loop Break System**: Programmatically terminates stuck episodes (20+ turns without score) - these are system timeouts, NOT game mechanics
3. **Objective System**: Discovers and tracks goals automatically
4. **Map System**: Manages spatial relationships and navigation

YOUR ROLE:
Extract ONLY universal game mechanics and strategic principles that apply REGARDLESS of location and are validated across multiple episodes.

SCOPE BOUNDARIES:
✅ INCLUDE: "EXAMINE reveals hidden information" (universal mechanic)
✅ INCLUDE: "Combat enemies exist" (danger category awareness)
✅ INCLUDE: "Multi-step procedures exist" (meta-pattern)
❌ EXCLUDE: "At Behind House, open window then enter window" (location-specific → Memory System)
❌ EXCLUDE: "Extended exploration causes mathematical death" (Loop Break timeout → System behavior)
❌ EXCLUDE: "Troll at Troll Room is dangerous" (specific instance → Memory System)

If knowledge requires phrases like "at Location X" or "in Room Y", it belongs in the Memory System instead.

Focus on UNIVERSAL patterns validated across episodes, not location-specific tactics or single-episode observations.""",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=analysis_sampling.get("temperature"),
            top_p=analysis_sampling.get("top_p"),
            top_k=analysis_sampling.get("top_k"),
            min_p=analysis_sampling.get("min_p"),
            max_tokens=analysis_sampling.get("max_tokens") or 2000,
            name="StrategyGenerator",
        )

        new_cross_episode_content = response.content.strip()

        # Update the knowledge base with new cross-episode section
        updated_knowledge = update_section_content(
            existing_knowledge,
            "CROSS-EPISODE INSIGHTS",
            new_cross_episode_content
        )

        # Save the updated knowledge base
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(updated_knowledge)

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
