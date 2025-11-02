# ABOUTME: Quality assessment logic for knowledge update decisions
# ABOUTME: Determines when knowledge updates should occur based on turn data variety and content

"""
Quality assessment module for knowledge management.

Provides heuristics and logic to determine when knowledge updates should occur
based on turn data quality, action variety, and content significance.
"""

import os
from typing import Dict, Tuple, Optional


def should_update_knowledge(turn_data: Dict, logger=None) -> Tuple[bool, str]:
    """
    Determine if turn data warrants a knowledge update using simple heuristics.

    Uses a sliding window approach to evaluate recent action variety rather than
    entire episode variety, preventing permanent blocking in long episodes.

    Args:
        turn_data: Dictionary containing actions_and_responses, death_events,
                   score_changes, location_changes, and episode_id
        logger: Optional logger for diagnostic information

    Returns:
        Tuple[bool, str]: (should_update, reason)
    """
    actions = turn_data["actions_and_responses"]

    # Always require minimum actions
    if len(actions) < 3:
        return False, "Too few actions (< 3)"

    # Always process death events (high learning value)
    if turn_data.get("death_events"):
        return True, f"Contains {len(turn_data['death_events'])} death event(s)"

    # Process if meaningful progress occurred
    if turn_data.get("score_changes"):
        return True, f"Score changed {len(turn_data['score_changes'])} times"

    if turn_data.get("location_changes"):
        return (
            True,
            f"Discovered {len(turn_data['location_changes'])} new locations",
        )

    # Check action variety using sliding window approach (last 75 turns)
    # This prevents permanent blocking in long episodes where episode-wide
    # variety naturally decreases over time
    window_size = min(75, len(actions))
    recent_actions = actions[-window_size:]

    # Calculate variety in recent window
    recent_unique = set(a["action"] for a in recent_actions)
    recent_variety = len(recent_unique) / len(recent_actions)

    # Also calculate episode-wide metrics for logging
    all_unique = set(a["action"] for a in actions)
    episode_variety = len(all_unique) / len(actions)

    # Detect stuck patterns - consecutive similar actions
    # Force update if agent is stuck, even if variety is low
    if len(recent_actions) >= 10:
        last_10_actions = [a["action"] for a in recent_actions[-10:]]
        unique_last_10 = len(set(last_10_actions))

        # If doing the same ~3 actions repeatedly, force update to help learn
        if unique_last_10 <= 3:
            if logger:
                logger.info(
                    "Stuck pattern detected - forcing knowledge update",
                    extra={
                        "event_type": "knowledge_update_quality",
                        "episode_id": turn_data.get("episode_id", "unknown"),
                        "stuck_pattern_unique_actions": unique_last_10,
                        "window_size": window_size,
                        "recent_variety": f"{recent_variety:.1%}",
                        "episode_variety": f"{episode_variety:.1%}",
                    }
                )
            return True, f"Stuck pattern detected (only {unique_last_10} unique actions in last 10 turns) - forcing update"

    # Use lower threshold (15%) for window-based variety
    # (lower than episode-wide 30% because window is smaller)
    if recent_variety < 0.15:
        if logger:
            logger.info(
                "Knowledge update decision: skip - Too repetitive",
                extra={
                    "event_type": "knowledge_update_quality",
                    "episode_id": turn_data.get("episode_id", "unknown"),
                    "window_size": window_size,
                    "recent_variety": f"{recent_variety:.1%}",
                    "episode_variety": f"{episode_variety:.1%}",
                    "recent_unique_count": len(recent_unique),
                    "total_actions": len(actions),
                    "threshold_used": "15%",
                    "decision": "skip",
                }
            )
        return False, f"Too repetitive in recent window ({recent_variety:.1%} unique actions in last {window_size} turns)"

    # Check response variety (ensure new information)
    unique_responses = set(a["response"][:50] for a in actions)

    if len(unique_responses) < 2:
        return False, "No new information in responses"

    # Check for meaningful content in responses
    total_response_length = sum(len(a["response"]) for a in actions)
    if total_response_length < 100:
        return False, "Responses too short/uninformative"

    # Log successful quality check
    if logger:
        logger.info(
            "Knowledge update decision: proceed - Varied gameplay",
            extra={
                "event_type": "knowledge_update_quality",
                "episode_id": turn_data.get("episode_id", "unknown"),
                "window_size": window_size,
                "recent_variety": f"{recent_variety:.1%}",
                "episode_variety": f"{episode_variety:.1%}",
                "recent_unique_count": len(recent_unique),
                "total_actions": len(actions),
                "threshold_used": "15%",
                "decision": "proceed",
            }
        )

    return True, f"Varied gameplay ({len(recent_unique)} unique actions in last {window_size} turns)"


def is_first_meaningful_update(output_file: str, logger=None) -> bool:
    """
    Check if this is the first meaningful knowledge update.

    Returns True if:
    1. No knowledge base exists, OR
    2. Knowledge base only contains auto-generated basic content (map + basic strategy)

    This handles the case where knowledgebase.md is auto-created for map updates
    but doesn't contain any LLM-generated strategic insights yet.

    Args:
        output_file: Path to the knowledge base file
        logger: Optional logger for diagnostic information

    Returns:
        True if this is the first meaningful update, False otherwise
    """
    if not os.path.exists(output_file):
        return True

    if os.path.getsize(output_file) == 0:
        return True

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if content only contains basic auto-generated sections
        # Look for indicators of LLM-generated strategic content

        # Basic strategy indicators that suggest auto-generated content
        basic_indicators = [
            "Always begin each location with 'look'",
            "Use systematic exploration patterns",
            "Execute 'take' commands for all portable items",
            "Parse all text output for puzzle-solving information",
            "Prioritize information extraction over rapid action execution",
        ]

        # Count how many basic indicators are present
        basic_indicator_count = sum(
            1 for indicator in basic_indicators if indicator in content
        )

        # If content is very short and mostly contains basic indicators, treat as first update
        content_lines = [
            line.strip() for line in content.split("\n") if line.strip()
        ]
        meaningful_content_lines = [
            line
            for line in content_lines
            if not line.startswith("#") and len(line) > 10
        ]

        # Heuristics for detecting auto-generated vs LLM-generated content:
        # 1. Very few meaningful content lines (< 10)
        # 2. High ratio of basic indicators to total content
        # 3. No complex strategic insights (no sentences > 100 chars with specific game references)

        if len(meaningful_content_lines) < 10:
            return True

        if basic_indicator_count >= 3 and len(meaningful_content_lines) < 15:
            return True

        # Look for complex strategic insights (longer sentences with game-specific terms)
        complex_insights = [
            line
            for line in meaningful_content_lines
            if len(line) > 80
            and any(
                term in line.lower()
                for term in [
                    "puzzle",
                    "treasure",
                    "combat",
                    "inventory",
                    "specific",
                    "strategy",
                    "avoid",
                    "danger",
                    "death",
                    "troll",
                    "grue",
                    "lamp",
                    "sword",
                ]
            )
        ]

        # If no complex insights found, likely still basic content
        if len(complex_insights) == 0:
            return True

        return False

    except Exception as e:
        if logger:
            logger.warning(
                f"Error checking knowledge base content: {e}",
                extra={"event_type": "knowledge_update"},
            )
        # If we can't read it, assume it's not meaningful yet
        return True
