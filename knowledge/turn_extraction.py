# ABOUTME: Turn data extraction from episode logs for knowledge analysis
# ABOUTME: Parses JSONL episode logs to extract actions, responses, events, and state changes

"""
Turn extraction module for knowledge management.

Provides utilities to extract turn-based data from episode log files,
including actions, responses, score/location/inventory changes, and death events.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Any


def get_episode_log_file(episode_id: str, workdir: str = "game_files") -> Path:
    """
    Get the log file path for a specific episode.

    Args:
        episode_id: The episode identifier
        workdir: Working directory containing episodes folder

    Returns:
        Path object pointing to the episode's log file
    """
    return Path(workdir) / "episodes" / episode_id / "episode_log.jsonl"


def extract_turn_window_data(
    episode_id: str,
    start_turn: int,
    end_turn: int,
    log_file: str,
    workdir: str = "game_files",
    logger=None
) -> Optional[Dict]:
    """
    Extract action-response data for a specific turn window.

    Parses episode log files to collect:
    - Action-response pairs with reasoning and critic scores
    - Score changes over time
    - Location changes (movement tracking)
    - Inventory changes
    - Death events with context
    - Game over events

    Args:
        episode_id: Episode identifier for the session
        start_turn: Starting turn number (inclusive)
        end_turn: Ending turn number (inclusive)
        log_file: Path to monolithic log file (fallback if episode log doesn't exist)
        workdir: Working directory containing episodes folder
        logger: Optional logger instance for warnings/errors

    Returns:
        Dictionary containing turn data with keys:
            - episode_id: Episode identifier
            - start_turn: Starting turn
            - end_turn: Ending turn
            - actions_and_responses: List of action-response pairs
            - score_changes: List of score change events
            - location_changes: List of location change events
            - inventory_changes: List of inventory change events
            - death_events: List of death events
            - game_over_events: List of all game over events
        Returns None if no data found for the turn window
    """
    turn_data = {
        "episode_id": episode_id,
        "start_turn": start_turn,
        "end_turn": end_turn,
        "actions_and_responses": [],
        "score_changes": [],
        "location_changes": [],
        "inventory_changes": [],
        "death_events": [],  # Track death events for knowledge base
        "game_over_events": [],  # Track all game over events
    }

    # Get episode-specific log file
    episode_log_file = get_episode_log_file(episode_id, workdir)
    if not episode_log_file.exists():
        # Fall back to monolithic file for backward compatibility
        episode_log_file = Path(log_file)

    try:
        with open(episode_log_file, "r", encoding="utf-8") as f:
            current_turn = 0
            current_score = 0
            current_location = ""

            # Store death messages temporarily for proper association
            death_messages_by_turn = {}

            for line in f:
                try:
                    log_entry = json.loads(line.strip())

                    # Skip entries not from this episode only if reading from monolithic file
                    if (
                        episode_log_file == Path(log_file)
                        and log_entry.get("episode_id") != episode_id
                    ):
                        continue

                    event_type = log_entry.get("event_type", "")

                    # Track turn progression - always update current_turn for this episode
                    if event_type == "turn_completed":
                        current_turn = log_entry.get("turn", 0)
                    elif event_type == "final_action_selection":
                        # Also update turn from action selection as backup
                        current_turn = log_entry.get("turn", current_turn)

                    # Collect action-response pairs - but only within our turn window
                    if event_type == "final_action_selection" and (
                        start_turn <= current_turn <= end_turn
                    ):
                        action_data = {
                            "turn": current_turn,
                            "action": log_entry.get("agent_action", ""),
                            "reasoning": log_entry.get("agent_reasoning", ""),
                            "critic_score": log_entry.get("critic_score", 0),
                            "response": "",  # Will be filled by next zork_response
                        }
                        turn_data["actions_and_responses"].append(action_data)

                    elif (
                        event_type == "zork_response"
                        and turn_data["actions_and_responses"]
                        and (start_turn <= current_turn <= end_turn)
                    ):
                        # Update the last action with its response
                        response = log_entry.get("zork_response", "")
                        turn_data["actions_and_responses"][-1]["response"] = (
                            response
                        )

                        # Check if this zork response contains death information and store it
                        if any(
                            death_indicator in response.lower()
                            for death_indicator in [
                                "you have died",
                                "you are dead",
                                "slavering fangs",
                                "eaten by a grue",
                                "you have been killed",
                                "****  you have died  ****",
                                "fatal",
                            ]
                        ):
                            action = log_entry.get("action", "")
                            # Create contextual description instead of bare action
                            death_context = (
                                f"{action} from {current_location}"
                                if current_location
                                else action
                            )
                            death_messages_by_turn[current_turn] = {
                                "detailed_death_message": response,
                                "death_context": death_context,
                                "death_location": current_location,
                                "fatal_action": action,  # Keep raw action for reference
                            }

                    # Only collect data within our turn window for other events
                    if not (start_turn <= current_turn <= end_turn):
                        continue

                    # Track death and game over events
                    if event_type in [
                        "game_over",
                        "game_over_final",
                        "death_during_inventory",
                    ]:
                        death_event = {
                            "turn": current_turn,
                            "event_type": event_type,
                            "reason": log_entry.get("reason", ""),
                            "action_taken": log_entry.get("action_taken", ""),
                            "final_score": log_entry.get(
                                "final_score", current_score
                            ),
                            "death_count": log_entry.get("death_count", 0),
                        }

                        # Add to both death_events and game_over_events for different analysis purposes
                        turn_data["game_over_events"].append(death_event)

                        # Check if this is specifically a death (vs victory)
                        reason = log_entry.get("reason", "").lower()
                        death_indicators = [
                            "died",
                            "death",
                            "eaten",
                            "grue",
                            "killed",
                            "fall",
                            "crushed",
                        ]
                        if any(
                            indicator in reason for indicator in death_indicators
                        ):
                            turn_data["death_events"].append(death_event)

                    # Track death state extraction for context
                    elif event_type == "death_state_extracted":
                        extracted_info = log_entry.get("extracted_info", {})
                        if extracted_info and turn_data["death_events"]:
                            # Add extraction details to the most recent death event
                            turn_data["death_events"][-1]["death_location"] = (
                                extracted_info.get("current_location_name", "")
                            )
                            turn_data["death_events"][-1]["death_objects"] = (
                                extracted_info.get("visible_objects", [])
                            )
                            turn_data["death_events"][-1]["death_messages"] = (
                                extracted_info.get("important_messages", [])
                            )

                    # Track score changes
                    elif event_type == "experience" and "zork_score" in log_entry:
                        new_score = log_entry.get("zork_score", 0)
                        if new_score != current_score:
                            turn_data["score_changes"].append(
                                {
                                    "turn": current_turn,
                                    "from_score": current_score,
                                    "to_score": new_score,
                                    "change": new_score - current_score,
                                }
                            )
                            current_score = new_score

                    # Track location changes
                    elif event_type == "extracted_info":
                        extracted_info = log_entry.get("extracted_info", {})
                        new_location = extracted_info.get(
                            "current_location_name", ""
                        )
                        if (
                            new_location
                            and new_location != current_location
                            and new_location != "Unknown Location"
                        ):
                            turn_data["location_changes"].append(
                                {
                                    "turn": current_turn,
                                    "from_location": current_location,
                                    "to_location": new_location,
                                }
                            )
                            current_location = new_location

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        if logger:
            logger.warning(
                f"Log file not found: {episode_log_file}",
                extra={"event_type": "knowledge_update"},
            )
        return None

    # Apply stored death messages to death events
    for turn_num, death_info in death_messages_by_turn.items():
        # Apply to death events
        for death_event in turn_data["death_events"]:
            if death_event["turn"] == turn_num:
                death_event.update(death_info)

        # Apply to game over events
        for game_over_event in turn_data["game_over_events"]:
            if game_over_event["turn"] == turn_num:
                game_over_event.update(death_info)

    return turn_data if turn_data["actions_and_responses"] else None


def episode_ended_in_loop_break(episode_id: str, workdir: str = "game_files") -> bool:
    """
    Check if episode ended due to Loop Break timeout (stuck_termination event).

    Loop Break system terminates episodes that are stuck without score progress
    for 20+ turns. These are system timeouts, NOT game mechanics/deaths.

    Args:
        episode_id: Episode identifier (e.g., "2025-11-08T23:14:34")
        workdir: Working directory containing episodes folder

    Returns:
        True if stuck_termination event found in episode log
    """
    episode_log_file = Path(workdir) / "episodes" / episode_id / "episode_log.jsonl"
    if not episode_log_file.exists():
        return False

    try:
        with open(episode_log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    if log_entry.get("event_type") == "stuck_termination":
                        return True
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return False

    return False
