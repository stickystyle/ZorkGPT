"""
AdaptiveKnowledgeManager module for turn-based knowledge management.

This module provides LLM-first adaptive knowledge management for ZorkGPT,
using turn-based sliding windows instead of episode-based analysis.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from map_graph import MapGraph
from movement_analyzer import MovementAnalyzer, create_movement_context
from llm_client import LLMClientWrapper
from config import get_config, get_client_api_key
import re
from pathlib import Path

# Import shared utilities
from shared_utils import estimate_tokens


class AdaptiveKnowledgeManager:
    """
    LLM-first adaptive knowledge management system for turn-based updates.

    This system lets the LLM manage its own knowledge lifecycle by:
    1. Assessing the quality of potential knowledge updates
    2. Determining appropriate update strategies
    3. Performing intelligent knowledge merging
    4. Self-monitoring for knowledge degradation
    """

    def __init__(
        self,
        log_file: str = "zork_episode_log.jsonl",
        output_file: str = "knowledgebase.md",
        logger=None,
        workdir: str = "game_files",
    ):
        self.log_file = log_file
        self.output_file = output_file
        self.logger = logger
        self.workdir = workdir

        # Initialize LLM client
        config = get_config()
        self.client = LLMClientWrapper(
            base_url=config.llm.get_base_url_for_model("analysis"),
            api_key=get_client_api_key(),
        )

        # Models for different tasks
        self.analysis_model = config.llm.analysis_model
        self.info_ext_model = config.llm.info_ext_model

        # Load sampling parameters from configuration
        self.analysis_sampling = config.analysis_sampling
        self.extractor_sampling = config.extractor_sampling

        # Turn-based configuration
        self.turn_window_size = config.gameplay.turn_window_size
        self.min_quality_threshold = config.gameplay.min_knowledge_quality

        # Knowledge base condensation configuration
        self.enable_condensation = config.gameplay.enable_knowledge_condensation
        self.condensation_threshold = config.gameplay.knowledge_condensation_threshold

        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = config.logging.enable_prompt_logging

        # Load agent instructions to avoid duplication
        self.agent_instructions = self._load_agent_instructions()

    def _get_all_episode_ids(self) -> List[str]:
        """
        Scan episodes directory and return all episode IDs in chronological order.
        """
        episodes_dir = Path(self.workdir) / "episodes"
        if not episodes_dir.exists():
            return []

        # Get all episode directories
        episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]

        # Sort chronologically (episode IDs are ISO8601 timestamps)
        episode_ids = sorted([d.name for d in episode_dirs])

        return episode_ids

    def _get_episode_log_file(self, episode_id: str) -> Path:
        """Get the log file path for a specific episode."""
        return Path(self.workdir) / "episodes" / episode_id / "episode_log.jsonl"

    def process_all_episodes_chronologically(self) -> List[Dict]:
        """
        Process all episodes in chronological order.
        """
        all_episode_data = []
        episode_ids = self._get_all_episode_ids()

        for episode_id in episode_ids:
            # Get total turns for this episode
            episode_log_file = self._get_episode_log_file(episode_id)
            if not episode_log_file.exists():
                continue

            # Extract data for entire episode
            with open(episode_log_file, "r", encoding="utf-8") as f:
                max_turn = 0
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if log_entry.get("event_type") == "turn_start":
                            turn = log_entry.get("turn", 0)
                            max_turn = max(max_turn, turn)
                    except json.JSONDecodeError:
                        continue

            if max_turn > 0:
                episode_data = self._extract_turn_window_data(episode_id, 1, max_turn)
                if episode_data:
                    all_episode_data.append(episode_data)

        return all_episode_data

    def _log_prompt_to_file(
        self, messages: List[Dict], prefix: str = "knowledge"
    ) -> None:
        """Log the full prompt to a temporary file for evaluation."""
        if not self.enable_prompt_logging:
            return

        self.prompt_counter += 1
        filename = f"tmp/{prefix}_{self.prompt_counter:03d}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"=== {prefix.upper()} PROMPT #{self.prompt_counter} ===\n")
                f.write(f"Model: {self.analysis_model}\n")
                f.write("=" * 50 + "\n\n")

                for i, message in enumerate(messages):
                    f.write(f"--- MESSAGE {i + 1} ({message['role'].upper()}) ---\n")
                    f.write(message["content"])
                    f.write("\n\n")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to log prompt to {filename}: {e}",
                    extra={"event_type": "knowledge_update"},
                )

    def _load_agent_instructions(self) -> str:
        """Load the agent.md prompt to understand what's already covered."""
        try:
            with open("agent.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Could not load agent.md: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return ""

    def _is_first_meaningful_update(self) -> bool:
        """
        Check if this is the first meaningful knowledge update.

        Returns True if:
        1. No knowledge base exists, OR
        2. Knowledge base only contains auto-generated basic content (map + basic strategy)

        This handles the case where knowledgebase.md is auto-created for map updates
        but doesn't contain any LLM-generated strategic insights yet.
        """
        if not os.path.exists(self.output_file):
            return True

        if os.path.getsize(self.output_file) == 0:
            return True

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if content only contains basic auto-generated sections
            # Look for indicators of LLM-generated strategic content

            # Remove map section for analysis
            content_without_map = self._trim_map_section(content)

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
                1 for indicator in basic_indicators if indicator in content_without_map
            )

            # If content is very short and mostly contains basic indicators, treat as first update
            content_lines = [
                line.strip() for line in content_without_map.split("\n") if line.strip()
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
            if self.logger:
                self.logger.warning(
                    f"Error checking knowledge base content: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            # If we can't read it, assume it's not meaningful yet
            return True

    def update_knowledge_from_turns(
        self,
        episode_id: str,
        start_turn: int,
        end_turn: int,
        is_final_update: bool = False,
    ) -> bool:
        """
        Update knowledge base from a specific turn range using single-stage generation.

        Args:
            episode_id: Current episode ID
            start_turn: Starting turn number
            end_turn: Ending turn number
            is_final_update: If True, more lenient about quality (episode-end updates)

        Returns:
            bool: True if knowledge was updated, False if skipped
        """
        if self.logger:
            self.logger.info(
                f"Knowledge update requested for turns {start_turn}-{end_turn}",
                extra={
                    "event_type": "knowledge_update_start",
                    "episode_id": episode_id,
                    "turn_range": f"{start_turn}-{end_turn}",
                    "is_final": is_final_update,
                },
            )

        # Step 1: Extract turn window data
        turn_data = self._extract_turn_window_data(episode_id, start_turn, end_turn)

        if not turn_data or not turn_data.get("actions_and_responses"):
            if self.logger:
                self.logger.warning(
                    "No turn data found for analysis",
                    extra={
                        "event_type": "knowledge_update_skipped",
                        "episode_id": episode_id,
                        "reason": "no_data",
                    },
                )
            return False

        # Step 2: Quality check using heuristics
        should_update, reason = self._should_update_knowledge(turn_data)

        # Override for final updates if there's a death or significant content
        if is_final_update and not should_update:
            if (
                turn_data.get("death_events")
                or len(turn_data["actions_and_responses"]) >= 5
            ):
                should_update = True
                reason = "Final update with significant content"

        if self.logger:
            self.logger.info(
                f"Knowledge update decision: {'proceed' if should_update else 'skip'} - {reason}",
                extra={
                    "event_type": "knowledge_update_decision",
                    "episode_id": episode_id,
                    "should_update": should_update,
                    "reason": reason,
                },
            )

        if not should_update:
            return False

        # Step 3: Load existing knowledge
        existing_knowledge = ""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, "r", encoding="utf-8") as f:
                    existing_knowledge = f.read()

                # Trim map section for LLM processing
                existing_knowledge = self._trim_map_section(existing_knowledge)

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Could not load existing knowledge: {e}",
                    extra={"event_type": "knowledge_update", "episode_id": episode_id},
                )

        # Step 4: Generate new knowledge in single pass
        if self.logger:
            self.logger.info(
                "Generating knowledge base update",
                extra={
                    "event_type": "knowledge_generation_start",
                    "episode_id": episode_id,
                },
            )

        new_knowledge = self._generate_knowledge_directly(turn_data, existing_knowledge)

        if not new_knowledge or new_knowledge.startswith("SKIP:"):
            if self.logger:
                self.logger.warning(
                    f"Knowledge generation returned skip or empty: {new_knowledge[:100]}",
                    extra={
                        "event_type": "knowledge_update_skipped",
                        "episode_id": episode_id,
                    },
                )
            return False

        # Step 5: Preserve map section if it exists
        if existing_knowledge and "## CURRENT WORLD MAP" in existing_knowledge:
            # Extract and preserve the map section
            original_with_map = ""
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    original_with_map = f.read()
                new_knowledge = self._preserve_map_section(
                    original_with_map, new_knowledge
                )
            except:
                pass  # Map preservation is non-critical

        # Step 6: Write updated knowledge to file
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(new_knowledge)

            if self.logger:
                self.logger.info(
                    "Knowledge base updated successfully",
                    extra={
                        "event_type": "knowledge_update_success",
                        "episode_id": episode_id,
                        "file": self.output_file,
                        "size": len(new_knowledge),
                    },
                )

            # Step 7: Log the prompt if in debug mode
            if hasattr(self, "log_prompts") and self.log_prompts:
                self._log_prompt_to_file("knowledge_update", turn_data, new_knowledge)

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Failed to write knowledge base: {e}",
                    extra={
                        "event_type": "knowledge_update_error",
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )
            return False

    def _extract_turn_window_data(
        self, episode_id: str, start_turn: int, end_turn: int
    ) -> Optional[Dict]:
        """Extract action-response data for a specific turn window."""
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
        episode_log_file = self._get_episode_log_file(episode_id)
        if not episode_log_file.exists():
            # Fall back to monolithic file for backward compatibility
            episode_log_file = self.log_file

        try:
            with open(episode_log_file, "r", encoding="utf-8") as f:
                current_turn = 0
                current_score = 0
                current_location = ""
                current_inventory = []

                # Store death messages temporarily for proper association
                death_messages_by_turn = {}

                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # Skip entries not from this episode only if reading from monolithic file
                        if (
                            episode_log_file == self.log_file
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
            if self.logger:
                self.logger.warning(
                    f"Log file {self.log_file} not found",
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

    def _should_update_knowledge(self, turn_data: Dict) -> Tuple[bool, str]:
        """
        Determine if turn data warrants a knowledge update using simple heuristics.

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

        # Check action variety (avoid pure repetition)
        unique_actions = set(a["action"] for a in actions)
        action_variety = len(unique_actions) / len(actions)

        if action_variety < 0.3:  # Less than 30% unique actions
            return False, f"Too repetitive ({action_variety:.1%} unique actions)"

        # Check response variety (ensure new information)
        unique_responses = set(a["response"][:50] for a in actions)

        if len(unique_responses) < 2:
            return False, "No new information in responses"

        # Check for meaningful content in responses
        total_response_length = sum(len(a["response"]) for a in actions)
        if total_response_length < 100:
            return False, "Responses too short/uninformative"

        return True, f"Varied gameplay ({len(unique_actions)} unique actions)"

    def _format_turn_data_for_prompt(self, turn_data: Dict) -> str:
        """Format turn data for LLM prompt with clear structure."""

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

    def _load_persistent_wisdom(self) -> str:
        """
        Load persistent wisdom from previous episodes.

        Returns:
            str: Formatted persistent wisdom or empty string if not available
        """
        try:
            from config import get_config

            config = get_config()
            persistent_wisdom_file = config.orchestrator.persistent_wisdom_file

            with open(persistent_wisdom_file, "r", encoding="utf-8") as f:
                wisdom = f.read().strip()

            if wisdom:
                return f"""
**PERSISTENT WISDOM FROM PREVIOUS EPISODES:**
{"-" * 50}
{wisdom}
{"-" * 50}
"""

        except FileNotFoundError:
            # No persistent wisdom file yet - this is fine for early episodes
            if self.logger:
                self.logger.debug(
                    "No persistent wisdom file found (normal for early episodes)"
                )
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Could not load persistent wisdom: {e}",
                    extra={"event_type": "knowledge_update"},
                )

        return ""

    def _format_death_analysis_section(self, turn_data: Dict) -> str:
        """Format death events for the knowledge base."""
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

    def _generate_knowledge_directly(
        self, turn_data: Dict, existing_knowledge: str
    ) -> str:
        """
        Generate knowledge base content in a single LLM call.

        Args:
            turn_data: Extracted turn data
            existing_knowledge: Current knowledge base content (if any)

        Returns:
            str: Complete knowledge base content
        """
        # Format turn data
        formatted_data = self._format_turn_data_for_prompt(turn_data)

        # Load persistent wisdom for context
        persistent_wisdom = self._load_persistent_wisdom()

        # Construct comprehensive prompt
        prompt = f"""Analyze this Zork gameplay data and create/update the knowledge base.

{formatted_data}

EXISTING KNOWLEDGE BASE:
{"-" * 50}
{existing_knowledge if existing_knowledge else "No existing knowledge - this is the first update"}
{"-" * 50}

{persistent_wisdom}

INSTRUCTIONS:
Create a comprehensive knowledge base with ALL of the following sections. If a section has no new information, keep the existing content for that section.

## WORLD KNOWLEDGE
List ALL specific facts discovered about the game world:
- **Item Locations**: Exact items and where found (e.g., "mailbox at West of House contains leaflet")
- **Room Connections**: Specific navigation paths (e.g., "north from West of House → North of House")
- **Dangers**: Specific threats and their locations (e.g., "grue in darkness east of North of House")
- **Object Interactions**: What happens with objects (e.g., "leaflet can be read, contains game introduction")
- **Puzzle Solutions**: Any puzzles solved and their solutions
- **Environmental Details**: Properties of locations, special features

## STRATEGIC PATTERNS
Identify patterns from this gameplay session:
- **Successful Actions**: What specific actions led to progress?
- **Failed Approaches**: What didn't work and why?
- **Exploration Strategies**: Effective methods for discovering new areas
- **Resource Management**: How to use items effectively
- **Objective Recognition**: How to identify new goals from game responses

## DEATH & DANGER ANALYSIS
{self._format_death_analysis_section(turn_data) if turn_data.get("death_events") else "No deaths occurred in this session."}

## COMMAND SYNTAX
List exact commands that worked:
- Movement: [specific successful movement commands]
- Interaction: [specific object interaction commands]
- Combat: [any combat-related commands]
- Special: [any special or unusual commands]

## LESSONS LEARNED
Specific insights from this session:
- **New Discoveries**: What was learned for the first time?
- **Confirmed Patterns**: What previous knowledge was validated?
- **Updated Understanding**: What previous assumptions were corrected?
- **Future Strategies**: What should be tried next based on these learnings?

## CROSS-EPISODE INSIGHTS
How this session relates to persistent wisdom:
- **Confirmations**: Which persistent patterns were observed again?
- **Contradictions**: What differed from previous episodes?
- **Extensions**: What new details extend existing knowledge?

CRITICAL REQUIREMENTS:
1. **Be Specific**: Include exact names, locations, and commands
2. **Preserve Details**: Never generalize specific facts into vague advice
3. **Additive Updates**: When updating, ADD new facts, don't remove existing ones
4. **Fact-First**: Prioritize concrete discoveries over abstract strategies
5. **Complete Sections**: Include all sections even if some have minimal updates

Remember: The agent needs BOTH specific facts ("mailbox contains leaflet") AND strategic insights ("reading items provides information")."""

        # Add qwen3-30b-a3b optimization if needed
        prompt = r"\no_think " + prompt

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are creating a knowledge base for an AI agent playing Zork. Focus on preserving specific, actionable facts from the gameplay while also identifying strategic patterns. Never abstract specific discoveries into generic advice.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 3000,
            )

            return response.content.strip()

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Knowledge generation failed: {e}",
                    extra={"event_type": "knowledge_update", "error": str(e)},
                )
            # Return existing knowledge on failure
            return existing_knowledge

    def _consolidate_existing_knowledge(self) -> Optional[str]:
        """Consolidate and improve existing knowledge without new data."""

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                full_knowledge = f.read()
        except:
            return None

        # Trim map section for LLM processing (map is handled separately)
        current_knowledge = self._trim_map_section(full_knowledge)

        prompt = f"""Review and consolidate this Zork strategy guide to improve its quality and organization.

**IMPORTANT**: This knowledge base is for an AI language model, not a human player.
- Remove any human-centric advice (drawing maps, taking notes, etc.)
- Focus on computational decision-making patterns
- Use precise command syntax and logical conditions
- Provide algorithmic approaches to problem-solving

CURRENT KNOWLEDGE BASE:
{current_knowledge}

Tasks:
1. Remove any contradictory or outdated information
2. Consolidate redundant advice into clearer statements
3. Improve organization and clarity
4. Ensure all advice is actionable and specific for an AI agent
5. **Remove human-centric references** (paper mapping, manual note-taking, etc.)
6. **Focus on algorithmic decision patterns** that an LLM can follow

Maintain the same general structure but improve the content quality.
Do not add new information - only reorganize and clarify existing knowledge for AI consumption.

**IMPORTANT**: Do not add any meta-commentary about the knowledge base structure or organization. Do not include sections like "Updated Knowledge Base Structure" or explanations of how the content is organized. Simply provide the improved content directly."""

        # Incase using Qwen qwen3-30b-a3b
        prompt = r"\no_think " + prompt
        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at consolidating and organizing strategic knowledge for AI language models playing interactive fiction games. Focus on algorithmic approaches and computational patterns rather than human-centric advice.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 3000,
            )

            return response.content.strip()

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Knowledge consolidation failed: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return None

    def _trim_map_section(self, knowledge_content: str) -> str:
        """Remove the map section from knowledge content for LLM processing."""
        if not knowledge_content or "## CURRENT WORLD MAP" not in knowledge_content:
            return knowledge_content

        # Remove the mermaid diagram section more precisely
        # Look for the pattern: ## CURRENT WORLD MAP followed by ```mermaid...```
        pattern = r"## CURRENT WORLD MAP\s*\n\s*```mermaid\s*\n.*?\n```"

        # Remove the mermaid diagram section while preserving other content
        knowledge_only = re.sub(pattern, "", knowledge_content, flags=re.DOTALL)

        # Clean up any extra whitespace that might be left
        knowledge_only = re.sub(r"\n\s*\n\s*\n", "\n\n", knowledge_only)

        return knowledge_only.strip()

    def _preserve_map_section(self, original_knowledge: str, new_knowledge: str) -> str:
        """Preserve the map section from original knowledge in the new knowledge."""
        if not original_knowledge or "## CURRENT WORLD MAP" not in original_knowledge:
            return new_knowledge

        # Extract map section from original
        map_start = original_knowledge.find("## CURRENT WORLD MAP")
        if map_start == -1:
            return new_knowledge

        map_section = original_knowledge[map_start:]

        # Add map section to new knowledge
        return f"{new_knowledge.rstrip()}\n\n{map_section}"

    def _condense_knowledge_base(self, verbose_knowledge: str) -> Optional[str]:
        """
        Use the info_ext_model to condense a knowledge base into a more concise format.

        This step focuses purely on reformatting and removing redundancy without
        adding new strategies or losing critical information.

        Args:
            verbose_knowledge: The full knowledge base content (without map section)

        Returns:
            Condensed knowledge base or None if condensation failed
        """

        if not verbose_knowledge or len(verbose_knowledge) < 1000:
            # Don't condense if content is already short
            return verbose_knowledge

        prompt = f"""You are tasked with condensing this Zork strategy guide into a more concise format while preserving ALL critical information.

**CRITICAL REQUIREMENTS**:
1. **NO NEW STRATEGIES**: Only reformat existing content - never invent or add new strategic advice
2. **PRESERVE ALL KEY INFORMATION**: Every important insight, danger warning, item detail, and strategic pattern must be retained
3. **REMOVE REDUNDANCY**: Eliminate repetitive statements and merge similar advice
4. **MAINTAIN STRUCTURE**: Keep the logical organization but make it more compact
5. **AI-FOCUSED LANGUAGE**: Use direct, actionable instructions for an AI language model
6. **CONSOLIDATE EXAMPLES**: Merge similar examples or scenarios into representative cases

**CURRENT KNOWLEDGE BASE TO CONDENSE**:
{verbose_knowledge}

**CONDENSATION GUIDELINES**:
- Merge repetitive advice into single, comprehensive statements
- Combine similar examples or scenarios into representative cases  
- Use bullet points and concise formatting for better readability
- Eliminate verbose explanations while keeping essential details
- Maintain all specific game elements (locations, items, commands, dangers)
- Preserve the strategic frameworks and decision-making patterns
- Keep all unique insights and specialized knowledge

**OUTPUT FORMAT**: Provide a condensed version that is 50-70% of the original length while maintaining 100% of the strategic value.

Focus on creating a guide that is information-dense but highly readable for an AI agent during gameplay.

**IMPORTANT**: Do not add any meta-commentary about the knowledge base structure or organization. Do not include sections like "Updated Knowledge Base Structure" or explanations of how the content is organized. Simply provide the condensed content directly."""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert technical writer specializing in condensing strategic guides for AI systems. Your goal is to maximize information density while preserving completeness and accuracy. Never add new information - only reorganize and consolidate existing content.",
                },
                {"role": "user", "content": prompt},
            ]

            # Log the condensation prompt if enabled
            self._log_prompt_to_file(messages, "knowledge_condensation")

            response = self.client.chat.completions.create(
                model=self.info_ext_model,
                messages=messages,
                temperature=self.extractor_sampling.temperature,
                top_p=getattr(self.extractor_sampling, "top_p", None),
                top_k=getattr(self.extractor_sampling, "top_k", None),
                min_p=getattr(self.extractor_sampling, "min_p", None),
                max_tokens=self.analysis_sampling.max_tokens or 5000,
            )

            condensed_content = response.content.strip()

            # Validate that condensation was successful and actually shorter
            if condensed_content and len(condensed_content) < len(verbose_knowledge):
                # Provide both character and token estimates for better feedback
                original_tokens = estimate_tokens(verbose_knowledge)
                condensed_tokens = estimate_tokens(condensed_content)

                if self.logger:
                    self.logger.info(
                        f"Knowledge condensed: {len(verbose_knowledge)} -> {len(condensed_content)} characters ({len(condensed_content) / len(verbose_knowledge) * 100:.1f}%)",
                        extra={
                            "event_type": "knowledge_update",
                            "details": f"Token estimate: {original_tokens} -> {condensed_tokens} tokens ({condensed_tokens / original_tokens * 100:.1f}%)",
                        },
                    )
                return condensed_content
            else:
                if self.logger:
                    self.logger.warning(
                        "Condensation failed or didn't reduce size - keeping original",
                        extra={"event_type": "knowledge_update"},
                    )
                return verbose_knowledge

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Knowledge condensation failed: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return verbose_knowledge  # Return original on failure

    def update_knowledge_with_map(self, episode_id: str, game_map: MapGraph) -> bool:
        """
        Update the knowledge base with current map information.

        Args:
            episode_id: Current episode ID
            game_map: The current MapGraph instance

        Returns:
            True if map was updated, False if skipped
        """
        if self.logger:
            self.logger.info(
                "Updating knowledge base with current map",
                extra={"event_type": "knowledge_update"},
            )

        # Generate mermaid diagram from current map
        mermaid_map = game_map.render_mermaid()
        if not mermaid_map or not mermaid_map.strip():
            if self.logger:
                self.logger.warning(
                    "No map data available to update",
                    extra={"event_type": "knowledge_update"},
                )
            return False

        # Load existing knowledge
        existing_knowledge = ""
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                existing_knowledge = f.read()
        except:
            existing_knowledge = ""

        # Update or add map section
        updated_knowledge = self._update_map_section(existing_knowledge, mermaid_map)

        if not updated_knowledge:
            if self.logger:
                self.logger.error(
                    "Failed to update map section",
                    extra={"event_type": "knowledge_update"},
                )
            return False

        # Save updated knowledge
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(updated_knowledge)
            if self.logger:
                self.logger.info(
                    "Map section updated in knowledge base",
                    extra={"event_type": "knowledge_update"},
                )
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Failed to save updated knowledge: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return False

    def _update_map_section(
        self, existing_knowledge: str, mermaid_map: str
    ) -> Optional[str]:
        """Update or add the map section to the knowledge base."""

        map_section = f"""

## CURRENT WORLD MAP

```mermaid
{mermaid_map}
```

"""

        # Check if there's already a map section
        if "## CURRENT WORLD MAP" in existing_knowledge:
            # Replace existing map section
            lines = existing_knowledge.split("\n")
            new_lines = []
            in_map_section = False
            in_mermaid_block = False

            for line in lines:
                if line.strip() == "## CURRENT WORLD MAP":
                    in_map_section = True
                    new_lines.append(line)
                    continue

                if in_map_section:
                    if line.strip().startswith("```mermaid"):
                        in_mermaid_block = True
                        new_lines.append(line)
                        new_lines.append(mermaid_map)
                        continue
                    elif line.strip() == "```" and in_mermaid_block:
                        in_mermaid_block = False
                        new_lines.append(line)
                        in_map_section = False
                        continue
                    elif line.strip().startswith("##") and not in_mermaid_block:
                        # Hit next section, end map section
                        in_map_section = False
                        new_lines.append(line)
                        continue
                    elif not in_mermaid_block:
                        # Skip old map content outside mermaid block
                        continue
                    else:
                        # Skip old mermaid content
                        continue
                else:
                    new_lines.append(line)

            return "\n".join(new_lines)
        else:
            # Add new map section at the end
            if existing_knowledge.strip():
                return existing_knowledge.rstrip() + map_section
            else:
                # Create minimal knowledge base with just the map
                return f"""# Zork Game World Knowledge Base

This knowledge base contains discovered information about the Zork game world, including specific items, puzzles, dangers, and strategic insights learned through gameplay.

{map_section}"""

    def _build_map_from_logs(self, episode_id: str) -> Optional[str]:
        """
        Build a mermaid map from log data for a specific episode.

        Args:
            episode_id: Episode ID to extract map data for

        Returns:
            Mermaid diagram string or None if failed
        """
        try:
            # Create a temporary MapGraph to build from logs
            temp_map = MapGraph()

            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # Skip entries not from this episode
                        if log_entry.get("episode_id") != episode_id:
                            continue

                        event_type = log_entry.get("event_type", "")

                        if event_type == "extracted_info":
                            extracted_info = log_entry.get("extracted_info", {})
                            location_name = extracted_info.get(
                                "current_location_name", ""
                            )
                            exits = extracted_info.get("exits", [])

                            if location_name and location_name != "Unknown Location":
                                # Add room and exits
                                temp_map.add_room(location_name)
                                temp_map.update_room_exits(location_name, exits)

                        elif event_type == "movement_connection_created":
                            from_room = log_entry.get("from_room", "")
                            to_room = log_entry.get("to_room", "")
                            action = log_entry.get("action", "")

                            if from_room and to_room and action:
                                temp_map.add_connection(from_room, action, to_room)

                    except json.JSONDecodeError:
                        continue

            # Generate mermaid representation
            mermaid_map = temp_map.render_mermaid()
            return mermaid_map if mermaid_map and mermaid_map.strip() else None

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Failed to build map from logs: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return None

    def update_knowledge_section(
        self, section_id: str, content: str, quality_score: float = None
    ) -> bool:
        """
        Update a specific section of the knowledge base without affecting other sections.

        This enables granular updates similar to the Pokemon agent's sectioned approach,
        while maintaining ZorkGPT's quality assessment principles.

        Args:
            section_id: The section to update (e.g., "items", "locations", "dangers")
            content: The new content for this section
            quality_score: Optional quality score for immediate updates

        Returns:
            True if the section was updated, False otherwise
        """
        if not os.path.exists(self.output_file):
            # If no knowledge base exists, create with this section
            self._create_sectioned_knowledge_base(section_id, content)
            return True

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                existing_content = f.read()

            # Parse existing sections
            sections = self._parse_knowledge_sections(existing_content)

            # Update the specific section
            sections[section_id] = content

            # Reassemble knowledge base
            updated_knowledge = self._reassemble_knowledge_sections(sections)

            # Preserve map section if it exists
            if "## CURRENT WORLD MAP" in existing_content:
                updated_knowledge = self._preserve_map_section(
                    existing_content, updated_knowledge
                )

            # Write updated knowledge base
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(updated_knowledge)

            if self.logger:
                self.logger.info(
                    f"Updated knowledge section: {section_id}",
                    extra={"event_type": "knowledge_update"},
                )
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Failed to update knowledge section {section_id}: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return False

    def _parse_knowledge_sections(self, content: str) -> Dict[str, str]:
        """Parse existing knowledge base into sections."""
        sections = {}

        # Define section patterns
        section_patterns = {
            "strategies": r"## 1\. \*\*Key Successful Strategies\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "mistakes": r"## 2\. \*\*Critical Mistakes\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "navigation": r"## 3\. \*\*Navigation Insights\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "items": r"## 4\. \*\*Item Management\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "combat": r"## 5\. \*\*Combat/Danger Handling\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "death_prevention": r"## 6\. \*\*Death Prevention\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "learning": r"## 7\. \*\*Learning Opportunities\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
        }

        for section_id, pattern in section_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[section_id] = match.group(1).strip()

        return sections

    def _reassemble_knowledge_sections(self, sections: Dict[str, str]) -> str:
        """Reassemble sections into a complete knowledge base."""
        header = """# **Zork Game World Knowledge Base (Merged and Enhanced)**

This knowledge base contains discovered information about the Zork game world, including specific items, puzzles, dangers, and strategic insights learned through gameplay. All advice uses algorithmic decision-making patterns, precise command syntax, and computational approaches for optimal performance.

---

"""

        section_templates = {
            "strategies": "## 1. **Key Successful Strategies**\n\n{content}\n\n---\n\n",
            "mistakes": "## 2. **Critical Mistakes**\n\n{content}\n\n---\n\n",
            "navigation": "## 3. **Navigation Insights**\n\n{content}\n\n---\n\n",
            "items": "## 4. **Item Management**\n\n{content}\n\n---\n\n",
            "combat": "## 5. **Combat/Danger Handling**\n\n{content}\n\n---\n\n",
            "death_prevention": "## 6. **Death Prevention**\n\n{content}\n\n---\n\n",
            "learning": "## 7. **Learning Opportunities**\n\n{content}\n\n---\n\n",
        }

        assembled = header
        for section_id, template in section_templates.items():
            if section_id in sections:
                assembled += template.format(content=sections[section_id])

        return assembled

    def _create_sectioned_knowledge_base(
        self, initial_section: str, content: str
    ) -> None:
        """Create a new knowledge base with sections, starting with the given section."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"# Zork Strategy Guide\n\n## {initial_section}\n{content}\n")

    def synthesize_inter_episode_wisdom(self, episode_data: Dict) -> bool:
        """
        Synthesize persistent wisdom from episode completion that should carry forward
        to future episodes. Focuses on deaths, major discoveries, and cross-episode patterns.

        Args:
            episode_data: Dictionary containing episode summary information

        Returns:
            True if synthesis was performed and wisdom was updated, False if skipped
        """
        from config import get_config

        config = get_config()

        persistent_wisdom_file = config.orchestrator.persistent_wisdom_file

        if self.logger:
            self.logger.info(
                f"Synthesizing inter-episode wisdom from episode {episode_data['episode_id']}",
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
            if self.logger:
                self.logger.info(
                    "Episode not significant enough for wisdom synthesis",
                    extra={
                        "event_type": "knowledge_update",
                        "details": f"Death: {episode_ended_in_death}, Score: {final_score}, Turns: {turn_count}, Avg Critic: {avg_critic_score:.2f}",
                    },
                )
            return False

        # Extract turn-by-turn data for death analysis and major discoveries
        turn_data = self._extract_turn_window_data(episode_id, 1, turn_count)
        if not turn_data:
            if self.logger:
                self.logger.warning(
                    "Could not extract turn data for wisdom synthesis",
                    extra={"event_type": "knowledge_update"},
                )
            return False

        # Load existing persistent wisdom
        existing_wisdom = ""
        try:
            with open(persistent_wisdom_file, "r", encoding="utf-8") as f:
                existing_wisdom = f.read()
        except FileNotFoundError:
            # No existing wisdom file - this is fine for first episode
            existing_wisdom = ""
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Could not load existing wisdom: {e}",
                    extra={"event_type": "knowledge_update"},
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
        prompt = f"""Analyze this completed Zork episode and update the persistent wisdom base with key learnings that should carry forward to future episodes.

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
{turn_data["actions_and_responses"][:10] if turn_data.get("actions_and_responses") else "No action data available"}

{death_analysis}

**EXISTING PERSISTENT WISDOM:**
{existing_wisdom if existing_wisdom else "No previous wisdom recorded."}

**SYNTHESIS TASK:**

Extract and synthesize the most critical learnings from this episode that should persist across future episodes. Focus on:

1. **Death Pattern Analysis**: If deaths occurred, what specific patterns, locations, or actions consistently lead to death? What environmental cues signal danger?

2. **Critical Environmental Knowledge**: What persistent facts about the game world were discovered? (Dangerous locations, item behaviors, puzzle mechanics)

3. **Strategic Patterns**: What meta-strategies or approaches proved consistently effective or ineffective across different situations?

4. **Discovery Insights**: What major discoveries about game mechanics, hidden areas, or puzzle solutions should be remembered?

5. **Cross-Episode Learning**: How does this episode's experience relate to patterns from previous episodes?

**REQUIREMENTS:**
- Focus on persistent, reusable knowledge rather than episode-specific details
- Emphasize death avoidance and danger recognition patterns
- Maintain existing wisdom while adding new insights
- Remove outdated or contradicted information
- Keep wisdom concise but actionable for an AI agent

**OUTPUT FORMAT:**
Provide the updated persistent wisdom as a well-organized markdown document. If no significant new wisdom emerged, return "NO_SIGNIFICANT_WISDOM" instead."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting persistent strategic wisdom from interactive fiction gameplay that can help AI agents improve across multiple game sessions. Focus on actionable patterns, danger recognition, and cross-episode learning.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 2000,
            )

            wisdom_response = response.content.strip()

            if wisdom_response == "NO_SIGNIFICANT_WISDOM":
                if self.logger:
                    self.logger.info(
                        "No significant wisdom to synthesize from this episode",
                        extra={"event_type": "knowledge_update"},
                    )
                return False

            # Save the updated persistent wisdom
            try:
                with open(persistent_wisdom_file, "w", encoding="utf-8") as f:
                    f.write(wisdom_response)

                if self.logger:
                    self.logger.info(
                        f"Persistent wisdom updated and saved to {persistent_wisdom_file}",
                        extra={
                            "event_type": "knowledge_update",
                            "details": f"Synthesized from episode with {turn_count} turns, score {final_score}",
                        },
                    )

                return True

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Failed to save persistent wisdom: {e}",
                        extra={"event_type": "knowledge_update"},
                    )
                return False

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Inter-episode wisdom synthesis failed: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return False
