import json
from openai import OpenAI
from zork_api import ZorkInterface
import time
import re
from pydantic import BaseModel
from collections import Counter
from typing import List, Optional, Tuple
import environs
from datetime import datetime
from map_graph import MapGraph, normalize_direction, is_non_movement_command
from logger import setup_logging, ZorkExperienceTracker


GENERIC_LOCATION_FALLBACKS = {
    "unknown location",
    "unknown area",
    "unclear area",
    "unspecified location",
    "same area",
    "same place",
    "no specific location",
    "not applicable",
    "na",
    "n/a",
    "",  # Empty string also a fallback
    # Add any other generic phrases observed from LLM outputs if necessary
}

env = environs.Env()
env.read_env()


MAX_TOKENS_AGENT = 4096  # Agent should output very short commands however, it needs more tokens for reasoning
MAX_TOKENS_CRITIC = 100  # Critic might need more tokens for justification
MAX_TOKENS_INFO_EXT = 300  # Adjust based on expected JSON size and complexity

TEMPERATURE_AGENT = 0.5  # Some creativity but not too random for commands
TEMPERATURE_CRITIC = 0.2  # Critic should be more deterministic
TEMPERATURE_INFO_EXT = 0.1  # Low temperature for deterministic extraction


class CriticResponse(BaseModel):
    score: float
    justification: str


class ExtractorResponse(BaseModel):
    current_location_name: str
    exits: List[str]
    visible_objects: List[str]
    visible_characters: List[str]
    important_messages: List[str]
    in_combat: bool


with open("agent.md") as fh:
    AGENT_SYSTEM_PROMPT = fh.read()

with open("critic.md") as fh:
    CRITIC_SYSTEM_PROMPT = fh.read()

with open("extractor.md") as fh:
    EXTRACTOR_SYSTEM_PROMPT = fh.read()


class ZorkAgent:
    """
    A class to manage Zork gameplay episodes with integrated logging and experience tracking.

    This class encapsulates all the logic for playing Zork episodes, including:
    - Agent action generation
    - Critic evaluation
    - Information extraction
    - Memory management
    - Logging and experience tracking
    """

    def __init__(
        self,
        agent_model: str = None,
        critic_model: str = None,
        info_ext_model: str = None,
        episode_log_file: str = "zork_episode_log.txt",
        json_log_file: str = "zork_episode_log.jsonl",
        experiences_file: str = "zork_experiences.json",
        max_turns_per_episode: int = 200,
        client_base_url: str = None,
        client_api_key: str = None,
    ):
        """
        Initialize the ZorkAgent.

        Args:
            agent_model: Model name for the agent
            critic_model: Model name for the critic
            info_ext_model: Model name for information extraction
            episode_log_file: Path for human-readable episode logs
            json_log_file: Path for JSON logs
            experiences_file: Path for experience data
            max_turns_per_episode: Maximum turns per episode
            client_base_url: OpenAI client base URL
            client_api_key: OpenAI client API key
        """
        # Configuration from environment or parameters
        self.agent_model = agent_model or env.str("AGENT_MODEL", "qwen3-30b-a3b-mlx")
        self.critic_model = critic_model or env.str("CRITIC_MODEL", "qwen3-30b-a3b-mlx")
        self.info_ext_model = info_ext_model or env.str(
            "INFO_EXT_MODEL", "qwen3-30b-a3b-mlx"
        )

        # File paths
        self.episode_log_file = episode_log_file
        self.json_log_file = json_log_file
        self.experiences_file = experiences_file

        # Game settings
        self.max_turns_per_episode = max_turns_per_episode

        # Model parameters
        self.max_tokens_agent = None
        self.max_tokens_critic = 100
        self.max_tokens_info_ext = 300
        self.temperature_agent = 0.5
        self.temperature_critic = 0.2
        self.temperature_info_ext = 0.1

        # Initialize logger and experience tracker
        self.logger = setup_logging(episode_log_file, json_log_file)
        self.experience_tracker = ZorkExperienceTracker()

        # Load system prompts
        self._load_system_prompts()

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=client_base_url or env.str("CLIENT_BASE_URL", None),
            api_key=client_api_key or env.str("CLIENT_API_KEY", None),
        )

        # Episode state (reset for each episode)
        self.reset_episode_state()

    def _load_system_prompts(self) -> None:
        """Load system prompts from markdown files."""
        try:
            with open("agent.md") as fh:
                self.agent_system_prompt = fh.read()
            with open("critic.md") as fh:
                self.critic_system_prompt = fh.read()
            with open("extractor.md") as fh:
                self.extractor_system_prompt = fh.read()
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load system prompt file: {e}")
            raise

    def reset_episode_state(self) -> None:
        """Reset state variables for a new episode."""
        self.memory_log_history = []
        self.current_inventory = []
        self.action_history = []
        self.action_counts = Counter()
        self.visited_locations = set()
        self.game_map = MapGraph()
        self.prev_room_for_prompt_context = None
        self.action_leading_to_current_room_for_prompt_context = None
        self.current_room_name_for_map = None
        self.turn_count = 0
        self.total_episode_reward = 0
        self.previous_zork_score = 0
        self.episode_id = None
        # Add failed actions tracking by location
        self.failed_actions_by_location = {}  # location_name -> set of failed actions

    def get_agent_action(
        self,
        game_state_text: str,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
        action_counts: Optional[Counter] = None,
        relevant_memories: Optional[str] = None,
    ) -> str:
        """
        Gets an action from the Agent LM.

        Args:
            game_state_text: Current game state text
            previous_actions_and_responses: List of (action, response) tuples for history
            action_counts: Counter of how many times each action has been tried
            relevant_memories: Formatted string of relevant memories

        Returns:
            The agent's chosen action as a string
        """
        messages = [{"role": "system", "content": self.agent_system_prompt}]

        # Add history if provided
        if previous_actions_and_responses:
            memory_context = "Here's what you've done so far:\n"

            # Add the most recent actions and responses (last 5-8 is usually sufficient)
            for i, (action, response) in enumerate(previous_actions_and_responses[-8:]):
                memory_context += f"Command: {action}\nResult: {response.strip()}\n\n"

            # Include information about repetitive actions
            if action_counts:
                repeated_actions = [
                    act for act, count in action_counts.items() if count > 2
                ]
                if repeated_actions:
                    memory_context += "\n**CRITICAL WARNING**: You've tried these actions multiple times with limited success: "
                    memory_context += ", ".join(repeated_actions)
                    memory_context += ". According to your instructions, you must AVOID repeating failed actions and try completely different approaches.\n"

            messages.append({"role": "system", "content": memory_context})

        # Combine game state with relevant memories if available
        user_content = game_state_text
        if relevant_memories:
            user_content = f"{user_content}\n\n{relevant_memories}"

        messages.append({"role": "user", "content": user_content})

        try:
            client_args = dict(
                model=self.agent_model,
                messages=messages,
                stop=None,
                temperature=self.temperature_agent,
                max_tokens=self.max_tokens_agent,
            )

            response = self.client.chat.completions.create(**client_args)
            action = response.choices[0].message.content.strip()
            # Clean up the action: remove any thinking
            action = re.sub(r"<think>.*?</think>\s*", "", action, flags=re.DOTALL)
            # Basic cleaning: Zork commands are usually lowercase
            action = action.lower()

            # Validate action is not empty
            if not action or action.isspace():
                self.logger.warning(
                    "Agent returned empty action, using 'look' as fallback"
                )
                action = "look"

            return action
        except Exception as e:
            self.logger.error(f"Error getting agent action: {e}")
            return "look"  # Default safe action on error

    def get_critic_evaluation(
        self,
        game_state_text: str,
        proposed_action: str,
        action_counts: Optional[Counter] = None,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
    ) -> CriticResponse:
        """
        Gets an evaluation from the Critic LM.

        Args:
            game_state_text: Current game state text
            proposed_action: The action to evaluate
            action_counts: Counter of action frequencies
            previous_actions_and_responses: Recent action history

        Returns:
            CriticResponse with score and justification
        """
        # Prepare context about repetitive actions for the critic
        repetition_context = ""
        if action_counts and action_counts[proposed_action] > 2:
            repetition_context = f"\nThis action '{proposed_action}' has been tried {action_counts[proposed_action]} times already."

        # Add context about the last few actions and responses
        recent_context = ""
        if previous_actions_and_responses and len(previous_actions_and_responses) > 0:
            recent_context = "\nRecent actions and responses:\n"
            for act, resp in previous_actions_and_responses[-3:]:
                recent_context += f"Command: {act}\nResult: {resp.strip()}\n"

        user_prompt = f"""Current Game State:
{game_state_text}

Proposed Agent Action:
{proposed_action}{repetition_context}{recent_context}

Evaluate this action based on your criteria. Respond in JSON format.
"""
        messages = [
            {"role": "system", "content": self.critic_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.critic_model,
                messages=messages,
                temperature=self.temperature_critic,
                max_tokens=self.max_tokens_critic,
                response_format={"type": "json_object"},
            )

            response_content = response.choices[0].message.content
            try:
                parsed_data = json.loads(response_content)
                return CriticResponse(**parsed_data)
            except Exception as e:
                self.logger.error(f"Error parsing critic response: {e}")
                self.logger.error(f"Response content: {response_content}")
                return CriticResponse(
                    score=0.0, justification="Critic evaluation error (parsing)."
                )
        except Exception as e:
            self.logger.error(f"Error getting critic evaluation: {e}")
            return CriticResponse(
                score=0.0, justification="Critic evaluation error (API)."
            )

    def get_extracted_info(
        self, game_text_from_zork: str
    ) -> Optional[ExtractorResponse]:
        """
        Uses an LLM to extract structured information from Zork's game text.

        Args:
            game_text_from_zork: The raw text output from the Zork game.

        Returns:
            ExtractorResponse containing the extracted information, or None if extraction fails.
        """
        if not game_text_from_zork or not game_text_from_zork.strip():
            return ExtractorResponse(
                current_location_name="Unknown (Empty Input)",
                exits=[],
                visible_objects=[],
                visible_characters=[],
                important_messages=["Received empty game text."],
                in_combat=False,
            )

        user_prompt_content = (
            f"Game Text:\n```\n{game_text_from_zork}\n```\n\nJSON Output:\n```json\n"
        )
        user_prompt_content = r"\no_think " + user_prompt_content

        messages = [
            {"role": "system", "content": self.extractor_system_prompt},
            {"role": "user", "content": user_prompt_content},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.info_ext_model,
                messages=messages,
                temperature=self.temperature_info_ext,
                max_tokens=self.max_tokens_info_ext,
                response_format={"type": "json_object"},
            )

            response_content = response.choices[0].message.content

            try:
                parsed_data = json.loads(response_content)
                return ExtractorResponse(**parsed_data)
            except Exception as e:
                self.logger.error(f"Error parsing extractor response: {e}")
                self.logger.error(f"Response content: {response_content}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting extracted info: {e}")
            return None

    def get_relevant_memories_for_prompt(
        self,
        current_location_name_from_current_extraction: str,
        memory_log_history: List[ExtractorResponse],
        current_inventory: List[str],
        game_map: MapGraph,
        previous_room_name_for_map_context: Optional[str] = None,
        action_taken_to_current_room: Optional[str] = None,
        in_combat: bool = False,
    ) -> str:
        """
        Generate relevant memories and context for the agent prompt.

        Args:
            current_location_name_from_current_extraction: Current room name
            memory_log_history: History of extracted information
            current_inventory: Current inventory items
            game_map: The game map object
            previous_room_name_for_map_context: Previous room name
            action_taken_to_current_room: Action that led to current room

        Returns:
            Formatted string of relevant memories for the agent
        """
        map_context_str = ""
        if game_map:
            map_info = game_map.get_context_for_prompt(
                current_room_name=current_location_name_from_current_extraction,
                previous_room_name=previous_room_name_for_map_context,
                action_taken_to_current=action_taken_to_current_room,
            )
            if map_info:
                map_context_str = map_info

        other_memory_strings = []

        # Add combat status information
        if in_combat:
            other_memory_strings.append(
                "- COMBAT SITUATION: You are currently in combat or facing an immediate threat! Be prepared to defend yourself or flee."
            )

        if current_inventory:
            other_memory_strings.append(
                f"- You are carrying: {', '.join(current_inventory)}."
            )

        # Add failed actions warning for current location
        if (
            hasattr(self, "failed_actions_by_location")
            and current_location_name_from_current_extraction
            in self.failed_actions_by_location
        ):
            failed_actions = self.failed_actions_by_location[
                current_location_name_from_current_extraction
            ]
            if failed_actions:
                other_memory_strings.append(
                    f"- FAILED ACTIONS in {current_location_name_from_current_extraction}: The following actions have already failed here and should NOT be repeated: {', '.join(sorted(failed_actions))}."
                )

        previous_observations_of_current_room = [
            obs
            for obs in reversed(memory_log_history[:-1])  # Exclude current observation
            if obs.current_location_name
            == current_location_name_from_current_extraction
        ]

        if previous_observations_of_current_room:
            last_relevant_obs = previous_observations_of_current_room[0]
            prev_objects = last_relevant_obs.visible_objects
            if prev_objects:
                other_memory_strings.append(
                    f"- Previously noted objects in {current_location_name_from_current_extraction}: {', '.join(prev_objects)}."
                )

        if memory_log_history:
            relevant_history_index = -1
            if (
                len(memory_log_history) > 1
            ):  # If there's more than the current observation
                relevant_history_index = (
                    -2
                )  # Use the one before current (last turn's result)

            last_turn_info = memory_log_history[relevant_history_index]
            important_msgs = last_turn_info.important_messages
            action_results = [
                msg
                for msg in important_msgs
                if not msg.lower().startswith("you are")
                and not msg.lower().startswith(
                    current_location_name_from_current_extraction.lower()
                )
                and len(msg) < 100
            ]
            if action_results:
                other_memory_strings.append(
                    f"- Last action result/event: {' '.join(action_results)}."
                )

        final_output_parts = []
        if map_context_str and map_context_str.strip():
            content_part = map_context_str.replace(
                "--- Map Information ---", ""
            ).strip()
            if content_part:  # Only add if there's more than just the header
                final_output_parts.append(map_context_str)

        if other_memory_strings:  # other_memory_strings is populated by existing logic
            if final_output_parts:
                final_output_parts.append("\n--- Other Relevant Memories ---")
            else:
                final_output_parts.append("--- Relevant Memories ---")
            final_output_parts.extend(other_memory_strings)

        if not final_output_parts:
            return ""
        return "\n".join(final_output_parts) + "\n"

    def extract_location_from_text(self, text: str) -> Optional[str]:
        """
        Extract a simplified location identifier from room descriptions.
        This helps track when the agent has changed locations.
        """
        # If no specific pattern matches, try to extract a generic location description
        # Look for phrases like "You are in/at/on..."
        location_match = re.search(r"You are (in|at|on) (the |a |an )?([\w\s]+)", text)
        if location_match:
            return location_match.group(3).strip().lower().replace(" ", "_")

        return None

    def play_episode(self, zork_interface_instance) -> Tuple[List, int]:
        """
        Play a single episode of Zork.

        Args:
            zork_interface_instance: The Zork game interface

        Returns:
            Tuple of (experiences, final_score)
        """
        # Reset state for new episode
        self.reset_episode_state()

        # Generate episode ID (ISO8601 timestamp with second resolution)
        episode_start_time = datetime.now()
        self.episode_id = episode_start_time.strftime("%Y-%m-%dT%H:%M:%S")

        # Log episode start
        self.logger.info(
            "Starting Zork episode...",
            extra={
                "extras": {
                    "event_type": "episode_start",
                    "episode_id": self.episode_id,
                    "agent_model": self.agent_model,
                    "critic_model": self.critic_model,
                    "info_ext_model": self.info_ext_model,
                    "timestamp": datetime.now().isoformat(),
                }
            },
        )

        current_game_state = zork_interface_instance.start()
        if not current_game_state:
            self.logger.error(
                "Failed to start Zork or get initial state.",
                extra={
                    "extras": {"event_type": "error", "episode_id": self.episode_id}
                },
            )
            return (
                self.experience_tracker.get_experiences(),
                0,
            )  # No score if failed to start

        # Log initial state
        self.logger.info(
            f"INITIAL STATE:\n{current_game_state}\n",
            extra={
                "extras": {
                    "event_type": "initial_state",
                    "episode_id": self.episode_id,
                    "game_state": current_game_state,
                }
            },
        )

        # Extract initial info and store in memory log
        extracted_info = self.get_extracted_info(current_game_state)
        self.memory_log_history.append(extracted_info)

        # Log extracted info
        self.logger.info(
            "Extracted info",
            extra={
                "extras": {
                    "event_type": "extracted_info",
                    "episode_id": self.episode_id,
                    "extracted_info": {
                        "current_location_name": extracted_info.current_location_name,
                        "exits": extracted_info.exits,
                        "visible_objects": extracted_info.visible_objects,
                        "visible_characters": extracted_info.visible_characters,
                        "important_messages": extracted_info.important_messages,
                    },
                }
            },
        )

        game_over = False

        # Initial extraction and map update
        if (
            extracted_info
        ):  # Should always be true due to get_extracted_info's error handling
            self.current_room_name_for_map = extracted_info.current_location_name
            self.game_map.add_room(self.current_room_name_for_map)
            self.game_map.update_room_exits(
                self.current_room_name_for_map, extracted_info.exits
            )
        else:
            self.current_room_name_for_map = "Unknown (Initial Extraction Failed)"
            self.game_map.add_room(self.current_room_name_for_map)

        while (
            not game_over
            and zork_interface_instance.is_running()
            and self.turn_count < self.max_turns_per_episode
        ):
            self.turn_count += 1
            # Log turn start
            self.logger.info(
                f"Turn {self.turn_count}",
                extra={
                    "extras": {
                        "event_type": "turn_start",
                        "episode_id": self.episode_id,
                        "turn": self.turn_count,
                    }
                },
            )

            # Check if we're in combat from the previous turn's extracted info
            in_combat = False
            if self.memory_log_history:
                last_extraction = self.memory_log_history[-1]
                in_combat = getattr(last_extraction, "in_combat", False)

            # Get inventory only if not in combat (to avoid death during inventory checks)
            if not in_combat:
                self.current_inventory, inventory_response = (
                    zork_interface_instance.inventory_with_response()
                )

                # Check if the inventory command caused game over
                if inventory_response:
                    game_over_flag, game_over_reason = (
                        zork_interface_instance.is_game_over(inventory_response)
                    )
                    if game_over_flag:
                        # Log game over from inventory
                        self.logger.info(
                            f"Game over during inventory check: {game_over_reason}",
                            extra={
                                "extras": {
                                    "event_type": "game_over",
                                    "episode_id": self.episode_id,
                                    "reason": f"Inventory check triggered: {game_over_reason}",
                                    "turn": self.turn_count,
                                }
                            },
                        )

                        # Get final score
                        current_zork_score_val, max_zork_score = (
                            zork_interface_instance.score(inventory_response)
                        )
                        self.previous_zork_score = current_zork_score_val

                        # Add death experience
                        reward = -20  # Death penalty
                        self.total_episode_reward += reward

                        experience = self.experience_tracker.add_experience(
                            state=current_game_state,
                            action="inventory",  # The action that caused death
                            reward=reward,
                            next_state=inventory_response,
                            done=True,
                            critic_score=0.0,
                            critic_justification="Death during inventory check",
                            zork_score=self.previous_zork_score,
                        )

                        # Log experience
                        self.logger.debug(
                            "Experience added (death during inventory)",
                            extra={
                                "extras": {
                                    "event_type": "experience",
                                    "episode_id": self.episode_id,
                                    "experience": experience,
                                }
                            },
                        )

                        # Episode ends here
                        break
            else:
                # In combat - skip inventory check and log the decision
                self.logger.info(
                    "Skipping inventory check due to combat situation",
                    extra={
                        "extras": {
                            "event_type": "inventory_skip",
                            "episode_id": self.episode_id,
                            "reason": "In combat - avoiding dangerous inventory check",
                            "turn": self.turn_count,
                        }
                    },
                )

            # Get relevant memories to include in the agent prompt
            relevant_memories = self.get_relevant_memories_for_prompt(
                self.current_room_name_for_map,  # Current location name
                self.memory_log_history,
                self.current_inventory,
                self.game_map,  # Pass the map object
                self.prev_room_for_prompt_context,  # Prev room name for context
                self.action_leading_to_current_room_for_prompt_context,  # Action that led here
                in_combat,  # Combat status
            )

            # Log memory context
            self.logger.info(
                f"Memory context for agent:\n{relevant_memories}",
                extra={
                    "extras": {
                        "event_type": "memory_context",
                        "episode_id": self.episode_id,
                        "memory_context": relevant_memories,
                    }
                },
            )

            # 1. Agent proposes an action with memory context and relevant memories
            agent_action = self.get_agent_action(
                current_game_state,  # This is next_game_state from previous turn, or initial state
                self.action_history,
                self.action_counts,
                relevant_memories,
            )

            # Log agent action
            self.logger.info(
                f"RAW AGENT PROPOSAL: {agent_action}",
                extra={
                    "extras": {
                        "event_type": "agent_action",
                        "episode_id": self.episode_id,
                        "agent_action": agent_action,
                    }
                },
            )

            # Update action count for repetition tracking
            self.action_counts[agent_action] += 1

            # Critic doesn't need to evaluate 'score' or 'inventory' as these are meta-actions
            if agent_action.lower() in [
                "score",
                "inventory",
                "quit",
                "save",
                "restore",
            ]:  # Handle meta commands
                critic_score_val = 0.05  # Small positive for useful meta commands
                critic_justification = "Meta-command execution."
            else:
                # 2. Critic evaluates the action with repetition context
                critic_evaluation = self.get_critic_evaluation(
                    current_game_state,
                    agent_action,
                    self.action_counts,
                    self.action_history,
                )
                critic_score_val = critic_evaluation.score
                critic_justification = critic_evaluation.justification

                # Log critic evaluation
                self.logger.info(
                    "Critic evaluation",
                    extra={
                        "extras": {
                            "event_type": "critic_evaluation",
                            "episode_id": self.episode_id,
                            "critic_score": critic_score_val,
                            "critic_justification": critic_justification,
                        }
                    },
                )

            # 3. Send the chosen action to Zork
            # Store current state for map connection before action is taken
            room_before_action = self.current_room_name_for_map
            action_taken = agent_action

            try:
                next_game_state = zork_interface_instance.send_command(
                    action_taken
                )  # Use action_taken

                # Log Zork response
                self.logger.info(
                    f"ZORK RESPONSE for '{action_taken}':\n{next_game_state}\n",
                    extra={
                        "extras": {
                            "event_type": "zork_response",
                            "episode_id": self.episode_id,
                            "action": action_taken,
                            "zork_response": next_game_state,
                        }
                    },
                )

                # Check if the game has ended based on the response
                game_over_flag, game_over_reason = zork_interface_instance.is_game_over(
                    next_game_state
                )
                if game_over_flag:
                    # Log game over
                    self.logger.info(
                        f"{game_over_reason}",
                        extra={
                            "extras": {
                                "event_type": "game_over",
                                "episode_id": self.episode_id,
                                "reason": game_over_reason,
                            }
                        },
                    )

                    game_over = True
                    current_zork_score_val, max_zork_score = (
                        zork_interface_instance.score(next_game_state)
                    )
                    self.previous_zork_score = current_zork_score_val
                    self.action_history.append(
                        (action_taken, next_game_state)
                    )  # Use action_taken
                    if game_over:
                        if "died" in game_over_reason.lower():
                            reward = -20
                        elif "victory" in game_over_reason.lower():
                            reward = 50
                        else:
                            reward = 0
                        self.total_episode_reward += reward

                        # Add experience and log it
                        experience = self.experience_tracker.add_experience(
                            state=current_game_state,
                            action=action_taken,
                            reward=reward,
                            next_state=next_game_state,
                            done=game_over,
                            critic_score=critic_score_val,
                            critic_justification=critic_justification,
                            zork_score=self.previous_zork_score,
                        )

                        # Log experience
                        self.logger.debug(
                            "Experience added",
                            extra={
                                "extras": {
                                    "event_type": "experience",
                                    "episode_id": self.episode_id,
                                    "experience": experience,
                                }
                            },
                        )
                        continue

                self.action_history.append(
                    (action_taken, next_game_state)
                )  # Use action_taken

                # Store the name of the room we were in *before* this action and new extraction.
                # current_room_name_for_map holds this value from the previous iteration or initial setup.
                room_before_action = self.current_room_name_for_map

                # 1. Get location using legacy regex-based function
                legacy_location_name = self.extract_location_from_text(next_game_state)

                # 2. Get location and other info using LLM-based extractor
                llm_extracted_info = self.get_extracted_info(
                    next_game_state
                )  # Renamed to avoid confusion

                final_current_room_name = ""
                source_of_location = ""

                if legacy_location_name:  # Regex succeeded
                    final_current_room_name = legacy_location_name
                    source_of_location = "Regex"
                elif llm_extracted_info:  # Regex failed, try LLM
                    # Use original casing for map storage, but lowercase for comparison
                    llm_room_name_original_case = (
                        llm_extracted_info.current_location_name.strip()
                    )
                    llm_room_name_lower = llm_room_name_original_case.lower()

                    if (
                        not llm_room_name_original_case
                        or llm_room_name_lower in GENERIC_LOCATION_FALLBACKS
                    ):
                        # LLM gave a generic name or empty string.
                        # Persist room_before_action if it's valid (i.e., not None or empty).
                        if (
                            room_before_action and room_before_action.strip()
                        ):  # Ensure room_before_action is substantial
                            final_current_room_name = room_before_action
                            source_of_location = f"Persisted ('{room_before_action}')"
                        else:
                            # This case occurs if regex failed AND LLM is generic AND it's the start of the game (room_before_action is None/empty)
                            # Fallback to a default name or the (generic) LLM name if it's not empty.
                            final_current_room_name = (
                                llm_room_name_original_case
                                if llm_room_name_original_case
                                else "Default Start Area"
                            )
                            source_of_location = (
                                "LLM (Initial Generic)"
                                if llm_room_name_original_case
                                else "Initial Default"
                            )
                    else:
                        # LLM gave a new, non-generic name. Use it.
                        final_current_room_name = (
                            llm_room_name_original_case  # Use original casing for map
                        )
                        source_of_location = "LLM (New)"
                else:  # Both regex and LLM extraction failed (e.g., llm_extracted_info is None)
                    if room_before_action and room_before_action.strip():
                        final_current_room_name = room_before_action
                        source_of_location = f"Persisted (Extraction Fail Fallback to '{room_before_action}')"
                    else:
                        # Absolute fallback, e.g. very first turn and all extractions failed
                        final_current_room_name = "Central Unknown Hub"
                        source_of_location = "Critical Fallback"

                # Ensure final_current_room_name is never an empty string at this point.
                if not final_current_room_name.strip():
                    # This is an ultimate safety net. If all logic above somehow results in an empty name.
                    final_current_room_name = "SafetyNet Unknown Place"
                    source_of_location += (
                        " (SafetyNet)" if source_of_location else "SafetyNet Default"
                    )

                # Use final_current_room_name for all map logic and state updates.
                # Update map with the new room and its exits
                self.game_map.add_room(final_current_room_name)
                # Exits should still come from the LLM extractor, as it's designed for that.
                # (and other fields like objects, characters, messages)
                if llm_extracted_info:  # Check if llm_extracted_info is not None before accessing its attributes
                    self.game_map.update_room_exits(
                        final_current_room_name, llm_extracted_info.exits
                    )

                # Check if room name changed but it wasn't actual movement
                if (
                    final_current_room_name != room_before_action
                    and room_before_action
                    and is_non_movement_command(action_taken)
                ):
                    # Log room name change without movement for debugging
                    self.logger.debug(
                        "Room name changed without movement",
                        extra={
                            "extras": {
                                "event_type": "room_name_change_no_movement",
                                "episode_id": self.episode_id,
                                "from_room": room_before_action,
                                "to_room": final_current_room_name,
                                "action": action_taken,
                                "reason": "Non-movement command detected",
                            }
                        },
                    )

                # If location changed AND it was actually a movement command, add connection
                if (
                    final_current_room_name != room_before_action
                    and room_before_action
                    and not is_non_movement_command(action_taken)
                ):
                    normalized_exit_taken = normalize_direction(action_taken)

                    effective_exit_for_connection = ""
                    effective_exit_for_prompt = ""

                    if normalized_exit_taken:
                        effective_exit_for_connection = normalized_exit_taken
                        effective_exit_for_prompt = normalized_exit_taken
                    else:
                        effective_exit_for_connection = (
                            action_taken.lower()
                        )  # Use lowercase raw action
                        effective_exit_for_prompt = action_taken.lower()

                    if effective_exit_for_connection:  # Ensure it's not empty
                        self.game_map.add_connection(
                            room_before_action,
                            effective_exit_for_connection,
                            final_current_room_name,
                        )
                        self.prev_room_for_prompt_context = room_before_action
                        self.action_leading_to_current_room_for_prompt_context = (
                            effective_exit_for_prompt
                        )

                # Update current_room_name_for_map for the next iteration.
                self.current_room_name_for_map = final_current_room_name

                # Prepare the ExtractorResponse object for memory_log_history
                # This ensures the history reflects the chosen final_current_room_name.
                if llm_extracted_info:  # Check if llm_extracted_info is not None
                    info_for_log = llm_extracted_info.model_copy(
                        update={"current_location_name": final_current_room_name}
                    )
                    self.memory_log_history.append(info_for_log)

                    # Log the extracted info with the source of location
                    original_llm_loc_name = (
                        llm_extracted_info.current_location_name
                        if llm_extracted_info.current_location_name
                        != final_current_room_name
                        else ""
                    )

                    # Log extracted info
                    self.logger.info(
                        "Extracted info",
                        extra={
                            "extras": {
                                "event_type": "extracted_info",
                                "episode_id": self.episode_id,
                                "extracted_info": {
                                    "current_location_name": info_for_log.current_location_name,
                                    "exits": info_for_log.exits,
                                    "visible_objects": info_for_log.visible_objects,
                                    "visible_characters": info_for_log.visible_characters,
                                    "important_messages": info_for_log.important_messages,
                                },
                                "source_of_location": source_of_location,
                                "original_location": original_llm_loc_name,
                            }
                        },
                    )
                else:
                    # This case should be rare due to get_extracted_info's error handling
                    # Log a minimal extraction info object
                    minimal_info = ExtractorResponse(
                        current_location_name=final_current_room_name,
                        exits=[],
                        visible_objects=[],
                        visible_characters=[],
                        important_messages=["Extraction failed"],
                        in_combat=False,
                    )
                    self.memory_log_history.append(minimal_info)

                    # Log minimal extracted info
                    self.logger.info(
                        "Extracted info",
                        extra={
                            "extras": {
                                "event_type": "extracted_info",
                                "episode_id": self.episode_id,
                                "extracted_info": {
                                    "current_location_name": minimal_info.current_location_name,
                                    "exits": minimal_info.exits,
                                    "visible_objects": minimal_info.visible_objects,
                                    "visible_characters": minimal_info.visible_characters,
                                    "important_messages": minimal_info.important_messages,
                                },
                                "source_of_location": source_of_location,
                            }
                        },
                    )

                if not game_over and zork_interface_instance.is_running():
                    current_zork_score_val, max_zork_score = (
                        zork_interface_instance.score()
                    )
                else:
                    current_zork_score_val, max_zork_score = (
                        zork_interface_instance.parse_zork_score(next_game_state)
                    )

            except RuntimeError as e:
                self.logger.error(
                    f"Zork process error: {e}",
                    extra={
                        "extras": {"event_type": "error", "episode_id": self.episode_id}
                    },
                )
                game_over = True
                next_game_state = "Game ended unexpectedly"
                continue

            # 4. Determine Reward & Game Over
            # Base reward from critic
            reward = critic_score_val

            # Apply repetition penalties
            if (
                self.action_counts[agent_action] > 2
                and "already" in next_game_state.lower()
            ):
                repetition_penalty = min(
                    0.2 * (self.action_counts[agent_action] - 2), 0.6
                )
                reward -= repetition_penalty

            # Check for location changes and reward exploration
            if (
                self.current_room_name_for_map
                and self.current_room_name_for_map != room_before_action
            ):  # room_before_action holds previous room name
                if (
                    self.current_room_name_for_map not in self.visited_locations
                ):  # visited_locations still useful
                    exploration_bonus = 0.5
                    reward += exploration_bonus
                    self.visited_locations.add(self.current_room_name_for_map)

            # Check for Zork's internal score changes
            score_change = current_zork_score_val - self.previous_zork_score
            if score_change > 0:
                reward += (
                    score_change * 5
                )  # Significant reward for actual score increase
            self.previous_zork_score = current_zork_score_val

            # Penalty for "I don't understand that" or similar parser failures
            parser_failure_phrases = [
                "i don't understand that.",
                "i don't know the word",
                "you can't see any such thing.",
                "you can't do that.",
                "what?",
                "huh?",
            ]

            # TODO: In the final version, we should not have `blocking_failure_phrases` - the LLM should be able to handle this.
            # But for now, we need to track failed actions to prevent repetition and training.
            # Track failed actions by location
            action_failed = False
            blocking_failure_phrases = [
                "there is a wall there",
                "it is too narrow",
                "you can't move the bolt",
                "i can't see one here",
                "pushing the bubble doesn't appear worthwhile",
                "playing in this way with a bubble doesn't appear worthwhile",
                "you certainly can't turn it with",
                "the bolt won't turn with your best effort",
            ]

            if any(
                phrase in next_game_state.lower() for phrase in parser_failure_phrases
            ):
                reward -= 0.2  # Small penalty
                action_failed = True
            elif any(
                phrase in next_game_state.lower() for phrase in blocking_failure_phrases
            ):
                action_failed = True

            # Track failed actions to prevent repetition
            if action_failed and hasattr(self, "failed_actions_by_location"):
                current_location = self.current_room_name_for_map or "unknown_location"
                if current_location not in self.failed_actions_by_location:
                    self.failed_actions_by_location[current_location] = set()
                self.failed_actions_by_location[current_location].add(
                    agent_action.lower()
                )

            self.total_episode_reward += reward

            # Log reward
            self.logger.info(
                "Reward",
                extra={
                    "extras": {
                        "event_type": "reward",
                        "episode_id": self.episode_id,
                        "reward": reward,
                        "total_reward": self.total_episode_reward,
                    }
                },
            )

            # Store experience for RL and log it
            experience = self.experience_tracker.add_experience(
                state=current_game_state,
                action=agent_action,
                reward=reward,
                next_state=next_game_state,
                done=game_over,
                critic_score=critic_score_val,
                critic_justification=critic_justification,
                zork_score=current_zork_score_val,
            )

            # Log experience
            self.logger.debug(
                "Experience added",
                extra={
                    "extras": {
                        "event_type": "experience",
                        "episode_id": self.episode_id,
                        "experience": experience,
                    }
                },
            )

            current_game_state = next_game_state

        # Debug: Log why the episode ended
        end_reasons = []
        if game_over:
            end_reasons.append("game_over=True")
        if not zork_interface_instance.is_running():
            end_reasons.append("zork_process_not_running")
        if self.turn_count >= self.max_turns_per_episode:
            end_reasons.append(
                f"max_turns_reached({self.turn_count}>={self.max_turns_per_episode})"
            )

        self.logger.info(
            f"Episode ended. Reasons: {', '.join(end_reasons) if end_reasons else 'unknown'}",
            extra={
                "extras": {
                    "event_type": "episode_end_debug",
                    "episode_id": self.episode_id,
                    "game_over": game_over,
                    "zork_running": zork_interface_instance.is_running(),
                    "turn_count": self.turn_count,
                    "max_turns": self.max_turns_per_episode,
                    "reasons": end_reasons,
                }
            },
        )

        # Log episode end
        self.logger.info(
            "Episode finished",
            extra={
                "extras": {
                    "event_type": "episode_end",
                    "episode_id": self.episode_id,
                    "turn_count": self.turn_count,
                    "zork_score": self.previous_zork_score,
                    "max_score": max_zork_score,
                    "total_reward": self.total_episode_reward,
                }
            },
        )

        # Save experiences to a separate file for RL
        self.experience_tracker.save_experiences(self.experiences_file)

        return self.experience_tracker.get_experiences(), self.previous_zork_score


if __name__ == "__main__":
    # Create ZorkAgent instance with default settings
    agent = ZorkAgent(
        agent_model="qwen/qwen3-14b:free",
        critic_model="google/gemini-2.5-flash-preview-05-20",
        info_ext_model="google/gemini-2.5-flash-preview-05-20",
    )

    with ZorkInterface(timeout=1.0) as zork_game:  # Increased timeout for stability
        try:
            episode_experiences, final_score = agent.play_episode(zork_game)
            print(f"\nPlayed one episode. Final Zork score: {final_score}")
            print(agent.game_map.render_ascii())
        except RuntimeError as e:
            print(f"ZorkInterface runtime error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("Ensuring Zork process is closed.")
