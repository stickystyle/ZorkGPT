"""
ZorkAgent module for generating actions and managing game memory.
"""

import re
from typing import Optional, List, Tuple, Dict
from openai import OpenAI
from collections import Counter
import environs
import os
from map_graph import MapGraph
from hybrid_zork_extractor import ExtractorResponse

# Load environment variables
env = environs.Env()
env.read_env()


class ZorkAgent:
    """
    Handles agent action generation and memory management for Zork gameplay.
    """

    def __init__(
        self,
        model: str = None,
        client: Optional[OpenAI] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.5,
        logger=None,
        episode_id: str = "unknown",
    ):
        """
        Initialize the ZorkAgent.

        Args:
            model: Model name for agent
            client: OpenAI client instance (if None, creates new one)
            max_tokens: Maximum tokens for agent responses
            temperature: Temperature for agent model
            logger: Logger instance for tracking
            episode_id: Current episode ID for logging
        """
        self.model = model or env.str("AGENT_MODEL", "qwen3-30b-a3b-mlx")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logger
        self.episode_id = episode_id
        
        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = env.bool("ENABLE_PROMPT_LOGGING", False)

        # Initialize OpenAI client if not provided
        if client is None:
            self.client = OpenAI(
                base_url=env.str("CLIENT_BASE_URL", None),
                api_key=env.str("CLIENT_API_KEY", None),
            )
        else:
            self.client = client

        # Load system prompt
        self._load_system_prompt()

    def _log_prompt_to_file(self, messages: List[Dict], prefix: str = "agent") -> None:
        """Log the full prompt to a temporary file for evaluation."""
        if not self.enable_prompt_logging:
            return
            
        self.prompt_counter += 1
        filename = f"tmp/{prefix}_{self.prompt_counter:03d}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== {prefix.upper()} PROMPT #{self.prompt_counter} ===\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"Max Tokens: {self.max_tokens}\n")
                f.write(f"Episode ID: {self.episode_id}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, message in enumerate(messages):
                    f.write(f"--- MESSAGE {i+1} ({message['role'].upper()}) ---\n")
                    f.write(message['content'])
                    f.write("\n\n")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to log prompt to {filename}: {e}")

    def _load_system_prompt(self) -> None:
        """Load agent system prompt from markdown files and enhance with knowledge."""
        try:
            # Load base agent prompt
            with open("agent.md") as fh:
                base_agent_prompt = fh.read()

            # Try to enhance with knowledge base
            self.system_prompt = self._enhance_prompt_with_knowledge(base_agent_prompt)

        except FileNotFoundError as e:
            if self.logger:
                self.logger.error(f"Failed to load agent prompt file: {e}")
            raise

    def _enhance_prompt_with_knowledge(self, base_prompt: str) -> str:
        """Enhance the agent prompt with accumulated knowledge."""
        knowledge_file = "knowledgebase.md"

        if not os.path.exists(knowledge_file):
            return base_prompt

        try:
            with open(knowledge_file, "r", encoding="utf-8") as f:
                knowledge_content = f.read()

            # Insert strategic guide before the "Output Format" section
            knowledge_section = f"""

**STRATEGIC GUIDE FROM PREVIOUS EPISODES:**

The following strategic guide has been compiled from analyzing previous episodes. Use this guide to improve your performance, prioritize important items, navigate efficiently, and avoid known dangers:

{knowledge_content}

**END OF STRATEGIC GUIDE**

"""

            if "**Output Format" in base_prompt:
                insertion_point = base_prompt.find("**Output Format")
                enhanced_prompt = (
                    base_prompt[:insertion_point]
                    + knowledge_section
                    + base_prompt[insertion_point:]
                )
            else:
                enhanced_prompt = base_prompt + knowledge_section

            # Log knowledge integration
            if self.logger:
                self.logger.info(
                    f"Enhanced prompt with knowledge base ({len(knowledge_content):,} characters)"
                )

            return enhanced_prompt

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Could not load knowledge from {knowledge_file}: {e}"
                )
            return base_prompt

    def get_action(
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
        if "o1" in self.model:
            # Use user prompt for o1 models
            messages = [{"role": "user", "content": self.system_prompt}]
        else:
            messages = [{"role": "system", "content": self.system_prompt}]

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

            if "o1" in self.model:
                # o1 models use user role for all messages
                messages.append({"role": "user", "content": memory_context})
            else:
                messages.append({"role": "system", "content": memory_context})

        # Combine game state with relevant memories if available
        user_content = game_state_text
        if relevant_memories:
            user_content = f"{user_content}\n\n{relevant_memories}"

        messages.append({"role": "user", "content": user_content})

        try:
            client_args = dict(
                model=self.model,
                messages=messages,
                stop=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers={
                    "X-Title": "ZorkGPT",
                },
            )

            response = self.client.chat.completions.create(**client_args)
            action = response.choices[0].message.content.strip()

            # Clean up the action: remove any thinking
            action = re.sub(r"<think>.*?</think>\s*", "", action, flags=re.DOTALL)
            action = re.sub(r"<thinking>.*?</thinking>\s*", "", action, flags=re.DOTALL)
            action = re.sub(
                r"<reflection>.*?</reflection>\s*", "", action, flags=re.DOTALL
            )
            # Basic cleaning: Zork commands are usually lowercase
            action = action.lower()

            # Validate action is not empty
            if not action or action.isspace():
                if self.logger:
                    self.logger.warning(
                        "Agent returned empty action, using 'look' as fallback"
                    )
                action = "look"
            return action
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting agent action: {e}")
            return "look"  # Default safe action on error

    def get_action_with_reasoning(
        self,
        game_state_text: str,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
        action_counts: Optional[Counter] = None,
        relevant_memories: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Gets an action from the Agent LM with reasoning preserved.

        Args:
            game_state_text: Current game state text
            previous_actions_and_responses: List of (action, response) tuples for history
            action_counts: Counter of how many times each action has been tried
            relevant_memories: Formatted string of relevant memories

        Returns:
            Dict with 'action' (cleaned) and 'reasoning' (raw thinking/reasoning)
        """
        if "o1" in self.model:
            # Use user prompt for o1 models
            messages = [{"role": "user", "content": self.system_prompt}]
        else:
            messages = [{"role": "system", "content": self.system_prompt}]

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

            if "o1" in self.model:
                # o1 models use user role for all messages
                messages.append({"role": "user", "content": memory_context})
            else:
                messages.append({"role": "system", "content": memory_context})

        # Combine game state with relevant memories if available
        user_content = game_state_text
        if relevant_memories:
            user_content = f"{user_content}\n\n{relevant_memories}"

        messages.append({"role": "user", "content": user_content})

        # Log the full prompt for evaluation
        self._log_prompt_to_file(messages, "agent")

        try:
            client_args = dict(
                model=self.model,
                messages=messages,
                stop=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers={
                    "X-Title": "ZorkGPT",
                },
            )

            response = self.client.chat.completions.create(**client_args)
            raw_response = response.choices[0].message.content.strip()

            # Extract reasoning from thinking tags
            reasoning_parts = []

            # Extract <think> tags
            think_matches = re.findall(
                r"<think>(.*?)</think>", raw_response, flags=re.DOTALL
            )
            reasoning_parts.extend(think_matches)

            # Extract <thinking> tags
            thinking_matches = re.findall(
                r"<thinking>(.*?)</thinking>", raw_response, flags=re.DOTALL
            )
            reasoning_parts.extend(thinking_matches)

            # Extract <reflection> tags
            reflection_matches = re.findall(
                r"<reflection>(.*?)</reflection>", raw_response, flags=re.DOTALL
            )
            reasoning_parts.extend(reflection_matches)

            # Combine all reasoning
            reasoning = "\n\n".join(
                part.strip() for part in reasoning_parts if part.strip()
            )

            # Clean up the action: remove any thinking
            action = re.sub(r"<think>.*?</think>\s*", "", raw_response, flags=re.DOTALL)
            action = re.sub(r"<thinking>.*?</thinking>\s*", "", action, flags=re.DOTALL)
            action = re.sub(
                r"<reflection>.*?</reflection>\s*", "", action, flags=re.DOTALL
            )
            # Basic cleaning: Zork commands are usually lowercase
            action = action.lower().strip()

            # Validate action is not empty
            if not action or action.isspace():
                if self.logger:
                    self.logger.warning(
                        "Agent returned empty action, using 'look' as fallback"
                    )
                action = "look"

            return {
                "action": action,
                "reasoning": reasoning if reasoning else None,
                "raw_response": raw_response,
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting agent action: {e}")
            return {
                "action": "look",
                "reasoning": None,
                "raw_response": None,
            }  # Default safe action on error

    def get_relevant_memories_for_prompt(
        self,
        current_location_name_from_current_extraction: str,
        memory_log_history: List[ExtractorResponse],
        current_inventory: List[str],
        game_map: MapGraph,
        previous_room_name_for_map_context: Optional[str] = None,
        action_taken_to_current_room: Optional[str] = None,
        in_combat: bool = False,
        failed_actions_by_location: Optional[dict] = None,
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
            in_combat: Whether currently in combat
            failed_actions_by_location: Dict of failed actions by location

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
            failed_actions_by_location
            and current_location_name_from_current_extraction
            in failed_actions_by_location
        ):
            failed_actions = failed_actions_by_location[
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
            # Always use the most recent memory entry
            # This contains the result from the last action taken
            relevant_history_index = -1
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

    def update_episode_id(self, episode_id: str) -> None:
        """Update the episode ID for logging purposes."""
        self.episode_id = episode_id

    def reload_knowledge_base(self) -> bool:
        """Reload the knowledge base from file and update the system prompt.
        
        Returns:
            True if knowledge base was successfully reloaded, False otherwise
        """
        try:
            # Load base agent prompt
            with open("agent.md") as fh:
                base_agent_prompt = fh.read()

            # Re-enhance with current knowledge base
            new_system_prompt = self._enhance_prompt_with_knowledge(base_agent_prompt)
            
            # Update the system prompt
            old_length = len(self.system_prompt) if hasattr(self, 'system_prompt') else 0
            self.system_prompt = new_system_prompt
            new_length = len(self.system_prompt)
            
            if self.logger:
                self.logger.info(
                    f"Knowledge base reloaded successfully (prompt: {old_length} -> {new_length} chars)",
                    extra={
                        "extras": {
                            "event_type": "knowledge_base_reloaded",
                            "episode_id": self.episode_id,
                            "old_prompt_length": old_length,
                            "new_prompt_length": new_length,
                        }
                    }
                )
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to reload knowledge base: {e}")
            return False
