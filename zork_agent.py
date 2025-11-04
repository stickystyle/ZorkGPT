"""
ZorkAgent module for generating actions and managing game memory.
"""

import re
from typing import Optional, List, Tuple, Dict
from collections import Counter
import os
from pathlib import Path
from map_graph import MapGraph
from hybrid_zork_extractor import ExtractorResponse
from llm_client import LLMClientWrapper
from session.game_configuration import GameConfiguration

try:
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    # Graceful fallback - no-op decorator
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGFUSE_AVAILABLE = False


class ZorkAgent:
    """
    Handles agent action generation and memory management for Zork gameplay.
    """

    def __init__(
        self,
        config: GameConfiguration,
        model: str = None,
        client: Optional[LLMClientWrapper] = None,
        max_tokens: Optional[int] = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        logger=None,
        episode_id: str = "unknown",
    ):
        """
        Initialize the ZorkAgent.

        Args:
            config: GameConfiguration instance
            model: Model name for agent
            client: OpenAI client instance (if None, creates new one)
            max_tokens: Maximum tokens for agent responses
            temperature: Temperature for agent model
            top_p: Top-p nucleus sampling
            top_k: Top-k sampling
            min_p: Minimum probability sampling
            logger: Logger instance for tracking
            episode_id: Current episode ID for logging
        """
        self.config = config

        self.model = model or self.config.agent_model
        self.max_tokens = max_tokens or self.config.agent_sampling.get("max_tokens")
        self.temperature = (
            temperature
            if temperature is not None
            else self.config.agent_sampling.get("temperature")
        )
        self.top_p = top_p if top_p is not None else self.config.agent_sampling.get("top_p")
        self.top_k = top_k if top_k is not None else self.config.agent_sampling.get("top_k")
        self.min_p = min_p if min_p is not None else self.config.agent_sampling.get("min_p")
        self.logger = logger
        self.episode_id = episode_id

        # Create sampling params object for LLM calls
        self.sampling_params = self.config.agent_sampling

        # Initialize LLM client if not provided
        if client is None:
            self.client = LLMClientWrapper(
                config=self.config,
                base_url=self.config.get_llm_base_url_for_model("agent"),
                api_key=self.config.get_effective_api_key(),
            )
        else:
            self.client = client

        # Load system prompt
        self._load_system_prompt()

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
                self.logger.error(
                    f"Failed to load agent prompt file: {e}",
                    extra={"episode_id": self.episode_id},
                )
            raise

    def _enhance_prompt_with_knowledge(self, base_prompt: str) -> str:
        """Enhance the agent prompt with accumulated knowledge."""
        knowledge_file = Path(self.config.zork_game_workdir) / self.config.knowledge_file

        if not os.path.exists(knowledge_file):
            return base_prompt

        try:
            with open(knowledge_file, "r", encoding="utf-8") as f:
                knowledge_content = f.read()

            # Strip map section from knowledge base (map is now passed dynamically in context)
            import re
            pattern = r"## CURRENT WORLD MAP\s*\n\s*```mermaid\s*\n.*?\n```"
            knowledge_content = re.sub(pattern, "", knowledge_content, flags=re.DOTALL)
            knowledge_content = knowledge_content.strip()

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

    @observe(name="agent-generate-action")
    def get_action_with_reasoning(
        self,
        game_state_text: str,
        relevant_memories: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Gets an action from the Agent LM with reasoning preserved.

        Args:
            game_state_text: Current game state text
            relevant_memories: Formatted string of relevant memories (includes reasoning history)

        Returns:
            Dict with 'action' (cleaned) and 'reasoning' (raw thinking/reasoning)
        """
        if "o1" in self.model:
            # Use user prompt for o1 models with caching
            messages = [
                {
                    "role": "user",
                    "content": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Combine game state with relevant memories if available
        user_content = game_state_text
        if relevant_memories:
            if user_content:
                user_content = f"{user_content}\n\n{relevant_memories}"
            else:
                user_content = relevant_memories

        messages.append({"role": "user", "content": user_content})

        try:
            client_args = dict(
                model=self.model,
                messages=messages,
                stop=None,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_tokens=self.max_tokens,
                name="Agent",
            )

            response = self.client.chat.completions.create(**client_args)
            raw_response = response.content.strip()

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

            # Fallback: if no reasoning found in tags, try to extract reasoning from the response
            if not reasoning_parts:
                # Look for reasoning patterns that might not be in tags
                lines = raw_response.split("\n")
                potential_reasoning = []

                for line in lines:
                    line = line.strip()
                    # Skip if it looks like a command
                    if len(line.split()) <= 3 and any(
                        word.lower() in line.lower()
                        for word in [
                            "north",
                            "south",
                            "east",
                            "west",
                            "up",
                            "down",
                            "look",
                            "examine",
                            "take",
                            "open",
                            "close",
                            "enter",
                            "exit",
                            "climb",
                            "go",
                        ]
                    ):
                        continue
                    # Skip empty lines
                    if not line:
                        continue
                    # If it's a longer explanatory line, consider it reasoning
                    if len(line) > 20 or any(
                        reasoning_word in line.lower()
                        for reasoning_word in [
                            "should",
                            "need",
                            "want",
                            "will",
                            "can",
                            "might",
                            "could",
                            "seems",
                            "appears",
                            "because",
                            "since",
                            "to explore",
                            "to find",
                        ]
                    ):
                        potential_reasoning.append(line)

                if potential_reasoning:
                    reasoning_parts.extend(potential_reasoning)

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

            # Remove any remaining markup tags (like <s>, </s>, etc.)
            action = re.sub(r"<[^>]*>", "", action)

            # Remove backticks and other formatting
            action = re.sub(
                r"`([^`]*)`", r"\1", action
            )  # Remove backticks but keep content
            action = re.sub(
                r"```[^`]*```", "", action, flags=re.DOTALL
            )  # Remove code blocks

            # Basic cleaning: Zork commands are usually lowercase
            action = action.lower().strip()

            # Remove any leading/trailing punctuation that might interfere
            action = action.strip(".,!?;:")

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
                self.logger.error(
                    f"Error getting agent action: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return {
                "action": "look",
                "reasoning": None,
                "raw_response": None,
            }  # Default safe action on error

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
            old_length = (
                len(self.system_prompt) if hasattr(self, "system_prompt") else 0
            )
            self.system_prompt = new_system_prompt
            new_length = len(self.system_prompt)

            if self.logger:
                self.logger.info(
                    f"Knowledge base reloaded successfully (prompt: {old_length} -> {new_length} chars)",
                    extra={
                        "event_type": "knowledge_base_reloaded",
                        "episode_id": self.episode_id,
                        "old_prompt_length": old_length,
                        "new_prompt_length": new_length,
                    },
                )

            return True

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to reload knowledge base: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return False
