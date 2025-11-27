"""
ZorkAgent module for generating actions and managing game memory.
"""

import re
import json
from typing import Optional, List, Tuple, Dict, TYPE_CHECKING, Any
from collections import Counter
import os
from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field
from map_graph import MapGraph
from hybrid_zork_extractor import ExtractorResponse
from llm_client import LLMClientWrapper
from session.game_configuration import GameConfiguration
from shared_utils import create_json_schema

if TYPE_CHECKING:
    from managers.mcp_manager import MCPManager

from managers.mcp_config import MCPError

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


class AgentResponse(BaseModel):
    """Structured response from the ZorkAgent LLM.

    This model defines the expected JSON structure for agent responses,
    including reasoning, action commands, and optional objective tracking.
    """
    thinking: str = Field(
        description="Your reasoning - what you observe, plan, and why"
    )
    action: str = Field(
        description="Single game command to execute"
    )
    new_objective: Optional[str] = Field(
        default=None,
        description="Optional multi-step objective to track. Only set when starting a new multi-turn plan. Should reference specific locations (e.g., 'get lamp from L124')"
    )


@dataclass
class MCPContext:
    """Context prepared by async setup for tool-calling loop."""
    messages: List[Dict[str, Any]]
    tool_schemas: Optional[List[Dict[str, Any]]]
    mcp_connected: bool


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
        mcp_manager: Optional["MCPManager"] = None,
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
            mcp_manager: Optional MCPManager instance for MCP tool calling support
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

        self.mcp_manager = mcp_manager

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

    def _check_model_tool_compatibility(self) -> None:
        """Check if model supports tool calling when MCP is enabled.

        Raises:
            MCPError: If model is incompatible and force_tool_support is False
        """
        if self.mcp_manager is None or self.mcp_manager.is_disabled:
            return

        # Check if force_tool_support bypasses the check
        if self.config.mcp_force_tool_support:
            return

        if not self.client.client._supports_tool_calling(self.model):
            raise MCPError(
                f"Model '{self.model}' does not support tool calling.\n"
                f"Please either:\n"
                f"  1. Set mcp.enabled = false in pyproject.toml, or\n"
                f"  2. Use a compatible model (gpt-4, claude-3, etc.), or\n"
                f"  3. Set mcp.force_tool_support = true to override"
            )

    async def _setup_mcp_context(
        self,
        game_state_text: str,
        relevant_memories: Optional[str] = None,
    ) -> MCPContext:
        """Setup MCP context for action generation with tool calling.

        Handles:
        - MCP session lifecycle (connect, will disconnect in finally of caller)
        - Tool schema retrieval when MCP enabled
        - Model compatibility checking
        - Message history building with cache_control

        Args:
            game_state_text: Current game state text
            relevant_memories: Formatted string of relevant memories

        Returns:
            MCPContext with messages, tool_schemas, and connection status
            for use in tool-calling loop

        Raises:
            MCPError: If model is incompatible with tool calling
        """
        tool_schemas = None
        mcp_connected = False

        # === MCP Session Connect (Req 3.1) ===
        if self.mcp_manager is not None and not self.mcp_manager.is_disabled:
            self._check_model_tool_compatibility()  # Req 12.5
            await self.mcp_manager.connect_session()  # Req 3.3
            mcp_connected = True
            tool_schemas = await self.mcp_manager.get_tool_schemas()  # Req 1.1, 3.4

            if self.logger:
                self.logger.debug(
                    f"MCP session connected, {len(tool_schemas)} tools available"
                )

        # === Build Messages with cache_control (Req 4.7) ===
        if "o1" in self.model or "o3" in self.model:
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

        # Combine game state with memories
        user_content = game_state_text
        if relevant_memories:
            if user_content:
                user_content = f"{user_content}\n\n{relevant_memories}"
            else:
                user_content = relevant_memories

        messages.append(
            {
                "role": "user",
                "content": user_content,
                "cache_control": {"type": "ephemeral"},
            }
        )

        return MCPContext(
            messages=messages,
            tool_schemas=tool_schemas,
            mcp_connected=mcp_connected,
        )

    async def _run_tool_calling_loop(
        self,
        mcp_context: MCPContext,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the tool-calling loop until LLM returns content.

        Implements Requirements:
        - 5.1: Execute tool calls via MCPManager
        - 5.2: Sequential tool execution (not parallel)
        - 5.3: JSON argument parsing before execution
        - 5.4: Append tool results to message history
        - 5.5: Call LLM again after tool results
        - 5.6: Exit loop when LLM returns content
        - 5.10: Log warning and exit on neither content nor tool_calls
        - 4.6: response_format NOT used during loop (OpenRouter compatibility)

        Args:
            mcp_context: MCPContext with messages, tool_schemas, and connection status
            max_iterations: Maximum number of iterations (defaults to config value)

        Returns:
            Dict with either:
            - 'action', 'reasoning', 'new_objective' (on success)
            - '_needs_forced_action': True, 'messages': [...] (on max iterations)
        """
        messages = mcp_context.messages.copy()
        iteration = 0

        if max_iterations is None:
            max_iterations = self.config.mcp_max_tool_iterations

        while iteration < max_iterations:
            iteration += 1

            # Call LLM with tools (NO response_format during loop - Req 4.6)
            response = self.client.client.chat_completions_create(
                model=self.model,
                messages=messages,
                tools=mcp_context.tool_schemas if mcp_context.mcp_connected else None,
                tool_choice="auto" if mcp_context.mcp_connected else None,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_tokens=self.max_tokens,
                # NOTE: response_format=None during loop for OpenRouter compatibility
            )

            # Exit on content (Req 5.6)
            if response.content:
                return self._parse_agent_response(response.content)

            # Execute tool calls (Req 5.1)
            if response.tool_calls:
                # Add assistant message with tool_calls to history
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response.tool_calls
                    ]
                })

                # Execute each tool sequentially (Req 5.2)
                for tool_call in response.tool_calls:
                    # Validate argument size before parsing
                    if len(tool_call.function.arguments) > 100_000:  # 100KB limit
                        if self.logger:
                            self.logger.error(
                                f"Tool arguments exceed size limit: {len(tool_call.function.arguments)} bytes"
                            )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": "Tool arguments exceed size limit"})
                        })
                        continue

                    # Parse JSON arguments (Req 5.3)
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        if self.logger:
                            self.logger.error(
                                f"Failed to parse tool arguments for {tool_call.function.name}: {e}"
                            )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": f"Invalid JSON arguments: {str(e)}"})
                        })
                        continue

                    # Execute via MCPManager (Req 5.1)
                    result = await self.mcp_manager.call_tool(
                        tool_name=tool_call.function.name,
                        arguments=arguments
                    )

                    # Append tool result to history (Req 5.4)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result.to_dict())
                    })

                # Log iteration info
                if self.logger:
                    self.logger.debug(
                        f"Tool-calling iteration {iteration}: executed {len(response.tool_calls)} tools"
                    )

                # Continue loop for next LLM call (Req 5.5)
                continue

            # Neither content nor tool_calls (Req 5.10)
            if self.logger:
                self.logger.warning(
                    f"LLM returned neither content nor tool_calls at iteration {iteration}"
                )
            break

        # Max iterations reached - will be handled by Task #11 (forced final action)
        return {"_needs_forced_action": True, "messages": messages}

    async def _generate_action_async(
        self,
        game_state_text: str,
        relevant_memories: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async implementation of action generation with tool calling.

        Orchestrates:
        1. MCP setup (connect session, get tools)
        2. Tool-calling loop (when MCP connected)
        3. Direct LLM call (when MCP not connected)

        Args:
            game_state_text: Current game state text
            relevant_memories: Formatted string of relevant memories

        Returns:
            Dict with 'action', 'reasoning', and optional 'new_objective'
            Or dict with '_needs_forced_action' if max iterations reached
        """
        # Step 1: Setup (current Task #8 implementation)
        mcp_context = await self._setup_mcp_context(game_state_text, relevant_memories)

        try:
            if mcp_context.mcp_connected:
                # Step 2: Run tool-calling loop
                return await self._run_tool_calling_loop(mcp_context)
            else:
                # No MCP - direct LLM call without tools
                response = self.client.client.chat_completions_create(
                    model=self.model,
                    messages=mcp_context.messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=self.min_p,
                    max_tokens=self.max_tokens,
                    response_format=create_json_schema(AgentResponse),
                    # No tools parameter when MCP not connected
                )

                if response.content:
                    return self._parse_agent_response(response.content)
                else:
                    # Fallback for unexpected case
                    if self.logger:
                        self.logger.warning("LLM returned no content without MCP")
                    return {"_needs_forced_action": True, "messages": mcp_context.messages}
        finally:
            # Ensure MCP session is always disconnected (Req 3.5)
            if mcp_context.mcp_connected and self.mcp_manager:
                await self.mcp_manager.disconnect_session()

    def _parse_agent_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response content into action data.

        Args:
            content: Raw LLM response content (should be JSON)

        Returns:
            Dict with 'action', 'reasoning', and optional 'new_objective'
        """
        try:
            data = json.loads(content)
            return {
                "action": data.get("action", "look"),
                "reasoning": data.get("thinking", ""),
                "new_objective": data.get("new_objective"),
            }
        except json.JSONDecodeError:
            # Fallback for non-JSON response
            if self.logger:
                self.logger.warning(f"Failed to parse agent response as JSON: {content[:100]}")
            return {
                "action": content.strip() if content else "look",
                "reasoning": "",
            }

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
                response_format=create_json_schema(AgentResponse),
            )

            response = self.client.chat.completions.create(**client_args)
            raw_response = response.content.strip()

            # Parse structured JSON response
            try:
                agent_response = AgentResponse.model_validate_json(raw_response)
            except Exception as e:
                # Fallback to safe defaults on parsing error
                if self.logger:
                    self.logger.error(f"Failed to parse agent response: {e}")
                    self.logger.error(f"Raw response: {raw_response}")
                agent_response = AgentResponse(
                    thinking="[Error parsing response]",
                    action="look",
                    new_objective=None
                )

            # Clean and validate action
            cleaned_action = self._clean_action(agent_response.action)

            return {
                "action": cleaned_action,
                "reasoning": agent_response.thinking,
                "new_objective": agent_response.new_objective,
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
                "new_objective": None,
                "raw_response": None,
            }  # Default safe action on error

    def _clean_action(self, action: str) -> str:
        """Clean and validate an action command from the agent.

        Args:
            action: Raw action string from agent response

        Returns:
            Cleaned action string (lowercase, trimmed, validated)
        """
        # Remove any remaining tags or formatting
        cleaned = action.strip()
        cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"`", "", cleaned)
        cleaned = cleaned.strip()

        # Convert to lowercase for game compatibility
        cleaned = cleaned.lower()

        # Remove leading/trailing punctuation (defensive)
        cleaned = cleaned.strip(".,!?;:")

        # Ensure non-empty
        if not cleaned:
            if self.logger:
                self.logger.warning("Agent returned empty action, defaulting to 'look'")
            cleaned = "look"

        return cleaned

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
