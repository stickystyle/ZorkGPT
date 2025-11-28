# MCP Integration Specification for ZorkGPT

**Version:** 1.0
**Date:** 2025-01-25
**Status:** Draft for Implementation
**Authors:** Ryan Parrish & Claude

---

## Executive Summary

This specification defines the integration of Model Context Protocol (MCP) client support into ZorkGPT, enabling the agent to use external tools during puzzle solving. The integration follows an **LLM-driven tool calling** approach where the agent autonomously decides when to use the `thoughtbox` tool for structured reasoning.

**Key Goals:**
- Enable ZorkGPT agent to call MCP tools during action generation
- Integrate thoughtbox MCP server for structured reasoning
- Maintain ZorkGPT's principle: "all reasoning from LLMs"
- Ensure tools are used for reasoning only - final output must be a game action
- Provide robust error handling and observability

**High-Level Approach:**
- Install Python MCP SDK (`mcp` package)
- Extend `LLMClient` to support OpenAI-style tool calling
- Implement tool-calling loop in `ZorkAgent` with async/sync bridge
- Create `MCPManager` for server lifecycle management
- Configure server via `mcp_config.json` in project root
- Track all tool calls in Langfuse and JSON logs

---

## Architecture Overview

### Design Principles

1. **LLM-Driven Tool Calling**: Agent decides when to use tools based on context and available tool schemas
2. **Fail-Fast Configuration**: Configuration errors crash early; runtime errors degrade gracefully
3. **Tools for Reasoning Only**: Final output must always be a Zork game command
4. **Async/Sync Bridge**: Single `asyncio.run()` boundary at agent level; orchestrator stays synchronous
5. **Single-Tier Lifecycle**: MCP session (and subprocess) per-turn via stdio transport (~50ms overhead)

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tool calling control | LLM-driven | Aligns with "all reasoning from LLMs" principle |
| Max iterations per turn | 20 | Prevents infinite loops while allowing complex reasoning |
| MCP session lifecycle | Per-turn (connect/disconnect) | Clean state, ~50ms overhead per turn; subprocess coupled to session with stdio transport |
| Server configuration | Dedicated `mcp_config.json` | Standard format, compatible with Claude Desktop/Cline |
| Server count | Single server (V1) | Simplicity; multi-server support deferred to future version |
| Tool name format | Prefix with server name | e.g., `thoughtbox.think` for namespacing (future-proofs for multi-server) |
| Async/sync bridge | Single boundary at agent | `asyncio.run()` in `get_action_with_reasoning()` only |
| Error handling | Fail-fast for config, graceful for runtime | Makes debugging easier, prevents silent failures |
| Observability | Langfuse + JSON logs | Separate events for each tool call |
| Tool message format | OpenAI-compatible | Works with OpenRouter's unified API |

### Design Notes

**Token Budget:**
Tool calling adds tokens to each request: tool schemas (~200-500 tokens) plus growing message
history with tool calls and results. Expect ~300-500 additional tokens per tool-calling iteration.
The 20-iteration cap provides a practical limit. Most turns will use zero tool calls - the LLM
decides autonomously when structured reasoning is needed. Token budget management can be added
in a future version if costs become a concern.

**Prompt Caching:**
Prompt caching remains effective during tool-calling loops. The system prompt and initial game
state (user message) are cached across iterations within a turn. Growing tool-call history does
not invalidate the cached prefix - it simply extends the non-cached portion.

**Tool Message Format:**
This spec uses OpenAI-compatible tool message format (`role: "tool"` with `tool_call_id`).
This works with OpenRouter's unified API which normalizes provider-specific formats.
Direct Anthropic API usage would require format translation to `tool_result` content blocks.

---

## Component Design

### 1. MCPManager

**Purpose**: Manage MCP server connections and tool discovery

**Location**: `managers/mcp_manager.py`

**Responsibilities:**
- Load MCP server configuration from `mcp_config.json`
- Connect MCP session (spawns subprocess) at turn start
- Disconnect MCP session (terminates subprocess) at turn end
- Discover available tools from server
- Execute tool calls with timeout handling
- Handle server startup failures with graceful degradation

**Lifecycle:**
- **Single-Tier**: Session and subprocess are coupled with stdio transport
- **Per-Turn**: Fresh subprocess + session each turn (~50ms overhead)
- **Health Check**: Tool timeout detects hung process; session errors trigger graceful degradation

**Key Methods:**

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPManager:
    def __init__(
        self,
        config: GameConfiguration,
        logger: Logger,
        langfuse_client: Optional[LangfuseClient] = None,
    ):
        """Initialize MCP manager with config, logger, and optional Langfuse client."""
        self._stdio_context = None  # Manages subprocess + stdio pipes
        self._session = None        # MCP protocol session
        self._disabled = False      # Set True after repeated failures (graceful degradation)
        self._retry_attempted = False  # Track retry attempts for graceful degradation
        self.langfuse_client = langfuse_client

    @property
    def is_disabled(self) -> bool:
        """Check if MCP has been disabled due to repeated failures."""
        return self._disabled

    async def connect_session(self) -> None:
        """Connect MCP session and spawn subprocess (once per turn).

        With stdio transport, the subprocess lifecycle is coupled to the session.
        Entering the stdio_client context spawns the process; exiting terminates it.

        Raises:
            MCPServerStartupError: If server fails to start
        """

    async def disconnect_session(self) -> None:
        """Disconnect MCP session and terminate subprocess (end of turn)."""

    async def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas from server.

        Returns:
            List of tool schemas with prefixed names (e.g., "thoughtbox.think")
        """

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_seconds: int = 30
    ) -> ToolCallResult:
        """Call a tool on the appropriate MCP server.

        Args:
            tool_name: Prefixed tool name (e.g., "thoughtbox.think")
            arguments: Tool arguments as dict
            timeout_seconds: Timeout for tool execution

        Returns:
            ToolCallResult with content or error

        Raises:
            asyncio.TimeoutError: If tool call exceeds timeout
        """
```

**MCP SDK Integration:**

```python
async def connect_session(self) -> None:
    """Connect to MCP server via stdio transport.

    Spawns subprocess and establishes MCP session (single-tier lifecycle).
    """
    server_params = StdioServerParameters(
        command=self.server_config.command,
        args=self.server_config.args,
        env={**os.environ, **(self.server_config.env or {})}
    )

    # stdio_client spawns subprocess and provides read/write streams
    self._stdio_context = stdio_client(server_params)
    read, write = await self._stdio_context.__aenter__()

    self._session = ClientSession(read, write)
    await self._session.initialize()

async def disconnect_session(self) -> None:
    """Disconnect MCP session and terminate subprocess (single-tier lifecycle)."""
    self._session = None
    if self._stdio_context:
        await self._stdio_context.__aexit__(None, None, None)
        self._stdio_context = None
```

**Server Discovery:**
```python
async def _discover_tools_from_server(
    self,
    server_name: str,
    session: ClientSession
) -> List[ToolSchema]:
    """Discover tools from a single MCP server.

    Translates MCP tool schemas to OpenAI tool calling format
    and prefixes tool names with server name.
    """
    tools = await session.list_tools()

    openai_tools = []
    for tool in tools.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": f"{server_name}.{tool.name}",
                "description": tool.description,
                "parameters": self._translate_schema(tool.inputSchema)
            }
        })

    return openai_tools
```

**Configuration Loading:**
```python
def _load_mcp_config(self) -> Dict[str, MCPServerConfig]:
    """Load MCP server configuration from mcp_config.json.

    Returns:
        Dict mapping server name to MCPServerConfig

    Raises:
        FileNotFoundError: If mcp_config.json not found and MCP is enabled
        ValidationError: If config format is invalid
    """
```

### 2. LLMClient Extensions

**Purpose**: Add tool calling support to existing LLM client

**Location**: `llm_client.py` (extend existing file)

**Changes Required:**

**2.1. Extend LLMResponse:**
```python
@dataclass
class LLMResponse:
    """Response object for LLM completions."""

    content: Optional[str]  # Now optional (None if tool_calls present)
    model: str
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[ToolCall]] = None  # NEW
    finish_reason: Optional[str] = None  # NEW: "stop" | "tool_calls" | "length"


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str  # Unique identifier for this tool call
    type: str  # Always "function" for now
    function: FunctionCall


@dataclass
class FunctionCall:
    """Function call details."""

    name: str  # Tool name (e.g., "thoughtbox.think")
    arguments: str  # JSON string of arguments
```

**2.2. Add tools parameter to chat_completions_create:**
```python
def chat_completions_create(
    self,
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,  # NEW
    tool_choice: Optional[Union[str, Dict]] = None,  # NEW
    **kwargs,
) -> LLMResponse:
    """Create a chat completion with advanced sampling parameters and retry logic.

    Args:
        tools: List of tool schemas in OpenAI format
        tool_choice: "auto" | "none" | {"type": "function", "function": {"name": "..."}}
    """
```

**2.3. Handle tool_calls in response parsing:**
```python
def _execute_request(
    self,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any]
) -> LLMResponse:
    """Execute the actual HTTP request to the LLM API."""

    # ... existing request logic ...

    response_data = response.json()

    # Extract tool calls if present
    tool_calls = None
    finish_reason = None

    if "choices" in response_data and len(response_data["choices"]) > 0:
        choice = response_data["choices"][0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message", {})

        # Check for tool calls
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    )
                )
                for tc in message["tool_calls"]
            ]
            content = None  # No content when tool calls present
        else:
            content = message.get("content")

    return LLMResponse(
        content=content,
        model=model,
        usage=usage,
        tool_calls=tool_calls,
        finish_reason=finish_reason
    )
```

**2.4. Model Compatibility Check:**
```python
def _supports_tool_calling(self, model: str, config: GameConfiguration) -> bool:
    """Check if model supports tool calling.

    Returns:
        True if model supports tools, False otherwise

    Note: This is a heuristic check. Use config.mcp_force_tool_support
    to override if auto-detection is wrong.
    """
    # Config escape hatch - bypass auto-detection
    if config.mcp_force_tool_support:
        return True

    model_lower = model.lower()

    # Reasoning models explicitly don't support tools
    NO_TOOL_PATTERNS = [
        "o1-", "o3-", "qwq", "deepseek-r1", "deepseek-reasoner",
        "-reasoning", "r1-"
    ]
    if any(pattern in model_lower for pattern in NO_TOOL_PATTERNS):
        return False

    # Known tool-supporting model families (OpenRouter naming)
    TOOL_PATTERNS = [
        "gpt-4", "gpt-3.5",           # OpenAI
        "claude-3", "claude-sonnet",   # Anthropic (covers claude-3.5-sonnet)
        "claude-opus", "claude-haiku",
        "gemini",                      # Google
        "mistral", "mixtral",          # Mistral
        "llama-3.1", "llama-3.2",      # Meta (tool support in 3.1+)
        "command-r",                   # Cohere
    ]
    if any(pattern in model_lower for pattern in TOOL_PATTERNS):
        return True

    # Default: assume support, fail at runtime if wrong
    return True
```

### 3. ZorkAgent Extensions

**Purpose**: Implement tool-calling loop in agent

**Location**: `zork_agent.py` (extend existing class)

**Changes Required:**

**3.1. Add MCPManager to Agent:**
```python
class ZorkAgent:
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
        mcp_manager: Optional[MCPManager] = None,  # NEW
    ):
        # ... existing initialization ...

        self.mcp_manager = mcp_manager
        self.mcp_enabled = config.mcp_enabled and mcp_manager is not None
```

**3.2. Implement Async Action Generation:**
```python
async def _generate_action_async(
    self,
    game_state_text: str,
    relevant_memories: Optional[str] = None,
) -> Dict[str, str]:
    """Async implementation of action generation with tool calling.

    This is the internal async method that handles the tool-calling loop.
    Session lifecycle (connect/disconnect) is managed here.

    Returns:
        Dict with 'action', 'reasoning', and optional 'new_objective'
    """

    # Build initial messages
    if "o1" in self.model:
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

    # Add user content
    user_content = game_state_text
    if relevant_memories:
        user_content = f"{user_content}\n\n{relevant_memories}"

    messages.append({"role": "user", "content": user_content})

    # Connect MCP session and get tool schemas if MCP enabled
    tools = None
    try:
        if self.mcp_enabled and not self.mcp_manager.is_disabled:
            try:
                # Connect session (spawns subprocess with stdio transport)
                await self.mcp_manager.connect_session()

                tools = await self.mcp_manager.get_tool_schemas()

                # Check model compatibility
                if tools and not self.client._supports_tool_calling(self.model):
                    raise MCPError(
                        f"Model {self.model} does not support tool calling. "
                        f"Disable MCP in config or use a compatible model."
                    )
            except MCPError:
                # If MCP was disabled due to repeated failures, continue without tools
                if self.mcp_manager.is_disabled:
                    if self.logger:
                        self.logger.warning(
                            "MCP disabled, continuing without tools",
                            extra={"event_type": "mcp_disabled_fallback"}
                        )
                    tools = None
                else:
                    raise  # Re-raise if not a graceful degradation case

        # Tool-calling loop
        iteration = 0
        max_iterations = self.config.mcp_max_tool_iterations

    while iteration < max_iterations:
        iteration += 1

        # Log iteration start
        if self.logger and tools:
            self.logger.info(
                f"Agent tool-calling iteration {iteration}/{max_iterations}",
                extra={
                    "event_type": "mcp_iteration_start",
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "episode_id": self.episode_id,
                }
            )

        # Call LLM
        # Note: response_format disabled during tool loop for OpenRouter compatibility
        # JSON schema enforcement applied only on forced final call
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
            response_format=None,  # Disabled during tool loop
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
        )

        response = self.client.chat.completions.create(**client_args)

        # Check for tool calls first (takes priority over content)
        if response.tool_calls:
            # LLM wants to call tools

            # Add assistant message with tool calls to history
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

            # Execute each tool call sequentially
            # Note: V1 processes tool calls serially to avoid MCP session concurrency issues.
            # Parallel execution could be added in future if performance becomes a concern.
            for tool_call in response.tool_calls:
                try:
                    # Parse arguments
                    import json
                    arguments = json.loads(tool_call.function.arguments)

                    # Log tool call
                    if self.logger:
                        self.logger.info(
                            f"Agent calling tool: {tool_call.function.name}",
                            extra={
                                "event_type": "mcp_tool_call",
                                "tool_name": tool_call.function.name,
                                "arguments": arguments,
                                "iteration": iteration,
                                "episode_id": self.episode_id,
                            }
                        )

                    # Execute tool via MCP
                    result = await self.mcp_manager.call_tool(
                        tool_name=tool_call.function.name,
                        arguments=arguments,
                        timeout_seconds=self.config.mcp_tool_call_timeout_seconds
                    )

                    # Add tool result to messages (consistent format via ToolCallResult)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result.to_dict())
                    })

                    # Log tool result
                    if self.logger:
                        self.logger.info(
                            f"Tool result from {tool_call.function.name}",
                            extra={
                                "event_type": "mcp_tool_result",
                                "tool_name": tool_call.function.name,
                                "result_length": len(str(result.content)),
                                "iteration": iteration,
                                "episode_id": self.episode_id,
                            }
                        )

                except asyncio.TimeoutError:
                    # Timeout - add error via ToolCallResult for consistent format
                    if self.logger:
                        self.logger.warning(
                            f"Tool call timeout: {tool_call.function.name}",
                            extra={
                                "event_type": "mcp_tool_timeout",
                                "tool_name": tool_call.function.name,
                                "iteration": iteration,
                                "episode_id": self.episode_id,
                            }
                        )

                    timeout_result = ToolCallResult(
                        content=None,
                        is_error=True,
                        error_message=f"Tool call timed out after {self.config.mcp_tool_call_timeout_seconds}s"
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(timeout_result.to_dict())
                    })
                    # Timeout is severe - skip remaining tool calls in this batch
                    # and let LLM decide how to proceed with partial results
                    break

                except Exception as e:
                    # Tool call failed - add error via ToolCallResult for consistent format
                    if self.logger:
                        self.logger.warning(
                            f"Tool call failed: {tool_call.function.name} - {e}",
                            extra={
                                "event_type": "mcp_tool_error",
                                "tool_name": tool_call.function.name,
                                "error": str(e),
                                "iteration": iteration,
                                "episode_id": self.episode_id,
                            }
                        )

                    error_result = ToolCallResult(
                        content=None,
                        is_error=True,
                        error_message=f"Tool call failed: {str(e)}"
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_result.to_dict())
                    })
                    # Non-timeout errors are recoverable - continue with remaining tool calls
                    continue

            # All tool calls processed (or batch aborted on timeout)
            # Continue outer loop to call LLM with results
            continue

        # No tool calls - check if we have content (final answer)
        if response.content:
            break

        # No tool calls and no content - unexpected state
        if self.logger:
            self.logger.warning(
                f"LLM returned no content and no tool_calls, finish_reason={response.finish_reason}",
                extra={
                    "event_type": "mcp_unexpected_state",
                    "finish_reason": response.finish_reason,
                    "iteration": iteration,
                    "episode_id": self.episode_id,
                }
            )
        break

    # After loop - check if we need to force a final answer
    if not response.content:
        if self.logger:
            self.logger.warning(
                f"Agent exited tool loop without content after {iteration} iterations",
                extra={
                    "event_type": "mcp_no_content",
                    "iterations": iteration,
                    "episode_id": self.episode_id,
                }
            )

        # Force LLM to give final answer with response_format for JSON guarantee
        messages.append({
            "role": "user",
            "content": "Please provide your final action now without calling any more tools."
        })

        client_args["tools"] = None
        client_args["tool_choice"] = None
        # Use existing ZorkGPT utilities (shared_utils.create_json_schema, zork_agent.AgentResponse)
        client_args["response_format"] = create_json_schema(AgentResponse)
        response = self.client.chat.completions.create(**client_args)

    # Parse final response
    if not response.content:
        raise MCPError(
            "Agent failed to return final action after tool calling. "
            f"Last finish_reason: {response.finish_reason}"
        )

        try:
            # AgentResponse is the existing Pydantic model from zork_agent.py:
            # - thinking: str (reasoning)
            # - action: str (game command)
            # - new_objective: Optional[str] (multi-step plan tracking)
            agent_response = AgentResponse.model_validate_json(response.content)
            result = {
                "action": agent_response.action,
                "reasoning": agent_response.thinking,
                "new_objective": agent_response.new_objective
            }
        except Exception as e:
            # Fallback to safe defaults on parsing error
            if self.logger:
                self.logger.error(
                    f"Failed to parse agent response: {e}",
                    extra={
                        "event_type": "agent_parse_error",
                        "raw_response": response.content,
                        "episode_id": self.episode_id,
                    }
                )
            result = {
                "action": "look",
                "reasoning": f"Parse error: {e}. Raw: {response.content[:100]}"
            }

        return result

    finally:
        # Always disconnect MCP session at end of turn (terminates subprocess)
        if self.mcp_enabled:
            await self.mcp_manager.disconnect_session()
```

**3.3. Sync Wrapper (Backward Compatibility):**
```python
def get_action_with_reasoning(
    self,
    game_state_text: str,
    relevant_memories: Optional[str] = None,
) -> Dict[str, str]:
    """Gets an action from the Agent LM with reasoning preserved.

    This is the public sync wrapper that maintains backward compatibility.
    Uses asyncio.run() to bridge sync/async boundary.

    Args:
        game_state_text: Current game state text
        relevant_memories: Formatted string of relevant memories

    Returns:
        Dict with 'action' (cleaned) and 'reasoning' (raw thinking/reasoning)
    """

    # If MCP is enabled, use async implementation
    if self.mcp_enabled:
        return asyncio.run(
            self._generate_action_async(game_state_text, relevant_memories)
        )

    # Otherwise, use existing synchronous implementation
    # ... existing code unchanged ...
```

### 4. Orchestrator Integration

**Purpose**: Initialize MCPManager and pass to agent (orchestrator stays synchronous)

**Location**: `orchestration/zork_orchestrator_v2.py`

**Architecture Note**: The orchestrator remains fully synchronous. All async MCP operations
are encapsulated within `ZorkAgent.get_action_with_reasoning()` which uses a single
`asyncio.run()` boundary internally. This minimizes changes to existing orchestrator code.

**Changes Required:**

**4.1. Add MCPManager Initialization:**
```python
def _initialize_managers(self) -> None:
    """Initialize all specialized managers."""

    # ... existing manager initialization ...

    # MCP Manager (if enabled)
    self.mcp_manager = None
    if self.config.mcp_enabled:
        try:
            self.mcp_manager = MCPManager(
                config=self.config,
                logger=self.logger,
                langfuse_client=self.langfuse_client,
            )
            self.logger.info(
                "MCP Manager initialized",
                extra={
                    "event_type": "mcp_manager_init",
                    "episode_id": self.game_state.episode_id,
                }
            )
        except Exception as e:
            # Fail fast on MCP initialization errors
            self.logger.error(
                f"Failed to initialize MCP Manager: {e}",
                extra={
                    "event_type": "mcp_init_error",
                    "error": str(e),
                    "episode_id": self.game_state.episode_id,
                }
            )
            raise MCPError(f"MCP initialization failed: {e}")
```

**4.2. Turn Processing (Unchanged):**

The orchestrator's `_process_turn()` method remains synchronous. MCP session lifecycle
(connect/disconnect, which also spawns/terminates subprocess) happens inside the agent's
async implementation via try/finally:

```python
def _process_turn(self) -> bool:
    """Process a single turn (remains synchronous).

    MCP lifecycle (single-tier: session + subprocess) is handled internally by
    ZorkAgent.get_action_with_reasoning() within its asyncio.run() boundary.
    """
    # ... existing turn processing logic ...

    # Agent generates action (handles MCP internally if enabled)
    action_data = self.agent.get_action_with_reasoning(
        game_state_text=context,
        relevant_memories=memories_context
    )

    # ... rest of turn processing ...
    return True
```

**4.3. Pass MCPManager to Agent:**
```python
def _initialize_game_components(self) -> None:
    """Initialize core game components (agent, critic, extractor)."""

    # Initialize agent with config and MCP manager
    self.agent = ZorkAgent(
        config=self.config,
        logger=self.logger,
        episode_id=self.game_state.episode_id,
        model=self.config.agent_model,
        mcp_manager=self.mcp_manager if hasattr(self, 'mcp_manager') else None,
    )

    # ... rest of initialization ...
```

---

## Data Structures

### MCPServerConfig

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    command: str = Field(
        description="Command to launch the server (e.g., 'npx', 'uvx')"
    )
    args: List[str] = Field(
        description="Arguments for the command (e.g., ['-y', '@kastalien-research/thoughtbox'])"
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        description="Environment variables for the server process"
    )
```

### ToolCallResult

```python
@dataclass
class ToolCallResult:
    """Result from an MCP tool call."""

    content: Any  # Tool result (can be dict, str, list, etc.)
    is_error: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        if self.is_error:
            return {
                "error": self.error_message,
                "content": self.content
            }
        return {"content": self.content}
```

### MCPError

```python
class MCPError(Exception):
    """Base exception for MCP-related errors."""
    pass

class MCPServerStartupError(MCPError):
    """Raised when an MCP server fails to start."""
    pass

class MCPToolCallError(MCPError):
    """Raised when a tool call fails."""
    pass
```

---

## Message Flow

### Tool-Calling Sequence Diagram

```
User Context
     ↓
┌────────────────────────────────────────────────────┐
│ 1. Agent receives context + tool schemas          │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│ 2. LLM Call #1 (with tools parameter)             │
│    Response: finish_reason="tool_calls"            │
│    tool_calls=[{name: "thoughtbox.think", ...}]   │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│ 3. Execute tool via MCP                            │
│    MCPManager.call_tool("thoughtbox.think", {...}) │
│    → Connect to server via stdio                   │
│    → Send JSON-RPC request                         │
│    → Receive result                                │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│ 4. Append tool result to message history           │
│    messages += [                                   │
│      {role: "assistant", tool_calls: [...]},       │
│      {role: "tool", tool_call_id: "...",          │
│       content: "...result..."}                     │
│    ]                                               │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│ 5. LLM Call #2 (with updated history)             │
│    Response: finish_reason="stop"                  │
│    content: {"thinking": "...", "action": "..."}   │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│ 6. Parse AgentResponse and return action          │
└────────────────────────────────────────────────────┘
     ↓
Action executed in Zork
```

### Message History Structure

Example of message history after tool calling:

```python
messages = [
    {
        "role": "system",
        "content": "You are ZorkGPT...",
        "cache_control": {"type": "ephemeral"}
    },
    {
        "role": "user",
        "content": "Current location: West of House\n..."
    },
    {
        # First LLM response with tool calls
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "thoughtbox.think",
                    "arguments": '{"thought": "I should explore the puzzle...", "thoughtNumber": 1, "totalThoughts": 5, "nextThoughtNeeded": true}'
                }
            }
        ]
    },
    {
        # Tool result (wrapped in ToolCallResult format)
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": '{"content": {"thoughtNumber": 1, "totalThoughts": 5, "nextThoughtNeeded": true}}'
    },
    {
        # Second LLM response with more tool calls
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_def456",
                "type": "function",
                "function": {
                    "name": "thoughtbox.think",
                    "arguments": '{"thought": "Based on the puzzle layout...", "thoughtNumber": 2, "totalThoughts": 5, "nextThoughtNeeded": true}'
                }
            }
        ]
    },
    {
        # Tool result (wrapped in ToolCallResult format)
        "role": "tool",
        "tool_call_id": "call_def456",
        "content": '{"content": {"thoughtNumber": 2, "totalThoughts": 5, "nextThoughtNeeded": true}}'
    },
    # ... more iterations ...
    {
        # Final LLM response with action (matches AgentResponse schema)
        "role": "assistant",
        "content": '{"thinking": "After analyzing...", "action": "examine door", "new_objective": null}'
    }
]
```

---

## Configuration Schema

### pyproject.toml

```toml
[tool.zorkgpt.mcp]
# Global MCP enable/disable flag
enabled = true

# Path to MCP server configuration file (relative to project root)
config_file = "mcp_config.json"

# Maximum tool-calling iterations per turn
max_tool_iterations = 20

# Timeout for individual tool calls (seconds)
tool_call_timeout_seconds = 30

# Timeout for MCP server startup (seconds)
server_startup_timeout_seconds = 10

# Override model compatibility auto-detection (escape hatch)
# Set to true if auto-detection incorrectly flags your model as incompatible
force_tool_support = false
```

### mcp_config.json

**Note:** V1 supports a single MCP server only. The config format uses `mcpServers` dict
for compatibility with Claude Desktop/Cline, but only the first server entry is used.
Multi-server support may be added in a future version.

```json
{
  "mcpServers": {
    "thoughtbox": {
      "command": "npx",
      "args": ["-y", "@kastalien-research/thoughtbox"],
      "env": {
        "DISABLE_THOUGHT_LOGGING": "true"
      }
    }
  }
}
```

**Schema Validation:**

```python
class MCPConfig(BaseModel):
    """Root MCP configuration schema.

    Note: V1 supports single server only. If multiple servers are configured,
    only the first one is used.
    """

    mcpServers: Dict[str, MCPServerConfig] = Field(
        description="Map of server name to server configuration (V1: first entry only)"
    )

    @classmethod
    def from_file(cls, path: str) -> "MCPConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.model_validate(data)

    def get_server_config(self) -> Tuple[str, MCPServerConfig]:
        """Get the single server config (V1: first entry only).

        Returns:
            Tuple of (server_name, server_config)

        Raises:
            ValueError: If no servers configured
        """
        if not self.mcpServers:
            raise ValueError("No MCP servers configured in mcp_config.json")
        server_name = next(iter(self.mcpServers))
        return server_name, self.mcpServers[server_name]
```

### GameConfiguration Extensions

Add to `session/game_configuration.py`:

```python
@dataclass
class GameConfiguration:
    # ... existing fields ...

    # MCP Configuration
    mcp_enabled: bool = False
    mcp_config_file: str = "mcp_config.json"
    mcp_max_tool_iterations: int = 20
    mcp_tool_call_timeout_seconds: int = 30
    mcp_server_startup_timeout_seconds: int = 10
    mcp_force_tool_support: bool = False  # Escape hatch for model compatibility

    @classmethod
    def from_toml(cls, config_path: str = "pyproject.toml") -> "GameConfiguration":
        """Load configuration from TOML file."""
        # ... existing loading logic ...

        # Load MCP config
        mcp_config = config.get("tool", {}).get("zorkgpt", {}).get("mcp", {})
        instance.mcp_enabled = mcp_config.get("enabled", False)
        instance.mcp_config_file = mcp_config.get("config_file", "mcp_config.json")
        instance.mcp_max_tool_iterations = mcp_config.get("max_tool_iterations", 20)
        instance.mcp_tool_call_timeout_seconds = mcp_config.get("tool_call_timeout_seconds", 30)
        instance.mcp_server_startup_timeout_seconds = mcp_config.get("server_startup_timeout_seconds", 10)
        instance.mcp_force_tool_support = mcp_config.get("force_tool_support", False)

        return instance
```

---

## Error Handling

### Failure Scenarios and Strategies

| Scenario | Detection | Strategy | Implementation |
|----------|-----------|----------|----------------|
| **MCP server fails to start** | Timeout or exception during `connect_session()` | **Fail fast**: Raise `MCPServerStartupError`, crash episode | Catch in orchestrator or agent |
| **Tool call fails** | Exception during `call_tool()` | **Skip**: Log warning, add error to message history, continue | Catch in agent tool-calling loop |
| **Tool call timeout** | `asyncio.TimeoutError` after N seconds | **End loop**: Add timeout message, force final action | Catch in agent tool-calling loop |
| **Model doesn't support tools** | Check model name against known patterns | **Fail fast**: Raise `MCPError` with clear message | Check before first tool call |
| **Max iterations reached** | No content after loop | **Force action**: Append message, call LLM with response_format | After loop condition check |
| **Session fails on subsequent turn** | Exception on `connect_session()` | **Retry once**: Try connect again, disable MCP if fails | Catch in session connect |
| **Server dies mid-turn** | Exception during tool call | **End loop**: Treat as tool error, force final action | Catch in tool execution |
| **Invalid tool arguments** | JSON parse error or validation error | **Skip tool**: Log error, add error to messages | Catch in tool execution |
| **MCP config file missing** | `FileNotFoundError` when loading config | **Fail fast**: Raise `MCPError` if MCP enabled | Catch in `MCPManager.__init__` |
| **MCP config invalid JSON** | JSON parse error | **Fail fast**: Raise `MCPError` with parse details | Catch in config loading |

### Graceful Degradation

When MCP session fails on a subsequent turn (after working earlier):

1. **On `connect_session()` failure**: Attempt to connect again once
2. **If retry succeeds**: Continue with MCP enabled
3. **If retry fails**: Disable MCP for rest of episode, log warning, continue without tools
4. **Agent continues**: Falls back to non-MCP action generation

```python
async def connect_session(self) -> None:
    """Connect MCP session with retry-on-failure."""
    try:
        await self._connect_session_impl()
    except Exception as e:
        if not self._retry_attempted:
            self._retry_attempted = True
            self.logger.warning(f"MCP session connect failed, retrying: {e}")
            await self._connect_session_impl()  # May raise if retry failed
        else:
            # Retry already attempted, disable MCP for rest of episode
            self.logger.warning(f"MCP retry failed, disabling MCP for episode: {e}")
            self._disabled = True
            raise
```

### Error Message Examples

```python
# Server startup failure
raise MCPServerStartupError(
    f"Failed to start MCP server 'thoughtbox': {error_details}\n"
    f"Command: npx -y @kastalien-research/thoughtbox\n"
    f"Check that Node.js is installed and the package is available."
)

# Model incompatibility
raise MCPError(
    f"Model '{self.model}' does not support tool calling.\n"
    f"Please either:\n"
    f"  1. Set mcp.enabled = false in pyproject.toml, or\n"
    f"  2. Use a compatible model (gpt-4, claude-3, etc.)"
)

# Configuration file missing
raise MCPError(
    f"MCP is enabled but config file not found: {config_path}\n"
    f"Please either:\n"
    f"  1. Create mcp_config.json in project root, or\n"
    f"  2. Set mcp.enabled = false in pyproject.toml"
)
```

---

## Observability

### Langfuse Integration

**Tool Call Events:**

Each tool call should be logged as a separate Langfuse span (matching ZorkGPT's existing pattern):

```python
async def call_tool(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: int = 30
) -> ToolCallResult:
    """Call a tool on the appropriate MCP server."""

    if self.langfuse_client:
        with self.langfuse_client.start_as_current_span(
            name=f"mcp-tool-{tool_name}",
            input=arguments,
            metadata={
                "tool_name": tool_name,
                "server_name": server_name,
                "timeout_seconds": timeout_seconds
            }
        ) as span:
            result = await self._execute_tool_call(...)

            span.update(
                output=result.to_dict(),
                metadata={
                    "is_error": result.is_error,
                    "duration_ms": duration
                }
            )

            return result
    else:
        return await self._execute_tool_call(...)
```

**Tool-Calling Session:**

The entire tool-calling session should be wrapped (matching ZorkGPT's existing pattern):

```python
async def _generate_action_async(...):
    if self.langfuse_client:
        with self.langfuse_client.start_as_current_span(
            name="agent-tool-calling-session",
            metadata={
                "mcp_enabled": self.mcp_enabled,
                "max_iterations": max_iterations
            }
        ) as session_span:
            result = await self._tool_calling_loop(...)

            session_span.update(
                metadata={
                    "iterations_used": iteration,
                    "tools_called": len(tool_calls),
                    "final_action": result["action"]
                }
            )

            return result
    else:
        return await self._tool_calling_loop(...)
```

### JSON Logging

**MCP Events:**

```python
# Tool call start
logger.info(
    f"MCP tool call: {tool_name}",
    extra={
        "event_type": "mcp_tool_call",
        "tool_name": tool_name,
        "server_name": server_name,
        "arguments": arguments,
        "iteration": iteration,
        "episode_id": episode_id,
        "turn": turn_number,
    }
)

# Tool call result
logger.info(
    f"MCP tool result: {tool_name}",
    extra={
        "event_type": "mcp_tool_result",
        "tool_name": tool_name,
        "server_name": server_name,
        "result_type": type(result.content).__name__,
        "result_length": len(str(result.content)),
        "is_error": result.is_error,
        "duration_ms": duration,
        "iteration": iteration,
        "episode_id": episode_id,
        "turn": turn_number,
    }
)

# Session summary
logger.info(
    f"MCP session complete: {iterations} iterations, {tool_calls_count} tool calls",
    extra={
        "event_type": "mcp_session_complete",
        "iterations": iterations,
        "tool_calls_count": tool_calls_count,
        "tools_used": list(set(tool_names)),
        "final_action": action,
        "episode_id": episode_id,
        "turn": turn_number,
    }
)
```

### Episode Logs

Human-readable format in episode log file:

```
[INFO] Turn 15: MCP session started (max 20 iterations)
[INFO] Turn 15 Iteration 1: Agent calling tool thoughtbox.think
[DEBUG] Turn 15 Iteration 1: Tool arguments: {"thought": "I should explore...", ...}
[INFO] Turn 15 Iteration 1: Tool result received (0.3s)
[INFO] Turn 15 Iteration 2: Agent calling tool thoughtbox.think
[DEBUG] Turn 15 Iteration 2: Tool arguments: {"thought": "Based on the puzzle...", ...}
[INFO] Turn 15 Iteration 2: Tool result received (0.2s)
[INFO] Turn 15: MCP session complete (2 iterations, 2 tool calls)
[INFO] Turn 15: Agent action: examine door
```

---

## Implementation Plan

### Phase 1: Core Infrastructure

**Goal**: Get basic MCP client working with single server

**Tasks:**
1. Install dependencies: `uv add "mcp[cli]"`
2. Create `managers/mcp_manager.py` with basic structure
3. Extend `LLMResponse` dataclass with `tool_calls` and `finish_reason`
4. Add `tools` parameter to `LLMClient.chat_completions_create()`
5. Implement tool call parsing in `_execute_request()`
6. Add MCP configuration to `GameConfiguration`
7. Create example `mcp_config.json` with thoughtbox only

**Testing:**
- Unit tests for `LLMResponse` extensions
- Unit tests for tool schema parsing
- Mock tests for MCP server connections

**Milestone**: LLMClient can parse tool calls from LLM responses

### Phase 2: Tool-Calling Loop

**Goal**: Implement agent tool-calling logic

**Tasks:**
1. Implement `MCPManager.connect_server()` and `disconnect_server()`
2. Implement `MCPManager.get_tool_schemas()` with schema translation
3. Implement `MCPManager.call_tool()` with timeout handling
4. Add `ZorkAgent._generate_action_async()` with tool-calling loop
5. Add `ZorkAgent.get_action_with_reasoning()` sync wrapper
6. Implement error handling for all failure scenarios
7. Add iteration counter and max iterations check

**Testing:**
- Integration tests with mock MCP server
- Test tool-calling loop with successful calls
- Test max iterations handling
- Test timeout handling
- Test tool call failures

**Milestone**: Agent can call MCP tools during action generation

### Phase 3: Orchestrator Integration

**Goal**: Connect MCP to orchestrator lifecycle

**Tasks:**
1. Initialize `MCPManager` in orchestrator
2. Pass `MCPManager` to `ZorkAgent`
3. Add MCP lifecycle to turn processing (connect/disconnect)
4. Make orchestrator turn processing async where needed
5. Add fail-fast error handling for startup errors
6. Test with real thoughtbox MCP server

**Testing:**
- End-to-end tests with real MCP server
- Test turn lifecycle (connect/disconnect)
- Test server startup failures
- Test episode with MCP enabled vs disabled

**Milestone**: Full turn processing with MCP working

### Phase 4: Observability & Polish

**Goal**: Add logging and production-readiness

**Tasks:**
1. Add Langfuse tracking for tool calls
2. Add JSON logging for all MCP events
3. Add episode log formatting for tool calls
4. Implement model compatibility checking
5. Add configuration validation
6. Write documentation and examples
7. Performance testing and optimization

**Testing:**
- Verify Langfuse events are created
- Verify JSON logs have correct structure
- Test with incompatible models
- Test with invalid configurations
- Load testing (many tool calls)

**Milestone**: Production-ready MCP integration

---

## Testing Strategy

### Unit Tests

**LLM Client Extensions:**
```python
def test_llm_response_with_tool_calls():
    """Test LLMResponse can represent tool calls."""
    response = LLMResponse(
        content=None,
        model="gpt-4",
        tool_calls=[
            ToolCall(
                id="call_123",
                type="function",
                function=FunctionCall(
                    name="thoughtbox.think",
                    arguments='{"thought": "test"}'
                )
            )
        ],
        finish_reason="tool_calls"
    )

    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.finish_reason == "tool_calls"

def test_llm_client_supports_tools_parameter():
    """Test LLMClient accepts tools parameter."""
    client = LLMClient(config=mock_config)

    # Should not raise
    client.chat_completions_create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test"}],
        tools=[{"type": "function", "function": {...}}]
    )
```

**MCP Manager:**
```python
def test_mcp_manager_load_config():
    """Test MCPManager loads config from JSON."""
    manager = MCPManager(config=mock_config, logger=mock_logger)

    # Should load mcp_config.json
    assert manager.server_config is not None
    assert manager.server_config.command == "npx"

def test_mcp_manager_schema_translation():
    """Test MCP schemas are translated to OpenAI format."""
    manager = MCPManager(config=mock_config, logger=mock_logger)

    mcp_schema = {
        "name": "think",
        "description": "Structured reasoning",
        "inputSchema": {...}
    }

    openai_schema = manager._translate_schema(mcp_schema, server_name="thoughtbox")

    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "thoughtbox.think"
    assert "parameters" in openai_schema["function"]
```

### Integration Tests

**Tool-Calling Loop:**
```python
@pytest.mark.asyncio
async def test_agent_tool_calling_loop():
    """Test agent executes tool-calling loop correctly."""

    # Mock MCP manager
    mock_mcp = MockMCPManager()
    mock_mcp.set_tool_result("thoughtbox.think", {"nextThoughtNeeded": False})

    # Mock LLM client
    mock_llm = MockLLMClient()
    mock_llm.add_response(
        tool_calls=[...],  # First response with tool call
        finish_reason="tool_calls"
    )
    mock_llm.add_response(
        content='{"thinking": "...", "action": "examine door"}',
        finish_reason="stop"
    )

    agent = ZorkAgent(config=config, client=mock_llm, mcp_manager=mock_mcp)

    result = await agent._generate_action_async(
        game_state_text="West of House",
        relevant_memories=None
    )

    assert result["action"] == "examine door"
    assert mock_mcp.call_count == 1  # Tool was called once

@pytest.mark.asyncio
async def test_agent_max_iterations():
    """Test agent stops after max iterations."""

    mock_mcp = MockMCPManager()
    mock_llm = MockLLMClient()

    # Always return tool calls (never stop)
    for _ in range(25):
        mock_llm.add_response(
            tool_calls=[...],
            finish_reason="tool_calls"
        )

    # Final response
    mock_llm.add_response(
        content='{"thinking": "...", "action": "look"}',
        finish_reason="stop"
    )

    agent = ZorkAgent(config=config, client=mock_llm, mcp_manager=mock_mcp)

    result = await agent._generate_action_async(...)

    # Should stop at 20 iterations and force final answer
    assert mock_mcp.call_count == 20
    assert result["action"] is not None
```

### End-to-End Tests

**With Real MCP Server:**
```python
@pytest.mark.slow
@pytest.mark.mcp
def test_full_turn_with_thoughtbox():
    """Test complete turn with real thoughtbox MCP server.

    Requires: Node.js installed, @kastalien-research/thoughtbox available
    """

    # Create config with MCP enabled
    config = GameConfiguration.from_toml()
    config.mcp_enabled = True

    # Create orchestrator
    orchestrator = ZorkOrchestratorV2(episode_id="test_mcp")

    # Process one turn
    success = orchestrator._process_turn()

    assert success
    # Verify action was generated
    assert orchestrator.game_state.last_action is not None
    # Verify MCP was used (check logs)
    # ...
```

### Mock Implementations

**MockMCPManager:**
```python
class MockMCPManager:
    """Mock MCP manager for testing."""

    def __init__(self):
        self.call_count = 0
        self.tool_results = {}

    def set_tool_result(self, tool_name: str, result: Dict):
        """Set the result for a specific tool."""
        self.tool_results[tool_name] = result

    async def get_tool_schemas(self) -> List[Dict]:
        return [{
            "type": "function",
            "function": {
                "name": "thoughtbox.think",
                "description": "Structured reasoning",
                "parameters": {...}
            }
        }]

    async def call_tool(self, tool_name: str, arguments: Dict) -> ToolCallResult:
        self.call_count += 1
        result = self.tool_results.get(tool_name, {"default": "result"})
        return ToolCallResult(content=result, is_error=False)

    async def connect_server(self):
        pass

    async def disconnect_server(self):
        pass
```

---

## Dependencies

### New Python Packages

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "mcp>=1.22.0",  # Model Context Protocol SDK
]
```

### System Requirements

- **Node.js**: Required for MCP servers that use `npx` (like thoughtbox)
  - Version: 22+ recommended
  - Installation: `brew install node` (macOS) or equivalent
- **Python**: 3.11+ (for async/await support)
- **uv**: Latest version for package management

### MCP Server

The thoughtbox server will be launched automatically when MCP is enabled:

```bash
# Thoughtbox (for structured reasoning)
npx -y @kastalien-research/thoughtbox
```

---

## Migration Path

### Enabling MCP

**Step 1: Add configuration to pyproject.toml**

```toml
[tool.zorkgpt.mcp]
enabled = true
config_file = "mcp_config.json"
max_tool_iterations = 20
tool_call_timeout_seconds = 30
```

**Step 2: Create mcp_config.json**

```json
{
  "mcpServers": {
    "thoughtbox": {
      "command": "npx",
      "args": ["-y", "@kastalien-research/thoughtbox"],
      "env": {
        "DISABLE_THOUGHT_LOGGING": "true"
      }
    }
  }
}
```

**Step 3: Verify model compatibility**

Ensure agent model supports tool calling:
- ✅ GPT-4, GPT-3.5-turbo
- ✅ Claude 3 (Opus, Sonnet, Haiku)
- ❌ o1/o3, DeepSeek R1, QwQ (reasoning models don't support tools)

**Step 4: Run episode**

```bash
uv run python main.py
```

MCP will connect at turn start and be available to the agent.

### Disabling MCP

Set `enabled = false` in pyproject.toml:

```toml
[tool.zorkgpt.mcp]
enabled = false
```

Or remove the `[tool.zorkgpt.mcp]` section entirely.

### Backward Compatibility

- MCP is **opt-in**: Default is disabled
- No changes to existing code when MCP is disabled
- Agent behavior unchanged when MCP is off
- All existing tests continue to pass

---

## Example Usage

### Example 1: Thoughtbox for Puzzle Solving

**Scenario**: Agent encounters locked door puzzle

**Turn Flow**:
```
Turn 47: Agent at Location 134 (Behind Door)

Context:
- Location: Behind locked door
- Inventory: brass key, lamp, sword
- Memory: "Door requires brass key"

Agent thinking with MCP:
1. Iteration 1: thoughtbox.think("Need to unlock door...")
   → Result: {"nextThoughtNeeded": true}

2. Iteration 2: thoughtbox.think("I have brass key in inventory...")
   → Result: {"nextThoughtNeeded": true}

3. Iteration 3: thoughtbox.think("Should use key on door...")
   → Result: {"nextThoughtNeeded": false}

4. Final action: "unlock door with brass key"

Logs:
- [INFO] Turn 47: MCP session started (max 20 iterations)
- [INFO] Turn 47 Iteration 1: Agent calling tool thoughtbox.think
- [INFO] Turn 47 Iteration 1: Tool result received (0.3s)
- [INFO] Turn 47 Iteration 2: Agent calling tool thoughtbox.think
- [INFO] Turn 47 Iteration 2: Tool result received (0.2s)
- [INFO] Turn 47 Iteration 3: Agent calling tool thoughtbox.think
- [INFO] Turn 47 Iteration 3: Tool result received (0.2s)
- [INFO] Turn 47: MCP session complete (3 iterations, 3 tool calls)
- [INFO] Turn 47: Agent action: unlock door with brass key
```

### Example 2: Tool Call Failure Handling

**Scenario**: Tool call times out

```
Turn 23: Agent calls thoughtbox.think
- Timeout after 30 seconds
- Error message added to history
- Tool-calling loop ends
- Agent forced to provide final action

Logs:
- [WARNING] Turn 23 Iteration 1: Tool call timeout: thoughtbox.think
- [INFO] Turn 23: Forcing final action after tool timeout
- [INFO] Turn 23: Agent action: look
```

---

## References

### MCP Documentation
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Build an MCP Client](https://modelcontextprotocol.io/docs/develop/build-client)
- [MCP Transports](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/)

### MCP Servers
- [Thoughtbox MCP Server](https://github.com/Kastalien-Research/thoughtbox)
- [Sequential Thinking MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking)
- [MCP Server Registry](https://github.com/modelcontextprotocol/servers)

### ZorkGPT Documentation
- [ZorkGPT Architecture](CLAUDE.md)
- [Manager Documentation](managers/CLAUDE.md)
- [Testing Guide](tests/CLAUDE.md)
- [Thoughtbox Brainstorming](ideas/thinking.md)

### Related Concepts
- [OpenAI Tool Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)
- [Langfuse Observability](https://langfuse.com/docs)

---

## Appendix: Tool Schema Translation

### MCP Tool Schema Format

```json
{
  "name": "think",
  "description": "Structured reasoning step",
  "inputSchema": {
    "type": "object",
    "properties": {
      "thought": {
        "type": "string",
        "description": "Your current thinking step"
      },
      "thoughtNumber": {
        "type": "integer",
        "minimum": 1
      },
      "nextThoughtNeeded": {
        "type": "boolean"
      }
    },
    "required": ["thought", "thoughtNumber", "nextThoughtNeeded"]
  }
}
```

### OpenAI Tool Schema Format (After Translation)

```json
{
  "type": "function",
  "function": {
    "name": "thoughtbox.think",
    "description": "Structured reasoning step",
    "parameters": {
      "type": "object",
      "properties": {
        "thought": {
          "type": "string",
          "description": "Your current thinking step"
        },
        "thoughtNumber": {
          "type": "integer",
          "minimum": 1
        },
        "nextThoughtNeeded": {
          "type": "boolean"
        }
      },
      "required": ["thought", "thoughtNumber", "nextThoughtNeeded"]
    }
  }
}
```

### Translation Logic

```python
def _translate_schema(
    self,
    mcp_tool: MCPTool,
    server_name: str
) -> Dict[str, Any]:
    """Translate MCP tool schema to OpenAI format.

    Args:
        mcp_tool: Tool object from MCP server
        server_name: Name of server (for prefixing)

    Returns:
        OpenAI-compatible tool schema
    """
    return {
        "type": "function",
        "function": {
            "name": f"{server_name}.{mcp_tool.name}",
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema  # Already JSON Schema format
        }
    }
```