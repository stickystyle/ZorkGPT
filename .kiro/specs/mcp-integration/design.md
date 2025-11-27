# Design Document: MCP Integration

## Overview

This design document describes the integration of Model Context Protocol (MCP) client support into ZorkGPT. The integration enables the ZorkGPT agent to use external tools during puzzle solving through an LLM-driven tool calling approach, where the agent autonomously decides when to use structured reasoning tools.

### Key Design Principles

1. **LLM-Driven Tool Calling**: The agent decides when to use tools based on context and available tool schemas
2. **Fail-Fast Configuration**: Configuration errors crash early; runtime errors degrade gracefully
3. **Tools for Reasoning Only**: Final output must always be a Zork game command
4. **Async/Sync Bridge**: Single `asyncio.run()` boundary at agent level; orchestrator stays synchronous
5. **Single-Tier Lifecycle**: MCP session (and subprocess) per-turn via stdio transport (~50ms overhead)

### Architecture Goals

- Minimal changes to existing ZorkGPT architecture
- Clean separation of concerns (MCP management, tool calling, action generation)
- Robust error handling with graceful degradation
- Comprehensive observability for debugging and analysis
- Backward compatibility when MCP is disabled

## Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestrator (Sync)                      │
│  - Initializes MCPManager                                    │
│  - Passes MCPManager to Agent                                │
│  - Remains synchronous throughout                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    ZorkAgent (Sync/Async)                    │
│  - Public API: get_action_with_reasoning() [sync]           │
│  - Internal: _generate_action_async() [async]               │
│  - Single asyncio.run() boundary                             │
│  - Manages tool-calling loop                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      MCPManager (Async)                      │
│  - Per-turn session lifecycle                                │
│  - Tool schema discovery and translation                     │
│  - Tool execution with timeout handling                      │
│  - Graceful degradation on failures                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   MCP Server (Subprocess)                    │
│  - Thoughtbox (structured reasoning)                         │
│  - Stdio transport                                           │
│  - Coupled to session lifecycle                              │
└─────────────────────────────────────────────────────────────┘
```

### Tool-Calling Flow

```
1. Turn Start
   ├─> MCPManager.connect_session()
   │   ├─> Spawn subprocess
   │   ├─> Create ClientSession
   │   ├─> session.initialize() (handshake)
   │   └─> Discover tool schemas
   │
2. Tool-Calling Loop (max 20 iterations)
   ├─> LLM call with tool schemas
   │   ├─> If tool_calls returned:
   │   │   ├─> Execute each tool sequentially
   │   │   ├─> Append results to history
   │   │   └─> Continue loop
   │   └─> If content returned:
   │       └─> Exit loop
   │
3. Turn End
   └─> MCPManager.disconnect_session()
       ├─> Close session
       └─> Terminate subprocess
```

### Async/Sync Boundary

```
Orchestrator (Sync)
    │
    ├─> agent.get_action_with_reasoning()  [Sync wrapper]
    │       │
    │       └─> asyncio.run(
    │               _generate_action_async()  [Async implementation]
    │                   │
    │                   ├─> mcp_manager.connect_session()  [Async]
    │                   ├─> Tool-calling loop  [Async]
    │                   │   ├─> mcp_manager.call_tool()  [Async]
    │                   │   └─> llm_client.create()  [Sync]
    │                   └─> mcp_manager.disconnect_session()  [Async]
    │           )
    │
    └─> Continue orchestrator logic  [Sync]
```

## Components and Interfaces

### MCPManager

**Purpose**: Manage MCP server connections and tool execution

**Location**: `managers/mcp_manager.py`

**Key Responsibilities**:
- Load MCP server configuration from `mcp_config.json`
- Connect/disconnect MCP session per turn
- Discover and translate tool schemas
- Execute tool calls with timeout handling
- Handle server failures with graceful degradation

**Public Interface**:

```python
class MCPManager:
    def __init__(
        self,
        config: GameConfiguration,
        logger: Logger,
        langfuse_client: Optional[LangfuseClient] = None,
    ):
        """Initialize MCP manager with config, logger, and optional Langfuse client."""
        
    @property
    def is_disabled(self) -> bool:
        """Check if MCP has been disabled due to repeated failures."""
        
    async def connect_session(self) -> None:
        """Connect MCP session and spawn subprocess (once per turn).
        
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

**Internal State**:
- `_stdio_context`: Manages subprocess + stdio pipes
- `_session`: MCP protocol session
- `_disabled`: Set True after repeated failures (graceful degradation)
- `_retry_attempted`: Track retry attempts for graceful degradation
- `server_config`: Loaded from mcp_config.json

### LLMClient Extensions

**Purpose**: Add tool calling support to existing LLM client

**Location**: `llm_client.py`

**New Data Structures**:

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

**Extended Interface**:

```python
def chat_completions_create(
    self,
    model: str,
    messages: List[Dict[str, str]],
    # ... existing parameters ...
    tools: Optional[List[Dict[str, Any]]] = None,  # NEW
    tool_choice: Optional[Union[str, Dict]] = None,  # NEW
    **kwargs,
) -> LLMResponse:
    """Create a chat completion with tool calling support.
    
    Args:
        tools: List of tool schemas in OpenAI format
        tool_choice: "auto" | "none" | {"type": "function", "function": {"name": "..."}}
    """

def _supports_tool_calling(self, model: str, config: GameConfiguration) -> bool:
    """Check if model supports tool calling.
    
    Returns:
        True if model supports tools, False otherwise
        
    Note: This is a heuristic check. Use config.mcp_force_tool_support
    to override if auto-detection is wrong.
    """
```

### ZorkAgent Extensions

**Purpose**: Implement tool-calling loop in agent

**Location**: `zork_agent.py`

**New Methods**:

```python
async def _generate_action_async(
    self,
    game_state_text: str,
    relevant_memories: Optional[str] = None,
) -> Dict[str, str]:
    """Async implementation of action generation with tool calling.
    
    This is the internal async method that handles the tool-calling loop.
    Session lifecycle (connect/disconnect) is managed here.
    
    Handles both MCP enabled and disabled cases:
    - When MCP enabled: connects session, uses tools, disconnects
    - When MCP disabled: skips MCP operations, generates action directly
    
    Returns:
        Dict with 'action', 'reasoning', and optional 'new_objective'
    """

def get_action_with_reasoning(
    self,
    game_state_text: str,
    relevant_memories: Optional[str] = None,
) -> Dict[str, str]:
    """Gets an action from the Agent LM with reasoning preserved.
    
    This is the public sync wrapper that maintains backward compatibility.
    Uses asyncio.run() to bridge sync/async boundary.
    
    Always uses the async implementation regardless of MCP configuration,
    providing a single unified code path.
    
    Args:
        game_state_text: Current game state text
        relevant_memories: Formatted string of relevant memories
        
    Returns:
        Dict with 'action' (cleaned) and 'reasoning' (raw thinking/reasoning)
    """
```

**Tool-Calling Loop Logic**:

1. Build initial messages (system prompt + user context)
2. Connect MCP session and get tool schemas
3. Enter loop (max 20 iterations):
   - Call LLM with tools parameter
   - If tool_calls returned:
     - Execute each tool sequentially
     - Append tool results to history
     - Continue loop
   - If content returned:
     - Exit loop
   - If neither:
     - Log warning and exit loop
4. If no content after loop:
   - Append "provide final action" message
   - Call LLM with response_format (force JSON)
5. Parse AgentResponse and return action
6. Finally: disconnect MCP session

## Data Models

### Configuration Models

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
    mcp_force_tool_support: bool = False

class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    command: str = Field(description="Command to launch the server (e.g., 'npx', 'uvx')")
    args: List[str] = Field(description="Arguments for the command")
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")

class MCPConfig(BaseModel):
    """Root MCP configuration schema."""
    mcpServers: Dict[str, MCPServerConfig] = Field(
        description="Map of server name to server configuration (V1: first entry only)"
    )
    
    def get_server_config(self) -> Tuple[str, MCPServerConfig]:
        """Get the single server config (V1: first entry only)."""
```

### Tool Call Models

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
            return {"error": self.error_message, "content": self.content}
        return {"content": self.content}
```

### Error Models

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

### Message History Structure

Example of message history during tool calling:

```python
messages = [
    {
        "role": "system",
        "content": "You are ZorkGPT...",
        "cache_control": {"type": "ephemeral"}
    },
    {
        "role": "user",
        "content": "Current location: West of House\n...",
        "cache_control": {"type": "ephemeral"}
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
                    "arguments": '{"thought": "I should explore...", ...}'
                }
            }
        ]
    },
    {
        # Tool result
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": '{"content": {"thoughtNumber": 1, ...}}'
    },
    # ... more iterations ...
    {
        # Final LLM response with action
        "role": "assistant",
        "content": '{"thinking": "After analyzing...", "action": "examine door"}'
    }
]
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Tool Schema Availability

*For any* agent with MCP enabled, when action generation begins, tool schemas should be available to the LLM.

**Validates: Requirements 1.1**

### Property 2: Final Output is Always a Game Command

*For any* tool-calling session, regardless of the number of tool calls or iterations, the final output must be a valid Zork game command string.

**Validates: Requirements 1.4**

### Property 3: Tool Results Appear in Message History

*For any* successful tool call, the tool result should appear in the message history before the next LLM call.

**Validates: Requirements 1.3, 5.4**

### Property 4: Configuration Loading Consistency

*For any* valid pyproject.toml with MCP settings, the GameConfiguration should load those settings with values matching the file.

**Validates: Requirements 2.4**

### Property 5: Invalid JSON Configuration Rejection

*For any* mcp_config.json file containing invalid JSON, the system should raise a configuration error.

**Validates: Requirements 2.3**

### Property 6: Session Lifecycle Coupling

*For any* turn with MCP enabled, the MCP session should be connected at turn start and disconnected at turn end.

**Validates: Requirements 3.1, 3.5**

### Property 7: Environment Variable Merging

*For any* MCP server configuration with environment variables, the spawned subprocess should have those variables merged with the system environment.

**Validates: Requirements 3.2**

### Property 8: Protocol Handshake Before Discovery

*For any* MCP session establishment, session.initialize() should be called before tool schema discovery.

**Validates: Requirements 3.3, 3.4**

### Property 9: Subprocess Termination on Disconnect

*For any* MCP session, when the session disconnects, the associated subprocess should be terminated.

**Validates: Requirements 3.7**

### Property 10: Tool Schemas in OpenAI Format

*For any* request with tools parameter, the tool schemas should be in valid OpenAI format.

**Validates: Requirements 4.1**

### Property 11: Tool Calls Parsing

*For any* LLM response containing tool_calls, the LLMClient should parse and return tool call information correctly.

**Validates: Requirements 4.2**

### Property 12: Response Content Exclusivity

*For any* LLM response, if tool_calls are present then content should be None, and if content is present then tool_calls should be None.

**Validates: Requirements 4.3, 4.4**

### Property 13: Model Compatibility Detection

*For any* model name, the compatibility check should correctly identify whether the model supports tool calling based on pattern matching.

**Validates: Requirements 4.5, 12.1, 12.2, 12.3**

### Property 14: Response Format Disabled During Loop

*For any* LLM call during the tool-calling loop (before max iterations), the response_format parameter should be None.

**Validates: Requirements 4.6**

### Property 15: Cache Control Metadata

*For any* message with system or user role, the message should include cache_control metadata.

**Validates: Requirements 4.7**

### Property 16: Tool Execution via MCPManager

*For any* tool_call received from the LLM, the agent should execute it via MCPManager.call_tool().

**Validates: Requirements 5.1**

### Property 17: Sequential Tool Execution

*For any* LLM response with multiple tool_calls, the agent should execute them sequentially (not in parallel).

**Validates: Requirements 5.2**

### Property 18: Tool Arguments JSON Parsing

*For any* tool call with arguments, the agent should parse the JSON string arguments before execution.

**Validates: Requirements 5.3**

### Property 19: Loop Continuation After Tool Results

*For any* tool result added to history, the agent should call the LLM again with the updated context.

**Validates: Requirements 5.5**

### Property 20: Loop Exit on Content

*For any* LLM response with content (and no tool_calls), the agent should exit the tool-calling loop.

**Validates: Requirements 5.6**

### Property 21: Error Logging and History

*For any* tool call that fails, the agent should log the error and add an error message to the message history.

**Validates: Requirements 6.2**

### Property 22: Batch Error Handling

*For any* LLM response with multiple tool_calls, the agent should treat them as a batch for error handling purposes.

**Validates: Requirements 6.3**

### Property 23: Non-Timeout Error Recovery

*For any* tool call that fails with a non-timeout error within a batch, the agent should continue processing remaining tool calls in that batch.

**Validates: Requirements 6.4**

### Property 24: Timeout Batch Abort

*For any* tool call that exceeds the timeout within a batch, the agent should abort the tool call and skip all remaining tool calls in that batch.

**Validates: Requirements 6.5**

### Property 25: Timeout Message in History

*For any* tool call timeout, the agent should add a timeout message to the message history.

**Validates: Requirements 6.6**

### Property 26: Connection Retry Logic

*For any* MCP session failure on a subsequent turn (not first turn), the system should retry connection exactly once.

**Validates: Requirements 6.7**

### Property 27: Graceful Degradation After Retry Failure

*For any* retry connection that fails, the system should disable MCP for the remainder of the episode and set the disabled flag.

**Validates: Requirements 6.8, 6.9**

### Property 28: Tool Call Logging

*For any* tool call, the system should log the tool name, arguments, and iteration number.

**Validates: Requirements 7.1**

### Property 29: Tool Result Logging

*For any* tool call completion, the system should log the result type, length, and duration.

**Validates: Requirements 7.2**

### Property 30: Session Summary Logging

*For any* tool-calling session completion, the system should log the total iterations and tools used.

**Validates: Requirements 7.3**

### Property 31: Langfuse Tool Call Spans

*For any* tool call when Langfuse is enabled, a separate Langfuse span should be created.

**Validates: Requirements 7.4**

### Property 32: Langfuse Session Span

*For any* tool-calling session when Langfuse is enabled, a session-level Langfuse span should be created.

**Validates: Requirements 7.5**

### Property 33: Schema Translation to OpenAI Format

*For any* MCP tool schema, the MCPManager should translate it to valid OpenAI format.

**Validates: Requirements 8.1**

### Property 34: Tool Name Format

*For any* translated tool name, it should follow the format {server_name}.{tool_name}.

**Validates: Requirements 8.2**

### Property 35: Tool Name Parsing for Routing

*For any* tool call execution, the MCPManager should correctly parse the tool name to extract the server name.

**Validates: Requirements 8.3**

### Property 36: InputSchema to Parameters Mapping

*For any* MCP tool schema translation, the inputSchema field should be mapped to the parameters field in OpenAI format.

**Validates: Requirements 8.4**

### Property 37: Complete Tool Translation

*For any* MCP server with multiple tools, all tools should be translated and made available.

**Validates: Requirements 8.5**

### Property 38: OpenAI-Compatible Tool Definitions

*For any* tool schema request, the MCPManager should return a list of valid OpenAI-compatible tool definitions.

**Validates: Requirements 8.6**

### Property 39: Single Asyncio.run Boundary

*For any* agent action generation with MCP enabled, asyncio.run should be called exactly once.

**Validates: Requirements 10.1**

### Property 40: No Nested Event Loops

*For any* async operation completion, control should return to synchronous code without creating nested event loops.

**Validates: Requirements 10.5**

### Property 41: ToolCallResult Success Format

*For any* successful tool call, the result should be wrapped in a ToolCallResult with content and is_error=False.

**Validates: Requirements 11.1**

### Property 42: ToolCallResult Error Format

*For any* failed tool call, the result should be wrapped in a ToolCallResult with is_error=True and error_message.

**Validates: Requirements 11.2, 11.4**

### Property 43: ToolCallResult Serialization

*For any* ToolCallResult, calling to_dict() should produce a valid dictionary that can be JSON-serialized.

**Validates: Requirements 11.3**

### Property 44: ToolCallResult JSON in History

*For any* ToolCallResult added to message history, it should be JSON-serialized.

**Validates: Requirements 11.5**

### Property 45: Configuration Defaults

*For any* GameConfiguration loaded without explicit MCP values, the system should apply the correct default values (max_tool_iterations=20, tool_call_timeout_seconds=30, server_startup_timeout_seconds=10, config_file="mcp_config.json", force_tool_support=False).

**Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5**

### Property 46: AgentResponse Parsing

*For any* final LLM response, the agent should parse it using the existing AgentResponse Pydantic model.

**Validates: Requirements 15.1**

### Property 47: AgentResponse Field Presence

*For any* successfully parsed AgentResponse, it should contain thinking, action, and optional new_objective fields.

**Validates: Requirements 15.2**

### Property 48: AgentResponse Backward Compatibility

*For any* agent with MCP disabled, the agent should continue using AgentResponse without any changes.

**Validates: Requirements 15.5**


## Error Handling

### Error Handling Strategy

The MCP integration follows a **fail-fast for configuration, graceful for runtime** approach:

- **Configuration errors** (missing files, invalid JSON, incompatible models) crash early with clear error messages
- **Runtime errors** (tool call failures, timeouts, session failures) degrade gracefully with logging and fallbacks

### Failure Scenarios

| Scenario | Detection | Strategy | Recovery |
|----------|-----------|----------|----------|
| **MCP server fails to start (first turn)** | Exception during `connect_session()` on turn 1 | **Fail fast**: Raise `MCPServerStartupError`, crash episode | None - requires fixing configuration |
| **Tool call fails** | Exception during `call_tool()` | **Skip**: Log warning, add error to message history, continue | LLM decides how to proceed with error info |
| **Tool call timeout** | `asyncio.TimeoutError` after N seconds | **End batch**: Add timeout message, skip remaining tools in batch | LLM decides how to proceed with partial results |
| **Model doesn't support tools** | Check model name against known patterns | **Fail fast**: Raise `MCPError` with clear message | Use compatible model or disable MCP |
| **Max iterations reached** | No content after loop | **Force action**: Append message, call LLM with response_format | Guaranteed valid action via JSON schema |
| **Session fails on subsequent turn** | Exception on `connect_session()` after turn 1 | **Retry once**: Try connect again, disable MCP if fails | Continue without tools for rest of episode |
| **Server dies mid-turn** | Exception during tool call | **End loop**: Treat as tool error, force final action | Guaranteed valid action via JSON schema |
| **Invalid tool arguments** | JSON parse error or validation error | **Skip tool**: Log error, add error to messages | LLM decides how to proceed |
| **MCP config file missing** | `FileNotFoundError` when loading config | **Fail fast**: Raise `MCPError` if MCP enabled | Create config file or disable MCP |
| **MCP config invalid JSON** | JSON parse error | **Fail fast**: Raise `MCPError` with parse details | Fix JSON syntax in config file |

### Error Messages

Error messages should be clear and actionable:

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

### First Turn vs Subsequent Turn Failures

The system distinguishes between first turn and subsequent turn failures:

**First Turn Failure** (turn 1):
- MCP server fails to start during initial connection
- **Strategy**: Fail fast - raise `MCPServerStartupError` and crash episode
- **Rationale**: Configuration is likely wrong, better to fail early than waste compute

**Subsequent Turn Failure** (turn 2+):
- MCP session fails to connect after working previously
- **Strategy**: Retry once, then graceful degradation
- **Rationale**: Transient network/process issues, worth retrying before giving up

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

## Testing Strategy

### Testing Approach

The MCP integration will be tested using both **unit tests** and **property-based tests**:

- **Unit tests** verify specific examples, edge cases, and error conditions
- **Property-based tests** verify universal properties that should hold across all inputs
- Together they provide comprehensive coverage: unit tests catch concrete bugs, property tests verify general correctness

### Unit Testing

Unit tests will cover:

**LLM Client Extensions:**
- LLMResponse can represent tool calls
- LLMClient accepts tools parameter
- Tool call parsing from responses
- Model compatibility checking with various model names

**MCP Manager:**
- Configuration loading from JSON
- Schema translation from MCP to OpenAI format
- Tool name prefixing with server name
- Environment variable merging
- Session lifecycle (connect/disconnect)
- Graceful degradation on failures

**Agent Extensions:**
- Tool-calling loop with mock MCP manager
- Max iterations handling
- Forced final action call
- Message history management
- Async/sync boundary

**Orchestrator Integration:**
- MCPManager initialization
- Passing MCPManager to agent
- Error handling on initialization failure

### Property-Based Testing

Property-based tests will use **Hypothesis** (Python's PBT library) to verify correctness properties across many randomly generated inputs.

**Configuration:**
- Each property-based test will run a minimum of 100 iterations
- Tests will use smart generators that constrain to valid input spaces
- Tests will be tagged with comments referencing design document properties

**Key Properties to Test:**

1. **Property 2: Final Output is Always a Game Command**
   - Generate random tool-calling scenarios
   - Verify final output is always a valid game command string

2. **Property 3: Tool Results Appear in Message History**
   - Generate random tool calls
   - Verify results appear in history before next LLM call

3. **Property 12: Response Content Exclusivity**
   - Generate random LLM responses
   - Verify tool_calls and content are mutually exclusive

4. **Property 17: Sequential Tool Execution**
   - Generate responses with multiple tool_calls
   - Verify execution is sequential, not parallel

5. **Property 34: Tool Name Format**
   - Generate random server and tool names
   - Verify translated names follow {server_name}.{tool_name} format

6. **Property 43: ToolCallResult Serialization**
   - Generate random ToolCallResults (success and error)
   - Verify to_dict() produces valid JSON-serializable dictionaries

7. **Property 45: Configuration Defaults**
   - Generate configurations with missing values
   - Verify correct defaults are applied

### Test Organization

```
tests/
├── test_mcp_manager.py           # Unit tests for MCPManager
├── test_llm_client_tools.py      # Unit tests for LLMClient extensions
├── test_agent_tool_calling.py    # Unit tests for agent tool-calling loop
├── test_orchestrator_mcp.py      # Unit tests for orchestrator integration
├── test_mcp_properties.py        # Property-based tests
└── test_mcp_integration.py       # End-to-end integration tests
```

### Mock Implementations

**MockMCPManager:**
```python
class MockMCPManager:
    """Mock MCP manager for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.tool_results = {}
        self.connected = False
        
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
        
    async def connect_session(self):
        self.connected = True
        
    async def disconnect_session(self):
        self.connected = False
```

### Integration Testing

End-to-end tests will verify the complete flow with a real MCP server:

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
    assert orchestrator.game_state.last_action is not None
```

### Test Coverage Goals

- **Unit test coverage**: >90% of new code
- **Property test coverage**: All critical correctness properties
- **Integration test coverage**: Happy path + major error scenarios
- **Edge case coverage**: Timeouts, failures, max iterations, graceful degradation


## Configuration Examples

### pyproject.toml Configuration

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

### mcp_config.json Configuration

**Note:** V1 supports a single MCP server only. The config format uses `mcpServers` dict for compatibility with Claude Desktop/Cline, but only the first server entry is used. Multi-server support may be added in a future version.

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

## Dependencies

### Python Packages

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "mcp>=1.22.0",  # Model Context Protocol SDK
]
```

Installation:
```bash
uv add "mcp>=1.22.0"
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
- [DEBUG] Turn 47 Iteration 1: Tool arguments: {"thought": "I should explore...", ...}
- [INFO] Turn 47 Iteration 1: Tool result received (0.3s)
- [INFO] Turn 47 Iteration 2: Agent calling tool thoughtbox.think
- [DEBUG] Turn 47 Iteration 2: Tool arguments: {"thought": "Based on the puzzle...", ...}
- [INFO] Turn 47 Iteration 2: Tool result received (0.2s)
- [INFO] Turn 47 Iteration 3: Agent calling tool thoughtbox.think
- [DEBUG] Turn 47 Iteration 3: Tool arguments: {"thought": "Should use key...", ...}
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
- Remaining tools in batch skipped
- Agent forced to provide final action

Logs:
- [WARNING] Turn 23 Iteration 1: Tool call timeout: thoughtbox.think
- [INFO] Turn 23: Skipping remaining tools in batch after timeout
- [INFO] Turn 23: Forcing final action after tool timeout
- [INFO] Turn 23: Agent action: look
```

### Example 3: Graceful Degradation

**Scenario**: MCP session fails on subsequent turn

```
Turn 15: MCP session works normally
- 2 tool calls executed successfully
- Action: "examine mailbox"

Turn 16: MCP session fails to connect
- First connection attempt fails
- Retry connection attempt
- Retry succeeds
- Turn continues normally with MCP

Turn 17: MCP session fails again
- First connection attempt fails
- Retry connection attempt
- Retry fails
- MCP disabled for rest of episode
- Agent continues without tools

Logs:
- [WARNING] Turn 16: MCP session connect failed, retrying: Connection refused
- [INFO] Turn 16: MCP session retry successful
- [WARNING] Turn 17: MCP session connect failed, retrying: Connection refused
- [WARNING] Turn 17: MCP retry failed, disabling MCP for episode
- [INFO] Turn 17: Continuing without MCP tools
```

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

### Tool Name Parsing

```python
def _parse_tool_name(self, tool_name: str) -> Tuple[str, str]:
    """Parse prefixed tool name to extract server and tool names.
    
    Args:
        tool_name: Prefixed tool name (e.g., "thoughtbox.think")
        
    Returns:
        Tuple of (server_name, tool_name)
        
    Raises:
        ValueError: If tool name format is invalid
    """
    parts = tool_name.split(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid tool name format: {tool_name}. "
            f"Expected format: {{server_name}}.{{tool_name}}"
        )
    return parts[0], parts[1]
```

## Migration and Deployment

### Enabling MCP

**Step 1: Install dependencies**

```bash
uv add "mcp>=1.22.0"
```

**Step 2: Add configuration to pyproject.toml**

```toml
[tool.zorkgpt.mcp]
enabled = true
config_file = "mcp_config.json"
max_tool_iterations = 20
tool_call_timeout_seconds = 30
```

**Step 3: Create mcp_config.json**

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

**Step 4: Verify model compatibility**

Ensure agent model supports tool calling:
- ✅ GPT-4, GPT-3.5-turbo
- ✅ Claude 3 (Opus, Sonnet, Haiku)
- ✅ Gemini, Mistral, Llama 3.1+
- ❌ o1/o3, DeepSeek R1, QwQ (reasoning models don't support tools)

**Step 5: Run episode**

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

