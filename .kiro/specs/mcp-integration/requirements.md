# Requirements Document

## Introduction

This document defines the requirements for integrating Model Context Protocol (MCP) client support into ZorkGPT. The integration will enable the ZorkGPT agent to use external tools during puzzle solving through an LLM-driven tool calling approach. The agent will autonomously decide when to use structured reasoning tools, maintaining ZorkGPT's core principle that "all reasoning comes from LLMs."

## Glossary

- **MCP (Model Context Protocol)**: A protocol for connecting LLM applications to external tools and data sources
- **ZorkGPT Agent**: The LLM-powered component that generates game actions based on current state
- **Tool Calling**: The process where an LLM requests to execute external functions during response generation
- **Thoughtbox**: An MCP server that provides structured reasoning capabilities
- **Stdio Transport**: Communication method using standard input/output streams with subprocess
- **Tool Schema**: JSON description of a tool's name, parameters, and behavior
- **LLMClient**: ZorkGPT's wrapper for LLM API interactions
- **MCPManager**: Component responsible for managing MCP server connections and tool execution
- **Orchestrator**: ZorkGPT's main coordination component that manages turn processing
- **Session**: A connection to an MCP server that lasts for one turn
- **Iteration**: One cycle of LLM call and potential tool execution within a turn
- **AgentResponse**: Existing Pydantic model with fields: thinking (agent's justification), action (game command), new_objective (optional multi-step plan)
- **thinking field**: The JSON field in AgentResponse where the agent writes its justification for the action
- **reasoning return key**: The dictionary key in the agent's return value that holds the thinking field value (for backward compatibility)
- **MCP structured reasoning**: External tool-based reasoning (e.g., thoughtbox.think) that happens during tool-calling loops, distinct from the agent's internal thinking field

## Requirements

### Requirement 1

**User Story:** As a ZorkGPT developer, I want the agent to use external reasoning tools during action generation, so that the agent can solve complex puzzles more effectively.

#### Acceptance Criteria

1. WHEN MCP is enabled in configuration THEN the ZorkGPT Agent SHALL have access to tool schemas during action generation
2. WHEN the Agent generates an action THEN the Agent SHALL autonomously decide whether to use available tools based on context
3. WHEN a tool is called THEN the tool result SHALL be incorporated into the Agent's reasoning process
4. WHEN the Agent completes tool calling THEN the Agent SHALL return a valid Zork game command as the final output
5. WHEN MCP is disabled in configuration THEN the Agent SHALL function normally without tool calling capabilities

### Requirement 2

**User Story:** As a ZorkGPT developer, I want to configure MCP servers through standard configuration files, so that I can easily enable or disable tool integrations.

#### Acceptance Criteria

1. WHEN the system starts THEN the MCPManager SHALL load server configuration from mcp_config.json
2. WHEN mcp_config.json is missing and MCP is enabled THEN the system SHALL raise a configuration error
3. WHEN mcp_config.json contains invalid JSON THEN the system SHALL raise a configuration error with details
4. WHEN pyproject.toml contains MCP settings THEN the GameConfiguration SHALL load those settings correctly
5. WHEN MCP is disabled in pyproject.toml THEN the system SHALL not attempt to load MCP server configuration
6. WHEN multiple servers are configured in mcp_config.json THEN the MCPManager SHALL use only the first server entry as a V1 limitation

### Requirement 3

**User Story:** As a ZorkGPT developer, I want MCP sessions to connect and disconnect cleanly per turn, so that each turn starts with fresh state.

#### Acceptance Criteria

1. WHEN a turn begins and MCP is enabled THEN the MCPManager SHALL connect to the configured MCP server
2. WHEN the MCP server subprocess is spawned THEN the MCPManager SHALL merge configured environment variables with the system environment
3. WHEN an MCP session is established THEN the MCPManager SHALL call session.initialize to complete the MCP protocol handshake
4. WHEN the MCP protocol handshake completes THEN the MCPManager SHALL discover available tool schemas
5. WHEN a turn ends THEN the MCPManager SHALL disconnect from the MCP server
6. WHEN the MCP server subprocess is spawned THEN the subprocess SHALL be coupled to the session lifecycle
7. WHEN the session disconnects THEN the subprocess SHALL be terminated

### Requirement 4

**User Story:** As a ZorkGPT developer, I want the LLMClient to support tool calling, so that the agent can request tool execution during LLM interactions.

#### Acceptance Criteria

1. WHEN the LLMClient makes a request with tools parameter THEN the request SHALL include tool schemas in OpenAI format
2. WHEN the LLM response contains tool_calls THEN the LLMClient SHALL parse and return tool call information
3. WHEN the LLM response contains tool_calls THEN the response content SHALL be None
4. WHEN the LLM response contains content THEN the response SHALL not contain tool_calls
5. WHEN a model does not support tool calling THEN the LLMClient SHALL detect this and raise an error
6. WHEN the Agent is in a tool-calling loop THEN the LLMClient SHALL NOT use response_format until the forced final action call for OpenRouter compatibility
7. WHEN the LLMClient sends messages with system or user roles THEN the LLMClient SHALL include cache_control metadata for prompt caching

### Requirement 5

**User Story:** As a ZorkGPT developer, I want the agent to execute tool-calling loops, so that complex reasoning can span multiple tool invocations.

#### Acceptance Criteria

1. WHEN the Agent receives tool_calls from the LLM THEN the Agent SHALL execute each tool call via MCPManager
2. WHEN the LLM returns multiple tool_calls in one response THEN the Agent SHALL execute them sequentially to avoid MCP session concurrency issues
3. WHEN the LLM returns tool call arguments THEN the Agent SHALL parse the JSON string arguments before execution
4. WHEN a tool call completes THEN the Agent SHALL append the tool result to the message history
5. WHEN tool results are added to history THEN the Agent SHALL call the LLM again with updated context
6. WHEN the LLM returns content instead of tool_calls THEN the Agent SHALL exit the tool-calling loop
7. WHEN the iteration count reaches the maximum THEN the Agent SHALL append a user message requesting final action
8. WHEN forcing a final action THEN the Agent SHALL set tools parameter to None and tool_choice parameter to None
9. WHEN forcing a final action THEN the Agent SHALL apply response_format with the AgentResponse JSON schema for guaranteed valid output
10. WHEN the LLM returns neither content nor tool_calls THEN the Agent SHALL log a warning and exit the loop

### Requirement 6

**User Story:** As a ZorkGPT developer, I want robust error handling for MCP operations, so that failures are handled gracefully or fail fast as appropriate.

#### Acceptance Criteria

1. WHEN the MCP server fails to start on turn 1 THEN the system SHALL raise MCPServerStartupError and terminate the episode without retry
2. WHEN a tool call fails during execution THEN the Agent SHALL log the error and add an error message to history
3. WHEN the LLM returns multiple tool_calls in one response THEN the Agent SHALL treat them as a batch for error handling purposes
4. WHEN a tool call fails with a non-timeout error within a batch THEN the Agent SHALL continue processing remaining tool calls in that batch
5. WHEN a tool call exceeds the timeout within a batch THEN the Agent SHALL abort the tool call and skip all remaining tool calls in that batch
6. WHEN a tool call timeout occurs THEN the Agent SHALL add a timeout message to history and let the LLM decide how to proceed
7. WHEN the MCP session fails to connect on turn 2 or later THEN the system SHALL retry connection exactly once
8. IF the retry connection succeeds THEN the system SHALL continue with MCP enabled for that turn
9. IF the retry connection fails THEN the system SHALL disable MCP for the remainder of the episode and continue without tools
10. WHEN MCP is disabled due to repeated failures THEN the system SHALL set an internal disabled flag to prevent further connection attempts

### Requirement 7

**User Story:** As a ZorkGPT developer, I want comprehensive observability for MCP operations, so that I can debug issues and analyze tool usage patterns.

#### Acceptance Criteria

1. WHEN a tool is called THEN the system SHALL log the tool name, arguments, and iteration number
2. WHEN a tool call completes THEN the system SHALL log the result type, length, and duration
3. WHEN a tool-calling session completes THEN the system SHALL log the total iterations and tools used
4. WHEN Langfuse is enabled THEN each tool call SHALL be tracked as a separate Langfuse span
5. WHEN Langfuse is enabled THEN the entire tool-calling session SHALL be tracked as a Langfuse span

### Requirement 8

**User Story:** As a ZorkGPT developer, I want tool schemas to be translated from MCP format to OpenAI format, so that they work with the unified LLM API.

#### Acceptance Criteria

1. WHEN tool schemas are discovered from an MCP server THEN the MCPManager SHALL translate them to OpenAI format
2. WHEN a tool name is translated THEN the tool name SHALL follow the format {server_name}.{tool_name}
3. WHEN a tool call is executed THEN the MCPManager SHALL parse the tool name to extract the server name and route the call
4. WHEN the inputSchema is translated THEN the schema SHALL be mapped to the parameters field
5. WHEN multiple tools exist on a server THEN all tools SHALL be translated and made available
6. WHEN tool schemas are requested THEN the MCPManager SHALL return a list of OpenAI-compatible tool definitions

### Requirement 9

**User Story:** As a ZorkGPT developer, I want the orchestrator to initialize and manage the MCPManager, so that MCP is properly integrated into the turn lifecycle.

#### Acceptance Criteria

1. WHEN the orchestrator initializes and MCP is enabled THEN the orchestrator SHALL create an MCPManager instance
2. WHEN the orchestrator creates an agent THEN the orchestrator SHALL pass the MCPManager to the agent
3. WHEN MCPManager initialization fails THEN the orchestrator SHALL raise an MCPError
4. WHEN MCP is disabled THEN the orchestrator SHALL not create an MCPManager
5. WHEN the agent processes a turn THEN the agent SHALL handle MCP session lifecycle internally

### Requirement 10

**User Story:** As a ZorkGPT developer, I want async/sync boundaries to be minimal, so that the codebase remains maintainable.

#### Acceptance Criteria

1. WHEN the agent generates an action with MCP enabled THEN the agent SHALL use a single asyncio.run boundary
2. WHEN the orchestrator processes a turn THEN the orchestrator SHALL remain synchronous
3. WHEN MCP operations execute THEN all async operations SHALL be contained within the agent's async method
4. WHEN the agent's public API is called THEN the API SHALL remain synchronous for backward compatibility
5. WHEN async operations complete THEN control SHALL return to synchronous code without nested event loops

### Requirement 11

**User Story:** As a ZorkGPT developer, I want tool call results to have a consistent format, so that error handling and success cases are uniform.

#### Acceptance Criteria

1. WHEN a tool call succeeds THEN the result SHALL be wrapped in a ToolCallResult with content
2. WHEN a tool call fails THEN the result SHALL be wrapped in a ToolCallResult with is_error=True
3. WHEN a ToolCallResult is serialized THEN the result SHALL be converted to a dictionary format
4. WHEN an error ToolCallResult is created THEN the result SHALL include an error_message field
5. WHEN a ToolCallResult is added to message history THEN the result SHALL be JSON-serialized

### Requirement 12

**User Story:** As a ZorkGPT developer, I want model compatibility checking, so that incompatible models are detected early.

#### Acceptance Criteria

1. WHEN a model is checked for tool support THEN the system SHALL use pattern matching on the model name
2. WHEN a reasoning model is detected THEN the system SHALL return False for tool support
3. WHEN a known tool-supporting model is detected THEN the system SHALL return True for tool support
4. WHEN the force_tool_support config is enabled THEN the system SHALL bypass auto-detection
5. WHEN an incompatible model is used with tools THEN the system SHALL raise an MCPError before the first tool call

### Requirement 13

**User Story:** As a ZorkGPT developer, I want sensible default configuration values, so that the system works correctly without extensive configuration.

#### Acceptance Criteria

1. WHEN MCP configuration is loaded without explicit max_tool_iterations THEN the system SHALL default to 20 iterations
2. WHEN MCP configuration is loaded without explicit tool_call_timeout_seconds THEN the system SHALL default to 30 seconds
3. WHEN MCP configuration is loaded without explicit server_startup_timeout_seconds THEN the system SHALL default to 10 seconds
4. WHEN MCP configuration is loaded without explicit config_file path THEN the system SHALL default to mcp_config.json
5. WHEN MCP configuration is loaded without explicit force_tool_support THEN the system SHALL default to False

### Requirement 14

**User Story:** As a ZorkGPT developer, I want clear system requirements documented, so that I can ensure the runtime environment is properly configured.

#### Acceptance Criteria

1. WHEN the thoughtbox MCP server is used THEN the system SHALL require Node.js version 22 or higher
2. WHEN MCP is enabled THEN the system SHALL require the mcp Python package version 1.22.0 or higher
3. WHEN MCP is enabled THEN the system SHALL require Python version 3.11 or higher for async support
4. WHEN system requirements are not met THEN the system SHALL provide clear error messages indicating missing dependencies
5. WHEN the MCP server command is not found THEN the system SHALL raise an error with installation instructions

### Requirement 15

**User Story:** As a ZorkGPT developer, I want the agent to use existing data structures for final responses, so that MCP integration maintains compatibility with the existing codebase.

#### Acceptance Criteria

1. WHEN the Agent parses a final LLM response THEN the Agent SHALL use the existing AgentResponse Pydantic model
2. WHEN the AgentResponse is parsed THEN the response SHALL contain thinking, action, and optional new_objective fields
3. WHEN the AgentResponse parsing fails THEN the Agent SHALL fall back to safe defaults with error logging
4. WHEN the Agent returns action data THEN the data SHALL be compatible with existing orchestrator expectations
5. WHEN MCP is disabled THEN the Agent SHALL continue using AgentResponse without any changes
