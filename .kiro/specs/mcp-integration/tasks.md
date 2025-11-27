# Implementation Plan

- [x] 1. Set up dependencies and configuration infrastructure
  - Install mcp Python package (version >=1.22.0)
  - Add MCP configuration fields to GameConfiguration dataclass
  - Create MCPConfig and MCPServerConfig Pydantic models for validation
  - Implement configuration loading from pyproject.toml and mcp_config.json
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 13.1, 13.2, 13.3, 13.4, 13.5, 14.2, 14.3_

- [x] 1.1 Write unit tests for configuration loading
  - Test loading MCP settings from pyproject.toml
  - Test loading server config from mcp_config.json
  - Test error handling for missing config file
  - Test error handling for invalid JSON
  - Test V1 single-server limitation
  - Test default values
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 13.1, 13.2, 13.3, 13.4, 13.5_

- [x] 1.2 Write property test for configuration defaults
  - **Property 45: Configuration Defaults**
  - **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5**

- [x] 2. Extend LLMClient with tool calling support
  - Create ToolCall, FunctionCall, and ToolCallResult dataclasses
  - Extend LLMResponse dataclass with tool_calls and finish_reason fields
  - Add tools and tool_choice parameters to chat_completions_create method
  - Implement tool call parsing in _execute_request method
  - Implement _supports_tool_calling method with model pattern matching
  - Add cache_control metadata to system and user messages
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7, 11.1, 11.2, 11.3, 11.4, 11.5, 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 2.1 Write unit tests for LLMClient extensions
  - Test LLMResponse with tool_calls
  - Test tool call parsing from API responses
  - Test tools parameter in requests
  - Test model compatibility checking
  - Test cache_control metadata
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7, 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 2.2 Write property test for response content exclusivity
  - **Property 12: Response Content Exclusivity**
  - **Validates: Requirements 4.3, 4.4**

- [x] 2.3 Write property test for model compatibility detection
  - **Property 13: Model Compatibility Detection**
  - **Validates: Requirements 4.5, 12.1, 12.2, 12.3**

- [x] 2.4 Write property test for ToolCallResult serialization
  - **Property 43: ToolCallResult Serialization**
  - **Validates: Requirements 11.3**

- [x] 3. Implement MCPManager core functionality
  - Create MCPManager class with initialization
  - Implement _load_mcp_config method to load from mcp_config.json
  - Implement connect_session method with stdio transport
  - Implement session.initialize() call for MCP protocol handshake
  - Implement disconnect_session method
  - Implement environment variable merging for subprocess
  - Add is_disabled property for graceful degradation tracking
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.5, 3.6, 3.7, 6.9_

- [x] 3.1 Write unit tests for MCPManager initialization
  - Test configuration loading
  - Test error handling for missing config
  - Test error handling for invalid JSON
  - Test environment variable merging
  - _Requirements: 2.1, 2.2, 2.3, 3.2_

- [x] 3.2 Write property test for session lifecycle
  - **Property 6: Session Lifecycle Coupling**
  - **Validates: Requirements 3.1, 3.5**

- [x] 3.3 Write property test for subprocess termination
  - **Property 9: Subprocess Termination on Disconnect**
  - **Validates: Requirements 3.7**

- [x] 4. Implement tool schema discovery and translation
  - Implement get_tool_schemas method to discover tools from MCP server
  - Implement _discover_tools_from_server method
  - Implement _translate_schema method to convert MCP schemas to OpenAI format
  - Implement tool name prefixing with server name (format: {server_name}.{tool_name})
  - Implement _parse_tool_name method to extract server name from prefixed tool name
  - _Requirements: 3.4, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 4.1 Write unit tests for schema translation
  - Test MCP to OpenAI schema conversion
  - Test tool name prefixing
  - Test tool name parsing
  - Test multiple tools translation
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [x] 4.2 Write property test for tool name format
  - **Property 34: Tool Name Format**
  - **Validates: Requirements 8.2**

- [x] 4.3 Write property test for complete tool translation
  - **Property 37: Complete Tool Translation**
  - **Validates: Requirements 8.5**

- [x] 5. Implement tool execution with error handling
  - Implement call_tool method with timeout handling
  - Implement ToolCallResult wrapping for success and error cases
  - Add timeout handling with asyncio.TimeoutError
  - Add error logging for tool call failures
  - Integrate Langfuse tracking for tool calls (if enabled)
  - _Requirements: 5.1, 6.2, 6.4, 6.5, 6.6, 7.1, 7.2, 7.4, 11.1, 11.2, 11.4_

- [x] 5.1 Write unit tests for tool execution
  - Test successful tool calls
  - Test tool call failures
  - Test timeout handling
  - Test ToolCallResult wrapping
  - Test error logging
  - _Requirements: 5.1, 6.2, 6.4, 6.5, 6.6, 11.1, 11.2, 11.4_

- [x] 5.2 Write property test for tool call logging
  - **Property 28: Tool Call Logging**
  - **Validates: Requirements 7.1**

- [x] 5.3 Write property test for tool result logging
  - **Property 29: Tool Result Logging**
  - **Validates: Requirements 7.2**

- [x] 6. Implement graceful degradation and retry logic
  - Implement retry logic in connect_session for subsequent turn failures
  - Add _retry_attempted flag to track retry attempts
  - Implement graceful degradation by setting _disabled flag after failed retry
  - Add logging for retry attempts and degradation
  - Distinguish between first-turn failures (fail-fast) and subsequent-turn failures (retry)
  - _Requirements: 6.1, 6.7, 6.8, 6.9_

- [x] 6.1 Write unit tests for graceful degradation
  - Test first-turn failure (fail-fast, no retry)
  - Test subsequent-turn failure with successful retry
  - Test subsequent-turn failure with failed retry (graceful degradation)
  - Test disabled flag behavior
  - _Requirements: 6.1, 6.7, 6.8, 6.9_

- [x] 6.2 Write property test for connection retry logic
  - **Property 26: Connection Retry Logic**
  - **Validates: Requirements 6.7**

- [x] 6.3 Write property test for graceful degradation
  - **Property 27: Graceful Degradation After Retry Failure**
  - **Validates: Requirements 6.8, 6.9**

- [ ] 7. Checkpoint - Ensure MCPManager tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement agent async action generation method
  - Create _generate_action_async method in ZorkAgent
  - Implement message history building with cache_control
  - Implement MCP session connection at turn start
  - Implement tool schema retrieval
  - Add model compatibility check before tool calling
  - _Requirements: 1.1, 3.1, 3.3, 3.4, 4.5, 4.7, 10.1, 10.3, 12.5_

- [x] 8.1 Write unit tests for async action generation setup
  - Test message history building
  - Test MCP session connection
  - Test tool schema retrieval
  - Test model compatibility check
  - _Requirements: 1.1, 3.1, 3.3, 3.4, 4.5, 12.5_

- [ ] 8.2 Write property test for tool schema availability
  - **Property 1: Tool Schema Availability**
  - **Validates: Requirements 1.1**

- [x] 9. Implement tool-calling loop
  - Implement iteration loop with max_tool_iterations limit
  - Implement LLM call with tools parameter (response_format disabled during loop)
  - Implement tool_calls detection and execution
  - Implement sequential tool execution (not parallel)
  - Implement JSON argument parsing
  - Implement tool result appending to message history
  - Implement loop continuation after tool results
  - Implement loop exit on content
  - Implement loop exit on neither content nor tool_calls
  - _Requirements: 1.2, 1.3, 4.6, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.10_

- [x] 9.1 Write unit tests for tool-calling loop
  - Test loop with successful tool calls
  - Test loop exit on content
  - Test loop exit on max iterations
  - Test loop exit on unexpected state
  - Test sequential execution
  - Test argument parsing
  - _Requirements: 1.2, 1.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.10_

- [ ] 9.2 Write property test for tool results in history
  - **Property 3: Tool Results Appear in Message History**
  - **Validates: Requirements 1.3, 5.4**

- [ ] 9.3 Write property test for sequential tool execution
  - **Property 17: Sequential Tool Execution**
  - **Validates: Requirements 5.2**

- [ ] 9.4 Write property test for loop continuation
  - **Property 19: Loop Continuation After Tool Results**
  - **Validates: Requirements 5.5**

- [ ] 9.5 Write property test for loop exit on content
  - **Property 20: Loop Exit on Content**
  - **Validates: Requirements 5.6**

- [x] 10. Implement batch error handling in tool-calling loop
  - Implement batch concept for multiple tool_calls in one response
  - Implement non-timeout error handling (continue with remaining tools)
  - Implement timeout error handling (abort batch, skip remaining tools)
  - Implement error message appending to history
  - Implement timeout message appending to history
  - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 10.1 Write unit tests for batch error handling
  - Test non-timeout error recovery
  - Test timeout batch abort
  - Test error messages in history
  - Test timeout messages in history
  - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 10.2 Write property test for batch error handling
  - **Property 22: Batch Error Handling**
  - **Validates: Requirements 6.3**

- [x] 10.3 Write property test for non-timeout error recovery
  - **Property 23: Non-Timeout Error Recovery**
  - **Validates: Requirements 6.4**

- [x] 10.4 Write property test for timeout batch abort
  - **Property 24: Timeout Batch Abort**
  - **Validates: Requirements 6.5**

- [ ] 11. Implement forced final action handling
  - Implement max iterations check
  - Implement user message appending ("provide final action")
  - Implement tools=None and tool_choice=None on forced call
  - Implement response_format with AgentResponse JSON schema on forced call
  - Implement AgentResponse parsing
  - Implement fallback to safe defaults on parsing failure
  - _Requirements: 1.4, 4.6, 5.7, 5.8, 5.9, 15.1, 15.2, 15.3, 15.4_

- [ ] 11.1 Write unit tests for forced final action
  - Test max iterations handling
  - Test forced call parameters
  - Test AgentResponse parsing
  - Test parsing failure fallback
  - _Requirements: 5.7, 5.8, 5.9, 15.1, 15.2, 15.3_

- [ ] 11.2 Write property test for final output is always a game command
  - **Property 2: Final Output is Always a Game Command**
  - **Validates: Requirements 1.4**

- [ ] 11.3 Write property test for AgentResponse parsing
  - **Property 46: AgentResponse Parsing**
  - **Validates: Requirements 15.1**

- [ ] 12. Implement MCP session cleanup and Langfuse tracking
  - Implement finally block for session disconnection
  - Implement Langfuse session span tracking (if enabled)
  - Implement session summary logging
  - _Requirements: 3.5, 7.3, 7.5_

- [ ] 12.1 Write unit tests for session cleanup
  - Test session disconnection in finally block
  - Test session summary logging
  - Test Langfuse session span
  - _Requirements: 3.5, 7.3, 7.5_

- [ ] 12.2 Write property test for session summary logging
  - **Property 30: Session Summary Logging**
  - **Validates: Requirements 7.3**

- [ ] 13. Implement sync wrapper to maintain existing agent API
  - Modify get_action_with_reasoning to always use asyncio.run to call _generate_action_async
  - The async implementation handles both MCP enabled and disabled cases internally
  - When MCP is disabled, _generate_action_async skips MCP session connection and tool calling
  - Ensure public API remains synchronous (orchestrator doesn't need changes)
  - This provides a single unified code path regardless of MCP configuration
  - _Requirements: 1.5, 10.1, 10.2, 10.4, 10.5, 15.5_

- [ ] 13.1 Write unit tests for sync wrapper
  - Test MCP enabled path (connects session, uses tools)
  - Test MCP disabled path (skips session connection, no tools)
  - Test asyncio.run boundary (only one call)
  - Test no nested event loops
  - Test both paths use same async implementation
  - _Requirements: 1.5, 10.1, 10.2, 10.4, 10.5, 15.5_

- [ ] 13.2 Write property test for single asyncio.run boundary
  - **Property 39: Single Asyncio.run Boundary**
  - **Validates: Requirements 10.1**

- [ ] 13.3 Write property test for MCP disabled behavior
  - **Property 48: AgentResponse Backward Compatibility**
  - **Validates: Requirements 15.5**
  - Ensures agent with MCP disabled works identically to pre-MCP agent

- [ ] 14. Checkpoint - Ensure agent tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - **Note**: Agent MCP functionality is complete, but orchestrator integration (Task 15) is needed for full system testing

- [ ] 15. Integrate MCPManager into orchestrator
  - Add MCPManager initialization in _initialize_managers
  - Add error handling for MCPManager initialization failures
  - Pass MCPManager to ZorkAgent during initialization
  - Add conditional initialization based on mcp_enabled config
  - Ensure orchestrator remains synchronous
  - **Note**: After this task, the full system is integrated and ready for end-to-end testing with real MCP servers
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.2_

- [ ] 15.1 Write unit tests for orchestrator integration
  - Test MCPManager initialization when enabled
  - Test no MCPManager when disabled
  - Test MCPManager passed to agent
  - Test error handling on initialization failure
  - Test orchestrator remains synchronous
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.2_

- [ ] 16. Create example configuration files
  - Create example mcp_config.json with thoughtbox server
  - Add MCP configuration section to pyproject.toml
  - Document configuration options
  - _Requirements: 2.1, 2.4, 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 17. Add comprehensive logging and observability
  - Add JSON logging for all MCP events
  - Add human-readable episode log formatting
  - Ensure all Langfuse spans are properly created
  - Add error logging with actionable messages
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 17.1 Write unit tests for logging
  - Test JSON log structure
  - Test episode log formatting
  - Test Langfuse span creation
  - Test error message formatting
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 18. Write end-to-end integration tests
  - Test full turn with mock MCP server
  - Test full turn with real thoughtbox server (marked as slow)
  - Test MCP disabled scenario
  - Test first-turn failure scenario
  - Test subsequent-turn failure with graceful degradation
  - Test timeout handling
  - Test max iterations handling
  - **Note**: This is the comprehensive end-to-end testing phase with full orchestrator integration
  - _Requirements: All requirements_

- [ ] 19. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Documentation and migration guide
  - Update README with MCP integration information
  - Create migration guide for enabling MCP
  - Document system requirements (Node.js, Python, mcp package)
  - Document model compatibility
  - Add troubleshooting section
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
