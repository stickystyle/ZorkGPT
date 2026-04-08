# ABOUTME: Unit tests for MCPManager tool schema discovery and translation.
# ABOUTME: Tests MCP to OpenAI schema translation and tool name handling.

import json
import logging
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from session.game_configuration import GameConfiguration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def mock_mcp_tool():
    """Create a mock MCP tool object with typical structure."""
    tool = MagicMock()
    tool.name = "think"
    tool.description = "Structured reasoning step"
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "Your thinking"},
            "thoughtNumber": {"type": "integer", "minimum": 1},
        },
        "required": ["thought", "thoughtNumber"],
    }
    return tool


@pytest.fixture
def mock_mcp_tool_complex():
    """Create a mock MCP tool with complex nested schema."""
    tool = MagicMock()
    tool.name = "search"
    tool.description = "Search with filters"
    tool.inputSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "filters": {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string", "format": "date"},
                    "date_to": {"type": "string", "format": "date"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "default": "asc",
            },
        },
        "required": ["query"],
    }
    return tool


@pytest.fixture
def mock_list_tools_result(mock_mcp_tool):
    """Mock result from session.list_tools()."""
    result = MagicMock()
    result.tools = [mock_mcp_tool]
    return result


@pytest.fixture
def mock_list_tools_result_multiple(mock_mcp_tool, mock_mcp_tool_complex):
    """Mock result from session.list_tools() with multiple tools."""
    result = MagicMock()
    result.tools = [mock_mcp_tool, mock_mcp_tool_complex]
    return result


@pytest.fixture
def create_mcp_config_file(tmp_path):
    """Helper fixture to create a mcp_config.json file."""

    def _create(config_data: dict) -> str:
        config_file = tmp_path / "mcp_config.json"
        config_file.write_text(json.dumps(config_data))
        return str(config_file)

    return _create


@pytest.fixture
def valid_mcp_config_data():
    """Valid MCP configuration data for sequential-thinking server."""
    return {
        "mcpServers": {
            "sequential-thinking": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            }
        }
    }


@pytest.fixture
def game_config_with_mcp(tmp_path, create_mcp_config_file, valid_mcp_config_data):
    """Create a GameConfiguration with MCP enabled and valid config file."""
    config_file = create_mcp_config_file(valid_mcp_config_data)
    return GameConfiguration(
        max_turns_per_episode=100,
        game_file_path="test.z5",
        mcp_enabled=True,
        mcp_config_file=config_file,
    )


@pytest.fixture
def mock_stdio_client():
    """Mock the stdio_client context manager from MCP SDK."""
    mock_read = AsyncMock()
    mock_write = AsyncMock()
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
    mock_context.__aexit__ = AsyncMock(return_value=None)

    with patch("managers.mcp_manager.stdio_client", return_value=mock_context):
        yield mock_context


@pytest.fixture
def mock_client_session():
    """Mock the ClientSession class from MCP SDK."""
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()

    with patch("managers.mcp_manager.ClientSession", return_value=mock_session):
        yield mock_session


# =============================================================================
# Task 4.1: Unit Tests for Schema Translation
# =============================================================================


class TestMCPManagerSchemaTranslation:
    """Test MCP tool schema discovery and translation (Requirements 3.4, 8.1-8.6)."""

    def test_translate_single_tool_schema(
        self, mock_logger, game_config_with_mcp, mock_mcp_tool
    ):
        """Test single MCP tool translates to OpenAI format (Req 8.1)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Call translation method
        translated = manager._translate_tool_schema(
            tool=mock_mcp_tool, server_name="sequential-thinking"
        )

        # Verify OpenAI function schema structure
        assert translated["type"] == "function"
        assert "function" in translated

        function_def = translated["function"]
        assert function_def["name"] == "sequential-thinking.think"
        assert function_def["description"] == "Structured reasoning step"
        assert "parameters" in function_def

        # Verify parameters mapping (inputSchema → parameters)
        parameters = function_def["parameters"]
        assert parameters["type"] == "object"
        assert "thought" in parameters["properties"]
        assert "thoughtNumber" in parameters["properties"]
        assert parameters["required"] == ["thought", "thoughtNumber"]

    def test_tool_name_prefixing(
        self, mock_logger, game_config_with_mcp, mock_mcp_tool
    ):
        """Test tool name gets server prefix: {server}.{tool} (Req 8.2)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Translate with explicit server name
        translated = manager._translate_tool_schema(
            tool=mock_mcp_tool, server_name="sequential-thinking"
        )

        # Verify prefixing format
        assert translated["function"]["name"] == "sequential-thinking.think"
        assert "." in translated["function"]["name"]
        parts = translated["function"]["name"].split(".")
        assert len(parts) == 2
        assert parts[0] == "sequential-thinking"
        assert parts[1] == "think"

    def test_tool_name_parsing(self, mock_logger, game_config_with_mcp):
        """Test parsing tool name from {server}.{tool} format (Req 8.3)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Parse valid tool name
        server_name, tool_name = manager._parse_tool_name("sequential-thinking.think")

        assert server_name == "sequential-thinking"
        assert tool_name == "think"

    def test_tool_name_parsing_invalid_format(
        self, mock_logger, game_config_with_mcp
    ):
        """Test parsing invalid tool name raises ValueError (Req 8.3)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Invalid format: no dot separator
        with pytest.raises(ValueError) as exc_info:
            manager._parse_tool_name("invalid_no_dot")

        assert "invalid tool name format" in str(exc_info.value).lower()

    def test_tool_name_parsing_multiple_dots(
        self, mock_logger, game_config_with_mcp
    ):
        """Test parsing tool name with multiple dots raises ValueError."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Invalid format: too many dots
        with pytest.raises(ValueError) as exc_info:
            manager._parse_tool_name("server.namespace.tool")

        assert "invalid tool name format" in str(exc_info.value).lower()

    def test_translate_multiple_tools(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_mcp_tool,
        mock_mcp_tool_complex,
    ):
        """Test multiple tools all get translated (Req 8.5)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Translate both tools
        translated1 = manager._translate_tool_schema(
            tool=mock_mcp_tool, server_name="sequential-thinking"
        )
        translated2 = manager._translate_tool_schema(
            tool=mock_mcp_tool_complex, server_name="sequential-thinking"
        )

        # Verify both translations
        assert translated1["function"]["name"] == "sequential-thinking.think"
        assert translated2["function"]["name"] == "sequential-thinking.search"

        # Verify both have proper structure
        assert translated1["type"] == "function"
        assert translated2["type"] == "function"
        assert "parameters" in translated1["function"]
        assert "parameters" in translated2["function"]

    def test_inputSchema_to_parameters_mapping(
        self, mock_logger, game_config_with_mcp, mock_mcp_tool
    ):
        """Test inputSchema correctly maps to parameters field (Req 8.4)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        translated = manager._translate_tool_schema(
            tool=mock_mcp_tool, server_name="sequential-thinking"
        )

        # Verify parameters is exact copy of inputSchema
        parameters = translated["function"]["parameters"]
        input_schema = mock_mcp_tool.inputSchema

        assert parameters == input_schema
        assert parameters["type"] == input_schema["type"]
        assert parameters["properties"] == input_schema["properties"]
        assert parameters["required"] == input_schema["required"]

    def test_inputSchema_to_parameters_preserves_nested_structure(
        self, mock_logger, game_config_with_mcp, mock_mcp_tool_complex
    ):
        """Test nested inputSchema structure is preserved in parameters."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        translated = manager._translate_tool_schema(
            tool=mock_mcp_tool_complex, server_name="sequential-thinking"
        )

        parameters = translated["function"]["parameters"]

        # Verify nested structure preserved
        assert "filters" in parameters["properties"]
        filters = parameters["properties"]["filters"]
        assert filters["type"] == "object"
        assert "date_from" in filters["properties"]
        assert "date_to" in filters["properties"]
        assert "max_results" in filters["properties"]

        # Verify enum preserved
        assert "sort_order" in parameters["properties"]
        sort_order = parameters["properties"]["sort_order"]
        assert sort_order["enum"] == ["asc", "desc"]
        assert sort_order["default"] == "asc"

    @pytest.mark.asyncio
    async def test_get_tool_schemas_without_session_raises(
        self, mock_logger, game_config_with_mcp
    ):
        """Test get_tool_schemas raises RuntimeError when session is None (Req 8.6)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Don't connect session
        assert manager._session is None

        with pytest.raises(RuntimeError) as exc_info:
            await manager.get_tool_schemas()

        assert "session not connected" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_tool_schemas_returns_openai_format(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
        mock_list_tools_result,
    ):
        """Test get_tool_schemas returns list of OpenAI-compatible schemas (Req 3.4, 8.6)."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Connect session
        await manager.connect_session()

        # Mock session.list_tools()
        mock_client_session.list_tools = AsyncMock(return_value=mock_list_tools_result)

        # Get tool schemas
        schemas = await manager.get_tool_schemas()

        # Verify return type and structure
        assert isinstance(schemas, list)
        assert len(schemas) == 1

        # Verify OpenAI format
        schema = schemas[0]
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "sequential-thinking.think"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

        # Verify list_tools was called
        mock_client_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tool_schemas_handles_multiple_tools(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
        mock_list_tools_result_multiple,
    ):
        """Test get_tool_schemas handles multiple tools correctly."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Connect session
        await manager.connect_session()

        # Mock session.list_tools() with multiple tools
        mock_client_session.list_tools = AsyncMock(
            return_value=mock_list_tools_result_multiple
        )

        # Get tool schemas
        schemas = await manager.get_tool_schemas()

        # Verify all tools translated
        assert len(schemas) == 2
        assert schemas[0]["function"]["name"] == "sequential-thinking.think"
        assert schemas[1]["function"]["name"] == "sequential-thinking.search"

        # Verify all are OpenAI format
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    @pytest.mark.asyncio
    async def test_get_tool_schemas_logs_discovery(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
        mock_list_tools_result,
    ):
        """Test get_tool_schemas logs successful tool discovery."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Connect session
        await manager.connect_session()

        # Mock session.list_tools()
        mock_client_session.list_tools = AsyncMock(return_value=mock_list_tools_result)

        # Get tool schemas
        await manager.get_tool_schemas()

        # Verify logging
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("tool" in call.lower() and "discovered" in call.lower() for call in log_calls)

    @pytest.mark.asyncio
    async def test_get_tool_schemas_empty_list(
        self,
        mock_logger,
        game_config_with_mcp,
        mock_stdio_client,
        mock_client_session,
    ):
        """Test get_tool_schemas handles empty tool list gracefully."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Connect session
        await manager.connect_session()

        # Mock empty tool list
        empty_result = MagicMock()
        empty_result.tools = []
        mock_client_session.list_tools = AsyncMock(return_value=empty_result)

        # Get tool schemas
        schemas = await manager.get_tool_schemas()

        # Should return empty list
        assert schemas == []
        assert isinstance(schemas, list)

    def test_translate_tool_schema_preserves_description(
        self, mock_logger, game_config_with_mcp, mock_mcp_tool
    ):
        """Test tool description is preserved in translation."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        translated = manager._translate_tool_schema(
            tool=mock_mcp_tool, server_name="sequential-thinking"
        )

        # Verify description preserved exactly
        assert translated["function"]["description"] == mock_mcp_tool.description
        assert translated["function"]["description"] == "Structured reasoning step"

    def test_translate_tool_schema_handles_missing_required(
        self, mock_logger, game_config_with_mcp
    ):
        """Test translation handles inputSchema without required field."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Create tool without required field
        tool = MagicMock()
        tool.name = "optional_tool"
        tool.description = "Tool with all optional parameters"
        tool.inputSchema = {
            "type": "object",
            "properties": {
                "optional_param": {"type": "string"},
            },
            # No required field
        }

        translated = manager._translate_tool_schema(
            tool=tool, server_name="sequential-thinking"
        )

        # Verify translation succeeds
        assert translated["function"]["name"] == "sequential-thinking.optional_tool"
        parameters = translated["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "optional_param" in parameters["properties"]

        # Required field should be preserved as-is (empty or missing)
        if "required" in tool.inputSchema:
            assert "required" in parameters
        # If not in inputSchema, it's fine if not in parameters

    @pytest.mark.asyncio
    async def test_get_tool_schemas_when_disabled(
        self, mock_logger, game_config_with_mcp
    ):
        """Test get_tool_schemas returns empty list when manager is disabled."""
        from managers.mcp_manager import MCPManager

        manager = MCPManager(
            config=game_config_with_mcp,
            logger=mock_logger,
        )

        # Manually disable
        manager._disabled = True

        # Should return empty list without requiring session
        schemas = await manager.get_tool_schemas()

        assert schemas == []
        assert isinstance(schemas, list)
