"""
ZorkGPT Game Interface Layer

This module provides a clean separation between the game interface and the AI orchestration logic.
It contains all components necessary for interacting with the Zork game, including:

- Server: FastAPI-based game server managing dfrotz processes
- Client: REST API client for connecting to the game server
- Core: Core game interface classes and parsers
"""

# Export main interfaces for convenient importing
from .core.zork_interface import ZorkInterface
from .core.structured_parser import StructuredZorkParser, StructuredZorkResponse
from .client.game_server_client import GameServerClient
from .server.models import (
    CommandRequest,
    CommandResponse,
    SessionState,
    HistoryEntry,
    SessionHistory
)
from .server.session_manager import GameSession

__all__ = [
    "ZorkInterface",
    "StructuredZorkParser", 
    "StructuredZorkResponse",
    "GameServerClient",
    "CommandRequest",
    "CommandResponse", 
    "SessionState",
    "HistoryEntry",
    "SessionHistory",
    "GameSession"
]