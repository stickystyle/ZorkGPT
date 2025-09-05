# ABOUTME: Pydantic models for the ZorkGPT Game Server API requests and responses
# ABOUTME: Contains all the data models used by the FastAPI server for type validation

from typing import List, Optional
from pydantic import BaseModel


# Request/Response models
class CommandRequest(BaseModel):
    command: str


class CommandResponse(BaseModel):
    session_id: str
    turn_number: int
    score: Optional[int]
    raw_response: str
    parsed: dict
    game_over: bool
    game_over_reason: Optional[str]


class SessionState(BaseModel):
    session_id: str
    turn_number: int
    last_score: int
    last_save_turn: int
    active: bool
    start_time: str
    last_command_time: str


class HistoryEntry(BaseModel):
    turn_number: int
    command: str
    raw_response: str
    timestamp: str


class SessionHistory(BaseModel):
    session_id: str
    turns: List[HistoryEntry]
