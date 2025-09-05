# ABOUTME: FastAPI-based game server for ZorkGPT managing dfrotz processes and REST API
# ABOUTME: Main server application providing HTTP endpoints for game session management

import os
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from .models import CommandRequest, CommandResponse, SessionState, SessionHistory
from .session_manager import GameSession


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GameServer:
    """Main game server managing multiple sessions."""

    def __init__(self, working_directory: str = "./game_files"):
        self.working_directory = working_directory
        self.sessions: Dict[str, GameSession] = {}

    async def create_or_restore_session(self, session_id: str) -> str:
        """Create a new session or restore existing one."""
        if session_id in self.sessions and self.sessions[session_id].active:
            # Session already active
            return "Session already active"

        # Create new session
        session = GameSession(session_id, self.working_directory)
        intro_text = await session.start()
        self.sessions[session_id] = session

        logger.info(f"Created/restored session: {session_id}")
        return intro_text

    def get_session(self, session_id: str) -> GameSession:
        """Get a session by ID."""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = self.sessions[session_id]
        if not session.active:
            raise HTTPException(status_code=400, detail="Session not active")

        return session

    def close_session(self, session_id: str):
        """Close a session."""
        if session_id in self.sessions:
            self.sessions[session_id].close()
            del self.sessions[session_id]


# Create FastAPI app
app = FastAPI(title="ZorkGPT Game Server")

# Create game server instance
game_server = GameServer()


@app.post("/sessions/{session_id}")
async def create_session(session_id: str) -> Dict[str, str]:
    """Create or restore a game session."""
    intro_text = await game_server.create_or_restore_session(session_id)
    return {"session_id": session_id, "intro_text": intro_text}


@app.post("/sessions/{session_id}/command")
async def send_command(session_id: str, request: CommandRequest) -> CommandResponse:
    """Send a command to the game."""
    session = game_server.get_session(session_id)
    return session.execute_command(request.command)


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str) -> SessionHistory:
    """Get full session history."""
    session = game_server.get_session(session_id)
    return session.get_history()


@app.get("/sessions/{session_id}/state")
async def get_state(session_id: str) -> SessionState:
    """Get current session state."""
    session = game_server.get_session(session_id)
    return session.get_state()


@app.post("/sessions/{session_id}/save")
async def force_save(session_id: str) -> Dict[str, str]:
    """Force an immediate save for the session."""
    session = game_server.get_session(session_id)
    session._force_save()
    return {
        "message": f"Save triggered for session {session_id}",
        "turn": str(session.turn_number),
    }


@app.delete("/sessions/{session_id}")
async def close_session(session_id: str) -> Dict[str, str]:
    """Close a session."""
    game_server.close_session(session_id)
    return {"message": f"Session {session_id} closed"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_sessions": len(game_server.sessions)}


if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
