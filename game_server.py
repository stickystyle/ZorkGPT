"""
Game Server for ZorkGPT
Manages dfrotz processes and provides REST API for game interaction.
"""

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

from zork_api import ZorkInterface
from structured_zork_parser import StructuredZorkParser


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response models
class CommandRequest(BaseModel):
    command: str


class CommandResponse(BaseModel):
    session_id: str
    turn_number: int
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


class GameSession:
    """Manages a single game session."""
    
    def __init__(self, session_id: str, working_directory: str):
        self.session_id = session_id
        self.working_directory = working_directory
        self.zork = None
        self.parser = StructuredZorkParser()
        self.turn_number = 0
        self.history: List[HistoryEntry] = []
        self.last_score = 0
        self.last_save_turn = 0
        self.active = False
        self.start_time = datetime.utcnow()
        self.last_command_time = self.start_time
        
        # Ensure working directory exists
        Path(working_directory).mkdir(parents=True, exist_ok=True)
        
    async def start(self) -> str:
        """Start or restore the game session."""
        if self.active:
            return "Session already active"
            
        # Check for existing save file (try both with and without .qzl extension)
        save_filename = f"autosave_{self.session_id}"
        save_path_qzl = os.path.join(self.working_directory, f"{save_filename}.qzl")
        save_path_no_ext = os.path.join(self.working_directory, save_filename)
        
        # Start Zork process
        self.zork = ZorkInterface(working_directory=self.working_directory)
        intro_text = self.zork.start()
        
        # Check for save file (try both extensions)
        save_exists = os.path.exists(save_path_qzl) or os.path.exists(save_path_no_ext)
        actual_save_path = save_path_qzl if os.path.exists(save_path_qzl) else save_path_no_ext
        
        if save_exists:
            # Attempt to restore from save
            logger.info(f"Found save file for session {self.session_id}, attempting restore")
            success = self.zork.trigger_zork_restore(save_filename)
            
            if success:
                # Load session metadata (history, turn count, etc.)
                self._load_session_metadata()
                
                # Get current state after restore
                response = self.zork.send_command("look")
                self.active = True
                logger.info(f"Successfully restored session {self.session_id}")
                return response
            else:
                logger.warning(f"Failed to restore session {self.session_id}, starting new game")
                
        self.active = True
        return intro_text
        
    def execute_command(self, command: str) -> CommandResponse:
        """Execute a command and return the response."""
        if not self.active or not self.zork:
            raise RuntimeError("Session not active")
            
        # Execute command
        raw_response = self.zork.send_command(command)
        self.turn_number += 1
        self.last_command_time = datetime.utcnow()
        
        # Parse response
        parsed = self.parser.parse_response(raw_response)
        
        # Check for game over
        game_over, game_over_reason = self.zork.is_game_over(raw_response)
        
        # Store in history
        history_entry = HistoryEntry(
            turn_number=self.turn_number,
            command=command,
            raw_response=raw_response,
            timestamp=self.last_command_time.isoformat()
        )
        self.history.append(history_entry)
        
        # Check if we need to save
        score_changed = parsed.score is not None and parsed.score != self.last_score
        turns_since_save = self.turn_number - self.last_save_turn
        
        if score_changed or turns_since_save >= 10:
            self._trigger_save()
            if parsed.score is not None:
                self.last_score = parsed.score
                
        # Build response
        return CommandResponse(
            session_id=self.session_id,
            turn_number=self.turn_number,
            raw_response=raw_response,
            parsed={
                "room_name": parsed.room_name,
                "score": parsed.score,
                "moves": parsed.moves,
                "game_text": parsed.game_text,
                "has_structured_header": parsed.has_structured_header
            },
            game_over=game_over,
            game_over_reason=game_over_reason
        )
        
    def _trigger_save(self):
        """Trigger an auto-save."""
        try:
            save_filename = f"autosave_{self.session_id}"
            success = self.zork.trigger_zork_save(save_filename)
            
            if success:
                # Also save the session history as JSON
                self._save_session_metadata()
                self.last_save_turn = self.turn_number
                logger.info(f"Auto-saved session {self.session_id} at turn {self.turn_number}")
            else:
                logger.warning(f"Failed to auto-save session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Error during auto-save for session {self.session_id}: {e}")
            
    def _save_session_metadata(self):
        """Save session metadata including history."""
        try:
            metadata = {
                "session_id": self.session_id,
                "turn_number": self.turn_number,
                "last_score": self.last_score,
                "last_save_turn": self.last_save_turn,
                "start_time": self.start_time.isoformat(),
                "last_command_time": self.last_command_time.isoformat(),
                "history": [h.model_dump() for h in self.history]
            }
            
            metadata_path = os.path.join(self.working_directory, f"autosave_{self.session_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved session metadata for {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")
            
    def _load_session_metadata(self):
        """Load session metadata including history."""
        try:
            metadata_path = os.path.join(self.working_directory, f"autosave_{self.session_id}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Restore session state
                self.turn_number = metadata.get("turn_number", 0)
                self.last_score = metadata.get("last_score", 0)
                self.last_save_turn = metadata.get("last_save_turn", 0)
                
                # Restore history
                history_data = metadata.get("history", [])
                self.history = []
                for h in history_data:
                    entry = HistoryEntry(**h)
                    self.history.append(entry)
                
                logger.info(f"Restored session metadata for {self.session_id} with {len(self.history)} turns")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load session metadata: {e}")
            
        return False
            
    def get_state(self) -> SessionState:
        """Get current session state."""
        return SessionState(
            session_id=self.session_id,
            turn_number=self.turn_number,
            last_score=self.last_score,
            last_save_turn=self.last_save_turn,
            active=self.active,
            start_time=self.start_time.isoformat(),
            last_command_time=self.last_command_time.isoformat()
        )
        
    def get_history(self) -> SessionHistory:
        """Get full session history."""
        return SessionHistory(
            session_id=self.session_id,
            turns=self.history
        )
        
    def close(self):
        """Close the session."""
        if self.zork:
            # Force save before closing (regardless of turn count)
            self._force_save()
            self.zork.close()
            self.zork = None
        self.active = False
        logger.info(f"Closed session {self.session_id}")
        
    def _force_save(self):
        """Force a save regardless of turn count or score."""
        try:
            save_filename = f"autosave_{self.session_id}"
            success = self.zork.trigger_zork_save(save_filename)
            
            if success:
                # Also save the session history as JSON
                self._save_session_metadata()
                self.last_save_turn = self.turn_number
                logger.info(f"Force-saved session {self.session_id} at turn {self.turn_number}")
            else:
                logger.warning(f"Failed to force-save session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Error during force-save for session {self.session_id}: {e}")


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
    return {"message": f"Save triggered for session {session_id}", "turn": str(session.turn_number)}


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