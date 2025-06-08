"""
REST API client for the ZorkGPT Game Server.
Provides a ZorkInterface-compatible API for the orchestrator.
"""

import requests
import time
from typing import List, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class GameServerClient:
    """Client for interacting with the ZorkGPT Game Server REST API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """Initialize the game server client.
        
        Args:
            base_url: Base URL of the game server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session_id = None
        self.turn_number = 0
        self._last_response = None
        
    def __enter__(self):
        """Support for context manager protocol."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting the context manager."""
        self.close()
        
    def start(self, session_id: Optional[str] = None) -> str:
        """Start or restore a game session.
        
        Args:
            session_id: Session ID (ISO8601 timestamp). If None, will be set later.
            
        Returns:
            Initial game text
        """
        # If session is already started and no new session_id provided, return cached intro
        if self.session_id and session_id is None and hasattr(self, '_intro_text'):
            return self._intro_text
            
        if session_id:
            self.session_id = session_id
        elif not self.session_id:
            # Generate a default session ID if none provided
            from datetime import datetime
            self.session_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        
        try:
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Started/restored session: {self.session_id}")
            
            # Rebuild state from history if this is a restore
            self._rebuild_state()
            
            intro_text = data.get("intro_text", "")
            self._intro_text = intro_text  # Cache for subsequent calls
            return intro_text
            
        except requests.RequestException as e:
            logger.error(f"Failed to start session: {e}")
            raise RuntimeError(f"Failed to start session: {e}")
            
    def _rebuild_state(self):
        """Rebuild internal state from session history."""
        try:
            response = requests.get(
                f"{self.base_url}/sessions/{self.session_id}/history",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            history = response.json()
            turns = history.get("turns", [])
            
            if turns:
                self.turn_number = len(turns)
                logger.info(f"Rebuilt state from {self.turn_number} turns")
                
        except requests.RequestException as e:
            logger.warning(f"Failed to rebuild state from history: {e}")
            
    def send_command(self, command: str) -> str:
        """Send a command to the game.
        
        Args:
            command: Command to send
            
        Returns:
            Raw game response
        """
        if not self.session_id:
            raise RuntimeError("No active session")
            
        try:
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}/command",
                json={"command": command},
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            self._last_response = data
            self.turn_number = data.get("turn_number", self.turn_number + 1)
            
            return data.get("raw_response", "")
            
        except requests.RequestException as e:
            logger.error(f"Failed to send command: {e}")
            raise RuntimeError(f"Failed to send command: {e}")
            
    def get_history(self) -> List[Tuple[str, str]]:
        """Get the full command/response history.
        
        Returns:
            List of (command, response) tuples
        """
        if not self.session_id:
            return []
            
        try:
            response = requests.get(
                f"{self.base_url}/sessions/{self.session_id}/history",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            history = response.json()
            turns = history.get("turns", [])
            
            return [(turn["command"], turn["raw_response"]) for turn in turns]
            
        except requests.RequestException as e:
            logger.error(f"Failed to get history: {e}")
            return []
            
    def is_running(self) -> bool:
        """Check if the session is active.
        
        Returns:
            True if session is active
        """
        if not self.session_id:
            return False
            
        try:
            response = requests.get(
                f"{self.base_url}/sessions/{self.session_id}/state",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            state = response.json()
            return state.get("active", False)
            
        except requests.RequestException:
            return False
            
    def close(self):
        """Close the session."""
        if self.session_id:
            try:
                response = requests.delete(
                    f"{self.base_url}/sessions/{self.session_id}",
                    timeout=self.timeout
                )
                response.raise_for_status()
                logger.info(f"Closed session: {self.session_id}")
                
            except requests.RequestException as e:
                logger.warning(f"Failed to close session: {e}")
                
            self.session_id = None
            
    def score(self, score_text=None) -> Tuple[int, int]:
        """Extract score from game text or last response.
        
        Args:
            score_text: Optional text to parse (uses last response if None)
            
        Returns:
            Tuple of (current_score, max_score)
        """
        if score_text is None and self._last_response:
            parsed = self._last_response.get("parsed", {})
            score = parsed.get("score", 0)
            return score, 350  # Zork I max score
            
        # Fallback to sending score command
        if score_text is None:
            score_text = self.send_command("score")
            
        # Parse score from text (compatibility with existing code)
        import re
        match = re.search(r"Your score is (\d+).*?of (\d+)", score_text)
        if match:
            return int(match.group(1)), int(match.group(2))
            
        return 0, 350
        
    def is_game_over(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if the game is over.
        
        Args:
            text: Game text to check
            
        Returns:
            Tuple of (is_over, reason)
        """
        if self._last_response:
            game_over = self._last_response.get("game_over", False)
            reason = self._last_response.get("game_over_reason")
            if game_over:
                return True, reason
                
        # Fallback to text parsing (compatibility)
        game_ending_phrases = [
            "you have died",
            "you are dead",
            "game over",
            "****  you have won  ****",
        ]
        
        text_lower = text.lower()
        for phrase in game_ending_phrases:
            if phrase in text_lower:
                return True, f"Game over - {phrase}"
                
        return False, None
        
    def inventory(self) -> List[str]:
        """Get the current inventory.
        
        Returns:
            List of inventory items
        """
        inv_text = self.send_command("inventory")
        return self._parse_inventory(inv_text)
        
    def inventory_with_response(self) -> Tuple[List[str], str]:
        """Get inventory with raw response.
        
        Returns:
            Tuple of (items, raw_response)
        """
        inv_text = self.send_command("inventory")
        return self._parse_inventory(inv_text), inv_text
        
    def _parse_inventory(self, inv_text: str) -> List[str]:
        """Parse inventory text (compatibility method)."""
        # Basic parsing - the orchestrator has more sophisticated parsing
        if "empty-handed" in inv_text.lower():
            return []
            
        items = []
        lines = inv_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('>') and line != "You are carrying:":
                if line.endswith('.'):
                    line = line[:-1]
                items.append(line)
                
        return items
        
    # Compatibility methods for save/restore (no-ops since server handles this)
    def trigger_zork_save(self, filename: str) -> bool:
        """Trigger a manual save via the game server."""
        if not self.session_id:
            logger.warning("No active session for manual save")
            return False
            
        try:
            logger.info(f"Manual save triggered for session {self.session_id} (filename: {filename})")
            # Use the force_save method which calls the dedicated endpoint
            return self.force_save()
            
        except Exception as e:
            logger.error(f"Failed to trigger manual save: {e}")
            return False
        
    def trigger_zork_restore(self, filename: str) -> bool:
        """Compatibility method - restores are handled by server."""
        logger.info(f"Restore request for {filename} - handled by server")
        return True
        
    def force_save(self) -> bool:
        """Force an immediate save via the REST API."""
        if not self.session_id:
            logger.warning("No active session for force save")
            return False
            
        try:
            # Use the dedicated save endpoint
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}/save",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Force save successful for session {self.session_id} at turn {data.get('turn')}")
                return True
            else:
                logger.warning(f"Force save failed with status {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Failed to force save: {e}")
            return False