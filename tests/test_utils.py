"""
Shared test utilities for ZorkGPT tests using the game server.
"""

from datetime import datetime
from typing import List, Dict, Any
import pytest

from game_server_client import GameServerClient


def setup_test_session(base_url: str = "http://localhost:8000") -> GameServerClient:
    """Create a new game server session for testing.
    
    Args:
        base_url: The game server URL
        
    Returns:
        GameServerClient instance with a new test session started
    """
    client = GameServerClient(base_url=base_url)
    session_id = f"test_{datetime.now().isoformat()}"
    client.start(session_id)
    return client


def run_test_commands(client: GameServerClient, commands: List[str]) -> List[Dict[str, Any]]:
    """Run a sequence of commands and return responses.
    
    Args:
        client: The game server client
        commands: List of commands to execute
        
    Returns:
        List of response dictionaries
    """
    responses = []
    for cmd in commands:
        response = client.send_command(cmd)
        responses.append(response)
    return responses


@pytest.fixture
def game_client():
    """Pytest fixture that provides a game server client with automatic cleanup."""
    client = setup_test_session()
    yield client
    client.close()


def skip_if_server_unavailable():
    """Skip test if game server is not available."""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=1)
        response.raise_for_status()
    except Exception:
        pytest.skip("Game server not available")