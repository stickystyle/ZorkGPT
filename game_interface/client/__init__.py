"""
Game Client Package

Contains the REST API client for connecting to the ZorkGPT game server.
"""

from .game_server_client import GameServerClient

__all__ = ["GameServerClient"]