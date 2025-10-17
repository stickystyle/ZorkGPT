"""
ZorkGPT Game Interface Layer

This module provides the Jericho-based game interface for interacting with Zork.
After Phase 2 migration, this module only contains the JerichoInterface.

The game server, client, and structured parser have been removed in favor of
direct Jericho integration with object tree access.
"""

# Export main interface for convenient importing
from .core.jericho_interface import JerichoInterface

__all__ = ["JerichoInterface"]
