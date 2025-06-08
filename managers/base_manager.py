"""
Base manager protocol and interface for ZorkGPT orchestration.

This module defines the common interface that all managers implement,
enabling clean composition and coordination in the orchestrator.
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Any, Dict
import logging

from session.game_state import GameState
from session.game_configuration import GameConfiguration


@runtime_checkable
class ManagerProtocol(Protocol):
    """
    Protocol defining the interface that all managers must implement.
    
    This enables the orchestrator to treat all managers uniformly
    while maintaining type safety and clear contracts.
    """
    
    def reset_episode(self) -> None:
        """Reset manager state for a new episode."""
        ...
    
    def process_turn(self) -> None:
        """Process manager-specific logic for the current turn."""
        ...
    
    def should_process_turn(self) -> bool:
        """Check if this manager needs to process the current turn."""
        ...


class BaseManager(ABC):
    """
    Abstract base class providing common functionality for all managers.
    
    Handles common dependencies (logger, config, game_state) and provides
    a foundation for manager-specific implementations.
    """
    
    def __init__(
        self, 
        logger: logging.Logger, 
        config: GameConfiguration, 
        game_state: GameState,
        component_name: str
    ):
        """
        Initialize base manager with common dependencies.
        
        Args:
            logger: Shared logger instance for structured logging
            config: Game configuration object
            game_state: Shared game state object
            component_name: Name for logging component field (e.g., "objective_manager")
        """
        self.logger = logger
        self.config = config
        self.game_state = game_state
        self.component_name = component_name
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log an info message with structured fields."""
        if self.logger:
            self.logger.info(message, extra={
                "event_type": "info",
                "component": self.component_name,
                "turn": self.game_state.turn_count,
                **kwargs
            })
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log a debug message with structured fields."""
        if self.logger:
            self.logger.debug(message, extra={
                "event_type": "debug",
                "component": self.component_name,
                "turn": self.game_state.turn_count,
                **kwargs
            })
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log a warning message with structured fields."""
        if self.logger:
            self.logger.warning(message, extra={
                "event_type": "warning",
                "component": self.component_name,
                "turn": self.game_state.turn_count,
                **kwargs
            })
    
    def log_error(self, message: str, **kwargs) -> None:
        """Log an error message with structured fields."""
        if self.logger:
            self.logger.error(message, extra={
                "event_type": "error",
                "component": self.component_name,
                "turn": self.game_state.turn_count,
                **kwargs
            })
    
    def log_progress(self, message: str, stage: str, **kwargs) -> None:
        """Log a progress message with structured fields."""
        if self.logger:
            self.logger.info(message, extra={
                "event_type": "progress",
                "component": self.component_name,
                "stage": stage,
                "turn": self.game_state.turn_count,
                **kwargs
            })
    
    @abstractmethod
    def reset_episode(self) -> None:
        """
        Reset manager state for a new episode.
        
        Called when starting a new episode to clean up any
        episode-specific state in the manager.
        """
        pass
    
    @abstractmethod
    def process_turn(self) -> None:
        """
        Process manager-specific logic for the current turn.
        
        This is the main entry point called by the orchestrator
        to trigger manager-specific processing.
        """
        pass
    
    def should_process_turn(self) -> bool:
        """
        Check if this manager needs to process the current turn.
        
        Default implementation always returns True, but managers
        can override this to implement turn-based intervals or
        other processing logic.
        
        Returns:
            True if process_turn() should be called this turn
        """
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current manager status for debugging and monitoring.
        
        Returns:
            Dictionary with manager status information
        """
        return {
            "component": self.component_name,
            "turn": self.game_state.turn_count,
            "episode_id": self.game_state.episode_id,
        }