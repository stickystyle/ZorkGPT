"""
GameState dataclass for ZorkGPT orchestration.

This module defines the central shared state object that all managers access.
Maintains clear separation between data (owned by GameState) and logic (owned by managers).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from datetime import datetime
from collections import Counter


@dataclass
class GameState:
    """
    Central shared state for a Zork gameplay episode.

    This class owns all mutable state that managers need to access and modify.
    Managers access this state directly but focus on their specific logic.
    """

    # Core game state
    episode_id: str = ""
    turn_count: int = 0
    current_room_id: int = 0  # Z-machine location ID (primary key for map operations)
    current_room_name: str = ""  # Display name only
    current_room_name_for_map: str = ""  # DEPRECATED: Use current_room_id instead
    current_inventory: List[str] = field(default_factory=list)
    previous_zork_score: int = 0
    game_over_flag: bool = False

    # Room description tracking (for agent context)
    last_room_description: str = ""
    last_room_description_turn: int = 0
    last_room_description_location_id: Optional[int] = None

    # Navigation and movement state
    prev_room_for_prompt_context: Optional[str] = None
    action_leading_to_current_room_for_prompt_context: Optional[str] = None
    visited_locations: Set[str] = field(default_factory=set)

    # Action and memory tracking
    action_counts: Counter = field(default_factory=Counter)
    action_history: List[Tuple[str, str]] = field(
        default_factory=list
    )  # (action, response)
    action_reasoning_history: List[Dict[str, Any]] = field(default_factory=list)  # Each entry: {"turn": int, "reasoning": str, "action": str, "timestamp": str}
    memory_log_history: List[Dict[str, Any]] = field(default_factory=list)
    failed_actions_by_location: Dict[str, List[str]] = field(default_factory=dict)

    # Critic evaluation tracking (for viewer compatibility)
    critic_evaluation_history: List[Dict[str, Any]] = field(default_factory=list)
    extracted_info_history: List[Dict[str, Any]] = field(default_factory=list)

    # Rejection tracking
    rejected_actions_per_turn: Dict[int, List[Dict[str, Any]]] = field(
        default_factory=dict
    )
    rejection_state: Optional[Dict[str, Any]] = (
        None  # RejectionManager state for persistence
    )

    # Objective management state
    discovered_objectives: List[str] = field(default_factory=list)
    completed_objectives: List[Dict[str, Any]] = field(default_factory=list)
    objective_update_turn: int = 0
    objective_staleness_tracker: Dict[str, int] = field(default_factory=dict)
    last_location_for_staleness: Optional[str] = None
    last_score_for_staleness: int = 0

    # Knowledge and learning state
    last_knowledge_update_turn: int = 0
    last_map_update_turn: int = 0

    # Session-persistent state (survives episode resets)
    death_count: int = 0  # Cumulative deaths across all episodes

    def reset_episode(self, episode_id: str = None) -> None:
        """
        Reset all episode-specific state while preserving session-persistent state.

        Args:
            episode_id: Episode ID to use (if None, generates one - for backward compatibility)

        This is called at the start of each new episode to clean the slate
        while maintaining things like death_count that persist across episodes.
        """
        # Core game state - use provided episode ID or generate one as fallback
        if episode_id:
            self.episode_id = episode_id
        else:
            # Fallback for backward compatibility
            self.episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.turn_count = 0
        self.current_room_id = 0
        self.current_room_name = ""
        self.current_room_name_for_map = ""
        self.current_inventory.clear()
        self.previous_zork_score = 0
        self.game_over_flag = False

        # Navigation and movement state
        self.prev_room_for_prompt_context = None
        self.action_leading_to_current_room_for_prompt_context = None
        self.visited_locations.clear()

        # Action and memory tracking
        self.action_counts.clear()
        self.action_history.clear()
        self.action_reasoning_history.clear()
        self.memory_log_history.clear()
        self.failed_actions_by_location.clear()
        self.critic_evaluation_history.clear()
        self.extracted_info_history.clear()

        # Rejection tracking
        self.rejected_actions_per_turn.clear()
        self.rejection_state = None

        # Objective management state
        self.discovered_objectives.clear()
        self.completed_objectives.clear()
        self.objective_update_turn = 0
        self.objective_staleness_tracker.clear()
        self.last_location_for_staleness = None
        self.last_score_for_staleness = 0

        # Knowledge and learning state
        self.last_knowledge_update_turn = 0
        self.last_map_update_turn = 0

        # Room description tracking
        self.last_room_description = ""
        self.last_room_description_turn = 0
        self.last_room_description_location_id = None

        # Note: death_count is NOT reset - it persists across episodes

    def get_export_data(self) -> Dict[str, Any]:
        """
        Get a dictionary representation for state export.

        Returns:
            Dictionary containing all state data suitable for JSON serialization
        """
        return {
            "metadata": {
                "episode_id": self.episode_id,
                "turn_count": self.turn_count,
                "export_timestamp": datetime.now().isoformat(),
                "game_over": self.game_over_flag,
            },
            "game_state": {
                "current_room_id": self.current_room_id,
                "current_room": self.current_room_name,  # Keep key name for backward compatibility
                "current_inventory": self.current_inventory,
                "current_score": self.previous_zork_score,
                "visited_locations": list(self.visited_locations),
                "death_count": self.death_count,
            },
            "navigation": {
                "prev_room": self.prev_room_for_prompt_context,
                "action_to_current_room": self.action_leading_to_current_room_for_prompt_context,
                "failed_actions_by_location": dict(self.failed_actions_by_location),
            },
            "objectives": {
                "discovered": self.discovered_objectives,
                "completed": self.completed_objectives,
                "staleness_tracker": dict(self.objective_staleness_tracker),
                "last_objective_update_turn": self.objective_update_turn,
            },
            "learning": {
                "last_knowledge_update_turn": self.last_knowledge_update_turn,
                "last_map_update_turn": self.last_map_update_turn,
            },
            "performance": {
                "action_counts": dict(self.action_counts),
                "total_actions": len(self.action_history),
                "total_memory_entries": len(self.memory_log_history),
            },
        }
