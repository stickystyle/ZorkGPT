"""
MapManager for ZorkGPT orchestration.

Handles all map-related responsibilities:
- Map building and updating from movement patterns
- Room tracking and location changes
- Map quality assessment and metrics
- Navigation analysis and movement tracking
- Integration with MapGraph and MovementAnalyzer
"""

from typing import Dict, Any, Optional

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from map_graph import MapGraph
from movement_analyzer import MovementAnalyzer, MovementContext


class MapManager(BaseManager):
    """
    Manages all map-related functionality for ZorkGPT.

    Responsibilities:
    - Map building and room tracking
    - Movement analysis and navigation
    - Map quality assessment and metrics
    - Integration with MapGraph and MovementAnalyzer
    """

    def __init__(self, logger, config: GameConfiguration, game_state: GameState):
        super().__init__(logger, config, game_state, "map_manager")

        # Initialize map components
        self.game_map = MapGraph(logger=logger)
        self.movement_analyzer = MovementAnalyzer()

        # Map update tracking
        self.last_map_update_turn = 0

    def reset_episode(self) -> None:
        """Reset map manager state for a new episode."""
        # Note: We typically don't reset the map itself as it persists across episodes
        # Only reset episode-specific tracking
        self.last_map_update_turn = 0
        self.log_debug("Map manager reset for new episode")

    def process_turn(self) -> None:
        """Process map management for the current turn."""
        # Check for periodic map updates
        if self.should_process_turn():
            self.check_map_update()

    def should_process_turn(self) -> bool:
        """Check if map needs processing this turn."""
        # Check if it's time for a map update
        turns_since_update = self.game_state.turn_count - self.last_map_update_turn
        return (
            self.game_state.turn_count > 0
            and turns_since_update >= self.config.map_update_interval
        )

    def add_initial_room(self, room_id: int, room_name: str) -> None:
        """Add the initial room to the map.

        Args:
            room_id: Jericho location ID (Z-machine object number)
            room_name: Human-readable room name
        """
        if room_name and room_id is not None:
            self.game_map.add_room(room_id, room_name)
            self.game_state.current_room_id = room_id
            self.game_state.current_room_name = room_name
            self.game_state.current_room_name_for_map = room_name  # DEPRECATED

            self.log_debug(f"Added initial room to map: {room_name} (ID: {room_id})")

            self.logger.info(
                f"Initial room added to map: {room_name}",
                extra={
                    "event_type": "map_initial_room_added",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "room_id": room_id,
                    "room_name": room_name,
                },
            )

    def update_from_movement(
        self,
        action_taken: str,
        new_room_id: int,
        new_room_name: str,
        previous_room_id: Optional[int] = None,
        previous_room_name: Optional[str] = None,
        game_response: str = "",
    ) -> None:
        """Update map based on movement action and result.

        Args:
            action_taken: The movement action taken
            new_room_id: Jericho location ID of destination room
            new_room_name: Human-readable name of destination room
            previous_room_id: Jericho location ID of origin room (optional)
            previous_room_name: Human-readable name of origin room (optional)
            game_response: Game's response to the action
        """
        try:
            if not new_room_name or new_room_id is None:
                return

            # Use provided previous room or get from game state
            prev_room_id = previous_room_id if previous_room_id is not None else self.game_state.current_room_id
            prev_room = previous_room_name or self.game_state.current_room_name_for_map

            # Add the new room to the map
            self.game_map.add_room(new_room_id, new_room_name)

            # Update movement tracking if we moved from a previous room
            if prev_room_id is not None and prev_room_id != new_room_id:
                self._update_movement_tracking(
                    action_taken, prev_room_id, prev_room, new_room_id, new_room_name, game_response
                )

            # Update current room tracking
            self.game_state.prev_room_for_prompt_context = (
                self.game_state.current_room_name_for_map
            )
            self.game_state.action_leading_to_current_room_for_prompt_context = (
                action_taken
            )
            self.game_state.current_room_id = new_room_id
            self.game_state.current_room_name = new_room_name
            self.game_state.current_room_name_for_map = new_room_name  # DEPRECATED

            self.log_debug(
                f"Updated map from movement: {prev_room} (ID:{prev_room_id}) --({action_taken})--> {new_room_name} (ID:{new_room_id})",
                details=f"Movement: {prev_room} to {new_room_name} via {action_taken}",
            )

            self.logger.info(
                "Map updated from movement",
                extra={
                    "event_type": "map_movement_update",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "action": action_taken,
                    "from_room_id": prev_room_id,
                    "from_room": prev_room,
                    "to_room_id": new_room_id,
                    "to_room": new_room_name,
                },
            )

        except Exception as e:
            self.log_error(f"Failed to update map from movement: {e}")

    def _update_movement_tracking(
        self,
        action: str,
        from_room_id: int,
        from_room_name: str,
        to_room_id: int,
        to_room_name: str,
        game_response: str = ""
    ) -> None:
        """Update movement tracking between rooms.

        Args:
            action: Movement action taken
            from_room_id: Origin room location ID
            from_room_name: Origin room name (for context)
            to_room_id: Destination room location ID
            to_room_name: Destination room name (for context)
            game_response: Game's response to action
        """
        try:
            # Create movement context
            movement_context = MovementContext(
                current_location=to_room_name,
                previous_location=from_room_name,
                action=action,
                game_response=game_response,
                turn_number=self.game_state.turn_count,
            )

            # Analyze the movement
            movement_analysis = self.movement_analyzer.analyze_movement(
                movement_context
            )

            # Update room exits based on analysis
            if (
                hasattr(movement_analysis, "from_exits")
                and movement_analysis.from_exits
            ):
                self.game_map.update_room_exits(
                    room_id=from_room_id,  # Use integer ID
                    new_exits=movement_analysis.from_exits
                )

            # Add connection if movement was successful
            if (
                hasattr(movement_analysis, "connection_created")
                and movement_analysis.connection_created
            ):
                # Extract direction from action if possible
                direction = self._extract_direction_from_action(action)
                if direction:
                    self.game_map.add_connection(
                        from_room_id=from_room_id,  # Use integer ID
                        exit_taken=direction,
                        to_room_id=to_room_id,  # Use integer ID
                        confidence=0.8,  # Default confidence
                    )

            # Note: Pending connections are automatically cleared by the movement analyzer
            # when they are resolved, so no manual clearing is needed here

        except Exception as e:
            self.log_error(f"Failed to update movement tracking: {e}")

    def _extract_direction_from_action(self, action: str) -> Optional[str]:
        """Extract direction from movement action."""
        if not action:
            return None

        action_lower = action.lower().strip()

        # Direct direction mappings
        direction_map = {
            "north": "north",
            "n": "north",
            "south": "south",
            "s": "south",
            "east": "east",
            "e": "east",
            "west": "west",
            "w": "west",
            "up": "up",
            "u": "up",
            "down": "down",
            "d": "down",
            "northwest": "northwest",
            "nw": "northwest",
            "northeast": "northeast",
            "ne": "northeast",
            "southwest": "southwest",
            "sw": "southwest",
            "southeast": "southeast",
            "se": "southeast",
        }

        # Check for exact matches first
        if action_lower in direction_map:
            return direction_map[action_lower]

        # Check for "go" commands
        for prefix in ["go ", "move ", "walk "]:
            if action_lower.startswith(prefix):
                direction_part = action_lower[len(prefix) :].strip()
                if direction_part in direction_map:
                    return direction_map[direction_part]

        return None

    def track_failed_action(self, action: str, location_id: int, location_name: str) -> None:
        """Track a failed action at a specific location.

        Args:
            action: The action that failed
            location_id: Jericho location ID where action failed
            location_name: Human-readable location name
        """
        try:
            # Initialize failed actions tracking for this location (use name for backward compat)
            if location_name not in self.game_state.failed_actions_by_location:
                self.game_state.failed_actions_by_location[location_name] = []

            # Add the failed action
            self.game_state.failed_actions_by_location[location_name].append(action)

            # Track exit failure in the map graph (use integer ID)
            failure_count = self.game_map.track_exit_failure(location_id, action)

            self.log_debug(f"Tracked failed action: {action} at {location_name} (ID: {location_id})")

            # Check if we should prune this exit due to repeated failures
            if failure_count >= 3:  # Threshold for exit pruning
                pruned_count = self.game_map.prune_invalid_exits(location_id, min_failure_count=3)
                if pruned_count > 0:
                    self.log_info(
                        f"Pruned {pruned_count} unreliable exit(s) from {location_name} (ID: {location_id})"
                    )

        except Exception as e:
            self.log_error(f"Failed to track failed action: {e}")


    def check_map_update(self) -> None:
        """Check if map update is needed and perform periodic map maintenance."""
        try:
            self.log_progress(
                f"Running periodic map update at turn {self.game_state.turn_count}",
                stage="map_update",
                details=f"Map update at turn {self.game_state.turn_count}",
            )

            # Get current map quality metrics
            quality_metrics = self.get_quality_metrics()

            # Log map status
            self.logger.info(
                "Periodic map update",
                extra={
                    "event_type": "map_periodic_update",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "room_count": len(self.game_map.rooms),
                    "connection_count": len(self.game_map.connections),
                    **quality_metrics,
                },
            )

            self.last_map_update_turn = self.game_state.turn_count

        except Exception as e:
            self.log_error(f"Failed during periodic map update: {e}")

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive map quality metrics."""
        try:
            return self.game_map.get_map_quality_metrics()
        except Exception as e:
            self.log_error(f"Failed to get map quality metrics: {e}")
            # Fallback basic metrics
            return {
                "room_count": len(getattr(self.game_map, "rooms", {})),
                "connection_count": len(getattr(self.game_map, "connections", {})),
                "confidence_score": 0.5,  # Default moderate confidence
                "fragmentation_score": 0.0,
                "isolated_room_count": 0,
            }

    def get_current_room_context(self) -> Dict[str, Any]:
        """Get current room context for agent prompts."""
        return {
            "current_room": self.game_state.current_room_name_for_map,
            "previous_room": self.game_state.prev_room_for_prompt_context,
            "action_to_current": self.game_state.action_leading_to_current_room_for_prompt_context,
            "failed_actions": self.game_state.failed_actions_by_location.get(
                self.game_state.current_room_name_for_map, []
            ),
        }

    def get_export_data(self) -> Dict[str, Any]:
        """Get map data for state export (matching old orchestrator format)."""
        try:
            return {
                "mermaid_diagram": self.game_map.render_mermaid(),
                "current_room": self.game_state.current_room_name_for_map,
                "current_room_id": self.game_state.current_room_id,  # Add ID for new consumers
                "total_rooms": len(self.game_map.rooms),
                "total_connections": sum(
                    len(connections)
                    for connections in self.game_map.connections.values()
                ),
                # Enhanced map metrics (like old orchestrator)
                "quality_metrics": self.game_map.get_map_quality_metrics(),
                "confidence_report": self.game_map.render_confidence_report(),
                # Exit failure tracking
                "exit_failure_stats": self.game_map.get_exit_failure_stats(),
                "exit_failure_report": self.game_map.render_exit_failure_report(),
                # Optional: Include raw data for advanced frontends
                "raw_data": {
                    "rooms": {
                        room_id: {"name": room.name, "exits": list(room.exits)}
                        for room_id, room in self.game_map.rooms.items()  # Fixed: room_id is int
                    },
                    "connections": self.game_map.connections,
                },
            }
        except Exception as e:
            self.log_error(f"Failed to get map export data: {e}")
            return {
                "mermaid_diagram": "graph LR\n    A[Error: Map unavailable]",
                "current_room": self.game_state.current_room_name_for_map,
                "current_room_id": self.game_state.current_room_id,
                "total_rooms": 0,
                "total_connections": 0,
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current map manager status."""
        status = super().get_status()
        quality_metrics = self.get_quality_metrics()

        status.update(
            {
                "current_room": self.game_state.current_room_name_for_map,
                "last_map_update_turn": self.last_map_update_turn,
                "turns_since_last_update": self.game_state.turn_count
                - self.last_map_update_turn,
                "map_update_interval": self.config.map_update_interval,
                **quality_metrics,
            }
        )
        return status
