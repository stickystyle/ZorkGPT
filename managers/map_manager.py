"""
MapManager for ZorkGPT orchestration.

Handles all map-related responsibilities:
- Map building and updating from movement patterns
- Room tracking and location changes
- Map quality assessment and metrics
- Navigation analysis and movement tracking
- Integration with MapGraph and MovementAnalyzer
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from map_graph import MapGraph
from movement_analyzer import MovementAnalyzer


class MapManager(BaseManager):
    """
    Manages all map-related functionality for ZorkGPT.
    
    Responsibilities:
    - Map building and room tracking
    - Movement analysis and navigation
    - Map quality assessment and metrics
    - Map consolidation and optimization
    - Integration with MapGraph and MovementAnalyzer
    """
    
    def __init__(
        self, 
        logger, 
        config: GameConfiguration, 
        game_state: GameState
    ):
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
        # Run map consolidation every turn
        self.run_map_consolidation()
        
        # Check for periodic map updates
        if self.should_process_turn():
            self.check_map_update()
    
    def should_process_turn(self) -> bool:
        """Check if map needs processing this turn."""
        # Check if it's time for a map update
        turns_since_update = self.game_state.turn_count - self.last_map_update_turn
        return (self.game_state.turn_count > 0 and 
                turns_since_update >= self.config.map_update_interval)
    
    def add_initial_room(self, room_name: str) -> None:
        """Add the initial room to the map."""
        if room_name:
            self.game_map.add_room(room_name)
            self.game_state.current_room_name_for_map = room_name
            
            self.log_debug(f"Added initial room to map: {room_name}")
            
            self.logger.info(
                f"Initial room added to map: {room_name}",
                extra={
                    "event_type": "map_initial_room_added",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "room_name": room_name,
                }
            )
    
    def update_from_movement(
        self, 
        action_taken: str, 
        new_room_name: str, 
        previous_room_name: Optional[str] = None
    ) -> None:
        """Update map based on movement action and result."""
        try:
            if not new_room_name:
                return
            
            # Use provided previous room or get from game state
            prev_room = previous_room_name or self.game_state.current_room_name_for_map
            
            # Add the new room to the map
            self.game_map.add_room(new_room_name)
            
            # Update movement tracking if we moved from a previous room
            if prev_room and prev_room != new_room_name:
                self._update_movement_tracking(action_taken, prev_room, new_room_name)
            
            # Update current room tracking
            self.game_state.prev_room_for_prompt_context = self.game_state.current_room_name_for_map
            self.game_state.action_leading_to_current_room_for_prompt_context = action_taken
            self.game_state.current_room_name_for_map = new_room_name
            
            self.log_debug(
                f"Updated map from movement: {prev_room} --({action_taken})--> {new_room_name}",
                details=f"Movement: {prev_room} to {new_room_name} via {action_taken}"
            )
            
            self.logger.info(
                f"Map updated from movement",
                extra={
                    "event_type": "map_movement_update",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "action": action_taken,
                    "from_room": prev_room,
                    "to_room": new_room_name,
                }
            )
            
        except Exception as e:
            self.log_error(f"Failed to update map from movement: {e}")
    
    def _update_movement_tracking(self, action: str, from_room: str, to_room: str) -> None:
        """Update movement tracking between rooms."""
        try:
            # Analyze the movement
            movement_analysis = self.movement_analyzer.analyze_movement(
                action=action,
                from_room=from_room,
                to_room=to_room,
                turn_number=self.game_state.turn_count
            )
            
            # Update room exits based on analysis
            self.game_map.update_room_exits(
                room_name=from_room,
                available_exits=movement_analysis.get("available_exits", [])
            )
            
            # Add connection if movement was successful
            if movement_analysis.get("successful_movement", False):
                direction = movement_analysis.get("direction")
                if direction:
                    self.game_map.add_connection(
                        from_room=from_room,
                        to_room=to_room,
                        direction=direction,
                        confidence=movement_analysis.get("confidence", 0.8)
                    )
            
            # Clear any pending connections that were resolved
            self.movement_analyzer.clear_pending_connections(from_room, to_room)
            
        except Exception as e:
            self.log_error(f"Failed to update movement tracking: {e}")
    
    def track_failed_action(self, action: str, location: str) -> None:
        """Track a failed action at a specific location."""
        try:
            # Initialize failed actions tracking for this location
            if location not in self.game_state.failed_actions_by_location:
                self.game_state.failed_actions_by_location[location] = []
            
            # Add the failed action
            self.game_state.failed_actions_by_location[location].append(action)
            
            # Track exit failure in the map graph
            self.game_map.track_exit_failure(location, action)
            
            self.log_debug(f"Tracked failed action: {action} at {location}")
            
            # Check if we should prune this exit due to repeated failures
            failure_count = self.game_state.failed_actions_by_location[location].count(action)
            if failure_count >= 3:  # Threshold for exit pruning
                self.game_map.prune_unreliable_exit(location, action)
                self.log_info(f"Pruned unreliable exit: {action} from {location} (failed {failure_count} times)")
            
        except Exception as e:
            self.log_error(f"Failed to track failed action: {e}")
    
    def run_map_consolidation(self) -> None:
        """Run map consolidation to merge similar locations and clean up fragmentation."""
        try:
            # Perform base name variant consolidation
            consolidation_results = self.game_map.consolidate_base_name_variants()
            
            if consolidation_results.get("consolidated_count", 0) > 0:
                self.log_debug(
                    f"Map consolidation merged {consolidation_results['consolidated_count']} rooms",
                    details=f"Consolidation results: {consolidation_results}"
                )
                
                self.logger.info(
                    f"Map consolidation completed",
                    extra={
                        "event_type": "map_consolidation",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "consolidated_count": consolidation_results.get("consolidated_count", 0),
                        "total_rooms": len(self.game_map.rooms),
                    }
                )
            
        except Exception as e:
            self.log_error(f"Failed to run map consolidation: {e}")
    
    def check_map_update(self) -> None:
        """Check if map update is needed and perform periodic map maintenance."""
        try:
            self.log_progress(
                f"Running periodic map update at turn {self.game_state.turn_count}",
                stage="map_update",
                details=f"Map update at turn {self.game_state.turn_count}"
            )
            
            # Get current map quality metrics
            quality_metrics = self.get_quality_metrics()
            
            # Log map status
            self.logger.info(
                f"Periodic map update",
                extra={
                    "event_type": "map_periodic_update",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "room_count": len(self.game_map.rooms),
                    "connection_count": len(self.game_map.connections),
                    **quality_metrics
                }
            )
            
            # Perform advanced consolidation if needed
            if quality_metrics.get("fragmentation_score", 0) > 0.3:  # High fragmentation
                self.game_map.consolidate_similar_locations()
                self.log_info("Performed similarity-based consolidation due to high fragmentation")
            
            # Prune fragmented nodes if needed
            if quality_metrics.get("isolated_room_count", 0) > 5:  # Too many isolated rooms
                pruned_count = self.game_map.prune_fragmented_nodes()
                if pruned_count > 0:
                    self.log_info(f"Pruned {pruned_count} fragmented nodes")
            
            self.last_map_update_turn = self.game_state.turn_count
            
        except Exception as e:
            self.log_error(f"Failed during periodic map update: {e}")
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get comprehensive map quality metrics."""
        try:
            if hasattr(self.game_map, 'get_quality_metrics'):
                return self.game_map.get_quality_metrics()
            else:
                # Fallback basic metrics
                return {
                    "room_count": len(self.game_map.rooms) if hasattr(self.game_map, 'rooms') else 0,
                    "connection_count": len(self.game_map.connections) if hasattr(self.game_map, 'connections') else 0,
                    "confidence_score": 0.5,  # Default moderate confidence
                    "fragmentation_score": 0.0,
                    "isolated_room_count": 0
                }
        except Exception as e:
            self.log_error(f"Failed to get map quality metrics: {e}")
            return {}
    
    def get_mermaid_representation(self) -> str:
        """Get mermaid diagram representation of the map."""
        try:
            if hasattr(self.game_map, 'to_mermaid'):
                return self.game_map.to_mermaid()
            else:
                return ""
        except Exception as e:
            self.log_error(f"Failed to get mermaid representation: {e}")
            return ""
    
    def get_current_room_context(self) -> Dict[str, Any]:
        """Get current room context for agent prompts."""
        return {
            "current_room": self.game_state.current_room_name_for_map,
            "previous_room": self.game_state.prev_room_for_prompt_context,
            "action_to_current": self.game_state.action_leading_to_current_room_for_prompt_context,
            "failed_actions": self.game_state.failed_actions_by_location.get(
                self.game_state.current_room_name_for_map, []
            )
        }
    
    def get_map_state_for_export(self) -> Dict[str, Any]:
        """Get map state for export to game state."""
        try:
            return {
                "room_count": len(self.game_map.rooms) if hasattr(self.game_map, 'rooms') else 0,
                "connection_count": len(self.game_map.connections) if hasattr(self.game_map, 'connections') else 0,
                "current_room": self.game_state.current_room_name_for_map,
                "quality_metrics": self.get_quality_metrics(),
                "mermaid_map": self.get_mermaid_representation()
            }
        except Exception as e:
            self.log_error(f"Failed to get map state for export: {e}")
            return {}
    
    def restore_map_state(self, map_data: Dict[str, Any]) -> None:
        """Restore map state from saved data."""
        try:
            # Restore basic room tracking
            if "current_room" in map_data:
                self.game_state.current_room_name_for_map = map_data["current_room"]
            
            # Additional restoration logic could be added here
            # depending on what map state needs to be preserved
            
            self.log_debug("Map state restored from saved data")
            
        except Exception as e:
            self.log_error(f"Failed to restore map state: {e}")
    
    def get_room_analysis_for_context(self, max_rooms: int = 10) -> str:
        """Get room analysis for agent context."""
        try:
            if hasattr(self.game_map, 'get_rooms_by_visit_frequency'):
                rooms = self.game_map.get_rooms_by_visit_frequency(limit=max_rooms)
                room_list = [f"- {room}" for room in rooms]
                return "\\n".join(room_list)
            else:
                # Fallback to basic room list
                rooms = list(self.game_map.rooms.keys()) if hasattr(self.game_map, 'rooms') else []
                return "\\n".join([f"- {room}" for room in rooms[:max_rooms]])
        except Exception as e:
            self.log_error(f"Failed to get room analysis: {e}")
            return ""
    
    def get_navigation_suggestions(self, target_room: Optional[str] = None) -> List[str]:
        """Get navigation suggestions based on map knowledge."""
        try:
            if hasattr(self.game_map, 'get_navigation_path') and target_room:
                path = self.game_map.get_navigation_path(
                    from_room=self.game_state.current_room_name_for_map,
                    to_room=target_room
                )
                return path if path else []
            else:
                # Return basic directional suggestions
                return ["north", "south", "east", "west", "up", "down"]
        except Exception as e:
            self.log_error(f"Failed to get navigation suggestions: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current map manager status."""
        status = super().get_status()
        quality_metrics = self.get_quality_metrics()
        
        status.update({
            "current_room": self.game_state.current_room_name_for_map,
            "last_map_update_turn": self.last_map_update_turn,
            "turns_since_last_update": self.game_state.turn_count - self.last_map_update_turn,
            "map_update_interval": self.config.map_update_interval,
            **quality_metrics
        })
        return status