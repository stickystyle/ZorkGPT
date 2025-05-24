"""
Shared Movement Analysis Logic for ZorkGPT

This module provides consistent movement detection and pending connection logic
that can be used by both real-time gameplay (main.py) and historical log analysis.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from map_graph import MapGraph, normalize_direction, is_non_movement_command


# Import generic location fallbacks from main.py constants
GENERIC_LOCATION_FALLBACKS = {
    "unknown location",
    "unknown area",
    "unclear area",
    "unspecified location",
    "same area",
    "same place",
    "no specific location",
    "not applicable",
    "na",
    "n/a",
    "",  # Empty string also a fallback
}


@dataclass
class MovementContext:
    """Represents the context needed for movement analysis"""

    current_location: str
    previous_location: Optional[str]
    action: str
    game_response: str
    turn_number: int


@dataclass
class MovementResult:
    """Result of movement analysis"""

    movement_occurred: bool
    from_location: Optional[str]
    to_location: Optional[str]
    action: str
    is_pending: bool
    environmental_factors: List[str]
    requires_resolution: bool
    connection_created: bool = False  # Whether a map connection should be created


class PendingConnection:
    """Represents a pending movement connection that needs later resolution"""

    def __init__(self, from_room: str, action: str, turn_created: int):
        self.from_room = from_room
        self.action = action
        self.turn_created = turn_created
        self.intermediate_actions: List[str] = []

    def add_intermediate_action(self, action: str) -> None:
        """Add an action that occurred while this connection was pending"""
        self.intermediate_actions.append(action)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "from_room": self.from_room,
            "action": self.action,
            "turn_created": self.turn_created,
            "intermediate_actions": self.intermediate_actions,
        }


class MovementAnalyzer:
    """
    Shared movement analysis logic used by both real-time and historical analysis.

    This class handles:
    - Detection of movement vs non-movement actions
    - Pending connection creation and resolution (for dark rooms, etc.)
    - Environmental factor detection
    - Consistent movement pattern analysis
    """

    def __init__(self):
        self.pending_connections: List[PendingConnection] = []
        self.max_pending_turns = 3

    def analyze_movement(self, context: MovementContext) -> MovementResult:
        """
        Analyze a single turn for movement patterns.
        Used by both main.py (real-time) and log parser (historical).

        Args:
            context: MovementContext containing turn information

        Returns:
            MovementResult with analysis results
        """
        # First, check if this resolves any pending connections
        resolved_connection = self._check_pending_resolution(context)
        if resolved_connection:
            return resolved_connection

        # Then check if this creates a new movement (immediate or pending)
        return self._analyze_new_movement(context)

    def _check_pending_resolution(
        self, context: MovementContext
    ) -> Optional[MovementResult]:
        """Check if current context resolves a pending connection"""
        for i, pending in enumerate(
            self.pending_connections[:]
        ):  # Copy for safe removal
            if self._should_resolve_pending(pending, context):
                # Remove from pending list
                resolved_pending = self.pending_connections.pop(i)

                return MovementResult(
                    movement_occurred=True,
                    from_location=resolved_pending.from_room,
                    to_location=context.current_location,
                    action=resolved_pending.action,  # Use original movement action
                    is_pending=False,
                    environmental_factors=self._detect_environmental_factors(
                        context.game_response
                    ),
                    requires_resolution=False,
                    connection_created=True,
                )
        return None

    def _analyze_new_movement(self, context: MovementContext) -> MovementResult:
        """Analyze if current context represents new movement"""
        # Non-movement actions don't create connections
        if not self._is_movement_action(context.action):
            return MovementResult(
                movement_occurred=False,
                from_location=context.previous_location,
                to_location=context.current_location,
                action=context.action,
                is_pending=False,
                environmental_factors=self._detect_environmental_factors(
                    context.game_response
                ),
                requires_resolution=False,
                connection_created=False,
            )

        # Check for immediate successful movement
        if (
            context.current_location != context.previous_location
            and context.current_location
            and context.current_location.lower() not in GENERIC_LOCATION_FALLBACKS
            and context.previous_location
        ):
            return MovementResult(
                movement_occurred=True,
                from_location=context.previous_location,
                to_location=context.current_location,
                action=context.action,
                is_pending=False,
                environmental_factors=self._detect_environmental_factors(
                    context.game_response
                ),
                requires_resolution=False,
                connection_created=True,
            )

        # Check for explicitly failed movement (blocked paths, walls, etc.) BEFORE pending movement
        elif self._is_blocked_movement(context.game_response):
            return MovementResult(
                movement_occurred=False,  # Movement was attempted but blocked
                from_location=context.previous_location,
                to_location=context.previous_location,  # Stay in same location
                action=context.action,
                is_pending=False,
                environmental_factors=self._detect_environmental_factors(
                    context.game_response
                ),
                requires_resolution=False,
                connection_created=False,
            )

        # Check for pending movement (dark room, unclear destination, etc.)
        elif self._indicates_pending_movement(
            context.game_response, context.current_location, context.previous_location
        ):
            # Create pending connection
            pending = PendingConnection(
                from_room=context.previous_location,
                action=context.action,
                turn_created=context.turn_number,
            )
            self.pending_connections.append(pending)

            return MovementResult(
                movement_occurred=True,  # Movement happened, just not resolved yet
                from_location=context.previous_location,
                to_location=None,  # To be determined later
                action=context.action,
                is_pending=True,
                environmental_factors=self._detect_environmental_factors(
                    context.game_response
                ),
                requires_resolution=True,
                connection_created=False,  # Will be created when resolved
            )

        # Unknown movement result - could be failed or unclear
        else:
            return MovementResult(
                movement_occurred=True,  # Assume movement was attempted
                from_location=context.previous_location,
                to_location=context.current_location,
                action=context.action,
                is_pending=False,
                environmental_factors=self._detect_environmental_factors(
                    context.game_response
                ),
                requires_resolution=False,
                connection_created=False,
            )

    def _should_resolve_pending(
        self, pending: PendingConnection, context: MovementContext
    ) -> bool:
        """Determine if a pending connection should be resolved with current context"""
        # Don't resolve if too much time has passed
        if context.turn_number - pending.turn_created > self.max_pending_turns:
            return False

        # Don't resolve if we don't have a clear location
        if not context.current_location:
            return False

        # Don't resolve to generic/fallback locations
        if context.current_location.lower() in GENERIC_LOCATION_FALLBACKS:
            return False

        # Don't resolve back to the same location we started from
        if context.current_location == pending.from_room:
            return False

        return True

    def _is_movement_action(self, action: str) -> bool:
        """Determine if an action represents movement"""
        if not action:
            return False
        return not is_non_movement_command(action)

    def _indicates_pending_movement(
        self, response: str, current_location: str, previous_location: str
    ) -> bool:
        """
        Determine if game response indicates movement that needs later resolution.

        This handles cases like dark rooms where movement occurred but destination is unclear.
        """
        if not response:
            return False

        response_lower = response.lower()

        # Dark room indicators
        dark_room_indicators = [
            "it is pitch dark",
            "pitch black",
            "too dark to see",
            "you are likely to be eaten by a grue",
            "darkness",
            "you can't see anything",
            "it's too dark",
        ]

        # Check for dark room response
        is_dark_response = any(
            indicator in response_lower for indicator in dark_room_indicators
        )

        # Also check if location extraction failed (stayed same) which might indicate unclear destination
        location_unclear = (
            current_location == previous_location
            and current_location
            and current_location.lower() not in GENERIC_LOCATION_FALLBACKS
        )

        return is_dark_response or location_unclear

    def _is_blocked_movement(self, response: str) -> bool:
        """Determine if game response indicates explicitly blocked movement."""
        if not response:
            return False

        response_lower = response.lower()

        # Explicit blocking indicators
        blocking_indicators = [
            "there is a wall there",
            "you can't go that way",
            "it is too narrow",
            "the door is closed",
            "the door is locked",
            "you can't move",
            "blocked",
            "impassable",
            "no exit",
            "way is blocked",
        ]

        return any(indicator in response_lower for indicator in blocking_indicators)

    def _detect_environmental_factors(self, response: str) -> List[str]:
        """Detect environmental factors that affect movement from game response"""
        factors = []
        if not response:
            return factors

        response_lower = response.lower()

        # Darkness-related factors
        if any(
            dark in response_lower for dark in ["dark", "pitch", "grue", "darkness"]
        ):
            factors.append("darkness")

        # Locked/blocked factors
        if any(
            locked in response_lower for locked in ["locked", "bolt", "key", "closed"]
        ):
            factors.append("locked")

        # Physical barriers
        if any(
            blocked in response_lower
            for blocked in ["wall", "barrier", "blocked", "impassable"]
        ):
            factors.append("blocked")

        # Climbing/elevation factors
        if any(
            climb in response_lower for climb in ["ladder", "climb", "rope", "steep"]
        ):
            factors.append("climbing")

        # Water/swimming factors
        if any(
            water in response_lower for water in ["water", "swim", "river", "stream"]
        ):
            factors.append("water")

        # Combat/danger factors
        if any(
            combat in response_lower
            for combat in ["troll", "attack", "fight", "hostile"]
        ):
            factors.append("combat")

        # Tool/item requirement indicators
        if any(
            tool in response_lower
            for tool in ["need", "require", "must have", "without"]
        ):
            factors.append("requires_item")

        return factors

    def add_intermediate_action_to_pending(self, action: str, turn_number: int) -> None:
        """Add an intermediate action to pending connections within the delay window"""
        for pending in self.pending_connections:
            if turn_number - pending.turn_created <= self.max_pending_turns:
                pending.add_intermediate_action(action)

    def cleanup_expired_pending(self, current_turn: int) -> List[PendingConnection]:
        """Remove expired pending connections and return them for logging"""
        expired = []
        remaining = []

        for pending in self.pending_connections:
            if current_turn - pending.turn_created > self.max_pending_turns:
                expired.append(pending)
            else:
                remaining.append(pending)

        self.pending_connections = remaining
        return expired

    def get_pending_connections(self) -> List[Dict]:
        """Get current pending connections for logging/debugging"""
        return [pending.to_dict() for pending in self.pending_connections]

    def has_pending_connections(self) -> bool:
        """Check if there are any pending connections"""
        return len(self.pending_connections) > 0

    def clear_pending_connections(self) -> None:
        """Clear all pending connections (useful for episode resets)"""
        self.pending_connections.clear()


# Utility functions for external use
def create_movement_context(
    current_location: str,
    previous_location: Optional[str],
    action: str,
    game_response: str,
    turn_number: int,
) -> MovementContext:
    """Convenience function to create MovementContext"""
    return MovementContext(
        current_location=current_location,
        previous_location=previous_location,
        action=action,
        game_response=game_response,
        turn_number=turn_number,
    )


def is_dark_room_response(game_text: str) -> bool:
    """
    Check if the game response indicates the player is in a dark room.
    This is a convenience function that matches the existing logic in main.py.
    """
    if not game_text:
        return False

    dark_room_indicators = [
        "it is pitch dark",
        "pitch black",
        "too dark to see",
        "darkness",
        "you are likely to be eaten by a grue",
    ]

    game_text_lower = game_text.lower()
    return any(indicator in game_text_lower for indicator in dark_room_indicators)
