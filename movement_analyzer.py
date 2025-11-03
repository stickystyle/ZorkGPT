"""
ABOUTME: Shared Movement Analysis Logic for ZorkGPT
ABOUTME: Provides ID-based movement detection using Jericho's stable location IDs
"""

from dataclasses import dataclass
from map_graph import is_non_movement_command


@dataclass
class MovementResult:
    """Result of movement analysis - simplified for ID-based detection.

    With Jericho providing stable location IDs, movement detection is trivial:
    if the ID changed, movement occurred. No pending connections or heuristics needed.
    """
    movement_occurred: bool
    from_location_id: int
    to_location_id: int
    action: str


class MovementAnalyzer:
    """
    Movement analysis logic using Jericho's stable location IDs.

    Handles:
    - Detection of movement vs non-movement actions
    - ID-based movement detection (simplified for Jericho integration)
    """

    def __init__(self):
        pass

    def analyze_movement(
        self,
        before_location_id: int,
        after_location_id: int,
        action: str
    ) -> MovementResult:
        """
        Analyze movement by comparing location IDs.

        With Jericho, movement detection is trivial: if the location ID
        changed, movement occurred. No heuristics needed.

        Args:
            before_location_id: Jericho location ID before the action
            after_location_id: Jericho location ID after the action
            action: The command that was executed

        Returns:
            MovementResult indicating whether movement occurred
        """
        movement_occurred = (before_location_id != after_location_id)

        return MovementResult(
            movement_occurred=movement_occurred,
            from_location_id=before_location_id,
            to_location_id=after_location_id,
            action=action,
        )

    def _is_movement_action(self, action: str) -> bool:
        """Determine if an action represents movement"""
        if not action:
            return False
        return not is_non_movement_command(action)
