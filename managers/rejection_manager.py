"""
RejectionManager handles critic rejection state tracking and override decisions.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration


@dataclass
class RejectionState:
    """Tracks rejection-related state for persistence."""

    trust_level: float = 0.8
    correct_rejections: int = 0
    incorrect_rejections: int = 0
    total_evaluations: int = 0
    recent_outcomes: List[bool] = field(default_factory=list)
    turns_since_movement: int = 0
    recent_critic_scores: List[float] = field(default_factory=list)


class RejectionManager(BaseManager):
    """Manages critic rejection state and override decisions."""

    def __init__(self, logger, config: GameConfiguration, game_state: GameState):
        super().__init__(logger, config, game_state, "rejection_manager")
        self.state = RejectionState()
        self.rejected_actions_this_turn: List[str] = []

    def reset_episode(self) -> None:
        """Reset manager state for new episode."""
        self.state = RejectionState()
        self.rejected_actions_this_turn = []
        self.log_info("RejectionManager reset for new episode")

    def process_turn(self) -> None:
        """Process turn-specific logic (not needed for rejection manager)."""
        pass

    def should_process_turn(self) -> bool:
        """Rejection manager doesn't need periodic processing."""
        return False

    def get_rejection_threshold(self) -> float:
        """Get adjusted rejection threshold based on current trust level."""
        base_threshold = self.config.critic_rejection_threshold
        return base_threshold * self.state.trust_level

    def update_trust(self, was_rejection_correct: bool) -> None:
        """Update trust based on whether a rejection was justified."""
        self.state.total_evaluations += 1
        self.state.recent_outcomes.append(was_rejection_correct)

        # Keep only recent outcomes (sliding window of 20)
        if len(self.state.recent_outcomes) > 20:
            self.state.recent_outcomes.pop(0)

        if was_rejection_correct:
            self.state.correct_rejections += 1
        else:
            self.state.incorrect_rejections += 1

        # Calculate trust based on recent performance
        if len(self.state.recent_outcomes) >= 5:
            recent_accuracy = sum(self.state.recent_outcomes) / len(
                self.state.recent_outcomes
            )
            self.state.trust_level = min(0.95, max(0.3, recent_accuracy))

    def should_override_rejection(
        self,
        action: str,
        current_location: str,
        failed_actions_by_location: Dict[str, Set[str]],
        context: dict,
    ) -> Tuple[bool, str]:
        """
        Determine if a critic rejection should be overridden using enhanced heuristics.

        Returns:
            Tuple of (should_override, reason)
        """
        # Never override if this action already failed at this location
        current_location_failed_actions = failed_actions_by_location.get(
            current_location, set()
        )
        if action.lower() in current_location_failed_actions:
            return False, "action_failed_in_current_location"

        # Extract context data
        recent_locations = context.get("recent_locations", [])
        recent_actions = context.get("recent_actions", [])
        previous_actions_and_responses = context.get(
            "previous_actions_and_responses", []
        )
        turns_without_progress = context.get(
            "turns_since_movement", self.state.turns_since_movement
        )

        # Quick check: not enough data for meaningful analysis
        if len(recent_locations) < 3 or len(recent_actions) < 3:
            return False, "insufficient_data"

        # Check for true loops vs productive exploration
        location_visits = {}
        for loc in recent_locations[-10:]:
            location_visits[loc] = location_visits.get(loc, 0) + 1

        # Get unique locations visited recently
        unique_recent_locations = len(set(recent_locations[-5:]))

        # Analyze action patterns
        action_lower = action.lower()
        recent_actions_lower = [a.lower() for a in recent_actions[-8:]]
        action_frequency = recent_actions_lower.count(action_lower)

        # Check parser responses for actual failures
        actual_failures = 0
        for action_taken, response in previous_actions_and_responses[-5:]:
            response_lower = response.lower()
            if any(
                phrase in response_lower
                for phrase in [
                    "i don't understand",
                    "you can't",
                    "that's not",
                    "invalid",
                    "what do you want",
                    "i beg your pardon",
                ]
            ):
                actual_failures += 1

        # Decision logic
        # 1. If we're exploring new locations, allow the action
        if unique_recent_locations >= 3:
            return True, "exploring_new_locations"

        # 2. If we haven't had many actual failures, trust the agent
        if actual_failures <= 1 and turns_without_progress < 5:
            return True, "low_failure_rate"

        # 3. If we're truly stuck (same location, high failures), override carefully
        if (
            turns_without_progress > 8
            and actual_failures > 2
            and unique_recent_locations == 1
            and action_frequency <= 1
        ):
            return True, "stuck_need_new_approach"

        # 4. If critic confidence is very low and we're not in a known bad pattern
        critic_confidence = context.get("critic_confidence", 0.8)
        if critic_confidence < 0.3 and action_frequency <= 2:
            return True, "low_critic_confidence"

        # 5. Check for oscillation pattern
        if len(recent_locations) >= 4:
            # A-B-A-B pattern
            if (
                recent_locations[-4] == recent_locations[-2]
                and recent_locations[-3] == recent_locations[-1]
                and recent_locations[-4] != recent_locations[-3]
            ):
                if action_lower not in ["look", "inventory", "examine"]:
                    return True, "breaking_oscillation"

        # Default: trust the critic
        return False, "default_trust_critic"

    def start_new_turn(self) -> None:
        """Reset per-turn tracking."""
        self.rejected_actions_this_turn = []

    def add_rejected_action(
        self, action: str, score: float, justification: str
    ) -> None:
        """Track a rejected action."""
        self.rejected_actions_this_turn.append(action)
        self.state.recent_critic_scores.append(score)

        # Keep only recent scores
        if len(self.state.recent_critic_scores) > 10:
            self.state.recent_critic_scores.pop(0)

    def update_movement_tracking(self, moved: bool) -> None:
        """Update turns since movement counter."""
        if moved:
            self.state.turns_since_movement = 0
        else:
            self.state.turns_since_movement += 1

    def get_state_for_export(self) -> dict:
        """Get rejection state for export."""
        return {
            "trust_level": self.state.trust_level,
            "correct_rejections": self.state.correct_rejections,
            "incorrect_rejections": self.state.incorrect_rejections,
            "total_evaluations": self.state.total_evaluations,
            "turns_since_movement": self.state.turns_since_movement,
            "recent_critic_scores": self.state.recent_critic_scores[-5:],  # Last 5
            "rejected_actions_this_turn": self.rejected_actions_this_turn,
        }

    def restore_state(self, state_dict: dict) -> None:
        """Restore rejection state from saved state."""
        if state_dict:
            self.state.trust_level = state_dict.get("trust_level", 0.8)
            self.state.correct_rejections = state_dict.get("correct_rejections", 0)
            self.state.incorrect_rejections = state_dict.get("incorrect_rejections", 0)
            self.state.total_evaluations = state_dict.get("total_evaluations", 0)
            self.state.turns_since_movement = state_dict.get("turns_since_movement", 0)
            self.state.recent_critic_scores = state_dict.get("recent_critic_scores", [])

    def get_status(self) -> dict:
        """Get current status of rejection tracking."""
        return {
            "trust_level": self.state.trust_level,
            "rejection_threshold": self.get_rejection_threshold(),
            "correct_rejections": self.state.correct_rejections,
            "incorrect_rejections": self.state.incorrect_rejections,
            "total_evaluations": self.state.total_evaluations,
            "turns_since_movement": self.state.turns_since_movement,
            "rejected_this_turn": len(self.rejected_actions_this_turn),
            "should_be_conservative": self.state.trust_level < 0.5,
        }
