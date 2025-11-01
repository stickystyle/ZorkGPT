"""
ZorkCritic module for evaluating actions and managing critic trust.
"""

import json
import re
from typing import Optional, List, Tuple, Dict
from pydantic import BaseModel
from collections import Counter
from llm_client import LLMClientWrapper
from config import get_config, get_client_api_key

# Import shared utilities
from shared_utils import create_json_schema, strip_markdown_json_fences

try:
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    # Graceful fallback - no-op decorator
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGFUSE_AVAILABLE = False


class CriticResponse(BaseModel):
    score: float
    justification: str
    confidence: float = 0.8  # Default confidence level


class FailureDetectionResponse(BaseModel):
    action_failed: bool
    reason: str


class ValidationResult(BaseModel):
    """Result of object tree validation."""
    valid: bool
    reason: str
    confidence: float = 0.9  # High confidence for Z-machine validated rejections


class CriticTrustTracker:
    """Tracks critic performance and adjusts trust levels accordingly."""

    def __init__(self):
        self.correct_rejections = 0  # Rejected actions that led to parser failures
        self.incorrect_rejections = 0  # Rejected actions that might have been good
        self.total_evaluations = 0
        self.trust_level = 0.8  # Start with high trust
        self.recent_outcomes = []  # Track recent decisions for moving average

    def update_trust(self, was_rejection_correct: bool):
        """Update trust based on whether a rejection was justified."""
        self.total_evaluations += 1
        self.recent_outcomes.append(was_rejection_correct)

        # Keep only recent outcomes (sliding window)
        if len(self.recent_outcomes) > 20:
            self.recent_outcomes.pop(0)

        if was_rejection_correct:
            self.correct_rejections += 1
        else:
            self.incorrect_rejections += 1

        # Calculate trust based on recent performance
        if len(self.recent_outcomes) >= 5:
            recent_accuracy = sum(self.recent_outcomes) / len(self.recent_outcomes)
            self.trust_level = min(0.95, max(0.3, recent_accuracy))

    def get_rejection_threshold(self, base_threshold: float = None) -> float:
        """Get adjusted rejection threshold based on current trust level."""
        if base_threshold is None:
            # Use configuration or default to a more reasonable threshold
            config = get_config()
            base_threshold = config.gameplay.critic_rejection_threshold
        return base_threshold * self.trust_level

    def should_be_conservative(self) -> bool:
        """Return True if we should be more conservative due to low trust."""
        return self.trust_level < 0.5


class ActionRejectionSystem:
    """Handles action rejection with enhanced override mechanisms using combined heuristics."""

    def __init__(self):
        self.rejected_actions_this_turn = []
        self.turns_since_movement = 0
        self.recent_critic_scores = []

    def should_override_rejection(
        self,
        action: str,
        current_location: str,
        failed_actions_by_location: Dict[str, set],
        context: dict,
    ) -> Tuple[bool, str]:
        """
        Determine if a critic rejection should be overridden using enhanced heuristics.

        This method combines multiple signals to distinguish between:
        - Productive exploration (allow to continue)
        - True loops/stagnation (trigger override)
        """

        # 1. NEVER override if this action already failed at this location
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
        turns_without_progress = context.get("turns_since_movement", 0)

        # 2. Quick check: not enough data for meaningful analysis
        if len(recent_locations) < 6:
            return self._check_other_override_conditions(action, context)

        # 3. Immediate action repetition (high priority detection)
        if len(recent_actions) >= 3:
            if len(set(recent_actions[-3:])) == 1:
                return True, "immediate_repetition_detected"

        # 4. Enhanced location-based loop detection
        unique_recent_locations = list(set(recent_locations[-6:]))
        if len(unique_recent_locations) <= 2:
            # 4a. Calculate action diversity in recent location stays
            recent_location_actions = (
                recent_actions[-6:] if len(recent_actions) >= 6 else recent_actions
            )
            action_diversity = self._calculate_action_diversity(recent_location_actions)

            # 4b. Assess whether recent actions show meaningful progress
            is_making_progress = self._assess_exploration_progress(
                previous_actions_and_responses[-6:]
            )

            # 4c. Analyze location type for exploration appropriateness
            location_context = self._analyze_location_type(current_location)

            # 4d. Decision matrix for location-based scenarios
            if action_diversity < 0.3 and not is_making_progress:
                return True, "low_diversity_no_progress_loop"
            elif (
                action_diversity < 0.5
                and not is_making_progress
                and turns_without_progress >= 4
            ):
                # Broader threshold for low diversity + no progress with some movement stagnation
                return True, "low_diversity_no_progress_loop"
            elif location_context == "simple" and len(recent_location_actions) > 4:
                return True, "over_exploring_simple_location"
            elif not is_making_progress and turns_without_progress > 10:
                return True, "extended_stagnation_detected"
            elif action_diversity < 0.4 and turns_without_progress > 6:
                return True, "repetitive_actions_no_movement"

            # 4e. If we're in a single location with good diversity and progress, allow continued exploration
            # This prevents productive exploration from being flagged as problematic
            if (
                len(unique_recent_locations) == 1
                and action_diversity >= 0.5
                and is_making_progress
            ):
                return (
                    False,
                    None,
                )  # Explicitly allow productive single-location exploration

        # 5. Action cycling detection (bouncing between small set of actions)
        if len(recent_actions) >= 8:
            cycling_detected = self._detect_action_cycling(recent_actions[-8:])
            if cycling_detected:
                return True, "action_cycling_detected"

        # 6. Fall back to other override conditions for non-loop scenarios
        return self._check_other_override_conditions(action, context)

    def _calculate_action_diversity(self, actions: List[str]) -> float:
        """
        Calculate action diversity score based on variety of verbs and action categories.

        Returns:
            float: Diversity score between 0.0 (no diversity) and 1.0 (high diversity)
        """
        if not actions:
            return 1.0  # Default to high diversity with no data

        # Extract unique action verbs (first word of each action)
        unique_verbs = set()
        for action in actions:
            verb = action.lower().split()[0] if action.strip() else ""
            if verb:
                unique_verbs.add(verb)

        # Calculate basic diversity ratio
        verb_diversity = len(unique_verbs) / len(actions)

        # Bonus for using different categories of actions
        exploration_verbs = {
            "look",
            "examine",
            "search",
            "inspect",
            "take",
            "get",
            "read",
        }
        interaction_verbs = {
            "open",
            "close",
            "push",
            "pull",
            "touch",
            "use",
            "move",
            "turn",
        }
        movement_verbs = {
            "go",
            "north",
            "south",
            "east",
            "west",
            "up",
            "down",
            "enter",
            "exit",
            "climb",
        }

        categories_used = set()
        if any(verb in unique_verbs for verb in exploration_verbs):
            categories_used.add("exploration")
        if any(verb in unique_verbs for verb in interaction_verbs):
            categories_used.add("interaction")
        if any(verb in unique_verbs for verb in movement_verbs):
            categories_used.add("movement")

        # Category diversity bonus (up to 0.3 additional points)
        category_bonus = len(categories_used) * 0.1

        return min(1.0, verb_diversity + category_bonus)

    def _assess_exploration_progress(
        self, action_response_pairs: List[Tuple[str, str]]
    ) -> bool:
        """
        Determine if recent actions show meaningful progress or discovery.

        Returns:
            bool: True if making progress, False if stagnating
        """
        if len(action_response_pairs) < 2:
            return True  # Assume progress with limited data

        recent_responses = [resp for _, resp in action_response_pairs]

        # Signs of meaningful progress/discovery
        progress_indicators = [
            "taken",
            "opened",
            "closed",
            "found",
            "see",
            "hear",
            "notice",
            "score",
            "points",
            "new",
            "different",
            "reveals",
            "discover",
            "inside",
            "contains",
            "appears",
            "seems",
            "looks like",
            "successful",
            "works",
            "activates",
            "changes",
        ]

        # Signs of stagnation or failure
        stagnation_indicators = [
            "can't go that way",
            "don't understand",
            "nothing happens",
            "already",
            "can't see",
            "not here",
            "doesn't work",
            "no way",
            "impossible",
            "can't do that",
            "too narrow",
            "there is a wall",
            "blocked",
            "locked",
            "won't budge",
        ]

        progress_count = 0
        stagnation_count = 0

        for response in recent_responses:
            response_lower = response.lower()
            if any(indicator in response_lower for indicator in progress_indicators):
                progress_count += 1
            if any(indicator in response_lower for indicator in stagnation_indicators):
                stagnation_count += 1

        # Consider progress if we have more positive than negative signals,
        # or if we have any progress signals without excessive stagnation
        return progress_count > stagnation_count or (
            progress_count > 0 and stagnation_count < len(recent_responses) * 0.7
        )

    def _analyze_location_type(self, location_name: str) -> str:
        """
        Analyze location type to determine appropriate exploration duration.

        Returns:
            str: "rich", "simple", or "neutral" indicating exploration appropriateness
        """
        if not location_name:
            return "neutral"

        location_lower = location_name.lower()

        # Rich locations that merit extensive exploration
        rich_indicators = [
            "treasure",
            "chest",
            "vault",
            "chamber",
            "room",
            "office",
            "study",
            "kitchen",
            "bedroom",
            "library",
            "attic",
            "basement",
            "cellar",
            "shop",
            "store",
            "museum",
            "gallery",
            "tower",
            "temple",
            "shrine",
        ]

        # Simple locations that need minimal exploration
        simple_indicators = [
            "path",
            "trail",
            "road",
            "hallway",
            "corridor",
            "passage",
            "tunnel",
            "bridge",
            "stairs",
            "steps",
            "landing",
            "junction",
            "crossroads",
            "intersection",
            "clearing",
            "field",
            "meadow",
        ]

        if any(indicator in location_lower for indicator in rich_indicators):
            return "rich"
        elif any(indicator in location_lower for indicator in simple_indicators):
            return "simple"
        else:
            return "neutral"

    def _detect_action_cycling(self, recent_actions: List[str]) -> bool:
        """
        Detect if the agent is cycling between a small set of actions.

        Returns:
            bool: True if cycling detected, False otherwise
        """
        if len(recent_actions) < 6:
            return False

        # Check for alternating patterns (like A-B-A-B-A-B)
        unique_actions = list(set(recent_actions))

        # Cycling: using only 2-3 unique actions across 6+ turns
        if len(unique_actions) <= 3 and len(recent_actions) >= 6:
            # Verify this is actually repetitive (not just similar actions with variety)
            action_counts = {
                action: recent_actions.count(action) for action in unique_actions
            }
            max_count = max(action_counts.values())

            # If any action appears more than half the time, it's likely cycling
            if max_count > len(recent_actions) * 0.4:
                return True

        return False

    def _check_other_override_conditions(
        self, action: str, context: dict
    ) -> Tuple[bool, str]:
        """
        Check non-loop override conditions (existing logic preserved).
        """
        turns_without_progress = context.get("turns_since_movement", 0)

        # Override for systematic exploration of standard directions - RESTRICTIVE BUT TARGETED
        standard_directions = {
            "north",
            "south",
            "east",
            "west",
            "up",
            "down",
            "northeast",
            "northwest",
            "southeast",
            "southwest",
            "enter",
            "exit",
        }

        # Check if this is systematic exploration of a standard direction
        if action.lower() in standard_directions:
            critic_confidence = context.get("critic_confidence", 0.5)

            # Override if critic is NOT confident about the rejection AND agent shows signs of being stuck
            if (
                critic_confidence < 0.7 and turns_without_progress >= 2
            ):  # Lowered threshold
                return True, "systematic_exploration"

        # Override if agent is very stuck and needs to explore non-movement actions
        if turns_without_progress >= 8:
            # But only for reasonable non-movement actions, not invalid directions or failed actions
            reasonable_actions = [
                "climb",
                "examine",
                "take",
                "open",
                "close",
                "look",
                "search",
                "enter",
                "exit",
            ]
            if any(keyword in action.lower() for keyword in reasonable_actions):
                return True, "exploration_stuck"

        # Override for novel non-movement actions (MUCH more restrictive)
        # Check if action failed in current location specifically
        current_location = context.get("current_location", "")
        failed_actions_by_location = context.get("failed_actions_by_location", {})
        current_location_failed_actions = failed_actions_by_location.get(
            current_location, set()
        )

        if action.lower() not in current_location_failed_actions:
            # Only override if critic confidence is low (if available)
            critic_confidence = context.get(
                "critic_confidence", 0.5
            )  # Default to low confidence
            if critic_confidence >= 0.8:  # Don't override confident rejections
                return False, None

            # But only if it's a reasonable action type (non-directional actions)
            reasonable_actions = [
                "climb",
                "examine",
                "take",
                "open",
                "close",
                "look",
                "search",
            ]
            if any(keyword in action.lower() for keyword in reasonable_actions):
                return True, "novel_action"

        # Override if we're in a performance decline
        recent_scores = context.get("recent_critic_scores", [])
        if len(recent_scores) >= 3 and sum(recent_scores[-3:]) / 3 < -0.3:
            return True, "emergency_exploration"

        # Override if all recent action attempts have been rejected (increased threshold)
        if len(self.rejected_actions_this_turn) >= 3:  # Increased from 2
            return True, "consensus_override"

        return False, None

    def reset_turn(self):
        """Reset per-turn tracking."""
        self.rejected_actions_this_turn = []


class ZorkCritic:
    """
    Handles critic evaluation of actions with confidence scoring and trust tracking.
    """

    def __init__(
        self,
        model: str = None,
        client: Optional[LLMClientWrapper] = None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        logger=None,
        episode_id: str = "unknown",
    ):
        """
        Initialize the ZorkCritic.

        Args:
            model: Model name for critic evaluation
            client: OpenAI client instance (if None, creates new one)
            max_tokens: Maximum tokens for critic responses
            temperature: Temperature for critic model
            top_p: Top-p nucleus sampling
            top_k: Top-k sampling
            min_p: Minimum probability sampling
            logger: Logger instance for tracking
            episode_id: Current episode ID for logging
        """
        config = get_config()

        self.model = model or config.llm.critic_model
        self.max_tokens = (
            max_tokens if max_tokens is not None else config.critic_sampling.max_tokens
        )
        self.temperature = (
            temperature
            if temperature is not None
            else config.critic_sampling.temperature
        )
        self.top_p = top_p if top_p is not None else config.critic_sampling.top_p
        self.top_k = top_k if top_k is not None else config.critic_sampling.top_k
        self.min_p = min_p if min_p is not None else config.critic_sampling.min_p
        self.logger = logger
        self.episode_id = episode_id

        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = config.logging.enable_prompt_logging

        # Initialize LLM client if not provided
        if client is None:
            self.client = LLMClientWrapper(
                base_url=config.llm.get_base_url_for_model("critic"),
                api_key=get_client_api_key(),
            )
        else:
            self.client = client

        # Load system prompt
        self._load_system_prompt()

        # Initialize trust and rejection systems
        self.trust_tracker = CriticTrustTracker()
        self.rejection_system = ActionRejectionSystem()

    def _log_prompt_to_file(self, messages: List[Dict], prefix: str = "critic") -> None:
        """Log the full prompt to a temporary file for evaluation."""
        if not self.enable_prompt_logging:
            return

        self.prompt_counter += 1
        filename = f"tmp/{prefix}_{self.prompt_counter:03d}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"=== {prefix.upper()} PROMPT #{self.prompt_counter} ===\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"Max Tokens: {self.max_tokens}\n")
                f.write(f"Episode ID: {self.episode_id}\n")
                f.write("=" * 50 + "\n\n")

                for i, message in enumerate(messages):
                    f.write(f"--- MESSAGE {i + 1} ({message['role'].upper()}) ---\n")
                    f.write(message["content"])
                    f.write("\n\n")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to log prompt to {filename}: {e}",
                    extra={"episode_id": self.episode_id},
                )

    def _load_system_prompt(self) -> None:
        """Load critic system prompt from markdown file."""
        try:
            with open("critic.md") as fh:
                self.system_prompt = fh.read()
        except FileNotFoundError as e:
            if self.logger:
                self.logger.error(
                    f"Failed to load critic prompt file: {e}",
                    extra={"episode_id": self.episode_id},
                )
            raise

    def _validate_against_object_tree(
        self,
        action: str,
        jericho_interface
    ) -> ValidationResult:
        """
        Validate action against Z-machine object tree.

        This method provides fast, high-confidence validation for common
        failure cases by checking the actual game state:
        - "take X" when X is not visible or not takeable
        - "open X" when X is not present or not openable
        - "go X" when X is not a valid exit

        Args:
            action: Action to validate (e.g., "take lamp", "open door")
            jericho_interface: JerichoInterface for object tree access

        Returns:
            ValidationResult indicating if action is valid
        """
        try:
            # Parse action to extract verb and object
            parts = action.lower().strip().split(maxsplit=1)
            if len(parts) < 2:
                # Single word commands are usually fine (look, inventory, etc.)
                return ValidationResult(valid=True, reason="Single word command")

            verb = parts[0]
            target = parts[1]

            # Strip prepositional phrases to get the direct object
            # Common prepositions: from, in, with, to, at, on, under, behind
            prepositions = ['from', 'in', 'with', 'to', 'at', 'on', 'under', 'behind', 'into']
            for prep in prepositions:
                prep_pattern = f' {prep} '
                if prep_pattern in target:
                    target = target.split(prep_pattern)[0]
                    break

            # Validate "take" actions
            if verb in ['take', 'get', 'grab', 'pick']:
                # Handle comma-separated multi-object commands (e.g., "take X, Y, Z")
                targets = [t.strip() for t in target.split(',')]

                # Get visible objects in location
                visible_objects = jericho_interface.get_visible_objects_in_location()

                # Also get objects in inventory (for "take X from sack" when sack is held)
                inventory_objects = jericho_interface.get_inventory_structured()

                # Expand to include objects inside open/transparent containers
                # This is needed because Zork allows "take X" for objects in open containers
                # We need to recursively check because containers can be inside containers
                # Start with both room objects AND inventory objects
                all_accessible_objects = list(visible_objects) + list(inventory_objects)
                world_objects = jericho_interface.env.get_world_objects()

                # Helper to recursively add children of transparent objects
                def add_accessible_children(obj_list):
                    """Recursively add children of transparent objects."""
                    new_objects = []
                    for obj in obj_list:
                        attrs = jericho_interface.get_object_attributes(obj)
                        # If object is transparent, we can see (and take) its contents
                        # In Zork, transparent=true means the container is open or see-through
                        if attrs.get('transparent'):
                            for child_obj in world_objects:
                                if child_obj.parent == obj.num and child_obj not in all_accessible_objects:
                                    new_objects.append(child_obj)
                                    all_accessible_objects.append(child_obj)
                    return new_objects

                # Recursively add children until no new objects found
                # Start with both room objects and inventory objects
                current_level = list(visible_objects) + list(inventory_objects)
                while current_level:
                    current_level = add_accessible_children(current_level)

                # Validate each target object individually
                for single_target in targets:
                    found = False
                    for obj in all_accessible_objects:
                        if single_target in obj.name.lower():
                            # Object exists and is visible/accessible
                            # Note: We don't check 'takeable' attribute because Jericho's
                            # attributes are unreliable (e.g., lunch has takeable=False but
                            # can actually be taken). We only validate visibility here.
                            found = True
                            break

                    if not found:
                        return ValidationResult(
                            valid=False,
                            reason=f"Object '{single_target}' is not visible or accessible",
                            confidence=0.9
                        )

            # Validate "open/close" actions
            elif verb in ['open', 'close']:
                visible_objects = jericho_interface.get_visible_objects_in_location()

                found = False
                for obj in visible_objects:
                    if target in obj.name.lower():
                        attrs = jericho_interface.get_object_attributes(obj)
                        # Allow if explicitly openable OR if it's a container
                        # (containers in Zork may not have the openable bit but are still openable)
                        if attrs.get('openable') or attrs.get('container'):
                            found = True
                            break
                        else:
                            return ValidationResult(
                                valid=False,
                                reason=f"Object '{target}' cannot be opened/closed",
                                confidence=0.9
                            )

                if not found:
                    return ValidationResult(
                        valid=False,
                        reason=f"Object '{target}' is not present",
                        confidence=0.9
                    )

            # For all other actions, validation passes (LLM will handle)
            return ValidationResult(valid=True, reason="Action not validated by object tree")

        except Exception as e:
            # If validation fails, default to allowing the action (let LLM decide)
            if self.logger:
                self.logger.warning(f"Object tree validation error: {e}")
            return ValidationResult(valid=True, reason="Validation error - defaulting to allow")

    @observe(name="critic-evaluate-action")
    def evaluate_action(
        self,
        game_state_text: str,
        proposed_action: str,
        available_exits: Optional[List[str]] = None,
        action_counts: Optional[Counter] = None,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
        current_location_name: Optional[str] = None,
        failed_actions_by_location: Optional[Dict[str, set]] = None,
        jericho_interface=None,  # NEW: Optional JerichoInterface for validation
        inventory: Optional[List[str]] = None,  # Current inventory for action validation
    ) -> CriticResponse:
        """
        Get an evaluation from the Critic LM.

        Args:
            game_state_text: Current game state text
            proposed_action: The action to evaluate
            available_exits: List of valid exits from current location for spatial awareness
            action_counts: Counter of action frequencies
            previous_actions_and_responses: Recent action history
            current_location_name: Name of the current location
            failed_actions_by_location: Dict mapping location names to sets of failed actions
            jericho_interface: Optional JerichoInterface for object tree validation
            inventory: Current inventory items for evaluating item-based actions

        Returns:
            CriticResponse with score and justification
        """
        # Validate against object tree if Jericho interface is available
        if jericho_interface:
            validation_result = self._validate_against_object_tree(
                proposed_action, jericho_interface
            )

            if not validation_result.valid:
                # Return high-confidence rejection based on Z-machine data
                return CriticResponse(
                    score=0.0,
                    justification=f"[Object Tree Validation] {validation_result.reason}",
                    confidence=validation_result.confidence
                )

        # If validation passes, continue with LLM-based evaluation
        # Prepare context about repetitive actions for the critic
        repetition_context = ""
        repetition_details = []

        # Global count context (useful for overall action frequency)
        if action_counts and action_counts[proposed_action] > 2:
            global_count = action_counts[proposed_action]
            repetition_details.append(
                f"Globally, '{proposed_action}' has been tried {global_count} times."
            )

        # Location-specific failure context
        if current_location_name and failed_actions_by_location:
            current_location_failed_actions = failed_actions_by_location.get(
                current_location_name, set()
            )
            if proposed_action.lower() in current_location_failed_actions:
                repetition_details.append(
                    f"IMPORTANT: The action '{proposed_action}' has previously FAILED in this specific location ('{current_location_name}')."
                )

            # Context about failures in other locations (if useful)
            other_failed_locations = []
            for loc, failed_set in failed_actions_by_location.items():
                if (
                    loc != current_location_name
                    and proposed_action.lower() in failed_set
                ):
                    other_failed_locations.append(loc)

            if other_failed_locations:
                if len(other_failed_locations) == 1:
                    repetition_details.append(
                        f"Note: '{proposed_action}' also failed in a different location: '{other_failed_locations[0]}'."
                    )
                else:
                    repetition_details.append(
                        f"Note: '{proposed_action}' also failed in {len(other_failed_locations)} other locations."
                    )

        if repetition_details:
            repetition_context = "\n" + "\n".join(repetition_details)

        # Add context about the last few actions and responses
        recent_context = ""
        if previous_actions_and_responses and len(previous_actions_and_responses) > 0:
            recent_context = "\nRecent actions and responses:\n"
            for act, resp in previous_actions_and_responses[-3:]:
                recent_context += f"Command: {act}\nResult: {resp.strip()}\n"

        # Add spatial context if available
        spatial_context = ""
        if available_exits:
            spatial_context = (
                f"\nAvailable exits from current location: {', '.join(available_exits)}"
            )

        # Add inventory context if available
        inventory_context = ""
        if inventory:
            inventory_context = (
                f"\nCurrent inventory: {', '.join(inventory)}"
            )

        user_prompt = f"""Current Game State:
{game_state_text}{spatial_context}{inventory_context}

Proposed Agent Action:
{proposed_action}{repetition_context}{recent_context}

Evaluate this action based on your criteria. Respond with ONLY a JSON object in this exact format:
{{"score": 0.0, "justification": "Your justification here", "confidence": 0.8}}
"""
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": user_prompt},
        ]

        # Log the full prompt for evaluation
        self._log_prompt_to_file(messages, "critic")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                name="Critic",
                response_format=create_json_schema(CriticResponse),
            )

            response_content = response.content
            try:
                # Strip markdown fences if present (some LLMs wrap JSON in ```json ... ```)
                json_content = strip_markdown_json_fences(response_content)

                # Clean up JSON content to handle common formatting issues
                # Fix positive numbers with + prefix (e.g., +0.2 -> 0.2)
                json_content = re.sub(r":\s*\+(\d+\.?\d*)", r": \1", json_content)

                # Fix unterminated strings by ensuring quotes are properly closed
                # This is a basic fix - if there's an odd number of quotes, add a closing quote
                quote_count = json_content.count('"')
                if quote_count % 2 == 1:
                    json_content += '"'

                # Ensure the JSON object is properly closed
                if json_content.strip() and not json_content.strip().endswith("}"):
                    json_content = json_content.strip() + "}"

                parsed_data = json.loads(json_content)
                return CriticResponse(**parsed_data)
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error parsing critic response: {e}",
                        extra={"episode_id": self.episode_id},
                    )
                    self.logger.error(
                        f"Response content: {response_content}",
                        extra={"episode_id": self.episode_id},
                    )
                return CriticResponse(
                    score=0.0, justification="Critic evaluation error (parsing)."
                )
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error getting critic evaluation: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return CriticResponse(
                score=0.0, justification="Critic evaluation error (API)."
            )

    def get_robust_evaluation(
        self,
        game_state_text: str,
        proposed_action: str,
        available_exits: Optional[List[str]] = None,
        action_counts: Optional[Counter] = None,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
        current_location_name: Optional[str] = None,
        failed_actions_by_location: Optional[Dict[str, set]] = None,
        max_attempts: int = 3,
        inventory: Optional[List[str]] = None,
    ) -> CriticResponse:
        """
        Get a robust critic evaluation with confidence scoring and consensus mechanism.

        Args:
            game_state_text: Current game state text
            proposed_action: The action to evaluate
            available_exits: List of valid exits from current location for spatial awareness
            action_counts: Counter of action frequencies
            previous_actions_and_responses: Recent action history
            current_location_name: Name of the current location
            failed_actions_by_location: Dict mapping location names to sets of failed actions
            max_attempts: Maximum number of evaluation attempts for consensus
            inventory: Current inventory items for evaluating item-based actions

        Returns:
            CriticResponse with score, justification, and confidence
        """
        evaluations = []

        for attempt in range(max_attempts):
            evaluation = self.evaluate_action(
                game_state_text,
                proposed_action,
                available_exits,
                action_counts,
                previous_actions_and_responses,
                current_location_name,
                failed_actions_by_location,
                inventory=inventory,
            )
            evaluations.append(evaluation)

            # Early exit if highly confident
            if evaluation.confidence > 0.9:
                break

        # Use consensus or most confident evaluation
        if len(evaluations) > 1:
            scores = [e.score for e in evaluations]
            score_range = max(scores) - min(scores)

            if score_range > 0.4:  # High disagreement between evaluations
                # Use most confident evaluation
                best_eval = max(evaluations, key=lambda e: e.confidence)
                if self.logger:
                    self.logger.info(
                        f"High disagreement in critic evaluations (range: {score_range:.2f}), using most confident",
                        extra={
                            "event_type": "critic_consensus",
                            "episode_id": self.episode_id,
                            "score_range": score_range,
                            "selected_confidence": best_eval.confidence,
                        },
                    )
                return best_eval
            else:
                # Use average with combined confidence, but keep the best justification
                avg_score = sum(scores) / len(scores)
                avg_confidence = sum(e.confidence for e in evaluations) / len(
                    evaluations
                )
                # Use the justification from the most confident evaluation
                best_justification = max(
                    evaluations, key=lambda e: e.confidence
                ).justification
                return CriticResponse(
                    score=avg_score,
                    justification=best_justification,
                    confidence=avg_confidence,
                )

        return evaluations[0]

    def track_rejection_outcome(
        self, rejected_actions: List[str], actual_action: str, outcome: str
    ):
        """Track whether critic rejections were justified based on actual outcomes."""
        if not rejected_actions:
            return

        # Define patterns that indicate the action failed
        failure_patterns = [
            "i don't understand",
            "there is a wall",
            "too narrow",
            "can't do that",
            "not possible",
            "doesn't work",
            "nothing happens",
            "that doesn't make sense",
            "there was no verb in that sentence",
        ]

        # Check if the actual action resulted in a parser failure
        action_failed = any(pattern in outcome.lower() for pattern in failure_patterns)

        # If the actual action failed, rejections were likely correct
        # If the actual action succeeded, we can't easily determine if rejections were wrong
        # (since we don't know what the rejected actions would have done)
        if action_failed:
            # Rejections were probably correct
            self.trust_tracker.update_trust(was_rejection_correct=True)
        else:
            # This is more complex - we'll be conservative and not penalize the critic
            # unless we have strong evidence the rejection was wrong
            pass

    def detect_action_failure(
        self, action: str, game_response: str
    ) -> FailureDetectionResponse:
        """
        Use LLM to determine if an action failed based on the game response.

        Args:
            action: The action that was attempted
            game_response: The response from the game

        Returns:
            FailureDetectionResponse indicating if the action failed and why
        """
        user_prompt = f"""Analyze this action and game response to determine if the action failed:

Action attempted: {action}
Game response: {game_response}

An action is considered "failed" if:
- The parser didn't understand the command ("I don't understand", "There was no verb in that sentence")
- The action was explicitly rejected ("You can't do that", "You can't go that way", "Nothing happens")
- The action was impossible ("There is a wall", "You can't see any such thing", "It's too narrow")
- The response indicates the action didn't work as intended

An action is NOT failed if:
- It provided useful information (descriptions, examinations)
- It worked but had unexpected consequences
- It moved the game state forward in any way
- The response is descriptive rather than rejective

Respond with ONLY a JSON object in this exact format:
{{"action_failed": true/false, "reason": "Brief explanation of why it failed or succeeded"}}
"""

        messages = [{"role": "user", "content": user_prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=100,
                name="Critic",
                response_format=create_json_schema(FailureDetectionResponse),
            )

            response_content = response.content
            try:
                # Strip markdown fences if present (some LLMs wrap JSON in ```json ... ```)
                json_content = strip_markdown_json_fences(response_content)

                parsed_data = json.loads(json_content)
                return FailureDetectionResponse(**parsed_data)
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error parsing failure detection response: {e}",
                        extra={"episode_id": self.episode_id},
                    )
                # Default to action succeeded if we can't parse
                return FailureDetectionResponse(
                    action_failed=False,
                    reason="Error parsing failure detection response",
                )
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error in failure detection: {e}",
                    extra={"episode_id": self.episode_id},
                )
            # Default to action succeeded if LLM call fails
            return FailureDetectionResponse(
                action_failed=False, reason="Error in failure detection API call"
            )

    def update_episode_id(self, episode_id: str) -> None:
        """Update the episode ID for logging purposes."""
        self.episode_id = episode_id
