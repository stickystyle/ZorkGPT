import json
from openai import OpenAI
from zork_api import ZorkInterface
import re
from pydantic import BaseModel
from collections import Counter
from typing import List, Optional, Tuple
import environs
from datetime import datetime
from map_graph import MapGraph, normalize_direction, is_non_movement_command
from movement_analyzer import (
    MovementAnalyzer,
    MovementContext,
    MovementResult,
    create_movement_context,
)
from logger import setup_logging, ZorkExperienceTracker
import os


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
    # Add any other generic phrases observed from LLM outputs if necessary
}

env = environs.Env()
env.read_env()


MAX_TOKENS_AGENT = 4096  # Agent should output very short commands however, it needs more tokens for reasoning
MAX_TOKENS_CRITIC = 100  # Critic might need more tokens for justification
MAX_TOKENS_INFO_EXT = 300  # Adjust based on expected JSON size and complexity

TEMPERATURE_AGENT = 0.5  # Some creativity but not too random for commands
TEMPERATURE_CRITIC = 0.2  # Critic should be more deterministic
TEMPERATURE_INFO_EXT = 0.1  # Low temperature for deterministic extraction


class CriticResponse(BaseModel):
    score: float
    justification: str
    confidence: float = 0.8  # Default confidence level


class ExtractorResponse(BaseModel):
    current_location_name: str
    exits: List[str]
    visible_objects: List[str]
    visible_characters: List[str]
    important_messages: List[str]
    in_combat: bool


with open("agent.md") as fh:
    AGENT_SYSTEM_PROMPT = fh.read()

with open("critic.md") as fh:
    CRITIC_SYSTEM_PROMPT = fh.read()

with open("extractor.md") as fh:
    EXTRACTOR_SYSTEM_PROMPT = fh.read()


class ZorkAgent:
    """
    A class to manage Zork gameplay episodes with integrated logging and experience tracking.

    This class encapsulates all the logic for playing Zork episodes, including:
    - Agent action generation
    - Critic evaluation
    - Information extraction
    - Memory management
    - Logging and experience tracking
    """

    def __init__(
        self,
        agent_model: str = None,
        critic_model: str = None,
        info_ext_model: str = None,
        episode_log_file: str = "zork_episode_log.txt",
        json_log_file: str = "zork_episode_log.jsonl",
        experiences_file: str = "zork_experiences.json",
        max_turns_per_episode: int = 200,
        client_base_url: str = None,
        client_api_key: str = None,
        # Dynamic turn limit parameters
        absolute_max_turns: int = 1000,
        turn_limit_increment: int = 50,
        performance_check_interval: int = 20,
        performance_threshold: float = 0.7,
        min_turns_for_increase: int = 50,
        # Automatic knowledge base updating
        auto_update_knowledge: bool = True,
    ):
        """
        Initialize the ZorkAgent.

        Args:
            agent_model: Model name for the agent
            critic_model: Model name for the critic
            info_ext_model: Model name for information extraction
            episode_log_file: Path for human-readable episode logs
            json_log_file: Path for JSON logs
            experiences_file: Path for experience data
            max_turns_per_episode: Initial maximum turns per episode
            client_base_url: OpenAI client base URL
            client_api_key: OpenAI client API key
            absolute_max_turns: Absolute maximum turns to prevent runaway costs
            turn_limit_increment: How much to increase turn limit when performance threshold is met
            performance_check_interval: How often (in turns) to check performance for turn limit increases
            performance_threshold: Average critic score threshold to trigger turn limit increase
            min_turns_for_increase: Minimum turns before first turn limit increase is possible
            auto_update_knowledge: Whether to automatically update knowledge base after each episode
        """
        # Configuration from environment or parameters
        self.agent_model = agent_model or env.str("AGENT_MODEL", "qwen3-30b-a3b-mlx")
        self.critic_model = critic_model or env.str("CRITIC_MODEL", "qwen3-30b-a3b-mlx")
        self.info_ext_model = info_ext_model or env.str(
            "INFO_EXT_MODEL", "qwen3-30b-a3b-mlx"
        )

        # File paths
        self.episode_log_file = episode_log_file
        self.json_log_file = json_log_file
        self.experiences_file = experiences_file

        # Game settings
        self.base_max_turns_per_episode = (
            max_turns_per_episode  # Store the initial/base value
        )
        self.max_turns_per_episode = max_turns_per_episode  # Current dynamic limit

        # Dynamic turn limit configuration
        self.absolute_max_turns = absolute_max_turns
        self.turn_limit_increment = turn_limit_increment
        self.performance_check_interval = performance_check_interval
        self.performance_threshold = performance_threshold
        self.min_turns_for_increase = min_turns_for_increase

        # Knowledge base configuration
        self.auto_update_knowledge = auto_update_knowledge

        # Model parameters
        self.max_tokens_agent = None
        self.max_tokens_critic = 100
        self.max_tokens_info_ext = 300
        self.temperature_agent = 0.5
        self.temperature_critic = 0.2
        self.temperature_info_ext = 0.1

        # Initialize logger and experience tracker
        self.logger = setup_logging(episode_log_file, json_log_file)
        self.experience_tracker = ZorkExperienceTracker()

        # Load system prompts
        self._load_system_prompts()

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=client_base_url or env.str("CLIENT_BASE_URL", None),
            api_key=client_api_key or env.str("CLIENT_API_KEY", None),
        )

        # Initialize shared movement analyzer for consistent movement detection
        self.movement_analyzer = MovementAnalyzer()

        # Initialize critic trust tracking and action rejection systems
        self.critic_trust_tracker = CriticTrustTracker()
        self.action_rejection_system = ActionRejectionSystem()

        # Episode state (reset for each episode)
        self.reset_episode_state()

    def _load_system_prompts(self) -> None:
        """Load system prompts from markdown files."""
        try:
            # Load base agent prompt
            with open("agent.md") as fh:
                base_agent_prompt = fh.read()

            # Try to enhance with knowledge base
            self.agent_system_prompt = self._enhance_prompt_with_knowledge(
                base_agent_prompt
            )

            with open("critic.md") as fh:
                self.critic_system_prompt = fh.read()
            
            # Try to use enhanced extractor, fall back to original if not found
            try:
                with open("enhanced_extractor.md") as fh:
                    self.extractor_system_prompt = fh.read()
                self.logger.info("Using enhanced extractor prompt")
            except FileNotFoundError:
                with open("extractor.md") as fh:
                    self.extractor_system_prompt = fh.read()
                self.logger.info("Using original extractor prompt")
                
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load system prompt file: {e}")
            raise

    def _enhance_prompt_with_knowledge(self, base_prompt: str) -> str:
        """Enhance the agent prompt with accumulated knowledge."""
        knowledge_file = "knowledgebase.md"

        if not os.path.exists(knowledge_file):
            return base_prompt

        try:
            with open(knowledge_file, "r", encoding="utf-8") as f:
                knowledge_content = f.read()

            # Insert strategic guide before the "Output Format" section
            knowledge_section = f"""

**STRATEGIC GUIDE FROM PREVIOUS EPISODES:**

The following strategic guide has been compiled from analyzing previous episodes. Use this guide to improve your performance, prioritize important items, navigate efficiently, and avoid known dangers:

{knowledge_content}

**END OF STRATEGIC GUIDE**

"""

            if "**Output Format" in base_prompt:
                insertion_point = base_prompt.find("**Output Format")
                enhanced_prompt = (
                    base_prompt[:insertion_point]
                    + knowledge_section
                    + base_prompt[insertion_point:]
                )
            else:
                enhanced_prompt = base_prompt + knowledge_section

            # Log knowledge integration
            self.logger.info(
                f"Enhanced prompt with knowledge base ({len(knowledge_content):,} characters)"
            )

            return enhanced_prompt

        except Exception as e:
            self.logger.warning(f"Could not load knowledge from {knowledge_file}: {e}")
            return base_prompt

    def reset_episode_state(self) -> None:
        """Reset all episode-specific state variables."""
        self.action_counts = Counter()
        self.action_history = []
        self.memory_log_history = []
        self.visited_locations = set()
        self.failed_actions_by_location = {}
        self.episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.previous_zork_score = 0
        self.current_episode_scores = []
        self.current_episode_turns = 0
        self.turn_count = 0  # Initialize turn counter
        self.total_episode_reward = 0  # Initialize total reward for episode
        self.game_map = MapGraph()
        self.current_room_name_for_map = ""  # Track current room for map updates
        self.prev_room_for_prompt_context: Optional[str] = None
        self.action_leading_to_current_room_for_prompt_context: Optional[str] = None
        # Reset movement analyzer for new episode
        self.movement_analyzer.clear_pending_connections()

        # Reset dynamic turn limit for the new episode
        self.max_turns_per_episode = self.base_max_turns_per_episode

        # Performance tracking for dynamic turn limits
        self.critic_scores_history = []  # Store all critic scores for this episode
        self.turn_limit_increases = (
            0  # Track how many times we've increased the limit this episode
        )
        self.last_performance_check_turn = 0  # Track when we last checked performance

    def get_agent_action(
        self,
        game_state_text: str,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
        action_counts: Optional[Counter] = None,
        relevant_memories: Optional[str] = None,
    ) -> str:
        """
        Gets an action from the Agent LM.

        Args:
            game_state_text: Current game state text
            previous_actions_and_responses: List of (action, response) tuples for history
            action_counts: Counter of how many times each action has been tried
            relevant_memories: Formatted string of relevant memories

        Returns:
            The agent's chosen action as a string
        """
        if "o1" in self.agent_model:
            # Use user prompt for o1 models
            messages = [{"role": "user", "content": self.agent_system_prompt}]
        else:
            messages = [{"role": "system", "content": self.agent_system_prompt}]

        # Add history if provided
        if previous_actions_and_responses:
            memory_context = "Here's what you've done so far:\n"

            # Add the most recent actions and responses (last 5-8 is usually sufficient)
            for i, (action, response) in enumerate(previous_actions_and_responses[-8:]):
                memory_context += f"Command: {action}\nResult: {response.strip()}\n\n"

            # Include information about repetitive actions
            if action_counts:
                repeated_actions = [
                    act for act, count in action_counts.items() if count > 2
                ]
                if repeated_actions:
                    memory_context += "\n**CRITICAL WARNING**: You've tried these actions multiple times with limited success: "
                    memory_context += ", ".join(repeated_actions)
                    memory_context += ". According to your instructions, you must AVOID repeating failed actions and try completely different approaches.\n"

            if "o1" in self.agent_model:
                # o1 models use user role for all messages
                messages.append({"role": "user", "content": memory_context})
            else:
                messages.append({"role": "system", "content": memory_context})

        # Combine game state with relevant memories if available
        user_content = game_state_text
        if relevant_memories:
            user_content = f"{user_content}\n\n{relevant_memories}"

        messages.append({"role": "user", "content": user_content})

        try:
            client_args = dict(
                model=self.agent_model,
                messages=messages,
                stop=None,
                temperature=self.temperature_agent,
                max_tokens=self.max_tokens_agent,
                extra_headers={
                    "X-Title": "ZorkGPT",
                },
            )

            response = self.client.chat.completions.create(**client_args)
            action = response.choices[0].message.content.strip()
            # Clean up the action: remove any thinking
            action = re.sub(r"<think>.*?</think>\s*", "", action, flags=re.DOTALL)
            action = re.sub(r"<thinking>.*?</thinking>\s*", "", action, flags=re.DOTALL)
            action = re.sub(
                r"<reflection>.*?</reflection>\s*", "", action, flags=re.DOTALL
            )
            # Basic cleaning: Zork commands are usually lowercase
            action = action.lower()

            # Validate action is not empty
            if not action or action.isspace():
                self.logger.warning(
                    "Agent returned empty action, using 'look' as fallback"
                )
                action = "look"
            return action
        except Exception as e:
            self.logger.error(f"Error getting agent action: {e}")
            return "look"  # Default safe action on error

    def get_robust_action_evaluation(
        self,
        game_state_text: str,
        proposed_action: str,
        action_counts: Optional[Counter] = None,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
        max_attempts: int = 3,
    ) -> CriticResponse:
        """
        Get a robust critic evaluation with confidence scoring and consensus mechanism.

        Args:
            game_state_text: Current game state text
            proposed_action: The action to evaluate
            action_counts: Counter of action frequencies
            previous_actions_and_responses: Recent action history
            max_attempts: Maximum number of evaluation attempts for consensus

        Returns:
            CriticResponse with score, justification, and confidence
        """
        evaluations = []

        for attempt in range(max_attempts):
            evaluation = self.get_critic_evaluation(
                game_state_text,
                proposed_action,
                action_counts,
                previous_actions_and_responses,
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
                self.logger.info(
                    f"High disagreement in critic evaluations (range: {score_range:.2f}), using most confident",
                    extra={
                        "extras": {
                            "event_type": "critic_consensus",
                            "episode_id": self.episode_id,
                            "score_range": score_range,
                            "selected_confidence": best_eval.confidence,
                        }
                    },
                )
                return best_eval
            else:
                # Use average with combined confidence
                avg_score = sum(scores) / len(scores)
                avg_confidence = sum(e.confidence for e in evaluations) / len(
                    evaluations
                )
                return CriticResponse(
                    score=avg_score,
                    justification=f"Consensus evaluation (range: {score_range:.2f})",
                    confidence=avg_confidence,
                )

        return evaluations[0]

    def get_action_with_safeguards(
        self,
        game_state_text: str,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
        action_counts: Optional[Counter] = None,
        relevant_memories: Optional[str] = None,
        max_rejection_attempts: int = 3,
    ) -> Tuple[str, CriticResponse, bool]:
        """
        Get an agent action with critic-based rejection and override safeguards.

        Args:
            game_state_text: Current game state text
            previous_actions_and_responses: Recent action history
            action_counts: Counter of action frequencies
            relevant_memories: Relevant memories for agent prompt
            max_rejection_attempts: Maximum number of rejection attempts

        Returns:
            Tuple of (selected_action, critic_evaluation, was_overridden)
        """
        rejection_attempt = 0
        attempted_actions = []
        rejection_feedback = ""
        was_overridden = False

        # Reset per-turn tracking
        self.action_rejection_system.reset_turn()

        while rejection_attempt < max_rejection_attempts:
            # Get agent action (with accumulated rejection feedback)
            enhanced_memories = relevant_memories
            if rejection_feedback:
                enhanced_memories = (
                    f"{relevant_memories}\n\n--- CRITIC FEEDBACK ---\n{rejection_feedback}"
                    if relevant_memories
                    else f"--- CRITIC FEEDBACK ---\n{rejection_feedback}"
                )

            agent_action = self.get_agent_action(
                game_state_text,
                previous_actions_and_responses,
                action_counts,
                enhanced_memories,
            )
            attempted_actions.append(agent_action)

            # Skip critic evaluation for meta commands
            if agent_action.lower() in [
                "score",
                "inventory",
                "quit",
                "save",
                "restore",
            ]:
                return (
                    agent_action,
                    CriticResponse(
                        score=0.05,
                        justification="Meta-command execution.",
                        confidence=1.0,
                    ),
                    False,
                )

            # Get robust critic evaluation
            evaluation = self.get_robust_action_evaluation(
                game_state_text,
                agent_action,
                action_counts,
                previous_actions_and_responses,
            )

            # Get context for override decisions
            current_location = getattr(
                self, "current_room_name_for_map", "unknown_location"
            )
            failed_actions = set()
            if (
                hasattr(self, "failed_actions_by_location")
                and current_location in self.failed_actions_by_location
            ):
                failed_actions = self.failed_actions_by_location[current_location]

            context = {
                "turns_since_movement": getattr(
                    self.action_rejection_system, "turns_since_movement", 0
                ),
                "recent_critic_scores": self.critic_scores_history[-10:]
                if self.critic_scores_history
                else [],
            }

            # Check for override conditions
            should_override, override_reason = (
                self.action_rejection_system.should_override_rejection(
                    agent_action, current_location, failed_actions, context
                )
            )

            if should_override:
                # Log the critic evaluation even when overridden
                self.logger.info(
                    f"Critic evaluation: Score={evaluation.score:.2f}, Justification='{evaluation.justification}'",
                    extra={
                        "extras": {
                            "event_type": "critic_evaluation",
                            "episode_id": self.episode_id,
                            "critic_score": evaluation.score,
                            "critic_justification": evaluation.justification,
                            "critic_confidence": evaluation.confidence,
                        }
                    },
                )

                self.logger.info(
                    f"Overriding critic rejection: {override_reason}",
                    extra={
                        "extras": {
                            "event_type": "critic_override",
                            "episode_id": self.episode_id,
                            "action": agent_action,
                            "critic_score": evaluation.score,
                            "override_reason": override_reason,
                        }
                    },
                )
                was_overridden = True
                return agent_action, evaluation, was_overridden

            # Adjust rejection threshold based on trust and confidence
            base_threshold = -0.6
            if self.critic_trust_tracker.should_be_conservative():
                base_threshold = -0.8  # More conservative when trust is low

            rejection_threshold = self.critic_trust_tracker.get_rejection_threshold(
                base_threshold
            )

            # Adjust threshold based on confidence
            confidence_adjusted_threshold = rejection_threshold * evaluation.confidence

            if evaluation.score > confidence_adjusted_threshold:
                # Log the accepted critic evaluation
                self.logger.info(
                    f"Critic evaluation: Score={evaluation.score:.2f}, Justification='{evaluation.justification}'",
                    extra={
                        "extras": {
                            "event_type": "critic_evaluation",
                            "episode_id": self.episode_id,
                            "critic_score": evaluation.score,
                            "critic_justification": evaluation.justification,
                            "critic_confidence": evaluation.confidence,
                        }
                    },
                )
                return agent_action, evaluation, was_overridden

            # Action rejected - prepare feedback for next attempt
            self.action_rejection_system.rejected_actions_this_turn.append(agent_action)
            rejection_feedback += f"- REJECTED: '{agent_action}' (Score: {evaluation.score:.2f}, Confidence: {evaluation.confidence:.2f}). {evaluation.justification}\n"

            self.logger.info(
                f"Critic rejected action '{agent_action}' (Score: {evaluation.score:.2f}, Confidence: {evaluation.confidence:.2f})",
                extra={
                    "extras": {
                        "event_type": "critic_rejection",
                        "episode_id": self.episode_id,
                        "action": agent_action,
                        "critic_score": evaluation.score,
                        "confidence": evaluation.confidence,
                        "threshold": confidence_adjusted_threshold,
                        "attempt": rejection_attempt + 1,
                    }
                },
            )

            rejection_attempt += 1

        # Final fallback: pick least bad attempted action or safe default
        if attempted_actions:
            # Re-evaluate all attempted actions and pick the best
            best_action = attempted_actions[0]
            best_evaluation = self.get_critic_evaluation(
                game_state_text,
                best_action,
                action_counts,
                previous_actions_and_responses,
            )

            for action in attempted_actions[1:]:
                eval_result = self.get_critic_evaluation(
                    game_state_text,
                    action,
                    action_counts,
                    previous_actions_and_responses,
                )
                if eval_result.score > best_evaluation.score:
                    best_action = action
                    best_evaluation = eval_result

            self.logger.info(
                f"All actions rejected, using least bad: '{best_action}' (Score: {best_evaluation.score:.2f})",
                extra={
                    "extras": {
                        "event_type": "fallback_action",
                        "episode_id": self.episode_id,
                        "action": best_action,
                        "critic_score": best_evaluation.score,
                    }
                },
            )
            return best_action, best_evaluation, was_overridden
        else:
            # Ultimate fallback
            safe_action = "look"
            safe_evaluation = CriticResponse(
                score=0.1, justification="Safe fallback action.", confidence=1.0
            )
            return safe_action, safe_evaluation, was_overridden

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
        ]

        # Check if the actual action resulted in a parser failure
        action_failed = any(pattern in outcome.lower() for pattern in failure_patterns)

        # If the actual action failed, rejections were likely correct
        # If the actual action succeeded, we can't easily determine if rejections were wrong
        # (since we don't know what the rejected actions would have done)
        if action_failed:
            # Rejections were probably correct
            self.critic_trust_tracker.update_trust(was_rejection_correct=True)
        else:
            # This is more complex - we'll be conservative and not penalize the critic
            # unless we have strong evidence the rejection was wrong
            pass

    def update_movement_tracking(
        self, action: str, location_before: str, location_after: str
    ):
        """Update movement tracking for the action rejection system."""
        # Check if this was a movement action that succeeded
        movement_words = [
            "north",
            "south",
            "east",
            "west",
            "up",
            "down",
            "enter",
            "exit",
            "climb",
            "go",
        ]
        is_movement = any(word in action.lower() for word in movement_words)

        if is_movement and location_before != location_after:
            # Reset movement counter on successful movement
            self.action_rejection_system.turns_since_movement = 0
        else:
            # Increment counter for non-movement or failed movement
            self.action_rejection_system.turns_since_movement += 1

    def get_critic_evaluation(
        self,
        game_state_text: str,
        proposed_action: str,
        action_counts: Optional[Counter] = None,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
    ) -> CriticResponse:
        """
        Gets an evaluation from the Critic LM.

        Args:
            game_state_text: Current game state text
            proposed_action: The action to evaluate
            action_counts: Counter of action frequencies
            previous_actions_and_responses: Recent action history

        Returns:
            CriticResponse with score and justification
        """
        # Prepare context about repetitive actions for the critic
        repetition_context = ""
        if action_counts and action_counts[proposed_action] > 2:
            repetition_context = f"\nThis action '{proposed_action}' has been tried {action_counts[proposed_action]} times already."

        # Add context about the last few actions and responses
        recent_context = ""
        if previous_actions_and_responses and len(previous_actions_and_responses) > 0:
            recent_context = "\nRecent actions and responses:\n"
            for act, resp in previous_actions_and_responses[-3:]:
                recent_context += f"Command: {act}\nResult: {resp.strip()}\n"

        user_prompt = f"""Current Game State:
{game_state_text}

Proposed Agent Action:
{proposed_action}{repetition_context}{recent_context}

Evaluate this action based on your criteria. Respond in JSON format.
"""
        messages = [
            {"role": "system", "content": self.critic_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.critic_model,
                messages=messages,
                temperature=self.temperature_critic,
                max_tokens=self.max_tokens_critic,
                response_format={"type": "json_object"},
                extra_headers={
                    "X-Title": "ZorkGPT",
                },
            )

            response_content = response.choices[0].message.content
            try:
                parsed_data = json.loads(response_content)
                return CriticResponse(**parsed_data)
            except Exception as e:
                self.logger.error(f"Error parsing critic response: {e}")
                self.logger.error(f"Response content: {response_content}")
                return CriticResponse(
                    score=0.0, justification="Critic evaluation error (parsing)."
                )
        except Exception as e:
            self.logger.error(f"Error getting critic evaluation: {e}")
            return CriticResponse(
                score=0.0, justification="Critic evaluation error (API)."
            )

    def get_extracted_info(
        self, game_text_from_zork: str, previous_location: Optional[str] = None
    ) -> Optional[ExtractorResponse]:
        """
        Uses an LLM to extract structured information from Zork's game text with location persistence.

        Args:
            game_text_from_zork: The raw text output from the Zork game.
            previous_location: The previous location name for persistence when no location change occurs.

        Returns:
            ExtractorResponse containing the extracted information, or None if extraction fails.
        """
        if not game_text_from_zork or not game_text_from_zork.strip():
            return ExtractorResponse(
                current_location_name=previous_location or "Unknown (Empty Input)",
                exits=[],
                visible_objects=[],
                visible_characters=[],
                important_messages=["Received empty game text."],
                in_combat=False,
            )

        user_prompt_content = (
            f"Game Text:\n```\n{game_text_from_zork}\n```\n\nJSON Output:\n```json\n"
        )
        user_prompt_content = r"\no_think " + user_prompt_content

        messages = [
            {"role": "system", "content": self.extractor_system_prompt},
            {"role": "user", "content": user_prompt_content},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.info_ext_model,
                messages=messages,
                temperature=self.temperature_info_ext,
                max_tokens=self.max_tokens_info_ext,
                response_format={"type": "json_object"},
                extra_headers={
                    "X-Title": "ZorkGPT",
                },
            )

            response_content = response.choices[0].message.content

            try:
                parsed_data = json.loads(response_content)
                extracted_response = ExtractorResponse(**parsed_data)
                
                # Handle location persistence for "Unknown Location" or similar responses
                if (extracted_response.current_location_name in GENERIC_LOCATION_FALLBACKS 
                    and previous_location 
                    and previous_location not in GENERIC_LOCATION_FALLBACKS):
                    
                    # Log the location persistence
                    self.logger.info(
                        f"Location persistence applied: '{extracted_response.current_location_name}' → '{previous_location}'",
                        extra={
                            "extras": {
                                "event_type": "location_persistence",
                                "episode_id": getattr(self, 'episode_id', 'unknown'),
                                "original_extraction": extracted_response.current_location_name,
                                "persisted_location": previous_location,
                                "game_text": game_text_from_zork[:100] + "..." if len(game_text_from_zork) > 100 else game_text_from_zork
                            }
                        }
                    )
                    
                    # Create new response with persisted location
                    extracted_response = ExtractorResponse(
                        current_location_name=previous_location,
                        exits=extracted_response.exits,
                        visible_objects=extracted_response.visible_objects,
                        visible_characters=extracted_response.visible_characters,
                        important_messages=extracted_response.important_messages,
                        in_combat=extracted_response.in_combat,
                    )
                
                return extracted_response
                
            except Exception as e:
                self.logger.error(f"Error parsing extractor response: {e}")
                self.logger.error(f"Response content: {response_content}")
                
                # Fallback with location persistence
                return ExtractorResponse(
                    current_location_name=previous_location or "Extraction Failed",
                    exits=[],
                    visible_objects=[],
                    visible_characters=[],
                    important_messages=["Extraction failed"],
                    in_combat=False,
                )
                
        except Exception as e:
            self.logger.error(f"Error getting extracted info: {e}")
            
            # Fallback with location persistence
            return ExtractorResponse(
                current_location_name=previous_location or "LLM Request Failed",
                exits=[],
                visible_objects=[],
                visible_characters=[],
                important_messages=["LLM request failed"],
                in_combat=False,
            )

    def get_relevant_memories_for_prompt(
        self,
        current_location_name_from_current_extraction: str,
        memory_log_history: List[ExtractorResponse],
        current_inventory: List[str],
        game_map: MapGraph,
        previous_room_name_for_map_context: Optional[str] = None,
        action_taken_to_current_room: Optional[str] = None,
        in_combat: bool = False,
    ) -> str:
        """
        Generate relevant memories and context for the agent prompt.

        Args:
            current_location_name_from_current_extraction: Current room name
            memory_log_history: History of extracted information
            current_inventory: Current inventory items
            game_map: The game map object
            previous_room_name_for_map_context: Previous room name
            action_taken_to_current_room: Action that led to current room

        Returns:
            Formatted string of relevant memories for the agent
        """
        map_context_str = ""
        if game_map:
            map_info = game_map.get_context_for_prompt(
                current_room_name=current_location_name_from_current_extraction,
                previous_room_name=previous_room_name_for_map_context,
                action_taken_to_current=action_taken_to_current_room,
            )
            if map_info:
                map_context_str = map_info

        other_memory_strings = []

        # Add combat status information
        if in_combat:
            other_memory_strings.append(
                "- COMBAT SITUATION: You are currently in combat or facing an immediate threat! Be prepared to defend yourself or flee."
            )

        if current_inventory:
            other_memory_strings.append(
                f"- You are carrying: {', '.join(current_inventory)}."
            )

        # Add failed actions warning for current location
        if (
            hasattr(self, "failed_actions_by_location")
            and current_location_name_from_current_extraction
            in self.failed_actions_by_location
        ):
            failed_actions = self.failed_actions_by_location[
                current_location_name_from_current_extraction
            ]
            if failed_actions:
                other_memory_strings.append(
                    f"- FAILED ACTIONS in {current_location_name_from_current_extraction}: The following actions have already failed here and should NOT be repeated: {', '.join(sorted(failed_actions))}."
                )

        previous_observations_of_current_room = [
            obs
            for obs in reversed(memory_log_history[:-1])  # Exclude current observation
            if obs.current_location_name
            == current_location_name_from_current_extraction
        ]

        if previous_observations_of_current_room:
            last_relevant_obs = previous_observations_of_current_room[0]
            prev_objects = last_relevant_obs.visible_objects
            if prev_objects:
                other_memory_strings.append(
                    f"- Previously noted objects in {current_location_name_from_current_extraction}: {', '.join(prev_objects)}."
                )

        if memory_log_history:
            # Always use the most recent memory entry
            # This contains the result from the last action taken
            relevant_history_index = -1
            last_turn_info = memory_log_history[relevant_history_index]
            important_msgs = last_turn_info.important_messages
            action_results = [
                msg
                for msg in important_msgs
                if not msg.lower().startswith("you are")
                and not msg.lower().startswith(
                    current_location_name_from_current_extraction.lower()
                )
                and len(msg) < 100
            ]
            if action_results:
                other_memory_strings.append(
                    f"- Last action result/event: {' '.join(action_results)}."
                )

        final_output_parts = []
        if map_context_str and map_context_str.strip():
            content_part = map_context_str.replace(
                "--- Map Information ---", ""
            ).strip()
            if content_part:  # Only add if there's more than just the header
                final_output_parts.append(map_context_str)

        if other_memory_strings:  # other_memory_strings is populated by existing logic
            if final_output_parts:
                final_output_parts.append("\n--- Other Relevant Memories ---")
            else:
                final_output_parts.append("--- Relevant Memories ---")
            final_output_parts.extend(other_memory_strings)

        if not final_output_parts:
            return ""
        return "\n".join(final_output_parts) + "\n"

    def play_episode(self, zork_interface_instance) -> Tuple[List, int]:
        """
        Play a single episode of Zork.

        Args:
            zork_interface_instance: The Zork game interface

        Returns:
            Tuple of (experiences, final_score)
        """
        # Reset state for new episode
        self.reset_episode_state()

        # Generate episode ID (ISO8601 timestamp with second resolution)
        episode_start_time = datetime.now()
        self.episode_id = episode_start_time.strftime("%Y-%m-%dT%H:%M:%S")

        # Log episode start
        self.logger.info(
            "Starting Zork episode...",
            extra={
                "extras": {
                    "event_type": "episode_start",
                    "episode_id": self.episode_id,
                    "agent_model": self.agent_model,
                    "critic_model": self.critic_model,
                    "info_ext_model": self.info_ext_model,
                    "timestamp": datetime.now().isoformat(),
                    # Dynamic turn limit configuration
                    "base_max_turns": self.base_max_turns_per_episode,
                    "absolute_max_turns": self.absolute_max_turns,
                    "turn_limit_increment": self.turn_limit_increment,
                    "performance_check_interval": self.performance_check_interval,
                    "performance_threshold": self.performance_threshold,
                    "min_turns_for_increase": self.min_turns_for_increase,
                }
            },
        )

        current_game_state = zork_interface_instance.start()
        if not current_game_state:
            self.logger.error(
                "Failed to start Zork or get initial state.",
                extra={
                    "extras": {"event_type": "error", "episode_id": self.episode_id}
                },
            )
            return (
                self.experience_tracker.get_experiences(),
                0,
            )  # No score if failed to start

        # Log initial state
        self.logger.info(
            f"INITIAL STATE:\n{current_game_state}\n",
            extra={
                "extras": {
                    "event_type": "initial_state",
                    "episode_id": self.episode_id,
                    "game_state": current_game_state,
                }
            },
        )

        # Extract initial info and store in memory log
        extracted_info = self.get_extracted_info(current_game_state)  # No previous location for initial extraction
        self.memory_log_history.append(extracted_info)

        # Log extracted info
        self.logger.info(
            "Extracted info",
            extra={
                "extras": {
                    "event_type": "extracted_info",
                    "episode_id": self.episode_id,
                    "extracted_info": {
                        "current_location_name": extracted_info.current_location_name,
                        "exits": extracted_info.exits,
                        "visible_objects": extracted_info.visible_objects,
                        "visible_characters": extracted_info.visible_characters,
                        "important_messages": extracted_info.important_messages,
                    },
                }
            },
        )

        game_over = False

        # Initial extraction and map update
        if (
            extracted_info
        ):  # Should always be true due to get_extracted_info's error handling
            self.current_room_name_for_map = extracted_info.current_location_name
            self.game_map.add_room(self.current_room_name_for_map)
            self.game_map.update_room_exits(
                self.current_room_name_for_map, extracted_info.exits
            )
        else:
            self.current_room_name_for_map = "Unknown (Initial Extraction Failed)"
            self.game_map.add_room(self.current_room_name_for_map)

        while (
            not game_over
            and zork_interface_instance.is_running()
            and self.turn_count < self.max_turns_per_episode
        ):
            self.turn_count += 1
            # Log turn start
            self.logger.info(
                f"Turn {self.turn_count}",
                extra={
                    "extras": {
                        "event_type": "turn_start",
                        "episode_id": self.episode_id,
                        "turn": self.turn_count,
                    }
                },
            )

            # Check if we're in combat from the previous turn's extracted info
            in_combat = False
            if self.memory_log_history:
                last_extraction = self.memory_log_history[-1]
                in_combat = getattr(last_extraction, "in_combat", False)

            # Get inventory only if not in combat (to avoid death during inventory checks)
            if not in_combat:
                self.current_inventory, inventory_response = (
                    zork_interface_instance.inventory_with_response()
                )

                # Check if the inventory command caused game over
                if inventory_response:
                    game_over_flag, game_over_reason = (
                        zork_interface_instance.is_game_over(inventory_response)
                    )
                    if game_over_flag:
                        # Log game over from inventory
                        self.logger.info(
                            f"Game over during inventory check: {game_over_reason}",
                            extra={
                                "extras": {
                                    "event_type": "game_over",
                                    "episode_id": self.episode_id,
                                    "reason": f"Inventory check triggered: {game_over_reason}",
                                    "turn": self.turn_count,
                                }
                            },
                        )

                        # Get final score
                        current_zork_score_val, max_zork_score = (
                            zork_interface_instance.score(inventory_response)
                        )
                        self.previous_zork_score = current_zork_score_val

                        # Add death experience
                        reward = -20  # Death penalty
                        self.total_episode_reward += reward

                        experience = self.experience_tracker.add_experience(
                            state=current_game_state,
                            action="inventory",  # The action that caused death
                            reward=reward,
                            next_state=inventory_response,
                            done=True,
                            critic_score=0.0,
                            critic_justification="Death during inventory check",
                            zork_score=self.previous_zork_score,
                        )

                        # Log experience
                        self.logger.debug(
                            "Experience added (death during inventory)",
                            extra={
                                "extras": {
                                    "event_type": "experience",
                                    "episode_id": self.episode_id,
                                    "experience": experience,
                                }
                            },
                        )

                        # Episode ends here
                        break
            else:
                # In combat - skip inventory check and log the decision
                self.logger.info(
                    "Skipping inventory check due to combat situation",
                    extra={
                        "extras": {
                            "event_type": "inventory_skip",
                            "episode_id": self.episode_id,
                            "reason": "In combat - avoiding dangerous inventory check",
                            "turn": self.turn_count,
                        }
                    },
                )

            # Get relevant memories to include in the agent prompt
            relevant_memories = self.get_relevant_memories_for_prompt(
                self.current_room_name_for_map,  # Current location name
                self.memory_log_history,
                self.current_inventory,
                self.game_map,  # Pass the map object
                self.prev_room_for_prompt_context,  # Prev room name for context
                self.action_leading_to_current_room_for_prompt_context,  # Action that led here
                in_combat,  # Combat status
            )

            # Log memory context
            self.logger.info(
                f"Memory context for agent:\n{relevant_memories}",
                extra={
                    "extras": {
                        "event_type": "memory_context",
                        "episode_id": self.episode_id,
                        "memory_context": relevant_memories,
                    }
                },
            )

            # 1. Get action with critic safeguards and rejection handling
            agent_action, critic_evaluation, was_overridden = (
                self.get_action_with_safeguards(
                    current_game_state,  # This is next_game_state from previous turn, or initial state
                    self.action_history,
                    self.action_counts,
                    relevant_memories,
                )
            )

            # Extract values from evaluation
            critic_score_val = critic_evaluation.score
            critic_justification = critic_evaluation.justification
            critic_confidence = getattr(critic_evaluation, "confidence", 0.8)

            # Log final selected action
            self.logger.info(
                f"SELECTED ACTION: {agent_action} (Score: {critic_score_val:.2f}, Confidence: {critic_confidence:.2f}, Override: {was_overridden})",
                extra={
                    "extras": {
                        "event_type": "final_action_selection",
                        "episode_id": self.episode_id,
                        "agent_action": agent_action,
                        "critic_score": critic_score_val,
                        "critic_confidence": critic_confidence,
                        "was_overridden": was_overridden,
                    }
                },
            )

            # Update action count for repetition tracking
            self.action_counts[agent_action] += 1

            # Track critic score for performance evaluation
            self.critic_scores_history.append(critic_score_val)

            # Evaluate performance and potentially increase turn limit
            self.evaluate_performance_and_adjust_turn_limit()

            # 3. Send the chosen action to Zork
            # Store current state for map connection before action is taken
            room_before_action = self.current_room_name_for_map
            action_taken = agent_action

            try:
                next_game_state = zork_interface_instance.send_command(
                    action_taken
                )  # Use action_taken

                # Log Zork response
                self.logger.info(
                    f"ZORK RESPONSE for '{action_taken}':\n{next_game_state}\n",
                    extra={
                        "extras": {
                            "event_type": "zork_response",
                            "episode_id": self.episode_id,
                            "action": action_taken,
                            "zork_response": next_game_state,
                        }
                    },
                )

                # Check if the game has ended based on the response
                game_over_flag, game_over_reason = zork_interface_instance.is_game_over(
                    next_game_state
                )
                if game_over_flag:
                    # Log game over
                    self.logger.info(
                        f"{game_over_reason}",
                        extra={
                            "extras": {
                                "event_type": "game_over",
                                "episode_id": self.episode_id,
                                "reason": game_over_reason,
                            }
                        },
                    )

                    game_over = True
                    current_zork_score_val, max_zork_score = (
                        zork_interface_instance.score(next_game_state)
                    )
                    self.previous_zork_score = current_zork_score_val
                    self.action_history.append(
                        (action_taken, next_game_state)
                    )  # Use action_taken
                    if game_over:
                        if "died" in game_over_reason.lower():
                            reward = -20
                        elif "victory" in game_over_reason.lower():
                            reward = 50
                        else:
                            reward = 0
                        self.total_episode_reward += reward

                        # Add experience and log it
                        experience = self.experience_tracker.add_experience(
                            state=current_game_state,
                            action=action_taken,
                            reward=reward,
                            next_state=next_game_state,
                            done=game_over,
                            critic_score=critic_score_val,
                            critic_justification=critic_justification,
                            zork_score=self.previous_zork_score,
                        )

                        # Log experience
                        self.logger.debug(
                            "Experience added",
                            extra={
                                "extras": {
                                    "event_type": "experience",
                                    "episode_id": self.episode_id,
                                    "experience": experience,
                                }
                            },
                        )
                        continue

                self.action_history.append(
                    (action_taken, next_game_state)
                )  # Use action_taken

                # Store the name of the room we were in *before* this action and new extraction.
                # current_room_name_for_map holds this value from the previous iteration or initial setup.
                room_before_action = self.current_room_name_for_map

                # 2. Extract information using enhanced LLM extractor
                llm_extracted_info = self.get_extracted_info(
                    next_game_state, room_before_action
                )

                # Use the extracted location directly - the enhanced extractor handles persistence internally
                if llm_extracted_info:
                    final_current_room_name = llm_extracted_info.current_location_name
                    source_of_location = "Enhanced LLM"
                else:
                    # Fallback only if extraction completely fails
                    final_current_room_name = room_before_action or "Unknown Location"
                    source_of_location = "Fallback (Extraction Failed)"

                # Update map with the new room and its exits
                self.game_map.add_room(final_current_room_name)
                if llm_extracted_info:
                    self.game_map.update_room_exits(
                        final_current_room_name, llm_extracted_info.exits
                    )

                # Track rejection outcomes and update movement tracking
                rejected_actions = getattr(
                    self.action_rejection_system, "rejected_actions_this_turn", []
                )
                self.track_rejection_outcome(
                    rejected_actions, action_taken, next_game_state
                )
                self.update_movement_tracking(
                    action_taken, room_before_action, final_current_room_name
                )

                # Use shared MovementAnalyzer for consistent movement detection
                movement_context = MovementContext(
                    current_location=final_current_room_name,
                    previous_location=room_before_action,
                    action=action_taken,
                    game_response=next_game_state,
                    turn_number=self.turn_count,
                )

                movement_result = self.movement_analyzer.analyze_movement(
                    movement_context
                )

                # Handle movement result
                if movement_result.connection_created:
                    # Create map connection
                    self.game_map.add_connection(
                        movement_result.from_location,
                        movement_result.action,
                        movement_result.to_location,
                    )

                    # Update prompt context
                    self.prev_room_for_prompt_context = movement_result.from_location
                    self.action_leading_to_current_room_for_prompt_context = (
                        movement_result.action
                    )

                    # Log the connection
                    if movement_result.is_pending:
                        self.logger.info(
                            "Resolved pending connection",
                            extra={
                                "extras": {
                                    "event_type": "pending_connection_resolved",
                                    "episode_id": self.episode_id,
                                    "from_room": movement_result.from_location,
                                    "action": movement_result.action,
                                    "to_room": movement_result.to_location,
                                    "environmental_factors": movement_result.environmental_factors,
                                }
                            },
                        )
                    else:
                        self.logger.debug(
                            "Immediate movement connection created",
                            extra={
                                "extras": {
                                    "event_type": "movement_connection_created",
                                    "episode_id": self.episode_id,
                                    "from_room": movement_result.from_location,
                                    "action": movement_result.action,
                                    "to_room": movement_result.to_location,
                                    "environmental_factors": movement_result.environmental_factors,
                                }
                            },
                        )

                elif movement_result.is_pending:
                    # Log pending connection creation
                    self.logger.info(
                        "Created pending connection",
                        extra={
                            "extras": {
                                "event_type": "pending_connection_created",
                                "episode_id": self.episode_id,
                                "from_room": movement_result.from_location,
                                "action": movement_result.action,
                                "environmental_factors": movement_result.environmental_factors,
                                "game_response": next_game_state[:100] + "..."
                                if len(next_game_state) > 100
                                else next_game_state,
                            }
                        },
                    )

                # Check if room name changed but it wasn't actual movement
                if (
                    final_current_room_name != room_before_action
                    and room_before_action
                    and is_non_movement_command(action_taken)
                ):
                    # Log room name change without movement for debugging
                    self.logger.debug(
                        "Room name changed without movement",
                        extra={
                            "extras": {
                                "event_type": "room_name_change_no_movement",
                                "episode_id": self.episode_id,
                                "from_room": room_before_action,
                                "to_room": final_current_room_name,
                                "action": action_taken,
                                "reason": "Non-movement command detected",
                            }
                        },
                    )

                # Add intermediate actions to pending connections for non-movement actions
                if not movement_result.movement_occurred and action_taken:
                    self.movement_analyzer.add_intermediate_action_to_pending(
                        action_taken, self.turn_count
                    )

                # Cleanup expired pending connections
                expired_connections = self.movement_analyzer.cleanup_expired_pending(
                    self.turn_count
                )
                for expired in expired_connections:
                    self.logger.debug(
                        "Pending connection expired",
                        extra={
                            "extras": {
                                "event_type": "pending_connection_expired",
                                "episode_id": self.episode_id,
                                "pending_connection": expired.to_dict(),
                            }
                        },
                    )

                # Update current_room_name_for_map for the next iteration.
                self.current_room_name_for_map = final_current_room_name

                # Prepare the ExtractorResponse object for memory_log_history
                # This ensures the history reflects the chosen final_current_room_name.
                if llm_extracted_info:  # Check if llm_extracted_info is not None
                    info_for_log = llm_extracted_info.model_copy(
                        update={"current_location_name": final_current_room_name}
                    )
                    self.memory_log_history.append(info_for_log)

                    # Log the extracted info with the source of location
                    original_llm_loc_name = (
                        llm_extracted_info.current_location_name
                        if llm_extracted_info.current_location_name
                        != final_current_room_name
                        else ""
                    )

                    # Log extracted info
                    self.logger.info(
                        "Extracted info",
                        extra={
                            "extras": {
                                "event_type": "extracted_info",
                                "episode_id": self.episode_id,
                                "extracted_info": {
                                    "current_location_name": info_for_log.current_location_name,
                                    "exits": info_for_log.exits,
                                    "visible_objects": info_for_log.visible_objects,
                                    "visible_characters": info_for_log.visible_characters,
                                    "important_messages": info_for_log.important_messages,
                                },
                                "source_of_location": source_of_location,
                                "original_location": original_llm_loc_name,
                            }
                        },
                    )
                else:
                    # This case should be rare due to get_extracted_info's error handling
                    # Log a minimal extraction info object
                    minimal_info = ExtractorResponse(
                        current_location_name=final_current_room_name,
                        exits=[],
                        visible_objects=[],
                        visible_characters=[],
                        important_messages=["Extraction failed"],
                        in_combat=False,
                    )
                    self.memory_log_history.append(minimal_info)

                    # Log minimal extracted info
                    self.logger.info(
                        "Extracted info",
                        extra={
                            "extras": {
                                "event_type": "extracted_info",
                                "episode_id": self.episode_id,
                                "extracted_info": {
                                    "current_location_name": minimal_info.current_location_name,
                                    "exits": minimal_info.exits,
                                    "visible_objects": minimal_info.visible_objects,
                                    "visible_characters": minimal_info.visible_characters,
                                    "important_messages": minimal_info.important_messages,
                                },
                                "source_of_location": source_of_location,
                            }
                        },
                    )

                if not game_over and zork_interface_instance.is_running():
                    current_zork_score_val, max_zork_score = (
                        zork_interface_instance.score()
                    )
                else:
                    current_zork_score_val, max_zork_score = (
                        zork_interface_instance.parse_zork_score(next_game_state)
                    )

            except RuntimeError as e:
                self.logger.error(
                    f"Zork process error: {e}",
                    extra={
                        "extras": {"event_type": "error", "episode_id": self.episode_id}
                    },
                )
                game_over = True
                next_game_state = "Game ended unexpectedly"
                continue

            # 4. Determine Reward & Game Over
            # Base reward from critic
            reward = critic_score_val

            # Apply repetition penalties
            if (
                self.action_counts[agent_action] > 2
                and "already" in next_game_state.lower()
            ):
                repetition_penalty = min(
                    0.2 * (self.action_counts[agent_action] - 2), 0.6
                )
                reward -= repetition_penalty

            # Check for location changes and reward exploration
            if (
                self.current_room_name_for_map
                and self.current_room_name_for_map != room_before_action
            ):  # room_before_action holds previous room name
                if (
                    self.current_room_name_for_map not in self.visited_locations
                ):  # visited_locations still useful
                    exploration_bonus = 0.5
                    reward += exploration_bonus
                    self.visited_locations.add(self.current_room_name_for_map)

            # Check for Zork's internal score changes
            score_change = current_zork_score_val - self.previous_zork_score
            if score_change > 0:
                reward += (
                    score_change * 5
                )  # Significant reward for actual score increase
            self.previous_zork_score = current_zork_score_val

            # Penalty for "I don't understand that" or similar parser failures
            parser_failure_phrases = [
                "i don't understand that.",
                "i don't know the word",
                "you can't see any such thing.",
                "you can't do that.",
                "what?",
                "huh?",
            ]

            # Track failed actions by location
            action_failed = False

            if any(
                phrase in next_game_state.lower() for phrase in parser_failure_phrases
            ):
                reward -= 0.2  # Small penalty
                action_failed = True

            # Track failed actions to prevent repetition
            if action_failed and hasattr(self, "failed_actions_by_location"):
                current_location = self.current_room_name_for_map or "unknown_location"
                if current_location not in self.failed_actions_by_location:
                    self.failed_actions_by_location[current_location] = set()
                self.failed_actions_by_location[current_location].add(
                    agent_action.lower()
                )

            self.total_episode_reward += reward

            # Log reward
            self.logger.info(
                "Reward",
                extra={
                    "extras": {
                        "event_type": "reward",
                        "episode_id": self.episode_id,
                        "reward": reward,
                        "total_reward": self.total_episode_reward,
                    }
                },
            )

            # Store experience for RL and log it
            experience = self.experience_tracker.add_experience(
                state=current_game_state,
                action=agent_action,
                reward=reward,
                next_state=next_game_state,
                done=game_over,
                critic_score=critic_score_val,
                critic_justification=critic_justification,
                zork_score=current_zork_score_val,
            )

            # Log experience
            self.logger.debug(
                "Experience added",
                extra={
                    "extras": {
                        "event_type": "experience",
                        "episode_id": self.episode_id,
                        "experience": experience,
                    }
                },
            )

            current_game_state = next_game_state

            # Update current room name for next iteration
            self.current_room_name_for_map = final_current_room_name

        # Debug: Log why the episode ended
        end_reasons = []
        if game_over:
            end_reasons.append("game_over=True")
        if not zork_interface_instance.is_running():
            end_reasons.append("zork_process_not_running")
        if self.turn_count >= self.max_turns_per_episode:
            end_reasons.append(
                f"max_turns_reached({self.turn_count}>={self.max_turns_per_episode})"
            )

        self.logger.info(
            f"Episode ended. Reasons: {', '.join(end_reasons) if end_reasons else 'unknown'}",
            extra={
                "extras": {
                    "event_type": "episode_end_debug",
                    "episode_id": self.episode_id,
                    "game_over": game_over,
                    "zork_running": zork_interface_instance.is_running(),
                    "turn_count": self.turn_count,
                    "max_turns": self.max_turns_per_episode,
                    "reasons": end_reasons,
                }
            },
        )

        # Log episode end
        self.logger.info(
            "Episode finished",
            extra={
                "extras": {
                    "event_type": "episode_end",
                    "episode_id": self.episode_id,
                    "turn_count": self.turn_count,
                    "zork_score": self.previous_zork_score,
                    "max_score": max_zork_score,
                    "total_reward": self.total_episode_reward,
                    # Dynamic turn limit information
                    "base_max_turns": self.base_max_turns_per_episode,
                    "final_max_turns": self.max_turns_per_episode,
                    "turn_limit_increases": self.turn_limit_increases,
                    "absolute_max_turns": self.absolute_max_turns,
                    # Performance metrics
                    "avg_critic_score": sum(self.critic_scores_history)
                    / len(self.critic_scores_history)
                    if self.critic_scores_history
                    else 0,
                    "total_critic_evaluations": len(self.critic_scores_history),
                }
            },
        )

        # Save experiences to a separate file for RL
        self.experience_tracker.save_experiences(self.experiences_file)

        # Automatically update knowledge base after episode completion
        if self.auto_update_knowledge:
            self._auto_update_knowledge_base()

        return self.experience_tracker.get_experiences(), self.previous_zork_score

    def evaluate_performance_and_adjust_turn_limit(self) -> bool:
        """
        Evaluate recent performance and potentially increase the turn limit.

        Returns:
            True if the turn limit was increased, False otherwise
        """
        # Don't check too early in the episode
        if self.turn_count < self.min_turns_for_increase:
            return False

        # Don't check too frequently
        turns_since_last_check = self.turn_count - self.last_performance_check_turn
        if turns_since_last_check < self.performance_check_interval:
            return False

        # Don't exceed absolute maximum
        if self.max_turns_per_episode >= self.absolute_max_turns:
            return False

        # Need sufficient critic score history to evaluate
        if len(self.critic_scores_history) < self.performance_check_interval:
            return False

        # Calculate recent performance metrics
        recent_scores = self.critic_scores_history[-self.performance_check_interval :]
        avg_recent_critic_score = sum(recent_scores) / len(recent_scores)

        # Additional performance indicators
        recent_rewards = []
        if hasattr(self, "experience_tracker") and self.experience_tracker.experiences:
            recent_experiences = self.experience_tracker.experiences[
                -self.performance_check_interval :
            ]
            recent_rewards = [exp["reward"] for exp in recent_experiences]

        avg_recent_reward = (
            sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        )

        # Count recent exploration (new rooms discovered in recent turns)
        recent_exploration_count = 0
        if len(self.action_history) >= self.performance_check_interval:
            recent_actions = self.action_history[-self.performance_check_interval :]
            # This is a proxy - in a more sophisticated version, we'd track when rooms were first discovered
            movement_actions = [
                action
                for action, _ in recent_actions
                if any(
                    direction in action.lower()
                    for direction in [
                        "north",
                        "south",
                        "east",
                        "west",
                        "up",
                        "down",
                        "enter",
                        "climb",
                        "go",
                    ]
                )
            ]
            recent_exploration_count = len(movement_actions)

        # Determine if performance warrants an increase
        performance_criteria_met = (
            avg_recent_critic_score >= self.performance_threshold
            and avg_recent_reward >= 0.1  # Positive average reward
            and recent_exploration_count >= 2  # Some exploration activity
        )

        self.last_performance_check_turn = self.turn_count

        if performance_criteria_met:
            new_limit = min(
                self.max_turns_per_episode + self.turn_limit_increment,
                self.absolute_max_turns,
            )

            old_limit = self.max_turns_per_episode
            self.max_turns_per_episode = new_limit
            self.turn_limit_increases += 1

            # Log the turn limit increase
            self.logger.info(
                f"Turn limit increased from {old_limit} to {new_limit} due to good performance",
                extra={
                    "extras": {
                        "event_type": "turn_limit_increase",
                        "episode_id": self.episode_id,
                        "turn": self.turn_count,
                        "old_limit": old_limit,
                        "new_limit": new_limit,
                        "avg_critic_score": avg_recent_critic_score,
                        "avg_reward": avg_recent_reward,
                        "exploration_count": recent_exploration_count,
                        "total_increases": self.turn_limit_increases,
                    }
                },
            )

            return True

        return False

    def _auto_update_knowledge_base(self) -> None:
        """
        Automatically update the comprehensive knowledge base after episode completion.
        Uses the integrated knowledge system combining spatial and gameplay insights.
        """
        try:
            self.logger.info(
                "Starting automatic knowledge base update...",
                extra={
                    "extras": {
                        "event_type": "auto_knowledge_update_start",
                        "episode_id": self.episode_id,
                    }
                },
            )

            # Use our integrated knowledge system
            from integrated_knowledge_system import create_integrated_knowledge_base

            output_file = create_integrated_knowledge_base(self.json_log_file)

            # Get file size and content stats for logging
            import os

            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                with open(output_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    char_count = len(content)
            else:
                file_size = 0
                char_count = 0

            self.logger.info(
                f"Comprehensive knowledge base updated successfully",
                extra={
                    "extras": {
                        "event_type": "auto_knowledge_update_success",
                        "episode_id": self.episode_id,
                        "output_file": output_file,
                        "file_size_bytes": file_size,
                        "character_count": char_count,
                    }
                },
            )

        except Exception as e:
            self.logger.warning(
                f"Automatic knowledge base update failed: {e}",
                extra={
                    "extras": {
                        "event_type": "auto_knowledge_update_failed",
                        "episode_id": self.episode_id,
                        "error": str(e),
                    }
                },
            )


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

    def get_rejection_threshold(self, base_threshold: float = -0.6) -> float:
        """Get adjusted rejection threshold based on current trust level."""
        return base_threshold * self.trust_level

    def should_be_conservative(self) -> bool:
        """Return True if we should be more conservative due to low trust."""
        return self.trust_level < 0.5


class ActionRejectionSystem:
    """Handles action rejection with override mechanisms."""

    def __init__(self):
        self.rejected_actions_this_turn = []
        self.turns_since_movement = 0
        self.recent_critic_scores = []

    def should_override_rejection(
        self, action: str, current_location: str, failed_actions: set, context: dict
    ) -> Tuple[bool, str]:
        """Determine if a critic rejection should be overridden."""

        # Override if agent is stuck and needs to explore
        if context.get("turns_since_movement", 0) >= 5:
            return True, "exploration_stuck"

        # Override for completely novel actions in this location
        if action.lower() not in failed_actions:
            # But only if it's a reasonable action type
            reasonable_actions = [
                "north",
                "south",
                "east",
                "west",
                "up",
                "down",
                "enter",
                "exit",
                "climb",
                "examine",
                "take",
                "open",
            ]
            if any(keyword in action.lower() for keyword in reasonable_actions):
                return True, "novel_action"

        # Override if we're in a performance decline
        recent_scores = context.get("recent_critic_scores", [])
        if len(recent_scores) >= 3 and sum(recent_scores[-3:]) / 3 < -0.3:
            return True, "emergency_exploration"

        # Override if all recent action attempts have been rejected
        if len(self.rejected_actions_this_turn) >= 2:
            return True, "consensus_override"

        return False, None

    def reset_turn(self):
        """Reset per-turn tracking."""
        self.rejected_actions_this_turn = []


if __name__ == "__main__":
    # Create ZorkAgent instance with default settings
    agent = ZorkAgent()

    with ZorkInterface(timeout=1.0) as zork_game:  # Increased timeout for stability
        try:
            episode_experiences, final_score = agent.play_episode(zork_game)
            print(f"\nPlayed one episode. Final Zork score: {final_score}")
            print(f"Turns taken: {agent.turn_count}")
            print(f"Base max turns: {agent.base_max_turns_per_episode}")
            print(f"Final max turns: {agent.max_turns_per_episode}")
            print(f"Turn limit increases: {agent.turn_limit_increases}")
            if agent.critic_scores_history:
                avg_critic_score = sum(agent.critic_scores_history) / len(
                    agent.critic_scores_history
                )
                print(f"Average critic score: {avg_critic_score:.3f}")
            print(agent.game_map.render_ascii())
        except RuntimeError as e:
            print(f"ZorkInterface runtime error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("Ensuring Zork process is closed.")
