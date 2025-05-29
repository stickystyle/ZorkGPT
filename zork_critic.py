"""
ZorkCritic module for evaluating actions and managing critic trust.
"""

import json
import re
from typing import Optional, List, Tuple, Any, Dict, Type
from pydantic import BaseModel
from collections import Counter
from llm_client import LLMClientWrapper
from config import get_config, get_client_api_key

# Import create_json_schema from shared utilities
from shared_utils import create_json_schema


class CriticResponse(BaseModel):
    score: float
    justification: str
    confidence: float = 0.8  # Default confidence level


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
    """Handles action rejection with override mechanisms."""

    def __init__(self):
        self.rejected_actions_this_turn = []
        self.turns_since_movement = 0
        self.recent_critic_scores = []

    def should_override_rejection(
        self, action: str, current_location: str, failed_actions: set, context: dict
    ) -> Tuple[bool, str]:
        """Determine if a critic rejection should be overridden."""

        # Override if agent is stuck and needs to explore (increased threshold)
        if context.get("turns_since_movement", 0) >= 8:  # Increased from 5
            return True, "exploration_stuck"

        # Override for novel actions (MUCH more restrictive)
        if action.lower() not in failed_actions:
            # NEW: Only override if critic confidence is low (if available)
            critic_confidence = context.get("critic_confidence", 0.5)  # Default to low confidence
            if critic_confidence >= 0.8:  # Don't override confident rejections
                return False, None
                
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
        self.max_tokens = max_tokens if max_tokens is not None else config.critic_sampling.max_tokens
        self.temperature = temperature if temperature is not None else config.critic_sampling.temperature
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
                base_url=config.llm.client_base_url,
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
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== {prefix.upper()} PROMPT #{self.prompt_counter} ===\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"Max Tokens: {self.max_tokens}\n")
                f.write(f"Episode ID: {self.episode_id}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, message in enumerate(messages):
                    f.write(f"--- MESSAGE {i+1} ({message['role'].upper()}) ---\n")
                    f.write(message['content'])
                    f.write("\n\n")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to log prompt to {filename}: {e}")

    def _load_system_prompt(self) -> None:
        """Load critic system prompt from markdown file."""
        try:
            with open("critic.md") as fh:
                self.system_prompt = fh.read()
        except FileNotFoundError as e:
            if self.logger:
                self.logger.error(f"Failed to load critic prompt file: {e}")
            raise

    def evaluate_action(
        self,
        game_state_text: str,
        proposed_action: str,
        action_counts: Optional[Counter] = None,
        previous_actions_and_responses: Optional[List[Tuple[str, str]]] = None,
    ) -> CriticResponse:
        """
        Get an evaluation from the Critic LM.

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

Evaluate this action based on your criteria. Respond with ONLY a JSON object in this exact format:
{{"score": 0.0, "justification": "Your justification here", "confidence": 0.8}}
"""
        messages = [
            {"role": "system", "content": self.system_prompt},
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
                response_format=create_json_schema(CriticResponse),
            )

            response_content = response.content
            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in response_content:
                    # Find the JSON content between ```json and ```
                    start_marker = "```json"
                    end_marker = "```"
                    start_idx = response_content.find(start_marker) + len(start_marker)
                    end_idx = response_content.find(end_marker, start_idx)
                    if end_idx != -1:
                        json_content = response_content[start_idx:end_idx].strip()
                    else:
                        json_content = response_content[start_idx:].strip()
                elif "```" in response_content:
                    # Handle generic code blocks
                    start_idx = response_content.find("```") + 3
                    end_idx = response_content.find("```", start_idx)
                    if end_idx != -1:
                        json_content = response_content[start_idx:end_idx].strip()
                    else:
                        json_content = response_content[start_idx:].strip()
                else:
                    json_content = response_content.strip()

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
                    self.logger.error(f"Error parsing critic response: {e}")
                    self.logger.error(f"Response content: {response_content}")
                return CriticResponse(
                    score=0.0, justification="Critic evaluation error (parsing)."
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting critic evaluation: {e}")
            return CriticResponse(
                score=0.0, justification="Critic evaluation error (API)."
            )

    def get_robust_evaluation(
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
            evaluation = self.evaluate_action(
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
                if self.logger:
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

    def update_episode_id(self, episode_id: str) -> None:
        """Update the episode ID for logging purposes."""
        self.episode_id = episode_id
