"""
ObjectiveManager for ZorkGPT orchestration.

Handles the complete lifecycle of objective management:
- Discovery and tracking of objectives through LLM analysis
- Completion detection and marking
- Staleness tracking and cleanup
- Refinement when too many objectives accumulate
"""

from typing import List, Dict, Any
from pydantic import BaseModel

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from knowledge import AdaptiveKnowledgeManager
from shared_utils import create_json_schema, strip_markdown_json_fences, extract_json_from_text


class ObjectiveDiscoveryResponse(BaseModel):
    """Response model for objective discovery."""

    objectives: List[str]
    reasoning: str = ""


class ObjectiveCompletionResponse(BaseModel):
    """Response model for objective completion evaluation."""

    completed_objectives: List[str]
    reasoning: str = ""


class ObjectiveRefinementResponse(BaseModel):
    """Response model for objective refinement."""

    refined_objectives: List[str]
    reasoning: str = ""


class ObjectiveManager(BaseManager):
    """
    Manages the complete lifecycle of game objectives.

    Responsibilities:
    - Discover objectives through LLM analysis of gameplay
    - Track objective completion through game responses
    - Clean up stale objectives that no longer seem relevant
    - Refine objectives when too many accumulate
    """

    def __init__(
        self,
        logger,
        config: GameConfiguration,
        game_state: GameState,
        adaptive_knowledge_manager: AdaptiveKnowledgeManager,
    ):
        super().__init__(logger, config, game_state, "objective_manager")
        self.adaptive_knowledge_manager = adaptive_knowledge_manager

        # Objective refinement tracking
        self.last_objective_refinement_turn = 0

    def reset_episode(self) -> None:
        """Reset objective manager state for a new episode."""
        self.last_objective_refinement_turn = 0
        self.log_debug("Objective manager reset for new episode")

    def process_turn(self, current_agent_reasoning: str = "") -> None:
        """Process objective management for the current turn."""
        # Check if this is a periodic update turn
        if self.should_process_turn():
            self.process_periodic_updates(current_agent_reasoning)

    def process_periodic_updates(self, current_agent_reasoning: str = "") -> None:
        """Process periodic objective updates (checking, refinement, staleness)."""
        # Check if objectives need updating
        self.check_and_update_objectives(current_agent_reasoning)

        # Check for objective refinement if enabled
        self.check_objective_refinement()

        # Check for stale objectives
        self.check_objective_staleness()

    def should_process_turn(self) -> bool:
        """Check if objectives need processing this turn."""
        # Check if it's time for an objective update
        turns_since_update = (
            self.game_state.turn_count - self.game_state.objective_update_turn
        )
        return (
            self.game_state.turn_count > 0
            and turns_since_update >= self.config.objective_update_interval
        )

    def check_and_update_objectives(self, current_agent_reasoning: str = "") -> None:
        """Check if it's time for an objective update and perform it if needed."""
        try:
            self.log_debug(
                f"Objective update check: turn={self.game_state.turn_count}, "
                f"interval={self.config.objective_update_interval}, "
                f"last_update={self.game_state.objective_update_turn}",
                details=f"turn={self.game_state.turn_count}, interval={self.config.objective_update_interval}, last_update={self.game_state.objective_update_turn}",
            )

            # Also log to the structured logger for permanent record
            self.logger.info(
                f"Objective update check: turn={self.game_state.turn_count}, last_update={self.game_state.objective_update_turn}",
                extra={
                    "event_type": "objective_update_check",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "objective_update_turn": self.game_state.objective_update_turn,
                    "current_objectives_count": len(
                        self.game_state.discovered_objectives
                    ),
                },
            )

            # Update objectives every turn, ensuring it's not a duplicate call for the same turn.
            if self.should_process_turn():
                self.log_progress(
                    f"Triggering objective update at turn {self.game_state.turn_count}",
                    stage="objective_update",
                    details=f"Starting objective update at turn {self.game_state.turn_count}",
                )

                self.logger.info(
                    f"Triggering objective update at turn {self.game_state.turn_count}",
                    extra={
                        "event_type": "objective_update_triggered",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                    },
                )
                self._update_discovered_objectives(current_agent_reasoning)
            else:
                self.log_debug(
                    f"Objective update skipped: turn_count={self.game_state.turn_count}, already updated this turn or turn 0",
                    details=f"Skipping objective update at turn {self.game_state.turn_count}",
                )

                self.logger.info(
                    f"Objective update skipped: turn_count={self.game_state.turn_count}, already updated this turn or turn 0",
                    extra={
                        "event_type": "objective_update_skipped",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "objective_update_turn": self.game_state.objective_update_turn,
                        "skip_reason": "turn_0"
                        if self.game_state.turn_count == 0
                        else "already_updated",
                    },
                )
        except Exception as e:
            self.log_error(
                f"Exception in _check_objective_update: {e}",
                details=f"Error during objective update check: {e}",
            )

            self.logger.error(
                f"Exception in _check_objective_update: {e}",
                extra={
                    "event_type": "objective_update_exception",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e),
                },
            )
            raise  # Re-raise to be caught by the outer try-catch

    def _update_discovered_objectives(self, current_agent_reasoning: str = "") -> None:
        """
        Use LLM to analyze recent gameplay and discover/update objectives.

        This maintains discovered objectives between turns while staying LLM-first.
        """
        try:
            self.log_progress(
                f"Updating discovered objectives (turn {self.game_state.turn_count})",
                stage="objective_update",
                details=f"Starting objective discovery update at turn {self.game_state.turn_count}",
            )

            # Log that we're starting the update
            self.logger.info(
                f"Starting objective discovery/update at turn {self.game_state.turn_count}",
                extra={
                    "event_type": "objective_discovery_start",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "current_objectives": self.game_state.discovered_objectives,
                    "current_score": self.game_state.previous_zork_score,
                },
            )

            # Get recent gameplay context for analysis
            recent_memory = (
                self.game_state.memory_log_history[-20:]
                if len(self.game_state.memory_log_history) > 20
                else self.game_state.memory_log_history
            )
            recent_actions = (
                self.game_state.action_history[-10:]
                if len(self.game_state.action_history) > 10
                else self.game_state.action_history
            )

            # Prepare context for LLM analysis
            gameplay_context = self._prepare_objective_analysis_context(
                recent_memory, recent_actions, current_agent_reasoning
            )

            # Create prompt for objective discovery/updating
            prompt = f"""Analyze the recent Zork gameplay to discover and maintain the agent's objectives.

CURRENT DISCOVERED OBJECTIVES:
{self.game_state.discovered_objectives if self.game_state.discovered_objectives else "None discovered yet"}

RECENTLY COMPLETED OBJECTIVES:
{[comp["objective"] for comp in self.game_state.completed_objectives[-5:]] if self.game_state.completed_objectives else "None completed yet"}

RECENT GAMEPLAY CONTEXT:
{gameplay_context}

CURRENT SCORE: {self.game_state.previous_zork_score}
CURRENT LOCATION: {self.game_state.current_room_name_for_map}
CURRENT INVENTORY: {self.game_state.current_inventory}

Based on this gameplay, identify the agent's discovered objectives. Look for:
1. **Score-increasing activities** (these reveal important objectives)
2. **Recurring patterns** in the agent's behavior that suggest goals
3. **Environmental clues** about what the agent should be doing
4. **Items or locations** that seem significant to the agent's progress
5. **Puzzles or challenges** the agent is actively working on

**IMPORTANT - COORDINATE WITH DISCOVERED OBJECTIVES SYSTEM**: 
The agent has a separate real-time objective tracking system that maintains current goals every 20 turns. Your knowledge base should COMPLEMENT this system by focusing on LONG-TERM strategic insights rather than current objectives.

Provide insights in these categories focused on strategic patterns and game world knowledge:

**AVOID (Handled by Objectives System)**:
- Specific current objectives or immediate tactical goals
- Real-time action prioritization advice
- "What should I do next" guidance
- Current situation analysis

Focus on actionable insights that help the agent become better at recognizing opportunities, avoiding dangers, and understanding the game world. Be specific about locations, items, commands, and game mechanics discovered through actual gameplay experience.

Focus on objectives the agent has actually discovered through gameplay patterns or its own novel reasoning, not general Zork knowledge.

**CRITICAL - OUTPUT FORMAT:**
YOU MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON. Do not include thinking tags or reasoning outside the JSON structure.

Required JSON format:
{{
  "objectives": ["objective 1", "objective 2", ...],
  "reasoning": "brief explanation"
}}

Example valid response:
{{
  "objectives": [
    "Find a light source to explore dark areas safely",
    "Acquire the brass lantern from the Living Room",
    "Search for valuable treasures to increase score"
  ],
  "reasoning": "Agent has discovered the importance of light from game feedback and is actively seeking the lantern."
}}"""

            # Get LLM response using adaptive knowledge manager's client
            if not self.adaptive_knowledge_manager:
                raise ValueError(
                    "Adaptive knowledge manager is required for objective completion evaluation"
                )

            messages = [{"role": "user", "content": prompt}]
            model_to_use = self.adaptive_knowledge_manager.analysis_model
            self.log_debug(
                f"Using model: {model_to_use}, prompt length: {len(prompt)} characters",
                details=f"Model: {model_to_use}, prompt length: {len(prompt)}",
            )

            # Log that we're about to make the LLM call
            self.logger.info(
                f"Making LLM call for objective discovery with model {model_to_use}",
                extra={
                    "event_type": "objective_llm_call_start",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "model": model_to_use,
                    "prompt_length": len(prompt),
                },
            )

            try:
                # Use response_format for structured output
                sampling_params = (
                    self.adaptive_knowledge_manager.analysis_sampling.model_dump(
                        exclude_unset=True
                    )
                    if self.adaptive_knowledge_manager
                    else {}
                )
                sampling_params.update(
                    {
                        "temperature": sampling_params.get("temperature", 0.3),
                        "max_tokens": sampling_params.get("max_tokens", 5000),
                        "response_format": create_json_schema(
                            ObjectiveDiscoveryResponse
                        ),
                    }
                )

                response = (
                    self.adaptive_knowledge_manager.client.chat.completions.create(
                        model=model_to_use, messages=messages, name="ObjectiveManager", **sampling_params
                    )
                )

                response_content = response.content
                self.log_debug(
                    f"LLM call successful, response length: {len(response_content)}",
                    details=f"Response type: {type(response)}, content length: {len(response_content)}",
                )

                # Parse JSON response
                try:
                    # Note: Empty response checking is now handled by llm_client with automatic retries
                    # Extract JSON (handles markdown fences, reasoning tags, and embedded JSON)
                    json_content = extract_json_from_text(response_content)
                    response_data = ObjectiveDiscoveryResponse.model_validate_json(
                        json_content
                    )
                    updated_objectives = response_data.objectives
                    reasoning = response_data.reasoning

                    self.log_debug(
                            f"Parsed {len(updated_objectives)} objectives from JSON response",
                            details=f"Reasoning: {reasoning[:100]}...",
                        )
                except Exception as e:
                    self.log_error(
                        f"Failed to parse JSON response: {e}",
                        details=f"Response content (first 500 chars): {response_content[:500]}"
                    )
                    updated_objectives = []

                if updated_objectives:
                    self.game_state.discovered_objectives = updated_objectives
                    self.game_state.objective_update_turn = self.game_state.turn_count

                    self.log_progress(
                        f"Objectives updated: {len(updated_objectives)} objectives discovered",
                        stage="objective_update",
                        details=f"Updated {len(updated_objectives)} objectives: {updated_objectives[:3]}",
                    )

                    # Log the update
                    self.logger.info(
                        "Discovered objectives updated",
                        extra={
                            "event_type": "objectives_updated",
                            "episode_id": self.game_state.episode_id,
                            "turn": self.game_state.turn_count,
                            "objective_count": len(updated_objectives),
                            "objectives": updated_objectives,
                        },
                    )
                else:
                    self.log_warning(
                        "No objectives parsed from LLM response",
                        details="LLM response did not contain parseable objectives",
                    )

                    self.logger.warning(
                        "No objectives parsed from LLM response",
                        extra={
                            "event_type": "objectives_parsing_failed",
                            "episode_id": self.game_state.episode_id,
                            "turn": self.game_state.turn_count,
                            "llm_response": response_content,
                        },
                    )

            except Exception as llm_error:
                self.log_error(
                    f"LLM call failed: {llm_error}",
                    details=f"LLM call failed with error: {llm_error}",
                )

                self.logger.error(
                    f"Objective LLM call failed: {llm_error}",
                    extra={
                        "event_type": "objective_llm_call_failed",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "error": str(llm_error),
                        "model": model_to_use,
                    },
                )

        except Exception as e:
            self.log_error(
                f"Failed to update objectives: {e}",
                details=f"Objective update failed with error: {e}",
            )

            self.logger.error(
                f"Objective update failed: {e}",
                extra={
                    "event_type": "objective_update_failed",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e),
                },
            )

    def _prepare_objective_analysis_context(
        self, recent_memory, recent_actions, current_agent_reasoning
    ) -> str:
        """Prepare gameplay context for objective analysis."""
        context_parts = []

        # Add recent actions and responses
        if recent_actions:
            context_parts.append("RECENT ACTIONS:")
            for action, response in recent_actions[-5:]:  # Last 5 actions
                context_parts.append(f"Action: {action}")
                context_parts.append(
                    f"Response: {response[:200]}..."
                )  # Truncate long responses
                context_parts.append("")

        # Add recent agent reasoning
        if current_agent_reasoning:
            context_parts.append("CURRENT AGENT REASONING:")
            context_parts.append(current_agent_reasoning)
            context_parts.append("")

        # Add recent memory highlights
        if recent_memory:
            context_parts.append("RECENT MEMORY HIGHLIGHTS:")
            for memory in recent_memory[-3:]:  # Last 3 memory entries
                if isinstance(memory, dict):
                    context_parts.append(
                        f"Turn {memory.get('turn', '?')}: {memory.get('summary', str(memory))}"
                    )
            context_parts.append("")

        return "\n".join(context_parts)

    def check_objective_completion(
        self, action_taken: str, game_response: str, extracted_info
    ) -> None:
        """Check if any objectives were completed based on the latest action and response."""
        if not self.game_state.discovered_objectives:
            return

        # Look for completion signals in the game response
        completion_signals = []

        # Score increase signals
        if extracted_info and extracted_info.score is not None:
            if (
                extracted_info.score
                and extracted_info.score > self.game_state.previous_zork_score
            ):
                completion_signals.append(
                    f"Score increased from {self.game_state.previous_zork_score} to {extracted_info.score}"
                )

        # Game response completion signals
        response_lower = game_response.lower()

        # "Taken" responses indicate successful item acquisition
        if response_lower.strip() == "taken.":
            completion_signals.append(
                f"Successfully acquired item via '{action_taken}'"
            )

        # Other positive feedback signals
        if any(
            signal in response_lower
            for signal in [
                "you have earned",
                "points",
                "score",
                "treasure",
                "valuable",
                "well done",
                "excellent",
                "congratulations",
                "success",
                "opened",
                "unlocked",
                "activated",
                "turned on",
                "lit",
            ]
        ):
            completion_signals.append("Positive feedback in game response")

        if completion_signals:
            self._evaluate_objective_completion(action_taken, completion_signals)

    def _evaluate_objective_completion(
        self, action_taken: str, completion_signals: List[str]
    ) -> None:
        """Evaluate which objectives might have been completed."""
        if not self.game_state.discovered_objectives or not completion_signals:
            return

        # Use LLM to determine which objectives were completed
        prompt = f"""Based on the recent action and completion signals, determine which of the current objectives have been completed.

CURRENT OBJECTIVES:
{chr(10).join(f"- {obj}" for obj in self.game_state.discovered_objectives)}

ACTION TAKEN: {action_taken}

COMPLETION SIGNALS:
{chr(10).join(f"- {signal}" for signal in completion_signals)}

CURRENT SCORE: {self.game_state.previous_zork_score}
CURRENT LOCATION: {self.game_state.current_room_name_for_map}

**IMPORTANT ZORK GAME MECHANICS**:
- "Taken." = Successfully picked up an item (positive completion)
- Score increases indicate major objective completion
- Actions like "opened", "unlocked", "activated" often complete specific objectives
- Item acquisition objectives are completed when you successfully "take" the item

Which objectives (if any) have been completed based on this action and the completion signals?

**CRITICAL - OUTPUT FORMAT:**
YOU MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON. Do not include thinking tags or reasoning outside the JSON structure.

Required JSON format:
{{
  "completed_objectives": ["exact objective text 1", "exact objective text 2", ...],
  "reasoning": "brief explanation"
}}

Example valid response:
{{
  "completed_objectives": ["Acquire the brass lantern from the Living Room"],
  "reasoning": "Successfully took the lantern, completing this specific acquisition objective."
}}"""

        if self.adaptive_knowledge_manager and self.adaptive_knowledge_manager.client:
            try:
                response = (
                    self.adaptive_knowledge_manager.client.chat.completions.create(
                        model=self.adaptive_knowledge_manager.analysis_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.adaptive_knowledge_manager.analysis_sampling.temperature,
                        top_p=self.adaptive_knowledge_manager.analysis_sampling.top_p,
                        top_k=self.adaptive_knowledge_manager.analysis_sampling.top_k,
                        min_p=self.adaptive_knowledge_manager.analysis_sampling.min_p,
                        max_tokens=self.adaptive_knowledge_manager.analysis_sampling.max_tokens,
                        name="ObjectiveManager",
                        response_format=create_json_schema(ObjectiveCompletionResponse),
                    )
                )

                response_content = response.content

                # Parse JSON response
                try:
                    # Note: Empty response checking is now handled by llm_client with automatic retries
                    # Extract JSON (handles markdown fences, reasoning tags, and embedded JSON)
                    json_content = extract_json_from_text(response_content)
                    response_data = ObjectiveCompletionResponse.model_validate_json(
                        json_content
                    )
                    completed_objectives = response_data.completed_objectives
                    # Filter to only include objectives that exist in discovered_objectives
                    completed_objectives = [
                        obj
                        for obj in completed_objectives
                        if obj in self.game_state.discovered_objectives
                    ]
                    if completed_objectives:
                        self._mark_objectives_complete(
                            completed_objectives, action_taken, completion_signals
                        )
                except Exception as e:
                    self.log_error(
                        f"Failed to parse completion JSON: {e}",
                        details=f"Response content (first 500 chars): {response_content[:500] if response_content else 'None'}"
                    )

            except Exception as e:
                self.log_error(f"Failed to evaluate objective completion: {e}")

    def _mark_objectives_complete(
        self,
        completed_objectives: List[str],
        action_taken: str,
        completion_signals: List[str],
    ) -> None:
        """Mark objectives as completed and track the completion."""
        for objective in completed_objectives:
            if objective in self.game_state.discovered_objectives:
                # Remove from discovered objectives
                self.game_state.discovered_objectives.remove(objective)

                # Add to completed objectives with metadata
                completion_record = {
                    "objective": objective,
                    "completed_turn": self.game_state.turn_count,
                    "completion_action": action_taken,
                    "completion_signals": completion_signals,
                    "completion_location": self.game_state.current_room_name_for_map,
                    "completion_score": self.game_state.previous_zork_score,
                }
                self.game_state.completed_objectives.append(completion_record)

                self.log_progress(
                    f"Objective completed: {objective}",
                    stage="objective_completion",
                    details=f"Completed objective: {objective}",
                )

                # Log the completion
                self.logger.info(
                    f"Objective completed: {objective}",
                    extra={
                        "event_type": "objective_completed",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "objective": objective,
                        "completion_action": action_taken,
                        "completion_location": self.game_state.current_room_name_for_map,
                        "completion_score": self.game_state.previous_zork_score,
                    },
                )

    def check_objective_staleness(self) -> None:
        """Check for stale objectives and remove them if they haven't seen progress."""
        current_location = self.game_state.current_room_name_for_map
        current_score = self.game_state.previous_zork_score

        # Initialize tracking if this is the first call
        if self.game_state.last_location_for_staleness is None:
            self.game_state.last_location_for_staleness = current_location
            self.game_state.last_score_for_staleness = current_score
            self.log_debug("Initializing staleness tracking")
            return

        # Check if we made any progress this turn
        made_progress = (
            current_location != self.game_state.last_location_for_staleness
            or current_score > self.game_state.last_score_for_staleness
        )

        # Log progress status
        self.log_debug(
            f"Staleness check - Progress: {made_progress}, "
            f"Location: {self.game_state.last_location_for_staleness} -> {current_location}, "
            f"Score: {self.game_state.last_score_for_staleness} -> {current_score}"
        )

        # Track staleness for each objective
        objectives_to_remove = []
        for objective in self.game_state.discovered_objectives:
            # Initialize staleness tracking if needed
            if objective not in self.game_state.objective_staleness_tracker:
                self.game_state.objective_staleness_tracker[objective] = 0
                self.log_debug(
                    f"Initializing staleness tracking for objective: {objective[:50]}..."
                )

            if made_progress:
                # Reset staleness counter if we made any progress
                if self.game_state.objective_staleness_tracker[objective] > 0:
                    self.log_debug(
                        f"Resetting staleness for objective due to progress: {objective[:50]}..."
                    )
                self.game_state.objective_staleness_tracker[objective] = 0
            else:
                # Increment staleness counter
                self.game_state.objective_staleness_tracker[objective] += 1
                staleness_count = self.game_state.objective_staleness_tracker[objective]

                # Log increasing staleness
                if staleness_count % 10 == 0:  # Log every 10 turns
                    self.log_debug(
                        f"Objective staleness increasing: {objective[:50]}... (count: {staleness_count})"
                    )

                # Mark for removal if stale for too long (30+ turns without progress)
                if staleness_count >= 30:
                    objectives_to_remove.append(objective)

                    self.log_progress(
                        f"Removing stale objective (30+ turns without progress): {objective[:50]}...",
                        stage="objective_cleanup",
                        details=f"Removed after {staleness_count} turns without progress",
                    )

                    self.logger.info(
                        f"Objective removed due to staleness: {objective}",
                        extra={
                            "event_type": "objective_removed_stale",
                            "episode_id": self.game_state.episode_id,
                            "turn": self.game_state.turn_count,
                            "objective": objective,
                            "staleness_count": staleness_count,
                        },
                    )

        # Remove stale objectives
        for objective in objectives_to_remove:
            self.game_state.discovered_objectives.remove(objective)
            del self.game_state.objective_staleness_tracker[objective]

        # Update tracking for next check
        self.game_state.last_location_for_staleness = current_location
        self.game_state.last_score_for_staleness = current_score

    def check_objective_refinement(self) -> None:
        """Check if objectives need refinement due to too many accumulating."""
        if not self.config.enable_objective_refinement:
            self.log_debug("Objective refinement disabled in config")
            return

        current_objective_count = len(self.game_state.discovered_objectives)
        turns_since_refinement = (
            self.game_state.turn_count - self.last_objective_refinement_turn
        )

        # Log current state
        self.log_debug(
            f"Refinement check - Objectives: {current_objective_count}, "
            f"Max before forced: {self.config.max_objectives_before_forced_refinement}, "
            f"Turns since last: {turns_since_refinement}, "
            f"Interval: {self.config.objective_refinement_interval}"
        )

        # Check if we have too many objectives or enough time has passed
        force_due_to_count = (
            current_objective_count
            >= self.config.max_objectives_before_forced_refinement
        )
        force_due_to_time = (
            turns_since_refinement >= self.config.objective_refinement_interval
        )

        should_refine = force_due_to_count or force_due_to_time

        if (
            should_refine
            and current_objective_count > self.config.refined_objectives_target_count
        ):
            self.log_progress(
                f"Triggering objective refinement - Count: {current_objective_count}, "
                f"Force due to count: {force_due_to_count}, Force due to time: {force_due_to_time}",
                stage="objective_refinement",
                details=f"Refining {current_objective_count} objectives down to {self.config.refined_objectives_target_count}",
            )

            self.logger.info(
                "Objective refinement triggered",
                extra={
                    "event_type": "objective_refinement_triggered",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "current_count": current_objective_count,
                    "target_count": self.config.refined_objectives_target_count,
                    "forced_by_count": force_due_to_count,
                    "forced_by_time": force_due_to_time,
                    "turns_since_last": turns_since_refinement,
                },
            )

            self._refine_discovered_objectives()
        else:
            if not should_refine:
                self.log_debug("Refinement not needed - thresholds not met")
            elif current_objective_count <= self.config.refined_objectives_target_count:
                self.log_debug(
                    f"Refinement not needed - already at or below target count "
                    f"({current_objective_count} <= {self.config.refined_objectives_target_count})"
                )

    def _refine_discovered_objectives(self) -> None:
        """Refine objectives by keeping only the most important ones."""
        if (
            len(self.game_state.discovered_objectives)
            <= self.config.refined_objectives_target_count
        ):
            return

        # Use LLM to refine objectives
        prompt = f"""Refine this list of objectives by selecting the {self.config.refined_objectives_target_count} most important and achievable ones.

CURRENT OBJECTIVES ({len(self.game_state.discovered_objectives)} total):
{chr(10).join(f"- {obj}" for obj in self.game_state.discovered_objectives)}

CURRENT CONTEXT:
- Score: {self.game_state.previous_zork_score}
- Location: {self.game_state.current_room_name_for_map}
- Turn: {self.game_state.turn_count}

Select the {self.config.refined_objectives_target_count} most important objectives that are:
1. Most likely to be achievable given current context
2. Most likely to lead to score increases
3. Most specific and actionable
4. Not redundant with each other

**CRITICAL - OUTPUT FORMAT:**
YOU MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON. Do not include thinking tags or reasoning outside the JSON structure.

Required JSON format:
{{{{
  "refined_objectives": ["objective 1", "objective 2", ...],
  "reasoning": "brief explanation"
}}}}

Example valid response:
{{{{
  "refined_objectives": [
    "Find and light the brass lantern to explore dark areas",
    "Locate and collect valuable treasures for points",
    "Solve the troll puzzle to access new areas"
  ],
  "reasoning": "Selected objectives that are most achievable with current inventory and most likely to increase score."
}}}}"""

        if self.adaptive_knowledge_manager and self.adaptive_knowledge_manager.client:
            try:
                response = (
                    self.adaptive_knowledge_manager.client.chat.completions.create(
                        model=self.adaptive_knowledge_manager.analysis_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=1000,
                        name="ObjectiveManager",
                        response_format=create_json_schema(ObjectiveRefinementResponse),
                    )
                )

                response_content = response.content

                # Parse JSON response
                try:
                    # Note: Empty response checking is now handled by llm_client with automatic retries
                    # Extract JSON (handles markdown fences, reasoning tags, and embedded JSON)
                    json_content = extract_json_from_text(response_content)
                    response_data = ObjectiveRefinementResponse.model_validate_json(
                        json_content
                    )
                    refined_objectives = response_data.refined_objectives
                    if refined_objectives and len(refined_objectives) <= len(
                        self.game_state.discovered_objectives
                    ):
                        old_count = len(self.game_state.discovered_objectives)
                        self.game_state.discovered_objectives = refined_objectives[
                            : self.config.refined_objectives_target_count
                        ]
                        self.last_objective_refinement_turn = self.game_state.turn_count

                        self.log_progress(
                            f"Objectives refined: {old_count} -> {len(self.game_state.discovered_objectives)}",
                            stage="objective_refinement",
                            details=f"Refined from {old_count} to {len(self.game_state.discovered_objectives)} objectives",
                        )
                except Exception as e:
                    self.log_error(
                        f"Failed to parse refinement JSON: {e}",
                        details=f"Response content (first 500 chars): {response_content[:500] if response_content else 'None'}"
                    )

            except Exception as e:
                self.log_error(f"Failed to refine objectives: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current objective manager status."""
        status = super().get_status()
        status.update(
            {
                "discovered_objectives_count": len(
                    self.game_state.discovered_objectives
                ),
                "completed_objectives_count": len(self.game_state.completed_objectives),
                "last_objective_update_turn": self.game_state.objective_update_turn,
                "last_refinement_turn": self.last_objective_refinement_turn,
                "staleness_tracker_size": len(
                    self.game_state.objective_staleness_tracker
                ),
            }
        )
        return status
