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
from pathlib import Path
from collections import deque
from contextlib import nullcontext

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from knowledge import AdaptiveKnowledgeManager
from shared_utils import create_json_schema, strip_markdown_json_fences, extract_json_from_text, estimate_tokens


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
        map_manager=None,  # NEW: Optional MapManager for spatial context
        simple_memory=None,  # NEW: Optional SimpleMemoryManager for memory access
        langfuse_client=None,  # NEW: Optional Langfuse client for span tracing
    ):
        super().__init__(logger, config, game_state, "objective_manager")
        self.adaptive_knowledge_manager = adaptive_knowledge_manager
        self.map_manager = map_manager
        self.simple_memory = simple_memory
        self.langfuse_client = langfuse_client

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

    def bootstrap_initial_objectives(self) -> None:
        """
        Bootstrap initial objectives before turn 1 from memories and knowledgebase.

        This method bypasses normal turn checks and is meant to be called once
        during episode initialization (at turn 0) to give the agent starting objectives
        based on cross-episode knowledge.
        """
        self.log_info(
            f"Bootstrapping initial objectives at turn {self.game_state.turn_count}",
            turn=self.game_state.turn_count,
        )

        # Directly call objective update, bypassing turn check
        # No agent reasoning available yet at turn 0
        self._update_discovered_objectives(current_agent_reasoning="")

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
        # Prepare span metadata for tracing
        should_process = self.should_process_turn()
        current_objective_count = len(self.game_state.discovered_objectives)
        turns_since_update = self.game_state.turn_count - self.game_state.objective_update_turn

        # Determine exit reason for tracing
        exit_reason = None
        if self.game_state.turn_count == 0:
            exit_reason = "turn_0"
        elif not should_process:
            exit_reason = "already_updated"
        else:
            exit_reason = "processing"

        # Create span for tracing (visible even if we exit early)
        if self.langfuse_client:
            span_metadata = {
                "turn_number": self.game_state.turn_count,
                "location_id": self.game_state.current_room_id,
                "location_name": self.game_state.current_room_name_for_map,
                "current_objective_count": current_objective_count,
                "objective_update_interval": self.config.objective_update_interval,
                "last_objective_update_turn": self.game_state.objective_update_turn,
                "turns_since_update": turns_since_update,
                "should_process": should_process,
                "exit_reason": exit_reason,
            }
            span_input = {
                "current_objective_count": current_objective_count,
                "turn_info": f"Turn {self.game_state.turn_count}, interval {self.config.objective_update_interval}, last update {self.game_state.objective_update_turn}",
                "reasoning_preview": current_agent_reasoning[:200] if current_agent_reasoning else "",
            }
            tracing_context = self.langfuse_client.start_as_current_span(
                name="objective-discovery-check",
                input=span_input,
                metadata=span_metadata,
            )
        else:
            tracing_context = nullcontext()

        with tracing_context as span:
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
                if should_process:
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
                # Update span metadata to reflect exception (if span exists)
                if span and hasattr(span, 'update'):
                    span.update(metadata={"exit_reason": "exception"})

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

            # Gather rich context from all available sources
            knowledge_content = self._get_full_knowledge()
            memories_content = self._get_all_memories_by_distance(self.game_state.current_room_id)
            map_context = self._get_map_context()
            gameplay_context = self._get_gameplay_context()

            # Log token counts for each context section
            knowledge_tokens = estimate_tokens(knowledge_content)
            memories_tokens = estimate_tokens(memories_content)
            map_tokens = estimate_tokens(map_context)
            gameplay_tokens = estimate_tokens(gameplay_context)
            total_context_tokens = knowledge_tokens + memories_tokens + map_tokens + gameplay_tokens

            self.log_debug(
                f"Objective context token breakdown: knowledge={knowledge_tokens}, "
                f"memories={memories_tokens}, map={map_tokens}, gameplay={gameplay_tokens}, "
                f"total={total_context_tokens}",
                details=f"Total context tokens: {total_context_tokens}"
            )

            self.logger.info(
                f"Objective discovery context prepared: {total_context_tokens} tokens",
                extra={
                    "event_type": "objective_context_prepared",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "knowledge_tokens": knowledge_tokens,
                    "memories_tokens": memories_tokens,
                    "map_tokens": map_tokens,
                    "gameplay_tokens": gameplay_tokens,
                    "total_context_tokens": total_context_tokens,
                }
            )

            # Create prompt for objective discovery/updating with enhanced context
            prompt = f"""Analyze recent gameplay and available knowledge to discover objectives.

=== STRATEGIC KNOWLEDGE (General Wisdom) ===
{knowledge_content}

=== LOCATION-SPECIFIC MEMORIES ===
{memories_content}

=== MAP & ROUTING INFORMATION ===
{map_context}

=== RECENT GAMEPLAY ===
{gameplay_context}

CURRENT STATE:
- Score: {self.game_state.previous_zork_score}
- Location: {self.game_state.current_room_name} (ID: {self.game_state.current_room_id})
- Inventory: {self.game_state.current_inventory}

Based on ALL of this context, identify objectives that:

1. **Align with strategic knowledge**
   - Avoid known dangers mentioned in knowledge base
   - Pursue known high-value goals (treasures, puzzle solutions)
   - Follow resource priority guidance (lantern > axe > sack)
   - Apply learned command patterns and syntax

2. **Leverage location-specific memories**
   - Use known procedures from current or adjacent locations
   - Example: "Memory shows window entry sequence at Location 79 → create objective to use it"
   - Build on previous discoveries rather than rediscovering

3. **Address exploration opportunities**
   - Prioritize unexplored exits shown in map
   - Investigate adjacent rooms with interesting memories
   - Example: "Adjacent Location 81 has lantern memory → objective to go north and investigate"

4. **Build on recent gameplay patterns**
   - Continue successful strategies from recent actions
   - Learn from recent failures or obstacles

**Good Objective Examples:**
✅ "Use window entry procedure from Location 79 memory (open window → enter window) to access Location 62 (Kitchen)"
✅ "Avoid troll encounter at Location 152 (knowledge warns: requires specific item or combat)"
✅ "Explore north to Location 81 (adjacent room with unexplored exits, lantern mentioned in memory)"
✅ "Secure brass lantern before dark areas (knowledge priority: light source critical)"

**Bad Objective Examples:**
❌ "Explore the house" (too vague, no location IDs)
❌ "Get items" (no specifics, doesn't leverage context)
❌ "Try random actions" (ignores knowledge and memories)

**IMPORTANT**:
- Use location IDs when referencing locations (e.g., "Location 79", not just "behind house")
- Reference specific memories or knowledge when creating objectives
- Prioritize objectives that leverage cross-episode learning

**RESPONSE LENGTH CONSTRAINTS:**
- Keep reasoning field under 500 characters (2-3 sentences maximum)
- Each objective should be 1 clear sentence
- Total response should be under 1000 tokens
- Focus on quality over quantity (3-5 objectives is ideal)

**CRITICAL - OUTPUT FORMAT:**
YOU MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.

Required JSON format:
{{
  "objectives": ["objective 1", "objective 2", ...],
  "reasoning": "brief explanation of how objectives align with knowledge/memories/map"
}}

Example valid response:
{{
  "objectives": [
    "Use window entry procedure from Location 79 memory to access Location 62 (Kitchen)",
    "Avoid troll at Location 152 per knowledge base warning",
    "Explore north to Location 81 to investigate lantern memory"
  ],
  "reasoning": "Objectives leverage window procedure memory (cross-episode learning), avoid known danger (strategic knowledge), and pursue exploration opportunity with useful item (map + memory combination)."
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
                # analysis_sampling is already a dict after config migration
                sampling_params = (
                    dict(self.adaptive_knowledge_manager.analysis_sampling)
                    if self.adaptive_knowledge_manager
                    else {}
                )
                # Token budget: ~2K prompt + ~1.5K completion = ~3.5K total
                # Objectives: 3-5 × ~125 tokens = ~500 tokens
                # Reasoning: ~75 tokens (brief, 2-3 sentences)
                # Total: ~600 tokens (well within 1500 limit)
                sampling_params.update(
                    {
                        "temperature": sampling_params.get("temperature", 0.3),
                        "max_tokens": sampling_params.get("max_tokens", 1500),
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

                    # Log the update with reasoning
                    self.logger.info(
                        "Discovered objectives updated",
                        extra={
                            "event_type": "objectives_updated",
                            "episode_id": self.game_state.episode_id,
                            "turn": self.game_state.turn_count,
                            "objective_count": len(updated_objectives),
                            "objectives": updated_objectives,
                            "reasoning": reasoning,  # Include LLM reasoning
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

    def add_agent_objective(self, objective_text: str) -> None:
        """
        Add an objective directly declared by the agent during reasoning.

        This provides a fast path for agent self-direction without waiting
        for periodic LLM-driven discovery.

        Args:
            objective_text: The objective description from agent response
        """
        # Basic validation
        if not objective_text or len(objective_text.strip()) == 0:
            self.log_warning("Agent provided empty objective, ignoring")
            return

        # Check if already exists (case-insensitive deduplication)
        for existing_obj in self.game_state.discovered_objectives:
            if existing_obj.lower() == objective_text.lower():
                self.log_info(f"Objective already exists: {objective_text}")
                return

        # Add to objectives list (simple string)
        self.game_state.discovered_objectives.append(objective_text)
        self.log_info(f"Added agent-declared objective: {objective_text}")

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
        """
        Check if any objectives were completed using LLM-based validation.

        Implements every-turn LLM checking with enhanced context (memories + action history).
        Replaces keyword-based gating to follow LLM-first architectural principle.

        Args:
            action_taken: Action that was executed
            game_response: Game's response text
            extracted_info: Structured info from extractor (score, location, etc.)
        """
        # Prepare span metadata for tracing
        has_objectives = bool(self.game_state.discovered_objectives)
        is_enabled = self.config.enable_objective_completion_llm_check
        is_correct_interval = (self.game_state.turn_count % self.config.completion_check_interval == 0)

        # Determine exit reason for tracing
        exit_reason = None
        if not has_objectives:
            exit_reason = "no_objectives"
        elif not is_enabled:
            exit_reason = "disabled_in_config"
        elif not is_correct_interval:
            exit_reason = "not_correct_interval"

        # Create span for tracing (visible even if we exit early)
        if self.langfuse_client:
            span_metadata = {
                "turn_number": self.game_state.turn_count,
                "location_id": self.game_state.current_room_id,
                "location_name": self.game_state.current_room_name_for_map,
                "has_objectives": has_objectives,
                "objective_count": len(self.game_state.discovered_objectives) if has_objectives else 0,
                "is_enabled": is_enabled,
                "is_correct_interval": is_correct_interval,
                "exit_reason": exit_reason,
            }
            span_input = {
                "action": action_taken,
                "response_preview": game_response[:200] if game_response else "",
            }
            tracing_context = self.langfuse_client.start_as_current_span(
                name="objective-completion-check",
                input=span_input,
                metadata=span_metadata,
            )
        else:
            tracing_context = nullcontext()

        with tracing_context:
            # Early exit: No objectives to check
            if not has_objectives:
                return

            # Early exit: Feature disabled in config
            if not is_enabled:
                return

            # Early exit: Not the right turn interval
            if not is_correct_interval:
                return

            # Call LLM evaluation with full parameters
            self._evaluate_objective_completion(action_taken, game_response, extracted_info)

    def _evaluate_objective_completion(
        self, action_taken: str, game_response: str, extracted_info
    ) -> None:
        """
        Evaluate which objectives might have been completed using enhanced LLM context.

        Enhanced context includes:
        - Recent action history (last N turns)
        - Location-specific memories (if enabled and available)
        - Current game state (score, location, inventory)
        - Before/after state comparison (score change, location change, inventory change)

        Args:
            action_taken: Action that was executed
            game_response: Game's response text
            extracted_info: Structured info from extractor (score, location, etc.)
        """
        if not self.game_state.discovered_objectives:
            return

        # Gather enhanced context
        action_history = self._get_recent_action_history()

        # Get location-specific memories if enabled
        memory_context = ""
        if self.config.completion_include_memories and self.game_state.current_room_id:
            memory_context = self._get_completion_memory_context(
                self.game_state.current_room_id
            )

        # Calculate state changes
        score_change = ""
        if extracted_info and extracted_info.score is not None:
            if extracted_info.score > self.game_state.previous_zork_score:
                score_change = f"Score increased from {self.game_state.previous_zork_score} to {extracted_info.score} (+{extracted_info.score - self.game_state.previous_zork_score})"
            else:
                score_change = f"Score unchanged: {self.game_state.previous_zork_score}"

        # Build enhanced prompt
        prompt = f"""Based on the recent gameplay context, determine which of the current objectives have been completed.

CURRENT OBJECTIVES:
{chr(10).join(f"- {obj}" for obj in self.game_state.discovered_objectives)}

## Recent Action History
{action_history if action_history else "No recent action history available."}

## Latest Action and Response
Action: {action_taken}
Response: {game_response}

## Current Game State
Location: {self.game_state.current_room_name_for_map} (ID: {self.game_state.current_room_id})
Current Score: {self.game_state.previous_zork_score}
Inventory: {", ".join(self.game_state.current_inventory) if self.game_state.current_inventory else "Empty"}

## State Changes
{score_change if score_change else "No score change detected."}

## Location-Specific Memories
{memory_context if memory_context else "No memories available for this location."}

**STRICT COMPLETION RULES**:

1. **Item Acquisition Objectives**: ONLY mark complete when:
   - The EXACT item name from the objective appears in the action (e.g., "take lantern" for "brass lantern" objective)
   - The response confirms acquisition (e.g., "Taken.")
   - If objective specifies a location, agent must be AT that location when acquiring

2. **Location Objectives**: ONLY mark complete when:
   - Current Location ID matches the exact ID in the objective
   - Agent has physically moved to that location (not just looking at it)

3. **Action Objectives**: ONLY mark complete when:
   - The EXACT action from objective was performed successfully
   - Game response confirms success (e.g., "opens" for "open door")

4. **Multi-Step Objectives**: ONLY mark complete when:
   - ALL steps in the sequence have been completed
   - Final step shows clear success in game response

**IMPORTANT - DO NOT MARK COMPLETE IF**:
- Agent took different items (e.g., taking "sack, bottle" does NOT complete "acquire lantern")
- Agent is at wrong location (e.g., Kitchen does NOT complete "acquire from Living Room")
- Score increased but objective's specific item/location wasn't achieved
- Action was similar but not exact (e.g., "examine door" does NOT complete "open door")
- Only partial progress made (e.g., first step of multi-step objective)

**NEGATIVE EXAMPLES** (DO NOT mark these as complete):
❌ Objective: "Acquire brass lantern from Location 193 (Living Room)"
   Action: "take sack, bottle" at Kitchen → NOT COMPLETE (wrong item, wrong location)

❌ Objective: "Open the trap door"
   Action: "examine trap door" → NOT COMPLETE (examined, not opened)

❌ Objective: "Visit Location 152 (Troll Room)"
   Current Location: 151 → NOT COMPLETE (not at target location yet)

**POSITIVE EXAMPLES** (mark these as complete):
✅ Objective: "Acquire brass lantern from Location 193 (Living Room)"
   Action: "take lantern" at Living Room, Response: "Taken." → COMPLETE

✅ Objective: "Open the trap door"
   Action: "open trap door", Response: "The trap door opens." → COMPLETE

✅ Objective: "Visit Location 152 (Troll Room)"
   Current Location: 152, Previous Location: 151 → COMPLETE

Which objectives (if any) have been completed based on these STRICT rules?

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
  "reasoning": "Agent took lantern at Living Room with 'Taken.' response, meeting all strict criteria."
}}"""

        if self.adaptive_knowledge_manager and self.adaptive_knowledge_manager.client:
            try:
                response = (
                    self.adaptive_knowledge_manager.client.chat.completions.create(
                        model=self.adaptive_knowledge_manager.analysis_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.adaptive_knowledge_manager.analysis_sampling.get("temperature"),
                        top_p=self.adaptive_knowledge_manager.analysis_sampling.get("top_p"),
                        top_k=self.adaptive_knowledge_manager.analysis_sampling.get("top_k"),
                        min_p=self.adaptive_knowledge_manager.analysis_sampling.get("min_p"),
                        max_tokens=self.adaptive_knowledge_manager.analysis_sampling.get("max_tokens"),
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
                            completed_objectives, action_taken, game_response
                        )
                except Exception as e:
                    self.log_error(
                        f"Failed to parse completion JSON: {e}",
                        details=f"Response content (first 500 chars): {response_content[:500] if response_content else 'None'}"
                    )

            except Exception as e:
                self.log_error(f"Failed to evaluate objective completion: {e}")

    def _get_recent_action_history(self) -> str:
        """
        Format recent action history for completion context.

        Returns formatted markdown string with last N turns (configurable).
        Uses same formatting pattern as SimpleMemoryManager for consistency.
        """
        if not self.game_state.action_history:
            return ""

        # Get window size from config
        window = self.config.completion_history_window

        # Get last N actions
        recent_actions = self.game_state.action_history[-window:]

        # Calculate starting turn number
        start_turn = max(1, self.game_state.turn_count - len(recent_actions) + 1)

        # Format using same pattern as SimpleMemoryManager
        lines = []
        for i, entry in enumerate(recent_actions):
            turn_num = start_turn + i
            lines.append(f"Turn {turn_num}: {entry.action}")
            lines.append(f"Response: {entry.response}")
            # Add blank line between entries (except after last)
            if i < len(recent_actions) - 1:
                lines.append("")

        return "\n".join(lines)

    def _get_completion_memory_context(self, location_id: int) -> str:
        """
        Get location-specific memories for completion context.

        Args:
            location_id: Z-machine location ID

        Returns:
            Formatted memory string or indication if no memories available
        """
        if not self.simple_memory or location_id is None:
            return "No memories available for this location."

        # Get location memories (SimpleMemoryManager handles formatting)
        memory_text = self.simple_memory.get_location_memory(location_id)

        if not memory_text or memory_text.strip() == "":
            return "No memories available for this location."

        return memory_text

    def _mark_objectives_complete(
        self,
        completed_objectives: List[str],
        action_taken: str,
        game_response: str,
    ) -> None:
        """
        Mark objectives as completed and track the completion.

        Args:
            completed_objectives: List of objective texts that were completed
            action_taken: Action that led to completion
            game_response: Game's response to the action
        """
        for objective in completed_objectives:
            if objective in self.game_state.discovered_objectives:
                # Remove from discovered objectives
                self.game_state.discovered_objectives.remove(objective)

                # Add to completed objectives with metadata
                completion_record = {
                    "objective": objective,
                    "completed_turn": self.game_state.turn_count,
                    "completion_action": action_taken,
                    "completion_response": game_response,  # Store full game response
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

    def _get_full_knowledge(self) -> str:
        """
        Load full knowledge base - strategic wisdom.

        Returns:
            Complete contents of knowledge base file or fallback message
        """
        kb_path = Path(self.config.zork_game_workdir) / self.config.knowledge_file
        if kb_path.exists():
            return kb_path.read_text(encoding="utf-8")
        return "No strategic knowledge available."

    def _calculate_distance_bfs(self, from_id: int, to_id: int) -> int:
        """
        Calculate shortest path distance between locations using BFS.

        Args:
            from_id: Starting location ID
            to_id: Target location ID

        Returns:
            Hop count (int) or float('inf') if unreachable
        """
        if from_id == to_id:
            return 0

        if not self.map_manager or not hasattr(self.map_manager, 'game_map'):
            return float('inf')

        visited = {from_id}
        queue = deque([(from_id, 0)])  # (location_id, distance)

        while queue:
            current_id, dist = queue.popleft()

            # Check outgoing connections
            if current_id in self.map_manager.game_map.connections:
                for exit_action, dest_id in self.map_manager.game_map.connections[current_id].items():
                    if dest_id == to_id:
                        return dist + 1
                    if dest_id not in visited:
                        visited.add(dest_id)
                        queue.append((dest_id, dist + 1))

        return float('inf')  # Unreachable

    def _get_all_memories_by_distance(self, current_location_id: int) -> str:
        """
        Get all ACTIVE strategic memories sorted by distance from current location.

        Filtering:
        - Status: ACTIVE only (excludes TENTATIVE, SUPERSEDED)

        Note: Persistence filtering (CORE + PERMANENT) is not implemented yet in the
        Memory dataclass. When the persistence field is added, this method should be
        updated to filter: m.persistence in ["core", "permanent"]

        Returns:
            Formatted string with memories grouped by location and sorted by distance
        """
        if not self.simple_memory or not self.map_manager:
            return "No memory data available"

        # Calculate distances to all locations with strategic memories
        location_distances = []
        for loc_id, memories in self.simple_memory.memory_cache.items():
            # Filter to ACTIVE memories only (strategic filtering)
            # TODO: Add persistence filtering when field is implemented:
            # strategic_memories = [m for m in memories
            #                       if m.status == "ACTIVE"
            #                       and m.persistence in ["core", "permanent"]]
            strategic_memories = [
                m for m in memories
                if m.status == "ACTIVE"
            ]

            if strategic_memories:  # Only include location if it has strategic memories
                distance = self._calculate_distance_bfs(current_location_id, loc_id)
                location_distances.append((distance, loc_id, strategic_memories))

        # Sort by distance (closest first)
        location_distances.sort(key=lambda x: x[0])

        # Format grouped by location
        lines = ["## All Game Memories (Sorted by Distance from Current Location)\n"]
        for distance, loc_id, memories in location_distances:
            loc_name = self.map_manager.game_map.room_names.get(
                loc_id, f"Location #{loc_id}"
            )
            lines.append(f"\n**Location {loc_id} ({loc_name}) - {distance} hops away:**")
            lines.append(self._format_memories(memories))

        return "\n".join(lines)

    def _format_memories(self, memories: List) -> str:
        """
        Format list of Memory objects for prompt.

        Args:
            memories: List of Memory objects to format

        Returns:
            Formatted string with memory details
        """
        if not memories:
            return "  (No memories)"

        formatted = []
        for mem in memories[:5]:  # Top 5 memories per location
            status_marker = f" [{mem.status}]" if mem.status != "ACTIVE" else ""
            formatted.append(f"  - [{mem.category}] {mem.title}{status_marker}")
            formatted.append(f"    {mem.text}")

        return "\n".join(formatted)

    def _get_map_context(self) -> str:
        """
        Get map context including Mermaid diagram and exploration summary.

        Returns:
            Human-readable map information with location IDs
        """
        if not self.map_manager:
            return "No map data available"

        lines = []

        # Mermaid diagram
        mermaid = self.map_manager.game_map.render_mermaid()
        lines.append("## Map Visualization (Mermaid Format)")
        lines.append(mermaid)
        lines.append("\nNote: Node IDs match location IDs in memories (L180 = Location 180)")
        lines.append(f"**Current location**: L{self.game_state.current_room_id}\n")

        # Current location routing
        lines.append(self._get_routing_summary(self.game_state.current_room_id))

        # Exploration statistics
        total_rooms = len(self.map_manager.game_map.rooms)
        lines.append(f"\n## Exploration Statistics")
        lines.append(f"- Rooms discovered: {total_rooms}")
        lines.append(f"- Current location: {self.game_state.current_room_name} (ID: {self.game_state.current_room_id})")

        return "\n".join(lines)

    def _get_routing_summary(self, current_location_id: int) -> str:
        """
        Generate text-based routing summary for current location.

        Shows available exits from current location only.

        Args:
            current_location_id: Location ID to generate routing for

        Returns:
            Human-readable routing information with location IDs
        """
        if not self.map_manager:
            return "No routing data available"

        map_graph = self.map_manager.game_map
        lines = []

        # Current location connections
        current_name = map_graph.room_names.get(current_location_id, f"Location #{current_location_id}")
        lines.append(f"## Current Location: {current_location_id} ({current_name})")

        if current_location_id in map_graph.connections and map_graph.connections[current_location_id]:
            lines.append("**Available Exits:**")
            for exit_action, dest_id in sorted(map_graph.connections[current_location_id].items()):
                dest_name = map_graph.room_names.get(dest_id, f"Location #{dest_id}")
                lines.append(f"  - {exit_action} → Location {dest_id} ({dest_name})")
        else:
            lines.append("  - No mapped exits")

        return "\n".join(lines)

    def _get_gameplay_context(self) -> str:
        """
        Format recent gameplay history for objective discovery context.

        Provides recent action/response pairs to show tactical patterns
        without redundant extracted state data.

        Returns:
            Formatted string with recent actions and responses
        """
        lines = []

        # Recent action history (last 10 actions)
        recent_actions = self.game_state.action_history[-10:] if self.game_state.action_history else []
        if recent_actions:
            lines.append("## Recent Actions (Last 10 Turns)")
            for i, entry in enumerate(recent_actions, start=1):
                turn_num = self.game_state.turn_count - len(recent_actions) + i
                lines.append(f"\nTurn {turn_num}:")
                lines.append(f"  Action: {entry.action}")
                # Truncate long responses to avoid token bloat
                response_preview = entry.response[:200] + "..." if len(entry.response) > 200 else entry.response
                lines.append(f"  Response: {response_preview}")
        else:
            lines.append("## Recent Actions\n(No actions yet - start of episode)")

        return "\n".join(lines)

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
