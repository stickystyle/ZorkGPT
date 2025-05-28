"""
ZorkOrchestrator module for coordinating Zork gameplay episodes.

This module contains the main game loop and ties together all other modules:
- ZorkAgent for action generation
- ZorkExtractor for information extraction
- ZorkCritic for action evaluation
- Movement analysis and mapping
- Logging and experience tracking
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
from datetime import datetime
import os
import json
import time

from zork_api import ZorkInterface
from llm_client import LLMClientWrapper
from map_graph import MapGraph
from movement_analyzer import MovementAnalyzer, MovementContext
from logger import setup_logging

# Import our refactored modules with aliases to avoid conflicts
from zork_agent import ZorkAgent as AgentModule
from hybrid_zork_extractor import ExtractorResponse
from hybrid_zork_extractor import HybridZorkExtractor
from zork_critic import ZorkCritic, CriticResponse
from zork_strategy_generator import AdaptiveKnowledgeManager
from config import get_config, get_client_api_key

# Optional S3 support
try:
    import boto3

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


class ZorkOrchestrator:
    """
    Orchestrates Zork gameplay episodes by coordinating all subsystems.

    This class manages the main game loop and ties together:
    - Agent action generation
    - Information extraction
    - Critic evaluation
    - Movement tracking
    - Experience logging
    """

    def __init__(
        self,
        agent_model: str = None,
        critic_model: str = None,
        info_ext_model: str = None,
        episode_log_file: str = None,
        json_log_file: str = None,
        max_turns_per_episode: int = None,
        client_base_url: str = None,
        client_api_key: str = None,
        # Turn-based knowledge updating
        knowledge_update_interval: int = None,
        # Map updating (more frequent than full knowledge updates)
        map_update_interval: int = None,
        # State export configuration
        enable_state_export: bool = None,
        state_export_file: str = None,
        s3_bucket: str = None,
        s3_key_prefix: str = None,
        # Gameplay delay for viewer experience
        turn_delay_seconds: float = None,
    ):
        """Initialize the ZorkOrchestrator with all subsystems."""
        # Load configuration
        config = get_config()
        
        # Store configuration with precedence: parameters > config > defaults
        self.episode_log_file = episode_log_file if episode_log_file is not None else config.files.episode_log_file
        self.json_log_file = json_log_file if json_log_file is not None else config.files.json_log_file

        # Game settings
        self.max_turns_per_episode = max_turns_per_episode if max_turns_per_episode is not None else config.orchestrator.max_turns_per_episode
        self.turn_delay_seconds = turn_delay_seconds if turn_delay_seconds is not None else config.gameplay.turn_delay_seconds

        # Adaptive knowledge management (always enabled)
        self.knowledge_update_interval = knowledge_update_interval if knowledge_update_interval is not None else config.orchestrator.knowledge_update_interval
        self.last_knowledge_update_turn = 0
        
        # Map updating (more frequent than full knowledge updates)
        self.map_update_interval = map_update_interval if map_update_interval is not None else config.orchestrator.map_update_interval
        self.last_map_update_turn = 0

        # Initialize logger
        self.logger = setup_logging(self.episode_log_file, self.json_log_file)

        # State export configuration
        self.enable_state_export = enable_state_export if enable_state_export is not None else config.orchestrator.enable_state_export
        self.state_export_file = state_export_file if state_export_file is not None else config.files.state_export_file
        self.s3_bucket = s3_bucket or config.aws.s3_bucket
        self.s3_key_prefix = s3_key_prefix if s3_key_prefix is not None else config.aws.s3_key_prefix

        # Initialize S3 client if available and configured
        self.s3_client = None
        if S3_AVAILABLE and self.s3_bucket:
            try:
                self.s3_client = boto3.client("s3")
                self.logger.info(f"S3 export enabled for bucket: {self.s3_bucket}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize S3 client: {e}")

        # Initialize LLM client (shared across components)
        self.client = LLMClientWrapper(
            base_url=client_base_url or config.llm.client_base_url,
            api_key=client_api_key or get_client_api_key(),
        )

        # Initialize core components
        self.agent = AgentModule(
            model=agent_model, client=self.client, logger=self.logger
        )

        # Initialize hybrid extractor (combines structured parsing with LLM extraction)
        self.extractor = HybridZorkExtractor(
            model=info_ext_model, client=self.client, logger=self.logger
        )
        if self.logger:
            self.logger.info("Using hybrid extractor (structured + LLM)")

        self.critic = ZorkCritic(
            model=critic_model, client=self.client, logger=self.logger
        )

        # Initialize shared movement analyzer
        self.movement_analyzer = MovementAnalyzer()

        # Initialize adaptive knowledge manager (always enabled)
        self.adaptive_knowledge_manager = AdaptiveKnowledgeManager(
            log_file=self.json_log_file, output_file="knowledgebase.md"
        )

        # Session-persistent state (survives episode resets)
        self.death_count = 0  # Track cumulative deaths across all episodes

        # Episode state (reset for each episode)
        self.reset_episode_state()

    def reset_episode_state(self) -> None:
        """Reset all episode-specific state variables."""
        self.action_counts = Counter()
        self.action_history = []
        self.memory_log_history = []
        self.visited_locations = set()
        self.failed_actions_by_location = {}
        self.episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.previous_zork_score = 0
        self.turn_count = 0
        self.game_over_flag = False  # Track game over state for state export
        # Use MapGraph with enhanced confidence tracking
        self.game_map = MapGraph()
        self.current_room_name_for_map = ""
        self.prev_room_for_prompt_context: Optional[str] = None
        self.action_leading_to_current_room_for_prompt_context: Optional[str] = None
        self.current_inventory = []

        # Reset movement analyzer for new episode
        self.movement_analyzer.clear_pending_connections()

        # Agent reasoning tracking for state export
        self.action_reasoning_history = []

        # Reset adaptive knowledge tracking for new episode
        self.last_knowledge_update_turn = 0
        self.last_map_update_turn = 0

        # Note: death_count is NOT reset here - it persists across episodes

        # Update episode IDs in components
        self.agent.update_episode_id(self.episode_id)
        self.extractor.update_episode_id(self.episode_id)
        self.critic.update_episode_id(self.episode_id)

    def play_episode(self, zork_interface_instance) -> int:
        """
        Play a single episode of Zork.

        Args:
            zork_interface_instance: The Zork game interface

        Returns:
            Final Zork score
        """
        # Reset state for new episode
        self.reset_episode_state()

        # Generate episode ID
        episode_start_time = datetime.now()
        self.episode_id = episode_start_time.strftime("%Y-%m-%dT%H:%M:%S")

        # Log episode start
        self.logger.info(
            "Starting Zork episode...",
            extra={
                "extras": {
                    "event_type": "episode_start",
                    "episode_id": self.episode_id,
                    "agent_model": self.agent.model,
                    "critic_model": self.critic.model,
                    "info_ext_model": self.extractor.model,
                    "timestamp": datetime.now().isoformat(),
                    "max_turns": self.max_turns_per_episode,
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
            return 0

        # Enable verbose mode to get full room descriptions on every visit
        verbose_response = zork_interface_instance.send_command("verbose")
        self.logger.info(
            f"Enabled verbose mode: {verbose_response}",
            extra={
                "extras": {
                    "event_type": "verbose_mode_enabled",
                    "episode_id": self.episode_id,
                    "verbose_response": verbose_response,
                }
            },
        )

        # Initialize game state variables
        game_over = False
        current_inventory = []

        # Initial extraction and map update
        extracted_info = self.extractor.extract_info(current_game_state)
        if extracted_info:
            self.current_room_name_for_map = extracted_info.current_location_name
            self.game_map.add_room(self.current_room_name_for_map)
            self.game_map.update_room_exits(
                self.current_room_name_for_map, extracted_info.exits
            )
            self.memory_log_history.append(extracted_info)

            # Log initial extraction
            self.logger.info(
                f"Initial state extracted: {extracted_info.current_location_name}",
                extra={
                    "extras": {
                        "event_type": "extracted_info",
                        "episode_id": self.episode_id,
                        "extracted_info": extracted_info.model_dump(),
                    }
                },
            )
        else:
            self.current_room_name_for_map = "Unknown (Initial Extraction Failed)"
            self.game_map.add_room(self.current_room_name_for_map)

        # Export initial state
        self.export_current_state()

        # Main game loop
        while (
            not game_over
            and zork_interface_instance.is_running()
            and self.turn_count < self.max_turns_per_episode
        ):
            self.turn_count += 1
            
            # Start timing the turn for delay calculation
            turn_start_time = time.time()

            # Log turn start
            self.logger.info(
                f"Turn {self.turn_count}",
                extra={
                    "extras": {
                        "event_type": "turn_start",
                        "episode_id": self.episode_id,
                        "turn": self.turn_count,
                        "turn_start_time": turn_start_time,
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
                current_inventory, inventory_response = (
                    zork_interface_instance.inventory_with_response()
                )
                # Update instance variable for state export
                self.current_inventory = current_inventory

                # Check if the inventory command caused game over
                if inventory_response:
                    game_over_flag, game_over_reason = (
                        zork_interface_instance.is_game_over(inventory_response)
                    )
                    if game_over_flag:
                        # Check if this is a death and increment counter
                        if self._is_death_reason(game_over_reason):
                            self.death_count += 1
                        
                        # Set the game over flag for state export
                        self.game_over_flag = True
                        
                        # Log game over from inventory
                        self.logger.info(
                            f"Game over during inventory check: {game_over_reason}",
                            extra={
                                "extras": {
                                    "event_type": "game_over",
                                    "episode_id": self.episode_id,
                                                                    "reason": f"Inventory check triggered: {game_over_reason}",
                                "turn": self.turn_count,
                                "death_count": self.death_count,
                                }
                            },
                        )

                        # Get final score
                        current_zork_score_val, max_zork_score = (
                            zork_interface_instance.score(inventory_response)
                        )
                        self.previous_zork_score = current_zork_score_val

                        # Log death during inventory
                        self.logger.info(
                            "Death during inventory check - episode ending",
                            extra={
                                "extras": {
                                    "event_type": "death_during_inventory",
                                    "episode_id": self.episode_id,
                                    "final_score": self.previous_zork_score,
                                }
                            },
                        )

                        # Store reasoning for inventory-based death (for state export)
                        self.action_reasoning_history.append(
                            {
                                "turn": self.turn_count,
                                "action": "inventory",
                                "reasoning": f"Death occurred during inventory check: {game_over_reason}",
                                "critic_score": 0.0,
                                "critic_justification": "Death during inventory - no action evaluation",
                                "was_overridden": False,
                                "rejected_actions": None,
                            }
                        )

                        # Export final state with death information
                        self.export_current_state()

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

            # Get relevant memories for agent prompt
            relevant_memories = self.agent.get_relevant_memories_for_prompt(
                current_location_name_from_current_extraction=self.current_room_name_for_map,
                memory_log_history=self.memory_log_history,
                current_inventory=current_inventory,
                game_map=self.game_map,
                previous_room_name_for_map_context=self.prev_room_for_prompt_context,
                action_taken_to_current_room=self.action_leading_to_current_room_for_prompt_context,
                in_combat=in_combat,
                failed_actions_by_location=self.failed_actions_by_location,
            )

            # Get agent action with reasoning
            agent_response = self.agent.get_action_with_reasoning(
                game_state_text=current_game_state,
                previous_actions_and_responses=self.action_history[
                    -5:
                ],  # Last 5 actions
                action_counts=self.action_counts,
                relevant_memories=relevant_memories,
            )
            agent_action = agent_response["action"]
            agent_reasoning = agent_response["reasoning"]

            # Get critic evaluation
            critic_response = self.critic.get_robust_evaluation(
                game_state_text=current_game_state,
                proposed_action=agent_action,
                action_counts=self.action_counts,
                previous_actions_and_responses=self.action_history[
                    -3:
                ],  # Last 3 actions
            )

            critic_score_val = critic_response.score
            critic_justification = critic_response.justification
            critic_confidence = getattr(critic_response, "confidence", 0.8)

            # Check for action rejection and override logic with retry loop
            was_overridden = False
            rejection_threshold = self.critic.trust_tracker.get_rejection_threshold()
            max_rejections = 3  # Prevent infinite loops

            # Track all rejected actions and their justifications for transparency
            rejected_actions_with_justifications = []

            for rejection_attempt in range(max_rejections):
                if critic_score_val >= rejection_threshold:
                    break  # Action is acceptable, exit loop

                # Action was rejected by critic
                override_needed, override_reason = (
                    self.critic.rejection_system.should_override_rejection(
                        agent_action,
                        self.current_room_name_for_map,
                        self.failed_actions_by_location.get(
                            self.current_room_name_for_map, set()
                        ),
                        {
                            "turns_since_movement": getattr(
                                self, "turns_since_movement", 0
                            ),
                        },
                    )
                )

                if override_needed:
                    was_overridden = True
                    self.logger.info(
                        f"Overriding critic rejection: {override_reason}",
                        extra={
                            "extras": {
                                "event_type": "critic_override",
                                "episode_id": self.episode_id,
                                "reason": override_reason,
                                "original_score": critic_score_val,
                                "rejection_attempt": rejection_attempt + 1,
                            }
                        },
                    )
                    break  # Override accepted, exit loop
                else:
                    # Track rejected action with its justification
                    rejected_actions_with_justifications.append(
                        {
                            "action": agent_action,
                            "score": critic_score_val,
                            "justification": critic_justification,
                        }
                    )

                    # Track rejected action and get a new one
                    self.critic.rejection_system.rejected_actions_this_turn.append(
                        agent_action
                    )

                    self.logger.info(
                        f"Action rejected (attempt {rejection_attempt + 1}/{max_rejections}): {agent_action} (score: {critic_score_val:.2f})",
                        extra={
                            "extras": {
                                "event_type": "action_rejected",
                                "episode_id": self.episode_id,
                                "rejected_action": agent_action,
                                "rejection_score": critic_score_val,
                                "rejection_attempt": rejection_attempt + 1,
                                "justification": critic_justification,
                            }
                        },
                    )

                    # Get new action from agent with reasoning
                    rejected_actions_context = ", ".join(
                        self.critic.rejection_system.rejected_actions_this_turn
                    )
                    agent_response = self.agent.get_action_with_reasoning(
                        game_state_text=current_game_state
                        + f"\n\n[Previous action(s) '{rejected_actions_context}' were rejected by critic: {critic_justification}]",
                        previous_actions_and_responses=self.action_history[-5:],
                        action_counts=self.action_counts,
                        relevant_memories=relevant_memories,
                    )
                    agent_action = agent_response["action"]
                    agent_reasoning = agent_response["reasoning"]

                    # Re-evaluate new action
                    critic_response = self.critic.get_robust_evaluation(
                        game_state_text=current_game_state,
                        proposed_action=agent_action,
                        action_counts=self.action_counts,
                        previous_actions_and_responses=self.action_history[-3:],
                    )
                    critic_score_val = critic_response.score
                    critic_justification = critic_response.justification

            # If we've exhausted all rejection attempts, log a warning
            if critic_score_val < rejection_threshold and not was_overridden:
                self.logger.warning(
                    f"Exhausted rejection attempts, proceeding with low-scoring action: {agent_action} (score: {critic_score_val:.2f})",
                    extra={
                        "extras": {
                            "event_type": "rejection_attempts_exhausted",
                            "episode_id": self.episode_id,
                            "final_action": agent_action,
                            "final_score": critic_score_val,
                            "threshold": rejection_threshold,
                        }
                    },
                )

            # Store reasoning for state export
            self.action_reasoning_history.append(
                {
                    "turn": self.turn_count,
                    "action": agent_action,
                    "reasoning": agent_reasoning,
                    "critic_score": critic_score_val,
                    "critic_justification": critic_justification,
                    "was_overridden": was_overridden,
                    "rejected_actions": rejected_actions_with_justifications
                    if rejected_actions_with_justifications
                    else None,
                }
            )

            # Log final selected action
            self.logger.info(
                f"SELECTED ACTION: {agent_action} (Score: {critic_score_val:.2f}, Confidence: {critic_confidence:.2f}, Override: {was_overridden})",
                extra={
                    "extras": {
                        "event_type": "final_action_selection",
                        "episode_id": self.episode_id,
                        "agent_action": agent_action,
                        "agent_reasoning": agent_reasoning,
                        "critic_score": critic_score_val,
                        "critic_confidence": critic_confidence,
                        "was_overridden": was_overridden,
                    }
                },
            )

            # Update action count for repetition tracking
            self.action_counts[agent_action] += 1

            # Send the chosen action to Zork
            room_before_action = self.current_room_name_for_map
            action_taken = agent_action

            try:
                next_game_state = zork_interface_instance.send_command(action_taken)

                # Get clean game text for display (without structured header)
                clean_game_text = next_game_state
                if hasattr(self.extractor, "get_clean_game_text"):
                    clean_game_text = self.extractor.get_clean_game_text(
                        next_game_state
                    )

                # Log Zork response (using clean text for display)
                self.logger.info(
                    f"ZORK RESPONSE for '{action_taken}':\n{clean_game_text}\n",
                    extra={
                        "extras": {
                            "event_type": "zork_response",
                            "episode_id": self.episode_id,
                            "action": action_taken,
                            "zork_response": clean_game_text,  # Store clean text for display
                            "raw_zork_response": next_game_state,  # Keep raw for parsing if needed
                        }
                    },
                )

                # Check if the game has ended based on the response
                game_over_flag, game_over_reason = zork_interface_instance.is_game_over(
                    next_game_state
                )
                if game_over_flag:
                    # Check if this is a death and track it
                    if self._is_death_reason(game_over_reason):
                        self.death_count += 1
                    
                    # Set the game over flag for state export
                    self.game_over_flag = True
                    
                    # Log game over
                    self.logger.info(
                        f"{game_over_reason}",
                        extra={
                            "extras": {
                                "event_type": "game_over",
                                "episode_id": self.episode_id,
                                "reason": game_over_reason,
                                "death_count": self.death_count,
                            }
                        },
                    )

                    game_over = True
                    current_zork_score_val, max_zork_score = (
                        zork_interface_instance.score(next_game_state)
                    )
                    self.previous_zork_score = current_zork_score_val
                    # Store clean game text in action history (without structured header)
                    self.action_history.append((action_taken, clean_game_text))

                    # Log game over details
                    self.logger.info(
                        f"Game over: {game_over_reason}",
                        extra={
                            "extras": {
                                "event_type": "game_over_final",
                                "episode_id": self.episode_id,
                                "reason": game_over_reason,
                                "final_score": self.previous_zork_score,
                                "action_taken": action_taken,
                            }
                        },
                    )
                    
                    # Store reasoning for death action (for state export)
                    if 'agent_reasoning' in locals():
                        self.action_reasoning_history.append(
                            {
                                "turn": self.turn_count,
                                "action": action_taken,
                                "reasoning": agent_reasoning,
                                "critic_score": critic_score_val if 'critic_score_val' in locals() else 0.0,
                                "critic_justification": critic_justification if 'critic_justification' in locals() else "Game Over",
                                "was_overridden": was_overridden if 'was_overridden' in locals() else False,
                                "rejected_actions": rejected_actions_with_justifications if 'rejected_actions_with_justifications' in locals() else None,
                            }
                        )
                    
                    # Extract information about the death state
                    llm_extracted_info = self.extractor.extract_info(
                        next_game_state, room_before_action
                    )
                    if llm_extracted_info:
                        self.memory_log_history.append(llm_extracted_info)
                        # Update current room for state export
                        self.current_room_name_for_map = llm_extracted_info.current_location_name
                        
                        # Log death extraction
                        self.logger.info(
                            f"Death state extracted: {llm_extracted_info.current_location_name}",
                            extra={
                                "extras": {
                                    "event_type": "death_state_extracted",
                                    "episode_id": self.episode_id,
                                    "extracted_info": llm_extracted_info.model_dump(),
                                }
                            },
                        )
                    
                    # Export final state with death information
                    self.export_current_state()
                    
                    # End the episode
                    break

                # Store clean game text in action history (without structured header)
                self.action_history.append((action_taken, clean_game_text))

                # Extract information using the extractor
                llm_extracted_info = self.extractor.extract_info(
                    next_game_state, room_before_action
                )

                # Use the extracted location directly - the enhanced extractor handles persistence internally
                if llm_extracted_info:
                    final_current_room_name = llm_extracted_info.current_location_name
                    source_of_location = "Enhanced LLM"
                    self.memory_log_history.append(llm_extracted_info)

                    # Log extraction
                    self.logger.info(
                        f"State extracted: {final_current_room_name}",
                        extra={
                            "extras": {
                                "event_type": "extracted_info",
                                "episode_id": self.episode_id,
                                "extracted_info": llm_extracted_info.model_dump(),
                            }
                        },
                    )
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
                    self.critic.rejection_system, "rejected_actions_this_turn", []
                )
                self.critic.track_rejection_outcome(
                    rejected_actions, action_taken, next_game_state
                )
                self._update_movement_tracking(
                    action_taken, room_before_action, final_current_room_name
                )

                # Create stable location identifiers using exit-based unique ID system
                # This prioritizes exit patterns over volatile descriptions for stability
                current_location_id = self.game_map._create_unique_location_id(
                    final_current_room_name,
                    description=' '.join(llm_extracted_info.important_messages) if llm_extracted_info else '',
                    objects=llm_extracted_info.visible_objects if llm_extracted_info else [],
                    exits=llm_extracted_info.exits if llm_extracted_info else []
                )
                
                # Use shared MovementAnalyzer for consistent movement detection
                movement_context = MovementContext(
                    current_location=current_location_id,
                    previous_location=room_before_action,  # Will be converted to unique ID in movement analysis
                    action=action_taken,
                    game_response=next_game_state,
                    turn_number=self.turn_count,
                )

                # Analyze potential connections
                movement_result = self.movement_analyzer.analyze_movement(
                    movement_context
                )
                if movement_result.connection_created:
                    # Use the improved unique ID system for both locations to ensure consistency
                    from_location_id = self.game_map._create_unique_location_id(
                        movement_result.from_location,
                        description='',  # No description available for previous location
                        objects=[],
                        exits=[]  # Exit info not available for previous location
                    )
                    to_location_id = current_location_id  # Already computed above
                    
                    # Add connection to map with unique identifiers
                    self.game_map.add_connection(
                        from_location_id,
                        movement_result.action,
                        to_location_id,
                    )

                    # Log the connection
                    self.logger.info(
                        f"Movement connection: {from_location_id} --[{movement_result.action}]--> {to_location_id}",
                        extra={
                            "extras": {
                                "event_type": "movement_connection_created",
                                "episode_id": self.episode_id,
                                "from_room": from_location_id,
                                "to_room": to_location_id,
                                "action": movement_result.action,
                                "confidence": getattr(
                                    movement_result, "confidence", 1.0
                                ),
                            }
                        },
                    )

                # Get score - try structured parser first if using hybrid extractor
                structured_score = None
                structured_moves = None

                if hasattr(self.extractor, "get_score_and_moves"):
                    structured_score, structured_moves = (
                        self.extractor.get_score_and_moves(next_game_state)
                    )

                if structured_score is not None:
                    # Use score from structured parser
                    current_zork_score_val = structured_score
                    max_zork_score = 585  # Known max score for Zork I

                    # Log successful structured score extraction
                    self.logger.debug(
                        f"Score extracted via structured parser: {current_zork_score_val} (moves: {structured_moves})",
                        extra={
                            "extras": {
                                "event_type": "structured_score_extraction",
                                "episode_id": self.episode_id,
                                "score": current_zork_score_val,
                                "moves": structured_moves,
                            }
                        },
                    )
                else:
                    # Fallback to traditional score extraction
                    try:
                        if not game_over and zork_interface_instance.is_running():
                            current_zork_score_val, max_zork_score = (
                                zork_interface_instance.score()
                            )
                        else:
                            # Use the score method with the game text for parsing
                            current_zork_score_val, max_zork_score = (
                                zork_interface_instance.score(next_game_state)
                            )
                        
                        # Check if score parsing returned 0 but we had a previous non-zero score
                        # This happens when the parser doesn't understand the command and returns default values
                        if (current_zork_score_val == 0 and max_zork_score == 0 and 
                            self.previous_zork_score > 0):
                            self.logger.warning(
                                f"Score parsing returned 0 but previous score was {self.previous_zork_score}. "
                                f"Likely parser error - maintaining previous score.",
                                extra={
                                    "extras": {
                                        "event_type": "score_parsing_zero_fallback",
                                        "episode_id": self.episode_id,
                                        "previous_score": self.previous_zork_score,
                                        "parsed_score": current_zork_score_val,
                                        "game_text": next_game_state[:100] + "..." if len(next_game_state) > 100 else next_game_state,
                                    }
                                },
                            )
                            current_zork_score_val = self.previous_zork_score
                            max_zork_score = 585  # Default max score for Zork I
                            
                    except Exception as score_parse_error:
                        # If score parsing fails completely, maintain the previous score
                        self.logger.warning(
                            f"Score parsing failed, maintaining previous score: {self.previous_zork_score}. Error: {score_parse_error}",
                            extra={
                                "extras": {
                                    "event_type": "score_parsing_exception_fallback",
                                    "episode_id": self.episode_id,
                                    "previous_score": self.previous_zork_score,
                                    "error": str(score_parse_error),
                                }
                            },
                        )
                        current_zork_score_val = self.previous_zork_score
                        max_zork_score = 585  # Default max score for Zork I

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

            # Check for Zork's internal score changes and log them
            score_change = current_zork_score_val - self.previous_zork_score
            if score_change > 0:
                self.logger.info(
                    f"Score increased by {score_change} points!",
                    extra={
                        "extras": {
                            "event_type": "score_increase",
                            "episode_id": self.episode_id,
                            "score_change": score_change,
                            "new_score": current_zork_score_val,
                        }
                    },
                )
            elif score_change < 0:
                self.logger.info(
                    f"Score decreased by {abs(score_change)} points",
                    extra={
                        "extras": {
                            "event_type": "score_decrease",
                            "episode_id": self.episode_id,
                            "score_change": score_change,
                            "new_score": current_zork_score_val,
                        }
                    },
                )

            self.previous_zork_score = current_zork_score_val

            current_game_state = next_game_state

            # Update current room name for next iteration
            self.current_room_name_for_map = current_location_id

            # Reset critic rejection system for next turn
            self.critic.rejection_system.reset_turn()

            # Check for adaptive knowledge update
            self._check_adaptive_knowledge_update()
            
            # Check for map update (more frequent than full knowledge updates)
            self._check_map_update()
            
            # Consolidate fragmented map locations only when new rooms have been added
            if self.game_map.needs_consolidation():
                consolidations = self.game_map.consolidate_similar_locations()
                if consolidations > 0:
                    self.logger.info(
                        f"Map consolidation: merged {consolidations} fragmented locations",
                        extra={
                            "extras": {
                                "event_type": "map_consolidation",
                                "episode_id": self.episode_id,
                                "turn": self.turn_count,
                                "consolidations_performed": consolidations,
                                "trigger": "new_rooms_added",
                            }
                        },
                    )

            # Export current state after each turn
            self.export_current_state()

            # Add configurable delay for viewer experience to ensure minimum turn duration
            if self.turn_delay_seconds > 0:
                turn_elapsed_time = time.time() - turn_start_time
                remaining_time = self.turn_delay_seconds - turn_elapsed_time
                
                if remaining_time > 0:
                    self.logger.info(
                        f"Turn took {turn_elapsed_time:.2f}s, pausing for {remaining_time:.2f}s more to reach minimum {self.turn_delay_seconds}s",
                        extra={
                            "extras": {
                                "event_type": "turn_delay",
                                "episode_id": self.episode_id,
                                "turn_elapsed_time": turn_elapsed_time,
                                "delay_seconds": remaining_time,
                                "target_turn_duration": self.turn_delay_seconds,
                                "turn": self.turn_count,
                            }
                        },
                    )
                    time.sleep(remaining_time)
                else:
                    self.logger.info(
                        f"Turn took {turn_elapsed_time:.2f}s (>= {self.turn_delay_seconds}s target), no additional delay needed",
                        extra={
                            "extras": {
                                "event_type": "turn_no_delay_needed",
                                "episode_id": self.episode_id,
                                "turn_elapsed_time": turn_elapsed_time,
                                "target_turn_duration": self.turn_delay_seconds,
                                "turn": self.turn_count,
                            }
                        },
                    )

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
                    "max_score": max_zork_score
                    if "max_zork_score" in locals()
                    else 585,
                    "final_max_turns": self.max_turns_per_episode,
                    # Performance metrics
                    "avg_critic_score": self.get_avg_critic_score(),
                }
            },
        )

        # Episode cleanup

        # Perform final adaptive knowledge update if there's been significant progress
        self._perform_final_knowledge_update()

        return self.previous_zork_score

    def _check_adaptive_knowledge_update(self) -> None:
        """Check if it's time for an adaptive knowledge update and perform it if needed."""
        if not self.adaptive_knowledge_manager:
            return

        # Check if enough turns have passed since last update
        turns_since_last_update = self.turn_count - self.last_knowledge_update_turn

        if turns_since_last_update >= self.knowledge_update_interval:
            # Calculate turn window for analysis
            start_turn = max(1, self.last_knowledge_update_turn + 1)
            end_turn = self.turn_count

            self.logger.info(
                f"Attempting adaptive knowledge update for turns {start_turn}-{end_turn}",
                extra={
                    "extras": {
                        "event_type": "adaptive_knowledge_update_start",
                        "episode_id": self.episode_id,
                        "start_turn": start_turn,
                        "end_turn": end_turn,
                        "turns_since_last_update": turns_since_last_update,
                    }
                },
            )

            try:
                # Include map quality metrics
                map_metrics = self.game_map.get_map_quality_metrics()
                self.logger.info(
                    f"📊 Map Quality: {map_metrics['average_confidence']:.2f} avg confidence, "
                    f"{map_metrics['high_confidence_ratio']:.1%} high confidence, "
                    f"{map_metrics['verified_connections']} verified connections"
                )

                # Attempt knowledge update
                update_success = (
                    self.adaptive_knowledge_manager.update_knowledge_from_turns(
                        episode_id=self.episode_id,
                        start_turn=start_turn,
                        end_turn=end_turn,
                    )
                )

                if update_success:
                    self.last_knowledge_update_turn = self.turn_count
                    self.logger.info(
                        "Adaptive knowledge update completed successfully",
                        extra={
                            "extras": {
                                "event_type": "adaptive_knowledge_update_success",
                                "episode_id": self.episode_id,
                                "updated_turn": self.turn_count,
                            }
                        },
                    )

                    # Update map in knowledge base after successful knowledge update
                    self._update_knowledge_base_map()

                    # Reload knowledge in agent for immediate use
                    self._reload_agent_knowledge()

                else:
                    self.logger.info(
                        "Adaptive knowledge update skipped (low quality data)",
                        extra={
                            "extras": {
                                "event_type": "adaptive_knowledge_update_skipped",
                                "episode_id": self.episode_id,
                                "reason": "low_quality_data",
                            }
                        },
                    )

            except Exception as e:
                self.logger.warning(
                    f"Adaptive knowledge update failed: {e}",
                    extra={
                        "extras": {
                            "event_type": "adaptive_knowledge_update_failed",
                            "episode_id": self.episode_id,
                            "error": str(e),
                        }
                    },
                )

    def _check_map_update(self) -> None:
        """Check if it's time for a map update and perform it if needed."""
        if not self.adaptive_knowledge_manager:
            return

        # Check if enough turns have passed since last map update
        turns_since_last_map_update = self.turn_count - self.last_map_update_turn

        if turns_since_last_map_update >= self.map_update_interval:
            self.logger.info(
                f"Updating map in knowledge base (turn {self.turn_count})",
                extra={
                    "extras": {
                        "event_type": "map_update_check",
                        "episode_id": self.episode_id,
                        "turn": self.turn_count,
                        "turns_since_last_map_update": turns_since_last_map_update,
                    }
                },
            )

            # Update map in knowledge base
            self._update_knowledge_base_map()
            self.last_map_update_turn = self.turn_count

    def _update_knowledge_base_map(self) -> None:
        """Update the mermaid map in the knowledge base."""
        if not self.adaptive_knowledge_manager:
            return
            
        try:
            map_updated = self.adaptive_knowledge_manager.update_knowledge_with_map(
                episode_id=self.episode_id,
                game_map=self.game_map
            )
            
            if map_updated:
                self.logger.info(
                    "Map updated in knowledge base",
                    extra={
                        "extras": {
                            "event_type": "knowledge_base_map_update_success",
                            "episode_id": self.episode_id,
                            "turn": self.turn_count,
                        }
                    },
                )
            else:
                self.logger.info(
                    "Map update skipped (no map data)",
                    extra={
                        "extras": {
                            "event_type": "knowledge_base_map_update_skipped",
                            "episode_id": self.episode_id,
                            "turn": self.turn_count,
                        }
                    },
                )
                
        except Exception as e:
            self.logger.warning(
                f"Failed to update map in knowledge base: {e}",
                extra={
                    "extras": {
                        "event_type": "knowledge_base_map_update_failed",
                        "episode_id": self.episode_id,
                        "error": str(e),
                    }
                },
            )

    def _reload_agent_knowledge(self) -> None:
        """Reload the knowledge base in the agent for immediate use."""
        try:
            # Actually reload the knowledge base in the agent
            success = self.agent.reload_knowledge_base()
            
            if success:
                self.logger.info(
                    "Knowledge base reloaded for agent use",
                    extra={
                        "extras": {
                            "event_type": "agent_knowledge_reload",
                            "episode_id": self.episode_id,
                        }
                    },
                )
            else:
                self.logger.warning("Failed to reload agent knowledge base")
                
        except Exception as e:
            self.logger.warning(f"Failed to reload agent knowledge: {e}")

    def _perform_final_knowledge_update(self) -> None:
        """Perform a final knowledge update at episode end if there's been significant progress."""
        if not self.adaptive_knowledge_manager:
            return

        # Calculate turns since last update
        turns_since_last_update = self.turn_count - self.last_knowledge_update_turn

        # Define minimum threshold for "significant progress" - much lower than normal interval
        min_significant_turns = max(
            10, self.knowledge_update_interval // 4
        )  # At least 10 turns, or 25% of normal interval

        if turns_since_last_update >= min_significant_turns:
            # Calculate turn window for analysis
            start_turn = max(1, self.last_knowledge_update_turn + 1)
            end_turn = self.turn_count

            self.logger.info(
                f"Performing final knowledge update for turns {start_turn}-{end_turn} (episode ended)",
                extra={
                    "extras": {
                        "event_type": "final_knowledge_update_start",
                        "episode_id": self.episode_id,
                        "start_turn": start_turn,
                        "end_turn": end_turn,
                        "turns_since_last_update": turns_since_last_update,
                        "reason": "episode_ended",
                    }
                },
            )

            try:
                # Attempt knowledge update with potentially lower quality threshold for final updates
                update_success = (
                    self.adaptive_knowledge_manager.update_knowledge_from_turns(
                        episode_id=self.episode_id,
                        start_turn=start_turn,
                        end_turn=end_turn,
                        is_final_update=True,
                    )
                )

                if update_success:
                    self.logger.info(
                        "Final knowledge update completed successfully",
                        extra={
                            "extras": {
                                "event_type": "final_knowledge_update_success",
                                "episode_id": self.episode_id,
                                "turns_analyzed": turns_since_last_update,
                            }
                        },
                    )
                    
                    # Update map in knowledge base after successful final update
                    self._update_knowledge_base_map()
                else:
                    self.logger.info(
                        "Final knowledge update skipped (low quality data)",
                        extra={
                            "extras": {
                                "event_type": "final_knowledge_update_skipped",
                                "episode_id": self.episode_id,
                                "reason": "low_quality_data",
                                "turns_analyzed": turns_since_last_update,
                            }
                        },
                    )

            except Exception as e:
                self.logger.warning(
                    f"Final knowledge update failed: {e}",
                    extra={
                        "extras": {
                            "event_type": "final_knowledge_update_failed",
                            "episode_id": self.episode_id,
                            "error": str(e),
                            "turns_analyzed": turns_since_last_update,
                        }
                    },
                )
        else:
            self.logger.info(
                f"Skipping final knowledge update - insufficient progress ({turns_since_last_update} turns < {min_significant_turns} minimum)",
                extra={
                    "extras": {
                        "event_type": "final_knowledge_update_skipped",
                        "episode_id": self.episode_id,
                        "reason": "insufficient_progress",
                        "turns_since_last_update": turns_since_last_update,
                        "min_required_turns": min_significant_turns,
                    }
                },
            )

    def _update_movement_tracking(
        self, action: str, from_room: str, to_room: str
    ) -> None:
        """Update movement tracking and failed actions."""
        # Track failed actions by location
        if from_room and from_room == to_room:
            # Action didn't result in movement, might be a failed action
            if from_room not in self.failed_actions_by_location:
                self.failed_actions_by_location[from_room] = set()

            # Only add to failed actions if it looks like a movement command that failed
            movement_keywords = [
                "north",
                "south",
                "east",
                "west",
                "up",
                "down",
                "enter",
                "exit",
                "go",
                "climb",
            ]
            if any(keyword in action.lower() for keyword in movement_keywords):
                self.failed_actions_by_location[from_room].add(action)

        # Update room context for next turn
        self.prev_room_for_prompt_context = from_room
        self.action_leading_to_current_room_for_prompt_context = action

    def get_current_state(self) -> Dict[str, Any]:
        """Get comprehensive current state for export."""
        return {
            "metadata": {
                "episode_id": self.episode_id,
                "timestamp": datetime.now().isoformat(),
                "turn_count": self.turn_count,
                "game_over": self.game_over_flag,
                "score": self.previous_zork_score,
                "max_turns": self.max_turns_per_episode,
                "models": {
                    "agent": self.agent.model,
                    "critic": self.critic.model,
                    "extractor": self.extractor.model,
                    "knowledge_base": self.adaptive_knowledge_manager.analysis_model if self.adaptive_knowledge_manager else "Not available",
                },
            },
            "current_state": {
                "location": self.current_room_name_for_map,
                "inventory": self.current_inventory,
                "in_combat": self._get_combat_status(),
                "death_count": self.death_count,
            },
            "recent_log": self.get_recent_log(20),
            "map": {
                "mermaid_diagram": self.game_map.render_mermaid(),
                "current_room": self.current_room_name_for_map,
                "total_rooms": len(self.game_map.rooms),
                "total_connections": sum(
                    len(connections)
                    for connections in self.game_map.connections.values()
                ),
                # Enhanced map metrics
                "quality_metrics": self.game_map.get_map_quality_metrics(),
                "confidence_report": self.game_map.render_confidence_report(),
                # Optional: Include raw data for advanced frontends
                "raw_data": {
                    "rooms": {
                        name: {"exits": list(room.exits)}
                        for name, room in self.game_map.rooms.items()
                    },
                    "connections": self.game_map.connections,
                },
            },
            "knowledge_base": self.get_knowledge_base_summary(),
            "performance": {
                "avg_critic_score": self.get_avg_critic_score(),
                "recent_actions": self.get_recent_action_summary(),
            },
        }

    def get_recent_log(self, length: int = 10) -> List[Dict[str, Any]]:
        """Get recent game log entries with reasoning."""
        recent_log = []

        # Get the most recent entries from action_history and memory_log_history
        recent_actions = self.action_history[-length:] if self.action_history else []
        recent_extractions = (
            self.memory_log_history[-length:] if self.memory_log_history else []
        )
        recent_reasoning = (
            self.action_reasoning_history[-length:]
            if self.action_reasoning_history
            else []
        )

        # Combine them by turn (assuming they're in sync)
        for i, (action, zork_response) in enumerate(recent_actions):
            turn_num = self.turn_count - len(recent_actions) + i + 1

            log_entry = {
                "turn": turn_num,
                "action": action,
                "zork_response": zork_response,
            }

            # Add reasoning if available
            reasoning_entry = None
            for reasoning in recent_reasoning:
                if reasoning["turn"] == turn_num and reasoning["action"] == action:
                    reasoning_entry = reasoning
                    break

            if reasoning_entry:
                log_entry["reasoning"] = reasoning_entry["reasoning"]
                log_entry["critic_score"] = reasoning_entry["critic_score"]
                log_entry["critic_justification"] = reasoning_entry.get(
                    "critic_justification"
                )
                log_entry["was_overridden"] = reasoning_entry["was_overridden"]
                log_entry["rejected_actions"] = reasoning_entry.get("rejected_actions")

            # Add extraction info if available
            if i < len(recent_extractions):
                extraction = recent_extractions[i]
                if hasattr(extraction, "model_dump"):
                    log_entry["extracted_info"] = extraction.model_dump()
                else:
                    log_entry["extracted_info"] = {
                        "current_location_name": getattr(
                            extraction, "current_location_name", ""
                        ),
                        "exits": getattr(extraction, "exits", []),
                        "visible_objects": getattr(extraction, "visible_objects", []),
                        "important_messages": getattr(
                            extraction, "important_messages", []
                        ),
                    }

            recent_log.append(log_entry)

        return recent_log

    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Get knowledge base without the embedded map."""
        try:
            with open("knowledgebase.md", "r") as f:
                content = f.read()

            # Split content and exclude the map section
            sections = content.split("## CURRENT WORLD MAP")
            knowledge_only = sections[0] if sections else content

            return {
                "content": knowledge_only.strip(),
                "last_updated": os.path.getmtime("knowledgebase.md")
                if os.path.exists("knowledgebase.md")
                else None,
            }
        except Exception:
            return {"content": "No knowledge base available", "last_updated": None}

    def _get_combat_status(self) -> bool:
        """Determine if currently in combat based on recent extractions."""
        if not self.memory_log_history:
            return False

        recent_extraction = self.memory_log_history[-1]
        return getattr(recent_extraction, "in_combat", False)

    def _is_death_reason(self, game_over_reason: str) -> bool:
        """Determine if a game over reason indicates a death (as opposed to victory)."""
        if not game_over_reason:
            return False
        
        reason_lower = game_over_reason.lower()
        
        # Death indicators from zork_api.py
        death_indicators = [
            "you have died",
            "you are dead",
            "you died",
            "you have been eaten",
            "eaten by a grue",
            "killed by",
            "you fall",
            "you are engulfed",
            "crushed",
            "blown up",
            "incinerated",
            "drowned",
            "suffocated",
            "starved",
            "frozen",
            "electrocuted",
            "poisoned",
            "dissolved",
            "obliterated",
            "annihilated",
            "terminated",
            "destroyed",
            "perished",
            "expired",
            "demise"
        ]
        
        # Victory indicators (explicitly not deaths)
        victory_indicators = [
            "you have won",
            "congratulations",
            "you win",
            "victory",
            "triumphant",
            "successful",
            "completed",
            "finished"
        ]
        
        # Check for victory first - if it's a victory, it's not a death
        for victory_word in victory_indicators:
            if victory_word in reason_lower:
                return False
        
        # Check for death indicators
        for death_word in death_indicators:
            if death_word in reason_lower:
                return True
        
        # Default: if game over but not explicitly victory, treat as death
        # This covers edge cases where the death message might be unusual
        return True

    def get_avg_critic_score(self) -> float:
        """Get average critic score for recent turns."""
        if (
            not hasattr(self, "action_reasoning_history")
            or not self.action_reasoning_history
        ):
            return 0.0

        recent_reasoning = self.action_reasoning_history[-10:]  # Last 10 turns
        critic_scores = [
            reasoning.get("critic_score", 0.0) for reasoning in recent_reasoning
        ]

        if not critic_scores:
            return 0.0

        return sum(critic_scores) / len(critic_scores)

    def get_recent_action_summary(self) -> List[str]:
        """Get summary of recent actions."""
        if not self.action_history:
            return []

        recent_actions = self.action_history[-5:]  # Last 5 actions
        return [action for action, _ in recent_actions]

    def export_current_state(self) -> None:
        """Export current state to file and optionally to S3."""
        if not self.enable_state_export:
            return

        try:
            state = self.get_current_state()

            # Export to local file
            temp_file = f"{self.state_export_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)

            # Atomic rename
            os.rename(temp_file, self.state_export_file)

            # Upload to S3 if configured
            if self.s3_client and self.s3_bucket:
                self.upload_state_to_s3(state)

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to export current state: {e}")

    def upload_state_to_s3(self, state: Dict[str, Any]) -> None:
        """Upload current state to S3."""
        try:
            # Current state file
            current_key = f"{self.s3_key_prefix}current_state.json"

            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=current_key,
                Body=json.dumps(state, indent=2),
                ContentType="application/json",
                CacheControl="no-cache, must-revalidate",  # Ensure fresh data
            )

            # Optional: Also save timestamped snapshot
            timestamp_key = f"{self.s3_key_prefix}snapshots/{state['metadata']['episode_id']}/turn_{state['metadata']['turn_count']}.json"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=timestamp_key,
                Body=json.dumps(state, indent=2),
                ContentType="application/json",
            )

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to upload state to S3: {e}")


if __name__ == "__main__":
    # Create ZorkOrchestrator instance with default settings
    orchestrator = ZorkOrchestrator()

    with ZorkInterface(timeout=1.0) as zork_game:
        try:
            final_score = orchestrator.play_episode(zork_game)
            print(f"\nPlayed one episode. Final Zork score: {final_score}")
            print(f"Turns taken: {orchestrator.turn_count}")
            print(orchestrator.game_map.render_ascii())
        except RuntimeError as e:
            print(f"ZorkInterface runtime error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("Ensuring Zork process is closed.")
