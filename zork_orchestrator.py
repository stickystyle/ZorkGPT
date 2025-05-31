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
import re

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

# Import shared utilities
from shared_utils import estimate_context_tokens


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
        # Objective updating (fastest update cycle for goal discovery)
        objective_update_interval: int = None,
        # State export configuration
        enable_state_export: bool = None,
        state_export_file: str = None,
        s3_bucket: str = None,
        s3_key_prefix: str = None,
        # Gameplay delay for viewer experience
        turn_delay_seconds: float = None,
        # Objective Refinement Configuration
        enable_objective_refinement: bool = None,
        objective_refinement_interval: int = None,
        max_objectives_before_forced_refinement: int = None,
        refined_objectives_target_count: int = None,
        last_objective_refinement_turn: int = None,
    ):
        """Initialize the ZorkOrchestrator with all subsystems."""
        # Load configuration
        config = get_config()
        
        # Store configuration with precedence: parameters > config > defaults
        self.episode_log_file = episode_log_file if episode_log_file is not None else config.files.episode_log_file
        self.json_log_file = json_log_file if json_log_file is not None else config.files.json_log_file

        # Save/restore configuration
        self.zork_save_filename = config.gameplay.zork_save_filename
        self.zork_game_workdir = config.gameplay.zork_game_workdir
        self.save_signal_filename = config.gameplay.save_signal_filename
        
        # Calculate absolute paths
        self.zork_workdir_abs_path = os.path.abspath(self.zork_game_workdir)
        self.zork_save_file_abs_path = os.path.join(self.zork_workdir_abs_path, self.zork_save_filename)
        self.save_signal_file_abs_path = os.path.abspath(self.save_signal_filename)
        
        # Ensure game directory exists
        os.makedirs(self.zork_workdir_abs_path, exist_ok=True)

        # Game settings
        self.max_turns_per_episode = max_turns_per_episode if max_turns_per_episode is not None else config.orchestrator.max_turns_per_episode
        self.turn_delay_seconds = turn_delay_seconds if turn_delay_seconds is not None else config.gameplay.turn_delay_seconds

        # Adaptive knowledge management (always enabled)
        self.knowledge_update_interval = knowledge_update_interval if knowledge_update_interval is not None else config.orchestrator.knowledge_update_interval
        self.last_knowledge_update_turn = 0
        
        # Map updating (more frequent than full knowledge updates)
        self.map_update_interval = map_update_interval if map_update_interval is not None else config.orchestrator.map_update_interval
        self.last_map_update_turn = 0

        # Objective updating (fastest update cycle for goal discovery)
        self.objective_update_interval = objective_update_interval if objective_update_interval is not None else config.orchestrator.objective_update_interval

        # Objective Refinement Configuration
        self.enable_objective_refinement = enable_objective_refinement if enable_objective_refinement is not None else config.orchestrator.enable_objective_refinement
        self.objective_refinement_interval = objective_refinement_interval if objective_refinement_interval is not None else config.orchestrator.objective_refinement_interval
        self.max_objectives_before_forced_refinement = max_objectives_before_forced_refinement if max_objectives_before_forced_refinement is not None else config.orchestrator.max_objectives_before_forced_refinement
        self.refined_objectives_target_count = refined_objectives_target_count if refined_objectives_target_count is not None else config.orchestrator.refined_objectives_target_count
        self.last_objective_refinement_turn = last_objective_refinement_turn if last_objective_refinement_turn is not None else 0

        # Context management configuration
        self.max_context_tokens = config.orchestrator.max_context_tokens if hasattr(config.orchestrator, 'max_context_tokens') else 150000
        self.context_overflow_threshold = config.orchestrator.context_overflow_threshold if hasattr(config.orchestrator, 'context_overflow_threshold') else 0.8
        self.last_summarization_turn = 0

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
            logger=self.logger,
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

        # Discovered objectives tracking - maintains goals discovered through gameplay
        self.discovered_objectives = []
        self.completed_objectives = []  # Track completed objectives for learning
        self.objective_update_turn = 0
        # Add objective staleness tracking
        self.objective_staleness_tracker = {}  # Track turns since progress on each objective
        self.last_location_for_staleness = None
        self.last_score_for_staleness = 0

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

        # Try to load previous state and merge learning data
        previous_state = self._load_previous_state()
        self._merge_previous_state(previous_state)

        # Try to restore from save file if it exists
        game_was_restored = self._attempt_restore_from_save(zork_interface_instance)
        
        if game_was_restored:
            # Get current state from restored game to reconcile with our tracking
            current_game_state = zork_interface_instance.send_command("look")
            
            # Get current inventory and score from restored game
            current_inventory, _ = zork_interface_instance.inventory_with_response()
            self.current_inventory = current_inventory
            
            current_zork_score, max_zork_score = zork_interface_instance.score()
            self.previous_zork_score = current_zork_score
            
            self.logger.info(f"Game restored - Score: {current_zork_score}, Inventory: {len(current_inventory)} items", extra={
                "extras": {
                    "event_type": "game_state_reconciled",
                    "episode_id": self.episode_id,
                    "score": current_zork_score,
                    "inventory_count": len(current_inventory),
                    "inventory": current_inventory
                }
            })

        # Enable verbose mode to get full room descriptions on every visit
        verbose_response = zork_interface_instance.send_command("verbose")
        self.logger.info(
            f"Enabled verbose mode: {verbose_response}",
            extra={
                "extras": {
                    "event_type": "verbose_mode_enabled",
                    "episode_id": self.episode_id,
                    "verbose_response": verbose_response,
                    "game_was_restored": game_was_restored,
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

            # Check for save signal from external process (like manage_ec2.py)
            save_signal_processed = self._handle_save_signal(zork_interface_instance)
            if save_signal_processed:
                # Save signal was processed, continue with normal gameplay
                # The system is ready for shutdown but will complete current turn
                pass

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

            # Add discovered objectives to the context if any exist
            if self.discovered_objectives:
                objectives_text = "\n--- Current Discovered Objectives ---\n"
                objectives_text += "🎯 Based on your recent gameplay patterns, you have discovered these objectives:\n"
                for i, obj in enumerate(self.discovered_objectives, 1):
                    objectives_text += f"  {i}. {obj}\n"
                objectives_text += "\n⚠️ FOCUS ON THESE OBJECTIVES when choosing your next action. Prioritize actions that advance these discovered goals rather than aimless exploration.\n"
                
                # Append to relevant memories
                if relevant_memories:
                    relevant_memories += objectives_text
                else:
                    relevant_memories = objectives_text.strip()

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
            # Get available exits from most recent extraction for spatial awareness
            current_exits = []
            if self.memory_log_history:
                last_extraction = self.memory_log_history[-1]
                current_exits = getattr(last_extraction, "exits", [])
            
            critic_response = self.critic.get_robust_evaluation(
                game_state_text=current_game_state,
                proposed_action=agent_action,
                available_exits=current_exits,
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
                            "critic_confidence": critic_confidence,
                            "recent_locations": [
                                getattr(entry, "current_location_name", "")
                                for entry in self.memory_log_history[-10:]
                                if hasattr(entry, 'current_location_name')
                            ],
                            "recent_actions": [
                                action for action, _ in self.action_history[-8:]
                            ],
                            "recent_critic_scores": [
                                reasoning.get("critic_score", 0.0)
                                for reasoning in self.action_reasoning_history[-5:]
                            ],
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
                        available_exits=current_exits,
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
            
            # Check for discovered objectives update every turn
            # Ensure agent_reasoning is available; default to empty string if not
            current_agent_reasoning = agent_reasoning if 'agent_reasoning' in locals() else ""
            self._check_objective_update(current_agent_reasoning)
            
            # Check for objective completion after each turn
            self._check_objective_completion(action_taken, next_game_state, llm_extracted_info if 'llm_extracted_info' in locals() else None)
            
            # Check for objective refinement
            self._check_objective_refinement()
            
            # Check for objective staleness
            self._check_objective_staleness()

            # Check for context overflow and trigger summarization if needed
            self._check_context_overflow()
            
            # Consolidate fragmented map locations only when new rooms have been added
            if self.game_map.needs_consolidation():
                try:
                    # First, perform map consolidation to merge similar locations
                    consolidations = self.game_map.consolidate_similar_locations()
                    if consolidations > 0:
                        self.logger.info(
                            f"Map consolidation completed: {consolidations} locations merged",
                            extra={
                                "extras": {
                                    "event_type": "map_consolidation",
                                    "episode_id": self.episode_id,
                                    "turn": self.turn_count,
                                    "consolidations_performed": consolidations,
                                }
                            },
                        )
                
                    # Enhanced base name consolidation to address main fragmentation source
                    base_consolidations = self.game_map.consolidate_base_name_variants()
                    if base_consolidations > 0:
                        self.logger.info(
                            f"Enhanced base name consolidation completed: {base_consolidations} locations merged",
                            extra={
                                "extras": {
                                    "event_type": "base_name_consolidation",
                                    "episode_id": self.episode_id,
                                    "turn": self.turn_count,
                                    "base_consolidations_performed": base_consolidations,
                                }
                            },
                        )
                
                    # Then, prune fragmented nodes that serve no navigation purpose
                    pruned_nodes = self.game_map.prune_fragmented_nodes()
                    if pruned_nodes > 0:
                        self.logger.info(
                            f"Map pruning completed: {pruned_nodes} fragmented nodes removed",
                            extra={
                                "extras": {
                                    "event_type": "map_pruning",
                                    "episode_id": self.episode_id,
                                    "turn": self.turn_count,
                                    "nodes_pruned": pruned_nodes,
                                }
                            },
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to consolidate map: {e}")

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

        # Perform final adaptive knowledge update if there's been significant progress
        self._perform_final_knowledge_update()

        # Check for periodic adaptive knowledge updates
        # Every 100 turns during gameplay (not just at episode end)
        if (
            self.turn_count - self.last_knowledge_update_turn
            >= self.knowledge_update_interval
        ):
            # Update knowledge during gameplay
            update_success = (
                self.adaptive_knowledge_manager.update_knowledge_from_turns(
                    episode_id=self.episode_id,
                    start_turn=self.last_knowledge_update_turn + 1,
                    end_turn=self.turn_count,
                    is_final_update=False,
                )
            )
            
            if update_success:
                self.last_knowledge_update_turn = self.turn_count


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

        # ALWAYS run consolidation every turn to prevent fragmentation buildup
        self._run_map_consolidation()
        
        # Check if enough turns have passed since last full map update (knowledge base sync)
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

            # Update map in knowledge base (but consolidation already ran above)
            self._update_knowledge_base_map()
            self.last_map_update_turn = self.turn_count

    def _run_map_consolidation(self) -> None:
        """Run map consolidation every turn to prevent fragmentation buildup."""
        if not self.game_map:
            return
            
        try:
            # Enhanced base name consolidation to address main fragmentation source
            base_consolidations = self.game_map.consolidate_base_name_variants()
            if base_consolidations > 0:
                self.logger.info(
                    f"Map consolidation (turn {self.turn_count}): {base_consolidations} locations merged",
                    extra={
                        "extras": {
                            "event_type": "turn_based_consolidation",
                            "episode_id": self.episode_id,
                            "turn": self.turn_count,
                            "base_consolidations_performed": base_consolidations,
                        }
                    },
                )
            
            # Also run legacy consolidation for any remaining case variations
            if self.game_map.needs_consolidation():
                consolidations = self.game_map.consolidate_similar_locations()
                if consolidations > 0:
                    self.logger.info(
                        f"Legacy consolidation (turn {self.turn_count}): {consolidations} locations merged",
                        extra={
                            "extras": {
                                "event_type": "turn_based_legacy_consolidation",
                                "episode_id": self.episode_id,
                                "turn": self.turn_count,
                                "consolidations_performed": consolidations,
                            }
                        },
                    )
            
            # Prune fragmented nodes that serve no navigation purpose
            pruned_nodes = self.game_map.prune_fragmented_nodes()
            if pruned_nodes > 0:
                self.logger.info(
                    f"Map pruning (turn {self.turn_count}): {pruned_nodes} fragmented nodes removed",
                    extra={
                        "extras": {
                            "event_type": "turn_based_pruning",
                            "episode_id": self.episode_id,
                            "turn": self.turn_count,
                            "nodes_pruned": pruned_nodes,
                        }
                    },
                )
                
        except Exception as e:
            self.logger.warning(
                f"Failed to run map consolidation on turn {self.turn_count}: {e}",
                extra={
                    "extras": {
                        "event_type": "map_consolidation_failed",
                        "episode_id": self.episode_id,
                        "turn": self.turn_count,
                        "error": str(e),
                    }
                },
            )

    def _update_knowledge_base_map(self) -> None:
        """Update the mermaid map in the knowledge base."""
        if not self.adaptive_knowledge_manager:
            return
            
        try:
            # Note: Consolidation and pruning already ran in _run_map_consolidation()
            # This method now focuses on syncing the clean map to the knowledge base
            
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
        config = get_config()
        
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
                
                # NEW: Track the failed exit in the map for potential pruning
                if config.gameplay.enable_exit_pruning and self.game_map:
                    failure_count = self.game_map.track_exit_failure(from_room, action)
                    
                    # Check if this exit should be pruned
                    if failure_count >= config.gameplay.exit_failure_threshold:
                        pruned_count = self.game_map.prune_invalid_exits(
                            from_room, config.gameplay.exit_failure_threshold
                        )
                        
                        if pruned_count > 0:
                            self.logger.info(
                                f"Exit pruning triggered for {from_room}: {pruned_count} invalid exits removed",
                                extra={
                                    "extras": {
                                        "event_type": "exit_pruning",
                                        "episode_id": self.episode_id,
                                        "turn": self.turn_count,
                                        "room": from_room,
                                        "failed_action": action,
                                        "failure_count": failure_count,
                                        "exits_pruned": pruned_count,
                                    }
                                },
                            )

        # Update room context for next turn
        self.prev_room_for_prompt_context = from_room
        self.action_leading_to_current_room_for_prompt_context = action

    def get_current_state(self) -> Dict[str, Any]:
        """Get comprehensive current state for export."""
        
        # Ensure current_room matches the actual room key in the map for proper highlighting
        # The orchestrator may track unique IDs that get consolidated, so we need to find the actual key
        actual_current_room = self.current_room_name_for_map
        
        # If the tracked room name doesn't exist in the map, find the best match
        if actual_current_room not in self.game_map.rooms:
            # Try to find a room that contains or is contained in the tracked name
            for room_key in self.game_map.rooms.keys():
                # Check if the base names match (e.g., "Kitchen" matches "Kitchen (3-way: east-up-west)")
                base_tracked = actual_current_room.split('(')[0].strip()
                base_room = room_key.split('(')[0].strip()
                
                if base_tracked.lower() == base_room.lower():
                    actual_current_room = room_key
                    break
        
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
                "location": actual_current_room,  # Use the matched room key
                "inventory": self.current_inventory,
                "in_combat": self._get_combat_status(),
                "death_count": self.death_count,
                "discovered_objectives": self.discovered_objectives,
                "completed_objectives": self.completed_objectives,
                "objective_update_turn": self.objective_update_turn,
            },
            "recent_log": self.get_recent_log(20),
            "map": {
                "mermaid_diagram": self.game_map.render_mermaid(),
                "current_room": actual_current_room,  # Use the matched room key for highlighting
                "total_rooms": len(self.game_map.rooms),
                "total_connections": sum(
                    len(connections)
                    for connections in self.game_map.connections.values()
                ),
                # Enhanced map metrics
                "quality_metrics": self.game_map.get_map_quality_metrics(),
                "confidence_report": self.game_map.render_confidence_report(),
                "fragmentation_report": self.game_map.get_fragmentation_report(),
                # Exit failure tracking
                "exit_failure_stats": self.game_map.get_exit_failure_stats(),
                "exit_failure_report": self.game_map.render_exit_failure_report(),
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

            # Remove the mermaid diagram section more precisely
            # Look for the pattern: ## CURRENT WORLD MAP followed by ```mermaid...```
            
            # Pattern to match the map section with mermaid diagram
            # This matches from "## CURRENT WORLD MAP" through the closing ```
            pattern = r'## CURRENT WORLD MAP\s*\n\s*```mermaid\s*\n.*?\n```'
            
            # Remove the mermaid diagram section while preserving other content
            knowledge_only = re.sub(pattern, '', content, flags=re.DOTALL)
            
            # Clean up any extra whitespace that might be left
            knowledge_only = re.sub(r'\n\s*\n\s*\n', '\n\n', knowledge_only)

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

    def _check_context_overflow(self) -> bool:
        """
        Monitor context size and trigger summarization if needed.
        
        Inspired by the Pokemon agent's context management approach.
        
        Returns:
            True if summarization was triggered, False otherwise
        """
        # Estimate total context tokens
        estimated_tokens = self._estimate_context_tokens()
        
        if estimated_tokens > (self.max_context_tokens * self.context_overflow_threshold):
            turns_since_last = self.turn_count - self.last_summarization_turn
            
            # Only summarize if we have meaningful content since last summarization
            if turns_since_last >= 20:  # Minimum turns before summarization
                print(f"🧠 Context overflow detected ({estimated_tokens} tokens), triggering summarization...")
                self._trigger_context_summarization()
                return True
                
        return False

    def _estimate_context_tokens(self) -> int:
        """
        Estimate total context tokens based on memory log history.
        
        Uses the shared token estimation utility.
        """
        return estimate_context_tokens(
            memory_history=self.memory_log_history,
            reasoning_history=self.action_reasoning_history,
            knowledge_base_path="knowledgebase.md"
        )

    def _trigger_context_summarization(self) -> None:
        """
        Generate a summary of recent gameplay and reset context.
        
        Similar to the Pokemon agent's summarization approach but tailored for Zork.
        """
        try:
            # Generate summary of recent progress
            summary = self._generate_gameplay_summary()
            
            # Create condensed memory log from summary
            condensed_memory = {
                "turn": self.turn_count,
                "type": "context_summary",
                "summary": summary,
                "turns_summarized": self.turn_count - self.last_summarization_turn,
                "timestamp": datetime.now().isoformat()
            }
            
            # Clear detailed memory but preserve recent critical information
            recent_critical_memories = self._extract_critical_memories(last_n_turns=10)
            
            # Reset memory log to summary + critical recent memories
            self.memory_log_history = [condensed_memory] + recent_critical_memories
            
            # Reset action reasoning history but preserve recent critic scores
            recent_reasoning = self.action_reasoning_history[-10:] if len(self.action_reasoning_history) > 10 else self.action_reasoning_history
            self.action_reasoning_history = recent_reasoning
            
            self.last_summarization_turn = self.turn_count
            
            print(f"  ✅ Context summarized, preserved {len(recent_critical_memories)} critical memories")
            
        except Exception as e:
            print(f"  ⚠️ Failed to perform context summarization: {e}")

    def _generate_gameplay_summary(self) -> str:
        """Generate a comprehensive summary of recent gameplay progress."""
        if not hasattr(self, 'client'):
            return "Summary generation unavailable (no LLM client)"
            
        # Prepare summary prompt
        recent_turns = self.memory_log_history[-50:] if len(self.memory_log_history) > 50 else self.memory_log_history
        
        summary_prompt = f"""Analyze the following Zork gameplay session and provide a comprehensive summary:

EPISODE ID: {self.episode_id}
TURNS COVERED: {self.last_summarization_turn + 1} to {self.turn_count}
CURRENT SCORE: {self.previous_zork_score}
DEATH COUNT: {self.death_count}

RECENT GAMEPLAY DATA:
{json.dumps(recent_turns, indent=2)}

CURRENT MAP STATE:
{self.game_map.generate_mermaid_diagram()}

Please provide a summary that includes:
1. Major discoveries and progress made
2. Key items obtained and used
3. Important locations visited and mapped
4. Puzzles solved or attempted
5. Deaths and what caused them
6. Strategic insights learned
7. Current objectives and next steps

Format as a clear, structured summary that preserves essential information for continued gameplay."""

        try:
            messages = [{"role": "user", "content": summary_prompt}]
            
            response = self.client.chat.completions.create(
                model=self.adaptive_knowledge_manager.analysis_model,
                messages=messages,
                **self.adaptive_knowledge_manager.analysis_sampling.model_dump(exclude_unset=True)
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"  ⚠️ Failed to generate LLM summary, using fallback: {e}")
            return self._generate_fallback_summary()

    def _extract_critical_memories(self, last_n_turns: int = 10) -> List[Dict]:
        """Extract the most critical memories from recent turns."""
        if len(self.memory_log_history) <= last_n_turns:
            return self.memory_log_history
            
        recent_memories = self.memory_log_history[-last_n_turns:]
        critical_memories = []
        
        for memory in recent_memories:
            # Preserve memories with important events
            if self._is_critical_memory(memory):
                critical_memories.append(memory)
                
        return critical_memories

    def _is_critical_memory(self, memory: Dict) -> bool:
        """Determine if a memory contains critical information that should be preserved."""
        memory_str = str(memory).lower()
        
        critical_indicators = [
            "death", "died", "killed", "grue",
            "new item", "took", "picked up",
            "opened", "unlocked", "solved",
            "score increased", "points",
            "new location", "room", "area",
            "combat", "fight", "attack",
            "puzzle", "riddle", "problem"
        ]
        
        return any(indicator in memory_str for indicator in critical_indicators)

    def _generate_fallback_summary(self) -> str:
        """Generate a basic summary without LLM assistance."""
        recent_actions = [memory.get("action", "") for memory in self.memory_log_history[-20:]]
        recent_locations = list(set([memory.get("location", "") for memory in self.memory_log_history[-20:] if memory.get("location")]))
        
        return f"""Gameplay Summary (Turns {self.last_summarization_turn + 1}-{self.turn_count}):
- Score: {self.previous_zork_score}
- Deaths: {self.death_count}
- Recent actions: {', '.join(recent_actions[-10:])}
- Locations visited: {', '.join(recent_locations)}
- Total turns: {self.turn_count}
"""

    def _immediate_knowledge_update(self, section_id: str, content: str, trigger_reason: str) -> None:
        """
        Perform immediate knowledge update for critical discoveries.
        
        Inspired by the Pokemon agent's runtime knowledge updates.
        Used for high-priority information that shouldn't wait for the next scheduled update.
        
        Args:
            section_id: Knowledge section to update (e.g., "dangers", "items")
            content: The critical information to add
            trigger_reason: Why this immediate update was triggered
        """
        try:
            success = self.adaptive_knowledge_manager.update_knowledge_section(
                section_id=section_id,
                content=content,
                quality_score=8.0  # High quality for immediate updates
            )
            
            if success:
                self.logger.info(
                    f"Immediate knowledge update triggered: {trigger_reason}",
                    extra={
                        "extras": {
                            "event_type": "immediate_knowledge_update",
                            "episode_id": self.episode_id,
                            "section_id": section_id,
                            "trigger_reason": trigger_reason,
                            "turn": self.turn_count,
                        }
                    },
                )
                
                # Reload agent knowledge immediately for current session benefit
                self._reload_agent_knowledge()
            else:
                self.logger.warning(
                    f"Failed immediate knowledge update for {trigger_reason}",
                    extra={
                        "extras": {
                            "event_type": "immediate_knowledge_update_failed",
                            "episode_id": self.episode_id,
                            "section_id": section_id,
                            "trigger_reason": trigger_reason,
                        }
                    },
                )
                
        except Exception as e:
            self.logger.error(
                f"Error during immediate knowledge update: {e}",
                extra={
                    "extras": {
                        "event_type": "immediate_knowledge_update_error",
                        "episode_id": self.episode_id,
                        "error": str(e),
                        "trigger_reason": trigger_reason,
                    }
                },
            )

    def _check_objective_update(self, current_agent_reasoning: str = "") -> None:
        """Check if it's time for an objective update and perform it if needed."""
        # Debug logging to help diagnose issues
        print(f"🔍 Objective update check: turn={self.turn_count}, interval={self.objective_update_interval}, last_update={self.objective_update_turn}")
        
        # Update objectives every turn, ensuring it's not a duplicate call for the same turn.
        if self.turn_count > 0 and self.turn_count != self.objective_update_turn:
            print(f"🎯 Triggering objective update at turn {self.turn_count}")
            self._update_discovered_objectives(current_agent_reasoning)
        else:
            print(f"🔍 Objective update skipped: turn_count={self.turn_count}, already updated this turn or turn 0.")

    def _update_discovered_objectives(self, current_agent_reasoning: str = "") -> None:
        """
        Use LLM to analyze recent gameplay and discover/update objectives.
        
        This maintains discovered objectives between turns while staying LLM-first.
        """
        try:
            print(f"🎯 Updating discovered objectives (turn {self.turn_count})...")
            
            # Get recent gameplay context for analysis
            recent_memory = self.memory_log_history[-20:] if len(self.memory_log_history) > 20 else self.memory_log_history
            recent_actions = self.action_history[-10:] if len(self.action_history) > 10 else self.action_history
            
            # Prepare context for LLM analysis
            gameplay_context = self._prepare_objective_analysis_context(recent_memory, recent_actions, current_agent_reasoning)
            
            # Create prompt for objective discovery/updating
            prompt = f"""Analyze the recent Zork gameplay to discover and maintain the agent's objectives.

CURRENT DISCOVERED OBJECTIVES:
{self.discovered_objectives if self.discovered_objectives else "None discovered yet"}

RECENTLY COMPLETED OBJECTIVES:
{[comp["objective"] for comp in self.completed_objectives[-5:]] if self.completed_objectives else "None completed yet"}

RECENT GAMEPLAY CONTEXT:
{gameplay_context}

CURRENT SCORE: {self.previous_zork_score}
CURRENT LOCATION: {self.current_room_name_for_map}
CURRENT INVENTORY: {self.current_inventory}

Based on this gameplay, identify the agent's discovered objectives. Look for:
1. **Score-increasing activities** (these reveal important objectives)
2. **Recurring patterns** in the agent's behavior that suggest goals
3. **Environmental clues** about what the agent should be doing
4. **Obstacles** that suggest significant rewards lie beyond them
5. **Items or locations** that appear strategically important

**IMPORTANT**: Consider the 'AGENT REASONING' provided in the context. If it contains new ideas, plans, or hypotheses for achieving goals, incorporate these into the objective list. However, if the reasoning is solely about escaping a loop, retrying a failed action, or simple exploration without a clear new goal, it should NOT lead to new objectives.

Update the objective list by:
- **Adding new objectives** discovered through recent gameplay patterns or valid agent reasoning
- **Updating existing objectives** with new information or progress
- **Removing objectives** that have been completed or proven incorrect
- **Prioritizing objectives** based on evidence of importance
- **Avoiding re-adding** objectives that were recently completed (listed above)

Format your response as:
OBJECTIVES:
- [objective 1]
- [objective 2]
- [etc.]

Focus on objectives the agent has actually discovered through gameplay patterns or its own novel reasoning, not general Zork knowledge."""

            # Log the full prompt for debugging
            self._log_objective_prompt(prompt)

            # Get LLM response
            if hasattr(self, 'client') and self.client:
                messages = [{"role": "user", "content": prompt}]
                
                model_to_use = self.adaptive_knowledge_manager.analysis_model if self.adaptive_knowledge_manager else "gpt-4"
                print(f"  🔍 Using model: {model_to_use}")
                print(f"  🔍 Prompt length: {len(prompt)} characters")
                print(f"  🔍 First 200 chars of prompt: {prompt[:200]}...")
                
                try:
                    response = self.client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        temperature=0.3,  # Lower temperature for consistent objective tracking
                    )
                    
                    print(f"  🔍 LLM call successful, response type: {type(response)}")
                    print(f"  🔍 Response content type: {type(response.content)}")
                    print(f"  🔍 Response content length: {len(response.content) if response.content else 0}")
                    
                    # Parse objectives from response
                    updated_objectives = self._parse_objectives_from_response(response.content)
                    
                    print(f"  🔍 Raw LLM response: '{response.content}'")
                    print(f"  🔍 Parsed objectives: {updated_objectives}")
                    
                    if updated_objectives:
                        self.discovered_objectives = updated_objectives
                        self.objective_update_turn = self.turn_count
                        
                        print(f"  ✅ Objectives updated: {len(updated_objectives)} objectives discovered")
                        for i, obj in enumerate(updated_objectives[:3], 1):  # Show first 3
                            print(f"    {i}. {obj}")
                        
                        # Log the update
                        self.logger.info(
                            "Discovered objectives updated",
                            extra={
                                "extras": {
                                    "event_type": "objectives_updated",
                                    "episode_id": self.episode_id,
                                    "turn": self.turn_count,
                                    "objective_count": len(updated_objectives),
                                    "objectives": updated_objectives,
                                }
                            },
                        )
                    else:
                        print("  ⚠️ No objectives parsed from LLM response")
                        
                except Exception as llm_error:
                    print(f"  ❌ LLM call failed: {llm_error}")
                    self.logger.warning(f"Objective LLM call failed: {llm_error}")
            else:
                print("  ⚠️ No LLM client available for objective analysis")
                
        except Exception as e:
            print(f"  ⚠️ Failed to update objectives: {e}")
            self.logger.warning(f"Objective update failed: {e}")

    def _prepare_objective_analysis_context(self, recent_memory, recent_actions, current_agent_reasoning) -> str:
        """Prepare gameplay context for objective analysis."""
        context_parts = []
        
        # Add recent actions and responses
        if recent_actions:
            context_parts.append("RECENT ACTIONS:")
            for action, response in recent_actions[-5:]:  # Last 5 actions
                context_parts.append(f"  Action: {action}")
                context_parts.append(f"  Result: {response[:200]}...")  # Truncate long responses
                context_parts.append("")
        
        # Add notable events from memory
        if recent_memory:
            notable_events = []
            for memory in recent_memory:
                if hasattr(memory, 'important_messages') and memory.important_messages:
                    for msg in memory.important_messages:
                        if any(keyword in msg.lower() for keyword in ['score', 'points', 'treasure', 'lamp', 'door', 'open', 'take']):
                            notable_events.append(msg)
            
            if notable_events:
                context_parts.append("NOTABLE EVENTS:")
                for event in notable_events[-10:]:  # Last 10 notable events
                    context_parts.append(f"  - {event}")
                context_parts.append("")
        
        # Add score changes
        score_changes = []
        for memory in recent_memory:
            # Check if this memory entry indicates a score change
            if hasattr(memory, 'important_messages'):
                for msg in memory.important_messages:
                    if 'score' in msg.lower() or 'points' in msg.lower():
                        score_changes.append(msg)
        
        if score_changes:
            context_parts.append("SCORE CHANGES:")
            for change in score_changes:
                context_parts.append(f"  - {change}")
            context_parts.append("")
        
        # Add agent reasoning if it doesn't seem to be solely about escaping a loop
        if current_agent_reasoning:
            # Keywords indicating loop-escaping or simple re-attempts, not new objectives
            loop_keywords = [
                'stuck', 'loop', 'repeat', 'try again', 'another way', 'instead',
                'alternative', 'avoid repeating', 'different action', 'failed before'
            ]
            # Keywords indicating new ideas or goals
            idea_keywords = [
                'idea', 'maybe if', 'i should try', 'plan to', 'goal is to', 'what if',
                'perhaps', 'new approach', 'strategy', 'objective is to', 'hypothesize'
            ]

            reasoning_lower = current_agent_reasoning.lower()
            is_loop_related = any(keyword in reasoning_lower for keyword in loop_keywords)
            has_new_idea = any(keyword in reasoning_lower for keyword in idea_keywords)

            # Include reasoning if it has new ideas OR if it's not clearly loop-related
            # This prioritizes including potentially useful reasoning unless it's obviously just about getting unstuck.
            if has_new_idea or not is_loop_related:
                context_parts.append("AGENT REASONING (Current Turn):")
                context_parts.append(f"  {current_agent_reasoning}")
                context_parts.append("")
            else:
                context_parts.append("AGENT REASONING (Current Turn - Filtered as loop-related):")
                context_parts.append(f"  {current_agent_reasoning}") # Still include for LLM to see it was considered
                context_parts.append("")
         
        return "\n".join(context_parts)

    def _parse_objectives_from_response(self, response: str) -> List[str]:
        """Parse objectives from the LLM response."""
        try:
            objectives = []
            lines = response.strip().split('\n')
            
            # Look for the OBJECTIVES: section
            in_objectives_section = False
            for line in lines:
                line = line.strip()
                
                if line.upper().startswith('OBJECTIVES:'):
                    in_objectives_section = True
                    continue
                
                if in_objectives_section:
                    # Stop if we hit another section header
                    if line.endswith(':') and len(line.split()) <= 3:
                        break
                    
                    # Look for bullet points
                    if line.startswith('- ') or line.startswith('* '):
                        objective = line[2:].strip()
                        if objective and len(objective) > 5:  # Filter out very short entries
                            objectives.append(objective)
            
            return objectives
            
        except Exception as e:
            self.logger.warning(f"Failed to parse objectives from response: {e}")
            return []

    def _check_objective_completion(self, action_taken: str, game_response: str, extracted_info) -> None:
        """Check if any discovered objectives have been completed this turn."""
        if not self.discovered_objectives:
            return
            
        try:
            # Look for completion signals in the game response and context
            score_change = self.previous_zork_score - getattr(self, '_last_score_for_completion_check', self.previous_zork_score)
            self._last_score_for_completion_check = self.previous_zork_score
            
            completion_signals = []
            
            # Score increase is a strong completion signal
            if score_change > 0:
                completion_signals.append(f"Score increased by {score_change} points")
            
            # Check for completion keywords in game response
            completion_keywords = [
                "well done", "congratulations", "you have", "successfully", 
                "completed", "solved", "unlocked", "opened", "found", 
                "treasure", "victory", "accomplished"
            ]
            
            response_lower = game_response.lower()
            for keyword in completion_keywords:
                if keyword in response_lower:
                    completion_signals.append(f"Response contains completion keyword: '{keyword}'")
            
            # Check for location/inventory changes that might indicate completion
            if extracted_info:
                # New location reached
                if (hasattr(extracted_info, 'current_location_name') and 
                    extracted_info.current_location_name != self.current_room_name_for_map):
                    completion_signals.append(f"Reached new location: {extracted_info.current_location_name}")
                
                # New items acquired  
                if hasattr(extracted_info, 'inventory'):
                    new_items = set(extracted_info.inventory) - set(self.current_inventory)
                    if new_items:
                        completion_signals.append(f"Acquired new items: {', '.join(new_items)}")
            
            # If we have completion signals, check objectives against them
            if completion_signals:
                self._evaluate_objective_completion(action_taken, completion_signals)
                
        except Exception as e:
            self.logger.warning(f"Failed to check objective completion: {e}")

    def _evaluate_objective_completion(self, action_taken: str, completion_signals: List[str]) -> None:
        """Use LLM to evaluate if any objectives were completed based on completion signals."""
        try:
            context = f"""
ACTION TAKEN: {action_taken}

COMPLETION SIGNALS:
{chr(10).join(f"- {signal}" for signal in completion_signals)}

CURRENT OBJECTIVES:
{chr(10).join(f"- {obj}" for obj in self.discovered_objectives)}

RECENT CONTEXT:
- Current Location: {self.current_room_name_for_map}
- Current Score: {self.previous_zork_score}
- Turn: {self.turn_count}
"""

            prompt = f"""Analyze if any of the current objectives were completed this turn.

{context}

Based on the action taken and completion signals, determine if any objectives were achieved. Be conservative - only mark objectives as completed if there's clear evidence of completion.

Look for:
- Direct achievement of stated objectives
- Score increases that correspond to objective completion
- Game responses that indicate success
- Location/item changes that fulfill objectives

Respond with:
COMPLETED: [list any completed objectives exactly as stated, or "None" if no clear completions]
REASONING: [brief explanation of why each objective was marked complete]

Only mark objectives as completed if you're confident they were achieved."""

            if hasattr(self, 'client') and self.client:
                messages = [{"role": "user", "content": prompt}]
                
                response = self.client.chat.completions.create(
                    model=self.adaptive_knowledge_manager.analysis_model if self.adaptive_knowledge_manager else "gpt-4",
                    messages=messages,
                    temperature=0.2,  # Low temperature for consistency
                    max_tokens=300
                )
                
                # Parse completed objectives
                completed_objectives = self._parse_completed_objectives(response.content)
                
                if completed_objectives:
                    self._mark_objectives_complete(completed_objectives, action_taken, completion_signals)
                    
        except Exception as e:
            self.logger.warning(f"Failed to evaluate objective completion: {e}")

    def _parse_completed_objectives(self, response: str) -> List[str]:
        """Parse completed objectives from LLM response."""
        try:
            completed = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line.upper().startswith('COMPLETED:'):
                    completed_text = line.split(':', 1)[1].strip()
                    if completed_text.lower() != "none":
                        # Split by commas and clean up
                        objectives = [obj.strip() for obj in completed_text.split(',')]
                        for obj in objectives:
                            if obj and obj in self.discovered_objectives:
                                completed.append(obj)
                    break
            
            return completed
            
        except Exception as e:
            self.logger.warning(f"Failed to parse completed objectives: {e}")
            return []

    def _mark_objectives_complete(self, completed_objectives: List[str], action_taken: str, completion_signals: List[str]) -> None:
        """Mark objectives as completed and move them to completed list."""
        try:
            for objective in completed_objectives:
                if objective in self.discovered_objectives:
                    # Remove from active objectives
                    self.discovered_objectives.remove(objective)
                    
                    # Add to completed objectives with context
                    completion_record = {
                        "objective": objective,
                        "completed_turn": self.turn_count,
                        "completion_action": action_taken,
                        "completion_signals": completion_signals,
                        "completion_score": self.previous_zork_score
                    }
                    self.completed_objectives.append(completion_record)
                    
                    print(f"  ✅ Objective completed: {objective}")
                    
                    # Log the completion
                    self.logger.info(
                        f"Objective completed: {objective}",
                        extra={
                            "extras": {
                                "event_type": "objective_completed",
                                "episode_id": self.episode_id,
                                "turn": self.turn_count,
                                "objective": objective,
                                "completion_action": action_taken,
                                "completion_signals": completion_signals,
                                "completion_score": self.previous_zork_score,
                            }
                        },
                    )
                    
        except Exception as e:
            self.logger.warning(f"Failed to mark objectives complete: {e}")

    def _log_objective_prompt(self, prompt_content: str) -> None:
        """Log the full objective generation prompt to a temporary file."""
        try:
            # Ensure tmp directory exists
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            
            # Simple counter for unique filenames, reset per orchestrator instance might be okay for debugging
            if not hasattr(self, '_objective_prompt_counter'):
                self._objective_prompt_counter = 0
            self._objective_prompt_counter += 1
            
            filename = f"tmp/objective_prompt_t{self.turn_count}_{self._objective_prompt_counter:03d}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== OBJECTIVE GENERATION PROMPT (Turn: {self.turn_count}, Episode: {self.episode_id}) ===\\n")
                model_to_use = self.adaptive_knowledge_manager.analysis_model if self.adaptive_knowledge_manager else "gpt-4" # Fallback for safety
                f.write(f"Model: {model_to_use}\\n")
                # Assuming sampling params are accessible or hardcoded for this debug log
                f.write(f"Temperature: 0.3 (default for objective update)\\n") 
                f.write("=" * 70 + "\\n\\n")
                f.write(prompt_content)
            
            if self.logger:
                self.logger.info(f"Objective prompt logged to {filename}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to log objective prompt: {e}")

    def _check_objective_refinement(self) -> None:
        """Checks if objectives need refinement and triggers it."""
        if not self.enable_objective_refinement:
            return

        time_for_scheduled_refinement = (self.turn_count - self.last_objective_refinement_turn) >= self.objective_refinement_interval
        forced_refinement_due_to_length = len(self.discovered_objectives) > self.max_objectives_before_forced_refinement

        if self.turn_count > 0 and (time_for_scheduled_refinement or forced_refinement_due_to_length):
            if time_for_scheduled_refinement:
                reason = f"interval met ({self.objective_refinement_interval} turns)"
            else:
                reason = f"max objectives exceeded ({len(self.discovered_objectives)} > {self.max_objectives_before_forced_refinement})"
            
            self.logger.info(
                f"Triggering objective refinement at turn {self.turn_count}. Reason: {reason}",
                extra={
                    "extras": {
                        "event_type": "objective_refinement_triggered",
                        "episode_id": self.episode_id,
                        "turn": self.turn_count,
                        "current_objective_count": len(self.discovered_objectives),
                        "reason": reason,
                    }
                }
            )
            self._refine_discovered_objectives()
            self.last_objective_refinement_turn = self.turn_count

    def _refine_discovered_objectives(self) -> None:
        """Uses an LLM to refine the current list of discovered objectives."""
        if not self.discovered_objectives:
            self.logger.info("No discovered objectives to refine.")
            return

        self.logger.info(f"🎯 Refining {len(self.discovered_objectives)} objectives (target: ~{self.refined_objectives_target_count})...")

        # Prepare a brief game state summary for context
        game_summary_parts = [
            f"- Current Location: {self.current_room_name_for_map}",
            f"- Current Score: {self.previous_zork_score}",
            f"- Turn: {self.turn_count}",
        ]
        if self.current_inventory:
            game_summary_parts.append(f"- Key Inventory: {', '.join(self.current_inventory[:5])}{'...' if len(self.current_inventory) > 5 else ''}")

        # Consider adding a brief summary from knowledgebase if available and concise
        # For now, keeping it simple to avoid excessive token usage here.

        game_state_summary = "\\n".join(game_summary_parts)

        # Corrected prompt construction
        objectives_str = "\\n".join([f"- {idx + 1}. {obj}" for idx, obj in enumerate(self.discovered_objectives)])
        completed_objectives_str = "\\n".join([f"- {comp['objective']}" for comp in self.completed_objectives[-10:]]) if self.completed_objectives else "None recently completed"

        refined_objectives_prompt = f"""You are tasked with refining a list of discovered objectives for a Zork gameplay session. The agent has discovered too many objectives and needs to focus on the most promising ones.

Current Game State:
{game_state_summary}

Current Objectives (to be refined):
{objectives_str}

Recently Completed Objectives:
{completed_objectives_str}

**CRITICAL: Loop and Stagnation Detection**
Review the current objectives carefully for signs of:
1. **Repetitive patterns** that may cause the agent to get stuck in loops
2. **Overly specific puzzle objectives** that may require tools/knowledge not yet available
3. **Location-specific goals** that trap the agent in one area (e.g., "experiment with egg/tree")
4. **Parser interaction objectives** that attempt complex commands the game doesn't understand

**Refinement Guidelines:**
1. **Prioritize exploration objectives** over specific puzzle-solving objectives
2. **Remove objectives** that seem to encourage repetitive behavior in one location
3. **Keep objectives** that promote visiting new areas or finding new items
4. **Prefer general goals** over highly specific interaction patterns
5. **Target ~{self.refined_objectives_target_count} objectives** total

**Examples of problematic objectives to REMOVE:**
- "Experiment with the egg's fragile clasp using specific tools"
- "Test vertical movement possibilities from specific location" 
- "Try different verbs with [specific object] at [specific location]"

**Examples of good objectives to KEEP:**
- "Explore unmapped areas to find new locations"
- "Look for light sources to safely explore dark areas"
- "Find keys or tools that can unlock doors or containers"
- "Locate and collect valuable items or treasures"

Please provide a refined list of objectives that encourages exploration and progress while avoiding repetitive loops."""


        try:
            if hasattr(self, 'client') and self.client:
                messages = [{"role": "user", "content": refined_objectives_prompt}]
                # Use analysis_model and sampling parameters similar to knowledge generation
                # Fallback to a default model if not configured
                model_to_use = self.adaptive_knowledge_manager.analysis_model if self.adaptive_knowledge_manager and self.adaptive_knowledge_manager.analysis_model else "gpt-4-turbo" 
                sampling_params = self.adaptive_knowledge_manager.analysis_sampling.model_dump(exclude_unset=True) if self.adaptive_knowledge_manager else {"temperature": 0.5, "max_tokens": 1000}


                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    **sampling_params
                )

                refined_objectives = self._parse_objectives_from_response(response.content)

                if refined_objectives:
                    self.logger.info(f"Objective list refined from {len(self.discovered_objectives)} to {len(refined_objectives)} objectives.")
                    self.discovered_objectives = refined_objectives
                elif response.content.strip().upper() == "OBJECTIVES:\nNONE":
                    self.logger.info("Objective list refined to None as per LLM response.")
                    self.discovered_objectives = []
                else:
                    self.logger.warning(f"Objective refinement LLM call returned an unexpected or empty response. Raw: '{response.content[:200]}...' No changes made to objectives.")

            else:
                self.logger.warning("No LLM client available for objective refinement.")
        except Exception as e:
            self.logger.error(f"Error during objective refinement: {e}", exc_info=True)

    def _check_objective_staleness(self) -> None:
        """Check for stale objectives and remove them if no progress has been made."""
        if not self.discovered_objectives:
            return
            
        current_location = self.current_room_name_for_map
        current_score = self.previous_zork_score
        
        # Detect if we're making progress (location change or score increase)
        made_progress = (
            current_location != self.last_location_for_staleness or
            current_score > self.last_score_for_staleness
        )
        
        # Update staleness tracking for all objectives
        for objective in self.discovered_objectives[:]:  # Copy list to allow modification
            if objective not in self.objective_staleness_tracker:
                self.objective_staleness_tracker[objective] = 0
            
            if made_progress:
                # Reset staleness counter if we made any progress
                self.objective_staleness_tracker[objective] = 0
            else:
                # Increment staleness counter
                self.objective_staleness_tracker[objective] += 1
                
                # Remove objectives that have been stale for too long (30+ turns without progress)
                if self.objective_staleness_tracker[objective] >= 30:
                    self.discovered_objectives.remove(objective)
                    del self.objective_staleness_tracker[objective]
                    
                    print(f"🗑️ Removed stale objective (30+ turns without progress): {objective}")
                    
                    self.logger.info(
                        f"Objective removed due to staleness: {objective}",
                        extra={
                            "extras": {
                                "event_type": "objective_staleness_removal",
                                "episode_id": self.episode_id,
                                "turn": self.turn_count,
                                "stale_objective": objective,
                                "turns_without_progress": 30,
                            }
                        },
                    )
        
        # Update tracking variables
        self.last_location_for_staleness = current_location
        self.last_score_for_staleness = current_score

    def _handle_save_signal(self, zork_interface_instance) -> bool:
        """Check for and handle save signal from external process (like manage_ec2.py).
        
        Returns:
            bool: True if save signal was processed (regardless of success)
        """
        if os.path.exists(self.save_signal_file_abs_path):
            self.logger.info("Save requested by external signal", extra={
                "extras": {
                    "event_type": "save_signal_received",
                    "episode_id": self.episode_id,
                    "turn": self.turn_count,
                    "signal_file": self.save_signal_file_abs_path
                }
            })
            
            # Attempt to save the Zork game state
            save_success = zork_interface_instance.trigger_zork_save(self.zork_save_filename)
            
            if save_success:
                self.logger.info("Zork game state saved successfully", extra={
                    "extras": {
                        "event_type": "zork_save_success",
                        "episode_id": self.episode_id,
                        "save_file": self.zork_save_file_abs_path
                    }
                })
                
                # Force save current ZorkGPT state to JSON
                self.export_current_state()
                
                self.logger.info("ZorkGPT state exported - ready for shutdown", extra={
                    "extras": {
                        "event_type": "state_export_complete",
                        "episode_id": self.episode_id,
                        "export_file": self.state_export_file
                    }
                })
            else:
                self.logger.error("Failed to save Zork game state", extra={
                    "extras": {
                        "event_type": "zork_save_failed",
                        "episode_id": self.episode_id,
                        "save_file": self.zork_save_file_abs_path
                    }
                })
            
            # Remove signal file regardless of save success/failure
            try:
                os.remove(self.save_signal_file_abs_path)
                self.logger.info("Save signal file removed", extra={
                    "extras": {
                        "event_type": "save_signal_removed",
                        "episode_id": self.episode_id
                    }
                })
            except OSError as e:
                self.logger.warning(f"Failed to remove save signal file: {e}", extra={
                    "extras": {
                        "event_type": "save_signal_removal_failed",
                        "episode_id": self.episode_id,
                        "error": str(e)
                    }
                })
            
            return True
        
        return False

    def _attempt_restore_from_save(self, zork_interface_instance) -> bool:
        """Attempt to restore from a previous save file.
        
        Returns:
            bool: True if restore was successful
        """
        # Check for save file with .qzl extension (Zork adds this automatically)
        save_file_with_qzl = self.zork_save_file_abs_path + ".qzl"
        
        if os.path.exists(save_file_with_qzl):
            save_file_to_use = save_file_with_qzl
        elif os.path.exists(self.zork_save_file_abs_path):
            save_file_to_use = self.zork_save_file_abs_path
        else:
            self.logger.info("No save file found - starting fresh game", extra={
                "extras": {
                    "event_type": "no_save_file",
                    "episode_id": self.episode_id,
                    "save_file": self.zork_save_file_abs_path,
                    "save_file_with_qzl": save_file_with_qzl
                }
            })
            return False
        
        self.logger.info("Save file found - attempting restore", extra={
            "extras": {
                "event_type": "save_file_found",
                "episode_id": self.episode_id,
                "save_file": save_file_to_use
            }
        })
        
        restore_success = zork_interface_instance.trigger_zork_restore(self.zork_save_filename)
        
        if restore_success:
            self.logger.info("Successfully restored from save file", extra={
                "extras": {
                    "event_type": "restore_success",
                    "episode_id": self.episode_id,
                    "save_file": save_file_to_use
                }
            })
            return True
        else:
            self.logger.error("Failed to restore from save file", extra={
                "extras": {
                    "event_type": "restore_failed",
                    "episode_id": self.episode_id,
                    "save_file": save_file_to_use
                }
            })
            
            # Delete corrupt save file
            try:
                os.remove(save_file_to_use)
                self.logger.info("Removed corrupt save file", extra={
                    "extras": {
                        "event_type": "corrupt_save_removed",
                        "episode_id": self.episode_id,
                        "save_file": save_file_to_use
                    }
                })
            except OSError as e:
                self.logger.warning(f"Failed to remove corrupt save file: {e}", extra={
                    "extras": {
                        "event_type": "corrupt_save_removal_failed",
                        "episode_id": self.episode_id,
                        "error": str(e)
                    }
                })
            
            return False

    def _load_previous_state(self) -> Optional[Dict[str, Any]]:
        """Load previous state from current_state.json if it exists.
        
        Returns:
            Dict containing previous state or None if not found/invalid
        """
        try:
            if os.path.exists(self.state_export_file):
                with open(self.state_export_file, 'r') as f:
                    previous_state = json.load(f)
                
                self.logger.info("Loaded previous state from JSON", extra={
                    "extras": {
                        "event_type": "previous_state_loaded",
                        "episode_id": self.episode_id,
                        "state_file": self.state_export_file,
                        "previous_episode_id": previous_state.get("metadata", {}).get("episode_id", "unknown"),
                        "previous_turn_count": previous_state.get("metadata", {}).get("turn_count", 0)
                    }
                })
                
                return previous_state
            else:
                self.logger.info("No previous state file found", extra={
                    "extras": {
                        "event_type": "no_previous_state",
                        "episode_id": self.episode_id,
                        "state_file": self.state_export_file
                    }
                })
                return None
                
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"Failed to load previous state: {e}", extra={
                "extras": {
                    "event_type": "previous_state_load_failed",
                    "episode_id": self.episode_id,
                    "error": str(e),
                    "state_file": self.state_export_file
                }
            })
            return None

    def _merge_previous_state(self, previous_state: Dict[str, Any]) -> None:
        """Merge relevant data from previous state into current session.
        
        Preserves learning (map, knowledge, objectives) while allowing fresh game state.
        """
        if not previous_state:
            return
        
        # Preserve map data
        if "map" in previous_state:
            try:
                # Load previous map data into current map
                map_data = previous_state["map"]
                if "raw_data" in map_data and "rooms" in map_data["raw_data"]:
                    for room_name, room_info in map_data["raw_data"]["rooms"].items():
                        self.game_map.add_room(room_name)
                        if "exits" in room_info:
                            self.game_map.update_room_exits(room_name, room_info["exits"])
                
                if "raw_data" in map_data and "connections" in map_data["raw_data"]:
                    for from_room, connections in map_data["raw_data"]["connections"].items():
                        for direction, to_room in connections.items():
                            self.game_map.add_connection(from_room, direction, to_room)
                
                self.logger.info("Merged previous map data", extra={
                    "extras": {
                        "event_type": "map_data_merged",
                        "episode_id": self.episode_id,
                        "rooms_loaded": len(map_data["raw_data"].get("rooms", {})),
                        "connections_loaded": sum(len(conns) for conns in map_data["raw_data"].get("connections", {}).values())
                    }
                })
            except Exception as e:
                self.logger.warning(f"Failed to merge map data: {e}", extra={
                    "extras": {
                        "event_type": "map_merge_failed",
                        "episode_id": self.episode_id,
                        "error": str(e)
                    }
                })
        
        # Preserve knowledge base
        if "knowledge_base" in previous_state:
            try:
                kb_content = previous_state["knowledge_base"].get("content", "")
                if kb_content and len(kb_content.strip()) > 50:  # Only if substantial content
                    with open("knowledgebase.md", "w") as f:
                        f.write(kb_content)
                    
                    # Update adaptive knowledge manager
                    self.adaptive_knowledge_manager.last_content = kb_content
                    
                    self.logger.info("Restored previous knowledge base", extra={
                        "extras": {
                            "event_type": "knowledge_restored",
                            "episode_id": self.episode_id,
                            "content_length": len(kb_content)
                        }
                    })
            except Exception as e:
                self.logger.warning(f"Failed to restore knowledge base: {e}", extra={
                    "extras": {
                        "event_type": "knowledge_restore_failed",
                        "episode_id": self.episode_id,
                        "error": str(e)
                    }
                })
        
        # Preserve death count and other persistent stats
        if "current_state" in previous_state:
            current_state = previous_state["current_state"]
            if "death_count" in current_state:
                self.death_count = current_state["death_count"]
                self.logger.info(f"Restored death count: {self.death_count}", extra={
                    "extras": {
                        "event_type": "death_count_restored",
                        "episode_id": self.episode_id,
                        "death_count": self.death_count
                    }
                })
        
        self.logger.info("Previous state merge completed", extra={
            "extras": {
                "event_type": "state_merge_complete",
                "episode_id": self.episode_id
            }
        })


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
