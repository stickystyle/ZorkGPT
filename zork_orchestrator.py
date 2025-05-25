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
import environs
import os
import json

from openai import OpenAI
from zork_api import ZorkInterface
from map_graph import MapGraph
from movement_analyzer import MovementAnalyzer, MovementContext
from logger import setup_logging, ZorkExperienceTracker

# Import our refactored modules with aliases to avoid conflicts
from zork_agent import ZorkAgent as AgentModule
from zork_extractor import ZorkExtractor, ExtractorResponse
from zork_critic import ZorkCritic, CriticResponse
from zork_strategy_generator import create_integrated_knowledge_base

# Optional S3 support
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# Load environment variables
env = environs.Env()
env.read_env()


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
        episode_log_file: str = "zork_episode_log.txt",
        json_log_file: str = "zork_episode_log.jsonl",
        experiences_file: str = "zork_experiences.json",
        max_turns_per_episode: int = 500,
        client_base_url: str = None,
        client_api_key: str = None,
        # Automatic knowledge base updating
        auto_update_knowledge: bool = True,
        # State export configuration
        enable_state_export: bool = True,
        state_export_file: str = "current_state.json",
        s3_bucket: str = None,
        s3_key_prefix: str = "zorkgpt/",
    ):
        """Initialize the ZorkOrchestrator with all subsystems."""
        # Store configuration
        self.episode_log_file = episode_log_file
        self.json_log_file = json_log_file
        self.experiences_file = experiences_file

        # Game settings
        self.max_turns_per_episode = max_turns_per_episode
        self.auto_update_knowledge = auto_update_knowledge

        # State export configuration
        self.enable_state_export = enable_state_export
        self.state_export_file = state_export_file
        self.s3_bucket = s3_bucket or env.str("ZORK_S3_BUCKET", None)
        self.s3_key_prefix = s3_key_prefix
        
        # Initialize S3 client if available and configured
        self.s3_client = None
        if S3_AVAILABLE and self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                if self.logger:
                    self.logger.info(f"S3 export enabled for bucket: {self.s3_bucket}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to initialize S3 client: {e}")

        # Initialize logger and experience tracker
        self.logger = setup_logging(episode_log_file, json_log_file)
        self.experience_tracker = ZorkExperienceTracker()

        # Initialize OpenAI client (shared across components)
        self.client = OpenAI(
            base_url=client_base_url or env.str("CLIENT_BASE_URL", None),
            api_key=client_api_key or env.str("CLIENT_API_KEY", None),
        )

        # Initialize core components
        self.agent = AgentModule(
            model=agent_model, client=self.client, logger=self.logger
        )

        self.extractor = ZorkExtractor(
            model=info_ext_model, client=self.client, logger=self.logger
        )

        self.critic = ZorkCritic(
            model=critic_model, client=self.client, logger=self.logger
        )

        # Initialize shared movement analyzer
        self.movement_analyzer = MovementAnalyzer()

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
        self.total_episode_reward = 0
        self.game_map = MapGraph()
        self.current_room_name_for_map = ""
        self.prev_room_for_prompt_context: Optional[str] = None
        self.action_leading_to_current_room_for_prompt_context: Optional[str] = None
        self.current_inventory = []

        # Reset movement analyzer for new episode
        self.movement_analyzer.clear_pending_connections()

        # Agent reasoning tracking for state export
        self.action_reasoning_history = []

        # Update episode IDs in components
        self.agent.update_episode_id(self.episode_id)
        self.extractor.update_episode_id(self.episode_id)
        self.critic.update_episode_id(self.episode_id)

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
            return self.experience_tracker.get_experiences(), 0

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

            # Check for action rejection and override logic
            was_overridden = False
            rejection_threshold = self.critic.trust_tracker.get_rejection_threshold()

            if critic_score_val < rejection_threshold:
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
                            }
                        },
                    )
                else:
                    # Track rejected action and get a new one
                    self.critic.rejection_system.rejected_actions_this_turn.append(
                        agent_action
                    )

                    # Get new action from agent with reasoning
                    agent_response = self.agent.get_action_with_reasoning(
                        game_state_text=current_game_state
                        + f"\n\n[Previous action '{agent_action}' was rejected by critic: {critic_justification}]",
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

            # Store reasoning for state export
            self.action_reasoning_history.append({
                "turn": self.turn_count,
                "action": agent_action,
                "reasoning": agent_reasoning,
                "critic_score": critic_score_val,
                "was_overridden": was_overridden
            })

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
                    self.action_history.append((action_taken, next_game_state))

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

                self.action_history.append((action_taken, next_game_state))

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

                # Use shared MovementAnalyzer for consistent movement detection
                movement_context = MovementContext(
                    current_location=final_current_room_name,
                    previous_location=room_before_action,
                    action=action_taken,
                    game_response=next_game_state,
                    turn_number=self.turn_count,
                )

                # Analyze potential connections
                movement_result = self.movement_analyzer.analyze_movement(
                    movement_context
                )
                if movement_result.connection_created:
                    # Add connection to map
                    self.game_map.add_connection(
                        movement_result.from_location,
                        movement_result.action,
                        movement_result.to_location,
                    )

                    # Log the connection
                    self.logger.info(
                        f"Movement connection: {movement_result.from_location} --[{movement_result.action}]--> {movement_result.to_location}",
                        extra={
                            "extras": {
                                "event_type": "movement_connection_created",
                                "episode_id": self.episode_id,
                                "from_room": movement_result.from_location,
                                "to_room": movement_result.to_location,
                                "action": movement_result.action,
                                "confidence": getattr(
                                    movement_result, "confidence", 1.0
                                ),
                            }
                        },
                    )

                # Get score
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

            # Determine Reward & Game Over
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
            ):
                if self.current_room_name_for_map not in self.visited_locations:
                    exploration_bonus = 0.5
                    reward += exploration_bonus
                    self.visited_locations.add(self.current_room_name_for_map)

            # Check for Zork's internal score changes
            score_change = current_zork_score_val - self.previous_zork_score
            if score_change > 0:
                score_bonus = score_change * 2  # Amplify positive score changes
                reward += score_bonus
                self.logger.info(
                    f"Score increased by {score_change} points! Bonus reward: +{score_bonus:.2f}",
                    extra={
                        "extras": {
                            "event_type": "score_increase",
                            "episode_id": self.episode_id,
                            "score_change": score_change,
                            "new_score": current_zork_score_val,
                            "bonus_reward": score_bonus,
                        }
                    },
                )
            elif score_change < 0:
                score_penalty = abs(score_change) * 1.5  # Penalize score losses
                reward -= score_penalty

            self.previous_zork_score = current_zork_score_val
            self.total_episode_reward += reward

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

            # Reset critic rejection system for next turn
            self.critic.rejection_system.reset_turn()

            # Export current state after each turn
            self.export_current_state()

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
                    "total_reward": self.total_episode_reward,
                    "final_max_turns": self.max_turns_per_episode,
                    # Performance metrics
                    "avg_critic_score": self.get_avg_critic_score(),
                }
            },
        )

        # Save experiences and optionally update knowledge base
        self.experience_tracker.save_experiences(self.experiences_file)

        if self.auto_update_knowledge:
            try:
                create_integrated_knowledge_base(self.json_log_file)
                self.logger.info("Knowledge base updated successfully")
            except Exception as e:
                self.logger.warning(f"Knowledge base update failed: {e}")

        return self.experience_tracker.get_experiences(), self.previous_zork_score



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
                "game_over": False,  # TODO: Track this properly
                "score": self.previous_zork_score,
                "total_reward": self.total_episode_reward,
                "max_turns": self.max_turns_per_episode
            },
            "current_state": {
                "location": self.current_room_name_for_map,
                "inventory": self.current_inventory,
                "in_combat": self._get_combat_status(),
            },
            "recent_log": self.get_recent_log(20),
            "map": {
                "mermaid_diagram": self.game_map.render_mermaid(),
                "current_room": self.current_room_name_for_map,
                "total_rooms": len(self.game_map.rooms),
                "total_connections": sum(len(connections) for connections in self.game_map.connections.values()),
                # Optional: Include raw data for advanced frontends
                "raw_data": {
                    "rooms": {name: {"exits": list(room.exits)} for name, room in self.game_map.rooms.items()},
                    "connections": self.game_map.connections
                }
            },
            "knowledge_base": self.get_knowledge_base_summary(),
            "performance": {
                "avg_critic_score": self.get_avg_critic_score(),
                "recent_actions": self.get_recent_action_summary()
            }
        }

    def get_recent_log(self, length: int = 10) -> List[Dict[str, Any]]:
        """Get recent game log entries with reasoning."""
        recent_log = []
        
        # Get the most recent entries from action_history and memory_log_history
        recent_actions = self.action_history[-length:] if self.action_history else []
        recent_extractions = self.memory_log_history[-length:] if self.memory_log_history else []
        recent_reasoning = self.action_reasoning_history[-length:] if self.action_reasoning_history else []
        
        # Combine them by turn (assuming they're in sync)
        for i, (action, zork_response) in enumerate(recent_actions):
            turn_num = self.turn_count - len(recent_actions) + i + 1
            
            log_entry = {
                "turn": turn_num,
                "action": action,
                "zork_response": zork_response
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
                log_entry["was_overridden"] = reasoning_entry["was_overridden"]
            
            # Add extraction info if available
            if i < len(recent_extractions):
                extraction = recent_extractions[i]
                if hasattr(extraction, 'model_dump'):
                    log_entry["extracted_info"] = extraction.model_dump()
                else:
                    log_entry["extracted_info"] = {
                        "current_location_name": getattr(extraction, 'current_location_name', ''),
                        "exits": getattr(extraction, 'exits', []),
                        "visible_objects": getattr(extraction, 'visible_objects', []),
                        "important_messages": getattr(extraction, 'important_messages', [])
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
                "last_updated": os.path.getmtime("knowledgebase.md") if os.path.exists("knowledgebase.md") else None
            }
        except Exception:
            return {"content": "No knowledge base available", "last_updated": None}

    def _get_combat_status(self) -> bool:
        """Determine if currently in combat based on recent extractions."""
        if not self.memory_log_history:
            return False
        
        recent_extraction = self.memory_log_history[-1]
        return getattr(recent_extraction, 'in_combat', False)

    def get_avg_critic_score(self) -> float:
        """Get average critic score for recent turns."""
        if not hasattr(self, "experience_tracker") or not self.experience_tracker.experiences:
            return 0.0
        
        recent_experiences = self.experience_tracker.experiences[-10:]  # Last 10 turns
        critic_scores = [exp.get("critic_score", 0.0) for exp in recent_experiences]
        
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
                ContentType='application/json',
                CacheControl='no-cache, must-revalidate'  # Ensure fresh data
            )
            
            # Optional: Also save timestamped snapshot
            timestamp_key = f"{self.s3_key_prefix}snapshots/{state['metadata']['episode_id']}/turn_{state['metadata']['turn_count']}.json"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=timestamp_key,
                Body=json.dumps(state, indent=2),
                ContentType='application/json'
            )
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to upload state to S3: {e}")


if __name__ == "__main__":
    # Create ZorkOrchestrator instance with default settings
    orchestrator = ZorkOrchestrator()

    with ZorkInterface(timeout=1.0) as zork_game:
        try:
            episode_experiences, final_score = orchestrator.play_episode(zork_game)
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
