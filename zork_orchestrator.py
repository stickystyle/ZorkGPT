"""
ZorkOrchestrator module for coordinating Zork gameplay episodes.

This module contains the main game loop and ties together all other modules:
- ZorkAgent for action generation
- ZorkExtractor for information extraction
- ZorkCritic for action evaluation
- Movement analysis and mapping
- Logging and experience tracking
"""

from typing import List, Tuple, Optional
from collections import Counter
from datetime import datetime
import environs
import os

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
        """Initialize the ZorkOrchestrator with all subsystems."""
        # Store configuration
        self.episode_log_file = episode_log_file
        self.json_log_file = json_log_file
        self.experiences_file = experiences_file

        # Game settings
        self.base_max_turns_per_episode = max_turns_per_episode
        self.max_turns_per_episode = max_turns_per_episode

        # Dynamic turn limit configuration
        self.absolute_max_turns = absolute_max_turns
        self.turn_limit_increment = turn_limit_increment
        self.performance_check_interval = performance_check_interval
        self.performance_threshold = performance_threshold
        self.min_turns_for_increase = min_turns_for_increase
        self.auto_update_knowledge = auto_update_knowledge

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

        # Reset movement analyzer for new episode
        self.movement_analyzer.clear_pending_connections()

        # Reset dynamic turn limit for the new episode
        self.max_turns_per_episode = self.base_max_turns_per_episode

        # Performance tracking for dynamic turn limits
        self.critic_scores_history = []
        self.turn_limit_increases = 0
        self.last_performance_check_turn = 0

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

            # Get agent action
            agent_action = self.agent.get_action(
                game_state_text=current_game_state,
                previous_actions_and_responses=self.action_history[
                    -5:
                ],  # Last 5 actions
                action_counts=self.action_counts,
                relevant_memories=relevant_memories,
            )

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
                            "recent_critic_scores": self.critic_scores_history[-5:],
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

                    # Get new action from agent
                    agent_action = self.agent.get_action(
                        game_state_text=current_game_state
                        + f"\n\n[Previous action '{agent_action}' was rejected by critic: {critic_justification}]",
                        previous_actions_and_responses=self.action_history[-5:],
                        action_counts=self.action_counts,
                        relevant_memories=relevant_memories,
                    )

                    # Re-evaluate new action
                    critic_response = self.critic.get_robust_evaluation(
                        game_state_text=current_game_state,
                        proposed_action=agent_action,
                        action_counts=self.action_counts,
                        previous_actions_and_responses=self.action_history[-3:],
                    )
                    critic_score_val = critic_response.score
                    critic_justification = critic_response.justification

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
            self._evaluate_performance_and_adjust_turn_limit()

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

        # Save experiences and optionally update knowledge base
        self.experience_tracker.save_experiences(self.experiences_file)

        if self.auto_update_knowledge:
            try:
                create_integrated_knowledge_base(self.json_log_file)
                self.logger.info("Knowledge base updated successfully")
            except Exception as e:
                self.logger.warning(f"Knowledge base update failed: {e}")

        return self.experience_tracker.get_experiences(), self.previous_zork_score

    def _evaluate_performance_and_adjust_turn_limit(self) -> bool:
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
