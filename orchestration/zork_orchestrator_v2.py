"""
Streamlined ZorkOrchestrator v2 - Clean orchestration layer.

This is the refactored orchestrator that coordinates specialized managers
instead of handling all responsibilities directly. It follows the
orchestration pattern, delegating work to focused manager classes.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from session.game_state import GameState
from session.game_configuration import GameConfiguration
from managers import (
    ObjectiveManager, 
    KnowledgeManager, 
    MapManager, 
    StateManager, 
    ContextManager, 
    EpisodeSynthesizer
)
from zork_agent import ZorkAgent
from zork_critic import ZorkCritic
from hybrid_zork_extractor import HybridZorkExtractor
from game_server_client import GameServerClient
from logger import setup_logging
from collections import Counter


class ZorkOrchestratorV2:
    """
    Streamlined orchestrator that coordinates specialized managers.
    
    This class is responsible for:
    - High-level game loop coordination
    - Manager initialization and lifecycle
    - Inter-manager communication
    - Game interface management
    
    All domain-specific logic is delegated to specialized managers.
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
        knowledge_update_interval: int = None,
        map_update_interval: int = None,
        objective_update_interval: int = None,
        enable_state_export: bool = None,
        state_export_file: str = None,
        s3_bucket: str = None,
        s3_key_prefix: str = None,
        turn_delay_seconds: float = None,
        game_server_url: str = None,
        enable_objective_refinement: bool = None,
        objective_refinement_interval: int = None,
        max_objectives_before_forced_refinement: int = None,
        refined_objectives_target_count: int = None,
    ):
        """Initialize the orchestrator with configuration and dependencies."""
        
        # Create configuration object with proper precedence
        self.config = GameConfiguration.create(
            agent_model=agent_model,
            critic_model=critic_model,
            info_ext_model=info_ext_model,
            episode_log_file=episode_log_file,
            json_log_file=json_log_file,
            max_turns_per_episode=max_turns_per_episode,
            client_base_url=client_base_url,
            client_api_key=client_api_key,
            knowledge_update_interval=knowledge_update_interval,
            map_update_interval=map_update_interval,
            objective_update_interval=objective_update_interval,
            enable_state_export=enable_state_export,
            state_export_file=state_export_file,
            s3_bucket=s3_bucket,
            s3_key_prefix=s3_key_prefix,
            turn_delay_seconds=turn_delay_seconds,
            game_server_url=game_server_url,
            enable_objective_refinement=enable_objective_refinement,
            objective_refinement_interval=objective_refinement_interval,
            max_objectives_before_forced_refinement=max_objectives_before_forced_refinement,
            refined_objectives_target_count=refined_objectives_target_count,
        )
        
        # Initialize logger
        self.logger = setup_logging(
            self.config.episode_log_file,
            self.config.json_log_file
        )
        
        # Initialize shared game state
        self.game_state = GameState()
        
        # Initialize core game components
        self._initialize_game_components()
        
        # Initialize managers
        self._initialize_managers()
        
        # Track critic confidence for synthesis decisions
        self.critic_confidence_history = []
        
        self.logger.info(
            "ZorkOrchestrator v2 initialized",
            extra={
                "event_type": "orchestrator_init",
                "agent_model": self.config.agent_model,
                "critic_model": self.config.critic_model,
                "info_ext_model": self.config.info_ext_model,
                "max_turns": self.config.max_turns_per_episode,
            }
        )
    
    def _initialize_game_components(self) -> None:
        """Initialize core game components (agent, critic, extractor)."""
        # Initialize agent
        self.agent = ZorkAgent(
            logger=self.logger,
            episode_id=self.game_state.episode_id,
            model=self.config.agent_model
        )
        
        # Initialize critic  
        self.critic = ZorkCritic(
            logger=self.logger,
            episode_id=self.game_state.episode_id,
            model=self.config.critic_model
        )
        
        # Initialize extractor
        self.extractor = HybridZorkExtractor(
            episode_id=self.game_state.episode_id,
            logger=self.logger,
            model=self.config.info_ext_model
        )
    
    def _initialize_managers(self) -> None:
        """Initialize all specialized managers."""
        # Initialize managers in dependency order
        
        # Map manager (no dependencies)
        self.map_manager = MapManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state
        )
        
        # Context manager (no dependencies)
        self.context_manager = ContextManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state
        )
        
        # State manager (needs potential S3 client)
        self.state_manager = StateManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            llm_client=self.agent.client  # Share LLM client
        )
        
        # Knowledge manager (needs agent and map references)
        self.knowledge_manager = KnowledgeManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            agent=self.agent,
            game_map=self.map_manager.game_map,
            json_log_file=self.config.json_log_file
        )
        
        # Objective manager (needs knowledge manager reference)
        self.objective_manager = ObjectiveManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            adaptive_knowledge_manager=self.knowledge_manager.adaptive_knowledge_manager
        )
        
        # Episode synthesizer (needs references to other managers)
        self.episode_synthesizer = EpisodeSynthesizer(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            knowledge_manager=self.knowledge_manager,
            state_manager=self.state_manager,
            llm_client=self.agent.client
        )
        
        # Create ordered manager list for processing
        self.managers = [
            self.map_manager,
            self.context_manager,
            self.state_manager,
            self.objective_manager,
            self.knowledge_manager,
            self.episode_synthesizer
        ]
    
    def create_game_interface(self) -> GameServerClient:
        """Create game interface for the orchestrator to use."""
        return GameServerClient(
            base_url=self.config.game_server_url
        )
    
    def play_episode(self, game_interface: GameServerClient) -> int:
        """
        Play a complete episode of Zork.
        
        Args:
            game_interface: The game interface to use
            
        Returns:
            Final score achieved in the episode
        """
        try:
            # Generate new episode ID (orchestrator owns episode lifecycle)
            episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            # Initialize new episode across all managers
            self.episode_synthesizer.initialize_episode(
                episode_id=episode_id,
                agent=self.agent,
                extractor=self.extractor,
                critic=self.critic
            )
            
            # Start new game session
            session_response = game_interface.start_session(session_id=episode_id)
            if not session_response["success"]:
                self.logger.error(f"Failed to start game session: {session_response}")
                return 0
            
            # Get initial game state
            initial_state_response = game_interface.send_command("look")
            if not initial_state_response["success"]:
                self.logger.error(f"Failed to get initial state: {initial_state_response}")
                return 0
            
            initial_game_state = initial_state_response["response"]
            
            # Extract initial state information
            initial_extracted_info = self.extractor.extract_info(initial_game_state)
            self._process_extraction(initial_extracted_info, "", initial_game_state)
            
            # Run the main game loop
            final_score = self._run_game_loop(game_interface, initial_game_state)
            
            # Finalize episode
            self.episode_synthesizer.finalize_episode(
                final_score=final_score,
                critic_confidence_history=self.critic_confidence_history
            )
            
            # Stop game session
            game_interface.stop_session()
            
            return final_score
            
        except Exception as e:
            self.logger.error(
                f"Episode failed with exception: {e}",
                extra={
                    "event_type": "episode_exception",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e)
                }
            )
            return self.game_state.previous_zork_score
    
    def _run_game_loop(self, game_interface: GameServerClient, initial_state: str) -> int:
        """Run the main game loop."""
        current_game_state = initial_state
        
        while not self.game_state.game_over_flag and self.game_state.turn_count < self.config.max_turns_per_episode:
            self.game_state.turn_count += 1
            
            # Add turn delay if configured
            if self.config.turn_delay_seconds > 0:
                time.sleep(self.config.turn_delay_seconds)
            
            # Run a single turn
            action_taken, next_game_state = self._run_turn(game_interface, current_game_state)
            
            if next_game_state:
                current_game_state = next_game_state
            
            # Check periodic updates for managers
            self._check_periodic_updates()
            
            # Export state periodically
            if self.game_state.turn_count % 10 == 0:
                self.state_manager.export_current_state()
        
        # Log episode completion
        self.logger.info(
            f"Episode completed",
            extra={
                "event_type": "episode_completed",
                "episode_id": self.game_state.episode_id,
                "turn": self.game_state.turn_count,
                "final_score": self.game_state.previous_zork_score,
                "game_over": self.game_state.game_over_flag,
                "reason": "game_over" if self.game_state.game_over_flag else "max_turns"
            }
        )
        
        return self.game_state.previous_zork_score
    
    def _run_turn(self, game_interface: GameServerClient, current_state: str) -> Tuple[str, str]:
        """Run a single game turn."""
        try:
            # Generate action using agent
            agent_context = self.context_manager.get_agent_context(
                current_state=current_state,
                inventory=self.game_state.current_inventory,
                location=self.game_state.current_room_name_for_map,
                game_map=self.map_manager.game_map,
                in_combat=self.state_manager.get_combat_status(),
                failed_actions=self.game_state.failed_actions_by_location.get(
                    self.game_state.current_room_name_for_map, []
                ),
                discovered_objectives=self.game_state.discovered_objectives
            )
            
            # Format context for agent
            formatted_context = self.context_manager.get_formatted_agent_prompt_context(agent_context)
            
            # Get agent action
            agent_result = self.agent.get_action_with_reasoning(
                game_state_text=current_state,
                previous_actions_and_responses=agent_context.get('recent_actions', []),
                action_counts=agent_context.get('action_counts'),
                relevant_memories=formatted_context
            )
            
            proposed_action = agent_result["action"]
            agent_reasoning = agent_result.get("reasoning", "")
            
            # Add reasoning to context
            self.context_manager.add_reasoning(agent_reasoning, proposed_action)
            
            # Get critic evaluation
            critic_context = self.context_manager.get_critic_context(
                current_state=current_state,
                proposed_action=proposed_action,
                location=self.game_state.current_room_name_for_map,
                available_exits=self.map_manager.game_map.get_available_exits(
                    self.game_state.current_room_name_for_map
                ) if hasattr(self.map_manager.game_map, 'get_available_exits') else [],
                failed_actions=self.game_state.failed_actions_by_location.get(
                    self.game_state.current_room_name_for_map, []
                )
            )
            
            critic_result = self.critic.evaluate_action(
                game_state_text=current_state,
                proposed_action=proposed_action,
                available_exits=critic_context.get('available_exits', []),
                action_counts=self.game_state.action_counts
            )
            
            # Critic evaluates the action but doesn't change it
            # For now, we'll use the proposed action (could add rejection logic later)
            action_to_take = proposed_action
            confidence = critic_result.confidence
            self.critic_confidence_history.append(confidence)
            
            # Update action counts
            self.game_state.action_counts[action_to_take] += 1
            
            # Execute action
            response = game_interface.send_command(action_to_take)
            if not response["success"]:
                self.logger.error(f"Failed to execute command: {response}")
                return action_to_take, current_state
            
            next_game_state = response["response"]
            
            # Add action to history
            self.context_manager.add_action(action_to_take, next_game_state)
            
            # Extract information from response
            extracted_info = self.extractor.extract_info(next_game_state)
            self._process_extraction(extracted_info, action_to_take, next_game_state)
            
            # Check for objective completion
            self.objective_manager.check_objective_completion(
                action_taken=action_to_take,
                game_response=next_game_state,
                extracted_info=extracted_info
            )
            
            # Log turn completion
            self.logger.info(
                f"Turn {self.game_state.turn_count} completed",
                extra={
                    "event_type": "turn_completed",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "action": action_to_take,
                    "score": self.game_state.previous_zork_score,
                    "location": self.game_state.current_room_name_for_map,
                    "confidence": confidence
                }
            )
            
            return action_to_take, next_game_state
            
        except Exception as e:
            self.logger.error(
                f"Turn failed with exception: {e}",
                extra={
                    "event_type": "turn_exception",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e)
                }
            )
            return "look", current_state
    
    def _process_extraction(self, extracted_info, action: str, response: str) -> None:
        """Process extracted information and update game state."""
        # Add to memory
        self.context_manager.add_memory(extracted_info)
        
        # Update score if present
        if hasattr(extracted_info, 'score') and extracted_info.score is not None:
            self.game_state.previous_zork_score = extracted_info.score
        
        # Update inventory if present
        if hasattr(extracted_info, 'inventory') and extracted_info.inventory:
            self.game_state.current_inventory = extracted_info.inventory
        
        # Update game over flag
        if hasattr(extracted_info, 'game_over') and extracted_info.game_over:
            self.game_state.game_over_flag = True
        
        # Update location and map
        if hasattr(extracted_info, 'current_location_name') and extracted_info.current_location_name:
            new_location = extracted_info.current_location_name
            
            # Add to visited locations
            self.game_state.visited_locations.add(new_location)
            
            # Update map
            if action and self.game_state.current_room_name_for_map:
                self.map_manager.update_from_movement(
                    action_taken=action,
                    new_room_name=new_location,
                    previous_room_name=self.game_state.current_room_name_for_map,
                    game_response=response
                )
            elif not self.game_state.current_room_name_for_map:
                # Initial room
                self.map_manager.add_initial_room(new_location)
        
        # Track failed actions
        if action and response:
            response_lower = response.lower()
            failure_indicators = ["you can't", "impossible", "don't understand", "nothing happens"]
            
            if any(indicator in response_lower for indicator in failure_indicators):
                self.map_manager.track_failed_action(action, self.game_state.current_room_name_for_map)
    
    def _check_periodic_updates(self) -> None:
        """Check and run periodic updates for managers."""
        # Map consolidation (runs every turn)
        self.map_manager.process_turn()
        
        # Objective updates
        if self.objective_manager.should_process_turn():
            current_reasoning = ""
            if self.game_state.action_reasoning_history:
                current_reasoning = self.game_state.action_reasoning_history[-1].get("reasoning", "")
            self.objective_manager.process_periodic_updates(current_reasoning)
        
        # Knowledge updates
        if self.knowledge_manager.should_process_turn():
            self.knowledge_manager.check_periodic_update()
        
        # State management (context overflow)
        self.state_manager.process_turn()
    
    def run_multiple_episodes(self, num_episodes: int = 1) -> List[int]:
        """
        Run multiple episodes sequentially.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            List of final scores for each episode
        """
        scores = []
        game_interface = self.create_game_interface()
        
        for i in range(num_episodes):
            self.logger.info(f"Starting episode {i+1} of {num_episodes}")
            
            # Reset managers for new episode
            for manager in self.managers:
                manager.reset_episode()
            
            # Clear critic confidence history
            self.critic_confidence_history = []
            
            # Play episode
            score = self.play_episode(game_interface)
            scores.append(score)
            
            self.logger.info(f"Episode {i+1} completed with score: {score}")
            
            # Brief pause between episodes
            if i < num_episodes - 1:
                time.sleep(2)
        
        return scores
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        status = {
            "orchestrator": "v2",
            "episode_id": self.game_state.episode_id,
            "turn_count": self.game_state.turn_count,
            "game_over": self.game_state.game_over_flag,
            "score": self.game_state.previous_zork_score,
            "managers": {}
        }
        
        # Get status from each manager
        for manager in self.managers:
            manager_name = manager.__class__.__name__
            status["managers"][manager_name] = manager.get_status()
        
        return status