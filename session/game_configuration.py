"""
Game configuration management for ZorkGPT orchestration.

This module provides a clean, typed interface to game configuration,
replacing the complex parameter handling in the original ZorkOrchestrator.__init__.
"""

from dataclasses import dataclass, field
from typing import Optional
from config import get_config, get_client_api_key


@dataclass
class GameConfiguration:
    """
    Typed configuration object for ZorkGPT orchestration.
    
    Handles configuration precedence: explicit parameters > config file > defaults.
    Provides clean, typed access to all configuration values.
    """
    
    # Core game settings
    max_turns_per_episode: int = 5000
    turn_delay_seconds: float = 0.0
    
    # File paths
    episode_log_file: str = "zork_episode_log.txt"
    json_log_file: str = "zork_episode_log.jsonl"
    state_export_file: str = "current_state.json"
    
    # Game server
    game_server_url: str = "http://localhost:8000"
    
    # LLM client settings
    client_base_url: Optional[str] = None
    client_api_key: Optional[str] = None
    
    # Model specifications
    agent_model: Optional[str] = None
    critic_model: Optional[str] = None
    info_ext_model: Optional[str] = None
    
    # Update intervals
    knowledge_update_interval: int = 100
    map_update_interval: int = 50
    objective_update_interval: int = 20
    
    # Objective refinement
    enable_objective_refinement: bool = True
    objective_refinement_interval: int = 200
    max_objectives_before_forced_refinement: int = 15
    refined_objectives_target_count: int = 10
    
    # Context management
    max_context_tokens: int = 150000
    context_overflow_threshold: float = 0.8
    
    # State export
    enable_state_export: bool = True
    s3_bucket: Optional[str] = None
    s3_key_prefix: str = "zorkgpt/"
    
    @classmethod
    def create(
        cls,
        # Allow explicit parameter overrides
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
    ) -> "GameConfiguration":
        """
        Create GameConfiguration with parameter precedence handling.
        
        Args:
            **kwargs: Optional parameter overrides
            
        Returns:
            GameConfiguration instance with proper precedence:
            explicit parameters > config file > class defaults
        """
        config = get_config()
        
        return cls(
            # Core game settings
            max_turns_per_episode=max_turns_per_episode if max_turns_per_episode is not None else config.orchestrator.max_turns_per_episode,
            turn_delay_seconds=turn_delay_seconds if turn_delay_seconds is not None else config.gameplay.turn_delay_seconds,
            
            # File paths
            episode_log_file=episode_log_file if episode_log_file is not None else config.files.episode_log_file,
            json_log_file=json_log_file if json_log_file is not None else config.files.json_log_file,
            state_export_file=state_export_file if state_export_file is not None else config.files.state_export_file,
            
            # Game server
            game_server_url=game_server_url if game_server_url is not None else "http://localhost:8000",
            
            # LLM client settings
            client_base_url=client_base_url,
            client_api_key=client_api_key if client_api_key is not None else get_client_api_key(),
            
            # Model specifications
            agent_model=agent_model,
            critic_model=critic_model,
            info_ext_model=info_ext_model,
            
            # Update intervals
            knowledge_update_interval=knowledge_update_interval if knowledge_update_interval is not None else config.orchestrator.knowledge_update_interval,
            map_update_interval=map_update_interval if map_update_interval is not None else config.orchestrator.map_update_interval,
            objective_update_interval=objective_update_interval if objective_update_interval is not None else config.orchestrator.objective_update_interval,
            
            # Objective refinement
            enable_objective_refinement=enable_objective_refinement if enable_objective_refinement is not None else config.orchestrator.enable_objective_refinement,
            objective_refinement_interval=objective_refinement_interval if objective_refinement_interval is not None else config.orchestrator.objective_refinement_interval,
            max_objectives_before_forced_refinement=max_objectives_before_forced_refinement if max_objectives_before_forced_refinement is not None else config.orchestrator.max_objectives_before_forced_refinement,
            refined_objectives_target_count=refined_objectives_target_count if refined_objectives_target_count is not None else config.orchestrator.refined_objectives_target_count,
            
            # Context management
            max_context_tokens=getattr(config.orchestrator, 'max_context_tokens', 150000),
            context_overflow_threshold=getattr(config.orchestrator, 'context_overflow_threshold', 0.8),
            
            # State export
            enable_state_export=enable_state_export if enable_state_export is not None else config.orchestrator.enable_state_export,
            s3_bucket=s3_bucket if s3_bucket is not None else config.aws.s3_bucket,
            s3_key_prefix=s3_key_prefix if s3_key_prefix is not None else config.aws.s3_key_prefix,
        )
    
    def get_effective_api_key(self) -> str:
        """Get the effective API key (parameter override or default)."""
        return self.client_api_key if self.client_api_key is not None else get_client_api_key()
    
    def get_llm_base_url_for_model(self, model_type: str) -> str:
        """
        Get the effective base URL for a specific model type.
        
        Args:
            model_type: One of 'agent', 'critic', 'info_ext', 'analysis'
            
        Returns:
            Base URL for the model type
        """
        if self.client_base_url:
            return self.client_base_url
            
        config = get_config()
        return config.llm.get_base_url_for_model(model_type)