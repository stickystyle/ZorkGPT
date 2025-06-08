"""
KnowledgeManager for ZorkGPT orchestration.

Handles all knowledge management responsibilities:
- Periodic knowledge updates and analysis from gameplay
- Integration with AdaptiveKnowledgeManager
- Knowledge base generation and maintenance
- Learning from gameplay patterns and synthesis
- Inter-episode wisdom synthesis and strategy updates
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from zork_strategy_generator import AdaptiveKnowledgeManager
from config import get_config


class KnowledgeManager(BaseManager):
    """
    Manages all knowledge-related functionality for ZorkGPT.
    
    Responsibilities:
    - Periodic knowledge updates from gameplay analysis
    - Final episode knowledge synthesis
    - Immediate knowledge updates for critical discoveries
    - Map updates in knowledge base
    - Agent knowledge reloading
    - Inter-episode wisdom synthesis
    """
    
    def __init__(
        self, 
        logger, 
        config: GameConfiguration, 
        game_state: GameState,
        agent,
        game_map,
        json_log_file: str = "zork_episode_log.jsonl"
    ):
        super().__init__(logger, config, game_state, "knowledge_manager")
        self.agent = agent
        self.game_map = game_map
        
        # Initialize AdaptiveKnowledgeManager
        self.adaptive_knowledge_manager = AdaptiveKnowledgeManager(
            log_file=json_log_file,
            output_file="knowledgebase.md",
            logger=logger
        )
        
        # Knowledge update tracking
        self.last_knowledge_update_turn = 0
    
    def reset_episode(self) -> None:
        """Reset knowledge manager state for a new episode."""
        self.last_knowledge_update_turn = 0
        self.log_debug("Knowledge manager reset for new episode")
    
    def process_turn(self) -> None:
        """Process knowledge management for the current turn."""
        # This is handled by process_periodic_updates
        pass
    
    def should_process_turn(self) -> bool:
        """Check if knowledge needs processing this turn."""
        # Check if it's time for a knowledge update
        turns_since_update = self.game_state.turn_count - self.last_knowledge_update_turn
        return (self.game_state.turn_count > 0 and 
                turns_since_update >= self.config.knowledge_update_interval)
    
    def check_periodic_update(self, current_agent_reasoning: str = "") -> None:
        """Check and perform periodic knowledge updates if needed."""
        if not self.should_process_turn():
            return
            
        try:
            self.log_progress(
                f"Starting periodic knowledge update at turn {self.game_state.turn_count}",
                stage="knowledge_update",
                details=f"Starting knowledge update at turn {self.game_state.turn_count}"
            )
            
            # Log that we're starting the update
            self.logger.info(
                f"Starting periodic knowledge update at turn {self.game_state.turn_count}",
                extra={
                    "event_type": "knowledge_update_start",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "last_update_turn": self.last_knowledge_update_turn,
                }
            )
            
            # Include map quality metrics for context
            map_metrics = {}
            if hasattr(self.game_map, 'get_quality_metrics'):
                try:
                    map_metrics = self.game_map.get_quality_metrics()
                    self.log_debug(f"Map quality metrics: {map_metrics}")
                except Exception as e:
                    self.log_warning(f"Failed to get map quality metrics: {e}")
            
            # Perform the knowledge update using "Method 2" - entire episode analysis
            self.log_debug("Calling adaptive knowledge manager update_knowledge_from_turns")
            success = self.adaptive_knowledge_manager.update_knowledge_from_turns(
                episode_id=self.game_state.episode_id,
                start_turn=1,
                end_turn=self.game_state.turn_count,
                is_final_update=False
            )
            
            if success:
                self.last_knowledge_update_turn = self.game_state.turn_count
                
                self.log_progress(
                    f"Knowledge update completed successfully at turn {self.game_state.turn_count}",
                    stage="knowledge_update",
                    details="Knowledge update completed successfully"
                )
                
                # Log successful update
                self.logger.info(
                    f"Knowledge update completed successfully at turn {self.game_state.turn_count}",
                    extra={
                        "event_type": "knowledge_update_success",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "update_method": "periodic_full_episode",
                    }
                )
                
                # Update map in knowledge base
                self.update_map_in_knowledge_base()
                
                # Reload agent knowledge
                self.reload_agent_knowledge()
                
            else:
                self.log_error(
                    f"Knowledge update failed at turn {self.game_state.turn_count}",
                    details="Knowledge update returned failure"
                )
                
                self.logger.error(
                    f"Knowledge update failed at turn {self.game_state.turn_count}",
                    extra={
                        "event_type": "knowledge_update_failed",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "update_method": "periodic_full_episode",
                    }
                )
                
        except Exception as e:
            self.log_error(
                f"Exception during knowledge update: {e}",
                details=f"Knowledge update failed with exception: {e}"
            )
            
            self.logger.error(
                f"Knowledge update exception: {e}",
                extra={
                    "event_type": "knowledge_update_exception",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e),
                }
            )
    
    def perform_final_update(self, death_count: int = 0) -> None:
        """Perform final knowledge update at episode end."""
        try:
            self.log_progress(
                f"Starting final knowledge update for episode {self.game_state.episode_id}",
                stage="final_knowledge_update",
                details=f"Final knowledge update for episode {self.game_state.episode_id}"
            )
            
            # Check if we've done a recent comprehensive update
            turns_since_last_update = self.game_state.turn_count - self.last_knowledge_update_turn
            skip_final_update = turns_since_last_update < (self.config.knowledge_update_interval / 2)
            
            # Special handling for death episodes - always do final update
            is_death_episode = death_count > 0
            
            self.logger.info(
                f"Final knowledge update decision: skip={skip_final_update}, death_episode={is_death_episode}",
                extra={
                    "event_type": "final_knowledge_update_decision",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "skip_update": skip_final_update,
                    "is_death_episode": is_death_episode,
                    "turns_since_last": turns_since_last_update,
                }
            )
            
            if not skip_final_update or is_death_episode:
                # Include map quality metrics
                map_metrics = {}
                if hasattr(self.game_map, 'get_quality_metrics'):
                    try:
                        map_metrics = self.game_map.get_quality_metrics()
                    except Exception as e:
                        self.log_warning(f"Failed to get map quality metrics: {e}")
                
                # Perform final knowledge update
                success = self.adaptive_knowledge_manager.update_knowledge_from_turns(
                    episode_id=self.game_state.episode_id,
                    start_turn=1,
                    end_turn=self.game_state.turn_count,
                    is_final_update=True
                )
                
                if success:
                    self.log_progress(
                        "Final knowledge update completed successfully",
                        stage="final_knowledge_update",
                        details="Final knowledge update completed"
                    )
                    
                    # Update map in knowledge base
                    self.update_map_in_knowledge_base()
                else:
                    self.log_error(
                        "Final knowledge update failed",
                        details="Final knowledge update returned failure"
                    )
            else:
                self.log_debug(
                    "Skipping final knowledge update - recent comprehensive update was done",
                    details=f"Last update was {turns_since_last_update} turns ago"
                )
                
        except Exception as e:
            self.log_error(
                f"Exception during final knowledge update: {e}",
                details=f"Final knowledge update failed with exception: {e}"
            )
    
    def immediate_update(self, section_id: str, content: str, trigger_reason: str) -> None:
        """Perform immediate knowledge update for critical discoveries."""
        try:
            self.log_progress(
                f"Immediate knowledge update: {section_id}",
                stage="immediate_knowledge_update",
                details=f"Immediate update for {section_id}: {trigger_reason}"
            )
            
            self.logger.info(
                f"Immediate knowledge update triggered: {section_id}",
                extra={
                    "event_type": "immediate_knowledge_update",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "section_id": section_id,
                    "trigger_reason": trigger_reason,
                }
            )
            
            # Perform immediate update
            success = self.adaptive_knowledge_manager.update_knowledge_section(
                section_id=section_id,
                content=content,
                episode_id=self.game_state.episode_id
            )
            
            if success:
                self.log_progress(
                    f"Immediate knowledge update completed: {section_id}",
                    stage="immediate_knowledge_update",
                    details="Immediate update completed successfully"
                )
                
                # Reload agent knowledge immediately
                self.reload_agent_knowledge()
            else:
                self.log_error(
                    f"Immediate knowledge update failed: {section_id}",
                    details="Immediate update returned failure"
                )
                
        except Exception as e:
            self.log_error(
                f"Exception during immediate knowledge update: {e}",
                details=f"Immediate update failed with exception: {e}"
            )
    
    def update_map_in_knowledge_base(self) -> None:
        """Update the mermaid map in knowledge base."""
        try:
            if hasattr(self.game_map, 'to_mermaid'):
                mermaid_map = self.game_map.to_mermaid()
                if mermaid_map:
                    self.adaptive_knowledge_manager.update_knowledge_with_map(
                        mermaid_content=mermaid_map,
                        episode_id=self.game_state.episode_id
                    )
                    self.log_debug("Updated map in knowledge base")
                else:
                    self.log_warning("No mermaid map content available")
            else:
                self.log_warning("Game map does not support mermaid export")
                
        except Exception as e:
            self.log_error(f"Failed to update map in knowledge base: {e}")
    
    def reload_agent_knowledge(self) -> None:
        """Reload knowledge base in agent for immediate use."""
        try:
            if hasattr(self.agent, 'reload_knowledge_base'):
                self.agent.reload_knowledge_base()
                self.log_debug("Agent knowledge base reloaded")
            else:
                self.log_warning("Agent does not support knowledge base reloading")
                
        except Exception as e:
            self.log_error(f"Failed to reload agent knowledge: {e}")
    
    def should_synthesize_inter_episode_wisdom(
        self, 
        final_score: int, 
        death_count: int, 
        critic_confidence_history: List[float]
    ) -> bool:
        """Determine if inter-episode wisdom synthesis should occur."""
        # Always synthesize on death episodes
        if death_count > 0:
            return True
        
        # Synthesize on significant score achievements
        if final_score >= 50:  # Significant progress threshold
            return True
        
        # Synthesize on long episodes (even if unsuccessful)
        if self.game_state.turn_count >= 500:
            return True
        
        # Synthesize based on critic confidence patterns
        if critic_confidence_history:
            avg_confidence = sum(critic_confidence_history) / len(critic_confidence_history)
            if avg_confidence >= 0.8:  # High confidence episode
                return True
        
        return False
    
    def perform_inter_episode_synthesis(
        self, 
        final_score: int, 
        death_count: int,
        critic_confidence_history: List[float]
    ) -> None:
        """Perform inter-episode wisdom synthesis."""
        try:
            # Check if synthesis should occur
            if not self.should_synthesize_inter_episode_wisdom(final_score, death_count, critic_confidence_history):
                self.log_debug("Skipping inter-episode synthesis - criteria not met")
                return
            
            self.log_progress(
                f"Starting inter-episode wisdom synthesis for episode {self.game_state.episode_id}",
                stage="wisdom_synthesis",
                details=f"Synthesis for episode {self.game_state.episode_id}"
            )
            
            # Collect episode data for synthesis
            episode_data = {
                "episode_id": self.game_state.episode_id,
                "turn_count": self.game_state.turn_count,
                "final_score": final_score,
                "death_count": death_count,
                "discovered_objectives": self.game_state.discovered_objectives.copy(),
                "completed_objectives": [obj["objective"] for obj in self.game_state.completed_objectives],
                "critic_confidence_avg": sum(critic_confidence_history) / len(critic_confidence_history) if critic_confidence_history else 0,
                "map_metrics": {}
            }
            
            # Add map metrics if available
            if hasattr(self.game_map, 'get_quality_metrics'):
                try:
                    episode_data["map_metrics"] = self.game_map.get_quality_metrics()
                except Exception as e:
                    self.log_warning(f"Failed to get map metrics for synthesis: {e}")
            
            self.logger.info(
                f"Inter-episode synthesis starting",
                extra={
                    "event_type": "inter_episode_synthesis_start",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "final_score": final_score,
                    "death_count": death_count,
                    "objectives_discovered": len(self.game_state.discovered_objectives),
                    "objectives_completed": len(self.game_state.completed_objectives),
                }
            )
            
            # Perform synthesis
            success = self.adaptive_knowledge_manager.synthesize_inter_episode_wisdom(
                episode_data=episode_data
            )
            
            if success:
                self.log_progress(
                    "Inter-episode wisdom synthesis completed successfully",
                    stage="wisdom_synthesis",
                    details="Synthesis completed successfully"
                )
            else:
                self.log_error(
                    "Inter-episode wisdom synthesis failed",
                    details="Synthesis returned failure"
                )
                
        except Exception as e:
            self.log_error(
                f"Exception during inter-episode synthesis: {e}",
                details=f"Synthesis failed with exception: {e}"
            )
    
    def get_knowledge_base_summary(self) -> str:
        """Get knowledge base content without the map section."""
        try:
            with open("knowledgebase.md", "r") as f:
                content = f.read()
            
            # Remove mermaid map sections using regex
            # This removes everything from "## Map" to the end or next major section
            content_without_map = re.sub(
                r'## Map.*?(?=\n## |\Z)', 
                '', 
                content, 
                flags=re.DOTALL
            )
            
            return content_without_map.strip()
            
        except FileNotFoundError:
            self.log_warning("Knowledge base file not found")
            return ""
        except Exception as e:
            self.log_error(f"Failed to read knowledge base: {e}")
            return ""
    
    def restore_knowledge_base(self, previous_content: str) -> None:
        """Restore knowledge base from previous state."""
        try:
            if previous_content:
                with open("knowledgebase.md", "w") as f:
                    f.write(previous_content)
                
                # Update adaptive knowledge manager's last content
                if hasattr(self.adaptive_knowledge_manager, 'last_content'):
                    self.adaptive_knowledge_manager.last_content = previous_content
                
                self.log_debug("Knowledge base restored from previous state")
            else:
                self.log_warning("No previous knowledge base content to restore")
                
        except Exception as e:
            self.log_error(f"Failed to restore knowledge base: {e}")
    
    def get_llm_client(self):
        """Access LLM client for knowledge-related analysis."""
        if hasattr(self.adaptive_knowledge_manager, 'client'):
            return self.adaptive_knowledge_manager.client
        return None
    
    def get_analysis_model(self) -> str:
        """Get the model used for analysis tasks."""
        if hasattr(self.adaptive_knowledge_manager, 'analysis_model'):
            return self.adaptive_knowledge_manager.analysis_model
        return "gpt-4"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current knowledge manager status."""
        status = super().get_status()
        status.update({
            "last_knowledge_update_turn": self.last_knowledge_update_turn,
            "turns_since_last_update": self.game_state.turn_count - self.last_knowledge_update_turn,
            "knowledge_update_interval": self.config.knowledge_update_interval,
            "has_adaptive_manager": self.adaptive_knowledge_manager is not None,
            "has_llm_client": self.get_llm_client() is not None,
        })
        return status