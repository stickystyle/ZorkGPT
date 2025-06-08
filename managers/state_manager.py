"""
StateManager for ZorkGPT orchestration.

Handles all state management responsibilities:
- Game state tracking and updates
- State export and import functionality
- Episode state initialization and cleanup
- Memory and history management
- Session state coordination
- Cross-episode persistent state
"""

import json
import boto3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class StateManager(BaseManager):
    """
    Manages all state-related functionality for ZorkGPT.
    
    Responsibilities:
    - Episode lifecycle and state management
    - State export and import to files and S3
    - Memory management and history tracking
    - Context overflow protection and summarization
    - Cross-episode persistent state tracking
    - State queries and reporting
    """
    
    def __init__(
        self, 
        logger, 
        config: GameConfiguration, 
        game_state: GameState,
        llm_client=None
    ):
        super().__init__(logger, config, game_state, "state_manager")
        self.llm_client = llm_client
        
        # S3 client for state uploads
        self.s3_client = None
        if config.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
            except Exception as e:
                self.log_warning(f"Failed to initialize S3 client: {e}")
        
        # Context management tracking
        self.last_summarization_turn = 0
    
    def reset_episode(self) -> None:
        """Reset episode-specific state for a new episode."""
        self.log_debug("Resetting episode state")
        
        # NOTE: Do NOT call game_state.reset_episode() here!
        # Episode ID generation and GameState reset is handled by EpisodeSynthesizer
        # This method only resets StateManager's internal state
        
        # Reset manager-specific tracking
        self.last_summarization_turn = 0
        
        self.log_debug("Episode state reset completed")
    
    def process_turn(self) -> None:
        """Process state management for the current turn."""
        # Check for context overflow
        self.check_context_overflow()
    
    def should_process_turn(self) -> bool:
        """Check if state needs processing this turn."""
        # Always process for context management
        return True
    
    def check_context_overflow(self) -> None:
        """Check if context is approaching token limits and trigger summarization if needed."""
        try:
            # Estimate current context tokens
            estimated_tokens = self.estimate_context_tokens()
            
            # Check if we're approaching the limit
            threshold = int(self.config.max_context_tokens * self.config.context_overflow_threshold)
            
            if estimated_tokens > threshold:
                self.log_progress(
                    f"Context overflow detected: {estimated_tokens}/{self.config.max_context_tokens} tokens",
                    stage="context_management",
                    details=f"Triggering context summarization at turn {self.game_state.turn_count}"
                )
                
                self.logger.info(
                    f"Context overflow - triggering summarization",
                    extra={
                        "event_type": "context_overflow",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "estimated_tokens": estimated_tokens,
                        "threshold": threshold,
                        "max_tokens": self.config.max_context_tokens,
                    }
                )
                
                self.trigger_context_summarization()
            
        except Exception as e:
            self.log_error(f"Failed to check context overflow: {e}")
    
    def estimate_context_tokens(self) -> int:
        """Estimate total context tokens based on memory log history."""
        try:
            # Rough estimation: ~3-4 tokens per word
            total_chars = 0
            
            # Count characters in memory history
            for memory in self.game_state.memory_log_history:
                if isinstance(memory, dict):
                    total_chars += len(str(memory))
                else:
                    total_chars += len(str(memory))
            
            # Count characters in action history
            for action, response in self.game_state.action_history:
                total_chars += len(action) + len(response)
            
            # Rough conversion: 4 characters per token
            estimated_tokens = total_chars // 4
            
            return estimated_tokens
            
        except Exception as e:
            self.log_error(f"Failed to estimate context tokens: {e}")
            return 0
    
    def trigger_context_summarization(self) -> None:
        """Generate a summary of recent gameplay and reset context."""
        try:
            self.log_progress(
                "Starting context summarization",
                stage="context_summarization",
                details=f"Summarizing context at turn {self.game_state.turn_count}"
            )
            
            # Generate summary of recent gameplay
            summary = self.generate_gameplay_summary()
            
            if summary:
                # Extract critical memories before clearing
                critical_memories = self.extract_critical_memories()
                
                # Clear old memory log history but keep recent critical memories
                self.game_state.memory_log_history = critical_memories[-10:]  # Keep last 10 critical memories
                
                # Add summary as a memory entry
                summary_memory = {
                    "turn": self.game_state.turn_count,
                    "type": "context_summary",
                    "summary": summary,
                    "summarized_turns": f"1-{self.game_state.turn_count - 10}"
                }
                self.game_state.memory_log_history.insert(0, summary_memory)
                
                # Keep recent action history but truncate older entries
                self.game_state.action_history = self.game_state.action_history[-20:]  # Keep last 20 actions
                
                self.last_summarization_turn = self.game_state.turn_count
                
                self.log_progress(
                    "Context summarization completed",
                    stage="context_summarization",
                    details=f"Generated summary, retained {len(critical_memories)} critical memories"
                )
                
                self.logger.info(
                    "Context summarization completed",
                    extra={
                        "event_type": "context_summarization_completed",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "critical_memories_retained": len(critical_memories),
                        "actions_retained": len(self.game_state.action_history),
                    }
                )
            else:
                self.log_warning("Failed to generate context summary")
                
        except Exception as e:
            self.log_error(f"Failed to trigger context summarization: {e}")
    
    def generate_gameplay_summary(self) -> str:
        """Generate a comprehensive summary of recent gameplay progress."""
        try:
            if not self.llm_client:
                return self.generate_fallback_summary()
            
            # Prepare context for summarization
            recent_actions = self.game_state.action_history[-30:]  # Last 30 actions
            recent_memories = self.game_state.memory_log_history[-20:]  # Last 20 memories
            
            # Create prompt for summarization
            prompt = f"""Summarize the recent gameplay progress in Zork for context preservation.

EPISODE: {self.game_state.episode_id}
TURNS: 1-{self.game_state.turn_count}
CURRENT SCORE: {self.game_state.previous_zork_score}
CURRENT LOCATION: {self.game_state.current_room_name_for_map}
CURRENT INVENTORY: {self.game_state.current_inventory}

RECENT ACTIONS ({len(recent_actions)} actions):
{chr(10).join([f"Turn {i}: {action} -> {response[:100]}..." for i, (action, response) in enumerate(recent_actions, start=max(1, self.game_state.turn_count-len(recent_actions)+1))])}

DISCOVERED OBJECTIVES:
{chr(10).join([f"- {obj}" for obj in self.game_state.discovered_objectives])}

COMPLETED OBJECTIVES:
{chr(10).join([f"- {obj['objective']} (Turn {obj['completed_turn']})" for obj in self.game_state.completed_objectives[-5:]])}

Create a concise summary covering:
1. Key locations visited and progress made
2. Important items acquired or puzzles solved
3. Current challenges or stuck points
4. Strategic insights for future gameplay

Keep the summary under 500 words and focus on actionable information for continuing gameplay."""

            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",  # Use a reliable model for summarization
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000
                )
                
                return response.content or ""
                
            except Exception as llm_error:
                self.log_warning(f"LLM summarization failed: {llm_error}")
                return self.generate_fallback_summary()
                
        except Exception as e:
            self.log_error(f"Failed to generate gameplay summary: {e}")
            return self.generate_fallback_summary()
    
    def generate_fallback_summary(self) -> str:
        """Generate a basic summary without LLM assistance."""
        try:
            summary_parts = [
                f"Episode {self.game_state.episode_id} - Turn {self.game_state.turn_count}",
                f"Score: {self.game_state.previous_zork_score}",
                f"Location: {self.game_state.current_room_name_for_map}",
                f"Inventory: {', '.join(self.game_state.current_inventory) if self.game_state.current_inventory else 'empty'}",
                f"Discovered objectives: {len(self.game_state.discovered_objectives)}",
                f"Completed objectives: {len(self.game_state.completed_objectives)}",
                f"Recent actions: {len(self.game_state.action_history[-10:])} logged"
            ]
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.log_error(f"Failed to generate fallback summary: {e}")
            return f"Context summarized at turn {self.game_state.turn_count}"
    
    def extract_critical_memories(self) -> List[Dict[str, Any]]:
        """Extract the most critical memories from recent turns."""
        try:
            critical_memories = []
            
            for memory in self.game_state.memory_log_history:
                if self.is_critical_memory(memory):
                    critical_memories.append(memory)
            
            # Sort by importance and return top memories
            critical_memories.sort(key=lambda m: m.get("turn", 0), reverse=True)
            return critical_memories[:15]  # Keep top 15 critical memories
            
        except Exception as e:
            self.log_error(f"Failed to extract critical memories: {e}")
            return []
    
    def is_critical_memory(self, memory: Any) -> bool:
        """Determine if a memory contains critical information."""
        try:
            if not isinstance(memory, dict):
                return False
            
            # Check for score increases
            if memory.get("score", 0) > 0:
                return True
            
            # Check for death events
            if memory.get("game_over") or "died" in str(memory).lower():
                return True
            
            # Check for new items or significant discoveries
            memory_text = str(memory).lower()
            critical_keywords = [
                "treasure", "valuable", "points", "earned", "found", "took",
                "opened", "unlocked", "solved", "completed", "achieved"
            ]
            
            if any(keyword in memory_text for keyword in critical_keywords):
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get comprehensive current state for export."""
        try:
            return {
                "metadata": {
                    "episode_id": self.game_state.episode_id,
                    "timestamp": datetime.now().isoformat(),
                    "turn_count": self.game_state.turn_count,
                    "game_over": self.game_state.game_over_flag,
                    "previous_score": self.game_state.previous_zork_score,
                    "max_turns_per_episode": self.config.max_turns_per_episode,
                    "agent_model": self.config.agent_model,
                    "critic_model": self.config.critic_model,
                    "info_ext_model": self.config.info_ext_model,
                },
                "current_state": {
                    "current_location": self.game_state.current_room_name_for_map,
                    "inventory": self.game_state.current_inventory,
                    "in_combat": self.get_combat_status(),
                    "death_count": self.game_state.death_count,
                    "discovered_objectives": self.game_state.discovered_objectives,
                    "completed_objectives": len(self.game_state.completed_objectives),
                },
                "recent_log": self.get_recent_log(),
                "performance": {
                    "avg_critic_score": self.get_avg_critic_score(),
                    "recent_actions": self.get_recent_action_summary(),
                    "total_actions": len(self.game_state.action_history),
                },
                "context_management": {
                    "memory_entries": len(self.game_state.memory_log_history),
                    "last_summarization_turn": self.last_summarization_turn,
                    "estimated_tokens": self.estimate_context_tokens(),
                }
            }
            
        except Exception as e:
            self.log_error(f"Failed to get current state: {e}")
            return {}
    
    def export_current_state(self) -> bool:
        """Export current state to file and optionally to S3."""
        try:
            if not self.config.enable_state_export:
                return True
            
            state_data = self.get_current_state()
            if not state_data:
                return False
            
            # Export to local file
            with open(self.config.state_export_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.log_debug(f"State exported to {self.config.state_export_file}")
            
            # Upload to S3 if configured
            if self.config.s3_bucket and self.s3_client:
                success = self.upload_state_to_s3(state_data)
                if success:
                    self.log_debug("State uploaded to S3")
                else:
                    self.log_warning("Failed to upload state to S3")
            
            return True
            
        except Exception as e:
            self.log_error(f"Failed to export current state: {e}")
            return False
    
    def upload_state_to_s3(self, state_data: Dict[str, Any]) -> bool:
        """Upload current state to S3."""
        try:
            if not self.s3_client or not self.config.s3_bucket:
                return False
            
            # Create S3 key
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"{self.config.s3_key_prefix}states/{self.game_state.episode_id}_{timestamp}.json"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=json.dumps(state_data, indent=2),
                ContentType='application/json'
            )
            
            return True
            
        except Exception as e:
            self.log_error(f"Failed to upload state to S3: {e}")
            return False
    
    def load_previous_state(self) -> Optional[Dict[str, Any]]:
        """Load previous state from current_state.json."""
        try:
            with open(self.config.state_export_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.log_debug("No previous state file found")
            return None
        except Exception as e:
            self.log_error(f"Failed to load previous state: {e}")
            return None
    
    def merge_previous_state(self, previous_state: Dict[str, Any]) -> None:
        """Merge relevant data from previous state into current session."""
        try:
            if not previous_state:
                return
            
            # Merge persistent cross-episode state
            if "current_state" in previous_state:
                prev_current = previous_state["current_state"]
                
                # Restore death count (persists across episodes)
                if "death_count" in prev_current:
                    self.game_state.death_count = prev_current["death_count"]
            
            self.log_debug("Previous state merged into current session")
            
        except Exception as e:
            self.log_error(f"Failed to merge previous state: {e}")
    
    def get_recent_log(self, num_entries: int = 10) -> List[Dict[str, Any]]:
        """Get recent game log entries with reasoning."""
        try:
            recent_log = []
            
            # Get recent action history with reasoning
            recent_actions = self.game_state.action_history[-num_entries:]
            recent_reasoning = self.game_state.action_reasoning_history[-num_entries:]
            
            for i, (action, response) in enumerate(recent_actions):
                reasoning = recent_reasoning[i] if i < len(recent_reasoning) else ""
                
                log_entry = {
                    "turn": self.game_state.turn_count - len(recent_actions) + i + 1,
                    "action": action,
                    "response": response[:200],  # Truncate long responses
                    "reasoning": reasoning[:100] if reasoning else ""  # Truncate reasoning
                }
                recent_log.append(log_entry)
            
            return recent_log
            
        except Exception as e:
            self.log_error(f"Failed to get recent log: {e}")
            return []
    
    def get_combat_status(self) -> bool:
        """Determine if currently in combat based on recent extractions."""
        try:
            # Check recent memory for combat indicators
            recent_memories = self.game_state.memory_log_history[-3:]  # Last 3 turns
            
            for memory in recent_memories:
                if isinstance(memory, dict):
                    # Check for combat-related fields or keywords
                    memory_text = str(memory).lower()
                    combat_keywords = ["combat", "fight", "attack", "enemy", "monster", "battle"]
                    
                    if any(keyword in memory_text for keyword in combat_keywords):
                        return True
            
            return False
            
        except Exception as e:
            self.log_error(f"Failed to get combat status: {e}")
            return False
    
    def get_avg_critic_score(self, num_recent: int = 10) -> float:
        """Get average critic score for recent turns."""
        try:
            # This would need to be tracked in game state or passed from orchestrator
            # For now, return a placeholder
            return 0.0
            
        except Exception as e:
            self.log_error(f"Failed to get avg critic score: {e}")
            return 0.0
    
    def get_recent_action_summary(self, num_actions: int = 5) -> List[str]:
        """Get summary of recent actions."""
        try:
            recent_actions = self.game_state.action_history[-num_actions:]
            return [action for action, _ in recent_actions]
            
        except Exception as e:
            self.log_error(f"Failed to get recent action summary: {e}")
            return []
    
    def is_death_episode(self) -> bool:
        """Determine if the current episode ended in death."""
        try:
            # Check recent memory for death indicators
            for memory in self.game_state.memory_log_history[-5:]:
                if isinstance(memory, dict):
                    memory_text = str(memory).lower()
                    death_keywords = ["died", "death", "killed", "fatal", "perish"]
                    
                    if any(keyword in memory_text for keyword in death_keywords):
                        return True
            
            return False
            
        except Exception as e:
            self.log_error(f"Failed to check death episode: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current state manager status."""
        status = super().get_status()
        status.update({
            "memory_entries": len(self.game_state.memory_log_history),
            "action_history_length": len(self.game_state.action_history),
            "estimated_tokens": self.estimate_context_tokens(),
            "last_summarization_turn": self.last_summarization_turn,
            "export_enabled": self.config.enable_state_export,
            "s3_configured": self.s3_client is not None,
        })
        return status