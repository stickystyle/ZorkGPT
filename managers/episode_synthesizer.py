"""
EpisodeSynthesizer for ZorkGPT orchestration.

Handles all episode-level coordination and synthesis responsibilities:
- Episode lifecycle coordination and management
- Final episode synthesis and analysis
- Episode cleanup and finalization
- Cross-component coordination for episode events
- Episode reporting and state export
- Inter-episode learning and wisdom synthesis
"""

from typing import List, Dict, Any
from datetime import datetime

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration

try:
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    # Graceful fallback - no-op decorator
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGFUSE_AVAILABLE = False


class EpisodeSynthesizer(BaseManager):
    """
    Manages episode-level synthesis and coordination for ZorkGPT.

    Responsibilities:
    - Episode lifecycle coordination (start, play, end)
    - Final episode synthesis and analysis
    - Episode cleanup and finalization
    - Cross-component episode coordination
    - Episode reporting and state export
    - Inter-episode wisdom synthesis and learning
    """

    def __init__(
        self,
        logger,
        config: GameConfiguration,
        game_state: GameState,
        knowledge_manager=None,
        state_manager=None,
        llm_client=None,
    ):
        super().__init__(logger, config, game_state, "episode_synthesizer")
        self.knowledge_manager = knowledge_manager
        self.state_manager = state_manager
        self.llm_client = llm_client

    def reset_episode(self) -> None:
        """Reset episode synthesizer state for a new episode."""
        self.log_debug("Episode synthesizer reset for new episode")

    def process_turn(self) -> None:
        """Process episode synthesis for the current turn."""
        # Episode synthesis is event-driven, not turn-based
        pass

    def should_process_turn(self) -> bool:
        """Check if episode synthesis needs processing this turn."""
        return False  # Episode synthesis is event-driven

    def initialize_episode(
        self, episode_id: str, agent=None, extractor=None, critic=None
    ) -> str:
        """Initialize a new episode and coordinate component updates.

        Args:
            episode_id: Episode ID provided by orchestrator
            agent: Agent component to update
            extractor: Extractor component to update
            critic: Critic component to update

        Returns:
            The episode ID (for convenience)
        """
        try:
            # Reset episode state in GameState with the provided episode ID
            self.game_state.reset_episode(episode_id=episode_id)

            self.log_progress(
                f"Initializing new episode: {episode_id}",
                stage="episode_initialization",
                details=f"Starting episode {episode_id}",
            )

            # Update episode ID in all components
            if agent:
                agent.update_episode_id(episode_id)

            if extractor:
                extractor.update_episode_id(episode_id)

            if critic:
                critic.update_episode_id(episode_id)

            self.logger.info(
                f"Episode initialized: {episode_id}",
                extra={
                    "event_type": "episode_initialized",
                    "episode_id": episode_id,
                    "turn": 0,
                    "max_turns": self.config.max_turns_per_episode,
                },
            )

            return episode_id

        except Exception as e:
            self.log_error(f"Failed to initialize episode: {e}")
            return f"{episode_id}_error"

    def finalize_episode(
        self, final_score: int, critic_confidence_history: List[float] = None
    ) -> None:
        """Finalize the current episode with synthesis and cleanup."""
        try:
            self.log_progress(
                f"Finalizing episode {self.game_state.episode_id}",
                stage="episode_finalization",
                details=f"Final score: {final_score}, turns: {self.game_state.turn_count}",
            )

            # Determine if this was a death episode
            is_death = self.is_death_episode()
            if is_death:
                self.game_state.death_count += 1

            # Perform final knowledge update if knowledge manager available
            if self.knowledge_manager:
                self.knowledge_manager.perform_final_update(
                    death_count=self.game_state.death_count
                )

            # Perform inter-episode synthesis if appropriate
            if critic_confidence_history is None:
                critic_confidence_history = []

            self.perform_inter_episode_synthesis(final_score, critic_confidence_history)

            # Note: State export is handled by orchestrator coordination
            # (orchestrator calls _export_coordinated_state after episode finalization)

            # Generate episode summary
            summary = self.generate_episode_summary(final_score, is_death)

            self.logger.info(
                f"Episode finalized: {self.game_state.episode_id}",
                extra={
                    "event_type": "episode_finalized",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "final_score": final_score,
                    "is_death_episode": is_death,
                    "death_count": self.game_state.death_count,
                    "objectives_discovered": len(self.game_state.discovered_objectives),
                    "objectives_completed": len(self.game_state.completed_objectives),
                },
            )

            self.log_progress(
                f"Episode {self.game_state.episode_id} completed successfully",
                stage="episode_completion",
                details=summary,
            )

        except Exception as e:
            self.log_error(f"Failed to finalize episode: {e}")

    def is_death_episode(self) -> bool:
        """Determine if the current episode ended in death."""
        try:
            # Check action reasoning history for death indicators
            for reasoning_entry in self.game_state.action_reasoning_history[
                -5:
            ]:  # Last 5 entries
                if isinstance(reasoning_entry, dict):
                    reasoning_text = reasoning_entry.get("reasoning", "").lower()
                    death_keywords = [
                        "died",
                        "death",
                        "killed",
                        "fatal",
                        "perish",
                        "dead",
                    ]

                    if any(keyword in reasoning_text for keyword in death_keywords):
                        return True

            # Check memory log for death indicators
            for memory in self.game_state.memory_log_history[-3:]:  # Last 3 memories
                if isinstance(memory, dict):
                    memory_text = str(memory).lower()
                    if any(
                        keyword in memory_text
                        for keyword in ["died", "death", "killed", "fatal"]
                    ):
                        return True

            # Check if game over flag indicates death
            if self.game_state.game_over_flag:
                # Additional death detection logic could be added here
                pass

            return False

        except Exception as e:
            self.log_error(f"Failed to check death episode: {e}")
            return False

    def should_synthesize_inter_episode_wisdom(
        self, final_score: int, critic_confidence_history: List[float]
    ) -> bool:
        """Determine if inter-episode wisdom synthesis should occur."""
        try:
            # Always synthesize on death episodes
            if self.is_death_episode():
                return True

            # Synthesize on significant score achievements
            if final_score >= 50:  # Significant progress threshold
                return True

            # Synthesize on long episodes (even if unsuccessful)
            if self.game_state.turn_count >= 500:
                return True

            # Synthesize based on critic confidence patterns
            if critic_confidence_history:
                avg_confidence = sum(critic_confidence_history) / len(
                    critic_confidence_history
                )
                if avg_confidence >= 0.8:  # High confidence episode
                    return True

            # Synthesize if significant objectives were completed
            if len(self.game_state.completed_objectives) >= 3:
                return True

            return False

        except Exception as e:
            self.log_error(f"Failed to check synthesis criteria: {e}")
            return False

    def perform_inter_episode_synthesis(
        self, final_score: int, critic_confidence_history: List[float]
    ) -> None:
        """Perform inter-episode wisdom synthesis."""
        try:
            # Check if synthesis should occur
            if not self.should_synthesize_inter_episode_wisdom(
                final_score, critic_confidence_history
            ):
                self.log_debug("Skipping inter-episode synthesis - criteria not met")
                return

            self.log_progress(
                f"Starting inter-episode synthesis for {self.game_state.episode_id}",
                stage="inter_episode_synthesis",
                details=f"Synthesis for episode {self.game_state.episode_id}",
            )

            # Use knowledge manager if available
            if self.knowledge_manager:
                self.knowledge_manager.perform_inter_episode_synthesis(
                    final_score=final_score,
                    death_count=self.game_state.death_count,
                    critic_confidence_history=critic_confidence_history,
                )
            else:
                # Fallback synthesis logic
                self.perform_fallback_synthesis(final_score, critic_confidence_history)

        except Exception as e:
            self.log_error(f"Failed to perform inter-episode synthesis: {e}")

    def perform_fallback_synthesis(
        self, final_score: int, critic_confidence_history: List[float]
    ) -> None:
        """Perform basic synthesis when knowledge manager is not available."""
        try:
            synthesis_data = {
                "episode_id": self.game_state.episode_id,
                "final_score": final_score,
                "turn_count": self.game_state.turn_count,
                "is_death": self.is_death_episode(),
                "objectives_discovered": len(self.game_state.discovered_objectives),
                "objectives_completed": len(self.game_state.completed_objectives),
                "avg_critic_confidence": sum(critic_confidence_history)
                / len(critic_confidence_history)
                if critic_confidence_history
                else 0,
            }

            self.logger.info(
                "Fallback inter-episode synthesis completed",
                extra={
                    "event_type": "fallback_synthesis",
                    "episode_id": self.game_state.episode_id,
                    "synthesis_data": synthesis_data,
                },
            )

        except Exception as e:
            self.log_error(f"Failed to perform fallback synthesis: {e}")

    def generate_episode_summary(self, final_score: int, is_death: bool) -> str:
        """Generate a comprehensive summary of the episode."""
        try:
            if self.llm_client:
                return self.generate_llm_episode_summary(final_score, is_death)
            else:
                return self.generate_fallback_episode_summary(final_score, is_death)

        except Exception as e:
            self.log_error(f"Failed to generate episode summary: {e}")
            return self.generate_fallback_episode_summary(final_score, is_death)

    @observe(name="episode-generate-synthesis")
    def generate_llm_episode_summary(self, final_score: int, is_death: bool) -> str:
        """Generate LLM-powered episode summary."""
        try:
            # Prepare context for summarization
            recent_actions = self.game_state.action_history[-10:]  # Last 10 actions

            prompt = f"""Summarize this Zork gameplay episode concisely.

EPISODE: {self.game_state.episode_id}
TURNS: {self.game_state.turn_count}
FINAL SCORE: {final_score}
DEATH EPISODE: {is_death}
CURRENT LOCATION: {self.game_state.current_room_name_for_map}
INVENTORY: {", ".join(self.game_state.current_inventory) if self.game_state.current_inventory else "empty"}

OBJECTIVES DISCOVERED: {len(self.game_state.discovered_objectives)}
OBJECTIVES COMPLETED: {len(self.game_state.completed_objectives)}

RECENT ACTIONS:
{chr(10).join([f"{action} -> {response[:100]}..." for action, response in recent_actions[-5:]])}

Provide a brief 2-3 sentence summary focusing on:
1. Key achievements or progress made
2. Challenges encountered or cause of failure
3. Strategic insights for future episodes

Keep it under 200 words."""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                name="EpisodeSynthesizer",
            )

            return response.content or self.generate_fallback_episode_summary(
                final_score, is_death
            )

        except Exception as e:
            self.log_error(f"LLM episode summary failed: {e}")
            return self.generate_fallback_episode_summary(final_score, is_death)

    def generate_fallback_episode_summary(
        self, final_score: int, is_death: bool
    ) -> str:
        """Generate basic episode summary without LLM assistance."""
        try:
            summary_parts = [
                f"Episode {self.game_state.episode_id} completed in {self.game_state.turn_count} turns",
                f"Final score: {final_score}",
                f"Death episode: {is_death}",
                f"Location: {self.game_state.current_room_name_for_map}",
                f"Objectives: {len(self.game_state.discovered_objectives)} discovered, {len(self.game_state.completed_objectives)} completed",
            ]

            if is_death:
                summary_parts.append(f"Total deaths: {self.game_state.death_count}")

            return ". ".join(summary_parts) + "."

        except Exception as e:
            self.log_error(f"Failed to generate fallback summary: {e}")
            return f"Episode {self.game_state.episode_id} completed with score {final_score}"

    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get comprehensive episode metrics for analysis."""
        try:
            return {
                "episode_id": self.game_state.episode_id,
                "turn_count": self.game_state.turn_count,
                "final_score": self.game_state.previous_zork_score,
                "is_death_episode": self.is_death_episode(),
                "death_count": self.game_state.death_count,
                "objectives_discovered": len(self.game_state.discovered_objectives),
                "objectives_completed": len(self.game_state.completed_objectives),
                "locations_visited": len(self.game_state.visited_locations),
                "actions_taken": len(self.game_state.action_history),
                "memory_entries": len(self.game_state.memory_log_history),
                "final_location": self.game_state.current_room_name_for_map,
                "final_inventory": self.game_state.current_inventory.copy(),
                "episode_duration": self.game_state.turn_count,  # Could be enhanced with actual time
            }

        except Exception as e:
            self.log_error(f"Failed to get episode metrics: {e}")
            return {}

    def export_episode_data(self) -> Dict[str, Any]:
        """Export comprehensive episode data for external analysis."""
        try:
            episode_data = {
                "metadata": {
                    "episode_id": self.game_state.episode_id,
                    "timestamp": datetime.now().isoformat(),
                    "turn_count": self.game_state.turn_count,
                    "is_death_episode": self.is_death_episode(),
                },
                "metrics": self.get_episode_metrics(),
                "objectives": {
                    "discovered": self.game_state.discovered_objectives.copy(),
                    "completed": [
                        obj.copy() for obj in self.game_state.completed_objectives
                    ],
                },
                "final_state": {
                    "score": self.game_state.previous_zork_score,
                    "location": self.game_state.current_room_name_for_map,
                    "inventory": self.game_state.current_inventory.copy(),
                    "game_over": self.game_state.game_over_flag,
                },
                "summary": self.generate_episode_summary(
                    self.game_state.previous_zork_score, self.is_death_episode()
                ),
            }

            return episode_data

        except Exception as e:
            self.log_error(f"Failed to export episode data: {e}")
            return {}

    def get_status(self) -> Dict[str, Any]:
        """Get current episode synthesizer status."""
        status = super().get_status()
        status.update(
            {
                "current_episode": self.game_state.episode_id,
                "turn_count": self.game_state.turn_count,
                "death_count": self.game_state.death_count,
                "objectives_discovered": len(self.game_state.discovered_objectives),
                "objectives_completed": len(self.game_state.completed_objectives),
                "is_death_episode": self.is_death_episode(),
                "has_knowledge_manager": self.knowledge_manager is not None,
                "has_state_manager": self.state_manager is not None,
            }
        )
        return status
