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
from datetime import datetime, timezone
from typing import List, Dict, Any

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from zork_strategy_generator import AdaptiveKnowledgeManager


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
        game_map,  # Actually receives MapManager
        json_log_file: str = "zork_episode_log.jsonl",
    ):
        super().__init__(logger, config, game_state, "knowledge_manager")
        self.agent = agent
        self.map_manager = game_map  # Rename for clarity - this is actually MapManager

        # Initialize AdaptiveKnowledgeManager
        self.adaptive_knowledge_manager = AdaptiveKnowledgeManager(
            log_file=json_log_file, output_file="knowledgebase.md", logger=logger
        )

        # Knowledge update tracking
        self.last_knowledge_update_turn = 0

        # Object event tracking (Phase 6)
        self.object_events: List[Dict[str, Any]] = []

    def reset_episode(self) -> None:
        """Reset knowledge manager state for a new episode."""
        self.last_knowledge_update_turn = 0
        self.log_debug("Knowledge manager reset for new episode")

        # Reset object event tracking (Phase 6)
        self.object_events.clear()

    def track_object_event(
        self,
        event_type: str,
        obj_id: int,
        obj_name: str,
        turn: int,
        additional_context: Dict[str, Any] = None,
    ) -> None:
        """
        Track object-related events for knowledge synthesis.

        Args:
            event_type: Type of event ("acquired", "dropped", "opened", "relocated", "closed", "examined")
            obj_id: Z-machine object ID
            obj_name: Object name
            turn: Turn number when event occurred
            additional_context: Optional additional context (location, action, etc.)
        """
        try:
            # Validate event type
            valid_event_types = {
                "acquired",
                "dropped",
                "opened",
                "relocated",
                "closed",
                "examined",
            }
            if event_type not in valid_event_types:
                self.log_warning(f"Unknown object event type: {event_type}")

            event = {
                "turn": turn,
                "event_type": event_type,
                "object_id": obj_id,
                "object_name": obj_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "episode_id": self.game_state.episode_id,
            }

            # Add optional context
            if additional_context:
                event.update(additional_context)

            self.object_events.append(event)

            self.log_debug(
                f"Tracked object event: {event_type} - {obj_name} (ID: {obj_id}) at turn {turn}"
            )

            # Log significant events
            if event_type in {"acquired", "opened"}:
                self.logger.info(
                    "Object event: %s - %s",
                    event_type,
                    obj_name,
                    extra={
                        "event_type": "object_event",
                        "object_event_type": event_type,
                        "object_id": obj_id,
                        "object_name": obj_name,
                        "turn": turn,
                        "episode_id": self.game_state.episode_id,
                    },
                )

        except Exception as e:
            self.log_error(f"Failed to track object event: {e}")

    def detect_object_events(
        self,
        prev_inventory: List[str],
        current_inventory: List[str],
        jericho_interface,
        action: str,
        turn: int,
    ) -> None:
        """
        Detect and track object events by comparing game states.

        Args:
            prev_inventory: Previous inventory
            current_inventory: Current inventory
            jericho_interface: JerichoInterface for getting object IDs
            action: Action that was taken
            turn: Current turn number
        """
        try:
            # Detect acquired items
            acquired = set(current_inventory) - set(prev_inventory)
            for item_name in acquired:
                # Get object ID from Jericho
                inv_objects = jericho_interface.get_inventory_structured()
                obj_id = next(
                    (obj.num for obj in inv_objects if obj.name == item_name), None
                )
                if obj_id:
                    self.track_object_event(
                        "acquired", obj_id, item_name, turn, {"action": action}
                    )

            # Detect dropped items
            dropped = set(prev_inventory) - set(current_inventory)
            for item_name in dropped:
                # For dropped items, we may not have the ID anymore
                # Track with placeholder ID if needed
                self.track_object_event(
                    "dropped", -1, item_name, turn, {"action": action}
                )

            # Detect open/close actions from action text
            action_lower = action.lower()
            if action_lower.startswith("open "):
                obj_name = action[5:].strip()
                self.track_object_event(
                    "opened", -1, obj_name, turn, {"action": action}
                )
            elif action_lower.startswith("close "):
                obj_name = action[6:].strip()
                self.track_object_event(
                    "closed", -1, obj_name, turn, {"action": action}
                )

        except Exception as e:
            self.log_error(f"Failed to detect object events: {e}")

    def process_turn(self) -> None:
        """Process knowledge management for the current turn."""
        # This is handled by process_periodic_updates
        pass

    def should_process_turn(self) -> bool:
        """Check if knowledge needs processing this turn."""
        # Check if it's time for a knowledge update
        turns_since_update = (
            self.game_state.turn_count - self.last_knowledge_update_turn
        )
        return (
            self.game_state.turn_count > 0
            and turns_since_update >= self.config.knowledge_update_interval
        )

    def check_periodic_update(self, current_agent_reasoning: str = "") -> None:
        """Check and perform periodic knowledge updates if needed."""
        if not self.should_process_turn():
            return

        try:
            self.log_progress(
                f"Starting periodic knowledge update at turn {self.game_state.turn_count}",
                stage="knowledge_update",
                details=f"Starting knowledge update at turn {self.game_state.turn_count}",
            )

            # Log that we're starting the update
            self.logger.info(
                f"Starting periodic knowledge update at turn {self.game_state.turn_count}",
                extra={
                    "event_type": "knowledge_update_start",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "last_update_turn": self.last_knowledge_update_turn,
                },
            )

            # Include map quality metrics for context
            map_metrics = {}
            try:
                map_metrics = self.map_manager.get_quality_metrics()
                self.log_debug(f"Map quality metrics: {map_metrics}")
            except Exception as e:
                self.log_warning(f"Failed to get map quality metrics: {e}")

            # Perform the knowledge update using "Method 2" - entire episode analysis
            self.log_debug(
                "Calling adaptive knowledge manager update_knowledge_from_turns"
            )
            success = self.adaptive_knowledge_manager.update_knowledge_from_turns(
                episode_id=self.game_state.episode_id,
                start_turn=1,
                end_turn=self.game_state.turn_count,
                is_final_update=False,
            )

            if success:
                self.last_knowledge_update_turn = self.game_state.turn_count

                self.log_progress(
                    f"Knowledge update completed successfully at turn {self.game_state.turn_count}",
                    stage="knowledge_update",
                    details="Knowledge update completed successfully",
                )

                # Log successful update
                self.logger.info(
                    f"Knowledge update completed successfully at turn {self.game_state.turn_count}",
                    extra={
                        "event_type": "knowledge_update_success",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "update_method": "periodic_full_episode",
                    },
                )

                # Update map in knowledge base
                self.update_map_in_knowledge_base()

                # Reload agent knowledge
                self.reload_agent_knowledge()

            else:
                self.log_error(
                    f"Knowledge update failed at turn {self.game_state.turn_count}",
                    details="Knowledge update returned failure",
                )

                self.logger.error(
                    f"Knowledge update failed at turn {self.game_state.turn_count}",
                    extra={
                        "event_type": "knowledge_update_failed",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "update_method": "periodic_full_episode",
                    },
                )

        except Exception as e:
            self.log_error(
                f"Exception during knowledge update: {e}",
                details=f"Knowledge update failed with exception: {e}",
            )

            self.logger.error(
                f"Knowledge update exception: {e}",
                extra={
                    "event_type": "knowledge_update_exception",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e),
                },
            )

    def perform_final_update(self, death_count: int = 0) -> None:
        """Perform final knowledge update at episode end."""
        try:
            self.log_progress(
                f"Starting final knowledge update for episode {self.game_state.episode_id}",
                stage="final_knowledge_update",
                details=f"Final knowledge update for episode {self.game_state.episode_id}",
            )

            # Check if we've done a recent comprehensive update
            turns_since_last_update = (
                self.game_state.turn_count - self.last_knowledge_update_turn
            )
            skip_final_update = turns_since_last_update < (
                self.config.knowledge_update_interval / 2
            )

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
                },
            )

            if not skip_final_update or is_death_episode:
                # Include map quality metrics
                try:
                    self.map_manager.get_quality_metrics()
                except Exception as e:
                    self.log_warning(f"Failed to get map quality metrics: {e}")

                # Perform final knowledge update
                success = self.adaptive_knowledge_manager.update_knowledge_from_turns(
                    episode_id=self.game_state.episode_id,
                    start_turn=1,
                    end_turn=self.game_state.turn_count,
                    is_final_update=True,
                )

                if success:
                    self.log_progress(
                        "Final knowledge update completed successfully",
                        stage="final_knowledge_update",
                        details="Final knowledge update completed",
                    )

                    # Update map in knowledge base
                    self.update_map_in_knowledge_base()
                else:
                    self.log_error(
                        "Final knowledge update failed",
                        details="Final knowledge update returned failure",
                    )
            else:
                self.log_debug(
                    "Skipping final knowledge update - recent comprehensive update was done",
                    details=f"Last update was {turns_since_last_update} turns ago",
                )

        except Exception as e:
            self.log_error(
                f"Exception during final knowledge update: {e}",
                details=f"Final knowledge update failed with exception: {e}",
            )

    def update_map_in_knowledge_base(self) -> None:
        """Update the mermaid map in knowledge base."""
        try:
            self.adaptive_knowledge_manager.update_knowledge_with_map(
                episode_id=self.game_state.episode_id,
                game_map=self.map_manager.game_map
            )
            self.log_debug("Updated map in knowledge base")

        except Exception as e:
            self.log_error(f"Failed to update map in knowledge base: {e}")

    def reload_agent_knowledge(self) -> None:
        """Reload knowledge base in agent for immediate use."""
        try:
            if hasattr(self.agent, "reload_knowledge_base"):
                self.agent.reload_knowledge_base()
                self.log_debug("Agent knowledge base reloaded")
            else:
                self.log_warning("Agent does not support knowledge base reloading")

        except Exception as e:
            self.log_error(f"Failed to reload agent knowledge: {e}")

    def should_synthesize_inter_episode_wisdom(
        self, final_score: int, death_count: int, critic_confidence_history: List[float]
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
            avg_confidence = sum(critic_confidence_history) / len(
                critic_confidence_history
            )
            if avg_confidence >= 0.8:  # High confidence episode
                return True

        return False

    def perform_inter_episode_synthesis(
        self, final_score: int, death_count: int, critic_confidence_history: List[float]
    ) -> None:
        """Perform inter-episode wisdom synthesis."""
        try:
            # Check if synthesis should occur
            if not self.should_synthesize_inter_episode_wisdom(
                final_score, death_count, critic_confidence_history
            ):
                self.log_debug("Skipping inter-episode synthesis - criteria not met")
                return

            self.log_progress(
                f"Starting inter-episode wisdom synthesis for episode {self.game_state.episode_id}",
                stage="wisdom_synthesis",
                details=f"Synthesis for episode {self.game_state.episode_id}",
            )

            # Collect episode data for synthesis
            episode_data = {
                "episode_id": self.game_state.episode_id,
                "turn_count": self.game_state.turn_count,
                "final_score": final_score,
                "death_count": death_count,
                "episode_ended_in_death": death_count > 0,  # Add missing key
                "discovered_objectives": self.game_state.discovered_objectives.copy(),
                "completed_objectives": [
                    obj["objective"] for obj in self.game_state.completed_objectives
                ],
                "avg_critic_score": sum(critic_confidence_history)
                / len(critic_confidence_history)
                if critic_confidence_history
                else 0,
                "map_metrics": {},
            }

            # Add map metrics if available
            try:
                episode_data["map_metrics"] = self.map_manager.get_quality_metrics()
            except Exception as e:
                self.log_warning(f"Failed to get map metrics for synthesis: {e}")

            self.logger.info(
                "Inter-episode synthesis starting",
                extra={
                    "event_type": "inter_episode_synthesis_start",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "final_score": final_score,
                    "death_count": death_count,
                    "objectives_discovered": len(self.game_state.discovered_objectives),
                    "objectives_completed": len(self.game_state.completed_objectives),
                },
            )

            # Perform synthesis
            success = self.adaptive_knowledge_manager.synthesize_inter_episode_wisdom(
                episode_data=episode_data
            )

            if success:
                self.log_progress(
                    "Inter-episode wisdom synthesis completed successfully",
                    stage="wisdom_synthesis",
                    details="Synthesis completed successfully",
                )
            else:
                self.log_error(
                    "Inter-episode wisdom synthesis failed",
                    details="Synthesis returned failure",
                )

        except Exception as e:
            self.log_error(
                f"Exception during inter-episode synthesis: {e}",
                details=f"Synthesis failed with exception: {e}",
            )

    def migrate_persistent_wisdom_to_knowledgebase(self) -> bool:
        """
        One-time migration: merge existing persistent_wisdom.md into knowledgebase.md
        CROSS-EPISODE INSIGHTS section. Can be safely called multiple times.

        Returns:
            True if migration occurred, False if skipped or failed
        """
        try:
            import os
            from config import get_config

            config = get_config()
            old_wisdom_file = config.orchestrator.persistent_wisdom_file

            # Check if old file exists
            if not os.path.exists(old_wisdom_file):
                self.log_debug("No persistent_wisdom.md file to migrate")
                return False

            # Read old wisdom content
            with open(old_wisdom_file, "r", encoding="utf-8") as f:
                old_wisdom = f.read().strip()

            if not old_wisdom:
                self.log_debug("persistent_wisdom.md is empty, skipping migration")
                os.remove(old_wisdom_file)  # Clean up empty file
                return False

            # Load current knowledge base
            kb_file = self.adaptive_knowledge_manager.output_file
            existing_kb = ""
            try:
                with open(kb_file, "r", encoding="utf-8") as f:
                    existing_kb = f.read()
            except FileNotFoundError:
                existing_kb = "# Zork Game World Knowledge Base\n\n"

            # Check if CROSS-EPISODE INSIGHTS section already exists
            if "## CROSS-EPISODE INSIGHTS" in existing_kb:
                self.log_info("CROSS-EPISODE INSIGHTS section already exists in knowledgebase.md")
                # Backup old file and remove it
                backup_path = f"{old_wisdom_file}.migrated"
                os.rename(old_wisdom_file, backup_path)
                self.log_info(f"Backed up persistent_wisdom.md to {backup_path}")
                return False

            # Merge old wisdom into new section
            self.log_info("Migrating persistent_wisdom.md to knowledgebase.md CROSS-EPISODE INSIGHTS section")

            # Use adaptive knowledge manager to update the section
            updated_kb = self.adaptive_knowledge_manager._update_section_content(
                existing_kb,
                "CROSS-EPISODE INSIGHTS",
                f"### Migrated from persistent_wisdom.md\n\n{old_wisdom}"
            )

            # Save updated knowledge base
            with open(kb_file, "w", encoding="utf-8") as f:
                f.write(updated_kb)

            # Backup and remove old file
            backup_path = f"{old_wisdom_file}.migrated"
            os.rename(old_wisdom_file, backup_path)

            self.log_progress(
                "Successfully migrated persistent_wisdom.md to knowledgebase.md",
                stage="migration",
                details=f"Old file backed up to {backup_path}"
            )

            return True

        except Exception as e:
            self.log_error(f"Failed to migrate persistent_wisdom.md: {e}")
            return False

    def get_knowledge_base_summary(self) -> str:
        """Get knowledge base content without the map section."""
        try:
            with open("knowledgebase.md", "r") as f:
                content = f.read()

            # Remove mermaid map sections using regex
            # This removes everything from "## Map" to the end or next major section
            content_without_map = re.sub(
                r"## Map.*?(?=\n## |\Z)", "", content, flags=re.DOTALL
            )

            return content_without_map.strip()

        except FileNotFoundError:
            self.log_warning("Knowledge base file not found")
            return ""
        except Exception as e:
            self.log_error(f"Failed to read knowledge base: {e}")
            return ""

    def get_llm_client(self):
        """Access LLM client for knowledge-related analysis."""
        if self.adaptive_knowledge_manager:
            return self.adaptive_knowledge_manager.client
        return None

    def get_export_data(self) -> Dict[str, Any]:
        """Get knowledge base data for state export (matching old orchestrator format)."""
        try:
            # Read knowledge base file (like old orchestrator did)
            import os

            with open("knowledgebase.md", "r") as f:
                content = f.read()

            # Remove the mermaid diagram section more precisely
            # (matching old orchestrator logic)
            pattern = r"## CURRENT WORLD MAP\s*\n\s*```mermaid\s*\n.*?\n```"
            knowledge_only = re.sub(pattern, "", content, flags=re.DOTALL)

            # Clean up any extra whitespace that might be left
            knowledge_only = re.sub(r"\n\s*\n\s*\n", "\n\n", knowledge_only)

            return {
                "content": knowledge_only.strip(),
                "last_updated": os.path.getmtime("knowledgebase.md")
                if os.path.exists("knowledgebase.md")
                else None,
                "object_events": self.object_events[-50:],  # Include recent 50 events
                "total_object_events": len(self.object_events),
            }
        except FileNotFoundError:
            self.log_debug("Knowledge base file not found, returning empty content")
            return {
                "content": "# Zork Game World Knowledge Base\n\nNo knowledge base content available yet.",
                "last_updated": None,
            }
        except Exception as e:
            self.log_error(f"Failed to get knowledge export data: {e}")
            return {
                "content": "# Zork Game World Knowledge Base\n\nError loading knowledge base content.",
                "last_updated": None,
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current knowledge manager status."""
        status = super().get_status()
        status.update(
            {
                "last_knowledge_update_turn": self.last_knowledge_update_turn,
                "turns_since_last_update": self.game_state.turn_count
                - self.last_knowledge_update_turn,
                "knowledge_update_interval": self.config.knowledge_update_interval,
                "has_adaptive_manager": self.adaptive_knowledge_manager is not None,
                "has_llm_client": self.get_llm_client() is not None,
                "object_events_tracked": len(self.object_events),
            }
        )
        return status
