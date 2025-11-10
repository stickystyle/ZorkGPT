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
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration

# Import boto3 only when needed
try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class StateManager(BaseManager):
    """
    Manages all state-related functionality for ZorkGPT.

    Responsibilities:
    - Episode lifecycle and state management
    - State export and import to files and S3
    - Memory management and history tracking
    - Cross-episode persistent state tracking
    - State queries and reporting
    """

    def __init__(
        self, logger, config: GameConfiguration, game_state: GameState, llm_client=None
    ):
        super().__init__(logger, config, game_state, "state_manager")
        self.llm_client = llm_client

        # S3 client for state uploads
        self.s3_client = None
        if config.s3_bucket and BOTO3_AVAILABLE:
            try:
                self.s3_client = boto3.client("s3")
            except Exception as e:
                self.log_warning(f"Failed to initialize S3 client: {e}")
        elif config.s3_bucket and not BOTO3_AVAILABLE:
            self.log_warning(
                "S3 bucket configured but boto3 not available. Install with: uv sync --extra s3"
            )

        # State loop detection (Phase 6)
        self.state_history: List[int] = []  # Track state hashes for loop detection
        self.max_state_history_size: int = 1000  # Prevent unbounded memory growth

    def reset_episode(self) -> None:
        """Reset episode-specific state for a new episode."""
        self.log_debug("Resetting episode state")

        # NOTE: Do NOT call game_state.reset_episode() here!
        # Episode ID generation and GameState reset is handled by EpisodeSynthesizer
        # This method only resets StateManager's internal state

        # Reset state loop detection (Phase 6)
        self.state_history.clear()

        self.log_debug("Episode state reset completed")

    def process_turn(self) -> None:
        """Process state management for the current turn."""
        # Note: State export is handled by orchestrator coordination
        # via _export_coordinated_state() which gathers data from all managers
        pass

    def should_process_turn(self) -> bool:
        """Check if state needs processing this turn."""
        return False  # No turn-by-turn processing needed

    def track_state_hash(self, jericho_interface) -> Optional[bool]:
        """
        Track current game state hash and detect loops.

        Uses Jericho's state tuple to generate a hash and detect if we've
        returned to an exact previous game state, which indicates the agent
        is stuck in a loop.

        Args:
            jericho_interface: JerichoInterface instance for accessing game state

        Returns:
            True if loop detected, False if no loop, None on error
        """
        try:
            # Get state hash from Jericho (hash of state tuple)
            state_tuple = jericho_interface.save_state()
            # Serialize the state tuple to bytes (handles numpy arrays inside)
            # and hash the bytes for consistent comparison
            state_hash = hash(pickle.dumps(state_tuple))

            # Check for loop
            if state_hash in self.state_history:
                loop_index = self.state_history.index(state_hash)
                turns_in_loop = len(self.state_history) - loop_index

                self.logger.warning(
                    "Exact game state loop detected",
                    extra={
                        "event_type": "state_loop_detected",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "state_hash": state_hash,
                        "loop_start_turn": loop_index + 1,
                        "turns_in_loop": turns_in_loop,
                    },
                )
                return True

            # Add to history (with size limit)
            self.state_history.append(state_hash)
            if len(self.state_history) > self.max_state_history_size:
                self.state_history.pop(0)  # Remove oldest

            return False

        except Exception as e:
            self.log_error(f"Failed to track state hash: {e}")
            return None


    def get_current_state(
        self, map_data: Dict[str, Any] = None, knowledge_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get comprehensive current state for export."""
        try:
            # Build models structure (matching old orchestrator format)
            models_data = {
                "agent": self.config.agent_model,
                "critic": self.config.critic_model,
                "extractor": self.config.info_ext_model,
                "knowledge_base": self.config.analysis_model,  # Use analysis model from config
            }

            state_data = {
                "metadata": {
                    "episode_id": self.game_state.episode_id,
                    "timestamp": datetime.now().isoformat(),
                    "turn_count": self.game_state.turn_count,
                    "game_over": self.game_state.game_over_flag,
                    "score": self.game_state.previous_zork_score,  # Viewer expects 'score'
                    "max_turns": self.config.max_turns_per_episode,  # Viewer expects 'max_turns'
                    "models": models_data,  # Structured models object
                },
                "current_state": {
                    "location": self.game_state.current_room_name_for_map,  # Viewer expects 'location'
                    "inventory": self.game_state.current_inventory,
                    "in_combat": self.get_combat_status(),
                    "death_count": self.game_state.death_count,
                    "discovered_objectives": self.game_state.discovered_objectives,
                    "completed_objectives": self.game_state.completed_objectives,  # Full objects, not count
                    "objective_update_turn": self.game_state.objective_update_turn,
                },
                "recent_log": self.get_recent_log(),
                "performance": {
                    "avg_critic_score": self.get_avg_critic_score(),
                    "recent_actions": self.get_recent_action_summary(),
                },
                "context_management": {
                    "memory_entries": len(self.game_state.memory_log_history),
                },
            }

            # Add map data if provided
            if map_data:
                state_data["map"] = map_data

            # Add knowledge base data if provided
            if knowledge_data:
                state_data["knowledge_base"] = knowledge_data

            return state_data

        except Exception as e:
            self.log_error(f"Failed to get current state: {e}")
            return {}

    def export_current_state(
        self, map_data: Dict[str, Any] = None, knowledge_data: Dict[str, Any] = None
    ) -> bool:
        """Export current state to file and optionally to S3."""
        try:
            if not self.config.enable_state_export:
                return True

            state_data = self.get_current_state(
                map_data=map_data, knowledge_data=knowledge_data
            )
            if not state_data:
                return False

            # Export to local file
            with open(self.config.state_export_file, "w") as f:
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
        """
        Upload current state to S3 using dual strategy for web viewer compatibility.

        This method uploads the state data to two S3 locations:
        1. current_state.json - Always overwritten, used for live monitoring by zork_viewer.html
        2. snapshots/{episode_id}/turn_{turn}.json - Historical preservation organized by episode

        This structure matches what the web viewer expects for both live monitoring
        and historical episode browsing functionality.

        Returns:
            bool: True if at least one upload succeeded, False if both failed or S3 not configured
        """
        try:
            if not self.s3_client or not self.config.s3_bucket:
                return False

            json_content = json.dumps(state_data, indent=2)
            upload_success_count = 0

            # 1. Upload current state (always overwritten for live monitoring)
            current_state_key = f"{self.config.s3_key_prefix}current_state.json"
            try:
                self.s3_client.put_object(
                    Bucket=self.config.s3_bucket,
                    Key=current_state_key,
                    Body=json_content,
                    ContentType="application/json",
                    CacheControl="no-cache, must-revalidate",  # Force fresh data for live monitoring
                )
                self.log_debug(f"Uploaded current state to S3: {current_state_key}")
                upload_success_count += 1
            except Exception as e:
                self.log_warning(f"Failed to upload current state to S3: {e}")

            # 2. Upload historical snapshot (organized by episode and turn)
            turn_count = self.game_state.turn_count
            snapshot_key = f"{self.config.s3_key_prefix}snapshots/{self.game_state.episode_id}/turn_{turn_count}.json"
            try:
                self.s3_client.put_object(
                    Bucket=self.config.s3_bucket,
                    Key=snapshot_key,
                    Body=json_content,
                    ContentType="application/json",
                )
                self.log_debug(f"Uploaded snapshot to S3: {snapshot_key}")
                upload_success_count += 1
            except Exception as e:
                self.log_warning(f"Failed to upload snapshot to S3: {e}")

            # Return True if at least one upload succeeded
            return upload_success_count > 0

        except Exception as e:
            self.log_error(f"Failed to upload state to S3: {e}")
            return False

    def get_recent_log(self, num_entries: int = 10) -> List[Dict[str, Any]]:
        """Get recent game log entries with reasoning."""
        try:
            recent_log = []

            # Get recent action history with reasoning
            recent_actions = self.game_state.action_history[-num_entries:]
            recent_reasoning = self.game_state.action_reasoning_history[-num_entries:]
            recent_critic_evals = self.game_state.critic_evaluation_history[
                -num_entries:
            ]
            recent_extracted_info = self.game_state.extracted_info_history[
                -num_entries:
            ]

            for i, entry in enumerate(recent_actions):
                reasoning_data = (
                    recent_reasoning[i] if i < len(recent_reasoning) else {}
                )
                reasoning_text = (
                    reasoning_data.get("reasoning", "")
                    if isinstance(reasoning_data, dict)
                    else str(reasoning_data)
                )

                log_entry = {
                    "turn": self.game_state.turn_count - len(recent_actions) + i + 1,
                    "action": entry.action,
                    "zork_response": entry.response,  # Viewer expects 'zork_response'
                    "reasoning": reasoning_text,
                    "location_id": entry.location_id,
                    "location_name": entry.location_name,
                }

                # Add critic data if available
                if i < len(recent_critic_evals):
                    critic_data = recent_critic_evals[i]
                    log_entry["critic_score"] = critic_data.get("critic_score", 0.0)
                    log_entry["critic_justification"] = critic_data.get(
                        "critic_justification", ""
                    )
                    log_entry["was_overridden"] = critic_data.get(
                        "was_overridden", False
                    )
                    log_entry["override_reason"] = critic_data.get(
                        "override_reason", None
                    )
                    log_entry["rejected_actions"] = critic_data.get(
                        "rejected_actions", []
                    )

                # Add extracted info if available
                if i < len(recent_extracted_info):
                    log_entry["extracted_info"] = recent_extracted_info[i]

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
                    combat_keywords = [
                        "combat",
                        "fight",
                        "attack",
                        "enemy",
                        "monster",
                        "battle",
                    ]

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
            return [entry.action for entry in recent_actions]

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
        status.update(
            {
                "memory_entries": len(self.game_state.memory_log_history),
                "action_history_length": len(self.game_state.action_history),
                "export_enabled": self.config.enable_state_export,
                "s3_configured": self.s3_client is not None,
                "state_history_size": len(self.state_history),
                "loop_detection_enabled": True,
            }
        )
        return status
