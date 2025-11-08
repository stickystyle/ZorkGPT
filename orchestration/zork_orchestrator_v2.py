
import time
import logging
from collections import deque
from contextlib import nullcontext
from typing import Dict, Any, List, Tuple, Optional

from session.game_state import GameState
from session.game_configuration import GameConfiguration
from managers import (
    ObjectiveManager,
    KnowledgeManager,
    MapManager,
    StateManager,
    ContextManager,
    EpisodeSynthesizer,
    RejectionManager,
)
from managers.simple_memory_manager import SimpleMemoryManager
from zork_agent import ZorkAgent
from zork_critic import ZorkCritic
from hybrid_zork_extractor import HybridZorkExtractor
from game_interface.core.jericho_interface import JerichoInterface
from logger import setup_logging

# Langfuse for observability
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None


class ZorkOrchestratorV2:
    """
    Streamlined orchestrator that coordinates specialized managers.

    This class is responsible for:
    - High-level game loop coordination
    - Manager initialization and lifecycle
    - Inter-manager communication
    - Game interface management (Jericho)

    All domain-specific logic is delegated to specialized managers.
    """

    def __init__(
        self,
        episode_id: str,
        max_turns_per_episode: int = None,
    ):
        """Initialize the orchestrator with configuration loaded from TOML."""

        # Load configuration from TOML file
        self.config = GameConfiguration.from_toml()

        # Override max_turns_per_episode if explicitly provided
        if max_turns_per_episode is not None:
            self.config.max_turns_per_episode = max_turns_per_episode

        # Initialize logger
        self.logger = setup_logging(
            self.config.episode_log_file,
            self.config.json_log_file,
            log_level=logging.DEBUG,
        )

        # Initialize shared game state
        self.game_state = GameState()
        self.game_state.episode_id = episode_id

        # Initialize Langfuse client for observability
        self.langfuse_client: Optional[Langfuse] = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_client = Langfuse()
                self.logger.info(
                    "Langfuse session tracking enabled",
                    extra={
                        "event_type": "langfuse_enabled",
                        "episode_id": episode_id,
                    }
                )
            except Exception as e:
                self.logger.warning(
                    f"Langfuse not available, continuing without session tracking: {e}",
                    extra={"event_type": "langfuse_unavailable"}
                )
        else:
            self.logger.info(
                "Langfuse not installed, continuing without session tracking",
                extra={"event_type": "langfuse_not_installed"}
            )

        # Setup episode-specific logging
        from logger import setup_episode_logging

        workdir = self.config.zork_game_workdir
        self.episode_log_file = setup_episode_logging(episode_id, workdir)

        # Initialize Jericho interface
        self.jericho_interface = JerichoInterface(
            game_file_path=self.config.game_file_path, logger=self.logger
        )

        # Initialize core game components
        self._initialize_game_components()

        # Initialize managers
        self._initialize_managers()

        # Track critic confidence for synthesis decisions
        self.critic_confidence_history = []

        self.logger.info(
            "ZorkOrchestrator v2 initialized with Jericho",
            extra={
                "event_type": "orchestrator_init",
                "episode_id": episode_id,
                "episode_log_file": self.episode_log_file,
                "game_file_path": self.config.game_file_path,
                "agent_model": self.config.agent_model,
                "critic_model": self.config.critic_model,
                "info_ext_model": self.config.info_ext_model,
                "max_turns": self.config.max_turns_per_episode,
            },
        )

    def _initialize_game_components(self) -> None:
        """Initialize core game components (agent, critic, extractor)."""
        # Initialize agent with config
        self.agent = ZorkAgent(
            config=self.config,
            logger=self.logger,
            episode_id=self.game_state.episode_id,
            model=self.config.agent_model,
        )

        # Initialize critic with config
        self.critic = ZorkCritic(
            config=self.config,
            logger=self.logger,
            episode_id=self.game_state.episode_id,
            model=self.config.critic_model,
        )

        # Initialize extractor with config and Jericho interface
        self.extractor = HybridZorkExtractor(
            jericho_interface=self.jericho_interface,
            config=self.config,
            episode_id=self.game_state.episode_id,
            logger=self.logger,
            model=self.config.info_ext_model,
        )

    def _initialize_managers(self) -> None:
        """Initialize all specialized managers."""
        # Initialize managers in dependency order

        # Map manager (no dependencies)
        self.map_manager = MapManager(
            logger=self.logger, config=self.config, game_state=self.game_state
        )

        # Context manager (no dependencies)
        self.context_manager = ContextManager(
            logger=self.logger, config=self.config, game_state=self.game_state
        )

        # Rejection manager (no dependencies)
        self.rejection_manager = RejectionManager(
            logger=self.logger, config=self.config, game_state=self.game_state
        )

        # Simple memory manager (needs LLM client)
        self.simple_memory = SimpleMemoryManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            llm_client=self.agent.client  # Share LLM client
        )

        # State manager (needs potential S3 client)
        self.state_manager = StateManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            llm_client=self.agent.client,  # Share LLM client
        )

        # Knowledge manager (needs agent and map manager references)
        self.knowledge_manager = KnowledgeManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            agent=self.agent,
            game_map=self.map_manager,
            json_log_file=self.config.json_log_file,
        )

        # Objective manager (needs knowledge, map, and memory managers)
        self.objective_manager = ObjectiveManager(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            adaptive_knowledge_manager=self.knowledge_manager.adaptive_knowledge_manager,
            map_manager=self.map_manager,  # NEW: Pass MapManager for spatial context
            simple_memory=self.simple_memory,  # NEW: Pass SimpleMemoryManager for memory access
            langfuse_client=self.langfuse_client,  # NEW: Pass Langfuse client for span tracing
        )

        # Episode synthesizer (needs references to other managers)
        self.episode_synthesizer = EpisodeSynthesizer(
            logger=self.logger,
            config=self.config,
            game_state=self.game_state,
            knowledge_manager=self.knowledge_manager,
            state_manager=self.state_manager,
            llm_client=self.agent.client,
        )

        # Inject simple_memory reference into context_manager
        self.context_manager.simple_memory = self.simple_memory

        # Create ordered manager list for processing
        self.managers = [
            self.map_manager,
            self.context_manager,
            self.rejection_manager,
            self.simple_memory,
            self.state_manager,
            self.objective_manager,
            self.knowledge_manager,
            self.episode_synthesizer,
        ]

    def play_episode(self) -> int:
        """
        Play a complete episode of Zork using Jericho.

        Returns:
            Final score achieved in the episode
        """
        try:
            # Initialize new episode across all managers
            self.episode_synthesizer.initialize_episode(
                episode_id=self.game_state.episode_id,
                agent=self.agent,
                extractor=self.extractor,
                critic=self.critic,
            )

            # Restore rejection state if available
            if self.game_state.rejection_state:
                self.rejection_manager.restore_state(self.game_state.rejection_state)

            # Start Jericho interface
            self.logger.debug("DEBUG: About to call jericho_interface.start()")
            initial_game_state = self.jericho_interface.start()
            self.logger.debug("DEBUG: jericho_interface.start() completed")

            self.logger.info(
                "Jericho interface started successfully",
                extra={
                    "event_type": "jericho_started",
                    "episode_id": self.game_state.episode_id,
                    "intro_length": len(initial_game_state),
                },
            )

            # Enable verbose mode to get full room descriptions on every visit
            verbose_response = self.jericho_interface.send_command("verbose")
            self.logger.info(
                f"Enabled verbose mode: {verbose_response}",
                extra={
                    "event_type": "verbose_mode_enabled",
                    "episode_id": self.game_state.episode_id,
                    "verbose_response": verbose_response,
                },
            )

            # Extract initial state information
            initial_extracted_info = self.extractor.extract_info(initial_game_state)
            self._process_extraction(initial_extracted_info, "", initial_game_state)

            # Generate turn 1 objectives BEFORE game loop starts (if memories exist)
            self._generate_turn_1_objectives()

            # Run the main game loop
            final_score = self._run_game_loop(initial_game_state)

            # Finalize episode
            self.episode_synthesizer.finalize_episode(
                final_score=final_score,
                critic_confidence_history=self.critic_confidence_history,
            )

            # Export final coordinated state (including map data)
            self._export_coordinated_state()

            # Save map state for cross-episode persistence
            self.map_manager.save_map_state()

            # Flush Langfuse traces if available (BEFORE closing Jericho to ensure delivery)
            if self.langfuse_client:
                try:
                    self.langfuse_client.flush()
                    self.logger.info(
                        "Langfuse traces flushed for episode",
                        extra={
                            "event_type": "langfuse_flushed",
                            "episode_id": self.game_state.episode_id,
                        }
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to flush Langfuse traces: {e}",
                        extra={
                            "event_type": "langfuse_flush_error",
                            "error": str(e)
                        }
                    )

            # Note: Jericho close() is handled in finally block to ensure cleanup
            # in all exit paths (success, exception, early return)
            return final_score

        except Exception as e:
            self.logger.error(
                f"Episode failed with exception: {e}",
                extra={
                    "event_type": "episode_exception",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e),
                },
            )
            return self.game_state.previous_zork_score
        finally:
            # Ensure Jericho interface is always closed, even on exceptions
            if hasattr(self, 'jericho_interface') and self.jericho_interface:
                try:
                    self.jericho_interface.close()
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Error during Jericho cleanup: {cleanup_error}",
                        extra={
                            "event_type": "jericho_cleanup_error",
                            "error": str(cleanup_error),
                        }
                    )

    def _track_score_for_progress_detection(self) -> None:
        """Track score changes for progress detection.

        Initializes tracking on first call with current turn and score.
        Detects both score increases and decreases, resetting the stuck
        counter on any change.

        Note: Works correctly even if game starts with non-zero score
        (e.g., restored save state or non-standard game start).

        Note: Stuck detection checks occur every N turns (stuck_check_interval),
        so there may be a delay of up to 1 turn between score change and
        detection of unstuck behavior.
        """
        if not hasattr(self, '_last_score_change_turn'):
            self._last_score_change_turn = 0
            self._last_tracked_score = self.game_state.previous_zork_score
            return

        current_score = self.game_state.previous_zork_score

        # Detect any score change (increase or decrease)
        # Note: Death/penalty resets counter because it represents discovery/learning,
        # even if progress is negative. This prevents premature termination during
        # dangerous exploration or combat sequences.
        if current_score != self._last_tracked_score:
            self._last_score_change_turn = self.game_state.turn_count

            self.logger.info(
                f"Score changed: {self._last_tracked_score} → {current_score}",
                extra={
                    "event_type": "score_change",
                    "turn": self.game_state.turn_count,
                    "old_score": self._last_tracked_score,
                    "new_score": current_score,
                }
            )

            self._last_tracked_score = current_score

    def _get_turns_since_score_change(self) -> int:
        """Calculate turns since last score change."""
        if not hasattr(self, '_last_score_change_turn'):
            return self.game_state.turn_count

        return self.game_state.turn_count - self._last_score_change_turn

    def _track_action_history(self, action: str) -> None:
        """Track actions for novelty detection.

        Maintains a sliding window of recent actions to detect repeated behaviors.
        """
        if not hasattr(self, '_action_history'):
            max_window = max(self.config.action_novelty_window, 20)
            self._action_history = deque(maxlen=max_window)

        self._action_history.append(action.strip().lower())

    def _detect_action_novelty(self, proposed_action: str) -> dict:
        """Detect if proposed action is novel (not recently tried).

        Returns:
            dict with keys:
            - is_novel (bool): True if action not in recent history
            - recent_actions (int): Number of actions in history
            - window_size (int): Configured window size
        """
        if not hasattr(self, '_action_history'):
            return {"is_novel": True, "recent_actions": 0, "window_size": 0}

        window = min(self.config.action_novelty_window, len(self._action_history))
        # Convert deque to list for slicing
        recent = list(self._action_history)[-window:] if window > 0 else []

        action_normalized = proposed_action.strip().lower()
        is_novel = action_normalized not in recent

        return {
            "is_novel": is_novel,
            "recent_actions": len(recent),
            "window_size": window
        }

    def _detect_unexplored_exits(self) -> dict:
        """Detect unexplored exits from current location.

        Returns:
            dict with keys:
            - has_unexplored (bool): True if unexplored exits exist
            - unexplored_exits (list): List of direction strings
            - all_exits (list): All known exits from room
        """
        current_location_id = self.game_state.current_room_id

        # Get room from map
        if current_location_id not in self.map_manager.game_map.rooms:
            return {
                "has_unexplored": False,
                "unexplored_exits": [],
                "all_exits": []
            }

        # Get connections for this room
        if current_location_id not in self.map_manager.game_map.connections:
            return {
                "has_unexplored": False,
                "unexplored_exits": [],
                "all_exits": []
            }

        room_connections = self.map_manager.game_map.connections[current_location_id]
        all_exits = list(room_connections.keys())

        # Check which exits lead to unexplored rooms (destination_id not in map)
        unexplored = []
        for direction, destination_id in room_connections.items():
            if destination_id not in self.map_manager.game_map.rooms:
                unexplored.append(direction)

        return {
            "has_unexplored": len(unexplored) > 0,
            "unexplored_exits": unexplored,
            "all_exits": all_exits
        }

    def _build_stuck_countdown_warning(self) -> str:
        """Build urgent countdown warning when agent is stuck.

        Returns escalating warnings based on how close to termination.
        Implements Ryan's requirement for explicit countdown messaging.
        """
        if not self.config.enable_stuck_warnings:
            return ""

        turns_stuck = self._get_turns_since_score_change()

        # Don't warn until threshold reached
        if turns_stuck < self.config.stuck_warning_threshold:
            return ""

        turns_until_death = self.config.max_turns_stuck - turns_stuck

        # Escalating urgency levels
        if turns_until_death <= 5:
            urgency = "🚨 CRITICAL EMERGENCY"
            tone = "IMMEDIATE"
        elif turns_until_death <= 10:
            urgency = "⚠️ URGENT WARNING"
            tone = "URGENT"
        else:
            urgency = "⚠️ SCORE STAGNATION DETECTED"
            tone = "IMPORTANT"

        warning = f"""
{'='*70}
{urgency}
{'='*70}

Your PRIMARY GOAL is to INCREASE YOUR SCORE.

You have made NO SCORE PROGRESS for {turns_stuck} turns.

If you do not increase your score, you will DIE in {turns_until_death} turns.

SUGGESTED STRATEGIES TO BREAK FREE:
• Try a completely different location (move 3+ rooms away)
• Attempt a different puzzle approach
• Explore unexplored exits
• Consider abandoning your current strategy

SURVIVAL DEPENDS ON SCORE INCREASE.
{'='*70}
"""
        return warning

    def _build_exploration_hints(
        self,
        proposed_action: str,
        novelty_info: dict,
        unexplored_info: dict
    ) -> str:
        """Build context hints for exploration.

        Returns informational guidance (not penalties) about:
        - Action novelty
        - Unexplored exits
        """
        if not self.config.enable_exploration_hints:
            return ""

        hints = []

        # Action novelty hint
        if not novelty_info["is_novel"] and novelty_info["recent_actions"] > 0:
            hints.append(
                f"Note: You recently tried '{proposed_action}' "
                f"in the last {novelty_info['window_size']} actions. "
                f"Consider trying something different."
            )

        # Unexplored exits hint
        if unexplored_info["has_unexplored"]:
            exits_str = ", ".join(unexplored_info["unexplored_exits"])
            hints.append(
                f"Exploration opportunity: This location has unexplored exits: {exits_str}"
            )

        if not hints:
            return ""

        return "\n\n" + "\n".join(hints) + "\n"

    def _track_location_history(self) -> None:
        """Track location at each turn for loop detection.

        Uses Z-machine location IDs (integers) - NOT room names.
        This is architecturally required for stability across episodes.

        Maintains a sliding window of the last 20 locations.
        """
        if not hasattr(self, '_location_id_history'):
            self._location_id_history = deque(maxlen=20)

        # Use Z-machine location ID (integer) - NOT room names
        location_obj = self.jericho_interface.get_location_structured()
        current_location_id = location_obj.num if location_obj else 0
        self._location_id_history.append(current_location_id)

    def _detect_location_revisit(self) -> dict:
        """Detect if current location was recently visited.

        Returns:
            dict with keys:
            - detected (bool): Whether revisit found
            - location_id (int): Current location ID
            - recent_visits (int): Number of times seen in window
            - window_size (int): Number of locations examined
        """
        if not hasattr(self, '_location_id_history'):
            return {
                "detected": False,
                "location_id": None,
                "recent_visits": 0,
                "window_size": 0
            }

        if len(self._location_id_history) == 0:
            return {
                "detected": False,
                "location_id": None,
                "recent_visits": 0,
                "window_size": 0
            }

        current_location_id = self._location_id_history[-1]

        # Check recent history (excluding current location)
        window_size = min(
            self.config.location_revisit_window,
            len(self._location_id_history) - 1  # Exclude current
        )

        if window_size < 1:
            return {
                "detected": False,
                "location_id": current_location_id,
                "recent_visits": 0,
                "window_size": 0
            }

        # Convert deque to list for slicing
        history_list = list(self._location_id_history)
        recent_history = history_list[-(window_size + 1):-1]
        recent_visits = recent_history.count(current_location_id)

        return {
            "detected": recent_visits > 0,
            "location_id": current_location_id,
            "recent_visits": recent_visits,
            "window_size": window_size
        }

    def _apply_location_revisit_penalty(
        self,
        base_score: float,
        revisit_info: dict
    ) -> Tuple[float, str]:
        """Apply programmatic penalty for location revisits.

        Args:
            base_score: Critic's original confidence score
            revisit_info: Dict from _detect_location_revisit()

        Returns:
            (adjusted_score, reason_string)
        """
        if not self.config.enable_location_penalty:
            return base_score, ""

        if not revisit_info["detected"]:
            return base_score, ""

        # Calculate penalty: -0.2 per revisit
        penalty = self.config.location_revisit_penalty * revisit_info["recent_visits"]
        adjusted_score = base_score + penalty

        # Clamp to [0.0, 1.0]
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        # Get location name for logging
        location_name = "Unknown"
        location_id = revisit_info["location_id"]
        if location_id in self.map_manager.game_map.rooms:
            location_name = self.map_manager.game_map.rooms[location_id].name

        reason = (
            f"Location revisit penalty {penalty:.2f} "
            f"({revisit_info['recent_visits']}x return to {location_name})"
        )

        return adjusted_score, reason

    def _run_game_loop(self, initial_state: str) -> int:
        """Run the main game loop."""
        current_game_state = initial_state

        while (
            not self.game_state.game_over_flag
            and self.game_state.turn_count < self.config.max_turns_per_episode
        ):
            self.game_state.turn_count += 1

            # Add turn delay if configured
            if self.config.turn_delay_seconds > 0:
                time.sleep(self.config.turn_delay_seconds)

            # Run a single turn
            action_taken, next_game_state = self._run_turn(current_game_state)

            if next_game_state:
                current_game_state = next_game_state

            # Track score for progress detection
            self._track_score_for_progress_detection()

            # Check for stuck behavior (every N turns)
            if self.game_state.turn_count % self.config.stuck_check_interval == 0:
                turns_stuck = self._get_turns_since_score_change()

                if turns_stuck >= self.config.max_turns_stuck:
                    self.logger.warning(
                        f"Terminating episode: no progress for {turns_stuck} turns "
                        f"(score stuck at {self.game_state.previous_zork_score})",
                        extra={
                            "event_type": "stuck_termination",
                            "episode_id": self.game_state.episode_id,
                            "turn": self.game_state.turn_count,
                            "score": self.game_state.previous_zork_score,
                            "turns_stuck": turns_stuck,
                        }
                    )
                    self.game_state.game_over_flag = True
                    self.game_state.termination_reason = "stuck_no_progress"
                    return self.game_state.previous_zork_score

            # Check periodic updates for managers
            self._check_periodic_updates()

            # Export state after every turn for live monitoring
            self._export_coordinated_state()

        # Log episode completion
        self.logger.info(
            "Episode completed",
            extra={
                "event_type": "episode_completed",
                "episode_id": self.game_state.episode_id,
                "turn": self.game_state.turn_count,
                "final_score": self.game_state.previous_zork_score,
                "game_over": self.game_state.game_over_flag,
                "reason": "game_over"
                if self.game_state.game_over_flag
                else "max_turns",
            },
        )

        return self.game_state.previous_zork_score

    def _run_turn(self, current_state: str) -> Tuple[str, str]:
        """Run a single game turn with optional Langfuse tracing and error recovery."""
        try:
            # Start Langfuse trace for this turn if available
            if self.langfuse_client:
                with self.langfuse_client.start_as_current_span(
                    name=f"turn-{self.game_state.turn_count}",
                    input={"game_state_preview": current_state[:200]},  # First 200 chars
                    metadata={
                        "turn_number": self.game_state.turn_count,
                        "score_before": self.game_state.previous_zork_score,
                        "location_id": self.game_state.current_room_id,
                        "location_name": self.game_state.current_room_name_for_map,
                    },
                ) as turn_span:
                    # Set trace-level attributes (session, user, tags)
                    turn_span.update_trace(
                        session_id=self.game_state.episode_id,
                        user_id="zorkgpt-agent",
                        tags=["zorkgpt", "game-turn"],
                    )

                    # Execute turn logic with trace context
                    action_taken, next_game_state = self._execute_turn_logic(current_state)

                    # Update span with outcome
                    turn_span.update(
                        output={
                            "action_taken": action_taken,
                            "score_after": self.game_state.previous_zork_score,
                            "game_over": self.game_state.game_over_flag,
                        }
                    )

                    return action_taken, next_game_state
            else:
                # No Langfuse - execute turn without tracing
                return self._execute_turn_logic(current_state)

        except Exception as e:
            self.logger.error(
                f"Turn failed with exception, using fallback 'look' action: {e}",
                extra={
                    "event_type": "turn_exception",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_action": "look",
                },
                exc_info=True,  # Include full traceback in logs
            )
            # Fallback action to maintain game continuity after exceptions
            return "look", current_state

    def _execute_critic_evaluation_loop(
        self,
        current_state: str,
        proposed_action: str,
        agent_context: Dict,
        formatted_context: str,
    ) -> Tuple[str, float, str, float, bool, Optional[str], List[Dict]]:
        """Execute critic evaluation and rejection loop with optional Langfuse tracing.

        Returns:
            Tuple of (action_to_take, final_critic_score, final_critic_justification,
                     final_critic_confidence, was_overridden, override_reason, rejected_actions_this_turn)
        """
        # Determine if we're tracing
        should_trace = self.langfuse_client is not None
        log_level = logging.DEBUG if should_trace else logging.INFO

        # Get critic context with ground-truth exits from Jericho (god-like view)
        critic_context = self.context_manager.get_critic_context(
            current_state=current_state,
            proposed_action=proposed_action,
            location=self.game_state.current_room_name_for_map,
            location_id=self.game_state.current_room_id,
            available_exits=self.jericho_interface.get_valid_exits(),  # Ground truth for validation
            failed_actions=self.game_state.failed_actions_by_location.get(
                self.game_state.current_room_name_for_map, []
            ),
        )

        # Setup tracing context if available
        if should_trace:
            rejection_threshold = self.rejection_manager.get_rejection_threshold()
            span_input = {
                "proposed_action": proposed_action,
                "current_location": self.game_state.current_room_name_for_map,
                "available_exits": critic_context.get("available_exits", []),
                "rejection_threshold": rejection_threshold,
            }
            span_metadata = {
                "turn_number": self.game_state.turn_count,
                "location_id": self.game_state.current_room_id,
                "location_name": self.game_state.current_room_name_for_map,
            }
            tracing_context = self.langfuse_client.start_as_current_span(
                name="critic-evaluation-loop",
                input=span_input,
                metadata=span_metadata,
            )
            attempt_details = []
        else:
            tracing_context = nullcontext()
            attempt_details = None

        with tracing_context as critic_span:
            # NEW: Detect action novelty and unexplored exits
            novelty_info = self._detect_action_novelty(proposed_action)
            unexplored_info = self._detect_unexplored_exits()

            # NEW: Build exploration hints for critic
            exploration_hints = self._build_exploration_hints(
                proposed_action,
                novelty_info,
                unexplored_info
            )

            # NEW: Add hints to critic context if present
            enhanced_critic_context = critic_context
            if exploration_hints:
                # Append hints to the current_state shown to critic
                enhanced_current_state = current_state + exploration_hints
            else:
                enhanced_current_state = current_state

            # Initial critic evaluation (with exploration hints influencing evaluation)
            critic_result = self.critic.evaluate_action(
                game_state_text=enhanced_current_state,
                proposed_action=proposed_action,
                available_exits=critic_context.get("available_exits", []),
                action_counts=self.game_state.action_counts,
                current_location_name=self.game_state.current_room_name_for_map,
                failed_actions_by_location=self.game_state.failed_actions_by_location,
                previous_actions_and_responses=self.game_state.action_history[-3:],
                jericho_interface=self.jericho_interface,
                inventory=self.game_state.current_inventory,
            )

            # Track initial attempt (if tracing)
            if attempt_details is not None:
                attempt_details.append({
                    "attempt_number": 1,
                    "action": proposed_action,
                    "score": critic_result.score,
                    "justification": critic_result.justification,
                    "confidence": critic_result.confidence,
                    "object_tree_validation_used": "[Object Tree Validation]" in critic_result.justification,
                })

            # NEW: Track location for revisit detection
            self._track_location_history()

            # NEW: Detect location revisit
            revisit_info = self._detect_location_revisit()

            # NEW: Apply programmatic penalty for location revisits
            base_confidence = critic_result.confidence
            adjusted_confidence, penalty_reason = self._apply_location_revisit_penalty(
                base_score=base_confidence,
                revisit_info=revisit_info
            )

            # NEW: Update critic result with adjusted confidence if penalty applied
            if penalty_reason:
                critic_result.confidence = adjusted_confidence

                self.logger.info(
                    f"Applied location penalty: {base_confidence:.2f} → {adjusted_confidence:.2f}",
                    extra={
                        "event_type": "location_penalty_applied",
                        "turn": self.game_state.turn_count,
                        "base_confidence": base_confidence,
                        "adjusted_confidence": adjusted_confidence,
                        "reason": penalty_reason,
                        "location_id": revisit_info["location_id"],
                        "recent_visits": revisit_info["recent_visits"],
                    }
                )

            # Start new turn for rejection tracking
            self.rejection_manager.start_new_turn()

            # Initialize tracking variables
            max_rejections = 3
            rejected_actions_this_turn = []
            action_to_take = proposed_action
            final_critic_score = critic_result.score
            final_critic_justification = critic_result.justification
            final_critic_confidence = critic_result.confidence
            was_overridden = False
            final_override_reason = None

            # Rejection loop
            for rejection_attempt in range(max_rejections):
                rejection_threshold = self.rejection_manager.get_rejection_threshold()

                # DEBUG: Log rejection loop entry
                self.logger.log(
                    log_level,
                    f"[REJECTION LOOP] Attempt {rejection_attempt + 1}/{max_rejections}: "
                    f"action='{action_to_take}', score={critic_result.score:.2f}, threshold={rejection_threshold}",
                    extra={
                        "event_type": "debug_rejection_loop_entry",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "rejection_attempt": rejection_attempt + 1,
                        "action": action_to_take,
                        "critic_score": critic_result.score,
                        "rejection_threshold": rejection_threshold,
                    },
                )

                if critic_result.score >= rejection_threshold:
                    self.logger.log(
                        log_level,
                        f"[REJECTION LOOP] Action accepted: score {critic_result.score:.2f} >= threshold {rejection_threshold}",
                        extra={
                            "event_type": "debug_rejection_loop_accepted",
                            "episode_id": self.game_state.episode_id,
                            "turn": self.game_state.turn_count,
                        },
                    )
                    break  # Action is acceptable

                # Check if we should override the rejection
                override_context = {
                    "recent_locations": [
                        getattr(entry, "current_location_name", "")
                        for entry in self.game_state.memory_log_history[-10:]
                        if hasattr(entry, "current_location_name")
                    ],
                    "recent_actions": [
                        action for action, _ in self.game_state.action_history[-8:]
                    ],
                    "previous_actions_and_responses": self.game_state.action_history[-8:],
                    "turns_since_movement": self.rejection_manager.state.turns_since_movement,
                    "critic_confidence": critic_result.confidence,
                }

                # DEBUG: Log before override check
                self.logger.log(
                    log_level,
                    f"[REJECTION LOOP] Checking override: turns_since_movement={override_context['turns_since_movement']}",
                    extra={
                        "event_type": "debug_before_override_check",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "override_context": override_context,
                    },
                )

                should_override, override_reason = (
                    self.rejection_manager.should_override_rejection(
                        action=action_to_take,
                        current_location=self.game_state.current_room_name_for_map,
                        failed_actions_by_location=self.game_state.failed_actions_by_location,
                        context=override_context,
                    )
                )

                # DEBUG: Log override decision
                self.logger.log(
                    log_level,
                    f"[REJECTION LOOP] Override decision: should_override={should_override}, reason='{override_reason}'",
                    extra={
                        "event_type": "debug_override_decision",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "should_override": should_override,
                        "override_reason": override_reason,
                    },
                )

                if should_override:
                    was_overridden = True
                    final_override_reason = override_reason
                    self.logger.info(
                        f"Overriding critic rejection: {override_reason}",
                        extra={
                            "event_type": "critic_override",
                            "episode_id": self.game_state.episode_id,
                            "reason": override_reason,
                            "turn": self.game_state.turn_count,
                            "original_action": action_to_take,
                            "original_score": critic_result.score,
                            "original_reasoning": critic_result.justification,
                        },
                    )
                    break

                # Action was rejected and not overridden
                rejected_actions_this_turn.append(
                    {
                        "action": action_to_take,
                        "score": critic_result.score,
                        "justification": critic_result.justification,
                    }
                )

                self.rejection_manager.add_rejected_action(
                    action_to_take, critic_result.score, critic_result.justification
                )

                # Log rejection
                self.logger.info(
                    f"Critic rejected action: {action_to_take} (score: {critic_result.score:.2f})",
                    extra={
                        "event_type": "action_rejected",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "action": action_to_take,
                        "score": critic_result.score,
                        "justification": critic_result.justification,
                        "rejection_attempt": rejection_attempt + 1,
                    },
                )

                # DEBUG: Log entering retry path
                self.logger.log(
                    log_level,
                    f"[REJECTION LOOP] Entering retry path: asking agent for new action",
                    extra={
                        "event_type": "debug_entering_retry",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "rejection_attempt": rejection_attempt + 1,
                        "rejected_action": action_to_take,
                    },
                )

                # Get new action from agent with rejection context
                rejected_actions_context = ", ".join(
                    self.rejection_manager.rejected_actions_this_turn
                )
                rejection_feedback = (
                    f"\n\n[Previous action(s) '{rejected_actions_context}' were rejected by critic: "
                    f"{critic_result.justification}]"
                )

                # Get new action with rejection context
                agent_result = self.agent.get_action_with_reasoning(
                    game_state_text=current_state + rejection_feedback,
                    relevant_memories=formatted_context,
                )

                action_to_take = agent_result["action"]
                agent_reasoning = agent_result.get("reasoning", "")

                # Track agent retry (if tracing)
                if attempt_details is not None:
                    attempt_details.append({
                        "attempt_number": rejection_attempt + 2,
                        "type": "agent_retry",
                        "new_action_generated": action_to_take,
                        "reasoning": agent_reasoning,
                    })

                # NEW: Re-detect novelty and unexplored exits for new action
                novelty_info = self._detect_action_novelty(action_to_take)
                unexplored_info = self._detect_unexplored_exits()

                # NEW: Build exploration hints for new action
                exploration_hints = self._build_exploration_hints(
                    action_to_take,
                    novelty_info,
                    unexplored_info
                )

                # NEW: Apply hints to critic context
                if exploration_hints:
                    enhanced_current_state = current_state + exploration_hints
                else:
                    enhanced_current_state = current_state

                # Re-evaluate new action (with exploration hints)
                critic_result = self.critic.evaluate_action(
                    game_state_text=enhanced_current_state,
                    proposed_action=action_to_take,
                    available_exits=critic_context.get("available_exits", []),
                    action_counts=self.game_state.action_counts,
                    current_location_name=self.game_state.current_room_name_for_map,
                    failed_actions_by_location=self.game_state.failed_actions_by_location,
                    previous_actions_and_responses=self.game_state.action_history[-3:],
                    jericho_interface=self.jericho_interface,
                    inventory=self.game_state.current_inventory,
                )

                # Track re-evaluation (if tracing)
                if attempt_details is not None:
                    attempt_details.append({
                        "attempt_number": rejection_attempt + 2,
                        "type": "critic_reevaluation",
                        "action": action_to_take,
                        "score": critic_result.score,
                        "justification": critic_result.justification,
                        "confidence": critic_result.confidence,
                        "object_tree_validation_used": "[Object Tree Validation]" in critic_result.justification,
                    })

                final_critic_score = critic_result.score
                final_critic_justification = critic_result.justification
                final_critic_confidence = critic_result.confidence

            # Check if we exhausted all rejection attempts
            exhausted_attempts = (
                rejection_attempt == max_rejections - 1
                and critic_result.score < rejection_threshold
                and not was_overridden
            )

            if exhausted_attempts:
                self.logger.warning(
                    f"Exhausted rejection attempts, proceeding with low-scoring action: "
                    f"{action_to_take} (score: {critic_result.score:.2f})",
                    extra={
                        "event_type": "rejection_attempts_exhausted",
                        "episode_id": self.game_state.episode_id,
                        "turn": self.game_state.turn_count,
                        "final_action": action_to_take,
                        "final_score": critic_result.score,
                        "threshold": rejection_threshold,
                    },
                )

            # Update span with comprehensive output (if tracing)
            if should_trace:
                # Determine final outcome
                if was_overridden:
                    outcome = "overridden"
                elif critic_result.score >= rejection_threshold:
                    outcome = "accepted"
                elif exhausted_attempts:
                    outcome = "exhausted_attempts"
                else:
                    outcome = "rejected"

                critic_span.update(
                    output={
                        "final_action": action_to_take,
                        "final_score": final_critic_score,
                        "final_confidence": final_critic_confidence,
                        "outcome": outcome,
                        "was_overridden": was_overridden,
                        "override_reason": final_override_reason if was_overridden else None,
                        "total_attempts": len(attempt_details),
                        "rejected_actions": rejected_actions_this_turn,
                    },
                    metadata={
                        "attempt_details": attempt_details,
                        "exhausted_attempts": exhausted_attempts,
                        "rejection_threshold": rejection_threshold,
                    },
                )

            return (
                action_to_take,
                final_critic_score,
                final_critic_justification,
                final_critic_confidence,
                was_overridden,
                final_override_reason,
                rejected_actions_this_turn,
            )

    def _sync_inventory_from_z_machine(self) -> None:
        """Sync game_state inventory with Z-machine reality every turn.

        This eliminates inventory desyncs by unconditionally updating game_state
        with authoritative Z-machine inventory data after every action execution.
        Fast operation (O(n) where n = inventory size, typically < 10 items).
        """
        try:
            structured_inv = self.jericho_interface.get_inventory_structured()
            self.game_state.current_inventory = [obj.name for obj in structured_inv]
        except RuntimeError as e:
            self.logger.error(
                f"Failed to sync inventory from Z-machine: {e}",
                extra={
                    "event_type": "inventory_sync_failure",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                }
            )
            # Don't update inventory if sync fails - keep previous state

    def _execute_turn_logic(self, current_state: str) -> Tuple[str, str]:
        """Execute the main turn logic (with or without Langfuse tracing)."""
        # Generate action using agent
        agent_context = self.context_manager.get_agent_context(
            current_state=current_state,
            inventory=self.game_state.current_inventory,
            location=self.game_state.current_room_name_for_map,
            location_id=self.game_state.current_room_id,
            game_map=self.map_manager.game_map,
            in_combat=self.state_manager.get_combat_status(),
            failed_actions=self.game_state.failed_actions_by_location.get(
                self.game_state.current_room_name_for_map, []
            ),
            discovered_objectives=self.game_state.discovered_objectives,
            jericho_interface=self.jericho_interface,  # NEW: Pass Jericho interface for structured data
        )

        # Format context for agent (including game response)
        formatted_context = self.context_manager.get_formatted_agent_prompt_context(
            agent_context,
            game_state_text=current_state
        )

        # CRITICAL: Add stuck countdown warning (highest priority)
        stuck_warning = self._build_stuck_countdown_warning()
        if stuck_warning:
            formatted_context = stuck_warning + "\n" + formatted_context

        # Get agent action (game_state_text no longer needed separately since it's in formatted_context)
        agent_result = self.agent.get_action_with_reasoning(
            game_state_text="",  # Empty since game response is now in formatted_context
            relevant_memories=formatted_context,
        )

        proposed_action = agent_result["action"]
        agent_reasoning = agent_result.get("reasoning", "")

        # Add reasoning to context
        self.context_manager.add_reasoning(agent_reasoning, proposed_action)

        # Extract and process agent-declared objective (if any)
        new_objective = agent_result.get("new_objective")
        if new_objective:
            self.objective_manager.add_agent_objective(new_objective)
            self.logger.info(f"Agent declared new objective: {new_objective}")

        # Execute critic evaluation with optional tracing
        action_to_take, final_critic_score, final_critic_justification, final_critic_confidence, was_overridden, override_reason, rejected_actions_this_turn = (
            self._execute_critic_evaluation_loop(
                current_state=current_state,
                proposed_action=proposed_action,
                agent_context=agent_context,
                formatted_context=formatted_context,
            )
        )

        # Store rejected actions for this turn
        if rejected_actions_this_turn:
            self.game_state.rejected_actions_per_turn[
                self.game_state.turn_count
            ] = rejected_actions_this_turn

        # Update confidence history
        self.critic_confidence_history.append(final_critic_confidence)

        # Store critic evaluation for viewer (state export)
        critic_eval_data = {
            "critic_score": final_critic_score,
            "critic_justification": final_critic_justification,
            "was_overridden": was_overridden,
            "override_reason": override_reason,  # Include override reason for viewer
            "rejected_actions": rejected_actions_this_turn,
        }
        self.game_state.critic_evaluation_history.append(critic_eval_data)

        # Update action counts
        self.game_state.action_counts[action_to_take] += 1

        # Log final action selection (for knowledge manager compatibility)
        self.logger.info(
            f"SELECTED ACTION: {action_to_take} (Score: {final_critic_score:.2f}, Confidence: {final_critic_confidence:.2f}, Override: {was_overridden})",
            extra={
                "event_type": "final_action_selection",
                "episode_id": self.game_state.episode_id,
                "turn": self.game_state.turn_count,
                "agent_action": action_to_take,
                "agent_reasoning": agent_reasoning,
                "critic_score": final_critic_score,
                "critic_confidence": final_critic_confidence,
                "was_overridden": was_overridden,
            },
        )

        # ================================================================
        # Phase 0: Capture state BEFORE action for source location storage
        # ================================================================
        # CRITICAL: Memories must be stored at SOURCE location (where action taken),
        # NOT destination location (where agent ends up after action).
        #
        # Rationale:
        # - Destination storage: "At Kitchen, I know window entry works" → Useless (already there)
        # - Source storage: "At Behind House, I know 'enter window' leads to Kitchen" → Useful for next visit
        #
        # This enables cross-episode learning: Episode 2 benefits from Episode 1 discoveries
        # when agent returns to same locations with prior knowledge.
        score_before, _ = self.jericho_interface.get_score()
        location_before = self.jericho_interface.get_location_structured()
        location_id_before = location_before.num if location_before else 0
        location_name_before = location_before.name if location_before else "Unknown"
        inventory_before = self.jericho_interface.get_inventory_structured()

        # Execute action using Jericho
        next_game_state = self.jericho_interface.send_command(action_to_take)

        # Sync inventory with Z-machine reality after every action
        self._sync_inventory_from_z_machine()

        # Track action after execution (for novelty detection)
        self._track_action_history(action_to_take)

        # Check for game over
        is_game_over, game_over_reason = self.jericho_interface.is_game_over(
            next_game_state
        )
        if is_game_over:
            self.game_state.game_over_flag = True
            self.logger.info(
                f"Game over detected: {game_over_reason}",
                extra={
                    "event_type": "game_over_detected",
                    "episode_id": self.game_state.episode_id,
                    "turn_number": self.game_state.turn_count,
                    "reason": game_over_reason,
                },
            )

        # Clean the game response before storing in history
        clean_response = self.extractor.get_clean_game_text(next_game_state)

        # Log zork response (for knowledge manager compatibility)
        self.logger.info(
            f"ZORK RESPONSE for '{action_to_take}':\n{clean_response}\n",
            extra={
                "event_type": "zork_response",
                "episode_id": self.game_state.episode_id,
                "turn": self.game_state.turn_count,
                "action": action_to_take,
                "zork_response": clean_response,
                "raw_zork_response": next_game_state,
            },
        )

        # Add action to history with source location (where action was taken)
        self.context_manager.add_action(
            action_to_take,
            clean_response,
            location_id_before,
            location_name_before
        )

        # Extract information from response
        extracted_info = self.extractor.extract_info(next_game_state)
        self._process_extraction(extracted_info, action_to_take, next_game_state)

        # Store room description if extractor flagged it
        if extracted_info.is_room_description:
            current_location = self.jericho_interface.get_location_structured()
            location_id = current_location.num if current_location else None

            self.game_state.last_room_description = clean_response
            self.game_state.last_room_description_turn = self.game_state.turn_count
            self.game_state.last_room_description_location_id = location_id

            self.logger.info(
                "Room description captured",
                extra={
                    "event_type": "room_description_captured",
                    "turn": self.game_state.turn_count,
                    "location_id": location_id,
                    "text_length": len(clean_response),
                }
            )

        # Capture state after action for memory system
        score_after, _ = self.jericho_interface.get_score()
        location_after = self.jericho_interface.get_location_structured()
        location_id_after = location_after.num if location_after else 0
        inventory_after = self.jericho_interface.get_inventory_structured()

        # Build Z-machine context for memory system
        z_machine_context = {
            'score_before': score_before,
            'score_after': score_after,
            'score_delta': score_after - score_before,
            'location_before': location_id_before,
            'location_after': location_id_after,
            'location_changed': location_id_before != location_id_after,
            'inventory_before': [obj.name for obj in inventory_before],
            'inventory_after': [obj.name for obj in inventory_after],
            'inventory_changed': set(o.name for o in inventory_before) != set(o.name for o in inventory_after),
            'died': self.game_state.game_over_flag,
            'response_length': len(clean_response),
            'first_visit': location_id_after not in self.simple_memory.memory_cache
        }

        # Record action outcome for memory synthesis
        # CRITICAL: Use location_id_before and location_name_before (SOURCE location)
        # NOT current location (destination). See Phase 0 comments above for rationale.
        self.simple_memory.record_action_outcome(
            location_id=location_id_before,      # SOURCE location (where action was taken)
            location_name=location_name_before,  # SOURCE location name
            action=action_to_take,
            response=clean_response,
            z_machine_context=z_machine_context
        )

        # Store extracted info for viewer (state export)
        extracted_dict = {}
        if hasattr(extracted_info, "__dict__"):
            extracted_dict = {
                k: v
                for k, v in extracted_info.__dict__.items()
                if not k.startswith("_")
            }
        elif isinstance(extracted_info, dict):
            extracted_dict = extracted_info
        self.game_state.extracted_info_history.append(extracted_dict)

        # Check for objective completion
        self.objective_manager.check_objective_completion(
            action_taken=action_to_take,
            game_response=next_game_state,
            extracted_info=extracted_info,
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
                "confidence": final_critic_confidence,
            },
        )

        # Track state for loop detection (Phase 6)
        loop_detected = self.state_manager.track_state_hash(self.jericho_interface)
        if loop_detected:
            self.logger.info(
                "State loop detected - agent may be stuck",
                extra={
                    "event_type": "stuck_behavior_detected",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                },
            )

        return action_to_take, next_game_state

    def _process_extraction(self, extracted_info, action: str, response: str) -> None:
        """Process extracted information and update game state."""
        # Add to memory
        self.context_manager.add_memory(extracted_info)

        # Update score if present
        if hasattr(extracted_info, "score") and extracted_info.score is not None:
            self.game_state.previous_zork_score = extracted_info.score

        # Note: Inventory sync now handled by _sync_inventory_from_z_machine()
        # which is called after every action execution. The Z-machine is the
        # authoritative source of truth for inventory state.
        # Object event detection can be reimplemented if needed using Z-machine data.

        # Update game over flag
        if hasattr(extracted_info, "game_over") and extracted_info.game_over:
            self.game_state.game_over_flag = True

        # Update location and map
        # CRITICAL: Always use Jericho's authoritative location data (both ID and name)
        # Do NOT use extractor's parsed location name - it's unreliable and causes map conflicts
        # when the extractor misreads room names from text responses
        try:
            location_obj = self.jericho_interface.get_location_structured()
            new_location_id = location_obj.num if location_obj else None
            new_location = location_obj.name if location_obj else None
        except Exception as e:
            self.logger.warning(f"Failed to get location from Jericho: {e}")
            new_location_id = None
            new_location = None

        # Only proceed with map updates if we have valid location data
        if new_location_id is not None and new_location is not None:
            # Add to visited locations
            self.game_state.visited_locations.add(new_location)

            # Update map
            if action and self.game_state.current_room_id:
                self.map_manager.update_from_movement(
                    action_taken=action,
                    new_room_id=new_location_id,
                    new_room_name=new_location,
                    previous_room_id=self.game_state.current_room_id,
                    previous_room_name=self.game_state.current_room_name_for_map,
                    game_response=response,
                )
            elif not self.game_state.current_room_id:
                # Initial room
                self.map_manager.add_initial_room(new_location_id, new_location)

            # Update rejection manager's movement tracking
            if self.game_state.current_room_name_for_map != new_location:
                self.rejection_manager.update_movement_tracking(moved=True)
            else:
                self.rejection_manager.update_movement_tracking(moved=False)

            # Update GameState with new location
            self.game_state.current_room_id = new_location_id
            self.game_state.current_room_name_for_map = new_location

        # Track failed actions
        if action and response:
            response_lower = response.lower()
            failure_indicators = [
                "you can't",
                "impossible",
                "don't understand",
                "nothing happens",
            ]

            if any(indicator in response_lower for indicator in failure_indicators):
                # Get current location ID for tracking
                try:
                    location_obj = self.jericho_interface.get_location_structured()
                    current_location_id = location_obj.num if location_obj else None
                    current_location_name = self.game_state.current_room_name_for_map

                    if current_location_id is not None:
                        self.map_manager.track_failed_action(
                            action, current_location_id, current_location_name
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to track failed action: {e}")

    def _check_periodic_updates(self) -> None:
        """Check and run periodic updates for managers."""
        # Map manager periodic check (currently no-op; map updates happen in real-time)
        self.map_manager.process_turn()

        # Objective updates (normal periodic check at interval)
        if self.objective_manager.should_process_turn():
            current_reasoning = ""
            if self.game_state.action_reasoning_history:
                current_reasoning = self.game_state.action_reasoning_history[-1].get(
                    "reasoning", ""
                )
            self.objective_manager.process_periodic_updates(current_reasoning)

        # Knowledge updates
        if self.knowledge_manager.should_process_turn():
            self.knowledge_manager.check_periodic_update()

        # State management (context overflow)
        self.state_manager.process_turn()

    def _generate_turn_1_objectives(self) -> None:
        """
        Generate initial objectives before turn 1 if memories exist from prior episodes.

        This ensures objectives are available to the agent when making its first decision.
        Only runs if persistent memories exist (cross-episode learning).
        """
        has_memories = self.simple_memory.get_persistent_count() > 0
        if has_memories:
            self.logger.info(
                "Pre-turn 1: Generating initial objectives from memories and knowledgebase",
                extra={
                    "event_type": "turn_1_objective_generation",
                    "episode_id": self.game_state.episode_id,
                    "memory_count": self.simple_memory.get_persistent_count(),
                }
            )
            # Use bootstrap method to bypass turn checks
            self.objective_manager.bootstrap_initial_objectives()
        else:
            self.logger.info(
                "Pre-turn 1: No persistent memories found, skipping initial objective generation",
                extra={
                    "event_type": "turn_1_objective_generation_skipped",
                    "episode_id": self.game_state.episode_id,
                    "reason": "no_memories",
                }
            )

    def _export_coordinated_state(self) -> None:
        """Coordinate data gathering from managers and export complete state."""
        try:
            # Gather data from specialized managers (orchestrator coordination)
            map_data = self.map_manager.get_export_data()
            knowledge_data = self.knowledge_manager.get_export_data()
            rejection_data = self.rejection_manager.get_state_for_export()

            # Store rejection state in GameState for persistence
            self.game_state.rejection_state = rejection_data

            # Pass to StateManager for assembly and export (delegation)
            self.state_manager.export_current_state(
                map_data=map_data, knowledge_data=knowledge_data
            )

        except Exception as e:
            self.logger.error(
                f"Failed to export coordinated state: {e}",
                extra={
                    "event_type": "state_export_error",
                    "episode_id": self.game_state.episode_id,
                    "turn": self.game_state.turn_count,
                    "error": str(e),
                },
            )

    def run_multiple_episodes(self, num_episodes: int = 1) -> List[int]:
        """
        Run multiple episodes sequentially.

        Args:
            num_episodes: Number of episodes to run

        Returns:
            List of final scores for each episode
        """
        scores = []

        for i in range(num_episodes):
            self.logger.info(f"Starting episode {i + 1} of {num_episodes}")

            # Reset managers for new episode
            for manager in self.managers:
                manager.reset_episode()

            # Clear critic confidence history
            self.critic_confidence_history = []

            # Play episode
            score = self.play_episode()
            scores.append(score)

            self.logger.info(f"Episode {i + 1} completed with score: {score}")

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
            "managers": {},
        }

        # Get status from each manager
        for manager in self.managers:
            manager_name = manager.__class__.__name__
            status["managers"][manager_name] = manager.get_status()

        return status
