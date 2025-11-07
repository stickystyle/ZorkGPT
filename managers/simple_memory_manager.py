"""
ABOUTME: SimpleMemoryManager for ZorkGPT - location-based memory system with multi-step synthesis.
ABOUTME: Coordinates memory operations via specialized components for caching, formatting, triggers, file I/O, and LLM synthesis.

This module implements the Simple Memory System:
- Location-based memory management with dual cache architecture (persistent + ephemeral)
- Multi-step procedure detection using recent action and reasoning history
- Memory status lifecycle (ACTIVE, TENTATIVE, SUPERSEDED)
- Supersession and standalone invalidation workflows
- Graceful handling of missing, empty, or corrupted files
- Delegation to specialized components for all operations
"""

import re
from typing import Dict, List, Optional, Any, Tuple

from managers.base_manager import BaseManager
from managers.memory import (
    Memory,
    MemoryStatus,
    MemorySynthesisResponse,
    INVALIDATION_MARKER,
    SynthesisTrigger,
    MemoryCacheManager,
    MemoryFileParser,
    MemoryFileWriter,
    HistoryFormatter,
    MemorySynthesizer,
)
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from llm_client import LLMClientWrapper

# Re-export for backward compatibility (tests import from this module)
__all__ = [
    "SimpleMemoryManager",
    "Memory",
    "MemoryStatus",
    "MemorySynthesisResponse",
    "INVALIDATION_MARKER",
]


class SimpleMemoryManager(BaseManager):
    """
    Manages location-based memory system for ZorkGPT with multi-step synthesis.

    Responsibilities:
    - Parse Memories.md file format on initialization
    - Maintain in-memory cache of memories per location ID
    - Synthesize memories with multi-step procedure detection (prerequisites, delayed consequences)
    - Manage memory status lifecycle (ACTIVE, TENTATIVE, SUPERSEDED)
    - Handle supersession when new evidence contradicts old memories
    - Gracefully handle missing, empty, or corrupted files
    - Provide memory retrieval interface with status filtering

    Cache Structure:
    - memory_cache: Dict[int, List[Memory]] - location ID to list of memories

    Multi-Step Synthesis:
    - Retrieves recent action and reasoning history (configurable window, default: 3 turns)
    - LLM detects procedures spanning multiple turns (e.g., "open window → enter window")
    - Stores memories at SOURCE location (where action taken) for cross-episode learning
    - Supports TENTATIVE memories that can be superseded by later evidence
    """

    def __init__(self, logger, config: GameConfiguration, game_state: GameState, llm_client=None):
        super().__init__(logger, config, game_state, "simple_memory")

        # Cache manager handles dual cache structure
        self.cache_manager = MemoryCacheManager()

        # File operation handlers
        self.file_parser = MemoryFileParser(self.logger, self.config, self.cache_manager)
        self.file_writer = MemoryFileWriter(self.logger, self.config)

        # Store LLM client (lazy initialization)
        self._llm_client = llm_client
        self._llm_client_initialized = llm_client is not None

        # Synthesis components
        self.trigger_detector = SynthesisTrigger(config, logger)
        self.formatter = HistoryFormatter()
        self.synthesizer = MemorySynthesizer(
            logger=self.logger,
            config=self.config,
            formatter=self.formatter,
            llm_client=self._llm_client  # Note: may be None (lazy init)
        )

        # Parse Memories.md file on initialization
        self.file_parser.load_from_file()

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if not self._llm_client_initialized:
            self._llm_client = LLMClientWrapper(logger=self.logger, config=self.config)
            self._llm_client_initialized = True
            # Update synthesizer with the lazily initialized client
            self.synthesizer.llm_client = self._llm_client
        return self._llm_client

    @property
    def memory_cache(self) -> Dict[int, List[Memory]]:
        """
        Backward compatibility property for external code accessing persistent cache.

        Returns:
            Reference to cache_manager's persistent cache (CORE + PERMANENT memories)
        """
        return self.cache_manager._memory_cache

    @memory_cache.setter
    def memory_cache(self, value: Dict[int, List[Memory]]) -> None:
        """
        Backward compatibility setter for external code setting persistent cache.

        Args:
            value: Dictionary mapping location IDs to lists of Memory objects
        """
        self.cache_manager._memory_cache = value

    @property
    def ephemeral_cache(self) -> Dict[int, List[Memory]]:
        """
        Backward compatibility property for external code accessing ephemeral cache.

        Returns:
            Reference to cache_manager's ephemeral cache (EPHEMERAL memories only)
        """
        return self.cache_manager._ephemeral_memory_cache

    @ephemeral_cache.setter
    def ephemeral_cache(self, value: Dict[int, List[Memory]]) -> None:
        """
        Backward compatibility setter for external code setting ephemeral cache.

        Args:
            value: Dictionary mapping location IDs to lists of Memory objects
        """
        self.cache_manager._ephemeral_memory_cache = value

    def reset_episode(self) -> None:
        """
        Reset manager state for new episode.

        CRITICAL: Clears ephemeral_cache to prevent false memories.
        Persistent cache (memory_cache) remains unchanged.
        """
        # Clear ephemeral memories via cache manager
        ephemeral_count = self.cache_manager.clear_ephemeral_cache()

        self.log_info(
            f"Episode reset: Cleared {ephemeral_count} ephemeral memories",
            ephemeral_count=ephemeral_count
        )

        # Note: memory_cache (persistent) is NOT cleared

    def process_turn(self) -> None:
        """
        Process manager-specific logic for the current turn.

        Phase 1 is read-only parsing, so no per-turn processing needed.
        Future phases may add memory recording logic here.
        """
        pass

    def should_process_turn(self) -> bool:
        """
        Check if this manager needs to process the current turn.

        Phase 1 is read-only, so no turn processing needed.
        """
        return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get current manager status for debugging and monitoring.

        Returns:
            Dictionary with manager status information including cache metrics
        """
        status = super().get_status()

        # Add memory cache statistics via cache manager
        status.update({
            "locations_tracked": self.cache_manager.get_locations_tracked(),
            "total_memories": self.cache_manager.get_total_memories(),
            "cache_populated": self.cache_manager.get_locations_tracked() > 0,
        })

        return status

    def get_ephemeral_count(self, location_id: Optional[int] = None) -> int:
        """
        Get count of ephemeral memories.

        Args:
            location_id: Specific location, or None for total across all locations

        Returns:
            Count of ephemeral memories
        """
        return self.cache_manager.get_ephemeral_count(location_id)

    def get_persistent_count(self, location_id: Optional[int] = None) -> int:
        """
        Get count of persistent memories (CORE + PERMANENT).

        Args:
            location_id: Specific location, or None for total across all locations

        Returns:
            Count of persistent memories
        """
        return self.cache_manager.get_persistent_count(location_id)

    def get_memory_breakdown(self, location_id: int) -> Dict[str, int]:
        """
        Get breakdown of memory types at location.

        Args:
            location_id: Location to get breakdown for

        Returns:
            {"core": count, "permanent": count, "ephemeral": count}
        """
        return self.cache_manager.get_memory_breakdown(location_id)


    def add_memory(
        self,
        location_id: int,
        location_name: str,
        memory: Memory
    ) -> bool:
        """
        Add a memory to file and/or cache based on persistence level.

        Routing logic:
        - Ephemeral memories: Added to ephemeral_cache only (NOT written to file)
        - Core/Permanent memories: Written to file AND added to memory_cache

        This method:
        1. Routes based on memory.persistence value
        2. Ephemeral: In-memory only (ephemeral_cache)
        3. Core/Permanent: File write with lock, backup, and atomic write (via file_writer)

        Args:
            location_id: Integer location ID from Z-machine
            location_name: Location name for display
            memory: Memory object to add

        Returns:
            True if successful, False if operation failed
        """
        # Route based on persistence level
        if memory.persistence == "ephemeral":
            # Ephemeral memories: in-memory only (NOT written to file)
            self.cache_manager.add_to_cache(location_id, memory)

            self.log_info(
                f"Added ephemeral memory [{memory.category}] {memory.title} to location {location_id} (in-memory only)",
                location_id=location_id,
                location_name=location_name,
                category=memory.category,
                title=memory.title,
                persistence=memory.persistence
            )

            return True

        # Core/Permanent memories: delegate to file_writer
        success = self.file_writer.write_memory(memory, location_id, location_name)

        if success:
            # Update cache after successful write via cache_manager
            self.cache_manager.add_to_cache(location_id, memory)

        return success

    def supersede_memory(
        self,
        location_id: int,
        location_name: str,
        old_memory_title: str,
        new_memory: Memory
    ) -> bool:
        """
        Supersede an existing memory with a new one, handling cache migration.

        This method handles three cases:
        1. Permanent → Permanent: Both in memory_cache
        2. Ephemeral → Ephemeral: Both in ephemeral_cache, no file write
        3. Ephemeral → Permanent: Migrate from ephemeral_cache to memory_cache + file

        Args:
            location_id: Location ID where memory exists
            location_name: Location name for logging
            old_memory_title: Title of memory to supersede
            new_memory: New memory to add

        Returns:
            True if supersession succeeded, False if old memory not found
        """
        # Search both caches for old memory using cache_manager
        persistent_memories = self.cache_manager.get_from_cache(location_id, persistent=True, include_superseded=True)
        ephemeral_memories = self.cache_manager.get_from_cache(location_id, persistent=False, include_superseded=True)

        old_memory = None
        old_in_persistent = False

        # Check persistent cache first
        for mem in persistent_memories:
            if mem.title == old_memory_title:
                old_memory = mem
                old_in_persistent = True
                break

        # Check ephemeral cache if not found
        if not old_memory:
            for mem in ephemeral_memories:
                if mem.title == old_memory_title:
                    old_memory = mem
                    old_in_persistent = False
                    break

        # Return False if not found in either cache
        if not old_memory:
            self.log_warning(
                f"Memory '{old_memory_title}' not found at location {location_id}",
                location_id=location_id,
                location_name=location_name,
                memory_title=old_memory_title
            )
            return False

        # Validate persistence level compatibility (prevent downgrade to ephemeral)
        if old_memory.persistence in ["core", "permanent"] and new_memory.persistence == "ephemeral":
            self.log_warning(
                f"Cannot downgrade {old_memory.persistence} memory to ephemeral - would cause data loss after episode reset",
                location_id=location_id,
                old_title=old_memory_title,
                old_persistence=old_memory.persistence,
                new_persistence=new_memory.persistence,
                reason=f"{old_memory.persistence.capitalize()} knowledge cannot be replaced by temporary state"
            )
            return False

        # Extract turn number from new_memory.turns for the superseded_at_turn parameter
        # new_memory.turns is a string like "20" or "20-25"
        try:
            turn_parts = new_memory.turns.split('-')
            superseded_at_turn = int(turn_parts[-1])  # Use the last turn number
        except (ValueError, IndexError):
            superseded_at_turn = self.game_state.turn_count

        # Mark old memory as SUPERSEDED (delegate to _update_memory_status)
        if old_in_persistent:
            # Update the file to mark old memory as SUPERSEDED
            self._update_memory_status(
                location_id=location_id,
                memory_title=old_memory_title,
                new_status=MemoryStatus.SUPERSEDED,
                superseded_by=new_memory.title,
                superseded_at_turn=superseded_at_turn,
                invalidation_reason=None
            )
        else:
            # Just update the in-memory cache
            old_memory.status = MemoryStatus.SUPERSEDED
            old_memory.superseded_by = new_memory.title
            old_memory.superseded_at_turn = superseded_at_turn

        # Add new memory (routing handled by add_memory)
        success = self.add_memory(location_id, location_name, new_memory)

        if success:
            self.log_info(
                f"Superseded memory: '{old_memory_title}' → '{new_memory.title}' "
                f"({old_memory.persistence} → {new_memory.persistence})",
                location_id=location_id,
                location_name=location_name,
                old_title=old_memory_title,
                new_title=new_memory.title,
                old_persistence=old_memory.persistence,
                new_persistence=new_memory.persistence
            )

        return success

    def invalidate_memory(
        self,
        location_id: int,
        memory_title: str,
        reason: str,
        turn: Optional[int] = None
    ) -> bool:
        """
        Invalidate a single memory without creating a replacement.

        This method marks a memory as SUPERSEDED with INVALIDATION_MARKER,
        indicating it was proven false without a specific replacement memory.

        Args:
            location_id: Integer location ID from Z-machine
            memory_title: Title of memory to invalidate (exact or substring match)
            reason: Explanation for why memory is invalid (e.g., "Proven false by death")
            turn: Optional turn number when invalidated (defaults to current turn)

        Returns:
            True if successful, False if operation failed
        """
        # Default to current turn if not provided
        if turn is None:
            turn = self.game_state.turn_count

        # Validate inputs
        if not reason or not reason.strip():
            self.log_error(
                "Invalidation reason cannot be empty",
                location_id=location_id,
                memory_title=memory_title
            )
            return False

        # Delegate to file_writer
        success = self.file_writer.update_memory_status(
            location_id=location_id,
            memory_title=memory_title,
            new_status=MemoryStatus.SUPERSEDED,
            superseded_by=None,
            superseded_at_turn=turn,
            invalidation_reason=reason
        )

        if success:
            # Update cache via cache_manager
            self.cache_manager.invalidate_in_cache(
                location_id=location_id,
                memory_title=memory_title,
                reason=reason,
                turn=turn
            )

            self.log_info(
                f"Invalidated memory: '{memory_title}' at location {location_id}",
                location_id=location_id,
                memory_title=memory_title,
                reason=reason,
                turn=turn
            )
        else:
            self.log_warning(
                f"Failed to invalidate memory: '{memory_title}'",
                location_id=location_id,
                memory_title=memory_title
            )

        return success

    def invalidate_memories(
        self,
        location_id: int,
        memory_titles: List[str],
        reason: str,
        turn: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Invalidate multiple memories at once (batch operation).

        All memories are invalidated with the same reason. Useful for invalidating
        related memories when a core assumption is proven false.

        Args:
            location_id: Integer location ID from Z-machine
            memory_titles: List of memory titles to invalidate
            reason: Shared explanation for invalidation
            turn: Optional turn number when invalidated (defaults to current turn)

        Returns:
            Dictionary mapping memory titles to success/failure (True/False)

        Example:
            >>> results = manager.invalidate_memories(
            ...     location_id=152,
            ...     memory_titles=["Troll is friendly", "Troll accepts gifts"],
            ...     reason="Both proven false by troll attack",
            ...     turn=25
            ... )
            >>> results
            {'Troll is friendly': True, 'Troll accepts gifts': True}
        """
        # Default to current turn if not provided
        if turn is None:
            turn = self.game_state.turn_count

        # Validate inputs
        if not memory_titles:
            self.log_warning(
                "No memory titles provided for batch invalidation",
                location_id=location_id
            )
            return {}

        if not reason or not reason.strip():
            self.log_error(
                "Invalidation reason cannot be empty",
                location_id=location_id,
                num_memories=len(memory_titles)
            )
            return {title: False for title in memory_titles}

        # Process each memory
        results = {}
        for memory_title in memory_titles:
            success = self.invalidate_memory(
                location_id=location_id,
                memory_title=memory_title,
                reason=reason,
                turn=turn
            )
            results[memory_title] = success

        # Log summary
        successful = sum(1 for v in results.values() if v)
        self.log_info(
            f"Batch invalidation: {successful}/{len(memory_titles)} succeeded",
            location_id=location_id,
            successful=successful,
            total=len(memory_titles),
            reason=reason
        )

        return results

    def _update_memory_status(
        self,
        location_id: int,
        memory_title: str,
        new_status: str,
        superseded_by: Optional[str] = None,
        superseded_at_turn: Optional[int] = None,
        invalidation_reason: Optional[str] = None
    ) -> bool:
        """
        Update the status of an existing memory in file and cache.

        Delegates to file_writer.update_memory_status() for file operations.

        Args:
            location_id: Location where memory exists
            memory_title: Title of memory to update (exact match or substring)
            new_status: New status (typically MemoryStatus.SUPERSEDED)
            superseded_by: Optional title of new memory that superseded this one
            superseded_at_turn: Turn number when superseded/invalidated
            invalidation_reason: Optional reason for standalone invalidation

        Returns:
            True if successful, False if memory not found or update failed
        """
        # Delegate to file_writer
        success = self.file_writer.update_memory_status(
            location_id=location_id,
            memory_title=memory_title,
            new_status=new_status,
            superseded_by=superseded_by,
            superseded_at_turn=superseded_at_turn,
            invalidation_reason=invalidation_reason
        )

        if success:
            # Update cache via cache_manager
            if invalidation_reason:
                # Standalone invalidation
                self.cache_manager.invalidate_in_cache(
                    location_id=location_id,
                    memory_title=memory_title,
                    reason=invalidation_reason,
                    turn=superseded_at_turn
                )
            elif superseded_by:
                # Traditional supersession
                self.cache_manager.supersede_in_cache(
                    location_id=location_id,
                    memory_title=memory_title,
                    superseded_by_title=superseded_by,
                    turn=superseded_at_turn
                )

        return success

    def _should_synthesize_memory(self, z_machine_context: Dict) -> bool:
        """
        Backward compatibility wrapper for trigger detection.

        This method was moved to managers/memory/triggers.py as SynthesisTrigger.should_synthesize().
        This wrapper delegates to the trigger detector for backward compatibility.

        Args:
            z_machine_context: Dictionary with Z-machine state changes

        Returns:
            True if synthesis should be triggered, False otherwise
        """
        return self.trigger_detector.should_synthesize(z_machine_context)

    def _synthesize_memory(
        self,
        location_id: int,
        location_name: str,
        action: str,
        response: str,
        z_machine_context: Dict
    ) -> Optional[MemorySynthesisResponse]:
        """
        Synthesize memory using MemorySynthesizer.

        This method prepares context and delegates to the synthesizer component.
        Langfuse instrumentation handled by caller (record_action_outcome).

        Args:
            location_id: Current location integer ID
            location_name: Location display name
            action: Action taken by agent
            response: Game response text
            z_machine_context: Ground truth state changes

        Returns:
            MemorySynthesisResponse if LLM says to remember, None otherwise
        """
        try:
            # Ensure LLM client is initialized (triggers lazy init via property)
            _ = self.llm_client

            # Get existing memories from cache for deduplication
            existing_memories = self.memory_cache.get(location_id, [])

            # Get configurable history window (default: 3 turns, validated >= 1)
            window_size = self.config.get_memory_history_window()

            # Retrieve recent actions and reasoning from shared game state
            recent_actions = self.game_state.action_history[-window_size:] if self.game_state.action_history else []
            current_turn = self.game_state.turn_count
            recent_reasoning = self.game_state.action_reasoning_history[-window_size:] if self.game_state.action_reasoning_history else []

            # Format using dedicated helpers (via formatter component)
            start_turn = max(1, current_turn - len(recent_actions) + 1)
            actions_formatted = self.formatter.format_recent_actions(recent_actions, start_turn)
            reasoning_formatted = self.formatter.format_recent_reasoning(
                recent_reasoning,
                self.game_state.action_history
            )

            # Delegate to synthesizer
            return self.synthesizer.synthesize_memory(
                location_id=location_id,
                location_name=location_name,
                action=action,
                response=response,
                existing_memories=existing_memories,
                z_machine_context=z_machine_context,
                actions_formatted=actions_formatted,
                reasoning_formatted=reasoning_formatted
            )

        except Exception as e:
            self.log_error(
                f"Failed to synthesize memory: {e}",
                location_id=location_id,
                error=str(e)
            )
            return None

    def record_action_outcome(
        self,
        location_id: int,
        location_name: str,
        action: str,
        response: str,
        z_machine_context: Dict
    ) -> None:
        """
        Record action outcome, invoking LLM synthesis if significant.

        This is the main entry point called by orchestrator after each action.

        Args:
            location_id: Current location integer ID
            location_name: Location display name
            action: Action taken by agent
            response: Game response text
            z_machine_context: Ground truth state changes
        """
        # Check if we should synthesize memory
        if not self.trigger_detector.should_synthesize(z_machine_context):
            self.log_debug("No trigger fired, skipping synthesis")
            return

        # Call LLM synthesis
        synthesis = self._synthesize_memory(
            location_id=location_id,
            location_name=location_name,
            action=action,
            response=response,
            z_machine_context=z_machine_context
        )

        if not synthesis:
            self.log_debug("LLM decided not to remember")
            return

        # Process supersessions BEFORE adding new memory
        if synthesis.supersedes_memory_titles:
            self.log_info(
                f"Superseding {len(synthesis.supersedes_memory_titles)} memories",
                location_id=location_id,
                titles=list(synthesis.supersedes_memory_titles)
            )

            for old_memory_title in synthesis.supersedes_memory_titles:
                success = self._update_memory_status(
                    location_id=location_id,
                    memory_title=old_memory_title,
                    new_status=MemoryStatus.SUPERSEDED,
                    superseded_by=synthesis.memory_title,
                    superseded_at_turn=self.game_state.turn_count,
                    invalidation_reason=None
                )

                if not success:
                    self.log_warning(
                        f"Failed to supersede memory: '{old_memory_title}'",
                        location_id=location_id,
                        old_title=old_memory_title,
                        new_title=synthesis.memory_title
                    )

        # Process standalone invalidations (no new memory created)
        if synthesis.invalidate_memory_titles:
            self.log_info(
                f"Invalidating {len(synthesis.invalidate_memory_titles)} memories",
                location_id=location_id,
                titles=list(synthesis.invalidate_memory_titles),
                reason=synthesis.invalidation_reason
            )

            for memory_title in synthesis.invalidate_memory_titles:
                success = self.invalidate_memory(
                    location_id=location_id,
                    memory_title=memory_title,
                    reason=synthesis.invalidation_reason,
                    turn=self.game_state.turn_count
                )

                if not success:
                    self.log_warning(
                        f"Failed to invalidate memory: '{memory_title}'",
                        location_id=location_id,
                        memory_title=memory_title
                    )

        # If LLM decided to create new memory, add it
        if not synthesis.should_remember:
            # No new memory to create, but may have invalidated existing ones
            self.log_debug(
                "LLM decided not to create new memory",
                location_id=location_id,
                invalidated_count=len(synthesis.invalidate_memory_titles) if synthesis.invalidate_memory_titles else 0
            )
            return

        # Extract episode number from game_state.episode_id
        # episode_id format: "ep_001" -> extract 1
        episode = 1  # Default
        if self.game_state.episode_id:
            try:
                # Handle formats like "ep_001", "episode_1", or just "1"
                episode_str = str(self.game_state.episode_id)
                # Extract digits from the string
                import re
                digits = re.findall(r'\d+', episode_str)
                if digits:
                    episode = int(digits[0])
            except (ValueError, AttributeError):
                episode = 1

        # Convert synthesis to Memory dataclass
        memory = Memory(
            category=synthesis.category,
            title=synthesis.memory_title,
            episode=episode,
            turns=str(self.game_state.turn_count),
            score_change=z_machine_context.get('score_delta'),
            text=synthesis.memory_text,
            persistence=synthesis.persistence,  # Use persistence from synthesis
            status=synthesis.status  # Include status from synthesis response
        )

        # Write to file and update cache
        success = self.add_memory(location_id, location_name, memory)

        if success:
            self.log_info(
                "Memory stored",
                location_id=location_id,
                location_name=location_name,
                category=synthesis.category,
                title=synthesis.memory_title
            )
        else:
            self.log_warning(
                "Failed to store memory",
                location_id=location_id,
                location_name=location_name
            )

    def get_location_memory(self, location_id: int) -> str:
        """
        Retrieve formatted memory text for a location.

        Combines memories from BOTH memory_cache (persistent) and ephemeral_cache (episode-only).

        Filters memories by status:
        - ACTIVE: Shown normally
        - TENTATIVE: Shown with warning marker
        - SUPERSEDED: Hidden (proven wrong by later evidence)

        Args:
            location_id: Location ID to retrieve memories for

        Returns:
            Formatted string with filtered memories
        """
        # Combine memories from both caches via cache_manager
        # get_from_cache() with persistent=None combines both caches and excludes SUPERSEDED
        all_memories = self.cache_manager.get_from_cache(location_id, persistent=None, include_superseded=False)

        if not all_memories:
            return ""

        # Separate memories by status
        active_memories = [m for m in all_memories if m.status == MemoryStatus.ACTIVE]
        tentative_memories = [m for m in all_memories if m.status == MemoryStatus.TENTATIVE]
        # SUPERSEDED memories are not shown to agent (proven wrong)

        # Format output
        lines = []

        # Show ACTIVE memories normally
        if active_memories:
            for mem in active_memories:
                lines.append(f"[{mem.category}] {mem.title}: {mem.text}")

        # Show TENTATIVE memories with warning
        if tentative_memories:
            if active_memories:
                lines.append("")  # Blank line separator
            lines.append("⚠️  TENTATIVE MEMORIES (unconfirmed, may be invalidated):")
            for mem in tentative_memories:
                lines.append(f"  [{mem.category}] {mem.title}: {mem.text}")

        return "\n".join(lines)
