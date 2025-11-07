"""
ABOUTME: Dual cache management for SimpleMemoryManager - persistent and ephemeral memory caches.
ABOUTME: Manages routing, retrieval, supersession, and invalidation across both cache types.
"""

from typing import Dict, List, Optional, Any

from managers.memory import Memory, MemoryStatus


class MemoryCacheManager:
    """
    Manages dual memory caches: persistent (cross-episode) and ephemeral (episode-only).

    The dual cache system supports three persistence levels:
    - CORE: Spawn state from first-visit room descriptions → persistent cache
    - PERMANENT: Game mechanics and reusable knowledge → persistent cache
    - EPHEMERAL: Agent-caused state changes → ephemeral cache

    Cache Structure:
    - memory_cache: Dict[int, List[Memory]] - CORE + PERMANENT (persists across episodes)
    - ephemeral_cache: Dict[int, List[Memory]] - EPHEMERAL only (cleared on episode reset)

    Key Operations:
    - add_to_cache(): Routes memories to appropriate cache based on persistence level
    - get_from_cache(): Retrieves memories with filtering (persistent, ephemeral, or both)
    - supersede_in_cache(): Marks memories as SUPERSEDED in both caches
    - invalidate_in_cache(): Marks memories as INVALIDATED in both caches
    - clear_ephemeral_cache(): Clears ephemeral cache for episode reset

    Thread Safety:
    This class does NOT handle locking. File-level locking is handled by SimpleMemoryManager
    for file write operations. Cache operations are single-threaded in the current architecture.
    """

    def __init__(self):
        """Initialize both caches as empty dictionaries."""
        # Persistent cache: CORE + PERMANENT memories (from Memories.md)
        self._memory_cache: Dict[int, List[Memory]] = {}

        # Ephemeral cache: EPHEMERAL memories only (in-memory, cleared on episode reset)
        self._ephemeral_memory_cache: Dict[int, List[Memory]] = {}

    def add_to_cache(self, location_id: int, memory: Memory) -> None:
        """
        Add memory to appropriate cache based on persistence_level.

        Routing logic:
        - EPHEMERAL → ephemeral_cache only
        - CORE or PERMANENT → memory_cache only

        Args:
            location_id: Integer location ID from Z-machine
            memory: Memory object to add

        Note:
            No file I/O here. File writes handled by SimpleMemoryManager.
        """
        if memory.persistence == "ephemeral":
            # Route to ephemeral cache
            if location_id not in self._ephemeral_memory_cache:
                self._ephemeral_memory_cache[location_id] = []
            self._ephemeral_memory_cache[location_id].append(memory)
        else:
            # Route to persistent cache (core or permanent)
            if location_id not in self._memory_cache:
                self._memory_cache[location_id] = []
            self._memory_cache[location_id].append(memory)

    def get_from_cache(
        self,
        location_id: int,
        persistent: Optional[bool] = None,
        include_superseded: bool = False
    ) -> List[Memory]:
        """
        Retrieve memories from cache(s) with filtering.

        Args:
            location_id: Location ID to retrieve memories for
            persistent: Filter by cache type:
                - None: Combine both caches (default)
                - True: Only persistent cache (CORE + PERMANENT)
                - False: Only ephemeral cache (EPHEMERAL)
            include_superseded: If True, include SUPERSEDED memories in results

        Returns:
            List of Memory objects matching criteria (may be empty)

        Example:
            # Get all active memories at location 15
            memories = cache_manager.get_from_cache(15)

            # Get only persistent memories
            persistent = cache_manager.get_from_cache(15, persistent=True)

            # Get ephemeral memories including superseded
            ephemeral = cache_manager.get_from_cache(15, persistent=False, include_superseded=True)
        """
        memories = []

        # Collect from persistent cache if requested
        if persistent is None or persistent is True:
            memories.extend(self._memory_cache.get(location_id, []))

        # Collect from ephemeral cache if requested
        if persistent is None or persistent is False:
            memories.extend(self._ephemeral_memory_cache.get(location_id, []))

        # Filter out SUPERSEDED memories unless explicitly requested
        if not include_superseded:
            memories = [m for m in memories if m.status != MemoryStatus.SUPERSEDED]

        return memories

    def supersede_in_cache(self, location_id: int, memory_title: str, superseded_by_title: str, turn: int) -> bool:
        """
        Mark memory as SUPERSEDED in both caches (cross-cache supersession).

        Searches both caches for memory matching title, marks as SUPERSEDED.
        This handles cases where we're not sure which cache contains the memory.

        Args:
            location_id: Location ID where memory exists
            memory_title: Title of memory to mark as SUPERSEDED
            superseded_by_title: Title of new memory that supersedes this one
            turn: Turn number when supersession occurred

        Returns:
            True if memory found and marked SUPERSEDED, False if not found

        Note:
            Only searches location_id. Memory must exist at that location.
            Uses substring matching (memory_title in mem.title OR mem.title in memory_title).
        """
        # Search persistent cache first
        if location_id in self._memory_cache:
            for mem in self._memory_cache[location_id]:
                if memory_title in mem.title or mem.title in memory_title:
                    mem.status = MemoryStatus.SUPERSEDED
                    mem.superseded_by = superseded_by_title
                    mem.superseded_at_turn = turn
                    return True

        # Search ephemeral cache
        if location_id in self._ephemeral_memory_cache:
            for mem in self._ephemeral_memory_cache[location_id]:
                if memory_title in mem.title or mem.title in memory_title:
                    mem.status = MemoryStatus.SUPERSEDED
                    mem.superseded_by = superseded_by_title
                    mem.superseded_at_turn = turn
                    return True

        # Not found in either cache
        return False

    def invalidate_in_cache(self, location_id: int, memory_title: str, reason: str, turn: int) -> bool:
        """
        Mark memory as INVALIDATED (standalone, no replacement) in both caches.

        Searches both caches for memory matching title, marks as SUPERSEDED with
        INVALIDATION_MARKER and invalidation_reason.

        Args:
            location_id: Location ID where memory exists
            memory_title: Title of memory to invalidate
            reason: Explanation for why memory is invalid
            turn: Turn number when invalidation occurred

        Returns:
            True if memory found and invalidated, False if not found

        Note:
            Sets superseded_by to INVALIDATION_MARKER (sentinel value).
            Uses substring matching (memory_title in mem.title OR mem.title in memory_title).
        """
        from managers.memory import INVALIDATION_MARKER

        # Search persistent cache first
        if location_id in self._memory_cache:
            for mem in self._memory_cache[location_id]:
                if memory_title in mem.title or mem.title in memory_title:
                    mem.status = MemoryStatus.SUPERSEDED
                    mem.superseded_by = INVALIDATION_MARKER
                    mem.superseded_at_turn = turn
                    mem.invalidation_reason = reason
                    return True

        # Search ephemeral cache
        if location_id in self._ephemeral_memory_cache:
            for mem in self._ephemeral_memory_cache[location_id]:
                if memory_title in mem.title or mem.title in memory_title:
                    mem.status = MemoryStatus.SUPERSEDED
                    mem.superseded_by = INVALIDATION_MARKER
                    mem.superseded_at_turn = turn
                    mem.invalidation_reason = reason
                    return True

        # Not found in either cache
        return False

    def clear_ephemeral_cache(self) -> int:
        """
        Clear ephemeral cache (episode reset).

        Returns:
            Number of ephemeral memories cleared

        Note:
            Persistent cache (memory_cache) is NOT cleared.
            This is called by SimpleMemoryManager.reset_episode().
        """
        ephemeral_count = sum(len(mems) for mems in self._ephemeral_memory_cache.values())
        self._ephemeral_memory_cache.clear()
        return ephemeral_count

    def get_ephemeral_count(self, location_id: Optional[int] = None) -> int:
        """
        Get count of ephemeral memories.

        Args:
            location_id: Specific location, or None for total across all locations

        Returns:
            Count of ephemeral memories
        """
        if location_id is not None:
            return len(self._ephemeral_memory_cache.get(location_id, []))
        else:
            return sum(len(mems) for mems in self._ephemeral_memory_cache.values())

    def get_persistent_count(self, location_id: Optional[int] = None) -> int:
        """
        Get count of persistent memories (CORE + PERMANENT).

        Args:
            location_id: Specific location, or None for total across all locations

        Returns:
            Count of persistent memories
        """
        if location_id is not None:
            return len(self._memory_cache.get(location_id, []))
        else:
            return sum(len(mems) for mems in self._memory_cache.values())

    def get_memory_breakdown(self, location_id: int) -> Dict[str, int]:
        """
        Get breakdown of memory types at location.

        Counts ACTIVE memories only (excludes SUPERSEDED).

        Args:
            location_id: Location to get breakdown for

        Returns:
            Dictionary with counts: {"core": count, "permanent": count, "ephemeral": count}
        """
        breakdown = {"core": 0, "permanent": 0, "ephemeral": 0}

        # Count from persistent cache (core + permanent)
        for mem in self._memory_cache.get(location_id, []):
            if mem.status != MemoryStatus.SUPERSEDED:
                breakdown[mem.persistence] += 1

        # Count from ephemeral cache
        for mem in self._ephemeral_memory_cache.get(location_id, []):
            if mem.status != MemoryStatus.SUPERSEDED:
                breakdown[mem.persistence] += 1

        return breakdown

    def get_locations_tracked(self) -> int:
        """
        Get count of distinct locations with memories in persistent cache.

        Returns:
            Number of locations in memory_cache

        Note:
            Only counts persistent cache locations. Ephemeral locations are excluded.
        """
        return len(self._memory_cache)

    def get_total_memories(self) -> int:
        """
        Get total count of memories in persistent cache.

        Returns:
            Total number of memories in memory_cache (CORE + PERMANENT)

        Note:
            Only counts persistent cache memories. Ephemeral memories are excluded.
        """
        return sum(len(memories) for memories in self._memory_cache.values())
