"""
ABOUTME: SimpleMemoryManager for ZorkGPT - location-based memory system (Phase 1: Parsing).
ABOUTME: Parses Memories.md format and maintains in-memory cache of location memories.

This module implements Phase 1 of the Simple Memory System:
- Parsing Memories.md file format with location headers and memory entries
- In-memory cache using Dict[int, List[Memory]] structure
- Graceful handling of missing, empty, or corrupted files
- Memory metadata parsing (episode, turns, score changes)
- Memory category extraction (SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE)
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Set, Tuple

from filelock import FileLock
from pydantic import BaseModel, Field, model_validator

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from shared_utils import create_json_schema, strip_markdown_json_fences, extract_json_from_text
from llm_client import LLMClientWrapper


# Type alias for valid status values
MemoryStatusType = Literal["ACTIVE", "TENTATIVE", "SUPERSEDED"]

class MemoryStatus:
    """
    Memory status constants.

    ACTIVE: Confirmed reliable memory (default)
    TENTATIVE: Appears true but may be invalidated by future evidence
    SUPERSEDED: Proven wrong and replaced by newer memory
    """
    ACTIVE: MemoryStatusType = "ACTIVE"
    TENTATIVE: MemoryStatusType = "TENTATIVE"
    SUPERSEDED: MemoryStatusType = "SUPERSEDED"


@dataclass
class Memory:
    """Represents a single location memory entry."""
    category: str              # SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE
    title: str                 # Short title of the memory
    episode: int               # Episode number
    turns: str                 # Turn range (e.g., "23-24" or "23")
    score_change: Optional[int]  # Score change (+5, +0, None if not specified)
    text: str                  # 1-2 sentence synthesized insight
    status: MemoryStatusType = MemoryStatus.ACTIVE  # Memory status
    superseded_by: Optional[str] = None  # Title of memory that superseded this
    superseded_at_turn: Optional[int] = None  # Turn when superseded


class MemorySynthesisResponse(BaseModel):
    """LLM response for memory synthesis."""
    model_config = {"strict": True}

    should_remember: bool
    category: Optional[str] = None  # Only required if should_remember=true
    memory_title: Optional[str] = None  # Only required if should_remember=true
    memory_text: Optional[str] = None  # Only required if should_remember=true
    status: MemoryStatusType = Field(default=MemoryStatus.ACTIVE)  # Default to ACTIVE
    supersedes_memory_titles: Set[str] = Field(default_factory=set)  # Titles to mark as superseded
    reasoning: str = ""  # Optional reasoning for debugging

    @model_validator(mode='after')
    def validate_remember_fields(self) -> 'MemorySynthesisResponse':
        """Ensure required fields are present when should_remember=true."""
        if self.should_remember:
            if not self.category:
                raise ValueError("category is required when should_remember=true")
            if not self.memory_title:
                raise ValueError("memory_title is required when should_remember=true")
            if not self.memory_text:
                raise ValueError("memory_text is required when should_remember=true")
        return self


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

    # Regex patterns for parsing Memories.md format
    LOCATION_HEADER_PATTERN = re.compile(r"^## Location (\d+): (.+)$")
    # Matches: **[CATEGORY] Title** or **[CATEGORY - STATUS] Title**
    MEMORY_ENTRY_PATTERN = re.compile(r"^\*\*\[(\w+)(?: - (\w+))?\] (.+?)\*\* \*\((.*?)\)\*$")

    def __init__(self, logger, config: GameConfiguration, game_state: GameState, llm_client=None):
        super().__init__(logger, config, game_state, "simple_memory")

        # Initialize in-memory cache
        self.memory_cache: Dict[int, List[Memory]] = {}

        # Store LLM client (lazy initialization)
        self._llm_client = llm_client
        self._llm_client_initialized = llm_client is not None

        # Parse Memories.md file on initialization
        self._load_memories_from_file()

    @property
    def llm_client(self):
        """Lazy initialization of LLM client."""
        if not self._llm_client_initialized:
            self._llm_client = LLMClientWrapper(logger=self.logger, config=self.config)
            self._llm_client_initialized = True
        return self._llm_client

    def reset_episode(self) -> None:
        """
        Reset manager state for a new episode.

        Note: Memory cache persists across episodes - only episode-specific
        tracking would be reset here. Currently no episode-specific state.
        """
        self.log_debug("Simple memory manager reset for new episode")

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

        # Add memory cache statistics
        total_memories = sum(len(memories) for memories in self.memory_cache.values())

        status.update({
            "locations_tracked": len(self.memory_cache),
            "total_memories": total_memories,
            "cache_populated": len(self.memory_cache) > 0,
        })

        return status

    def _load_memories_from_file(self) -> None:
        """
        Load and parse Memories.md file into memory cache.

        Handles missing, empty, or corrupted files gracefully with appropriate
        logging. Parsing continues after encountering corrupted sections.
        """
        memories_path = Path(self.config.zork_game_workdir) / "Memories.md"

        # Handle missing file gracefully
        if not memories_path.exists():
            self.log_warning(f"Memories.md not found at {memories_path}")
            return

        try:
            content = memories_path.read_text(encoding="utf-8")
            self._parse_memories_content(content)

            total_memories = sum(len(memories) for memories in self.memory_cache.values())
            self.log_info(
                f"Loaded {total_memories} memories from {len(self.memory_cache)} locations",
                locations=len(self.memory_cache),
                total_memories=total_memories,
            )

        except Exception as e:
            self.log_error(f"Failed to load Memories.md: {e}", error=str(e))

    def _parse_memories_content(self, content: str) -> None:
        """
        Parse Memories.md content into memory cache.

        Format:
        ## Location 15: West of House
        **Visits:** 3 | **Episodes:** 1, 2, 3

        ### Memories

        **[SUCCESS] Title** *(Ep1, T23-24, +0)*
        Memory text content here.

        Args:
            content: Full text content of Memories.md file
        """
        lines = content.split("\n")
        current_location_id: Optional[int] = None
        current_memory_header: Optional[tuple] = None  # (category, title, metadata)
        current_memory_text_lines: List[str] = []

        for line in lines:
            line = line.rstrip()

            # Check for location header (valid or malformed)
            if line.startswith("## Location"):
                location_match = self.LOCATION_HEADER_PATTERN.match(line)

                # Save any pending memory from previous location
                if current_memory_header and current_location_id is not None:
                    self._add_memory_to_cache(
                        current_location_id,
                        current_memory_header,
                        current_memory_text_lines
                    )
                    current_memory_header = None
                    current_memory_text_lines = []

                if location_match:
                    # Valid location header - parse it
                    try:
                        location_id = int(location_match.group(1))
                        location_name = location_match.group(2).strip()
                        current_location_id = location_id

                        # Initialize location in cache if not present
                        if location_id not in self.memory_cache:
                            self.memory_cache[location_id] = []

                        self.log_debug(
                            f"Parsing location {location_id}: {location_name}",
                            location_id=location_id,
                            location_name=location_name
                        )

                    except (ValueError, IndexError) as e:
                        self.log_warning(
                            f"Error parsing location header: {line}",
                            error=str(e)
                        )
                        current_location_id = None
                else:
                    # Malformed location header (e.g., "## Location Invalid: Not a Number")
                    self.log_warning(
                        f"Skipping malformed location header: {line}",
                        line=line
                    )
                    current_location_id = None
                    current_memory_header = None
                    current_memory_text_lines = []

                continue

            # Check for memory entry header
            memory_match = self.MEMORY_ENTRY_PATTERN.match(line)
            if memory_match:
                # Save any pending memory from previous entry
                if current_memory_header and current_location_id is not None:
                    self._add_memory_to_cache(
                        current_location_id,
                        current_memory_header,
                        current_memory_text_lines
                    )

                # Only parse new memory header if we have a valid location
                if current_location_id is not None:
                    # Parse new memory header
                    category = memory_match.group(1)

                    # Validate status value
                    status_str = memory_match.group(2)
                    if status_str and status_str not in [MemoryStatus.ACTIVE, MemoryStatus.TENTATIVE, MemoryStatus.SUPERSEDED]:
                        self.log_warning(
                            f"Invalid memory status '{status_str}', defaulting to ACTIVE",
                            line=line,
                            status=status_str
                        )
                        status = MemoryStatus.ACTIVE
                    else:
                        status = status_str or MemoryStatus.ACTIVE

                    title = memory_match.group(3).strip()
                    metadata = memory_match.group(4).strip()

                    # Store status along with other header info
                    current_memory_header = (category, status, title, metadata)
                    current_memory_text_lines = []
                else:
                    # Skip memory entries when not in a valid location
                    self.log_warning(
                        f"Skipping memory entry (no valid location): {line}",
                        line=line
                    )
                    current_memory_header = None
                    current_memory_text_lines = []

                continue

            # Check if this looks like a malformed memory header (bold text + metadata pattern)
            # Format: **TEXT** *(metadata)*
            # This catches malformed entries that don't match the strict pattern
            if line.strip().startswith("**") and "**" in line[2:] and "*(" in line and ")*" in line:
                # Save any pending memory first
                if current_memory_header and current_location_id is not None:
                    self._add_memory_to_cache(
                        current_location_id,
                        current_memory_header,
                        current_memory_text_lines
                    )
                    current_memory_header = None
                    current_memory_text_lines = []

                # Log and skip malformed entry
                self.log_warning(
                    f"Skipping malformed memory entry: {line}",
                    line=line
                )
                continue

            # Accumulate memory text lines
            if current_memory_header and line.strip():
                # Skip separator lines
                if line.strip() in ["---", "###", "### Memories"]:
                    continue
                # Skip metadata lines (Visits, Episodes)
                if line.startswith("**Visits:**") or line.startswith("**Episodes:**"):
                    continue
                # Skip headers
                if line.startswith("#"):
                    continue
                # Skip supersession reference lines (strict validation)
                if line.strip().startswith("[Superseded at") and 'by "' in line and line.strip().endswith('"]'):
                    continue  # Don't include in memory text

                current_memory_text_lines.append(line.strip())

        # Save final pending memory
        if current_memory_header and current_location_id is not None:
            self._add_memory_to_cache(
                current_location_id,
                current_memory_header,
                current_memory_text_lines
            )

    def _add_memory_to_cache(
        self,
        location_id: int,
        memory_header: tuple,
        text_lines: List[str]
    ) -> None:
        """
        Add a parsed memory entry to the cache.

        Args:
            location_id: Integer location ID
            memory_header: Tuple of (category, status, title, metadata)
            text_lines: List of text lines for memory content
        """
        category, status, title, metadata = memory_header  # Unpack 4 values now

        try:
            # Parse metadata: "Ep1, T23-24, +0" or "Ep1, T100"
            episode, turns, score_change = self._parse_metadata(metadata)

            # Join text lines into single string
            text = " ".join(text_lines)

            # Remove strikethrough markers ONLY if status is SUPERSEDED
            # and text is wrapped (not embedded)
            if status == MemoryStatus.SUPERSEDED:
                # Only strip if text starts AND ends with ~~ (wrapping, not embedded)
                if text.startswith("~~") and text.endswith("~~"):
                    text = text[2:-2]  # Remove wrapping strikethrough markers only

            # Create Memory object with status
            memory = Memory(
                category=category,
                title=title,
                episode=episode,
                turns=turns,
                score_change=score_change,
                text=text,
                status=status  # Add status field
            )

            # Add to cache
            if location_id not in self.memory_cache:
                self.memory_cache[location_id] = []

            self.memory_cache[location_id].append(memory)

            self.log_debug(
                f"Added memory: [{category} - {status}] {title} at location {location_id}",
                location_id=location_id,
                category=category,
                status=status,
                title=title
            )

        except Exception as e:
            self.log_warning(
                f"Skipping malformed memory entry: [{category}] {title}",
                error=str(e)
            )

    def _parse_metadata(self, metadata: str) -> tuple[int, str, Optional[int]]:
        """
        Parse memory metadata string.

        Format: "Ep1, T23-24, +0" or "Ep1, T100" or "Ep2, T50, -5"

        Args:
            metadata: Metadata string from memory entry

        Returns:
            Tuple of (episode, turns, score_change)
        """
        # Split by comma
        parts = [p.strip() for p in metadata.split(",")]

        # Parse episode: "Ep1" -> 1
        episode = 1
        if parts[0].startswith("Ep"):
            episode = int(parts[0][2:])

        # Parse turns: "T23-24" or "T23"
        turns = ""
        if len(parts) > 1 and parts[1].startswith("T"):
            turns = parts[1][1:]  # Remove "T" prefix

        # Parse score change: "+0", "+5", "-2", or None
        score_change = None
        if len(parts) > 2:
            score_str = parts[2].strip()
            if score_str.startswith("+") or score_str.startswith("-"):
                score_change = int(score_str)

        return episode, turns, score_change

    # ========================================================================
    # Phase 1.2: File Writing and Memory Addition
    # ========================================================================

    def add_memory(
        self,
        location_id: int,
        location_name: str,
        memory: Memory
    ) -> bool:
        """
        Add a memory to file and update cache atomically.

        This method:
        1. Acquires file lock for thread safety
        2. Creates backup of existing file
        3. Reads entire file and parses structure
        4. Appends memory to existing location or creates new section
        5. Writes updated content atomically
        6. Updates in-memory cache

        Args:
            location_id: Integer location ID from Z-machine
            location_name: Location name for display
            memory: Memory object to add

        Returns:
            True if successful, False if operation failed
        """
        memories_path = Path(self.config.zork_game_workdir) / "Memories.md"
        lock_path = str(memories_path) + ".lock"

        try:
            # Acquire lock with 10 second timeout
            with FileLock(lock_path, timeout=10):
                # Create backup before write
                self._create_backup(memories_path)

                # Read existing content or start fresh
                if memories_path.exists():
                    content = memories_path.read_text(encoding="utf-8")
                else:
                    content = "# Location Memories\n\n"

                # Update content with new memory
                updated_content = self._add_memory_to_content(
                    content,
                    location_id,
                    location_name,
                    memory
                )

                # Write atomically
                memories_path.write_text(updated_content, encoding="utf-8")

                # Update cache after successful write
                if location_id not in self.memory_cache:
                    self.memory_cache[location_id] = []
                self.memory_cache[location_id].append(memory)

                self.log_info(
                    f"Added memory [{memory.category}] {memory.title} to location {location_id}",
                    location_id=location_id,
                    location_name=location_name,
                    category=memory.category,
                    title=memory.title
                )

                return True

        except Exception as e:
            self.log_error(
                f"Failed to add memory to location {location_id}: {e}",
                location_id=location_id,
                error=str(e)
            )
            return False

    def _create_backup(self, memories_path: Path) -> None:
        """
        Create backup of existing Memories.md file.

        Args:
            memories_path: Path to Memories.md file
        """
        if memories_path.exists():
            backup_path = Path(str(memories_path) + ".backup")
            shutil.copy2(memories_path, backup_path)
            self.log_debug(f"Created backup: {backup_path}")

    def _add_memory_to_content(
        self,
        content: str,
        location_id: int,
        location_name: str,
        memory: Memory
    ) -> str:
        """
        Add memory to content by either appending to existing location or creating new section.

        Args:
            content: Current file content
            location_id: Location ID
            location_name: Location name
            memory: Memory to add

        Returns:
            Updated file content
        """
        # Parse existing content to find location sections
        location_sections = self._parse_location_sections(content)

        if location_id in location_sections:
            # Append to existing location
            return self._append_to_location_section(
                content,
                location_id,
                location_sections[location_id],
                memory
            )
        else:
            # Create new location section at end
            return self._create_new_location_section(
                content,
                location_id,
                location_name,
                memory
            )

    def _parse_location_sections(self, content: str) -> Dict[int, Dict[str, Any]]:
        """
        Parse content to identify location sections and their positions.

        Args:
            content: File content

        Returns:
            Dictionary mapping location_id to section metadata
            {location_id: {"start": line_num, "end": line_num, "name": str, "memories": list}}
        """
        sections = {}
        lines = content.split("\n")
        current_location_id = None
        current_section_start = None

        for i, line in enumerate(lines):
            if line.startswith("## Location"):
                # Save previous section
                if current_location_id is not None and current_section_start is not None:
                    sections[current_location_id]["end"] = i - 1

                # Parse new location header
                match = self.LOCATION_HEADER_PATTERN.match(line)
                if match:
                    location_id = int(match.group(1))
                    location_name = match.group(2).strip()
                    current_location_id = location_id
                    current_section_start = i

                    sections[location_id] = {
                        "start": i,
                        "end": None,  # Will be set when next section starts or at EOF
                        "name": location_name,
                    }

        # Close final section
        if current_location_id is not None and current_section_start is not None:
            sections[current_location_id]["end"] = len(lines) - 1

        return sections

    def _append_to_location_section(
        self,
        content: str,
        location_id: int,
        section_info: Dict[str, Any],
        memory: Memory
    ) -> str:
        """
        Append memory to existing location section.

        Args:
            content: Current file content
            location_id: Location ID
            section_info: Section metadata from _parse_location_sections
            memory: Memory to add

        Returns:
            Updated file content
        """
        lines = content.split("\n")
        section_start = section_info["start"]
        section_end = section_info["end"]

        # Find insertion point (before separator or at section end)
        insert_idx = section_end
        for i in range(section_start, section_end + 1):
            if lines[i].strip() == "---":
                insert_idx = i
                break

        # Update visit metadata in header
        lines = self._update_visit_metadata(lines, section_start, memory.episode)

        # Format memory entry
        memory_entry = self._format_memory_entry(memory)

        # Insert memory entry before separator
        lines.insert(insert_idx, memory_entry)
        lines.insert(insert_idx + 1, "")

        return "\n".join(lines)

    def _create_new_location_section(
        self,
        content: str,
        location_id: int,
        location_name: str,
        memory: Memory
    ) -> str:
        """
        Create new location section at end of file.

        Args:
            content: Current file content
            location_id: Location ID
            location_name: Location name
            memory: Memory to add

        Returns:
            Updated file content
        """
        # Remove trailing whitespace and separators
        content = content.rstrip()
        if content and not content.endswith("\n"):
            content += "\n"

        # Ensure proper spacing before new section
        if content and not content.endswith("\n\n"):
            content += "\n"

        # Format new location section
        section = self._format_location_section(
            location_id,
            location_name,
            [memory],
            episode=memory.episode
        )

        return content + section

    def _format_location_section(
        self,
        location_id: int,
        location_name: str,
        memories: List[Memory],
        episode: int
    ) -> str:
        """
        Format complete location section.

        Args:
            location_id: Location ID
            location_name: Location name
            memories: List of memories
            episode: Current episode number

        Returns:
            Formatted location section
        """
        lines = []

        # Location header
        lines.append(f"## Location {location_id}: {location_name}")
        lines.append(f"**Visits:** 1 | **Episodes:** {episode}")
        lines.append("")
        lines.append("### Memories")
        lines.append("")

        # Memory entries
        for memory in memories:
            lines.append(self._format_memory_entry(memory))
            lines.append("")

        # Section separator
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _format_memory_entry(self, memory: Memory) -> str:
        """
        Format single memory entry with status indicator.

        Format: **[CATEGORY] Title** *(EpX, TY, +/-Z)*
                text

        For TENTATIVE/SUPERSEDED: **[CATEGORY - STATUS] Title** *(metadata)*
                                   [Superseded at TurnX by "Title"]
                                   ~~text~~

        Args:
            memory: Memory object

        Returns:
            Formatted memory entry (multi-line string)
        """
        # Format metadata
        metadata_parts = [f"Ep{memory.episode}", f"T{memory.turns}"]

        # Only include score if not None
        if memory.score_change is not None:
            score_str = f"+{memory.score_change}" if memory.score_change >= 0 else str(memory.score_change)
            metadata_parts.append(score_str)

        metadata = ", ".join(metadata_parts)

        # Format header with optional status
        if memory.status == MemoryStatus.ACTIVE:
            header = f"**[{memory.category}] {memory.title}** *({metadata})*"
        else:
            header = f"**[{memory.category} - {memory.status}] {memory.title}** *({metadata})*"

        # Format text (strikethrough if superseded)
        if memory.status == MemoryStatus.SUPERSEDED:
            text = f"~~{memory.text}~~"
        else:
            text = memory.text

        # Build lines
        lines = [header]

        # Add supersession reference if applicable
        if memory.status == MemoryStatus.SUPERSEDED and memory.superseded_by:
            lines.append(f"[Superseded at T{memory.superseded_at_turn} by \"{memory.superseded_by}\"]")

        lines.append(text)

        return "\n".join(lines)

    def _update_memory_status(
        self,
        location_id: int,
        memory_title: str,
        new_status: str,
        superseded_by: str,
        superseded_at_turn: int
    ) -> bool:
        """
        Update the status of an existing memory in file and cache.

        This method:
        1. Reads entire Memories.md file
        2. Finds memory by title at specific location
        3. Updates status in header (**[CATEGORY - STATUS] Title**)
        4. Adds supersession reference line
        5. Wraps text in strikethrough if SUPERSEDED
        6. Writes back to file atomically
        7. Updates cache

        Args:
            location_id: Location where memory exists
            memory_title: Title of memory to update (exact match or substring)
            new_status: New status (typically MemoryStatus.SUPERSEDED)
            superseded_by: Title of new memory that superseded this one
            superseded_at_turn: Turn number when superseded

        Returns:
            True if successful, False if memory not found or update failed
        """
        memories_path = Path(self.config.zork_game_workdir) / "Memories.md"
        lock_path = str(memories_path) + ".lock"

        try:
            # Acquire lock
            with FileLock(lock_path, timeout=10):
                # Backup before modification
                self._create_backup(memories_path)

                # Read entire file
                if not memories_path.exists():
                    self.log_warning(f"Cannot update memory: Memories.md not found")
                    return False

                content = memories_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                # Find the memory entry and update it
                updated_lines = []
                in_target_location = False
                in_target_memory = False
                memory_found = False
                i = 0

                while i < len(lines):
                    line = lines[i]

                    # Check for location header
                    location_match = self.LOCATION_HEADER_PATTERN.match(line)
                    if location_match:
                        loc_id = int(location_match.group(1))
                        in_target_location = (loc_id == location_id)
                        in_target_memory = False
                        updated_lines.append(line)
                        i += 1
                        continue

                    # Check for memory entry header (only in target location)
                    if in_target_location:
                        memory_match = self.MEMORY_ENTRY_PATTERN.match(line)
                        if memory_match:
                            category = memory_match.group(1)
                            status = memory_match.group(2) or MemoryStatus.ACTIVE
                            title = memory_match.group(3).strip()
                            metadata = memory_match.group(4).strip()

                            # Check if this is the memory to update (exact or substring match)
                            if memory_title in title or title in memory_title:
                                # Found the memory - update header
                                memory_found = True
                                in_target_memory = True

                                # Format new header with status
                                updated_header = f"**[{category} - {new_status}] {title}** *({metadata})*"
                                updated_lines.append(updated_header)

                                # Add supersession reference on next line
                                supersession_ref = f'[Superseded at T{superseded_at_turn} by "{superseded_by}"]'
                                updated_lines.append(supersession_ref)

                                i += 1

                                # Now collect and update the memory text lines
                                while i < len(lines):
                                    text_line = lines[i]

                                    # Stop at next memory or section end
                                    if (text_line.startswith("**[") or
                                        text_line.strip() == "---" or
                                        text_line.startswith("##")):
                                        in_target_memory = False
                                        break

                                    # Skip existing supersession references
                                    if text_line.strip().startswith("[Superseded at"):
                                        i += 1
                                        continue

                                    # Wrap text in strikethrough if not already
                                    if text_line.strip() and not text_line.strip().startswith("~~"):
                                        text_line = f"~~{text_line}~~"  # Preserve original formatting

                                    updated_lines.append(text_line)
                                    i += 1

                                continue
                            else:
                                # Not the target memory - keep as is
                                in_target_memory = False
                                updated_lines.append(line)
                                i += 1
                                continue

                    # Default: keep line as-is
                    updated_lines.append(line)
                    i += 1

                if not memory_found:
                    self.log_warning(
                        f"Memory '{memory_title}' not found at location {location_id}",
                        location_id=location_id,
                        memory_title=memory_title
                    )
                    return False

                # Write updated content
                memories_path.write_text("\n".join(updated_lines), encoding="utf-8")

                # Update cache - find and update the memory object
                if location_id in self.memory_cache:
                    for memory in self.memory_cache[location_id]:
                        if memory_title in memory.title or memory.title in memory_title:
                            memory.status = new_status
                            memory.superseded_by = superseded_by
                            memory.superseded_at_turn = superseded_at_turn
                            break

                self.log_info(
                    f"Updated memory status: '{memory_title}' → {new_status}",
                    location_id=location_id,
                    memory_title=memory_title,
                    new_status=new_status,
                    superseded_by=superseded_by
                )

                return True

        except Exception as e:
            self.log_error(
                f"Failed to update memory status: {e}",
                location_id=location_id,
                memory_title=memory_title,
                error=str(e)
            )
            return False

    def _update_visit_metadata(
        self,
        lines: List[str],
        section_start: int,
        episode: int
    ) -> List[str]:
        """
        Update visit count and episodes list in location header.

        Args:
            lines: File lines
            section_start: Starting line of location section
            episode: Current episode number

        Returns:
            Updated lines
        """
        # Find visits/episodes line (should be line after header)
        visits_line_idx = section_start + 1

        if visits_line_idx < len(lines):
            visits_line = lines[visits_line_idx]

            # Parse current metadata
            visits = 1
            episodes = set()

            visits_match = re.search(r"\*\*Visits:\*\* (\d+)", visits_line)
            if visits_match:
                visits = int(visits_match.group(1))

            episodes_match = re.search(r"\*\*Episodes:\*\* ([\d, ]+)", visits_line)
            if episodes_match:
                episode_strs = episodes_match.group(1).split(",")
                episodes = {int(e.strip()) for e in episode_strs if e.strip()}

            # Update with new visit
            visits += 1
            episodes.add(episode)

            # Format updated line
            episodes_list = ", ".join(str(e) for e in sorted(episodes))
            lines[visits_line_idx] = f"**Visits:** {visits} | **Episodes:** {episodes_list}"

        return lines

    # ========================================================================
    # Phase 1.3: Z-machine Trigger Detection
    # ========================================================================

    def _should_synthesize_memory(self, z_machine_context: Dict) -> bool:
        """
        Determine if action outcome warrants LLM memory synthesis.

        Uses Z-machine ground truth data to make fast boolean decision.
        No LLM calls - pure boolean logic based on state changes.

        Args:
            z_machine_context: Dict with keys:
                - score_before, score_after, score_delta
                - location_before, location_after, location_changed
                - inventory_before, inventory_after, inventory_changed
                - died (bool)
                - response_length (int)
                - first_visit (bool)

        Returns:
            True if LLM synthesis should be invoked, False otherwise
        """
        # Trigger 1: Score changed
        if z_machine_context.get('score_delta', 0) != 0:
            self.log_debug(
                "Trigger: Score changed",
                score_delta=z_machine_context.get('score_delta')
            )
            return True

        # Trigger 2: Location changed
        if z_machine_context.get('location_changed', False):
            self.log_debug(
                "Trigger: Location changed",
                location_before=z_machine_context.get('location_before'),
                location_after=z_machine_context.get('location_after')
            )
            return True

        # Trigger 3: Inventory changed
        if z_machine_context.get('inventory_changed', False):
            self.log_debug(
                "Trigger: Inventory changed",
                inventory_before=z_machine_context.get('inventory_before'),
                inventory_after=z_machine_context.get('inventory_after')
            )
            return True

        # Trigger 4: Death occurred
        if z_machine_context.get('died', False):
            self.log_debug("Trigger: Death occurred")
            return True

        # Trigger 5: First visit to location
        if z_machine_context.get('first_visit', False):
            self.log_debug(
                "Trigger: First visit to location",
                location=z_machine_context.get('location_after')
            )
            return True

        # Trigger 6: Substantial response (>100 characters, not >=100)
        if z_machine_context.get('response_length', 0) > 100:
            self.log_debug(
                "Trigger: Substantial response",
                response_length=z_machine_context.get('response_length')
            )
            return True

        # No triggers fired
        return False

    # ========================================================================
    # Phase 1.4: LLM Memory Synthesis
    # ========================================================================

    def _synthesize_memory(
        self,
        location_id: int,
        location_name: str,
        action: str,
        response: str,
        z_machine_context: Dict
    ) -> Optional[MemorySynthesisResponse]:
        """
        Invoke LLM to synthesize memory from action outcome.

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
            # Get existing memories from cache for deduplication
            existing_memories = self.memory_cache.get(location_id, [])

            # Format existing memories - TITLES ONLY for conciseness
            if existing_memories:
                memory_titles = "\n".join(
                    f"  • [{mem.category}] {mem.title}"
                    for mem in existing_memories
                )
                existing_section = f"""
EXISTING MEMORIES AT THIS LOCATION:
{memory_titles}
"""
            else:
                existing_section = "\nEXISTING MEMORIES: None (first memory for this location)\n"

            # ================================================================
            # Phase 3: Multi-Step Procedure Detection - History Retrieval
            # ================================================================
            # Original system was turn-atomic: LLM only saw current action/response pair.
            # This prevented capturing procedures spanning multiple turns:
            # - Prerequisites: "open window" (turn N) → "enter window" (turn N+1)
            # - Delayed consequences: "give lunch to troll" → troll attacks later
            # - Progressive discovery: "examine door" → "unlock" → "open"
            #
            # Solution: Retrieve recent action and reasoning history, inject into synthesis prompt.
            # This gives LLM temporal context to recognize multi-step patterns.

            # Get configurable history window (default: 3 turns, validated >= 1)
            window_size = self.config.get_memory_history_window()

            # Retrieve recent actions from shared game state (action, response) tuples
            # Uses sliding window: if window_size=3, gets last 3 entries from action_history
            recent_actions = self.game_state.action_history[-window_size:] if self.game_state.action_history else []
            current_turn = self.game_state.turn_count

            # Retrieve recent reasoning from shared game state (reasoning history entries)
            # Each entry: {"turn": int, "reasoning": str, "action": str, "timestamp": str}
            recent_reasoning = self.game_state.action_reasoning_history[-window_size:] if self.game_state.action_reasoning_history else []

            # Format using dedicated helpers (matches ContextManager conventions for consistency)
            # _format_recent_actions: Creates "Turn N: action\nResponse: response" format
            # _format_recent_reasoning: Adds agent reasoning with response lookup via reverse iteration
            start_turn = max(1, current_turn - len(recent_actions) + 1)
            actions_formatted = self._format_recent_actions(recent_actions, start_turn)
            reasoning_formatted = self._format_recent_reasoning(recent_reasoning, self.game_state.action_history)

            # Build concise, focused prompt optimized for reasoning model
            prompt = f"""Location: {location_name} (ID: {location_id})
{existing_section}
═══════════════════════════════════════════════════════════════
🚨 CRITICAL DEDUPLICATION CHECK 🚨

Before remembering ANYTHING, compare against existing memories above.
These are SEMANTICALLY DUPLICATE (DO NOT remember):
  ❌ "Leaflet reveals message" vs "Leaflet provides message"
  ❌ "Mailbox contains leaflet" vs "Leaflet found in mailbox"
  ❌ "Egg can be taken" vs "Taking egg succeeds"

Only remember if this provides NEW information not semantically captured above.
═══════════════════════════════════════════════════════════════

CONTRADICTION CHECK:
═══════════════════════════════════════════════════════════════
Review existing memories above. Does this action outcome:

1. CONTRADICT any existing memory? (proves it wrong)
   Example: Memory says "troll accepts gifts peacefully" but troll attacks after accepting
   → Mark that memory as SUPERSEDED, create new DANGER memory

2. REVEAL DELAYED CONSEQUENCES? (success wasn't really success)
   Example: "Door opens" seemed successful but leads to death trap
   → Mark optimistic memory as SUPERSEDED, create WARNING memory

3. CLARIFY a TENTATIVE memory? (confirms or denies uncertain outcome)
   Example: TENTATIVE "troll might be friendly" → CONFIRMED as false by attack
   → Mark tentative memory as SUPERSEDED

If yes to any: list specific memory TITLES in supersedes_memory_titles field.
If contradicting multiple memories: list ALL relevant titles.
Use EXACT titles from existing memories above. If title is long, unique substring is sufficient.
═══════════════════════════════════════════════════════════════

MEMORY STATUS DECISION:
═══════════════════════════════════════════════════════════════
WORKFLOW: First check duplicates → Then check contradictions → Then determine status

Choose status based on outcome certainty:

**ACTIVE** (default) - Use when:
✓ Outcome is immediate and certain
✓ Consequence is fully understood
✓ No delayed effects expected
Examples:
  • "Mailbox contains leaflet" (examined, saw leaflet, certain)
  • "Window is locked" (tried to open, failed, certain)
  • "Lamp provides light" (lit lamp, room illuminated, confirmed)

**TENTATIVE** - Use when:
⚠️  Immediate action succeeds BUT long-term consequence unclear
⚠️  Entity accepts action BUT reaction not yet known
⚠️  Effect seems positive BUT might have hidden downsides
Examples:
  • "Troll accepts lunch gift" (took it but might attack later) → TENTATIVE
  • "Door unlocked successfully" (opened but don't know what's inside) → TENTATIVE
  • "Drank mysterious potion" (consumed but effect not yet clear) → TENTATIVE

**Rule of thumb:** If you think "this worked... for now", mark it TENTATIVE.
═══════════════════════════════════════════════════════════════
"""

            # Add history sections if available (for multi-step procedure detection)
            if actions_formatted or reasoning_formatted:
                prompt += "\n═══════════════════════════════════════════════════════════════\n"
                prompt += "RECENT ACTION SEQUENCE:\n"
                prompt += "═══════════════════════════════════════════════════════════════\n"

                if actions_formatted:
                    prompt += f"\n{actions_formatted}\n"
                else:
                    prompt += "\n(No recent actions available - this is one of the first turns)\n"

                if reasoning_formatted:
                    prompt += "\nAGENT'S REASONING:\n"
                    prompt += f"{reasoning_formatted}\n"

                prompt += "\n═══════════════════════════════════════════════════════════════\n"

            # Continue with ACTION ANALYSIS section
            prompt += f"""
ACTION ANALYSIS:
Action: {action}
Response: {response}

State Changes (ground truth from Z-machine):
• Score: {z_machine_context.get('score_delta', 0):+d} points
• Location changed: {z_machine_context.get('location_changed', False)}
• Inventory changed: {z_machine_context.get('inventory_changed', False)}
• Died: {z_machine_context.get('died', False)}
• First visit: {z_machine_context.get('first_visit', False)}

REASONING STEPS (use your reasoning capabilities):
1. Identify the KEY object/entity in this action (e.g., "leaflet", "troll", "egg")
2. Identify the KEY relationship/insight (e.g., "contains item", "blocks path", "is takeable")
3. Check existing memories: Does ANY memory mention this object + relationship?
4. Use semantic matching:
   - "reveals" = "provides" = "shows" = "contains"
   - "take" = "pick up" = "grab" = "obtain"
   - "blocks" = "prevents" = "stops"
5. If semantic match found → should_remember=false
6. If truly new insight → should_remember=true

MULTI-STEP PROCEDURE DETECTION:
═══════════════════════════════════════════════════════════════
Review the RECENT ACTION SEQUENCE above. Does the current outcome depend on previous actions?

**Look for these patterns:**

1. **Prerequisites** (action B requires action A first):
   Example: "open window" (turn N) → "enter window" (turn N+1) → success
   Memory: "To enter kitchen: (1) open window, (2) enter window"

2. **Delayed Consequences** (action seemed successful but had delayed effect):
   Example: "give lunch to troll" (turn N, seemed ok) → troll attacks (turn N+1)
   Memory: "Troll attacks after accepting gift - gift strategy fails"
   Action: Mark previous TENTATIVE memory as SUPERSEDED

3. **Progressive Discovery** (understanding deepens over multiple turns):
   Example: Turn N "examine door" (locked) → Turn N+1 "unlock with key" → Turn N+2 "open door" (success)
   Memory: "Door requires key to unlock before opening"

**How to capture multi-step procedures:**
- If outcome required previous actions: Include steps in memory_text
- Format: "To achieve X: (1) step1, (2) step2" or "After A, then B occurs"
- Don't duplicate if existing memory already captures the complete procedure

═══════════════════════════════════════════════════════════════

REMEMBER (actionable game mechanics):
✅ Object interactions (how to use items, what works/fails)
✅ Item discoveries (finding items, understanding purpose)
✅ Dangers (death, hazards, threats)
✅ Puzzle mechanics (how things operate)
✅ Score-earning actions

SKIP (handled elsewhere or not actionable):
❌ Navigation/directions (tracked by MapGraph)
❌ Room descriptions (in game output)
❌ Movement commands (north/south/etc.)
❌ DUPLICATES (semantically similar to existing memories)

**CRITICAL - OUTPUT FORMAT:**
YOU MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON. Do not include thinking tags, reasoning outside the JSON structure, or markdown fences.

If should_remember=false (duplicate/navigation/not actionable):
{{
  "should_remember": false,
  "reasoning": "explain why not remembering (semantic duplicate, navigation, etc.)"
}}

If should_remember=true (new actionable insight):
{{
  "should_remember": true,
  "category": "SUCCESS"|"FAILURE"|"DISCOVERY"|"DANGER"|"NOTE",
  "memory_title": "3-6 words, evergreen",
  "memory_text": "1-2 sentences, actionable insight",
  "status": "ACTIVE"|"TENTATIVE",
  "supersedes_memory_titles": ["Title1", "Title2"],
  "reasoning": "explain semantic comparison, contradiction detection, status choice"
}}

Example valid response for NOT remembering:
{{
  "should_remember": false,
  "reasoning": "Semantic duplicate - existing memory 'Mailbox contains leaflet' already captures this insight"
}}

Example valid response for remembering:
{{
  "should_remember": true,
  "category": "DANGER",
  "memory_title": "Troll attacks after accepting gift",
  "memory_text": "Troll accepts lunch gift but then becomes hostile and attacks. Gift strategy ineffective.",
  "status": "ACTIVE",
  "supersedes_memory_titles": ["Troll accepts lunch gift"],
  "reasoning": "Contradicts previous tentative memory - troll is not pacified by gifts"
}}"""

            # Call LLM with structured output
            llm_response = self.llm_client.chat.completions.create(
                model=self.config.memory_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.memory_sampling.get('temperature', 0.3),
                max_tokens=self.config.memory_sampling.get('max_tokens', 1000),
                name="SimpleMemory",
                response_format=create_json_schema(MemorySynthesisResponse)
            )

            # Note: Empty response checking is now handled by llm_client with automatic retries
            # Extract JSON (handles markdown fences, reasoning tags, and embedded JSON)
            json_content = extract_json_from_text(llm_response.content)

            # Parse response
            synthesis = MemorySynthesisResponse.model_validate_json(json_content)

            # Check if should remember
            if not synthesis.should_remember:
                self.log_debug(
                    "LLM decided not to remember",
                    location_id=location_id,
                    reasoning=synthesis.reasoning
                )
                return None

            self.log_debug(
                "LLM synthesis complete",
                location_id=location_id,
                category=synthesis.category,
                title=synthesis.memory_title,
                reasoning=synthesis.reasoning
            )

            return synthesis

        except Exception as e:
            # Get response preview safely (llm_response may not be defined if error occurred during call)
            try:
                response_preview = llm_response.content[:500] if llm_response and llm_response.content else "No response content"
            except (NameError, AttributeError):
                response_preview = "Error occurred before LLM response received"

            self.log_error(
                f"Failed to synthesize memory: {e}",
                location_id=location_id,
                error=str(e),
                response_preview=response_preview
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
        if not self._should_synthesize_memory(z_machine_context):
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
                    superseded_at_turn=self.game_state.turn_count
                )

                if not success:
                    self.log_warning(
                        f"Failed to supersede memory: '{old_memory_title}'",
                        location_id=location_id,
                        old_title=old_memory_title,
                        new_title=synthesis.memory_title
                    )

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

        Filters memories by status:
        - ACTIVE: Shown normally
        - TENTATIVE: Shown with warning marker
        - SUPERSEDED: Hidden (proven wrong by later evidence)

        Args:
            location_id: Location ID to retrieve memories for

        Returns:
            Formatted string with filtered memories
        """
        if location_id not in self.memory_cache:
            return ""

        memories = self.memory_cache[location_id]
        if not memories:
            return ""

        # Separate memories by status
        active_memories = [m for m in memories if m.status == MemoryStatus.ACTIVE]
        tentative_memories = [m for m in memories if m.status == MemoryStatus.TENTATIVE]
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

    # ========================================================================
    # Phase 2: History Formatting Helpers
    # ========================================================================

    def _format_recent_actions(
        self,
        actions: List[Tuple[str, str]],
        start_turn: int
    ) -> str:
        """
        Format recent action/response pairs into markdown for multi-step synthesis.

        Part of Phase 2 helpers used by Phase 3's multi-step procedure detection.
        Matches ContextManager formatting conventions for consistency across systems.

        Args:
            actions: List of (action, response) tuples from game_state.action_history
            start_turn: Turn number of the first action in the list

        Returns:
            Formatted markdown string with turn context, empty string if no actions

        Example output:
            Turn 47: go north
            Response: You are in a forest clearing.

            Turn 48: examine trees
            Response: The trees are ordinary pine trees.

        Usage:
            Injected into synthesis prompt's "RECENT ACTION SEQUENCE" section to give
            LLM temporal context for detecting prerequisites and delayed consequences.
        """
        # Handle empty list
        if not actions:
            return ""

        lines = []
        for i, (action, response) in enumerate(actions):
            turn_num = start_turn + i
            lines.append(f"Turn {turn_num}: {action}")
            lines.append(f"Response: {response}")
            # Add blank line between entries (except after last entry)
            if i < len(actions) - 1:
                lines.append("")

        return "\n".join(lines)

    def _format_recent_reasoning(
        self,
        reasoning_entries: List[Dict[str, Any]],
        action_history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        Format recent reasoning history into markdown for multi-step synthesis.

        Part of Phase 2 helpers used by Phase 3's multi-step procedure detection.
        Matches ContextManager formatting conventions (Turn → Reasoning → Action → Response).
        Uses reverse iteration through action_history to match actions to responses.

        Args:
            reasoning_entries: List of reasoning history dicts from game_state.action_reasoning_history
                Each dict has: turn, reasoning, action, timestamp
            action_history: Optional list of (action, response) tuples for response lookup
                Uses reverse iteration to handle duplicate actions correctly

        Returns:
            Formatted markdown string with reasoning context, empty string if no entries

        Example output:
            Turn 47:
            Reasoning: I need to explore north systematically.
            Action: go north
            Response: You are in a forest clearing.

            Turn 48:
            Reasoning: Will examine objects before moving on.
            Action: examine trees
            Response: The trees are ordinary pine trees.

        Usage:
            Injected into synthesis prompt's "AGENT'S REASONING" section to help LLM
            understand strategic intent behind multi-step procedures.
        """
        # Handle empty list
        if not reasoning_entries:
            return ""

        lines = []
        for i, entry in enumerate(reasoning_entries):
            # Skip non-dict entries gracefully
            if not isinstance(entry, dict):
                self.log_debug(
                    "Skipping non-dict reasoning entry",
                    entry_type=type(entry).__name__
                )
                continue

            # Extract fields with fallbacks
            turn = entry.get("turn")
            if turn is None:
                turn = "?"
            reasoning = entry.get("reasoning", "(No reasoning recorded)")
            action = entry.get("action", "(No action recorded)")

            # Find matching game response from action_history
            # Iterate in reverse to match the most recent occurrence
            response = "(Response not recorded)"
            if action_history:
                for hist_action, hist_response in reversed(action_history):
                    if hist_action == action:
                        response = hist_response
                        break

            # Format this entry
            lines.append(f"Turn {turn}:")
            lines.append(f"Reasoning: {reasoning}")
            lines.append(f"Action: {action}")
            lines.append(f"Response: {response}")
            # Add blank line between entries (except after last entry)
            if i < len(reasoning_entries) - 1:
                lines.append("")

        return "\n".join(lines)
