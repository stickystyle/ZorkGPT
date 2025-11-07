"""
ABOUTME: File I/O operations for ZorkGPT memory system - reading and writing Memories.md.
ABOUTME: Provides thread-safe file operations with locking, backups, and atomic writes.
"""

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from filelock import FileLock

from .models import Memory, MemoryStatus, MemoryStatusType, INVALIDATION_MARKER
from .cache_manager import MemoryCacheManager
from session.game_configuration import GameConfiguration


class MemoryFileParser:
    """
    Parses Memories.md file and populates memory caches.

    Responsible for:
    - Reading and parsing Memories.md file format
    - Extracting location sections and memory entries
    - Parsing metadata (episode, turns, score changes)
    - Populating cache_manager with parsed memories
    - Handling malformed sections gracefully
    """

    # Regex patterns for parsing Memories.md format
    LOCATION_HEADER_PATTERN = re.compile(r"^## Location (\d+): (.+)$")
    # Matches: **[CATEGORY] Title** or **[CATEGORY - PERSISTENCE] Title** or **[CATEGORY - PERSISTENCE - STATUS] Title**
    # Group 1: Category (e.g., "NOTE", "SUCCESS")
    # Group 2: Optional middle field (persistence or status)
    # Group 3: Optional final field (status when persistence present)
    # Group 4: Title
    # Group 5: Metadata
    MEMORY_ENTRY_PATTERN = re.compile(r"^\*\*\[(\w+)(?: - (\w+))?(?: - (\w+))?\] (.+?)\*\* \*\((.*?)\)\*$")
    # Matches: [Invalidated at T<turn>: "<reason>"]
    # Use negated character class to exclude quotes from capture
    INVALIDATION_REF_PATTERN = re.compile(r'^\[Invalidated at T(\d+): "([^"]*)"\]$')

    def __init__(self, logger, config: GameConfiguration, cache_manager: MemoryCacheManager):
        """
        Initialize file parser.

        Args:
            logger: Logger instance for debugging
            config: GameConfiguration for file paths
            cache_manager: MemoryCacheManager to populate during parsing
        """
        self.logger = logger
        self.config = config
        self.cache_manager = cache_manager

    def load_from_file(self) -> None:
        """
        Load and parse Memories.md file into memory cache.

        Handles missing, empty, or corrupted files gracefully with appropriate
        logging. Parsing continues after encountering corrupted sections.
        """
        memories_path = Path(self.config.zork_game_workdir) / "Memories.md"

        # Handle missing file gracefully
        if not memories_path.exists():
            self.logger.warning(f"Memories.md not found at {memories_path}")
            return

        try:
            content = memories_path.read_text(encoding="utf-8")
            self._parse_memories_content(content)

            # Get cache statistics via cache_manager
            locations_tracked = self.cache_manager.get_locations_tracked()
            total_memories = self.cache_manager.get_total_memories()

            self.logger.info(
                f"Loaded {total_memories} memories from {locations_tracked} locations",
                extra={
                    "locations": locations_tracked,
                    "total_memories": total_memories,
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to load Memories.md: {e}", extra={"error": str(e)})

    def _parse_memories_content(self, content: str) -> None:
        """
        Parse Memories.md content into memory cache.

        Format:
        ## Location 15: West of House
        **Visits:** 3 | **Episodes:** 1, 2, 3

        ### Memories

        **[SUCCESS] Title** *(Ep1, T23-24, +0)*
        Memory text content here.

        **[DANGER - SUPERSEDED] Bad Title** *(Ep1, T50)*
        [Invalidated at T55: "Proven false by death"]
        ~~Memory text content here.~~

        Args:
            content: Full text content of Memories.md file
        """
        lines = content.split("\n")
        current_location_id: Optional[int] = None
        current_memory_header: Optional[tuple] = None  # (category, title, metadata)
        current_memory_text_lines: List[str] = []
        current_invalidation_info: Optional[tuple] = None  # (turn, reason)

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
                        current_memory_text_lines,
                        current_invalidation_info
                    )
                    current_memory_header = None
                    current_memory_text_lines = []
                    current_invalidation_info = None

                if location_match:
                    # Valid location header - parse it
                    try:
                        location_id = int(location_match.group(1))
                        location_name = location_match.group(2).strip()
                        current_location_id = location_id

                        # Cache manager handles location initialization automatically

                        self.logger.debug(
                            f"Parsing location {location_id}: {location_name}",
                            extra={
                                "location_id": location_id,
                                "location_name": location_name
                            }
                        )

                    except (ValueError, IndexError) as e:
                        self.logger.warning(
                            f"Error parsing location header: {line}",
                            extra={"error": str(e)}
                        )
                        current_location_id = None
                else:
                    # Malformed location header (e.g., "## Location Invalid: Not a Number")
                    self.logger.warning(
                        f"Skipping malformed location header: {line}",
                        extra={"line": line}
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
                        current_memory_text_lines,
                        current_invalidation_info
                    )
                    current_invalidation_info = None  # Reset for next memory

                # Only parse new memory header if we have a valid location
                if current_location_id is not None:
                    # Parse new memory header with updated regex groups
                    # Group 1: Category
                    # Group 2: Optional middle field (persistence or status)
                    # Group 3: Optional final field (status when persistence present)
                    # Group 4: Title
                    # Group 5: Metadata
                    category = memory_match.group(1)
                    second_field = memory_match.group(2)
                    third_field = memory_match.group(3)
                    title = memory_match.group(4).strip()
                    metadata = memory_match.group(5).strip()

                    # Determine persistence and status
                    persistence = "permanent"  # Default
                    status = MemoryStatus.ACTIVE  # Default

                    # If third_field exists, we have: CATEGORY - PERSISTENCE - STATUS
                    if third_field:
                        # Second field is persistence, third field is status
                        if second_field:
                            second_field_upper = second_field.upper()
                            if second_field_upper in ["CORE", "PERMANENT", "EPHEMERAL"]:
                                persistence = second_field.lower()
                            else:
                                self.logger.warning(
                                    f"Expected persistence marker, got '{second_field}', defaulting to 'permanent'",
                                    extra={"line": line, "field": second_field}
                                )
                        if third_field in [MemoryStatus.ACTIVE, MemoryStatus.TENTATIVE, MemoryStatus.SUPERSEDED]:
                            status = third_field
                        else:
                            self.logger.warning(
                                f"Expected status marker, got '{third_field}', defaulting to ACTIVE",
                                extra={"line": line, "field": third_field}
                            )
                    # If only second_field exists, we have: CATEGORY - (PERSISTENCE or STATUS)
                    elif second_field:
                        second_field_upper = second_field.upper()

                        # Check if it's a persistence marker (CORE, PERMANENT, EPHEMERAL)
                        if second_field_upper in ["CORE", "PERMANENT", "EPHEMERAL"]:
                            persistence = second_field.lower()
                        # Check if it's a status marker (ACTIVE, TENTATIVE, SUPERSEDED)
                        elif second_field in [MemoryStatus.ACTIVE, MemoryStatus.TENTATIVE, MemoryStatus.SUPERSEDED]:
                            status = second_field
                        else:
                            self.logger.warning(
                                f"Unknown second field '{second_field}', treating as status and defaulting to ACTIVE",
                                extra={"line": line, "field": second_field}
                            )

                    # Store persistence, status, and other header info
                    current_memory_header = (category, persistence, status, title, metadata)
                    current_memory_text_lines = []
                else:
                    # Skip memory entries when not in a valid location
                    self.logger.warning(
                        f"Skipping memory entry (no valid location): {line}",
                        extra={"line": line}
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
                        current_memory_text_lines,
                        current_invalidation_info
                    )
                    current_memory_header = None
                    current_memory_text_lines = []
                    current_invalidation_info = None

                # Log and skip malformed entry
                self.logger.warning(
                    f"Skipping malformed memory entry: {line}",
                    extra={"line": line}
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
                # Detect and skip invalidation reference lines
                if line.strip().startswith("[Invalidated at") and ': "' in line and line.strip().endswith('"]'):
                    invalidation_match = self.INVALIDATION_REF_PATTERN.match(line.strip())
                    if invalidation_match:
                        turn = int(invalidation_match.group(1))
                        reason = invalidation_match.group(2)
                        current_invalidation_info = (turn, reason)
                    continue  # Don't include in memory text

                current_memory_text_lines.append(line.strip())

        # Save final pending memory
        if current_memory_header and current_location_id is not None:
            self._add_memory_to_cache(
                current_location_id,
                current_memory_header,
                current_memory_text_lines,
                current_invalidation_info
            )

    def _add_memory_to_cache(
        self,
        location_id: int,
        memory_header: tuple,
        text_lines: List[str],
        invalidation_info: Optional[tuple] = None
    ) -> None:
        """
        Add a parsed memory entry to the cache.

        Args:
            location_id: Integer location ID
            memory_header: Tuple of (category, persistence, status, title, metadata)
            text_lines: List of text lines for memory content
            invalidation_info: Optional tuple of (turn, reason) for invalidated memories
        """
        category, persistence, status, title, metadata = memory_header  # Unpack 5 values now

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

            # Extract invalidation details if present
            invalidation_reason = None
            if invalidation_info:
                _, invalidation_reason = invalidation_info

            # Create Memory object with parsed persistence and status
            memory = Memory(
                category=category,
                title=title,
                episode=episode,
                turns=turns,
                score_change=score_change,
                text=text,
                persistence=persistence,  # Use parsed persistence value
                status=status,
                superseded_by=INVALIDATION_MARKER if invalidation_reason else None,
                invalidation_reason=invalidation_reason
            )

            # Add to cache via cache_manager (always goes to persistent cache during file load)
            self.cache_manager.add_to_cache(location_id, memory)

            self.logger.debug(
                f"Added memory: [{category} - {status}] {title} at location {location_id}",
                extra={
                    "location_id": location_id,
                    "category": category,
                    "status": status,
                    "title": title
                }
            )

        except Exception as e:
            self.logger.warning(
                f"Skipping malformed memory entry: [{category}] {title}",
                extra={"error": str(e)}
            )

    def _parse_metadata(self, metadata: str) -> Tuple[int, str, Optional[int]]:
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


class MemoryFileWriter:
    """
    Writes memories to Memories.md with file locking and backups.

    Responsible for:
    - Thread-safe file writes with filelock library
    - Atomic backup creation before modifications
    - Memory entry formatting with persistence markers
    - Location section management (append vs create)
    - Memory status updates (supersession/invalidation)
    - Visit metadata tracking
    """

    # Regex patterns for file parsing during writes
    LOCATION_HEADER_PATTERN = re.compile(r"^## Location (\d+): (.+)$")
    MEMORY_ENTRY_PATTERN = re.compile(r"^\*\*\[(\w+)(?: - (\w+))?(?: - (\w+))?\] (.+?)\*\* \*\((.*?)\)\*$")

    def __init__(self, logger, config: GameConfiguration):
        """
        Initialize file writer.

        Args:
            logger: Logger instance for debugging
            config: GameConfiguration for file paths
        """
        self.logger = logger
        self.config = config

    def write_memory(
        self,
        memory: Memory,
        location_id: int,
        location_name: str
    ) -> bool:
        """
        Write memory to file with locking and backup.

        Thread-safe write operation:
        1. Acquire file lock (10 second timeout)
        2. Create timestamped backup
        3. Read current file content
        4. Add/update memory in content
        5. Write atomically

        Args:
            memory: Memory object to write
            location_id: Integer location ID from Z-machine
            location_name: Location name for display

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

                self.logger.info(
                    f"Added {memory.persistence} memory to file: [{memory.category}] {memory.title} to location {location_id}",
                    extra={
                        "location_id": location_id,
                        "location_name": location_name,
                        "category": memory.category,
                        "title": memory.title,
                        "persistence": memory.persistence
                    }
                )

                return True

        except Exception as e:
            self.logger.error(
                f"Failed to add memory to location {location_id}: {e}",
                extra={
                    "location_id": location_id,
                    "error": str(e)
                }
            )
            return False

    def update_memory_status(
        self,
        location_id: int,
        memory_title: str,
        new_status: str,
        superseded_by: Optional[str] = None,
        superseded_at_turn: Optional[int] = None,
        invalidation_reason: Optional[str] = None
    ) -> bool:
        """
        Update the status of an existing memory in file.

        Supports two workflows:
        1. Supersession: New memory replaces old memory (superseded_by provided)
        2. Standalone Invalidation: Memory proven false (invalidation_reason provided)

        This method:
        1. Reads entire Memories.md file
        2. Finds memory by title at specific location
        3. Updates status in header (**[CATEGORY - STATUS] Title**)
        4. Adds reference line (supersession or invalidation)
        5. Wraps text in strikethrough if SUPERSEDED
        6. Writes back to file atomically

        Args:
            location_id: Location where memory exists
            memory_title: Title of memory to update (exact match or substring)
            new_status: New status (typically MemoryStatus.SUPERSEDED)
            superseded_by: Optional title of new memory that superseded this one
            superseded_at_turn: Turn number when superseded/invalidated
            invalidation_reason: Optional reason for standalone invalidation

        Returns:
            True if successful, False if memory not found or update failed

        Raises:
            ValueError: If neither superseded_by nor invalidation_reason provided
        """
        memories_path = Path(self.config.zork_game_workdir) / "Memories.md"
        lock_path = str(memories_path) + ".lock"

        # Validate that exactly one of superseded_by or invalidation_reason is provided
        if not superseded_by and not invalidation_reason:
            self.logger.error("Either superseded_by or invalidation_reason must be provided")
            return False
        if superseded_by and invalidation_reason:
            self.logger.error("Cannot provide both superseded_by and invalidation_reason")
            return False

        # Validate non-empty strings
        if superseded_by is not None and not superseded_by.strip():
            self.logger.error(
                "superseded_by cannot be empty or whitespace",
                extra={
                    "location_id": location_id,
                    "memory_title": memory_title
                }
            )
            return False
        if invalidation_reason is not None and not invalidation_reason.strip():
            self.logger.error(
                "invalidation_reason cannot be empty or whitespace",
                extra={
                    "location_id": location_id,
                    "memory_title": memory_title
                }
            )
            return False

        # Validate turn number when superseding
        if new_status == MemoryStatus.SUPERSEDED:
            if superseded_at_turn is None:
                self.logger.error(
                    "superseded_at_turn is required when new_status is SUPERSEDED",
                    extra={
                        "location_id": location_id,
                        "memory_title": memory_title
                    }
                )
                return False
            if superseded_at_turn < 1:
                self.logger.error(
                    f"superseded_at_turn must be >= 1, got {superseded_at_turn}",
                    extra={
                        "location_id": location_id,
                        "memory_title": memory_title
                    }
                )
                return False

        try:
            # Acquire lock
            with FileLock(lock_path, timeout=10):
                # Backup before modification
                self._create_backup(memories_path)

                # Read entire file
                if not memories_path.exists():
                    self.logger.warning(f"Cannot update memory: Memories.md not found")
                    return False

                content = memories_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                # Determine reference line format based on workflow
                if invalidation_reason:
                    # Standalone invalidation
                    reference_line = f'[Invalidated at T{superseded_at_turn}: "{invalidation_reason}"]'
                else:
                    # Traditional supersession
                    reference_line = f'[Superseded at T{superseded_at_turn} by "{superseded_by}"]'

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
                            # Parse header with new regex groups
                            category = memory_match.group(1)
                            second_field = memory_match.group(2)
                            third_field = memory_match.group(3)
                            title = memory_match.group(4).strip()
                            metadata = memory_match.group(5).strip()

                            # Determine current persistence marker
                            persistence_marker = None
                            if third_field:
                                # Format is: CATEGORY - PERSISTENCE - STATUS
                                if second_field and second_field.upper() in ["CORE", "PERMANENT", "EPHEMERAL"]:
                                    persistence_marker = second_field.upper()
                            elif second_field:
                                # Format is: CATEGORY - (PERSISTENCE or STATUS)
                                if second_field.upper() in ["CORE", "PERMANENT", "EPHEMERAL"]:
                                    persistence_marker = second_field.upper()

                            # Check if this is the memory to update (exact or substring match)
                            if memory_title in title or title in memory_title:
                                # Found the memory - update header
                                memory_found = True
                                in_target_memory = True

                                # Format new header preserving persistence marker
                                if persistence_marker:
                                    updated_header = f"**[{category} - {persistence_marker} - {new_status}] {title}** *({metadata})*"
                                else:
                                    updated_header = f"**[{category} - {new_status}] {title}** *({metadata})*"
                                updated_lines.append(updated_header)

                                # Add reference line (supersession or invalidation)
                                updated_lines.append(reference_line)

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

                                    # Skip existing supersession/invalidation references
                                    if text_line.strip().startswith("[Superseded at") or text_line.strip().startswith("[Invalidated at"):
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
                    self.logger.warning(
                        f"Memory '{memory_title}' not found at location {location_id}",
                        extra={
                            "location_id": location_id,
                            "memory_title": memory_title
                        }
                    )
                    return False

                # Write updated content
                memories_path.write_text("\n".join(updated_lines), encoding="utf-8")

                # Log with appropriate context
                log_context = {
                    "location_id": location_id,
                    "memory_title": memory_title,
                    "new_status": new_status,
                }
                if superseded_by:
                    log_context["superseded_by"] = superseded_by
                if invalidation_reason:
                    log_context["invalidation_reason"] = invalidation_reason

                self.logger.info(
                    f"Updated memory status: '{memory_title}' → {new_status}",
                    extra=log_context
                )

                return True

        except Exception as e:
            self.logger.error(
                f"Failed to update memory status: {e}",
                extra={
                    "location_id": location_id,
                    "memory_title": memory_title,
                    "error": str(e)
                }
            )
            return False

    def _create_backup(self, memories_path: Path) -> None:
        """
        Create timestamped backup of existing Memories.md file.

        Args:
            memories_path: Path to Memories.md file
        """
        if memories_path.exists():
            backup_path = Path(str(memories_path) + ".backup")
            shutil.copy2(memories_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")

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
            {location_id: {"start": line_num, "end": line_num, "name": str}}
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
        Format single memory entry with persistence markers and status indicator.

        Format for ACTIVE:
            - CORE: **[CATEGORY - CORE] Title** *(EpX, TY, +/-Z)*
            - PERMANENT: **[CATEGORY - PERMANENT] Title** *(EpX, TY, +/-Z)*
            - No marker: **[CATEGORY] Title** *(EpX, TY, +/-Z)* (legacy/ephemeral)

        Format for TENTATIVE/SUPERSEDED:
            **[CATEGORY - PERSISTENCE - STATUS] Title** *(metadata)*
            [Superseded at TurnX by "Title"] or [Invalidated at TurnX: "reason"]
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

        # Build category string with persistence marker
        category_str = memory.category
        if memory.persistence in ["core", "permanent"]:
            category_str = f"{memory.category} - {memory.persistence.upper()}"

        # Format header with optional status
        if memory.status == MemoryStatus.ACTIVE:
            header = f"**[{category_str}] {memory.title}** *({metadata})*"
        else:
            header = f"**[{category_str} - {memory.status}] {memory.title}** *({metadata})*"

        # Format text (strikethrough if superseded)
        if memory.status == MemoryStatus.SUPERSEDED:
            text = f"~~{memory.text}~~"
        else:
            text = memory.text

        # Build lines
        lines = [header]

        # Add reference line if applicable (supersession or invalidation)
        if memory.status == MemoryStatus.SUPERSEDED:
            if memory.invalidation_reason:
                # Standalone invalidation
                lines.append(f"[Invalidated at T{memory.superseded_at_turn}: \"{memory.invalidation_reason}\"]")
            elif memory.superseded_by:
                # Traditional supersession
                lines.append(f"[Superseded at T{memory.superseded_at_turn} by \"{memory.superseded_by}\"]")

        lines.append(text)

        return "\n".join(lines)

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
