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
from typing import Dict, List, Optional, Any

from filelock import FileLock
from pydantic import BaseModel, Field

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration
from shared_utils import create_json_schema
from llm_client import LLMClientWrapper


@dataclass
class Memory:
    """Represents a single location memory entry."""
    category: str              # SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE
    title: str                 # Short title of the memory
    episode: int               # Episode number
    turns: str                 # Turn range (e.g., "23-24" or "23")
    score_change: Optional[int]  # Score change (+5, +0, None if not specified)
    text: str                  # 1-2 sentence synthesized insight


class MemorySynthesisResponse(BaseModel):
    """LLM response for memory synthesis."""
    model_config = {"strict": True}

    should_remember: bool
    category: str  # SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE
    memory_title: str
    memory_text: str
    reasoning: str = ""  # Optional reasoning for debugging


class SimpleMemoryManager(BaseManager):
    """
    Manages location-based memory system for ZorkGPT.

    Responsibilities:
    - Parse Memories.md file format on initialization
    - Maintain in-memory cache of memories per location ID
    - Gracefully handle missing, empty, or corrupted files
    - Provide memory retrieval interface (Phase 2)

    Cache Structure:
    - memory_cache: Dict[int, List[Memory]] - location ID to list of memories
    """

    # Regex patterns for parsing Memories.md format
    LOCATION_HEADER_PATTERN = re.compile(r"^## Location (\d+): (.+)$")
    MEMORY_ENTRY_PATTERN = re.compile(r"^\*\*\[(\w+)\] (.+?)\*\* \*\((.*?)\)\*$")

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
                    title = memory_match.group(2).strip()
                    metadata = memory_match.group(3).strip()

                    current_memory_header = (category, title, metadata)
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
            memory_header: Tuple of (category, title, metadata)
            text_lines: List of text lines for memory content
        """
        category, title, metadata = memory_header

        try:
            # Parse metadata: "Ep1, T23-24, +0" or "Ep1, T100"
            episode, turns, score_change = self._parse_metadata(metadata)

            # Join text lines into single string
            text = " ".join(text_lines)

            # Create Memory object
            memory = Memory(
                category=category,
                title=title,
                episode=episode,
                turns=turns,
                score_change=score_change,
                text=text
            )

            # Add to cache
            if location_id not in self.memory_cache:
                self.memory_cache[location_id] = []

            self.memory_cache[location_id].append(memory)

            self.log_debug(
                f"Added memory: [{category}] {title} at location {location_id}",
                location_id=location_id,
                category=category,
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
        Format single memory entry.

        Format: **[CATEGORY] Title** *(EpX, TY, +/-Z)*
                text

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

        # Format entry
        lines = [
            f"**[{memory.category}] {memory.title}** *({metadata})*",
            memory.text
        ]

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

OUTPUT FORMAT:
- should_remember: true/false (MUST be false if duplicate)
- category: SUCCESS/FAILURE/DISCOVERY/DANGER/NOTE
- memory_title: 3-6 words, evergreen (no "reveals" vs "provides" variations)
- memory_text: 1-2 sentences, actionable insight
- reasoning: Explain semantic comparison with existing memories

Return JSON only."""

            # Call LLM with structured output
            llm_response = self.llm_client.chat.completions.create(
                model=self.config.memory_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.memory_sampling.get('temperature', 0.3),
                max_tokens=self.config.memory_sampling.get('max_tokens', 1000),
                name="SimpleMemory",
                response_format=create_json_schema(MemorySynthesisResponse)
            )

            # Parse response
            synthesis = MemorySynthesisResponse.model_validate_json(llm_response.content)

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
            text=synthesis.memory_text
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

        Args:
            location_id: Location ID to retrieve memories for

        Returns:
            Formatted string with all memories for location
        """
        if location_id not in self.memory_cache:
            return ""

        memories = self.memory_cache[location_id]
        if not memories:
            return ""

        # Format memories as text
        lines = []
        for mem in memories:
            lines.append(f"[{mem.category}] {mem.title}: {mem.text}")

        return "\n".join(lines)
