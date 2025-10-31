"""
Jericho-based Zork Extractor using object tree for structured data.

This extractor uses Jericho's Z-machine object tree to directly extract
inventory, location, and visible objects WITHOUT regex parsing. LLM is used
only for exits, combat detection, and important message extraction.

This is the Phase 2 implementation with NO backwards compatibility.
"""

import json
from typing import Optional, List
from pydantic import BaseModel
from llm_client import LLMClientWrapper

from game_interface.core.jericho_interface import JerichoInterface
from shared_utils import create_json_schema, strip_markdown_json_fences
from config import get_config, get_client_api_key

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

# Generic location fallbacks for location persistence (used by movement_analyzer)
GENERIC_LOCATION_FALLBACKS = {
    "unknown location",
    "unknown area",
    "unclear area",
    "unspecified location",
    "same area",
    "same place",
    "no specific location",
    "not applicable",
    "na",
    "n/a",
    "",  # Empty string also a fallback
}


class ExtractorResponse(BaseModel):
    current_location_name: str
    exits: List[str]
    visible_objects: List[str]
    visible_characters: List[str]
    inventory: List[str]
    important_messages: List[str]
    in_combat: bool
    score: Optional[int] = None
    moves: Optional[int] = None


class HybridZorkExtractor:
    """
    Jericho-based extractor that uses object tree for structured data.

    Uses Jericho's Z-machine object tree for:
    - Inventory (from get_inventory_structured)
    - Location name (from get_location_structured)
    - Visible objects (from object tree traversal)
    - Score/moves (from get_score)

    Uses LLM only for:
    - Exits (direction parsing is game-state dependent)
    - Combat detection (requires semantic understanding)
    - Important messages (requires semantic filtering)
    """

    def __init__(
        self,
        jericho_interface: JerichoInterface,
        model: str = None,
        client: Optional[LLMClientWrapper] = None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        logger=None,
        episode_id: str = "unknown",
    ):
        """
        Initialize the HybridZorkExtractor.

        Args:
            jericho_interface: JerichoInterface instance for accessing game state
            model: Model name for LLM extraction (when needed)
            client: OpenAI client instance (if None, creates new one)
            max_tokens: Maximum tokens for LLM extraction
            temperature: Temperature for LLM extraction
            top_p: Top-p nucleus sampling
            top_k: Top-k sampling
            min_p: Minimum probability sampling
            logger: Logger instance for tracking
            episode_id: Current episode ID for logging
        """
        config = get_config()

        self.jericho = jericho_interface
        self.model = model or config.llm.info_ext_model
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else config.extractor_sampling.max_tokens
        )
        self.temperature = (
            temperature
            if temperature is not None
            else config.extractor_sampling.temperature
        )
        self.top_p = top_p if top_p is not None else config.extractor_sampling.top_p
        self.top_k = top_k if top_k is not None else config.extractor_sampling.top_k
        self.min_p = min_p if min_p is not None else config.extractor_sampling.min_p
        self.logger = logger
        self.episode_id = episode_id

        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = config.logging.enable_prompt_logging

        # Initialize LLM client if not provided
        if client is None:
            self.client = LLMClientWrapper(
                base_url=config.llm.get_base_url_for_model("info_ext"),
                api_key=get_client_api_key(),
            )
        else:
            self.client = client

        # Load system prompt for LLM extraction
        self._load_system_prompt()

        # Previous state tracking for context
        self.previous_combat_state = False
        self.previous_location = None

    def _log_prompt_to_file(
        self, messages: List[dict], prefix: str = "extractor"
    ) -> None:
        """Log the full prompt to a temporary file for evaluation."""
        if not self.enable_prompt_logging:
            return

        self.prompt_counter += 1
        filename = f"tmp/{prefix}_{self.prompt_counter:03d}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"=== {prefix.upper()} PROMPT #{self.prompt_counter} ===\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"Max Tokens: {self.max_tokens}\n")
                f.write(f"Episode ID: {self.episode_id}\n")
                f.write("=" * 50 + "\n\n")

                for i, message in enumerate(messages):
                    f.write(f"--- MESSAGE {i + 1} ({message['role'].upper()}) ---\n")
                    f.write(message["content"])
                    f.write("\n\n")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to log prompt to {filename}: {e}",
                    extra={"episode_id": self.episode_id},
                )

    def _load_system_prompt(self) -> None:
        """Load the system prompt for LLM extraction."""
        try:
            with open("extractor.md", "r") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            # Fallback if extractor.md is missing
            self.system_prompt = """You are an expert data extraction assistant for a text adventure game.
Extract key information from the game text and return it as JSON with these fields:
- exits: List of available exits (directions like north, south, east, west, etc.)
- important_messages: List of important messages (gameplay events, not flavor text)
- in_combat: Boolean indicating combat status"""

    @observe(name="extractor-extract-information")
    def extract_info(
        self, game_text_from_zork: str, previous_location: str = None
    ) -> Optional[ExtractorResponse]:
        """
        Extract structured information from Zork using hybrid approach.

        Uses Jericho object tree for location, inventory, visible objects.
        Uses LLM for exits, combat detection, and important messages.

        Args:
            game_text_from_zork: Raw text from Zork (for LLM context)
            previous_location: Previous location name for context

        Returns:
            Extracted information or None if extraction fails
        """
        try:
            # Extract structured data from Jericho object tree
            location_name = self._get_location_from_jericho()
            inventory = self._get_inventory_from_jericho()
            visible_objects = self._get_visible_objects_from_jericho()
            visible_characters = self._get_visible_characters_from_jericho()
            score, moves = self._get_score_from_jericho()

            # Use LLM for exits, combat, and important messages
            llm_extracted = self._extract_with_llm(
                game_text_from_zork, location_name, previous_location
            )

            # Combine structured and LLM data
            extracted_info = ExtractorResponse(
                current_location_name=location_name,
                exits=llm_extracted.get("exits", []),
                visible_objects=visible_objects,
                visible_characters=visible_characters,
                inventory=inventory,
                important_messages=llm_extracted.get("important_messages", []),
                in_combat=llm_extracted.get("in_combat", False),
                score=score,
                moves=moves,
            )

            # Log successful extraction
            if self.logger:
                self.logger.info(
                    f"Jericho extraction successful: {location_name}",
                    extra={
                        "event_type": "jericho_extraction_success",
                        "episode_id": self.episode_id,
                        "extracted_location": location_name,
                        "visible_objects_count": len(visible_objects),
                        "visible_characters_count": len(visible_characters),
                        "exits_count": len(llm_extracted.get("exits", [])),
                        "in_combat": llm_extracted.get("in_combat", False),
                    },
                )

            # Update previous state tracking
            self.previous_combat_state = llm_extracted.get("in_combat", False)
            self.previous_location = location_name

            return extracted_info

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"[{self.episode_id}] Extraction failed: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return self._create_fallback_response(
                game_text_from_zork, previous_location
            )

    def _get_location_from_jericho(self) -> str:
        """Get current location from Jericho object tree."""
        try:
            location_obj = self.jericho.get_location_structured()
            return location_obj.name if location_obj else "Unknown Location"
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to get location from Jericho: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return "Unknown Location"

    def _get_inventory_from_jericho(self) -> List[str]:
        """Get inventory from Jericho object tree."""
        try:
            inventory_objects = self.jericho.get_inventory_structured()
            return [obj.name for obj in inventory_objects]
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to get inventory from Jericho: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return []

    def _get_visible_objects_from_jericho(self) -> List[str]:
        """
        Get visible objects from Jericho object tree.

        Traverses the Z-machine object tree to find objects in the current location.
        Objects are considered visible if they are children of the current location.
        """
        try:
            location_obj = self.jericho.get_location_structured()
            if not location_obj:
                return []

            visible_objects = []
            all_objects = self.jericho.get_all_objects()

            # Find objects whose parent is the current location
            for obj in all_objects:
                if obj.parent == location_obj.num and obj.name:
                    # Filter out the player object and empty names
                    if obj.name.strip() and not obj.name.lower() == "cretin":
                        visible_objects.append(obj.name)

            return visible_objects

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to get visible objects from Jericho: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return []

    def _get_visible_characters_from_jericho(self) -> List[str]:
        """
        Get visible characters/NPCs from Jericho object tree.

        Characters are typically distinguished by having certain attributes
        or properties in the Z-machine. This is a heuristic approach.
        """
        try:
            location_obj = self.jericho.get_location_structured()
            if not location_obj:
                return []

            characters = []
            all_objects = self.jericho.get_all_objects()

            # Character detection heuristics for Zork
            character_keywords = [
                "thief",
                "troll",
                "cyclops",
                "bat",
                "grue",
                "ghost",
                "spirit",
                "demon",
            ]

            for obj in all_objects:
                if obj.parent == location_obj.num and obj.name:
                    obj_name_lower = obj.name.lower()
                    # Check if object name contains character keywords
                    if any(keyword in obj_name_lower for keyword in character_keywords):
                        characters.append(obj.name)

            return characters

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to get visible characters from Jericho: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return []

    def _get_score_from_jericho(self) -> tuple[Optional[int], Optional[int]]:
        """Get score and moves from Jericho."""
        try:
            score, max_score = self.jericho.get_score()
            # Jericho doesn't track moves separately, so return None for moves
            return score, None
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to get score from Jericho: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return None, None

    def _extract_with_llm(
        self, game_text: str, current_location: str, previous_location: str = None
    ) -> dict:
        """
        Use LLM to extract exits, combat status, and important messages.

        This is the ONLY place where LLM is used for extraction.
        """
        try:
            # Build extraction prompt
            extraction_prompt = self._build_llm_extraction_prompt(
                game_text, current_location, previous_location
            )

            # Use proper system/user message structure with caching
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
                {"role": "user", "content": extraction_prompt},
            ]

            # Log the full prompt for evaluation if enabled
            self._log_prompt_to_file(messages, "extractor")

            # Define minimal schema for LLM extraction
            class LLMExtraction(BaseModel):
                exits: List[str]
                important_messages: List[str]
                in_combat: bool

            llm_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                name="Extractor",
                response_format=create_json_schema(LLMExtraction),
            )

            if not llm_response:
                return {"exits": [], "important_messages": [], "in_combat": False}

            # Extract content from the response
            response_content = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )

            # Parse the LLM response
            parsed_response = self._parse_llm_response(response_content)
            return parsed_response

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"LLM extraction failed: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return {
                "exits": [],
                "important_messages": [],
                "in_combat": self.previous_combat_state,
            }

    def _build_llm_extraction_prompt(
        self, game_text: str, current_location: str, previous_location: str = None
    ) -> str:
        """Build the extraction prompt for the LLM."""
        prompt_parts = []

        # Add context information
        if previous_location:
            prompt_parts.append(f"Previous Location: {previous_location}")
        prompt_parts.append(f"Current Location: {current_location}")

        # Add previous combat state context
        if self.previous_location or self.previous_combat_state:
            prompt_parts.append(f"Previous Combat State: {self.previous_combat_state}")

        # Add the game text
        prompt_parts.append(f"Game Text:\n```\n{game_text}\n```")

        # Simple instruction
        prompt_parts.append(
            "Extract ONLY the following information from the game text:\n"
            "1. exits: List of available directions (north, south, east, west, up, down, etc.)\n"
            "2. important_messages: List of significant gameplay events (not flavor text)\n"
            "3. in_combat: Boolean indicating if the player is currently in combat\n\n"
            "Return as JSON with these three fields."
        )

        return "\n\n".join(prompt_parts)

    def _parse_llm_response(self, llm_response: str) -> dict:
        """Parse the LLM response and return a dict."""
        try:
            # Strip markdown fences if present (some LLMs wrap JSON in ```json ... ```)
            json_content = strip_markdown_json_fences(llm_response)

            parsed_data = json.loads(json_content)
            return {
                "exits": parsed_data.get("exits", []),
                "important_messages": parsed_data.get("important_messages", []),
                "in_combat": parsed_data.get("in_combat", False),
            }

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error parsing LLM extractor response: {e}",
                    extra={"episode_id": self.episode_id},
                )
            return {
                "exits": [],
                "important_messages": [],
                "in_combat": self.previous_combat_state,
            }

    def _create_fallback_response(
        self, game_text: str, previous_location: str
    ) -> ExtractorResponse:
        """Create a fallback response when extraction fails."""
        # Try to get location from Jericho even in fallback
        try:
            fallback_location = self._get_location_from_jericho()
        except Exception:
            fallback_location = previous_location or "Extraction Failed"

        # Create minimal response
        return ExtractorResponse(
            current_location_name=fallback_location,
            exits=[],
            visible_objects=[],
            visible_characters=[],
            inventory=[],
            important_messages=[game_text] if game_text else [],
            in_combat=self.previous_combat_state,
            score=None,
            moves=None,
        )

    def update_episode_id(self, episode_id: str) -> None:
        """Update the episode ID for logging purposes."""
        self.episode_id = episode_id

    def get_score_and_moves(
        self, game_text_from_zork: str
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Extract score and moves directly from Jericho.

        Args:
            game_text_from_zork: Unused, kept for backwards compatibility

        Returns:
            Tuple of (score, moves) if available, (None, None) otherwise.
        """
        return self._get_score_from_jericho()

    def get_clean_game_text(self, game_text_from_zork: str) -> str:
        """
        Get the game text as-is (no structured header to remove with Jericho).

        Args:
            game_text_from_zork: The raw text output from the Zork game.

        Returns:
            Clean game text (same as input with Jericho)
        """
        return game_text_from_zork
