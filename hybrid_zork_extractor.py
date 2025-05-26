"""
Hybrid Zork Extractor that combines structured parsing with LLM extraction.

This extractor uses the new structured Zork output format to directly extract
room names, scores, and moves when available, while still using LLM extraction
for other information like exits, objects, characters, and important messages.

This approach maintains the LLM-first philosophy while taking advantage of the
structured data that's now available.
"""

import json
from typing import Optional, List, Any, Dict, Type
from pydantic import BaseModel
from openai import OpenAI
import environs
import os

from structured_zork_parser import StructuredZorkParser, StructuredZorkResponse
from zork_extractor import (
    ExtractorResponse,
    GENERIC_LOCATION_FALLBACKS,
    create_json_schema,
)

# Load environment variables
env = environs.Env()
env.read_env()


class HybridZorkExtractor:
    """
    Hybrid extractor that combines structured parsing with LLM extraction.

    Uses structured parsing for location/score/moves when available,
    and LLM extraction for detailed game state analysis.
    """

    def __init__(
        self,
        model: str = None,
        client: Optional[OpenAI] = None,
        max_tokens: int = 300,
        temperature: float = 0.1,
        logger=None,
        episode_id: str = "unknown",
    ):
        """
        Initialize the HybridZorkExtractor.

        Args:
            model: Model name for LLM extraction (when needed)
            client: OpenAI client instance (if None, creates new one)
            max_tokens: Maximum tokens for LLM extraction
            temperature: Temperature for LLM extraction
            logger: Logger instance for tracking
            episode_id: Current episode ID for logging

        """
        self.model = model or env.str("INFO_EXT_MODEL", "qwen3-30b-a3b-mlx")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logger
        self.episode_id = episode_id
        
        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = env.bool("ENABLE_PROMPT_LOGGING", False)
        
        # Initialize structured parser (always enabled for hybrid extractor)
        self.structured_parser = StructuredZorkParser()

        # Initialize OpenAI client if not provided
        if client is None:
            self.client = OpenAI(
                base_url=env.str("CLIENT_BASE_URL", None),
                api_key=env.str("CLIENT_API_KEY", None),
            )
        else:
            self.client = client

        # Load system prompt for LLM extraction
        self._load_system_prompt()

    def _log_prompt_to_file(self, messages: List[Dict], prefix: str = "extractor") -> None:
        """Log the full prompt to a temporary file for evaluation."""
        if not self.enable_prompt_logging:
            return
            
        self.prompt_counter += 1
        filename = f"tmp/{prefix}_{self.prompt_counter:03d}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== {prefix.upper()} PROMPT #{self.prompt_counter} ===\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"Max Tokens: {self.max_tokens}\n")
                f.write(f"Episode ID: {self.episode_id}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, message in enumerate(messages):
                    f.write(f"--- MESSAGE {i+1} ({message['role'].upper()}) ---\n")
                    f.write(message['content'])
                    f.write("\n\n")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to log prompt to {filename}: {e}")

    def _load_system_prompt(self) -> None:
        """Load the system prompt for LLM extraction."""
        try:
            with open("extractor.md", "r") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            # Ultimate fallback if extractor.md is missing
            self.system_prompt = """You are an expert data extraction assistant for a text adventure game.
Extract key information from the game text and return it as JSON with these fields:
- current_location_name: The name of the current room/area
- exits: List of available exits
- visible_objects: List of significant objects
- visible_characters: List of characters/creatures
- important_messages: List of important messages
- in_combat: Boolean indicating combat status"""

    def extract_info(
        self, game_text_from_zork: str, previous_location: Optional[str] = None
    ) -> Optional[ExtractorResponse]:
        """
        Extract structured information using hybrid approach.

        Args:
            game_text_from_zork: The raw text output from the Zork game.
            previous_location: The previous location name for persistence.

        Returns:
            ExtractorResponse containing the extracted information, or None if extraction fails.
        """
        if not game_text_from_zork or not game_text_from_zork.strip():
            return ExtractorResponse(
                current_location_name=previous_location or "Unknown (Empty Input)",
                exits=[],
                visible_objects=[],
                visible_characters=[],
                important_messages=["Received empty game text."],
                in_combat=False,
            )

        # Step 1: Try structured parsing first
        structured_result = self.structured_parser.parse_response(game_text_from_zork)

        # Step 2: Determine location using structured parser or fallback
        current_location_name = None
        if (
            structured_result
            and structured_result.has_structured_header
            and structured_result.room_name
        ):
            # Use structured parser result
            current_location_name = self.structured_parser.get_canonical_room_name(
                structured_result.room_name
            )
            extraction_source = "Structured Parser"

            # Log successful structured extraction
            if self.logger:
                self.logger.info(
                    f"Location extracted via structured parser: {current_location_name}",
                    extra={
                        "extras": {
                            "event_type": "structured_location_extraction",
                            "episode_id": self.episode_id,
                            "location": current_location_name,
                            "score": structured_result.score,
                            "moves": structured_result.moves,
                        }
                    },
                )
        else:
            # No structured location available, will need LLM or fallback
            extraction_source = "LLM Fallback"

        # Step 3: Prepare game text for LLM extraction (remove structured header if present)
        game_text_for_llm = game_text_from_zork
        if structured_result and structured_result.has_structured_header:
            # Use the clean game text without the structured header for LLM analysis
            game_text_for_llm = structured_result.game_text

        # Use LLM extraction for detailed analysis
        # We still use LLM for exits, objects, characters, and messages
        # even when we have structured location data
        llm_extraction = self._extract_with_llm(game_text_for_llm, previous_location)

        if llm_extraction:
            # If we got location from structured parser, use that; otherwise use LLM result
            if current_location_name:
                # Use structured location but LLM details
                final_response = ExtractorResponse(
                    current_location_name=current_location_name,
                    exits=llm_extraction.exits,
                    visible_objects=llm_extraction.visible_objects,
                    visible_characters=llm_extraction.visible_characters,
                    important_messages=llm_extraction.important_messages,
                    in_combat=llm_extraction.in_combat,
                )
            else:
                # Use LLM result entirely
                final_response = llm_extraction
                current_location_name = llm_extraction.current_location_name
                extraction_source = "LLM Only"
        else:
            # LLM extraction failed
            if current_location_name:
                # We have structured location but no LLM details
                final_response = ExtractorResponse(
                    current_location_name=current_location_name,
                    exits=[],
                    visible_objects=[],
                    visible_characters=[],
                    important_messages=[structured_result.game_text]
                    if structured_result.game_text
                    else [],
                    in_combat=False,
                )
                extraction_source = "Structured Only"
            else:
                # Complete extraction failure
                final_response = ExtractorResponse(
                    current_location_name=previous_location or "Extraction Failed",
                    exits=[],
                    visible_objects=[],
                    visible_characters=[],
                    important_messages=["Both structured and LLM extraction failed"],
                    in_combat=False,
                )
                extraction_source = "Fallback"

        # Step 4: Apply location persistence if needed
        if (
            final_response.current_location_name.lower() in GENERIC_LOCATION_FALLBACKS
            and previous_location
            and previous_location.lower() not in GENERIC_LOCATION_FALLBACKS
        ):
            if self.logger:
                self.logger.info(
                    f"Location persistence applied: '{final_response.current_location_name}' → '{previous_location}'",
                    extra={
                        "extras": {
                            "event_type": "location_persistence",
                            "episode_id": self.episode_id,
                            "original_extraction": final_response.current_location_name,
                            "persisted_location": previous_location,
                            "extraction_source": extraction_source,
                        }
                    },
                )

            final_response = ExtractorResponse(
                current_location_name=previous_location,
                exits=final_response.exits,
                visible_objects=final_response.visible_objects,
                visible_characters=final_response.visible_characters,
                important_messages=final_response.important_messages,
                in_combat=final_response.in_combat,
            )

        # Log final extraction result
        if self.logger:
            self.logger.debug(
                f"Hybrid extraction complete: {final_response.current_location_name} (source: {extraction_source})",
                extra={
                    "extras": {
                        "event_type": "hybrid_extraction_complete",
                        "episode_id": self.episode_id,
                        "location": final_response.current_location_name,
                        "extraction_source": extraction_source,
                        "has_structured_data": structured_result.has_structured_header
                        if structured_result
                        else False,
                    }
                },
            )

        return final_response

    def _extract_with_llm(
        self, game_text_from_zork: str, previous_location: Optional[str] = None
    ) -> Optional[ExtractorResponse]:
        """
        Extract information using LLM.

        Args:
            game_text_from_zork: The raw text output from the Zork game.
            previous_location: The previous location name for context.

        Returns:
            ExtractorResponse containing the extracted information, or None if extraction fails.
        """
        user_prompt_content = (
            f"Game Text:\n```\n{game_text_from_zork}\n```\n\nJSON Output:\n```json\n"
        )
        user_prompt_content = r"\no_think " + user_prompt_content

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt_content},
        ]

        # Log the full prompt for evaluation
        self._log_prompt_to_file(messages, "extractor")

        try:
            # Use structured output for OpenAI models, no response_format for others
            if "gpt-" in self.model.lower() or "o1-" in self.model.lower():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=create_json_schema(ExtractorResponse),
                    extra_headers={
                        "X-Title": "ZorkGPT",
                    },
                )
            else:
                # For non-OpenAI models, rely on prompt instructions for JSON format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_headers={
                        "X-Title": "ZorkGPT",
                    },
                )

            response_content = response.choices[0].message.content

            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in response_content:
                    start_marker = "```json"
                    end_marker = "```"
                    start_idx = response_content.find(start_marker) + len(start_marker)
                    end_idx = response_content.find(end_marker, start_idx)
                    if end_idx != -1:
                        json_content = response_content[start_idx:end_idx].strip()
                    else:
                        json_content = response_content[start_idx:].strip()
                elif "```" in response_content:
                    start_idx = response_content.find("```") + 3
                    end_idx = response_content.find("```", start_idx)
                    if end_idx != -1:
                        json_content = response_content[start_idx:end_idx].strip()
                    else:
                        json_content = response_content[start_idx:].strip()
                else:
                    json_content = response_content.strip()

                parsed_data = json.loads(json_content)
                return ExtractorResponse(**parsed_data)

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error parsing LLM extractor response: {e}")
                    self.logger.error(f"Response content: {response_content}")
                return None

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting LLM extracted info: {e}")
            return None

    def update_episode_id(self, episode_id: str) -> None:
        """Update the episode ID for logging purposes."""
        self.episode_id = episode_id

    def get_score_and_moves(
        self, game_text_from_zork: str
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Extract score and moves directly from structured response.

        Args:
            game_text_from_zork: The raw text output from the Zork game.

        Returns:
            Tuple of (score, moves) if available, (None, None) otherwise.
        """
        return self.structured_parser.extract_score_and_moves(game_text_from_zork)

    def get_clean_game_text(self, game_text_from_zork: str) -> str:
        """
        Get the game text without the structured header for display purposes.

        Args:
            game_text_from_zork: The raw text output from the Zork game.

        Returns:
            Clean game text without structured header, or original text if no header found.
        """
        structured_result = self.structured_parser.parse_response(game_text_from_zork)
        if structured_result.has_structured_header:
            return structured_result.game_text

        return game_text_from_zork
