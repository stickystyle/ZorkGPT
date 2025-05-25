"""
ZorkExtractor module for extracting structured information from Zork game text.
"""

import json
from typing import Optional, List, Any, Dict, Type
from pydantic import BaseModel
from openai import OpenAI
import environs
import os

# Load environment variables
env = environs.Env()
env.read_env()

# Generic location fallbacks for location persistence
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


def create_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    schema = model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "strict": True,
            "schema": schema,
        },
    }


class ExtractorResponse(BaseModel):
    current_location_name: str
    exits: List[str]
    visible_objects: List[str]
    visible_characters: List[str]
    important_messages: List[str]
    in_combat: bool


class ZorkExtractor:
    """
    Handles extraction of structured information from Zork game text using LLMs.
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
        Initialize the ZorkExtractor.

        Args:
            model: Model name for information extraction
            client: OpenAI client instance (if None, creates new one)
            max_tokens: Maximum tokens for extraction
            temperature: Temperature for extraction model
            logger: Logger instance for tracking
            episode_id: Current episode ID for logging
        """
        self.model = model or env.str("INFO_EXT_MODEL", "qwen3-30b-a3b-mlx")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logger
        self.episode_id = episode_id

        # Initialize OpenAI client if not provided
        if client is None:
            self.client = OpenAI(
                base_url=env.str("CLIENT_BASE_URL", None),
                api_key=env.str("CLIENT_API_KEY", None),
            )
        else:
            self.client = client

        # Load system prompt
        self._load_system_prompt()

    def _load_system_prompt(self) -> None:
        """Load extractor system prompt from markdown files."""
        try:
            # Try to use enhanced extractor, fall back to original if not found
            try:
                with open("enhanced_extractor.md") as fh:
                    self.system_prompt = fh.read()
                if self.logger:
                    self.logger.info("Using enhanced extractor prompt")
            except FileNotFoundError:
                with open("extractor.md") as fh:
                    self.system_prompt = fh.read()
                if self.logger:
                    self.logger.info("Using original extractor prompt")

        except FileNotFoundError as e:
            if self.logger:
                self.logger.error(f"Failed to load extractor prompt file: {e}")
            raise

    def extract_info(
        self, game_text_from_zork: str, previous_location: Optional[str] = None
    ) -> Optional[ExtractorResponse]:
        """
        Extract structured information from Zork's game text with location persistence.

        Args:
            game_text_from_zork: The raw text output from the Zork game.
            previous_location: The previous location name for persistence when no location change occurs.

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

        user_prompt_content = (
            f"Game Text:\n```\n{game_text_from_zork}\n```\n\nJSON Output:\n```json\n"
        )
        user_prompt_content = r"\no_think " + user_prompt_content

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt_content},
        ]

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
                    # Find the JSON content between ```json and ```
                    start_marker = "```json"
                    end_marker = "```"
                    start_idx = response_content.find(start_marker) + len(start_marker)
                    end_idx = response_content.find(end_marker, start_idx)
                    if end_idx != -1:
                        json_content = response_content[start_idx:end_idx].strip()
                    else:
                        json_content = response_content[start_idx:].strip()
                elif "```" in response_content:
                    # Handle generic code blocks
                    start_idx = response_content.find("```") + 3
                    end_idx = response_content.find("```", start_idx)
                    if end_idx != -1:
                        json_content = response_content[start_idx:end_idx].strip()
                    else:
                        json_content = response_content[start_idx:].strip()
                else:
                    json_content = response_content.strip()

                parsed_data = json.loads(json_content)
                extracted_response = ExtractorResponse(**parsed_data)

                # Handle location persistence for "Unknown Location" or similar responses
                if (
                    extracted_response.current_location_name.lower()
                    in GENERIC_LOCATION_FALLBACKS
                    and previous_location
                    and previous_location.lower() not in GENERIC_LOCATION_FALLBACKS
                ):
                    # Log the location persistence
                    if self.logger:
                        self.logger.info(
                            f"Location persistence applied: '{extracted_response.current_location_name}' → '{previous_location}'",
                            extra={
                                "extras": {
                                    "event_type": "location_persistence",
                                    "episode_id": self.episode_id,
                                    "original_extraction": extracted_response.current_location_name,
                                    "persisted_location": previous_location,
                                    "game_text": game_text_from_zork[:100] + "..."
                                    if len(game_text_from_zork) > 100
                                    else game_text_from_zork,
                                }
                            },
                        )

                    # Create new response with persisted location
                    extracted_response = ExtractorResponse(
                        current_location_name=previous_location,
                        exits=extracted_response.exits,
                        visible_objects=extracted_response.visible_objects,
                        visible_characters=extracted_response.visible_characters,
                        important_messages=extracted_response.important_messages,
                        in_combat=extracted_response.in_combat,
                    )

                return extracted_response

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error parsing extractor response: {e}")
                    self.logger.error(f"Response content: {response_content}")

                # Fallback with location persistence
                return ExtractorResponse(
                    current_location_name=previous_location or "Extraction Failed",
                    exits=[],
                    visible_objects=[],
                    visible_characters=[],
                    important_messages=["Extraction failed"],
                    in_combat=False,
                )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting extracted info: {e}")

            # Fallback with location persistence
            return ExtractorResponse(
                current_location_name=previous_location or "LLM Request Failed",
                exits=[],
                visible_objects=[],
                visible_characters=[],
                important_messages=["LLM request failed"],
                in_combat=False,
            )

    def update_episode_id(self, episode_id: str) -> None:
        """Update the episode ID for logging purposes."""
        self.episode_id = episode_id
