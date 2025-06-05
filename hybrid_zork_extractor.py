"""
Hybrid Zork Extractor that combines structured parsing with LLM extraction.

This extractor uses the structured Zork output format to directly extract
room names, scores, and moves when available, while still using LLM extraction
for other information like exits, objects, characters, and important messages.

This approach maintains the LLM-first philosophy while taking advantage of the
structured data that's now available.
"""

import json
from typing import Optional, List, Any, Dict, Type, Tuple
from pydantic import BaseModel
from llm_client import LLMClientWrapper
import os

from structured_zork_parser import StructuredZorkParser, StructuredZorkResponse
from shared_utils import create_json_schema
from config import get_config, get_client_api_key

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


class ExtractorResponse(BaseModel):
    current_location_name: str
    exits: List[str]
    visible_objects: List[str]
    visible_characters: List[str]
    important_messages: List[str]
    in_combat: bool
    score: Optional[int] = None
    moves: Optional[int] = None


class HybridZorkExtractor:
    """
    Hybrid extractor that combines structured parsing with LLM extraction.

    Uses structured parsing for location/score/moves when available,
    and LLM extraction for detailed game state analysis.
    """

    def __init__(
        self,
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
        
        self.model = model or config.llm.info_ext_model
        self.max_tokens = max_tokens if max_tokens is not None else config.extractor_sampling.max_tokens
        self.temperature = temperature if temperature is not None else config.extractor_sampling.temperature
        self.top_p = top_p if top_p is not None else config.extractor_sampling.top_p
        self.top_k = top_k if top_k is not None else config.extractor_sampling.top_k
        self.min_p = min_p if min_p is not None else config.extractor_sampling.min_p
        self.logger = logger
        self.episode_id = episode_id
        
        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = config.logging.enable_prompt_logging
        
        # Initialize structured parser (always enabled for hybrid extractor)
        self.structured_parser = StructuredZorkParser()

        # Initialize LLM client if not provided
        if client is None:
            self.client = LLMClientWrapper(
                base_url=config.llm.get_base_url_for_model('info_ext'),
                api_key=get_client_api_key(),
            )
        else:
            self.client = client

        # Load system prompt for LLM extraction
        self._load_system_prompt()
        
        # Previous state tracking for context
        self.previous_combat_state = False
        self.previous_location = None

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
                self.logger.warning(f"Failed to log prompt to {filename}: {e}", extra={
                    "episode_id": self.episode_id
                })

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

    def extract_info(self, game_text_from_zork: str, previous_location: str = None) -> Optional[ExtractorResponse]:
        """
        Extract structured information from Zork game text using hybrid approach.
        
        Args:
            game_text_from_zork: Raw text from Zork
            previous_location: Previous location name for context
            
        Returns:
            Extracted information or None if extraction fails
        """
        try:
            # Get clean game text for LLM processing
            clean_game_text = self.get_clean_game_text(game_text_from_zork)
            
            # First, try structured parsing for key information
            structured_info = self._extract_structured_info(game_text_from_zork)
            
            # Enhanced location change detection
            location_changed = False
            location_change_reason = "No previous location provided"
            if previous_location:
                location_changed, location_change_reason = self._detect_location_change_from_response(
                    game_text_from_zork, previous_location
                )
            
            # Use LLM for comprehensive extraction with structured info as context
            extraction_prompt = self._build_extraction_prompt(
                clean_game_text, 
                previous_location, 
                structured_info,
                location_changed,
                location_change_reason
            )
            
            # Use proper system/user message structure
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": extraction_prompt}
            ]
            
            # Log the full prompt for evaluation if enabled
            self._log_prompt_to_file(messages, "extractor")
            
            llm_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                response_format=create_json_schema(ExtractorResponse),
            )
            
            if not llm_response:
                self.logger.warning(f"[{self.episode_id}] LLM extraction returned empty response", extra={
                    "episode_id": self.episode_id
                })
                return self._create_fallback_response(clean_game_text, previous_location, structured_info)
            
            # Extract content from the response
            response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Parse the LLM response
            parsed_response = self._parse_llm_response(response_content, previous_location, structured_info)
            
            if parsed_response:
                # Enhance with structured data where available
                parsed_response = self._enhance_with_structured_data(parsed_response, structured_info)
                
                # Log successful extraction
                self.logger.info(
                    f"[{self.episode_id}] Hybrid extraction successful: {parsed_response.current_location_name}",
                    extra={
                        "event_type": "hybrid_extraction_success",
                        "episode_id": self.episode_id,
                        "extracted_location": parsed_response.current_location_name,
                        "location_changed": location_changed,
                        "location_change_reason": location_change_reason,
                        "structured_available": bool(structured_info.get("current_location_name")),
                        "combat_state_transition": f"{self.previous_combat_state} -> {parsed_response.in_combat}",
                    }
                )
                
                # Update previous state tracking for next turn
                self.previous_combat_state = parsed_response.in_combat
                self.previous_location = parsed_response.current_location_name
                
                return parsed_response
            else:
                self.logger.warning(f"[{self.episode_id}] Failed to parse LLM extraction response", extra={
                    "episode_id": self.episode_id
                })
                fallback_response = self._create_fallback_response(clean_game_text, previous_location, structured_info)
                
                # Update previous state tracking even for fallback
                self.previous_combat_state = fallback_response.in_combat
                self.previous_location = fallback_response.current_location_name
                
                return fallback_response
        
        except Exception as e:
            self.logger.error(f"[{self.episode_id}] Extraction failed: {e}", extra={
                "episode_id": self.episode_id
            })
            # Pass structured_info to fallback even on exception
            structured_info = self._extract_structured_info(game_text_from_zork)
            fallback_response = self._create_fallback_response(game_text_from_zork, previous_location, structured_info)
            
            # Update previous state tracking even for exception case
            self.previous_combat_state = fallback_response.in_combat
            self.previous_location = fallback_response.current_location_name
            
            return fallback_response

    def _extract_structured_info(self, game_text_from_zork: str) -> Dict:
        """Extract structured information from the game text."""
        structured_result = self.structured_parser.parse_response(game_text_from_zork)
        return {
            "current_location_name": self.structured_parser.get_canonical_room_name(structured_result.room_name) if structured_result.has_structured_header else None,
            "exits": [],  # Structured parser doesn't extract exits, leave empty for LLM
            "visible_objects": [],  # Structured parser doesn't extract objects, leave empty for LLM
            "visible_characters": [],  # Structured parser doesn't extract characters, leave empty for LLM
            "important_messages": [structured_result.game_text] if structured_result.game_text else [],
            "in_combat": False,  # Structured parser doesn't detect combat, leave false for LLM
            "score": structured_result.score if structured_result.has_structured_header else None,
            "moves": structured_result.moves if structured_result.has_structured_header else None,
        }

    def _detect_location_change_from_response(self, game_response: str, previous_location: str) -> Tuple[bool, str]:
        """
        LLM-based location change detection that avoids hardcoded patterns.
        
        Args:
            game_response: The game's response text
            previous_location: The previous location name
            
        Returns:
            Tuple of (location_changed, new_location_or_reason)
        """
        try:
            # Use LLM to analyze the movement instead of hardcoded patterns
            analysis_prompt = self._build_movement_analysis_prompt(game_response, previous_location)
            
            movement_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=200,
            )
            
            if not movement_response:
                return False, "LLM movement analysis failed"
            
            # Extract content from the response
            response_content = movement_response.content if hasattr(movement_response, 'content') else str(movement_response)
            
            # Parse the LLM response
            return self._parse_movement_analysis(response_content)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[{self.episode_id}] Movement analysis failed: {e}", extra={
                    "episode_id": self.episode_id
                })
            # Fallback: assume no movement on error
            return False, f"Movement analysis error: {str(e)}"

    def _build_movement_analysis_prompt(self, game_response: str, previous_location: str) -> str:
        """Build a prompt for LLM-based movement analysis."""
        return f"""Analyze this text adventure game response to determine if the player's location changed.

Previous Location: {previous_location}

Game Response:
```
{game_response}
```

Please analyze whether the player moved to a new location and respond with JSON:
{{
    "location_changed": true/false,
    "new_location": "name of new location if moved, or null",
    "reason": "brief explanation of your analysis"
}}

Look for indicators such as:
- New room descriptions
- Movement success messages
- Location headers
- Transition descriptions
- Movement failure messages

Respond only with the JSON, no other text."""

    def _parse_movement_analysis(self, llm_response: str) -> Tuple[bool, str]:
        """Parse the LLM movement analysis response."""
        try:
            # Extract JSON from the response
            json_content = llm_response.strip()
            
            # Handle markdown code blocks if present
            if "```json" in json_content:
                start_idx = json_content.find("```json") + 7
                end_idx = json_content.find("```", start_idx)
                if end_idx != -1:
                    json_content = json_content[start_idx:end_idx].strip()
            elif "```" in json_content:
                start_idx = json_content.find("```") + 3
                end_idx = json_content.find("```", start_idx)
                if end_idx != -1:
                    json_content = json_content[start_idx:end_idx].strip()
            
            # Parse the JSON response
            analysis = json.loads(json_content)
            
            location_changed = analysis.get("location_changed", False)
            new_location = analysis.get("new_location")
            reason = analysis.get("reason", "No reason provided")
            
            if location_changed and new_location:
                return True, new_location
            elif location_changed:
                return True, f"Location changed: {reason}"
            else:
                return False, f"No movement: {reason}"
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[{self.episode_id}] Failed to parse movement analysis: {e}", extra={
                    "episode_id": self.episode_id
                })
                self.logger.warning(f"LLM response was: {llm_response}", extra={
                    "episode_id": self.episode_id
                })
            # Fallback: return the raw reason
            return False, f"Parse error: {llm_response[:100]}..."

    def _build_extraction_prompt(self, game_text: str, previous_location: str, structured_info: Dict, location_changed: bool, location_change_reason: str) -> str:
        """Build the extraction prompt for the LLM."""
        prompt_parts = []
        
        # Add context information
        if previous_location:
            prompt_parts.append(f"Previous Location: {previous_location}")
        
        # Add previous combat state context for persistence reasoning
        if self.previous_location or self.previous_combat_state:
            prompt_parts.append(f"Previous Combat State: {self.previous_combat_state}")
            if self.previous_location:
                prompt_parts.append(f"Previous Turn Location: {self.previous_location}")
        
        # Add movement analysis context
        prompt_parts.append(f"Movement Analysis: {location_change_reason}")
        
        # Add structured info if available
        if any(structured_info.values()):
            prompt_parts.append("Structured Parser Results:")
            for key, value in structured_info.items():
                if value:
                    prompt_parts.append(f"  {key}: {value}")
        
        # Add the game text
        prompt_parts.append(f"Game Text:\n```\n{game_text}\n```")
        
        # Simple instruction - let the system prompt handle the details
        prompt_parts.append("Please extract the key information from this game text and return it as JSON.")
        
        return "\n\n".join(prompt_parts)

    def _parse_llm_response(self, llm_response: str, previous_location: str, structured_info: Dict) -> Optional[ExtractorResponse]:
        """Parse the LLM response and return an ExtractorResponse."""
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in llm_response:
                start_marker = "```json"
                end_marker = "```"
                start_idx = llm_response.find(start_marker) + len(start_marker)
                end_idx = llm_response.find(end_marker, start_idx)
                if end_idx != -1:
                    json_content = llm_response[start_idx:end_idx].strip()
                else:
                    json_content = llm_response[start_idx:].strip()
            elif "```" in llm_response:
                start_idx = llm_response.find("```") + 3
                end_idx = llm_response.find("```", start_idx)
                if end_idx != -1:
                    json_content = llm_response[start_idx:end_idx].strip()
                else:
                    json_content = llm_response[start_idx:].strip()
            else:
                json_content = llm_response.strip()

            parsed_data = json.loads(json_content)
            return ExtractorResponse(**parsed_data)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error parsing LLM extractor response: {e}", extra={
                    "episode_id": self.episode_id
                })
                self.logger.error(f"Response content: {llm_response}", extra={
                    "episode_id": self.episode_id
                })
            return None

    def _enhance_with_structured_data(self, extracted_response: ExtractorResponse, structured_info: Dict) -> ExtractorResponse:
        """Enhance the extracted response with structured data where available."""
        if structured_info.get("current_location"):
            extracted_response.current_location_name = structured_info["current_location_name"]
        if structured_info.get("exits"):
            extracted_response.exits = structured_info["exits"]
        if structured_info.get("visible_objects"):
            extracted_response.visible_objects = structured_info["visible_objects"]
        if structured_info.get("visible_characters"):
            extracted_response.visible_characters = structured_info["visible_characters"]
        if structured_info.get("important_messages"):
            extracted_response.important_messages = structured_info["important_messages"]
        if structured_info.get("in_combat"):
            extracted_response.in_combat = structured_info["in_combat"]
        if structured_info.get("score"):
            extracted_response.score = structured_info["score"]
        if structured_info.get("moves"):
            extracted_response.moves = structured_info["moves"]
        return extracted_response

    def _create_fallback_response(self, game_text: str, previous_location: str, structured_info: Dict) -> ExtractorResponse:
        """Create a fallback response when extraction fails."""
        # Use structured location if available, otherwise fall back to previous location
        fallback_location = (
            structured_info.get("current_location_name") or 
            previous_location or 
            "Extraction Failed"
        )
        
        # For combat state, use structured info if available, otherwise maintain previous state
        fallback_combat_state = structured_info.get("in_combat")
        if fallback_combat_state is None:
            # No structured combat info, use previous state for persistence
            fallback_combat_state = self.previous_combat_state
        
        # Create response with best available information
        return ExtractorResponse(
            current_location_name=fallback_location,
            exits=[],
            visible_objects=[],
            visible_characters=[],
            important_messages=structured_info.get("important_messages", [game_text] if game_text else []),
            in_combat=fallback_combat_state,
            score=structured_info.get("score"),
            moves=structured_info.get("moves"),
        )

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
