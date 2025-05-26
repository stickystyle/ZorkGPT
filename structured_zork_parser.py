"""
Structured Zork Parser for parsing the new Zork output format.

The new Zork version provides structured output in the format:
> Room Name                                    Score: X        Moves: Y

Game response text here...

This parser extracts the room name, score, and move counter directly from this format,
eliminating the need for LLM-based location extraction in most cases.
"""

import re
from typing import Optional, Tuple, List
from pydantic import BaseModel


class StructuredZorkResponse(BaseModel):
    """Structured representation of Zork's response."""

    room_name: Optional[str] = None
    score: Optional[int] = None
    moves: Optional[int] = None
    game_text: str = ""
    has_structured_header: bool = False


class StructuredZorkParser:
    """
    Parser for the new structured Zork output format.

    This parser can extract room names, scores, and move counts directly
    from the Zork response header, reducing reliance on LLM extraction.
    """

    def __init__(self):
        # Regex pattern to match the structured header
        # Format: > Room Name                                    Score: X        Moves: Y
        self.header_pattern = re.compile(
            r"^>\s*(.+?)\s+Score:\s*(\d+)\s+Moves:\s*(\d+)\s*$", re.MULTILINE
        )

        # Alternative pattern for responses without room names (like error messages)
        self.simple_pattern = re.compile(r"^>(.+)$", re.MULTILINE)

    def parse_response(self, zork_response: str) -> StructuredZorkResponse:
        """
        Parse a Zork response and extract structured information.

        Args:
            zork_response: Raw response text from Zork

        Returns:
            StructuredZorkResponse with extracted information
        """
        if not zork_response or not zork_response.strip():
            return StructuredZorkResponse(game_text="")

        # Try to match the structured header pattern first
        header_match = self.header_pattern.search(zork_response)

        if header_match:
            room_name = header_match.group(1).strip()
            score = int(header_match.group(2))
            moves = int(header_match.group(3))

            # Extract the game text (everything after the header line)
            lines = zork_response.split("\n")
            header_line_found = False
            game_text_lines = []

            for line in lines:
                if header_line_found:
                    game_text_lines.append(line)
                elif self.header_pattern.match(line):
                    header_line_found = True

            game_text = "\n".join(game_text_lines).strip()

            return StructuredZorkResponse(
                room_name=room_name,
                score=score,
                moves=moves,
                game_text=game_text,
                has_structured_header=True,
            )

        # If no structured header, check for simple command responses
        simple_match = self.simple_pattern.search(zork_response)
        if simple_match:
            # This is likely an error message or simple response
            return StructuredZorkResponse(
                game_text=zork_response.strip(), has_structured_header=False
            )

        # Fallback: treat entire response as game text
        return StructuredZorkResponse(
            game_text=zork_response.strip(), has_structured_header=False
        )

    def extract_room_name(self, zork_response: str) -> Optional[str]:
        """
        Quick extraction of just the room name from a Zork response.

        Args:
            zork_response: Raw response text from Zork

        Returns:
            Room name if found, None otherwise
        """
        parsed = self.parse_response(zork_response)
        return parsed.room_name

    def extract_score_and_moves(
        self, zork_response: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Quick extraction of score and move count from a Zork response.

        Args:
            zork_response: Raw response text from Zork

        Returns:
            Tuple of (score, moves) if found, (None, None) otherwise
        """
        parsed = self.parse_response(zork_response)
        return parsed.score, parsed.moves

    def is_structured_response(self, zork_response: str) -> bool:
        """
        Check if the response has the new structured format.

        Args:
            zork_response: Raw response text from Zork

        Returns:
            True if response has structured header, False otherwise
        """
        return self.parse_response(zork_response).has_structured_header

    def get_canonical_room_name(self, room_name: str) -> str:
        """
        Convert room name to canonical format for consistency.

        Args:
            room_name: Raw room name from Zork

        Returns:
            Canonical room name
        """
        if not room_name:
            return "Unknown Location"

        # Clean up the room name
        canonical = room_name.strip()

        # Handle common patterns
        if canonical.lower().startswith("west of"):
            canonical = "West Of " + canonical[7:].strip().title()
        elif canonical.lower().startswith("east of"):
            canonical = "East Of " + canonical[7:].strip().title()
        elif canonical.lower().startswith("north of"):
            canonical = "North Of " + canonical[8:].strip().title()
        elif canonical.lower().startswith("south of"):
            canonical = "South Of " + canonical[8:].strip().title()
        else:
            # Title case for other room names
            canonical = canonical.title()

        return canonical


def test_parser():
    """Test function to verify the parser works correctly."""
    parser = StructuredZorkParser()

    # Test cases
    test_cases = [
        # Structured response with room name
        "> West of House                                    Score: 0        Moves: 2\n\nThe small mailbox is closed.",
        # Error message
        '>I don\'t know the word "east".',
        # Simple response
        ">Taken.",
        # Empty response
        "",
        # Multi-line game text
        "> Forest Clearing                                  Score: 5        Moves: 15\n\nYou are in a forest clearing.\nThere is a path to the north.\nA small cottage is visible to the east.",
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i + 1}:")
        print(f"Input: {repr(test_case)}")
        result = parser.parse_response(test_case)
        print(f"Result: {result}")
        print(
            f"Canonical room: {parser.get_canonical_room_name(result.room_name) if result.room_name else 'None'}"
        )


if __name__ == "__main__":
    test_parser()
