"""
Tests for StructuredZorkParser score parsing functionality.
"""

import pytest
from game_interface.core.structured_parser import StructuredZorkParser


class TestStructuredZorkParserScore:
    """Test score parsing in StructuredZorkParser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = StructuredZorkParser()

    def test_score_parsing_with_structured_header(self):
        """Test score parsing when structured header is present."""
        response_text = "> Kitchen                                          Score: 25       Moves: 12\n\nKitchen\nYou are in the kitchen of the white house."
        
        result = self.parser.parse_response(response_text)
        
        assert result.score == 25
        assert result.moves == 12
        assert result.room_name == "Kitchen"
        assert result.has_structured_header is True
        assert "kitchen of the white house" in result.game_text

    def test_score_parsing_zero_score(self):
        """Test that score of 0 is parsed correctly."""
        response_text = "> West of House                                    Score: 0        Moves: 1\n\nWest of House\nYou are standing in an open field."
        
        result = self.parser.parse_response(response_text)
        
        assert result.score == 0
        assert result.moves == 1
        assert result.room_name == "West of House"
        assert result.has_structured_header is True

    def test_score_parsing_high_score(self):
        """Test parsing of high score values."""
        response_text = "> Treasure Room                                    Score: 350      Moves: 543\n\nTreasure Room\nYou have found the ultimate treasure!"
        
        result = self.parser.parse_response(response_text)
        
        assert result.score == 350
        assert result.moves == 543
        assert result.room_name == "Treasure Room"
        assert result.has_structured_header is True

    def test_score_parsing_no_structured_header(self):
        """Test that score is None when no structured header is present."""
        response_text = ">I don't understand that."
        
        result = self.parser.parse_response(response_text)
        
        assert result.score is None
        assert result.moves is None
        assert result.room_name is None
        assert result.has_structured_header is False

    def test_score_parsing_simple_response(self):
        """Test score parsing with simple responses like '>Taken.'"""
        response_text = ">Taken."
        
        result = self.parser.parse_response(response_text)
        
        assert result.score is None
        assert result.moves is None
        assert result.room_name is None
        assert result.has_structured_header is False

    def test_score_parsing_empty_response(self):
        """Test score parsing with empty response."""
        response_text = ""
        
        result = self.parser.parse_response(response_text)
        
        assert result.score is None
        assert result.moves is None
        assert result.room_name is None
        assert result.has_structured_header is False

    def test_score_parsing_various_room_names(self):
        """Test score parsing with various room name formats."""
        test_cases = [
            ("> Forest Clearing                                  Score: 5        Moves: 15", "Forest Clearing", 5, 15),
            ("> North of House                                   Score: 0        Moves: 3", "North of House", 0, 3),
            ("> Behind House                                     Score: 10       Moves: 4", "Behind House", 10, 4),
            ("> Kitchen                                          Score: 10       Moves: 4", "Kitchen", 10, 4),
        ]
        
        for response_text, expected_room, expected_score, expected_moves in test_cases:
            full_response = f"{response_text}\n\nSome game text here."
            result = self.parser.parse_response(full_response)
            
            assert result.room_name == expected_room
            assert result.score == expected_score
            assert result.moves == expected_moves
            assert result.has_structured_header is True

    def test_extract_score_and_moves_method(self):
        """Test the extract_score_and_moves convenience method."""
        response_text = "> Kitchen                                          Score: 15       Moves: 8\n\nKitchen description."
        
        score, moves = self.parser.extract_score_and_moves(response_text)
        
        assert score == 15
        assert moves == 8

    def test_extract_score_and_moves_no_header(self):
        """Test extract_score_and_moves when no header is present."""
        response_text = ">Invalid command."
        
        score, moves = self.parser.extract_score_and_moves(response_text)
        
        assert score is None
        assert moves is None

    def test_score_parsing_multiline_game_text(self):
        """Test score parsing with multiline game text after header."""
        response_text = """> Forest Clearing                                  Score: 5        Moves: 15

You are in a forest clearing.
There is a path to the north.
A small cottage is visible to the east.
The sun is shining brightly."""
        
        result = self.parser.parse_response(response_text)
        
        assert result.score == 5
        assert result.moves == 15
        assert result.room_name == "Forest Clearing"
        assert result.has_structured_header is True
        assert "You are in a forest clearing." in result.game_text
        assert "The sun is shining brightly." in result.game_text

    def test_score_parsing_edge_cases(self):
        """Test score parsing edge cases."""
        # Score with leading/trailing spaces
        response_text = ">  Kitchen                                        Score:  25      Moves:  12  \n\nKitchen text."
        result = self.parser.parse_response(response_text)
        assert result.score == 25
        assert result.moves == 12
        
        # Very long room name
        long_room = "Very Long Room Name That Goes On And On"
        response_text = f"> {long_room}                    Score: 100      Moves: 200\n\nRoom description."
        result = self.parser.parse_response(response_text)
        assert result.room_name == long_room
        assert result.score == 100
        assert result.moves == 200