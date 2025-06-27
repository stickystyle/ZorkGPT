"""
Test Gemma3's JSON generation with different approaches.

This script tests:
1. response_format with structured outputs (current approach)
2. response_format with simple JSON mode
3. JSON example in prompt WITH response_format
4. JSON example in prompt WITHOUT response_format
"""

import json
from typing import Optional
from pydantic import BaseModel
from llm_client import LLMClientWrapper
from shared_utils import create_json_schema
from config import get_config, get_client_api_key


class TestResponse(BaseModel):
    """Simple test response for JSON generation."""
    current_location_name: str
    exits: list[str]
    visible_objects: list[str]
    visible_characters: list[str]
    important_messages: list[str]
    in_combat: bool
    score: Optional[int] = None
    moves: Optional[int] = None


# Test game text
TEST_GAME_TEXT = """West of House
You are standing in an open field west of a white house, with a boarded front door.
There is a small mailbox here.

Exits: north, south, east"""


SYSTEM_PROMPT = """You are an expert data extraction assistant for a text adventure game.
Extract key information from the game text and return it as JSON."""


def test_method_1_structured_outputs():
    """Test 1: Current approach - structured outputs with strict mode"""
    print("\n=== Method 1: Structured Outputs (strict=True) ===")
    
    config = get_config()
    client = LLMClientWrapper(
        base_url=config.llm.get_base_url_for_model('info_ext'),
        api_key=get_client_api_key(),
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract information from this game text:\n\n{TEST_GAME_TEXT}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=config.llm.info_ext_model,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format=create_json_schema(TestResponse),
        )
        
        print(f"Raw response: {response.content}")
        
        # Try to parse
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        
        # Validate with Pydantic
        validated = TestResponse(**parsed)
        print(f"Pydantic validation: SUCCESS")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


def test_method_2_json_mode():
    """Test 2: Simple JSON mode (not strict)"""
    print("\n=== Method 2: JSON Mode (type=json_object) ===")
    
    config = get_config()
    client = LLMClientWrapper(
        base_url=config.llm.get_base_url_for_model('info_ext'),
        api_key=get_client_api_key(),
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract information from this game text:\n\n{TEST_GAME_TEXT}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=config.llm.info_ext_model,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        
        print(f"Raw response: {response.content}")
        
        # Try to parse
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        
        # Validate with Pydantic
        validated = TestResponse(**parsed)
        print(f"Pydantic validation: SUCCESS")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


def test_method_3_example_with_format():
    """Test 3: JSON example in prompt WITH response_format"""
    print("\n=== Method 3: JSON Example + response_format ===")
    
    config = get_config()
    client = LLMClientWrapper(
        base_url=config.llm.get_base_url_for_model('info_ext'),
        api_key=get_client_api_key(),
    )
    
    enhanced_prompt = f"""Extract information from this game text and return EXACTLY this JSON structure:

{{
  "current_location_name": "West of House",
  "exits": ["north", "south", "east"],
  "visible_objects": ["small mailbox"],
  "visible_characters": [],
  "important_messages": [],
  "in_combat": false,
  "score": null,
  "moves": null
}}

IMPORTANT: 
- Empty arrays must be [] not whitespace
- All arrays must contain quoted strings or be empty
- Do not add extra fields

Game text:
{TEST_GAME_TEXT}"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": enhanced_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=config.llm.info_ext_model,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        
        print(f"Raw response: {response.content}")
        
        # Try to parse
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        
        # Validate with Pydantic
        validated = TestResponse(**parsed)
        print(f"Pydantic validation: SUCCESS")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


def test_method_4_example_no_format():
    """Test 4: JSON example in prompt WITHOUT response_format"""
    print("\n=== Method 4: JSON Example WITHOUT response_format ===")
    
    config = get_config()
    client = LLMClientWrapper(
        base_url=config.llm.get_base_url_for_model('info_ext'),
        api_key=get_client_api_key(),
    )
    
    enhanced_prompt = f"""Extract information from this game text and return ONLY valid JSON matching this exact structure:

{{
  "current_location_name": "West of House",
  "exits": ["north", "south", "east"],
  "visible_objects": ["small mailbox"],
  "visible_characters": [],
  "important_messages": [],
  "in_combat": false,
  "score": null,
  "moves": null
}}

Rules:
- Return ONLY the JSON object, no other text
- Empty arrays must be [] not whitespace or newlines
- All arrays must contain quoted strings or be empty
- Use double quotes for all strings
- No trailing commas
- No comments

Game text:
{TEST_GAME_TEXT}"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nYou must respond with valid JSON only. No explanations or other text."},
        {"role": "user", "content": enhanced_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=config.llm.info_ext_model,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            # NO response_format parameter
        )
        
        print(f"Raw response: {response.content}")
        
        # Try to parse
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        
        # Validate with Pydantic
        validated = TestResponse(**parsed)
        print(f"Pydantic validation: SUCCESS")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


def test_method_5_minimal_schema():
    """Test 5: Minimal schema approach"""
    print("\n=== Method 5: Minimal Schema ===")
    
    config = get_config()
    client = LLMClientWrapper(
        base_url=config.llm.get_base_url_for_model('info_ext'),
        api_key=get_client_api_key(),
    )
    
    minimal_prompt = f"""Return a JSON object with these exact fields:
- current_location_name: string
- exits: array of strings
- visible_objects: array of strings  
- visible_characters: array of strings
- important_messages: array of strings
- in_combat: boolean
- score: number or null
- moves: number or null

Game text:
{TEST_GAME_TEXT}"""
    
    messages = [
        {"role": "system", "content": "You are a JSON generator. Return only valid JSON."},
        {"role": "user", "content": minimal_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=config.llm.info_ext_model,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        
        print(f"Raw response: {response.content}")
        
        # Try to parse
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        
        # Validate with Pydantic
        validated = TestResponse(**parsed)
        print(f"Pydantic validation: SUCCESS")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


def main():
    """Run all tests."""
    print(f"Testing JSON generation with model: {get_config().llm.info_ext_model}")
    print("=" * 60)
    
    test_method_1_structured_outputs()
    test_method_2_json_mode()
    test_method_3_example_with_format()
    test_method_4_example_no_format()
    test_method_5_minimal_schema()
    
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()