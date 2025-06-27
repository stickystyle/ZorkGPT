"""
Test Gemma3's structured output support with OpenRouter's require_parameters.

According to OpenRouter docs, require_parameters ensures the provider
supports all requested parameters (like response_format).
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


TEST_GAME_TEXT = """West of House
You are standing in an open field west of a white house, with a boarded front door.
There is a small mailbox here.

Exits: north, south, east"""

SYSTEM_PROMPT = """You are an expert data extraction assistant for a text adventure game.
Extract key information from the game text and return it as JSON."""


def test_1_without_require_parameters():
    """Test 1: Current approach without require_parameters"""
    print("\n=== Test 1: WITHOUT require_parameters ===")
    
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
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        validated = TestResponse(**parsed)
        print(f"✓ Pydantic validation: SUCCESS")
        
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")


def test_2_with_require_parameters():
    """Test 2: With require_parameters in provider config"""
    print("\n=== Test 2: WITH require_parameters ===")
    
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
        # Add provider configuration with require_parameters
        response = client.chat.completions.create(
            model=config.llm.info_ext_model,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format=create_json_schema(TestResponse),
            provider={
                "require_parameters": True
            }
        )
        
        print(f"Raw response: {response.content}")
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        validated = TestResponse(**parsed)
        print(f"✓ Pydantic validation: SUCCESS")
        
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")


def test_3_json_mode_with_require_parameters():
    """Test 3: Simple JSON mode with require_parameters"""
    print("\n=== Test 3: JSON Mode + require_parameters ===")
    
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
            provider={
                "require_parameters": True
            }
        )
        
        print(f"Raw response: {response.content}")
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")


def test_4_with_example_and_require_parameters():
    """Test 4: JSON example in prompt + require_parameters"""
    print("\n=== Test 4: JSON Example + require_parameters ===")
    
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

IMPORTANT: Empty arrays must be [] not whitespace

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
            provider={
                "require_parameters": True
            }
        )
        
        print(f"Raw response: {response.content}")
        parsed = json.loads(response.content)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)}")
        validated = TestResponse(**parsed)
        print(f"✓ Pydantic validation: SUCCESS")
        
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")


def test_5_raw_request_with_provider():
    """Test 5: Raw request to see exact behavior"""
    print("\n=== Test 5: Raw Request with Provider Config ===")
    
    import requests
    config = get_config()
    
    url = f"{config.llm.get_base_url_for_model('info_ext')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_client_api_key()}",
        "X-Title": "ZorkGPT",
        "HTTP-Referer": "https://zorkgpt.com",
    }
    
    payload = {
        "model": config.llm.info_ext_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract information from this game text:\n\n{TEST_GAME_TEXT}"}
        ],
        "temperature": 0.1,
        "max_tokens": 300,
        "response_format": create_json_schema(TestResponse),
        "provider": {
            "require_parameters": True
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        # Print response headers
        print("Response headers:")
        for key, value in response.headers.items():
            if key.lower().startswith('x-'):
                print(f"  {key}: {value}")
        
        if response.ok:
            data = response.json()
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
                print(f"\nContent: {content}")
                parsed = json.loads(content)
                validated = TestResponse(**parsed)
                print(f"✓ Pydantic validation: SUCCESS")
        else:
            print(f"\n✗ HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")


def main():
    """Run all tests."""
    print(f"Testing structured output with require_parameters for: {get_config().llm.info_ext_model}")
    print("=" * 70)
    
    test_1_without_require_parameters()
    test_2_with_require_parameters()
    test_3_json_mode_with_require_parameters()
    test_4_with_example_and_require_parameters()
    test_5_raw_request_with_provider()
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("\nKey findings:")
    print("- If Test 2 succeeds but Test 1 fails: require_parameters forces proper provider")
    print("- Check Test 5 headers to see which provider was actually used")


if __name__ == "__main__":
    main()