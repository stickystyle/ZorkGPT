"""
Quick test of key models for structured output support.
"""

import json
from typing import Optional, List
from pydantic import BaseModel
from llm_client import LLMClientWrapper
from shared_utils import create_json_schema
from config import get_config, get_client_api_key


class ExtractorResponse(BaseModel):
    current_location_name: str
    exits: List[str]
    visible_objects: List[str]
    visible_characters: List[str]
    important_messages: List[str]
    in_combat: bool
    score: Optional[int] = None
    moves: Optional[int] = None


# Simple test
SYSTEM_PROMPT = "You are an expert data extraction assistant. Extract information and return as JSON."

GAME_TEXT = """West of House
You are standing in an open field west of a white house, with a boarded front door.
There is a small mailbox here."""

USER_PROMPT = f"""Game Text:
```
{GAME_TEXT}
```

Please extract the key information from this game text and return it as JSON."""

# Key models to test
TEST_MODELS = [
    "google/gemma-3-12b-it",         # Currently failing
    "openai/gpt-4o-mini",            # Should work
    "mistralai/mistral-nemo",        # Test Mistral
    "meta-llama/llama-3.1-8b-instruct",  # Test Llama
]


def test_model(model_name: str):
    """Quick test of a model."""
    print(f"\n=== {model_name} ===")
    
    config = get_config()
    client = LLMClientWrapper(
        base_url=config.llm.get_base_url_for_model('info_ext'),
        api_key=get_client_api_key(),
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    
    # Test 1: With structured output + require_parameters
    print("Test 1: Structured output + require_parameters")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format=create_json_schema(ExtractorResponse),
            provider={"require_parameters": True}
        )
        
        # Check response
        if "\n  \n  \n" in response.content:
            print("  ✗ FAIL: Malformed array with newlines detected")
        else:
            parsed = json.loads(response.content)
            validated = ExtractorResponse(**parsed)
            print(f"  ✓ SUCCESS: {validated.current_location_name}, exits: {validated.exits}")
    except Exception as e:
        if "404" in str(e):
            print("  ✗ NO PROVIDER: Not supported")
        else:
            print(f"  ✗ ERROR: {str(e)[:80]}")
    
    # Test 2: Without require_parameters
    print("Test 2: Structured output WITHOUT require_parameters")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format=create_json_schema(ExtractorResponse),
        )
        
        if "\n  \n  \n" in response.content:
            print("  ✗ FAIL: Malformed array with newlines")
        else:
            parsed = json.loads(response.content)
            print(f"  ○ Works but may not respect schema")
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)[:80]}")
    
    # Test 3: JSON mode only
    print("Test 3: Simple JSON mode")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        
        parsed = json.loads(response.content)
        print(f"  ○ JSON produced: {list(parsed.keys())}")
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)[:80]}")


def main():
    print("Quick structured output test for key models")
    print("=" * 60)
    
    for model in TEST_MODELS:
        test_model(model)
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("- If gemma-3 fails with structured output, it's an OpenRouter limitation")
    print("- Switch to a model that shows SUCCESS in Test 1")
    print("- Or implement JSON mode + prompt examples as a fallback")


if __name__ == "__main__":
    main()