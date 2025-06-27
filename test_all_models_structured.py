"""
Test structured output support for all requested models.
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


# Use actual extractor prompt
try:
    with open("extractor.md", "r") as f:
        SYSTEM_PROMPT = f.read()
except:
    SYSTEM_PROMPT = "You are an expert data extraction assistant. Extract information and return as JSON."

# Real game text
GAME_TEXT = """West of House
You are standing in an open field west of a white house, with a boarded front door.
There is a small mailbox here."""

USER_PROMPT = f"""Previous Location: None

Movement Analysis: No previous location provided

Game Text:
```
{GAME_TEXT}
```

Please extract the key information from this game text and return it as JSON."""

# Models to test
TEST_MODELS = [
    "mistralai/mistral-nemo",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-small-24b-instruct-2501",
    "mistralai/mistral-small-3.1-24b-instruct",
    "meta-llama/llama-4-scout",
    "qwen/qwen3-32b",
]


def test_model(model_name: str):
    """Test a model with structured output + require_parameters."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    config = get_config()
    client = LLMClientWrapper(
        base_url=config.llm.get_base_url_for_model('info_ext'),
        api_key=get_client_api_key(),
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    
    try:
        print("Sending request...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format=create_json_schema(ExtractorResponse),
            provider={"require_parameters": True}
        )
        
        # Check for malformed arrays
        raw_content = response.content
        print(f"Response received ({len(raw_content)} chars)")
        
        if "\n  \n  \n" in raw_content:
            print("⚠️  WARNING: Malformed array pattern detected")
            # Find and show the problematic section
            lines = raw_content.split('\n')
            for i, line in enumerate(lines):
                if i > 0 and not line.strip() and not lines[i-1].strip():
                    print(f"   Problem at line {i}: multiple empty lines in array")
                    break
        
        # Parse and validate
        parsed = json.loads(raw_content)
        validated = ExtractorResponse(**parsed)
        
        print(f"✓ SUCCESS - Structured output works correctly!")
        print(f"  Location: {validated.current_location_name}")
        print(f"  Exits: {validated.exits}")
        print(f"  Objects: {validated.visible_objects}")
        
        # Check if it properly detected the exits (should include cardinal directions)
        if not any(direction in validated.exits for direction in ["north", "south", "east"]):
            print(f"  ⚠️  Note: Model may not be following exit detection rules (no cardinal directions)")
        
        return "success"
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON PARSE ERROR: {e}")
        if 'response' in locals():
            print(f"  First 300 chars: {response.content[:300]}")
        return "json_error"
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "No endpoints found" in error_msg:
            print(f"✗ NO PROVIDER SUPPORT - No provider can handle structured output for this model")
            return "no_provider"
        else:
            print(f"✗ ERROR: {error_msg}")
            return "error"


def main():
    """Test all models."""
    print("Testing structured output support for extractor models")
    print("=" * 80)
    print(f"Using {len(TEST_MODELS)} models")
    print("This may take several minutes...")
    
    results = {}
    
    for i, model in enumerate(TEST_MODELS):
        print(f"\n[{i+1}/{len(TEST_MODELS)}] Testing {model}")
        result = test_model(model)
        results[model] = result
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    working_models = [m for m, r in results.items() if r == "success"]
    json_error_models = [m for m, r in results.items() if r == "json_error"]
    no_provider_models = [m for m, r in results.items() if r == "no_provider"]
    other_error_models = [m for m, r in results.items() if r == "error"]
    
    print(f"\n✅ FULLY WORKING MODELS ({len(working_models)}):")
    for model in working_models:
        print(f"  - {model}")
    
    if json_error_models:
        print(f"\n❌ JSON PARSE ERRORS ({len(json_error_models)}):")
        for model in json_error_models:
            print(f"  - {model}")
    
    if no_provider_models:
        print(f"\n⚠️  NO PROVIDER SUPPORT ({len(no_provider_models)}):")
        for model in no_provider_models:
            print(f"  - {model}")
    
    if other_error_models:
        print(f"\n❓ OTHER ERRORS ({len(other_error_models)}):")
        for model in other_error_models:
            print(f"  - {model}")
    
    print(f"\n📋 RECOMMENDATIONS:")
    print(f"1. Current model (google/gemma-3-12b-it) doesn't support structured outputs")
    if working_models:
        print(f"2. Switch info_ext_model in pyproject.toml to one of these:")
        for model in working_models[:3]:
            print(f"   info_ext_model = \"{model}\"")
    print(f"3. These models guarantee proper JSON structure without parsing issues")


if __name__ == "__main__":
    main()