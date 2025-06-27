"""
Test structured output support for different models using actual extractor prompts.

This uses the real system prompt and game data from the episode log to test
which models properly support structured outputs.
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


# Load the actual extractor system prompt
try:
    with open("extractor.md", "r") as f:
        SYSTEM_PROMPT = f.read()
except:
    SYSTEM_PROMPT = """You are an expert data extraction assistant for a text adventure game.
Extract key information from the game text and return it as JSON."""

# Real game text samples
GAME_TEXT_SAMPLES = [
    # Starting location - this is what Zork actually outputs
    """West of House
You are standing in an open field west of a white house, with a boarded front door.
There is a small mailbox here.""",
    
    # Behind house - a location with a window
    """Behind House
You are behind the white house. A path leads into the forest to the north. In one corner of the house there is a small window which is slightly ajar.""",
    
    # Simple action result
    """Taken.""",
    
    # Kitchen with multiple exits
    """Kitchen
You are in the kitchen of the white house. A table seems to have been used recently for the preparation of food. A passage leads to the west and a dark staircase can be seen leading upward. A dark chimney leads down and to the east is a small window which is open.
On the table is an elongated brown sack, smelling of hot peppers.
A bottle is sitting on the table.
The glass bottle contains:
  A quantity of water""",
]

# Models to test
TEST_MODELS = [
    "google/gemma-3-12b-it",  # Currently failing
    "mistralai/mistral-nemo",
    "meta-llama/llama-3.1-8b-instruct", 
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-small-24b-instruct-2501",
    "mistralai/mistral-small-3.1-24b-instruct",
    "meta-llama/llama-4-scout",
    "qwen/qwen3-32b",
    "openai/gpt-4o-mini",  # Known working
]


def build_extraction_prompt(game_text: str, previous_location: str = None) -> str:
    """Build the extraction prompt exactly as the extractor does."""
    prompt_parts = []
    
    if previous_location:
        prompt_parts.append(f"Previous Location: {previous_location}")
    
    # Movement analysis context (simulated)
    prompt_parts.append(f"Movement Analysis: No previous location provided")
    
    # Add the game text
    prompt_parts.append(f"Game Text:\n```\n{game_text}\n```")
    
    # Simple instruction - let the system prompt handle the details
    prompt_parts.append("Please extract the key information from this game text and return it as JSON.")
    
    return "\n\n".join(prompt_parts)


def test_model_with_game_text(model_name: str, game_text: str, sample_name: str):
    """Test a specific model with actual game text."""
    print(f"\n  Sample: {sample_name}")
    print(f"  Game text preview: {game_text[:50]}...")
    
    config = get_config()
    client = LLMClientWrapper(
        base_url=config.llm.get_base_url_for_model('info_ext'),
        api_key=get_client_api_key(),
    )
    
    # Build prompt exactly as extractor does
    extraction_prompt = build_extraction_prompt(game_text)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": extraction_prompt}
    ]
    
    # Test with structured output + require_parameters
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
            response_format=create_json_schema(ExtractorResponse),
            provider={
                "require_parameters": True
            }
        )
        
        # Check if response contains the malformed array issue
        raw_content = response.content
        print(f"    Raw response length: {len(raw_content)} chars")
        
        # Check for the specific malformed array pattern
        if "\n  \n  \n  " in raw_content:
            print(f"    ⚠️  MALFORMED ARRAY DETECTED - contains excessive newlines")
            # Show snippet of the problem
            import re
            match = re.search(r'([\[\{][^\[\]]*\n\s*\n\s*\n[^\[\]]*[\]\}])', raw_content)
            if match:
                print(f"    Problem area: {match.group(1)[:100]}...")
        
        parsed = json.loads(raw_content)
        validated = ExtractorResponse(**parsed)
        print(f"    ✓ SUCCESS: Properly formatted JSON with correct schema")
        
        # Show the actual extracted data
        print(f"    Location: {validated.current_location_name}")
        print(f"    Exits: {validated.exits}")
        
        return "success"
        
    except json.JSONDecodeError as e:
        print(f"    ✗ JSON PARSE ERROR: {e}")
        print(f"    First 200 chars of response: {response.content[:200] if 'response' in locals() else 'No response'}")
        return "json_error"
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "No endpoints found" in error_msg:
            print(f"    ✗ NO PROVIDER: Structured output not supported")
            return "no_provider"
        else:
            print(f"    ✗ ERROR: {error_msg[:100]}")
            return "error"


def main():
    """Test all models with actual extractor setup."""
    print("Testing structured output with actual extractor prompts and game data")
    print("=" * 80)
    
    results = {}
    
    for model in TEST_MODELS:
        print(f"\n{'='*60}")
        print(f"Testing: {model}")
        print(f"{'='*60}")
        
        model_results = []
        
        # Test with first sample (most important - the one that failed)
        result = test_model_with_game_text(model, GAME_TEXT_SAMPLES[0], "West of House")
        model_results.append(result)
        
        # If first test succeeded, try a more complex one
        if result == "success":
            result2 = test_model_with_game_text(model, GAME_TEXT_SAMPLES[3], "Kitchen (complex)")
            model_results.append(result2)
        
        results[model] = model_results
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS:")
    print("=" * 80)
    
    working_models = []
    json_error_models = []
    no_provider_models = []
    
    for model, model_results in results.items():
        if all(r == "success" for r in model_results):
            working_models.append(model)
        elif any(r == "json_error" for r in model_results):
            json_error_models.append(model)
        elif any(r == "no_provider" for r in model_results):
            no_provider_models.append(model)
    
    print(f"\n✓ WORKING MODELS ({len(working_models)}):")
    for model in working_models:
        print(f"  - {model}")
    
    print(f"\n✗ JSON PARSE ERRORS ({len(json_error_models)}):")
    for model in json_error_models:
        print(f"  - {model} (produces malformed JSON)")
    
    print(f"\n⚠️  NO PROVIDER SUPPORT ({len(no_provider_models)}):")
    for model in no_provider_models:
        print(f"  - {model} (no provider supports structured output)")
    
    print(f"\nRECOMMENDATIONS:")
    if working_models:
        print(f"1. Switch info_ext_model to one of these working models:")
        for model in working_models[:3]:
            print(f"   - {model}")
    print(f"2. Current model ({get_config().llm.info_ext_model}) appears to not support structured outputs properly")
    print(f"3. Consider implementing fallback to JSON mode with examples for unsupported models")


if __name__ == "__main__":
    main()