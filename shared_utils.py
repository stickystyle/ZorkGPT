"""
Shared utilities for ZorkGPT components.

This module contains common utility functions used across multiple components
to avoid code duplication.
"""

from typing import Dict, Any, Type, Union, List, Optional
from pydantic import BaseModel
from pathlib import Path


def create_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Create OpenAI structured output schema from a Pydantic model.

    Args:
        model: Pydantic model class to create schema for

    Returns:
        Dictionary containing the JSON schema formatted for OpenAI structured output
    """
    schema = model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "strict": True,
            "schema": schema,
        },
    }


def estimate_tokens(content: Union[str, List[Any], Dict[Any, Any]]) -> int:
    """
    Estimate the number of tokens in given content.

    Uses a rough approximation of 4 characters per token, which works reasonably
    well for English text and structured data.

    Args:
        content: Text string, list of objects, or dictionary to estimate tokens for

    Returns:
        Estimated number of tokens
    """
    if isinstance(content, str):
        return len(content) // 4
    elif isinstance(content, (list, dict)):
        return len(str(content)) // 4
    else:
        return len(str(content)) // 4


def strip_markdown_json_fences(content: str) -> str:
    """
    Strip markdown code fences from JSON content.

    Some LLMs return JSON wrapped in markdown code fences like:
    ```json
    { "key": "value" }
    ```

    This function extracts the JSON content from within the fences.

    Args:
        content: String that may contain markdown-wrapped JSON

    Returns:
        Stripped JSON string, or original content if no fences found
    """
    if "```json" in content:
        # Find the JSON content between ```json and ```
        start_marker = "```json"
        end_marker = "```"
        start_idx = content.find(start_marker) + len(start_marker)
        end_idx = content.find(end_marker, start_idx)
        if end_idx != -1:
            return content[start_idx:end_idx].strip()

    # Also check for plain ``` fences without json language tag
    if content.strip().startswith("```") and content.strip().endswith("```"):
        lines = content.strip().split("\n")
        # Remove first and last lines if they are fence markers
        if lines[0].strip().startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()

    return content


def extract_json_from_text(content: str) -> str:
    """
    Extract JSON from text that may contain reasoning or other content.

    Reasoning models like DeepSeek R1 may return JSON embedded in thinking tags
    or surrounded by explanatory text. This function tries to find and extract
    the JSON object from anywhere in the text.

    Args:
        content: String that may contain JSON embedded in other text

    Returns:
        Extracted JSON string, or original content if no JSON found
    """
    import re
    import json

    # First try the standard markdown fence extraction
    stripped = strip_markdown_json_fences(content)
    if stripped != content:
        return stripped

    # Try to find JSON objects by looking for balanced braces
    # Start from first { and find matching }
    start_idx = content.find('{')
    if start_idx == -1:
        return content

    # Find the matching closing brace
    brace_count = 0
    end_idx = -1
    for i in range(start_idx, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx != -1:
        potential_json = content[start_idx:end_idx]
        # Validate it's actually JSON
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass

    # If all else fails, return original content
    return content


def estimate_context_tokens(
    memory_history: List[Any] = None,
    reasoning_history: List[Any] = None,
    knowledge_base_path: Optional[str] = None,
    additional_content: str = "",
) -> int:
    """
    Estimate total context tokens from various sources.

    This function aggregates token estimates from multiple sources commonly
    used in ZorkGPT context windows.

    Args:
        memory_history: List of memory log entries
        reasoning_history: List of action reasoning entries
        knowledge_base_path: Path to knowledge base file (defaults to game_files/knowledgebase.md from config)
        additional_content: Any additional string content to include

    Returns:
        Total estimated tokens across all sources
    """
    total_tokens = 0

    # Count memory log history
    if memory_history:
        for memory in memory_history:
            total_tokens += estimate_tokens(memory)

    # Count action reasoning history
    if reasoning_history:
        for reasoning in reasoning_history:
            total_tokens += estimate_tokens(reasoning)

    # Count knowledge base
    if knowledge_base_path is None:
        # Construct path from config if not provided
        from config import get_config
        config = get_config()
        knowledge_base_path = str(Path(config.gameplay.zork_game_workdir) / config.files.knowledge_file)

    try:
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            total_tokens += estimate_tokens(f.read())
    except FileNotFoundError:
        pass

    # Count additional content
    if additional_content:
        total_tokens += estimate_tokens(additional_content)

    return total_tokens
