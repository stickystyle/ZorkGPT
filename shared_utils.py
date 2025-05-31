"""
Shared utilities for ZorkGPT components.

This module contains common utility functions used across multiple components
to avoid code duplication.
"""

from typing import Dict, Any, Type, Union, List
from pydantic import BaseModel


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


def estimate_context_tokens(memory_history: List[Any] = None, 
                          reasoning_history: List[Any] = None,
                          knowledge_base_path: str = "knowledgebase.md",
                          additional_content: str = "") -> int:
    """
    Estimate total context tokens from various sources.
    
    This function aggregates token estimates from multiple sources commonly
    used in ZorkGPT context windows.
    
    Args:
        memory_history: List of memory log entries
        reasoning_history: List of action reasoning entries
        knowledge_base_path: Path to knowledge base file
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
    try:
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            total_tokens += estimate_tokens(f.read())
    except FileNotFoundError:
        pass
    
    # Count additional content
    if additional_content:
        total_tokens += estimate_tokens(additional_content)
    
    return total_tokens 