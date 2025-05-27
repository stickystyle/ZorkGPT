"""
Shared utilities for ZorkGPT components.

This module contains common utility functions used across multiple components
to avoid code duplication.
"""

from typing import Dict, Any, Type
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