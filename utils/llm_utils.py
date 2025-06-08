"""
LLM utility functions for handling different response formats.

This module provides utilities to handle various LLM response formats
consistently across the ZorkGPT system.
"""

def extract_llm_content(response):
    """
    Extract content from LLM response, handling common formats.
    
    Args:
        response: LLM response object from various clients
        
    Returns:
        str: The extracted content text, or empty string if none found
    """
    if not response:
        return ""
    
    try:
        # OpenAI format - check this first and return if successful
        if hasattr(response, 'choices') and response.choices:
            try:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content or ""
            except (IndexError, AttributeError, TypeError):
                pass
        
        # Direct content attribute
        if hasattr(response, 'content'):
            return response.content or ""
        
        # String response
        if isinstance(response, str):
            return response
        
        # Dict response
        if isinstance(response, dict):
            if 'choices' in response and response['choices']:
                try:
                    return response['choices'][0]['message']['content'] or ""
                except (KeyError, IndexError, TypeError):
                    pass
            elif 'content' in response:
                return response['content'] or ""
            else:
                return str(response)
        
        # Fallback to string conversion
        return str(response)
        
    except Exception:
        # If anything goes wrong, return empty string
        return ""