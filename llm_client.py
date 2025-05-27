"""
LLM Client module for making direct API calls with advanced sampling parameters.

This module replaces the OpenAI SDK to enable fine-tuned control over sampling
parameters like top_k, min_p, etc. that are not supported by the OpenAI SDK.
"""

import json
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from config import get_config, get_client_api_key


@dataclass
class LLMResponse:
    """Response object that mimics OpenAI's response structure."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None


class LLMClient:
    """
    Custom LLM client that uses requests to make direct API calls.
    
    This allows us to use advanced sampling parameters like top_k and min_p
    that aren't supported by the OpenAI SDK.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM client.
        
        Args:
            base_url: Base URL for the API endpoint
            api_key: API key for authentication
        """
        config = get_config()
        self.base_url = base_url or config.llm.client_base_url
        self.api_key = api_key or get_client_api_key() or "not-needed"
        
        # Set default headers that will be included in all requests
        self.default_headers = {
            "X-Title": "ZorkGPT",
            "HTTP-Referer": "https://zorkgpt.com",
        }
        
        # Ensure base_url ends with /v1 if it doesn't already
        if not self.base_url.endswith('/v1'):
            if not self.base_url.endswith('/'):
                self.base_url += '/'
            self.base_url += 'v1'
    
    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Create a chat completion with advanced sampling parameters.
        
        Args:
            model: Model name to use
            messages: List of message dictionaries
            temperature: Sampling temperature (None = exclude from payload)
            top_p: Top-p nucleus sampling (None = exclude from payload)
            top_k: Top-k sampling (None = exclude from payload)
            min_p: Minimum probability sampling (None = exclude from payload)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            response_format: Response format specification
            extra_headers: Additional headers to send
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object with the generated content
        """
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Add default headers
        headers.update(self.default_headers)
        
        # Add any extra headers (will override defaults if same key)
        if extra_headers:
            headers.update(extra_headers)
        
        # Build the request payload
        payload = {
            "model": model,
            "messages": messages,
        }
        
        # Handle model-specific parameter restrictions
        is_o1_model = "o1-" in model.lower()
        
        # Only include sampling parameters if they're not None and supported by the model
        if temperature is not None:
            if not is_o1_model:  # o1 models don't support temperature
                payload["temperature"] = temperature
            
        if top_p is not None:
            if not is_o1_model:  # o1 models don't support top_p
                payload["top_p"] = top_p
            
        if top_k is not None:
            if not is_o1_model:  # o1 models don't support top_k
                payload["top_k"] = top_k
            
        if min_p is not None:
            if not is_o1_model:  # o1 models don't support min_p
                payload["min_p"] = min_p
        
        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if stop is not None:
            payload["stop"] = stop
            
        if response_format is not None:
            if not is_o1_model:  # o1 models don't support response_format
                payload["response_format"] = response_format
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Extract content from response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
            else:
                raise ValueError("No valid choices in response")
            
            # Extract usage information if available
            usage = response_data.get("usage")
            
            return LLMResponse(
                content=content,
                model=model,
                usage=usage
            )
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM API request failed: {e}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Invalid LLM API response format: {e}")


class ChatCompletions:
    """Nested class to mimic OpenAI SDK structure."""
    
    def __init__(self, client: LLMClient):
        self.client = client
    
    def create(self, **kwargs) -> LLMResponse:
        """Create a chat completion."""
        return self.client.chat_completions_create(**kwargs)


class Chat:
    """Nested class to mimic OpenAI SDK structure."""
    
    def __init__(self, client: LLMClient):
        self.completions = ChatCompletions(client)


class LLMClientWrapper:
    """
    Wrapper class that provides OpenAI SDK-compatible interface.
    
    This allows us to use the new client as a drop-in replacement
    for the OpenAI client in existing code.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the wrapper client.
        
        Args:
            base_url: Base URL for the API endpoint
            api_key: API key for authentication
            **kwargs: Additional arguments for LLMClient
        """
        self.client = LLMClient(base_url=base_url, api_key=api_key, **kwargs)
        self.chat = Chat(self.client)


# For convenience, provide a function that creates the wrapper
def create_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMClientWrapper:
    """
    Create an LLM client wrapper.
    
    Args:
        base_url: Base URL for the API endpoint  
        api_key: API key for authentication
        **kwargs: Additional arguments for LLMClient
        
    Returns:
        LLMClientWrapper instance
    """
    return LLMClientWrapper(base_url=base_url, api_key=api_key, **kwargs) 