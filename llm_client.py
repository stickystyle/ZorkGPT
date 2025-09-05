"""
LLM Client module for making direct API calls with advanced sampling parameters.

This module replaces the OpenAI SDK to enable fine-tuned control over sampling
parameters like top_k, min_p, etc. that are not supported by the OpenAI SDK.
"""

import json
import requests
import random
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from config import get_config, get_client_api_key
from enum import Enum


@dataclass
class LLMResponse:
    """Response object for LLM completions."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None


class RetryableError(Exception):
    """Exception for errors that should trigger a retry."""

    pass


class RateLimitError(RetryableError):
    """Exception for rate limit errors."""

    pass


class ServerError(RetryableError):
    """Exception for server errors (5xx)."""

    pass


class LLMTimeoutError(RetryableError):
    """Exception for timeout errors."""

    pass


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(
        self, failure_threshold: int, recovery_timeout: float, success_threshold: int
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0

    def call_succeeded(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def call_failed(self):
        """Record a failed call."""
        self.failure_count += 1
        self.success_count = 0  # Reset success count
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self._open_circuit()

    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self._half_open_circuit()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True

        return False

    def _close_circuit(self):
        """Close the circuit (normal operation)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

    def _open_circuit(self):
        """Open the circuit (failing)."""
        self.state = CircuitState.OPEN

    def _half_open_circuit(self):
        """Half-open the circuit (testing recovery)."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


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
        logger=None,
    ):
        """
        Initialize the LLM client.

        Args:
            base_url: Base URL for the API endpoint
            api_key: API key for authentication
            logger: Logger instance for tracking retry attempts
        """
        config = get_config()
        self.base_url = base_url or config.llm.client_base_url
        self.api_key = api_key or get_client_api_key() or "not-needed"
        self.retry_config = config.retry
        self.logger = logger

        # Initialize circuit breaker if enabled
        if self.retry_config.circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.retry_config.circuit_breaker_failure_threshold,
                recovery_timeout=self.retry_config.circuit_breaker_recovery_timeout,
                success_threshold=self.retry_config.circuit_breaker_success_threshold,
            )
        else:
            self.circuit_breaker = None

        # Set default headers that will be included in all requests
        self.default_headers = {
            "X-Title": "ZorkGPT",
            "HTTP-Referer": "https://zorkgpt.com",
        }

        # Ensure base_url ends with /v1 if it doesn't already
        # if not self.base_url.endswith('/v1'):
        #     if not self.base_url.endswith('/'):
        #         self.base_url += '/'
        #     self.base_url += 'v1'

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate the delay for exponential backoff with jitter."""
        # Calculate exponential delay
        delay = self.retry_config.initial_delay * (
            self.retry_config.exponential_base**attempt
        )

        # Cap at max_delay
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter to prevent thundering herd
        if self.retry_config.jitter_factor > 0:
            jitter = delay * self.retry_config.jitter_factor * random.random()
            delay += jitter

        return delay

    def _classify_error(
        self, response: requests.Response = None, exception: Exception = None
    ) -> Optional[Exception]:
        """Classify an error to determine if it should trigger a retry."""
        # Handle HTTP response errors
        if response is not None:
            status_code = response.status_code

            # Rate limit errors (429, or provider-specific patterns)
            if status_code == 429:
                return RateLimitError(f"Rate limit error: {status_code}")

            # Check for rate limit in response text (some providers use 400 with specific messages)
            try:
                response_text = response.text.lower()
                if any(
                    phrase in response_text
                    for phrase in ["rate limit", "too many requests", "quota exceeded"]
                ):
                    return RateLimitError(
                        f"Rate limit detected in response: {response.text[:200]}"
                    )
            except:
                pass  # If we can't parse response text, continue with status code logic

            # Server errors (5xx)
            if 500 <= status_code < 600 and self.retry_config.retry_on_server_error:
                return ServerError(f"Server error: {status_code}")

            # Other client errors (4xx) generally shouldn't be retried
            return None

        # Handle request exceptions
        if exception is not None:
            if (
                isinstance(exception, requests.exceptions.Timeout)
                and self.retry_config.retry_on_timeout
            ):
                return LLMTimeoutError(f"Request timeout: {exception}")
            elif isinstance(
                exception,
                (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException,
                ),
            ):
                # Network issues might be transient
                return RetryableError(f"Network error: {exception}")

        return None

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
        **kwargs,
    ) -> LLMResponse:
        """
        Create a chat completion with advanced sampling parameters and retry logic.

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
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            error_msg = (
                f"Circuit breaker is open. Service unavailable until recovery timeout."
            )
            if self.logger:
                self.logger.error(
                    error_msg,
                    extra={
                        "extras": {
                            "event_type": "circuit_breaker_open",
                            "circuit_state": self.circuit_breaker.state.value,
                            "failure_count": self.circuit_breaker.failure_count,
                            "model": model,
                        }
                    },
                )
            raise CircuitOpenError(error_msg)

        last_exception = None

        for attempt in range(
            self.retry_config.max_retries + 1
        ):  # +1 for initial attempt
            try:
                result = self._make_request(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    response_format=response_format,
                    extra_headers=extra_headers,
                    **kwargs,
                )

                # Record success in circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker.call_succeeded()

                return result

            except RetryableError as e:
                last_exception = e

                # Record failure in circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker.call_failed()

                # Don't retry on the last attempt
                if attempt >= self.retry_config.max_retries:
                    break

                # Calculate backoff delay
                delay = self._calculate_backoff_delay(attempt)

                # Log retry attempt with proper logging if available
                retry_msg = f"API call failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}"
                backoff_msg = f"Retrying in {delay:.2f} seconds..."

                if self.logger:
                    self.logger.warning(
                        retry_msg,
                        extra={
                            "extras": {
                                "event_type": "llm_retry",
                                "attempt": attempt + 1,
                                "max_attempts": self.retry_config.max_retries + 1,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "backoff_delay": delay,
                                "model": model,
                                "circuit_state": self.circuit_breaker.state.value
                                if self.circuit_breaker
                                else "disabled",
                            }
                        },
                    )
                    self.logger.info(backoff_msg)
                else:
                    # Fallback to print if no logger
                    print(retry_msg)
                    print(backoff_msg)

                time.sleep(delay)
                continue

            except Exception as e:
                # Record failure in circuit breaker for non-retryable errors too
                if self.circuit_breaker:
                    self.circuit_breaker.call_failed()

                # Non-retryable error, re-raise immediately
                raise Exception(f"LLM API request failed: {e}")

        # If we get here, all retries were exhausted
        raise Exception(
            f"LLM API request failed after {self.retry_config.max_retries + 1} attempts. Last error: {last_exception}"
        )

    def _make_request(
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
        **kwargs,
    ) -> LLMResponse:
        """Make the actual HTTP request."""
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

        # Handle model-specific parameter restrictions
        is_o1_model = "o1-" in model.lower()
        is_o3_model = "o3-" in model.lower()
        is_reasoning_model = is_o1_model or is_o3_model

        # For o1/o3 models, set all message roles to "user"
        if is_reasoning_model:
            messages = [
                {"role": "user", "content": msg.get("content", "")} for msg in messages
            ]

        # Build the request payload
        payload = {
            "model": model,
            "messages": messages,
        }

        # Only include sampling parameters if they're not None and supported by the model
        if temperature is not None:
            if not is_reasoning_model:  # o1/o3 models don't support temperature
                payload["temperature"] = temperature

        if top_p is not None:
            if not is_reasoning_model:  # o1/o3 models don't support top_p
                payload["top_p"] = top_p

        if top_k is not None:
            if not is_reasoning_model:  # o1/o3 models don't support top_k
                payload["top_k"] = top_k

        if min_p is not None:
            if not is_reasoning_model:  # o1/o3 models don't support min_p
                payload["min_p"] = min_p

        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if stop is not None:
            payload["stop"] = stop

        if response_format is not None:
            if not is_reasoning_model:  # o1/o3 models don't support response_format
                payload["response_format"] = response_format

        # Add any additional kwargs
        payload.update(kwargs)

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.retry_config.timeout_seconds,
            )

            # Check for errors that should trigger retry
            if not response.ok:
                retryable_error = self._classify_error(response=response)
                if retryable_error:
                    raise retryable_error
                else:
                    # Non-retryable error
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

            response_data = response.json()

            # Extract content from response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
            else:
                raise ValueError("No valid choices in response")

            # Extract usage information if available
            usage = response_data.get("usage")

            return LLMResponse(content=content, model=model, usage=usage)

        except requests.exceptions.RequestException as e:
            # Classify and potentially convert to retryable error
            retryable_error = self._classify_error(exception=e)
            if retryable_error:
                raise retryable_error
            else:
                raise Exception(f"Request failed: {e}")
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
        logger=None,
        **kwargs,
    ):
        """
        Initialize the wrapper client.

        Args:
            base_url: Base URL for the API endpoint
            api_key: API key for authentication
            logger: Logger instance for tracking retry attempts
            **kwargs: Additional arguments for LLMClient
        """
        self.client = LLMClient(
            base_url=base_url, api_key=api_key, logger=logger, **kwargs
        )
        self.chat = Chat(self.client)


# For convenience, provide a function that creates the wrapper
def create_llm_client(
    base_url: Optional[str] = None, api_key: Optional[str] = None, logger=None, **kwargs
) -> LLMClientWrapper:
    """
    Create an LLM client wrapper.

    Args:
        base_url: Base URL for the API endpoint
        api_key: API key for authentication
        logger: Logger instance for tracking retry attempts
        **kwargs: Additional arguments for LLMClient

    Returns:
        LLMClientWrapper instance
    """
    return LLMClientWrapper(base_url=base_url, api_key=api_key, logger=logger, **kwargs)
