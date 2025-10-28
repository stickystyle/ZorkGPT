"""Tests for Langfuse usage details extraction.

This test suite validates that usage details are correctly extracted from LLM
responses for accurate cost tracking and analytics in Langfuse.
"""

import pytest
from llm_client import LLMClient


class TestUsageExtraction:
    """Test suite for _extract_usage_details method."""

    @pytest.fixture
    def client(self):
        """Create LLM client instance for testing."""
        return LLMClient(base_url="http://test", api_key="test-key", logger=None)

    def test_extract_usage_standard_openai_format(self, client):
        """Test extraction of standard OpenAI-style usage."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }

        result = client._extract_usage_details(usage)

        assert result == {
            "input": 100,
            "output": 50,
            "total": 150
        }

    def test_extract_usage_with_anthropic_cache(self, client):
        """Test extraction includes Anthropic cache fields."""
        usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 300
        }

        result = client._extract_usage_details(usage)

        assert result["input"] == 1000
        assert result["output"] == 200
        assert result["total"] == 1200
        assert result["cache_creation_input_tokens"] == 500
        assert result["cache_read_input_tokens"] == 300

    def test_extract_usage_none_returns_none(self, client):
        """Test that None usage returns None."""
        assert client._extract_usage_details(None) is None

    def test_extract_usage_empty_dict_returns_none(self, client):
        """Test that empty dict returns None."""
        assert client._extract_usage_details({}) is None

    def test_extract_usage_partial_fields_prompt_only(self, client):
        """Test extraction with only prompt_tokens present."""
        usage = {"prompt_tokens": 100}  # Missing completion_tokens and total_tokens
        result = client._extract_usage_details(usage)

        assert result == {"input": 100}

    def test_extract_usage_partial_fields_completion_only(self, client):
        """Test extraction with only completion_tokens present."""
        usage = {"completion_tokens": 50}
        result = client._extract_usage_details(usage)

        assert result == {"output": 50}

    def test_extract_usage_partial_fields_total_only(self, client):
        """Test extraction with only total_tokens present."""
        usage = {"total_tokens": 150}
        result = client._extract_usage_details(usage)

        assert result == {"total": 150}

    def test_extract_usage_cache_fields_only(self, client):
        """Test extraction with only cache fields present."""
        usage = {
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 300
        }

        result = client._extract_usage_details(usage)

        assert result == {
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 300
        }

    def test_extract_usage_invalid_type_string(self, client):
        """Test handling of string instead of dict."""
        assert client._extract_usage_details("not a dict") is None

    def test_extract_usage_invalid_type_int(self, client):
        """Test handling of integer instead of dict."""
        assert client._extract_usage_details(123) is None

    def test_extract_usage_invalid_type_list(self, client):
        """Test handling of list instead of dict."""
        assert client._extract_usage_details([1, 2, 3]) is None

    def test_extract_usage_invalid_type_bool(self, client):
        """Test handling of boolean instead of dict."""
        assert client._extract_usage_details(True) is None
        assert client._extract_usage_details(False) is None

    def test_extract_usage_with_extra_fields(self, client):
        """Test that extra/unknown fields are ignored."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "unknown_field": "should be ignored",
            "random_number": 999,
            "extra_data": {"nested": "value"}
        }

        result = client._extract_usage_details(usage)

        # Should only extract known fields
        assert result == {"input": 100, "output": 50}
        assert "unknown_field" not in result
        assert "random_number" not in result
        assert "extra_data" not in result

    def test_extract_usage_zero_values(self, client):
        """Test extraction with zero token values (edge case but valid)."""
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        result = client._extract_usage_details(usage)

        assert result == {
            "input": 0,
            "output": 0,
            "total": 0
        }

    def test_extract_usage_mixed_valid_and_cache_fields(self, client):
        """Test extraction with mix of standard and cache fields."""
        usage = {
            "prompt_tokens": 100,
            "total_tokens": 150,
            "cache_read_input_tokens": 50,
            # Missing completion_tokens but has cache field
        }

        result = client._extract_usage_details(usage)

        assert result == {
            "input": 100,
            "total": 150,
            "cache_read_input_tokens": 50
        }

    def test_extract_usage_realistic_openrouter_response(self, client):
        """Test extraction from realistic OpenRouter API response format."""
        usage = {
            "prompt_tokens": 523,
            "completion_tokens": 142,
            "total_tokens": 665
        }

        result = client._extract_usage_details(usage)

        assert result["input"] == 523
        assert result["output"] == 142
        assert result["total"] == 665

    def test_extract_usage_realistic_anthropic_with_caching(self, client):
        """Test extraction from realistic Anthropic API response with caching."""
        usage = {
            "prompt_tokens": 2048,
            "completion_tokens": 512,
            "total_tokens": 2560,
            "cache_creation_input_tokens": 1024,
            "cache_read_input_tokens": 512
        }

        result = client._extract_usage_details(usage)

        # Verify all fields are extracted
        assert result["input"] == 2048
        assert result["output"] == 512
        assert result["total"] == 2560
        assert result["cache_creation_input_tokens"] == 1024
        assert result["cache_read_input_tokens"] == 512

    def test_extract_usage_float_values(self, client):
        """Test handling of float values (some providers might send floats)."""
        usage = {
            "prompt_tokens": 100.0,
            "completion_tokens": 50.5,
            "total_tokens": 150.7
        }

        result = client._extract_usage_details(usage)

        # Should preserve the values as-is (Langfuse will handle the type)
        assert result["input"] == 100.0
        assert result["output"] == 50.5
        assert result["total"] == 150.7

    def test_extract_usage_negative_values(self, client):
        """Test handling of negative values (invalid but should not crash)."""
        usage = {
            "prompt_tokens": -100,
            "completion_tokens": 50,
            "total_tokens": -50
        }

        result = client._extract_usage_details(usage)

        # Should extract even invalid values (let Langfuse handle validation)
        assert result["input"] == -100
        assert result["output"] == 50
        assert result["total"] == -50

    def test_extract_usage_string_number_values(self, client):
        """Test handling of string number values (malformed response)."""
        usage = {
            "prompt_tokens": "100",
            "completion_tokens": "50",
            "total_tokens": "150"
        }

        result = client._extract_usage_details(usage)

        # Should extract string values (Langfuse may handle conversion)
        assert result["input"] == "100"
        assert result["output"] == "50"
        assert result["total"] == "150"
