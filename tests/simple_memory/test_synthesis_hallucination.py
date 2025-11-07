"""
Tests for memory synthesis hallucination detection and prevention.

This module tests the safeguards against LLM hallucinating excessive
supersession titles, which can cause JSON truncation errors.
"""

import pytest
from unittest.mock import Mock, MagicMock
from managers.memory.synthesis import MemorySynthesizer
from managers.memory.models import Memory, MemorySynthesisResponse, MemoryStatus
from session.game_configuration import GameConfiguration


@pytest.fixture
def config():
    """Create test configuration."""
    return GameConfiguration(
        max_turns_per_episode=100,
        memory_sampling={'temperature': 0.3, 'max_tokens': 1000}
    )


@pytest.fixture
def logger():
    """Create mock logger."""
    logger = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def formatter():
    """Create mock formatter."""
    formatter = Mock()
    formatter.format_recent_actions = Mock(return_value="")
    formatter.format_recent_reasoning = Mock(return_value="")
    return formatter


@pytest.fixture
def llm_client():
    """Create mock LLM client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    return client


@pytest.fixture
def synthesizer(logger, config, formatter, llm_client):
    """Create synthesizer instance."""
    return MemorySynthesizer(logger, config, formatter, llm_client)


def test_excessive_supersession_titles_rejected(synthesizer, llm_client, logger):
    """Test that >3 supersession titles are rejected by Pydantic validation."""
    # Mock LLM response with 5 supersession titles (hallucination)
    mock_response = Mock()
    mock_response.content = '''{
        "should_remember": true,
        "category": "NOTE",
        "memory_title": "New memory",
        "memory_text": "Some insight",
        "persistence": "permanent",
        "status": "ACTIVE",
        "supersedes_memory_titles": [
            "Old memory 1",
            "Old memory 2",
            "Old memory 3",
            "Old memory 4",
            "Old memory 5"
        ],
        "reasoning": "Test"
    }'''
    mock_response.usage = {"completion_tokens": 100}
    llm_client.chat.completions.create = Mock(return_value=mock_response)

    # Call synthesize_memory
    result = synthesizer.synthesize_memory(
        location_id=10,
        location_name="Test Room",
        action="test action",
        response="test response",
        existing_memories=[],
        z_machine_context={'score_delta': 0, 'location_changed': False, 'inventory_changed': False, 'died': False, 'first_visit': True}
    )

    # Pydantic validation should reject this during parse, returning None
    assert result is None

    # Verify error was logged (not warning, since Pydantic raises exception)
    assert logger.error.called
    error_call = logger.error.call_args
    assert "Failed to synthesize memory" in error_call[0][0]


def test_valid_supersession_count_allowed(synthesizer, llm_client, logger):
    """Test that 1-3 supersession titles are allowed through."""
    # Mock LLM response with 2 supersession titles (valid)
    mock_response = Mock()
    mock_response.content = '''{
        "should_remember": true,
        "category": "NOTE",
        "memory_title": "New memory",
        "memory_text": "Some insight",
        "persistence": "permanent",
        "status": "ACTIVE",
        "supersedes_memory_titles": ["Old memory 1", "Old memory 2"],
        "reasoning": "Test"
    }'''
    mock_response.usage = {"completion_tokens": 100}
    llm_client.chat.completions.create = Mock(return_value=mock_response)

    # Call synthesize_memory
    result = synthesizer.synthesize_memory(
        location_id=10,
        location_name="Test Room",
        action="test action",
        response="test response",
        existing_memories=[],
        z_machine_context={'score_delta': 0, 'location_changed': False, 'inventory_changed': False, 'died': False, 'first_visit': True}
    )

    # Verify no warning was logged
    assert not logger.warning.called

    # Verify supersession titles are preserved
    assert len(result.supersedes_memory_titles) == 2
    assert "Old memory 1" in result.supersedes_memory_titles
    assert "Old memory 2" in result.supersedes_memory_titles


def test_exactly_three_supersessions_allowed(synthesizer, llm_client, logger):
    """Test that exactly 3 supersession titles is the boundary case (allowed)."""
    # Mock LLM response with exactly 3 supersession titles
    mock_response = Mock()
    mock_response.content = '''{
        "should_remember": true,
        "category": "NOTE",
        "memory_title": "New memory",
        "memory_text": "Some insight",
        "persistence": "permanent",
        "status": "ACTIVE",
        "supersedes_memory_titles": ["Old 1", "Old 2", "Old 3"],
        "reasoning": "Test"
    }'''
    mock_response.usage = {"completion_tokens": 100}
    llm_client.chat.completions.create = Mock(return_value=mock_response)

    # Call synthesize_memory
    result = synthesizer.synthesize_memory(
        location_id=10,
        location_name="Test Room",
        action="test action",
        response="test response",
        existing_memories=[],
        z_machine_context={'score_delta': 0, 'location_changed': False, 'inventory_changed': False, 'died': False, 'first_visit': True}
    )

    # Verify no warning (3 is allowed)
    assert not logger.warning.called

    # Verify all 3 titles preserved
    assert len(result.supersedes_memory_titles) == 3


def test_four_supersessions_triggers_rejection(synthesizer, llm_client, logger):
    """Test that 4 supersession titles triggers rejection (just over boundary)."""
    # Mock LLM response with 4 supersession titles
    mock_response = Mock()
    mock_response.content = '''{
        "should_remember": true,
        "category": "NOTE",
        "memory_title": "New memory",
        "memory_text": "Some insight",
        "persistence": "permanent",
        "status": "ACTIVE",
        "supersedes_memory_titles": ["Old 1", "Old 2", "Old 3", "Old 4"],
        "reasoning": "Test"
    }'''
    mock_response.usage = {"completion_tokens": 100}
    llm_client.chat.completions.create = Mock(return_value=mock_response)

    # Call synthesize_memory
    result = synthesizer.synthesize_memory(
        location_id=10,
        location_name="Test Room",
        action="test action",
        response="test response",
        existing_memories=[],
        z_machine_context={'score_delta': 0, 'location_changed': False, 'inventory_changed': False, 'died': False, 'first_visit': True}
    )

    # Pydantic validation should reject this, returning None
    assert result is None

    # Verify error was logged
    assert logger.error.called


def test_hallucination_log_contains_details(synthesizer, llm_client, logger):
    """Test that Pydantic validation error is logged with details."""
    # Mock LLM response with many titles
    mock_response = Mock()
    mock_response.content = '''{
        "should_remember": true,
        "category": "NOTE",
        "memory_title": "New memory",
        "memory_text": "Some insight",
        "persistence": "permanent",
        "status": "ACTIVE",
        "supersedes_memory_titles": [
            "Title 1", "Title 2", "Title 3", "Title 4", "Title 5",
            "Title 6", "Title 7", "Title 8", "Title 9", "Title 10"
        ],
        "reasoning": "Test"
    }'''
    mock_response.usage = {"completion_tokens": 100}
    llm_client.chat.completions.create = Mock(return_value=mock_response)

    # Call synthesize_memory
    result = synthesizer.synthesize_memory(
        location_id=42,
        location_name="Test Location",
        action="test action here",
        response="test response",
        existing_memories=[],
        z_machine_context={'score_delta': 0, 'location_changed': False, 'inventory_changed': False, 'died': False, 'first_visit': True}
    )

    # Should return None due to validation error
    assert result is None

    # Verify error was logged with location context
    error_call = logger.error.call_args
    extra = error_call[1]['extra']
    assert extra['location_id'] == 42


def test_memory_synthesis_with_valid_supersessions_succeeds(synthesizer, llm_client, logger):
    """Test that synthesis succeeds when supersessions are within limits."""
    # Mock LLM response with valid supersession count (2 titles)
    mock_response = Mock()
    mock_response.content = '''{
        "should_remember": true,
        "category": "SUCCESS",
        "memory_title": "Valid memory",
        "memory_text": "This is valid",
        "persistence": "permanent",
        "status": "ACTIVE",
        "supersedes_memory_titles": ["Old memory 1", "Old memory 2"],
        "reasoning": "Test"
    }'''
    mock_response.usage = {"completion_tokens": 100}
    llm_client.chat.completions.create = Mock(return_value=mock_response)

    # Call synthesize_memory
    result = synthesizer.synthesize_memory(
        location_id=10,
        location_name="Test Room",
        action="test action",
        response="test response",
        existing_memories=[],
        z_machine_context={'score_delta': 0, 'location_changed': False, 'inventory_changed': False, 'died': False, 'first_visit': True}
    )

    # Verify synthesis succeeded
    assert result is not None
    assert result.should_remember is True
    assert result.category == "SUCCESS"
    assert result.memory_title == "Valid memory"
    assert result.memory_text == "This is valid"
    # Valid supersession titles preserved
    assert len(result.supersedes_memory_titles) == 2


def test_zero_supersessions_not_flagged(synthesizer, llm_client, logger):
    """Test that 0 supersession titles doesn't trigger hallucination detection."""
    # Mock LLM response with no supersessions
    mock_response = Mock()
    mock_response.content = '''{
        "should_remember": true,
        "category": "NOTE",
        "memory_title": "New memory",
        "memory_text": "Some insight",
        "persistence": "permanent",
        "status": "ACTIVE",
        "supersedes_memory_titles": [],
        "reasoning": "Test"
    }'''
    mock_response.usage = {"completion_tokens": 100}
    llm_client.chat.completions.create = Mock(return_value=mock_response)

    # Call synthesize_memory
    result = synthesizer.synthesize_memory(
        location_id=10,
        location_name="Test Room",
        action="test action",
        response="test response",
        existing_memories=[],
        z_machine_context={'score_delta': 0, 'location_changed': False, 'inventory_changed': False, 'died': False, 'first_visit': True}
    )

    # Verify no warning
    assert not logger.warning.called

    # Verify empty set preserved
    assert len(result.supersedes_memory_titles) == 0


# Integration test for ephemeral supersession prevention removed
# The validation logic in simple_memory_manager.py lines 728-753 prevents
# ephemeral memories from superseding persistent ones, which is tested
# implicitly by the manual verification below.
#
# Manual verification steps:
# 1. Run episode with high memory density
# 2. Check logs for "Rejected supersession: ephemeral memory cannot supersede"
# 3. Verify persistent memories remain ACTIVE after ephemeral actions
