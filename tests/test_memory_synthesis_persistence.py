# ABOUTME: Tests for MemorySynthesisResponse persistence field validation (Phase 1.2 of ephemeral memory system)
# ABOUTME: Verifies Pydantic validator logic for conditional persistence requirement

import pytest
from pydantic import ValidationError
from managers.simple_memory_manager import MemorySynthesisResponse, MemoryStatus


class TestMemorySynthesisResponsePersistence:
    """Test suite for MemorySynthesisResponse persistence field validation."""

    def test_persistence_required_when_creating_memory(self):
        """Test that persistence is required when should_remember=True."""
        # Attempt to create memory without persistence field should fail
        with pytest.raises(ValidationError) as exc_info:
            MemorySynthesisResponse(
                should_remember=True,
                category="observation",
                memory_title="Test Memory",
                memory_text="This is a test memory",
                reasoning="Test reasoning"
            )

        error_msg = str(exc_info.value)
        assert "persistence required" in error_msg.lower(), (
            f"Expected error about 'persistence required', got: {error_msg}"
        )

        # Explicitly passing persistence=None should also fail
        with pytest.raises(ValidationError) as exc_info:
            MemorySynthesisResponse(
                should_remember=True,
                category="observation",
                memory_title="Test Memory",
                memory_text="This is a test memory",
                persistence=None,
                reasoning="Test reasoning"
            )

        error_msg = str(exc_info.value)
        assert "persistence required" in error_msg.lower(), (
            f"Expected error about 'persistence required', got: {error_msg}"
        )

    def test_persistence_optional_when_not_creating_memory(self):
        """Test that persistence can be None when should_remember=False."""
        # Without persistence field - should work
        response = MemorySynthesisResponse(
            should_remember=False,
            reasoning="Not relevant enough to remember"
        )
        assert response.should_remember is False
        assert response.persistence is None, (
            f"Expected persistence=None when should_remember=False, got: {response.persistence}"
        )

        # With explicit persistence=None - should also work
        response = MemorySynthesisResponse(
            should_remember=False,
            persistence=None,
            reasoning="Not relevant enough to remember"
        )
        assert response.should_remember is False
        assert response.persistence is None, (
            f"Expected persistence=None when should_remember=False, got: {response.persistence}"
        )

    def test_valid_persistence_values_accepted(self):
        """Test that valid persistence values (core, permanent, ephemeral) are accepted."""
        valid_values = ["core", "permanent", "ephemeral"]

        for persistence_value in valid_values:
            response = MemorySynthesisResponse(
                should_remember=True,
                category="observation",
                memory_title=f"Test {persistence_value} Memory",
                memory_text=f"This is a {persistence_value} memory",
                persistence=persistence_value,
                reasoning="Test reasoning"
            )

            assert response.should_remember is True, (
                f"Expected should_remember=True for persistence={persistence_value}"
            )
            assert response.persistence == persistence_value, (
                f"Expected persistence={persistence_value}, got: {response.persistence}"
            )

    def test_invalid_persistence_values_rejected(self):
        """Test that invalid persistence values are rejected with clear error message."""
        invalid_values = ["invalid", "temporary", "", "CORE", "Permanent"]

        for invalid_value in invalid_values:
            with pytest.raises(ValidationError) as exc_info:
                MemorySynthesisResponse(
                    should_remember=True,
                    category="observation",
                    memory_title="Test Memory",
                    memory_text="This is a test memory",
                    persistence=invalid_value,
                    reasoning="Test reasoning"
                )

            error_msg = str(exc_info.value).lower()

            # Check that error mentions valid values
            assert "core" in error_msg or "permanent" in error_msg or "ephemeral" in error_msg, (
                f"Expected error to list valid persistence values for input '{invalid_value}', got: {exc_info.value}"
            )

            # Check that error mentions the invalid value or persistence field
            assert invalid_value.lower() in error_msg or "persistence" in error_msg, (
                f"Expected error to mention invalid value '{invalid_value}' or 'persistence', got: {exc_info.value}"
            )

    def test_persistence_field_stored(self):
        """Test that persistence field is stored correctly for all valid values."""
        test_cases = [
            ("core", "Core memory - fundamental game mechanics"),
            ("permanent", "Permanent memory - important discovery"),
            ("ephemeral", "Ephemeral memory - transient observation")
        ]

        for persistence_value, description in test_cases:
            response = MemorySynthesisResponse(
                should_remember=True,
                category="observation",
                memory_title=f"{persistence_value.capitalize()} Test",
                memory_text=description,
                persistence=persistence_value,
                reasoning=f"Testing {persistence_value} persistence"
            )

            # Verify field is stored exactly as provided
            assert response.persistence == persistence_value, (
                f"Expected persistence to be stored as '{persistence_value}', got: '{response.persistence}'"
            )

            # Verify other fields are also stored correctly
            assert response.should_remember is True
            assert response.memory_title == f"{persistence_value.capitalize()} Test"
            assert response.memory_text == description

            # Verify field can be accessed via dict
            response_dict = response.dict()
            assert response_dict["persistence"] == persistence_value, (
                f"Expected persistence in dict to be '{persistence_value}', got: '{response_dict['persistence']}'"
            )
