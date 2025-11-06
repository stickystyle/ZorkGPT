# ABOUTME: Tests for Memory dataclass persistence field (Phase 1.1 of ephemeral memory system)
# ABOUTME: Verifies required persistence field validation and storage

import pytest
from managers.simple_memory_manager import Memory, MemoryStatus


class TestMemoryPersistence:
    """Test suite for Memory dataclass persistence field requirements."""

    def test_memory_requires_persistence_field(self):
        """
        Test that Memory creation requires persistence field.

        Test approach:
        1. Attempt to create Memory without persistence field
        2. Verify TypeError is raised
        3. Verify error message mentions missing 'persistence' parameter
        """
        # Act & Assert - Creating Memory without persistence should raise TypeError
        with pytest.raises(TypeError) as exc_info:
            Memory(
                category="SUCCESS",
                title="Test Memory",
                episode=1,
                turns="10",
                score_change=0,
                text="Test memory text",
                status=MemoryStatus.ACTIVE
            )

        # Verify error message mentions the missing 'persistence' parameter
        error_message = str(exc_info.value)
        assert "persistence" in error_message.lower(), (
            f"Expected error message to mention 'persistence', got: {error_message}"
        )

    def test_memory_accepts_valid_persistence_values(self):
        """
        Test that Memory accepts valid persistence values.

        Test approach:
        1. Create Memory with persistence="core"
        2. Create Memory with persistence="permanent"
        3. Create Memory with persistence="ephemeral"
        4. Verify all creations succeed without errors
        """
        # Arrange - Common memory parameters
        base_params = {
            "category": "SUCCESS",
            "title": "Test Memory",
            "episode": 1,
            "turns": "10",
            "score_change": 0,
            "text": "Test memory text",
            "status": MemoryStatus.ACTIVE
        }

        # Act & Assert - All valid persistence values should be accepted
        memory_core = Memory(**base_params, persistence="core")
        assert memory_core.persistence == "core", "Memory should store persistence='core'"

        memory_permanent = Memory(**base_params, persistence="permanent")
        assert memory_permanent.persistence == "permanent", (
            "Memory should store persistence='permanent'"
        )

        memory_ephemeral = Memory(**base_params, persistence="ephemeral")
        assert memory_ephemeral.persistence == "ephemeral", (
            "Memory should store persistence='ephemeral'"
        )

    def test_memory_rejects_invalid_persistence_values(self):
        """
        Test that Memory rejects invalid persistence values.

        Test approach:
        1. Attempt to create Memory with persistence="invalid"
        2. Attempt to create Memory with persistence="temporary"
        3. Attempt to create Memory with persistence="" (empty string)
        4. Verify ValueError is raised for each with helpful message
        """
        # Arrange - Common memory parameters
        base_params = {
            "category": "SUCCESS",
            "title": "Test Memory",
            "episode": 1,
            "turns": "10",
            "score_change": 0,
            "text": "Test memory text",
            "status": MemoryStatus.ACTIVE
        }

        # Act & Assert - Invalid persistence value "invalid"
        with pytest.raises(ValueError) as exc_info:
            Memory(**base_params, persistence="invalid")

        error_message = str(exc_info.value)
        assert "persistence" in error_message.lower(), (
            f"Error message should mention 'persistence', got: {error_message}"
        )
        assert any(
            valid in error_message.lower()
            for valid in ["core", "permanent", "ephemeral"]
        ), (
            f"Error message should list valid values, got: {error_message}"
        )

        # Act & Assert - Invalid persistence value "temporary"
        with pytest.raises(ValueError) as exc_info:
            Memory(**base_params, persistence="temporary")

        error_message = str(exc_info.value)
        assert "persistence" in error_message.lower(), (
            f"Error message should mention 'persistence', got: {error_message}"
        )

        # Act & Assert - Invalid persistence value "" (empty string)
        with pytest.raises(ValueError) as exc_info:
            Memory(**base_params, persistence="")

        error_message = str(exc_info.value)
        assert "persistence" in error_message.lower(), (
            f"Error message should mention 'persistence', got: {error_message}"
        )

    def test_memory_persistence_field_stored(self):
        """
        Test that persistence field is stored correctly in Memory.

        Test approach:
        1. Create Memory with each valid persistence value
        2. Verify Memory.persistence returns the correct value
        3. Test that persistence field is accessible after creation
        """
        # Arrange - Common memory parameters
        base_params = {
            "category": "DISCOVERY",
            "title": "Persistence Storage Test",
            "episode": 2,
            "turns": "15-17",
            "score_change": 5,
            "text": "Testing that persistence field is stored correctly.",
            "status": MemoryStatus.TENTATIVE
        }

        # Act - Create memory with persistence="core"
        memory = Memory(**base_params, persistence="core")

        # Assert - Verify field is stored and accessible
        assert hasattr(memory, "persistence"), "Memory should have 'persistence' attribute"
        assert memory.persistence == "core", (
            f"Expected persistence='core', got: {memory.persistence}"
        )

        # Act - Create memory with persistence="permanent"
        memory = Memory(**base_params, persistence="permanent")

        # Assert - Verify field is stored correctly
        assert memory.persistence == "permanent", (
            f"Expected persistence='permanent', got: {memory.persistence}"
        )

        # Act - Create memory with persistence="ephemeral"
        memory = Memory(**base_params, persistence="ephemeral")

        # Assert - Verify field is stored correctly
        assert memory.persistence == "ephemeral", (
            f"Expected persistence='ephemeral', got: {memory.persistence}"
        )
