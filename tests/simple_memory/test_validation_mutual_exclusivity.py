"""
Test mutual exclusivity validation in MemorySynthesisResponse.

Tests that memory titles cannot be both superseded and invalidated simultaneously,
which would create ambiguous state.
"""

import pytest
from pydantic import ValidationError

from managers.simple_memory_manager import MemorySynthesisResponse, MemoryStatus


class TestMutualExclusivityValidation:
    """Test that supersession and invalidation are mutually exclusive for same titles."""

    def test_supersede_and_invalidate_different_titles_allowed(self):
        """Different titles can be superseded and invalidated in same response."""
        response = MemorySynthesisResponse(
            should_remember=True,
            category="SUCCESS",
            memory_title="New memory",
            memory_text="Some text",
            status=MemoryStatus.ACTIVE,
            supersedes_memory_titles={"Old memory 1"},
            invalidate_memory_titles={"Old memory 2"},
            invalidation_reason="Memory 2 is wrong"
        )

        # Should succeed - no overlap
        assert response.supersedes_memory_titles == {"Old memory 1"}
        assert response.invalidate_memory_titles == {"Old memory 2"}

    def test_supersede_and_invalidate_same_title_raises_error(self):
        """Same title in both supersedes and invalidate raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySynthesisResponse(
                should_remember=True,
                category="SUCCESS",
                memory_title="New memory",
                memory_text="Some text",
                status=MemoryStatus.ACTIVE,
                supersedes_memory_titles={"Old memory"},
                invalidate_memory_titles={"Old memory"},  # CONFLICT
                invalidation_reason="Memory is wrong"
            )

        # Check error message contains the conflicting title
        error_message = str(exc_info.value)
        assert "cannot be both superseded and invalidated" in error_message
        assert "Old memory" in error_message

    def test_multiple_overlapping_titles_raises_error(self):
        """Multiple overlapping titles all reported in error."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySynthesisResponse(
                should_remember=True,
                category="SUCCESS",
                memory_title="New memory",
                memory_text="Some text",
                status=MemoryStatus.ACTIVE,
                supersedes_memory_titles={"Memory A", "Memory B", "Memory C"},
                invalidate_memory_titles={"Memory B", "Memory C", "Memory D"},  # B and C overlap
                invalidation_reason="Some memories are wrong"
            )

        error_message = str(exc_info.value)
        assert "cannot be both superseded and invalidated" in error_message
        # Should mention both overlapping titles (set, so order may vary)
        assert "Memory B" in error_message
        assert "Memory C" in error_message

    def test_only_supersedes_no_invalidate_allowed(self):
        """Only superseding without invalidation is allowed."""
        response = MemorySynthesisResponse(
            should_remember=True,
            category="SUCCESS",
            memory_title="New memory",
            memory_text="Some text",
            status=MemoryStatus.ACTIVE,
            supersedes_memory_titles={"Old memory 1", "Old memory 2"},
            invalidate_memory_titles=set(),  # Empty
            invalidation_reason=None
        )

        assert response.supersedes_memory_titles == {"Old memory 1", "Old memory 2"}
        assert response.invalidate_memory_titles == set()

    def test_only_invalidate_no_supersedes_allowed(self):
        """Only invalidating without supersession is allowed."""
        response = MemorySynthesisResponse(
            should_remember=True,
            category="SUCCESS",
            memory_title="New memory",
            memory_text="Some text",
            status=MemoryStatus.ACTIVE,
            supersedes_memory_titles=set(),  # Empty
            invalidate_memory_titles={"Wrong memory"},
            invalidation_reason="This memory is incorrect"
        )

        assert response.supersedes_memory_titles == set()
        assert response.invalidate_memory_titles == {"Wrong memory"}

    def test_neither_supersedes_nor_invalidate_allowed(self):
        """Neither supersession nor invalidation is also valid."""
        response = MemorySynthesisResponse(
            should_remember=True,
            category="DISCOVERY",
            memory_title="New discovery",
            memory_text="Found something new",
            status=MemoryStatus.ACTIVE,
            supersedes_memory_titles=set(),
            invalidate_memory_titles=set()
        )

        assert response.supersedes_memory_titles == set()
        assert response.invalidate_memory_titles == set()

    def test_invalidation_reason_still_required_with_invalidate_titles(self):
        """Existing validation: invalidation_reason required when invalidate_memory_titles not empty."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySynthesisResponse(
                should_remember=True,
                category="SUCCESS",
                memory_title="New memory",
                memory_text="Some text",
                status=MemoryStatus.ACTIVE,
                invalidate_memory_titles={"Wrong memory"},
                invalidation_reason=None  # Missing required field
            )

        error_message = str(exc_info.value)
        assert "invalidation_reason must be non-empty" in error_message

    def test_invalidation_reason_rejects_whitespace_only(self):
        """Validation should reject whitespace-only invalidation_reason."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySynthesisResponse(
                should_remember=True,
                category="SUCCESS",
                memory_title="New memory",
                memory_text="Some text",
                status=MemoryStatus.ACTIVE,
                invalidate_memory_titles={"Wrong memory"},
                invalidation_reason="   "  # Whitespace-only
            )

        error_message = str(exc_info.value)
        assert "invalidation_reason must be non-empty" in error_message

    def test_invalidation_reason_rejects_empty_string(self):
        """Validation should reject empty string invalidation_reason."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySynthesisResponse(
                should_remember=True,
                category="SUCCESS",
                memory_title="New memory",
                memory_text="Some text",
                status=MemoryStatus.ACTIVE,
                invalidate_memory_titles={"Wrong memory"},
                invalidation_reason=""  # Empty string
            )

        error_message = str(exc_info.value)
        assert "invalidation_reason must be non-empty" in error_message


class TestInvalidationMarkerConstant:
    """Test INVALIDATION_MARKER constant is accessible at module level."""

    def test_invalidation_marker_accessible(self):
        """INVALIDATION_MARKER should be accessible from module."""
        from managers.simple_memory_manager import INVALIDATION_MARKER

        assert INVALIDATION_MARKER == "INVALIDATED"
        assert isinstance(INVALIDATION_MARKER, str)

    def test_invalidation_marker_not_in_memory_status_class(self):
        """INVALIDATION_MARKER should not be an attribute of MemoryStatus class."""
        from managers.simple_memory_manager import MemoryStatus

        # Should not have INVALIDATION_MARKER as class attribute
        assert not hasattr(MemoryStatus, "INVALIDATION_MARKER")

    def test_memory_status_only_has_valid_status_constants(self):
        """MemoryStatus class should only have ACTIVE, TENTATIVE, SUPERSEDED."""
        from managers.simple_memory_manager import MemoryStatus

        # Should have these three
        assert hasattr(MemoryStatus, "ACTIVE")
        assert hasattr(MemoryStatus, "TENTATIVE")
        assert hasattr(MemoryStatus, "SUPERSEDED")

        # Values should be strings
        assert MemoryStatus.ACTIVE == "ACTIVE"
        assert MemoryStatus.TENTATIVE == "TENTATIVE"
        assert MemoryStatus.SUPERSEDED == "SUPERSEDED"
