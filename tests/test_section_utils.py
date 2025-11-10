# ABOUTME: Tests for knowledge base section manipulation utilities
# ABOUTME: Tests extract_section_content, update_section_content, and remove_section

"""
Tests for knowledge/section_utils.py

Tests the section manipulation functions for knowledge base content including:
- Extracting section content
- Updating section content
- Removing sections
"""

import pytest
from knowledge import section_utils


class TestRemoveSection:
    """Tests for remove_section() function."""

    def test_remove_existing_section(self):
        """Test removing an existing section from knowledge base."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## PUZZLE SOLUTIONS
Window entry requires open then enter.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        result = section_utils.remove_section(knowledge, "PUZZLE SOLUTIONS")

        # Section should be removed
        assert "## PUZZLE SOLUTIONS" not in result
        assert "Window entry requires" not in result

        # Other sections should be preserved
        assert "## DANGERS & THREATS" in result
        assert "Grue attacks in dark locations." in result
        assert "## STRATEGIC PATTERNS" in result
        assert "Examine before taking objects." in result

    def test_remove_missing_section(self):
        """Test removing a section that doesn't exist returns unchanged content."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## PUZZLE SOLUTIONS
Window entry requires open then enter.
"""

        result = section_utils.remove_section(knowledge, "NONEXISTENT SECTION")

        # Content should be unchanged
        assert result == knowledge

    def test_remove_from_empty_knowledge(self):
        """Test removing section from empty knowledge returns empty string."""
        result = section_utils.remove_section("", "SOME SECTION")
        assert result == ""

    def test_remove_first_section(self):
        """Test removing the first section preserves sections after it."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## PUZZLE SOLUTIONS
Window entry requires open then enter.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        result = section_utils.remove_section(knowledge, "DANGERS & THREATS")

        # First section should be removed
        assert "## DANGERS & THREATS" not in result
        assert "Grue attacks" not in result

        # Subsequent sections should be preserved
        assert "## PUZZLE SOLUTIONS" in result
        assert "Window entry" in result
        assert "## STRATEGIC PATTERNS" in result
        assert "Examine before" in result

    def test_remove_last_section(self):
        """Test removing the last section preserves sections before it."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## PUZZLE SOLUTIONS
Window entry requires open then enter.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        result = section_utils.remove_section(knowledge, "STRATEGIC PATTERNS")

        # Last section should be removed
        assert "## STRATEGIC PATTERNS" not in result
        assert "Examine before" not in result

        # Previous sections should be preserved
        assert "## DANGERS & THREATS" in result
        assert "Grue attacks" in result
        assert "## PUZZLE SOLUTIONS" in result
        assert "Window entry" in result

    def test_remove_middle_section(self):
        """Test removing a middle section preserves sections before and after."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## PUZZLE SOLUTIONS
Window entry requires open then enter.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        result = section_utils.remove_section(knowledge, "PUZZLE SOLUTIONS")

        # Middle section should be removed
        assert "## PUZZLE SOLUTIONS" not in result
        assert "Window entry" not in result

        # Sections before and after should be preserved
        assert "## DANGERS & THREATS" in result
        assert "Grue attacks" in result
        assert "## STRATEGIC PATTERNS" in result
        assert "Examine before" in result

    def test_remove_section_with_special_characters(self):
        """Test removing section with special regex characters in name."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS (UPDATED)
Grue attacks in dark locations.

## PUZZLE SOLUTIONS
Window entry requires open then enter.
"""

        result = section_utils.remove_section(knowledge, "DANGERS & THREATS (UPDATED)")

        # Section with special characters should be removed
        assert "## DANGERS & THREATS (UPDATED)" not in result
        assert "Grue attacks" not in result

        # Other sections should be preserved
        assert "## PUZZLE SOLUTIONS" in result

    def test_remove_cross_episode_insights_section(self):
        """Test removing CROSS-EPISODE INSIGHTS section (main use case)."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## CROSS-EPISODE INSIGHTS
This is persistent wisdom that should be preserved.
Multiple lines of cross-episode content here.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        result = section_utils.remove_section(knowledge, "CROSS-EPISODE INSIGHTS")

        # CROSS-EPISODE INSIGHTS should be removed
        assert "## CROSS-EPISODE INSIGHTS" not in result
        assert "persistent wisdom" not in result
        assert "Multiple lines of cross-episode" not in result

        # Other sections should be preserved
        assert "## DANGERS & THREATS" in result
        assert "Grue attacks" in result
        assert "## STRATEGIC PATTERNS" in result
        assert "Examine before" in result


class TestExtractSectionContent:
    """Tests for extract_section_content() function."""

    def test_extract_existing_section(self):
        """Test extracting content from an existing section."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## PUZZLE SOLUTIONS
Window entry requires open then enter.
"""

        result = section_utils.extract_section_content(knowledge, "DANGERS & THREATS")

        assert "Grue attacks in dark locations." in result
        assert "##" not in result  # Header should not be in extracted content

    def test_extract_missing_section(self):
        """Test extracting non-existent section returns empty string."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.
"""

        result = section_utils.extract_section_content(knowledge, "NONEXISTENT")
        assert result == ""

    def test_extract_from_empty_knowledge(self):
        """Test extracting from empty knowledge returns empty string."""
        result = section_utils.extract_section_content("", "SOME SECTION")
        assert result == ""


class TestUpdateSectionContent:
    """Tests for update_section_content() function."""

    def test_update_existing_section(self):
        """Test updating an existing section."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Old content here.

## PUZZLE SOLUTIONS
Window entry requires open then enter.
"""

        new_content = "Updated content about dangers."
        result = section_utils.update_section_content(knowledge, "DANGERS & THREATS", new_content)

        assert "Updated content about dangers." in result
        assert "Old content here." not in result
        assert "## PUZZLE SOLUTIONS" in result  # Other sections preserved

    def test_add_new_section(self):
        """Test adding a new section that doesn't exist."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.
"""

        new_content = "New strategic patterns."
        result = section_utils.update_section_content(knowledge, "STRATEGIC PATTERNS", new_content)

        assert "## STRATEGIC PATTERNS" in result
        assert "New strategic patterns." in result
        assert "## DANGERS & THREATS" in result  # Existing sections preserved

    def test_update_empty_knowledge(self):
        """Test updating section in empty knowledge creates initial structure."""
        new_content = "Some danger content."
        result = section_utils.update_section_content("", "DANGERS & THREATS", new_content)

        assert "# Zork Game World Knowledge Base" in result
        assert "## DANGERS & THREATS" in result
        assert "Some danger content." in result


class TestRemoveAndRestoreWorkflow:
    """Tests simulating the remove/restore workflow for CROSS-EPISODE preservation."""

    def test_preserve_and_restore_cross_episode(self):
        """Test the full workflow: extract, remove, process, restore."""
        original_knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## CROSS-EPISODE INSIGHTS
Important cross-episode wisdom here.
Should be preserved exactly.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        # Step 1: Extract CROSS-EPISODE INSIGHTS
        preserved = section_utils.extract_section_content(
            original_knowledge, "CROSS-EPISODE INSIGHTS"
        )
        assert "Important cross-episode wisdom here." in preserved

        # Step 2: Remove CROSS-EPISODE INSIGHTS from knowledge
        without_cross_episode = section_utils.remove_section(
            original_knowledge, "CROSS-EPISODE INSIGHTS"
        )
        assert "## CROSS-EPISODE INSIGHTS" not in without_cross_episode
        assert "Important cross-episode wisdom" not in without_cross_episode

        # Step 3: Simulate LLM regenerating knowledge (might include CROSS-EPISODE)
        llm_generated = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations - updated by LLM.

## CROSS-EPISODE INSIGHTS
LLM tried to regenerate this - should be removed.

## STRATEGIC PATTERNS
Examine before taking objects - updated by LLM.
"""

        # Step 4: Defensively remove any CROSS-EPISODE the LLM generated
        cleaned = section_utils.remove_section(
            llm_generated, "CROSS-EPISODE INSIGHTS"
        )
        assert "LLM tried to regenerate" not in cleaned

        # Step 5: Restore the preserved CROSS-EPISODE INSIGHTS
        final = section_utils.update_section_content(
            cleaned, "CROSS-EPISODE INSIGHTS", preserved
        )

        # Verify: ORIGINAL cross-episode content restored
        assert "Important cross-episode wisdom here." in final
        assert "Should be preserved exactly." in final
        assert "LLM tried to regenerate" not in final

        # Verify: LLM updates to other sections preserved
        assert "updated by LLM" in final

    def test_preserve_when_no_existing_cross_episode(self):
        """Test workflow when there's no existing CROSS-EPISODE INSIGHTS."""
        knowledge = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks in dark locations.

## STRATEGIC PATTERNS
Examine before taking objects.
"""

        # Step 1: Try to extract CROSS-EPISODE (should be empty)
        preserved = section_utils.extract_section_content(
            knowledge, "CROSS-EPISODE INSIGHTS"
        )
        assert preserved == ""

        # Step 2: Remove (should be no-op)
        without_cross_episode = section_utils.remove_section(
            knowledge, "CROSS-EPISODE INSIGHTS"
        )
        assert without_cross_episode == knowledge

        # Step 3: Simulate LLM generation (no CROSS-EPISODE section)
        llm_generated = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Grue attacks - updated.

## STRATEGIC PATTERNS
Examine before taking - updated.
"""

        # Step 4: Defensively remove (should be no-op)
        cleaned = section_utils.remove_section(
            llm_generated, "CROSS-EPISODE INSIGHTS"
        )

        # Step 5: Only restore if preserved content exists
        if preserved:
            final = section_utils.update_section_content(
                cleaned, "CROSS-EPISODE INSIGHTS", preserved
            )
        else:
            final = cleaned

        # Verify: No CROSS-EPISODE section in final result
        assert "## CROSS-EPISODE INSIGHTS" not in final
        assert final == llm_generated
