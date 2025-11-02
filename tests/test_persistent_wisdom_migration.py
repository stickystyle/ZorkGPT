"""
ABOUTME: Tests for persistent_wisdom.md to knowledgebase.md migration functionality.
ABOUTME: Validates helper methods for section extraction and cross-episode insights.
"""

import pytest
import re
from unittest.mock import Mock
from knowledge import AdaptiveKnowledgeManager


class TestSectionExtractionMethods:
    """Test suite for section extraction helper methods."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager instance."""
        log_file = tmp_path / "test_log.jsonl"
        output_file = tmp_path / "test_knowledge.md"

        # Create minimal config files to avoid errors
        workdir = tmp_path / "game_files"
        workdir.mkdir(exist_ok=True)

        manager = AdaptiveKnowledgeManager(
            log_file=str(log_file),
            output_file=str(output_file),
            logger=Mock(),
            workdir=str(workdir),
        )
        return manager

    def test_extract_cross_episode_section_with_content(self, manager):
        """Test extracting CROSS-EPISODE INSIGHTS section from knowledge base."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Some danger info here.

## CROSS-EPISODE INSIGHTS
This is cross-episode wisdom.
It spans multiple lines.

### Death Patterns
Grues are dangerous in the dark.

## CURRENT WORLD MAP
Map goes here.
"""
        result = manager._extract_cross_episode_section(knowledge_content)

        assert "CROSS-EPISODE INSIGHTS FROM PREVIOUS EPISODES:" in result
        assert "This is cross-episode wisdom." in result
        assert "Grues are dangerous in the dark." in result
        assert "Map goes here" not in result  # Should not include next section

    def test_extract_cross_episode_section_empty(self, manager):
        """Test extracting when CROSS-EPISODE INSIGHTS section doesn't exist."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Some danger info here.

## CURRENT WORLD MAP
Map goes here.
"""
        result = manager._extract_cross_episode_section(knowledge_content)

        assert result == ""

    def test_extract_cross_episode_section_empty_content(self, manager):
        """Test extracting when section exists but is empty."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Some danger info here.

## CROSS-EPISODE INSIGHTS

## CURRENT WORLD MAP
Map goes here.
"""
        result = manager._extract_cross_episode_section(knowledge_content)

        # Empty section should return empty string
        assert result == ""

    def test_extract_section_content_generic(self, manager):
        """Test extracting any section by name."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Danger line 1.
Danger line 2.

## PUZZLE SOLUTIONS
Puzzle line 1.
Puzzle line 2.

## CURRENT WORLD MAP
Map goes here.
"""
        # Extract DANGERS section
        dangers = manager._extract_section_content(knowledge_content, "DANGERS & THREATS")
        assert "Danger line 1." in dangers
        assert "Danger line 2." in dangers
        assert "Puzzle line" not in dangers

        # Extract PUZZLE section
        puzzles = manager._extract_section_content(knowledge_content, "PUZZLE SOLUTIONS")
        assert "Puzzle line 1." in puzzles
        assert "Puzzle line 2." in puzzles
        assert "Danger line" not in puzzles

    def test_extract_section_content_nonexistent(self, manager):
        """Test extracting a section that doesn't exist."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Some danger info here.
"""
        result = manager._extract_section_content(
            knowledge_content, "NONEXISTENT SECTION"
        )
        assert result == ""

    def test_update_section_content_replace_existing(self, manager):
        """Test updating an existing section."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Old danger info.

## PUZZLE SOLUTIONS
Puzzle info.

## CURRENT WORLD MAP
Map goes here.
"""
        new_content = "New danger info.\nUpdated content."

        updated = manager._update_section_content(
            knowledge_content, "DANGERS & THREATS", new_content
        )

        assert "New danger info." in updated
        assert "Updated content." in updated
        assert "Old danger info." not in updated
        assert "Puzzle info." in updated  # Other sections preserved
        assert "Map goes here." in updated

    def test_update_section_content_add_new_before_map(self, manager):
        """Test adding a new section before the map."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Danger info.

## CURRENT WORLD MAP
Map goes here.
"""
        new_content = "Cross-episode wisdom here."

        updated = manager._update_section_content(
            knowledge_content, "CROSS-EPISODE INSIGHTS", new_content
        )

        assert "## CROSS-EPISODE INSIGHTS" in updated
        assert "Cross-episode wisdom here." in updated

        # Verify it's before the map section
        cross_ep_index = updated.find("## CROSS-EPISODE INSIGHTS")
        map_index = updated.find("## CURRENT WORLD MAP")
        assert cross_ep_index < map_index

    def test_update_section_content_add_new_at_end(self, manager):
        """Test adding a new section at the end when no map exists."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS
Danger info.

## PUZZLE SOLUTIONS
Puzzle info.
"""
        new_content = "Cross-episode wisdom here."

        updated = manager._update_section_content(
            knowledge_content, "CROSS-EPISODE INSIGHTS", new_content
        )

        assert "## CROSS-EPISODE INSIGHTS" in updated
        assert "Cross-episode wisdom here." in updated

        # Verify it's at the end
        assert updated.rstrip().endswith("Cross-episode wisdom here.")

    def test_update_section_content_empty_knowledge_base(self, manager):
        """Test updating when knowledge base is empty."""
        knowledge_content = ""
        new_content = "First content."

        updated = manager._update_section_content(
            knowledge_content, "CROSS-EPISODE INSIGHTS", new_content
        )

        assert "# Zork Game World Knowledge Base" in updated
        assert "## CROSS-EPISODE INSIGHTS" in updated
        assert "First content." in updated

    def test_section_extraction_preserves_formatting(self, manager):
        """Test that section extraction preserves markdown formatting."""
        knowledge_content = """# Zork Game World Knowledge Base

## CROSS-EPISODE INSIGHTS

### Death Patterns
- **Grue attacks**: Always fatal in dark rooms
- **Troll combat**: Requires sword

### Environmental Knowledge
* West of House is safe
* Cellar has no light

#### Sub-subsection
More details here.
"""
        result = manager._extract_section_content(
            knowledge_content, "CROSS-EPISODE INSIGHTS"
        )

        # Verify formatting is preserved
        assert "### Death Patterns" in result
        assert "- **Grue attacks**:" in result
        assert "* West of House" in result
        assert "#### Sub-subsection" in result


class TestCrossEpisodeInsightsFormatting:
    """Test the formatted output of cross-episode section extraction."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager instance."""
        log_file = tmp_path / "test_log.jsonl"
        output_file = tmp_path / "test_knowledge.md"
        workdir = tmp_path / "game_files"
        workdir.mkdir(exist_ok=True)

        manager = AdaptiveKnowledgeManager(
            log_file=str(log_file),
            output_file=str(output_file),
            logger=Mock(),
            workdir=str(workdir),
        )
        return manager

    def test_cross_episode_formatting_with_separators(self, manager):
        """Test that cross-episode section includes proper formatting."""
        knowledge_content = """# Zork Game World Knowledge Base

## CROSS-EPISODE INSIGHTS
Important wisdom here.
Multiple lines of wisdom.
"""
        result = manager._extract_cross_episode_section(knowledge_content)

        # Verify the formatted output includes separators
        assert "**CROSS-EPISODE INSIGHTS FROM PREVIOUS EPISODES:**" in result
        assert "-" * 50 in result
        assert "Important wisdom here." in result
        assert "Multiple lines of wisdom." in result

    def test_cross_episode_section_in_prompt_context(self, manager):
        """Test that extracted section is suitable for prompt context."""
        knowledge_content = """# Zork Game World Knowledge Base

## CROSS-EPISODE INSIGHTS

### Death Patterns Across Episodes
- Dark rooms without light source cause instant death
- Troll requires weapon or avoidance

### Strategic Meta-Patterns
- Always check inventory after scoring
- Map dark areas after obtaining lamp
"""
        result = manager._extract_cross_episode_section(knowledge_content)

        # Verify it's well-formatted for LLM context
        assert result.startswith("\n")
        assert "CROSS-EPISODE INSIGHTS FROM PREVIOUS EPISODES:" in result
        assert "Death Patterns Across Episodes" in result
        assert "Strategic Meta-Patterns" in result

        # Verify separators are present and balanced
        separator = "-" * 50
        assert result.count(separator) == 2  # Opening and closing


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a test manager instance."""
        log_file = tmp_path / "test_log.jsonl"
        output_file = tmp_path / "test_knowledge.md"
        workdir = tmp_path / "game_files"
        workdir.mkdir(exist_ok=True)

        manager = AdaptiveKnowledgeManager(
            log_file=str(log_file),
            output_file=str(output_file),
            logger=Mock(),
            workdir=str(workdir),
        )
        return manager

    def test_extract_from_none_content(self, manager):
        """Test extracting from None content."""
        result = manager._extract_cross_episode_section(None)
        assert result == ""

        result = manager._extract_section_content(None, "ANY SECTION")
        assert result == ""

    def test_update_none_content(self, manager):
        """Test updating None content."""
        result = manager._update_section_content(None, "NEW SECTION", "Content")

        assert "# Zork Game World Knowledge Base" in result
        assert "## NEW SECTION" in result
        assert "Content" in result

    def test_section_name_with_special_characters(self, manager):
        """Test section names with regex special characters."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS & THREATS (CRITICAL!)
Danger info here.

## PUZZLE [SOLUTIONS]
Puzzle info here.
"""
        # Extract section with parentheses
        result = manager._extract_section_content(
            knowledge_content, "DANGERS & THREATS (CRITICAL!)"
        )
        assert "Danger info here." in result

        # Extract section with brackets
        result = manager._extract_section_content(
            knowledge_content, "PUZZLE [SOLUTIONS]"
        )
        assert "Puzzle info here." in result

    def test_multiple_sections_with_same_start(self, manager):
        """Test when section names share a common prefix."""
        knowledge_content = """# Zork Game World Knowledge Base

## DANGERS
General dangers.

## DANGERS & THREATS
Specific threats.

## DANGERS ANALYSIS
Deep analysis.
"""
        # Should extract the exact match, not partial
        result = manager._extract_section_content(knowledge_content, "DANGERS")
        assert "General dangers." in result
        assert "Specific threats." not in result
        assert "Deep analysis." not in result

        result = manager._extract_section_content(
            knowledge_content, "DANGERS & THREATS"
        )
        assert "Specific threats." in result
        assert "General dangers." not in result
