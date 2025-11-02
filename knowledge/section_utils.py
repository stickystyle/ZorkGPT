# ABOUTME: Section manipulation utilities for knowledge base content.
# ABOUTME: Provides functions to extract, update, trim, and preserve knowledge base sections.

"""
Knowledge base section manipulation utilities.

This module provides standalone functions for manipulating sections within the
knowledge base markdown content, including extracting specific sections, updating
section content, trimming the map section for LLM processing, and preserving the
map section when updating knowledge.
"""

import re
from typing import Optional


def trim_map_section(knowledge_content: str) -> str:
    """
    Remove the map section from knowledge content for LLM processing.

    The map section (CURRENT WORLD MAP) can be very large and contains
    structured data that doesn't need to be processed by the LLM during
    knowledge updates. This function removes it to reduce token usage.

    Args:
        knowledge_content: Full knowledge base content

    Returns:
        Knowledge content with map section removed
    """
    if not knowledge_content or "## CURRENT WORLD MAP" not in knowledge_content:
        return knowledge_content

    # Remove the mermaid diagram section more precisely
    # Look for the pattern: ## CURRENT WORLD MAP followed by ```mermaid...```
    pattern = r"## CURRENT WORLD MAP\s*\n\s*```mermaid\s*\n.*?\n```"

    # Remove the mermaid diagram section while preserving other content
    knowledge_only = re.sub(pattern, "", knowledge_content, flags=re.DOTALL)

    # Clean up any extra whitespace that might be left
    knowledge_only = re.sub(r"\n\s*\n\s*\n", "\n\n", knowledge_only)

    return knowledge_only.strip()


def preserve_map_section(original_knowledge: str, new_knowledge: str) -> str:
    """
    Preserve the map section from original knowledge in the new knowledge.

    When the LLM generates updated knowledge content, it doesn't include the
    map section (which is trimmed before processing). This function restores
    the map section from the original content.

    Args:
        original_knowledge: Original knowledge base with map section
        new_knowledge: New knowledge base without map section

    Returns:
        New knowledge with map section appended from original
    """
    if not original_knowledge or "## CURRENT WORLD MAP" not in original_knowledge:
        return new_knowledge

    # Extract map section from original
    map_start = original_knowledge.find("## CURRENT WORLD MAP")
    if map_start == -1:
        return new_knowledge

    map_section = original_knowledge[map_start:]

    # Add map section to new knowledge
    return f"{new_knowledge.rstrip()}\n\n{map_section}"


def extract_section_content(knowledge_content: str, section_name: str) -> str:
    """
    Extract content from a specific section of the knowledge base.

    Sections are identified by markdown headers (## SECTION_NAME). This function
    extracts all content from the specified section until the next section header
    or end of file.

    Args:
        knowledge_content: Full knowledge base content
        section_name: Name of section to extract (without ## prefix)

    Returns:
        Section content or empty string if not found
    """
    if not knowledge_content:
        return ""

    # Look for the section heading followed by content until next section or end
    pattern = rf"## {re.escape(section_name)}(.*?)(?=\n## |$)"
    match = re.search(pattern, knowledge_content, re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


def update_section_content(
    knowledge_content: str, section_name: str, new_content: str
) -> str:
    """
    Update or add a section in the knowledge base.

    If the section exists, it's replaced with new content. If it doesn't exist,
    it's added before the map section (if present) or at the end.

    Args:
        knowledge_content: Full knowledge base content
        section_name: Name of section to update (without ## prefix)
        new_content: New content for the section

    Returns:
        Updated knowledge base content
    """
    if not knowledge_content:
        knowledge_content = "# Zork Game World Knowledge Base\n\n"

    section_header = f"## {section_name}"

    # Check if section exists
    pattern = rf"## {re.escape(section_name)}(.*?)(?=\n## |$)"
    match = re.search(pattern, knowledge_content, re.DOTALL)

    if match:
        # Replace existing section (only first occurrence)
        old_section = match.group(0)
        new_section = f"{section_header}\n\n{new_content}\n"
        updated = knowledge_content.replace(old_section, new_section, 1)
        return updated
    else:
        # Add new section before the map section (if it exists) or at the end
        if "## CURRENT WORLD MAP" in knowledge_content:
            # Insert before map
            map_index = knowledge_content.find("## CURRENT WORLD MAP")
            before_map = knowledge_content[:map_index]
            map_section = knowledge_content[map_index:]
            return f"{before_map}\n{section_header}\n\n{new_content}\n\n{map_section}"
        else:
            # Append at end
            return f"{knowledge_content}\n\n{section_header}\n\n{new_content}\n"


def extract_cross_episode_section(knowledge_content: str) -> str:
    """
    Extract the CROSS-EPISODE INSIGHTS section from existing knowledge base.

    This section contains persistent wisdom that carries across episodes and
    should be preserved and highlighted during knowledge updates.

    Args:
        knowledge_content: Full knowledge base content

    Returns:
        Formatted cross-episode section or empty string if not found
    """
    if not knowledge_content:
        return ""

    # Look for the CROSS-EPISODE INSIGHTS section
    pattern = r"## CROSS-EPISODE INSIGHTS(.*?)(?=\n## |$)"
    match = re.search(pattern, knowledge_content, re.DOTALL)

    if match:
        section_content = match.group(1).strip()
        if section_content:
            return f"""
**CROSS-EPISODE INSIGHTS FROM PREVIOUS EPISODES:**
{"-" * 50}
{section_content}
{"-" * 50}
"""

    return ""
