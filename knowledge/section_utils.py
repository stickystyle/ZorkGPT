# ABOUTME: Section manipulation utilities for knowledge base content.
# ABOUTME: Provides functions to extract and update knowledge base sections.

"""
Knowledge base section manipulation utilities.

This module provides standalone functions for manipulating sections within the
knowledge base markdown content, including extracting specific sections and
updating section content.
"""

import re
from typing import Optional


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
    it's added at the end.

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
        # Append at end
        return f"{knowledge_content}\n\n{section_header}\n\n{new_content}\n"


def remove_section(knowledge_content: str, section_name: str) -> str:
    """
    Remove a section from knowledge base content.

    Args:
        knowledge_content: Full knowledge base markdown content
        section_name: Name of section to remove (without ## prefix)

    Returns:
        Knowledge content with specified section removed
    """
    if not knowledge_content:
        return ""

    # Use regex pattern to match section header and all content until next section or end
    pattern = rf"## {re.escape(section_name)}(.*?)(?=\n## |$)"

    # Remove the section (replaces with empty string)
    result = re.sub(pattern, "", knowledge_content, flags=re.DOTALL)

    # Clean up any resulting multiple consecutive newlines (more than 2)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result


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
