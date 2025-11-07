# ABOUTME: Knowledge management package initialization for ZorkGPT
# ABOUTME: Exposes core knowledge management classes and utilities
"""
Knowledge management modules for ZorkGPT.

This package handles all knowledge-related functionality including:
- Quality assessment for knowledge updates
- Turn-based data extraction from episode logs
- LLM-based knowledge generation and synthesis
- Cross-episode wisdom synthesis
- Section-based knowledge management utilities

Main class:
- AdaptiveKnowledgeManager: Orchestrates all knowledge management operations

Modules:
- quality_assessment: Knowledge update quality checks
- turn_extraction: Turn-based data extraction from logs
- knowledge_generation: LLM-based knowledge generation
- cross_episode_synthesis: Cross-episode wisdom synthesis
- section_utils: Section-based utilities
- adaptive_manager: Main orchestrator class
"""

from knowledge.adaptive_manager import AdaptiveKnowledgeManager

__version__ = "2.0.0"  # Post-modularization version
__all__ = ['AdaptiveKnowledgeManager']
