"""
DEPRECATED: This module is maintained for backward compatibility.
New code should import from knowledge.adaptive_manager directly.

Legacy import wrapper for AdaptiveKnowledgeManager.

This module previously contained the full implementation of the
AdaptiveKnowledgeManager class. The implementation has been refactored
into the knowledge package with a modular architecture:

- knowledge.quality_assessment: Knowledge update quality checks
- knowledge.turn_extraction: Turn-based data extraction from logs
- knowledge.knowledge_generation: LLM-based knowledge generation
- knowledge.knowledge_condensation: Knowledge condensation for efficiency
- knowledge.cross_episode_synthesis: Cross-episode wisdom synthesis
- knowledge.section_utils: Section-based utilities
- knowledge.adaptive_manager: Main orchestrator class

For new code, import directly from knowledge:
    from knowledge import AdaptiveKnowledgeManager

This wrapper ensures backward compatibility for existing code that imports from
zork_strategy_generator.
"""

from knowledge.adaptive_manager import AdaptiveKnowledgeManager

__all__ = ['AdaptiveKnowledgeManager']
