# ABOUTME: Main orchestrator for adaptive knowledge management in ZorkGPT
# ABOUTME: Coordinates quality assessment, turn extraction, generation, condensation, and synthesis

"""
Adaptive knowledge manager for ZorkGPT.

Main orchestrator class that coordinates all knowledge management operations
using the modular components for quality assessment, turn extraction,
knowledge generation, condensation, and cross-episode synthesis.
"""

import os
from typing import Dict, Optional
from pathlib import Path

from llm_client import LLMClientWrapper
from config import get_config, get_client_api_key

from knowledge import quality_assessment
from knowledge import turn_extraction
from knowledge import knowledge_generation
from knowledge import knowledge_condensation
from knowledge import cross_episode_synthesis
from knowledge import section_utils

try:
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    # Graceful fallback - no-op decorator
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    LANGFUSE_AVAILABLE = False


class AdaptiveKnowledgeManager:
    """
    LLM-first adaptive knowledge management system for turn-based updates.

    This system lets the LLM manage its own knowledge lifecycle by:
    1. Assessing the quality of potential knowledge updates
    2. Determining appropriate update strategies
    3. Performing intelligent knowledge merging
    4. Self-monitoring for knowledge degradation
    """

    def __init__(
        self,
        log_file: str = "zork_episode_log.jsonl",
        output_file: str = "knowledgebase.md",
        logger=None,
        workdir: str = "game_files",
    ):
        self.log_file = log_file
        self.output_file = output_file
        self.logger = logger
        self.workdir = workdir

        # Initialize LLM client
        config = get_config()
        self.client = LLMClientWrapper(
            base_url=config.llm.get_base_url_for_model("analysis"),
            api_key=get_client_api_key(),
        )

        # Models for different tasks
        self.analysis_model = config.llm.analysis_model
        self.condensation_model = config.llm.condensation_model

        # Load sampling parameters from configuration
        self.analysis_sampling = config.analysis_sampling
        self.condensation_sampling = config.condensation_sampling

        # Turn-based configuration
        self.turn_window_size = config.gameplay.turn_window_size
        self.min_quality_threshold = config.gameplay.min_knowledge_quality

        # Knowledge base condensation configuration
        self.enable_condensation = config.gameplay.enable_knowledge_condensation
        self.condensation_threshold = config.gameplay.knowledge_condensation_threshold

        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = config.logging.enable_prompt_logging

        # Load agent instructions to avoid duplication
        self.agent_instructions = self._load_agent_instructions()

    def _get_episode_log_file(self, episode_id: str) -> Path:
        """Get the log file path for a specific episode."""
        return Path(self.workdir) / "episodes" / episode_id / "episode_log.jsonl"

    def _log_prompt_to_file(
        self, messages: list, prefix: str = "knowledge"
    ) -> None:
        """Log the full prompt to a temporary file for evaluation."""
        if not self.enable_prompt_logging:
            return

        self.prompt_counter += 1
        filename = f"tmp/{prefix}_{self.prompt_counter:03d}.txt"

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"=== {prefix.upper()} PROMPT #{self.prompt_counter} ===\n")
                f.write(f"Model: {self.analysis_model}\n")
                f.write("=" * 50 + "\n\n")

                for i, message in enumerate(messages):
                    f.write(f"--- MESSAGE {i + 1} ({message['role'].upper()}) ---\n")
                    f.write(message["content"])
                    f.write("\n\n")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to log prompt to {filename}: {e}",
                    extra={"event_type": "knowledge_update"},
                )

    def _load_agent_instructions(self) -> str:
        """Load the agent.md prompt to understand what's already covered."""
        try:
            with open("agent.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Could not load agent.md: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            return ""

    def _is_first_meaningful_update(self) -> bool:
        """
        Check if this is the first meaningful knowledge update.

        Returns True if:
        1. No knowledge base exists, OR
        2. Knowledge base only contains auto-generated basic content (map + basic strategy)

        This handles the case where knowledgebase.md is auto-created for map updates
        but doesn't contain any LLM-generated strategic insights yet.
        """
        if not os.path.exists(self.output_file):
            return True

        if os.path.getsize(self.output_file) == 0:
            return True

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if content only contains basic auto-generated sections
            # Look for indicators of LLM-generated strategic content

            # Remove map section for analysis
            content_without_map = section_utils.trim_map_section(content)

            # Basic strategy indicators that suggest auto-generated content
            basic_indicators = [
                "Always begin each location with 'look'",
                "Use systematic exploration patterns",
                "Execute 'take' commands for all portable items",
                "Parse all text output for puzzle-solving information",
                "Prioritize information extraction over rapid action execution",
            ]

            # Count how many basic indicators are present
            basic_indicator_count = sum(
                1 for indicator in basic_indicators if indicator in content_without_map
            )

            # If content is very short and mostly contains basic indicators, treat as first update
            content_lines = [
                line.strip() for line in content_without_map.split("\n") if line.strip()
            ]
            meaningful_content_lines = [
                line
                for line in content_lines
                if not line.startswith("#") and len(line) > 10
            ]

            # Heuristics for detecting auto-generated vs LLM-generated content:
            # 1. Very few meaningful content lines (< 10)
            # 2. High ratio of basic indicators to total content
            # 3. No complex strategic insights (no sentences > 100 chars with specific game references)

            if len(meaningful_content_lines) < 10:
                return True

            if basic_indicator_count >= 3 and len(meaningful_content_lines) < 15:
                return True

            # Look for complex strategic insights (longer sentences with game-specific terms)
            complex_insights = [
                line
                for line in meaningful_content_lines
                if len(line) > 80
                and any(
                    term in line.lower()
                    for term in [
                        "puzzle",
                        "treasure",
                        "combat",
                        "inventory",
                        "specific",
                        "strategy",
                        "avoid",
                        "danger",
                        "death",
                        "troll",
                        "grue",
                        "lamp",
                        "sword",
                    ]
                )
            ]

            # If no complex insights found, likely still basic content
            if len(complex_insights) == 0:
                return True

            return False

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Error checking knowledge base content: {e}",
                    extra={"event_type": "knowledge_update"},
                )
            # If we can't read it, assume it's not meaningful yet
            return True

    def update_knowledge_from_turns(
        self,
        episode_id: str,
        start_turn: int,
        end_turn: int,
        is_final_update: bool = False,
    ) -> bool:
        """
        Update knowledge base from a specific turn range using single-stage generation.

        Args:
            episode_id: Current episode ID
            start_turn: Starting turn number
            end_turn: Ending turn number
            is_final_update: If True, more lenient about quality (episode-end updates)

        Returns:
            bool: True if knowledge was updated, False if skipped
        """
        if self.logger:
            self.logger.info(
                f"Knowledge update requested for turns {start_turn}-{end_turn}",
                extra={
                    "event_type": "knowledge_update_start",
                    "episode_id": episode_id,
                    "turn_range": f"{start_turn}-{end_turn}",
                    "is_final": is_final_update,
                },
            )

        # Step 1: Extract turn window data
        turn_data = turn_extraction.extract_turn_window_data(
            episode_id=episode_id,
            start_turn=start_turn,
            end_turn=end_turn,
            log_file=self.log_file,
            workdir=self.workdir,
            logger=self.logger
        )

        if not turn_data or not turn_data.get("actions_and_responses"):
            if self.logger:
                self.logger.warning(
                    "No turn data found for analysis",
                    extra={
                        "event_type": "knowledge_update_skipped",
                        "episode_id": episode_id,
                        "reason": "no_data",
                    },
                )
            return False

        # Step 2: Quality check using heuristics
        should_update, reason = quality_assessment.should_update_knowledge(
            turn_data=turn_data,
            logger=self.logger
        )

        # Override for final updates if there's a death or significant content
        if is_final_update and not should_update:
            if (
                turn_data.get("death_events")
                or len(turn_data["actions_and_responses"]) >= 5
            ):
                should_update = True
                reason = "Final update with significant content"

        if self.logger:
            self.logger.info(
                f"Knowledge update decision: {'proceed' if should_update else 'skip'} - {reason}",
                extra={
                    "event_type": "knowledge_update_decision",
                    "episode_id": episode_id,
                    "should_update": should_update,
                    "reason": reason,
                },
            )

        if not should_update:
            return False

        # Step 3: Load existing knowledge
        existing_knowledge = ""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, "r", encoding="utf-8") as f:
                    existing_knowledge = f.read()

                # Trim map section for LLM processing
                existing_knowledge = section_utils.trim_map_section(existing_knowledge)

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Could not load existing knowledge: {e}",
                    extra={"event_type": "knowledge_update", "episode_id": episode_id},
                )

        # Step 4: Generate new knowledge in single pass
        if self.logger:
            self.logger.info(
                "Generating knowledge base update",
                extra={
                    "event_type": "knowledge_generation_start",
                    "episode_id": episode_id,
                },
            )

        new_knowledge = knowledge_generation.generate_knowledge_directly(
            turn_data=turn_data,
            existing_knowledge=existing_knowledge,
            client=self.client,
            analysis_model=self.analysis_model,
            analysis_sampling=self.analysis_sampling,
            logger=self.logger,
            log_prompt_callback=self._log_prompt_to_file if self.enable_prompt_logging else None
        )

        if not new_knowledge or new_knowledge.startswith("SKIP:"):
            if self.logger:
                self.logger.warning(
                    f"Knowledge generation returned skip or empty: {new_knowledge[:100] if new_knowledge else 'None'}",
                    extra={
                        "event_type": "knowledge_update_skipped",
                        "episode_id": episode_id,
                    },
                )
            return False

        # Step 5: Preserve map section if it exists
        if existing_knowledge and "## CURRENT WORLD MAP" in existing_knowledge:
            # Extract and preserve the map section
            original_with_map = ""
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    original_with_map = f.read()
                new_knowledge = section_utils.preserve_map_section(
                    original_with_map, new_knowledge
                )
            except:
                pass  # Map preservation is non-critical

        # Step 6: Write updated knowledge to file
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(new_knowledge)

            if self.logger:
                self.logger.info(
                    "Knowledge base updated successfully",
                    extra={
                        "event_type": "knowledge_update_success",
                        "episode_id": episode_id,
                        "file": self.output_file,
                        "size": len(new_knowledge),
                    },
                )

            # Step 7: Log the prompt if in debug mode
            if hasattr(self, "log_prompts") and self.log_prompts:
                self._log_prompt_to_file("knowledge_update", turn_data, new_knowledge)

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Failed to write knowledge base: {e}",
                    extra={
                        "event_type": "knowledge_update_error",
                        "episode_id": episode_id,
                        "error": str(e),
                    },
                )
            return False

    @observe(name="knowledge-synthesize-strategic")
    def synthesize_inter_episode_wisdom(self, episode_data: Dict) -> bool:
        """
        Synthesize persistent wisdom from episode completion into the CROSS-EPISODE INSIGHTS
        section of knowledgebase.md. Focuses on deaths, major discoveries, and cross-episode patterns.

        Args:
            episode_data: Dictionary containing episode summary information

        Returns:
            True if synthesis was performed and wisdom was updated, False if skipped
        """
        # Add callback for extracting turn data if not already present
        if "extract_turn_data_callback" not in episode_data:
            episode_data = episode_data.copy()  # Don't mutate original
            episode_data["extract_turn_data_callback"] = self._extract_turn_window_data

        return cross_episode_synthesis.synthesize_inter_episode_wisdom(
            episode_data=episode_data,
            output_file=self.output_file,
            client=self.client,
            analysis_model=self.analysis_model,
            analysis_sampling=self.analysis_sampling,
            logger=self.logger
        )

    # Backward compatibility wrappers for tests
    def _should_update_knowledge(self, turn_data: Dict):
        """
        Backward compatibility wrapper for tests.
        Delegates to quality_assessment module.
        """
        return quality_assessment.should_update_knowledge(
            turn_data=turn_data,
            logger=self.logger
        )

    def _extract_turn_window_data(self, episode_id: str, start_turn: int, end_turn: int):
        """
        Backward compatibility wrapper for tests.
        Delegates to turn_extraction module.
        """
        return turn_extraction.extract_turn_window_data(
            episode_id=episode_id,
            start_turn=start_turn,
            end_turn=end_turn,
            log_file=self.log_file,
            workdir=self.workdir,
            logger=self.logger
        )

    def _extract_cross_episode_section(self, knowledge_content: str) -> str:
        """
        Backward compatibility wrapper for tests.
        Delegates to section_utils module.
        """
        return section_utils.extract_cross_episode_section(knowledge_content)

    def _extract_section_content(self, knowledge_content: str, section_name: str) -> str:
        """
        Backward compatibility wrapper for tests.
        Delegates to section_utils module.
        """
        return section_utils.extract_section_content(knowledge_content, section_name)

    def _update_section_content(self, knowledge_content: str, section_name: str, new_content: str) -> str:
        """
        Backward compatibility wrapper for tests.
        Delegates to section_utils module.
        """
        return section_utils.update_section_content(knowledge_content, section_name, new_content)

    def _format_turn_data_for_prompt(self, turn_data: Dict) -> str:
        """
        Backward compatibility wrapper for tests.
        Delegates to turn_extraction module.
        """
        return turn_extraction.format_turn_data_for_prompt(turn_data)

    def _format_death_analysis_section(self, turn_data: Dict) -> str:
        """
        Backward compatibility wrapper for tests.
        Delegates to turn_extraction module.
        """
        return turn_extraction.format_death_analysis_section(turn_data)
