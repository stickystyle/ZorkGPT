"""
AdaptiveKnowledgeManager module for turn-based knowledge management.

This module provides LLM-first adaptive knowledge management for ZorkGPT,
using turn-based sliding windows instead of episode-based analysis.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import environs
from openai import OpenAI
from map_graph import MapGraph
from movement_analyzer import MovementAnalyzer, create_movement_context

# Load environment variables
env = environs.Env()
env.read_env()


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
    ):
        self.log_file = log_file
        self.output_file = output_file

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=env.str("CLIENT_BASE_URL", None),
            api_key=env.str("CLIENT_API_KEY", None),
        )

        # Model for analysis
        self.analysis_model = env.str("ANALYSIS_MODEL", "gpt-4")

        # Turn-based configuration
        self.turn_window_size = env.int("TURN_WINDOW_SIZE", 100)
        self.min_quality_threshold = env.float("MIN_KNOWLEDGE_QUALITY", 6.0)
        
        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = env.bool("ENABLE_PROMPT_LOGGING", False)
        
        # Load agent instructions to avoid duplication
        self.agent_instructions = self._load_agent_instructions()

    def _log_prompt_to_file(self, messages: List[Dict], prefix: str = "knowledge") -> None:
        """Log the full prompt to a temporary file for evaluation."""
        if not self.enable_prompt_logging:
            return
            
        self.prompt_counter += 1
        filename = f"tmp/{prefix}_{self.prompt_counter:03d}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== {prefix.upper()} PROMPT #{self.prompt_counter} ===\n")
                f.write(f"Model: {self.analysis_model}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, message in enumerate(messages):
                    f.write(f"--- MESSAGE {i+1} ({message['role'].upper()}) ---\n")
                    f.write(message['content'])
                    f.write("\n\n")
        except Exception as e:
            print(f"Failed to log prompt to {filename}: {e}")

    def _load_agent_instructions(self) -> str:
        """Load the agent.md prompt to understand what's already covered."""
        try:
            with open("agent.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"  ⚠️ Could not load agent.md: {e}")
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
            content_without_map = self._trim_map_section(content)
            
            # Basic strategy indicators that suggest auto-generated content
            basic_indicators = [
                "Always begin each location with 'look'",
                "Use systematic exploration patterns",
                "Execute 'take' commands for all portable items",
                "Parse all text output for puzzle-solving information",
                "Prioritize information extraction over rapid action execution"
            ]
            
            # Count how many basic indicators are present
            basic_indicator_count = sum(1 for indicator in basic_indicators if indicator in content_without_map)
            
            # If content is very short and mostly contains basic indicators, treat as first update
            content_lines = [line.strip() for line in content_without_map.split('\n') if line.strip()]
            meaningful_content_lines = [line for line in content_lines if not line.startswith('#') and len(line) > 10]
            
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
                line for line in meaningful_content_lines 
                if len(line) > 80 and any(term in line.lower() for term in [
                    'puzzle', 'treasure', 'combat', 'inventory', 'specific', 'strategy',
                    'avoid', 'danger', 'death', 'troll', 'grue', 'lamp', 'sword'
                ])
            ]
            
            # If no complex insights found, likely still basic content
            if len(complex_insights) == 0:
                return True
                
            return False
            
        except Exception as e:
            print(f"  ⚠️ Error checking knowledge base content: {e}")
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
        Update knowledge base from a specific turn range.

        Args:
            episode_id: Current episode ID
            start_turn: Starting turn number
            end_turn: Ending turn number
            is_final_update: If True, use lower quality threshold for episode-end updates

        Returns:
            True if knowledge was updated, False if skipped
        """
        print(f"🧠 Evaluating knowledge update for turns {start_turn}-{end_turn}...")

        # Extract turn window data
        turn_data = self._extract_turn_window_data(episode_id, start_turn, end_turn)
        if not turn_data:
            print("  ⚠️ No turn data found for analysis")
            return False

        # Check if this is the very first knowledge update (no meaningful knowledge exists)
        is_first_update = self._is_first_meaningful_update()

        # Step 1: LLM assesses if this data is worth analyzing
        quality_score, quality_reason = self._assess_knowledge_update_quality(
            turn_data, is_final_update
        )
        print(
            f"  📊 Knowledge quality score: {quality_score:.1f}/10 - {quality_reason}"
        )

        # Use lower threshold for final episode updates to capture important death/danger scenarios
        effective_threshold = self.min_quality_threshold
        if is_final_update:
            effective_threshold = max(
                3.0, self.min_quality_threshold - 2.0
            )  # Lower threshold for final updates
            print(
                f"  🎯 Using final update threshold: {effective_threshold:.1f} (vs normal {self.min_quality_threshold:.1f})"
            )

        # Allow first update regardless of quality to bootstrap learning
        if is_first_update:
            print(
                f"  🌱 First knowledge update - proceeding regardless of quality score to bootstrap learning"
            )
        elif quality_score < effective_threshold:
            print(
                f"  ⏭️ Skipping update - quality below threshold ({effective_threshold})"
            )
            return False

        # Step 2: LLM determines update strategy
        update_strategy = self._determine_update_strategy(turn_data, quality_score)
        print(f"  🎯 Update strategy: {update_strategy}")

        # Step 3: Perform analysis based on strategy
        new_insights = self._perform_targeted_analysis(turn_data, update_strategy)
        if not new_insights:
            print("  ⚠️ Analysis failed to generate insights")
            return False

        # Step 4: Intelligent knowledge merging
        success = self._intelligent_knowledge_merge(new_insights, update_strategy)
        if success:
            print("  ✅ Knowledge base updated successfully")
        else:
            print("  ⚠️ Knowledge merge failed")

        return success

    def _extract_turn_window_data(
        self, episode_id: str, start_turn: int, end_turn: int
    ) -> Optional[Dict]:
        """Extract action-response data for a specific turn window."""
        turn_data = {
            "episode_id": episode_id,
            "start_turn": start_turn,
            "end_turn": end_turn,
            "actions_and_responses": [],
            "score_changes": [],
            "location_changes": [],
            "inventory_changes": [],
        }

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                current_turn = 0
                current_score = 0
                current_location = ""
                current_inventory = []

                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # Skip entries not from this episode
                        if log_entry.get("episode_id") != episode_id:
                            continue

                        event_type = log_entry.get("event_type", "")

                        # Track turn progression
                        if event_type == "turn_start":
                            current_turn = log_entry.get("turn", 0)

                        # Only collect data within our turn window
                        if not (start_turn <= current_turn <= end_turn):
                            continue

                        # Collect action-response pairs
                        if event_type == "final_action_selection":
                            action_data = {
                                "turn": current_turn,
                                "action": log_entry.get("agent_action", ""),
                                "reasoning": log_entry.get("agent_reasoning", ""),
                                "critic_score": log_entry.get("critic_score", 0),
                                "response": "",  # Will be filled by next zork_response
                            }
                            turn_data["actions_and_responses"].append(action_data)

                        elif (
                            event_type == "zork_response"
                            and turn_data["actions_and_responses"]
                        ):
                            # Update the last action with its response
                            turn_data["actions_and_responses"][-1]["response"] = (
                                log_entry.get("zork_response", "")
                            )

                        # Track score changes
                        elif event_type == "experience" and "zork_score" in log_entry:
                            new_score = log_entry.get("zork_score", 0)
                            if new_score != current_score:
                                turn_data["score_changes"].append(
                                    {
                                        "turn": current_turn,
                                        "from_score": current_score,
                                        "to_score": new_score,
                                        "change": new_score - current_score,
                                    }
                                )
                                current_score = new_score

                        # Track location changes
                        elif event_type == "extracted_info":
                            extracted_info = log_entry.get("extracted_info", {})
                            new_location = extracted_info.get(
                                "current_location_name", ""
                            )
                            if (
                                new_location
                                and new_location != current_location
                                and new_location != "Unknown Location"
                            ):
                                turn_data["location_changes"].append(
                                    {
                                        "turn": current_turn,
                                        "from_location": current_location,
                                        "to_location": new_location,
                                    }
                                )
                                current_location = new_location

                    except json.JSONDecodeError:
                        continue

        except FileNotFoundError:
            print(f"  ⚠️ Log file {self.log_file} not found")
            return None

        return turn_data if turn_data["actions_and_responses"] else None

    def _assess_knowledge_update_quality(
        self, turn_data: Dict, is_final_update: bool = False
    ) -> Tuple[float, str]:
        """Let LLM assess if this turn window would produce useful knowledge."""

        # Prepare summary of turn data
        num_actions = len(turn_data["actions_and_responses"])
        num_score_changes = len(turn_data["score_changes"])
        num_location_changes = len(turn_data["location_changes"])

        # Sample some actions for context (limit to avoid token overflow)
        sample_actions = turn_data["actions_and_responses"][:10]
        actions_summary = "\n".join(
            [
                f"Turn {action['turn']}: {action['action']} -> {action['response'][:100]}..."
                if len(action["response"]) > 100
                else f"Turn {action['turn']}: {action['action']} -> {action['response']}"
                for action in sample_actions
            ]
        )

        # Adjust prompt based on whether this is a final update
        context_note = ""
        if is_final_update:
            context_note = """
**IMPORTANT**: This is a FINAL EPISODE UPDATE. Even brief encounters with danger, death, or new areas 
can provide valuable learning opportunities. Be more lenient in scoring - focus on whether ANY useful 
strategic insights could be extracted, especially about dangers to avoid or new discoveries."""

        prompt = f"""You are about to analyze gameplay data to extract strategic insights for a Zork knowledge base.
{context_note}

TURN WINDOW SUMMARY:
- Turns: {turn_data["start_turn"]}-{turn_data["end_turn"]}
- Total actions: {num_actions}
- Score changes: {num_score_changes}
- Location changes: {num_location_changes}

SAMPLE ACTIONS:
{actions_summary}

Before doing the full analysis, assess the potential value of this data:

1. How much NEW strategic value could be extracted from this data?
2. Are there repetitive patterns that would dilute useful insights?
3. Would analyzing this data improve or degrade the knowledge base?
4. What type of gameplay situation does this represent?
5. {"Are there any dangers, deaths, or new discoveries that should be documented?" if is_final_update else ""}

Rate the potential knowledge value from 0-10 and provide a brief explanation.
{"For final updates: Focus on ANY useful insights, especially dangers or new areas." if is_final_update else "Focus on whether this data would generate actionable, non-repetitive insights."}

Format your response as:
SCORE: [0-10]
REASON: [brief explanation]"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at evaluating gameplay data quality for knowledge extraction. Be honest about data that would produce low-quality or repetitive insights.",
                },
                {"role": "user", "content": prompt},
            ]
            
            # Log the full prompt for evaluation
            self._log_prompt_to_file(messages, "knowledge_quality")
            
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Parse score and reason
            lines = content.split("\n")
            score = 5.0  # Default
            reason = "Assessment failed"

            for line in lines:
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except:
                        pass
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            return score, reason

        except Exception as e:
            print(f"  ⚠️ Quality assessment failed: {e}")
            return 5.0, "Assessment failed due to API error"

    def _determine_update_strategy(self, turn_data: Dict, quality_score: float) -> str:
        """Let LLM determine the best update strategy for this data."""

        # Load current knowledge base for context
        current_knowledge = ""
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                current_knowledge = f.read()[:2000]  # First 2000 chars for context
        except:
            current_knowledge = "No existing knowledge base"

        prompt = f"""Based on the gameplay data and current knowledge base, determine the best analysis strategy.

CURRENT KNOWLEDGE BASE (excerpt):
{current_knowledge}

TURN DATA QUALITY: {quality_score}/10

TURN DATA SUMMARY:
- Actions: {len(turn_data["actions_and_responses"])}
- Score changes: {len(turn_data["score_changes"])}
- Location changes: {len(turn_data["location_changes"])}

Choose the most appropriate strategy:

1. FULL_UPDATE: Comprehensive analysis of all aspects (exploration, combat, puzzles, items)
2. SELECTIVE_UPDATE: Focus on specific aspects that show promise in this data
3. CONSOLIDATION_ONLY: Don't analyze new data, just consolidate existing knowledge
4. ESCAPE_ANALYSIS: Focus on identifying and solving stuck/loop situations

Consider:
- What type of gameplay situation this represents
- Whether the data shows progress or stagnation
- How it relates to existing knowledge

Respond with just the strategy name: FULL_UPDATE, SELECTIVE_UPDATE, CONSOLIDATION_ONLY, or ESCAPE_ANALYSIS"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at determining optimal knowledge extraction strategies for interactive fiction gameplay.",
                },
                {"role": "user", "content": prompt},
            ]
            
            # Log the full prompt for evaluation
            self._log_prompt_to_file(messages, "knowledge_strategy")
            
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=messages,
                temperature=0.2,
                max_tokens=100,
            )

            strategy = response.choices[0].message.content.strip().upper()

            # Validate strategy
            valid_strategies = [
                "FULL_UPDATE",
                "SELECTIVE_UPDATE",
                "CONSOLIDATION_ONLY",
                "ESCAPE_ANALYSIS",
            ]
            if strategy not in valid_strategies:
                return "SELECTIVE_UPDATE"  # Default fallback

            return strategy

        except Exception as e:
            print(f"  ⚠️ Strategy determination failed: {e}")
            return "SELECTIVE_UPDATE"

    def _perform_targeted_analysis(
        self, turn_data: Dict, strategy: str
    ) -> Optional[str]:
        """Perform analysis based on the determined strategy."""

        if strategy == "CONSOLIDATION_ONLY":
            return self._consolidate_existing_knowledge()
        elif strategy == "ESCAPE_ANALYSIS":
            return self._analyze_escape_strategies(turn_data)
        elif strategy == "SELECTIVE_UPDATE":
            return self._analyze_selective_insights(turn_data)
        else:  # FULL_UPDATE
            return self._analyze_full_insights(turn_data)

    def _analyze_selective_insights(self, turn_data: Dict) -> Optional[str]:
        """Focus on the most promising aspects of the turn data."""

        # Prepare action sequence
        actions_text = ""
        for action in turn_data["actions_and_responses"][
            :20
        ]:  # Limit to avoid token overflow
            actions_text += f"Turn {action['turn']}: {action['action']} -> {action['response'][:150]}...\n"

        # Include agent instructions context if available
        agent_context = ""
        if self.agent_instructions:
            agent_context = f"""
**IMPORTANT - EXISTING AGENT INSTRUCTIONS CONTEXT:**
The agent already has comprehensive instructions covering:
- Basic command syntax and parser rules
- General exploration and combat strategies  
- Anti-repetition rules and failure handling
- Command format requirements and common actions

Focus your analysis on DISCOVERED KNOWLEDGE that goes BEYOND these basics:
- Specific location details and secrets
- Exact item locations and usage sequences  
- Specific puzzle solutions and tricks
- Particular dangers and how to handle them
- Advanced tactics discovered through play
- Specific sequences that work well

DO NOT repeat basic information already covered in the agent instructions."""

        prompt = f"""Analyze this Zork gameplay data and extract the most valuable strategic insights.{agent_context}

TURN RANGE: {turn_data["start_turn"]}-{turn_data["end_turn"]}
SCORE CHANGES: {turn_data["score_changes"]}
LOCATION CHANGES: {turn_data["location_changes"]}

ACTION SEQUENCE:
{actions_text}

Focus on extracting insights that are:
1. Actionable and specific
2. Non-obvious or newly discovered
3. Likely to improve future performance
4. Beyond basic gameplay mechanics (which are already covered)

Provide insights in these categories only if they contain valuable information:
- **Navigation Discoveries**: New paths, connections, or movement strategies
- **Item Insights**: Useful items found, usage patterns, or combinations
- **Puzzle Solutions**: Successful problem-solving approaches
- **Danger Avoidance**: Threats identified and how to handle them
- **Efficiency Improvements**: Better action sequences or time-saving approaches

Skip categories that don't have meaningful insights from this data.
Be specific about locations, items, and sequences when relevant."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting actionable strategic insights from interactive fiction gameplay. Focus on quality over quantity.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  ⚠️ Selective analysis failed: {e}")
            return None

    def _analyze_escape_strategies(self, turn_data: Dict) -> Optional[str]:
        """Analyze data to identify escape strategies from stuck situations."""

        # Focus on recent actions and any changes
        recent_actions = turn_data["actions_and_responses"][-10:]
        actions_text = "\n".join(
            [
                f"Turn {action['turn']}: {action['action']} -> {action['response'][:100]}..."
                for action in recent_actions
            ]
        )

        prompt = f"""This gameplay data appears to show a stuck or repetitive situation. Analyze it to identify escape strategies.

RECENT ACTIONS:
{actions_text}

LOCATION CHANGES: {turn_data["location_changes"]}
SCORE CHANGES: {turn_data["score_changes"]}

Focus on:
1. **Situation Analysis**: What type of stuck situation is this? (maze, puzzle, combat, etc.)
2. **Escape Strategies**: What actions might break the pattern or lead to progress?
3. **Alternative Approaches**: What different strategies should be tried?
4. **Warning Signs**: How to recognize and avoid this situation in the future?

Provide specific, actionable advice for escaping this type of situation."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at identifying and solving stuck situations in interactive fiction games.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=1000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  ⚠️ Escape analysis failed: {e}")
            return None

    def _analyze_full_insights(self, turn_data: Dict) -> Optional[str]:
        """Perform comprehensive analysis of all aspects."""

        # This is similar to the original episode analysis but for turn ranges
        actions_text = ""
        for action in turn_data["actions_and_responses"][:30]:
            actions_text += f"Turn {action['turn']}: {action['action']} -> {action['response'][:150]}...\n"

        prompt = f"""Analyze this Zork gameplay data and provide comprehensive strategic insights.

TURN RANGE: {turn_data["start_turn"]}-{turn_data["end_turn"]}
SCORE CHANGES: {turn_data["score_changes"]}
LOCATION CHANGES: {turn_data["location_changes"]}

ACTION SEQUENCE:
{actions_text}

Provide insights in these categories:
1. **Key Successful Strategies**: What actions or patterns led to progress?
2. **Critical Mistakes**: What actions hindered progress or caused setbacks?
3. **Navigation Insights**: How effectively was the world navigated?
4. **Item Management**: Were items collected and used effectively?
5. **Combat/Danger Handling**: How well were threats managed?
6. **Learning Opportunities**: What should be done differently?

Focus on actionable insights that would help improve future gameplay. Be specific about locations, items, and sequences when relevant."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing interactive fiction gameplay to identify successful strategies and common mistakes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  ⚠️ Full analysis failed: {e}")
            return None

    def _consolidate_existing_knowledge(self) -> Optional[str]:
        """Consolidate and improve existing knowledge without new data."""

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                full_knowledge = f.read()
        except:
            return None
            
        # Trim map section for LLM processing (map is handled separately)
        current_knowledge = self._trim_map_section(full_knowledge)

        prompt = f"""Review and consolidate this Zork strategy guide to improve its quality and organization.

**IMPORTANT**: This knowledge base is for an AI language model, not a human player.
- Remove any human-centric advice (drawing maps, taking notes, etc.)
- Focus on computational decision-making patterns
- Use precise command syntax and logical conditions
- Provide algorithmic approaches to problem-solving

CURRENT KNOWLEDGE BASE:
{current_knowledge}

Tasks:
1. Remove any contradictory or outdated information
2. Consolidate redundant advice into clearer statements
3. Improve organization and clarity
4. Ensure all advice is actionable and specific for an AI agent
5. **Remove human-centric references** (paper mapping, manual note-taking, etc.)
6. **Focus on algorithmic decision patterns** that an LLM can follow

Maintain the same general structure but improve the content quality.
Do not add new information - only reorganize and clarify existing knowledge for AI consumption."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at consolidating and organizing strategic knowledge for AI language models playing interactive fiction games. Focus on algorithmic approaches and computational patterns rather than human-centric advice.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  ⚠️ Knowledge consolidation failed: {e}")
            return None

    def _intelligent_knowledge_merge(self, new_insights: str, strategy: str) -> bool:
        """Intelligently merge new insights with existing knowledge."""

        # Load existing knowledge
        existing_knowledge = ""
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                existing_knowledge = f.read()
        except:
            existing_knowledge = ""

        if strategy == "CONSOLIDATION_ONLY":
            # Replace entirely with consolidated version, but preserve map section
            merged_knowledge = self._preserve_map_section(existing_knowledge, new_insights)
        else:
            # Intelligent merge
            merged_knowledge = self._merge_insights_with_existing(
                existing_knowledge, new_insights, strategy
            )
            # Preserve map section after merge
            if merged_knowledge:
                merged_knowledge = self._preserve_map_section(existing_knowledge, merged_knowledge)

        if not merged_knowledge:
            return False

        # Save merged knowledge
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(merged_knowledge)
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to save knowledge: {e}")
            return False

    def _merge_insights_with_existing(
        self, existing: str, new_insights: str, strategy: str
    ) -> Optional[str]:
        """Use LLM to intelligently merge new insights with existing knowledge."""

        if not existing.strip():
            # No existing knowledge, create new guide
            return self._create_new_knowledge_base(new_insights)
            
        # Trim map section for LLM processing (map is handled separately)
        existing_without_map = self._trim_map_section(existing)

        prompt = f"""Merge these new strategic insights with the existing Zork knowledge base.

**IMPORTANT**: This knowledge base is for an AI language model, not a human player. 
- Use direct, actionable instructions
- Avoid references to human activities (drawing maps, taking notes, etc.)
- Focus on computational decision-making patterns
- Use precise command syntax and logical conditions

EXISTING KNOWLEDGE BASE:
{existing_without_map}

NEW INSIGHTS ({strategy}):
{new_insights}

Merge guidelines:
1. Preserve all valuable information from both sources
2. Resolve contradictions by favoring more specific/recent insights
3. Organize information clearly with consistent structure
4. Remove redundancy while maintaining completeness
5. Update outdated information with newer insights
6. **Remove any human-centric advice** (paper mapping, manual note-taking, etc.)
7. **Focus on algorithmic decision patterns** that an LLM can follow

The merged guide should be more comprehensive and accurate than either individual source.
Maintain the existing structure but enhance it with the new insights."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at merging and consolidating strategic knowledge for AI language models. Create guides that focus on algorithmic decision-making patterns, precise command syntax, and computational approaches rather than human intuition or manual activities.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  ⚠️ Knowledge merging failed: {e}")
            return None

    def _create_new_knowledge_base(self, insights: str) -> str:
        """Create a new knowledge base from insights."""

        prompt = f"""Create a comprehensive Zork strategy guide from these insights.

**IMPORTANT**: This knowledge base is for an AI language model, not a human player.
- Use direct, actionable instructions that an LLM can follow
- Avoid human-centric advice (drawing maps, taking notes, etc.)
- Focus on computational decision-making patterns
- Use precise command syntax and logical conditions
- Provide algorithmic approaches to problem-solving

INSIGHTS:
{insights}

Structure the guide with these sections:
## PRIORITY OBJECTIVES
## NAVIGATION STRATEGY  
## ITEM COLLECTION & USAGE
## COMBAT & DANGER MANAGEMENT
## COMMON MISTAKES TO AVOID
## ADVANCED TACTICS

Make the guide practical and actionable for an AI agent. Use specific locations, items, and commands when relevant.
Focus on decision trees, conditional logic, and systematic approaches rather than human intuition."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert strategy guide writer for AI language models playing interactive fiction games. Focus on algorithmic approaches, decision trees, and computational patterns rather than human-centric advice.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  ⚠️ New knowledge base creation failed: {e}")
            return f"# Zork Strategy Guide\n\n{insights}"

    def _trim_map_section(self, knowledge_content: str) -> str:
        """Remove the map section from knowledge content for LLM processing."""
        if not knowledge_content or "## CURRENT WORLD MAP" not in knowledge_content:
            return knowledge_content
            
        # Split on map section and take only the part before it
        sections = knowledge_content.split("## CURRENT WORLD MAP")
        return sections[0].strip() if sections else knowledge_content

    def _preserve_map_section(self, original_knowledge: str, new_knowledge: str) -> str:
        """Preserve the map section from original knowledge in the new knowledge."""
        if not original_knowledge or "## CURRENT WORLD MAP" not in original_knowledge:
            return new_knowledge
            
        # Extract map section from original
        map_start = original_knowledge.find("## CURRENT WORLD MAP")
        if map_start == -1:
            return new_knowledge
            
        map_section = original_knowledge[map_start:]
        
        # Add map section to new knowledge
        return f"{new_knowledge.rstrip()}\n\n{map_section}"

    def update_knowledge_with_map(self, episode_id: str, game_map: MapGraph) -> bool:
        """
        Update the knowledge base with current map information.
        
        Args:
            episode_id: Current episode ID
            game_map: The current MapGraph instance
            
        Returns:
            True if map was updated, False if skipped
        """
        print("🗺️ Updating knowledge base with current map...")
        
        # Generate mermaid diagram from current map
        mermaid_map = game_map.render_mermaid()
        if not mermaid_map or not mermaid_map.strip():
            print("  ⚠️ No map data available to update")
            return False
            
        # Load existing knowledge
        existing_knowledge = ""
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                existing_knowledge = f.read()
        except:
            existing_knowledge = ""
            
        # Update or add map section
        updated_knowledge = self._update_map_section(existing_knowledge, mermaid_map)
        
        if not updated_knowledge:
            print("  ⚠️ Failed to update map section")
            return False
            
        # Save updated knowledge
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(updated_knowledge)
            print("  ✅ Map section updated in knowledge base")
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to save updated knowledge: {e}")
            return False

    def _update_map_section(self, existing_knowledge: str, mermaid_map: str) -> Optional[str]:
        """Update or add the map section to the knowledge base."""
        
        map_section = f"""

## CURRENT WORLD MAP

```mermaid
{mermaid_map}
```

"""
        
        # Check if there's already a map section
        if "## CURRENT WORLD MAP" in existing_knowledge:
            # Replace existing map section
            lines = existing_knowledge.split('\n')
            new_lines = []
            in_map_section = False
            in_mermaid_block = False
            
            for line in lines:
                if line.strip() == "## CURRENT WORLD MAP":
                    in_map_section = True
                    new_lines.append(line)
                    continue
                    
                if in_map_section:
                    if line.strip().startswith("```mermaid"):
                        in_mermaid_block = True
                        new_lines.append(line)
                        new_lines.append(mermaid_map)
                        continue
                    elif line.strip() == "```" and in_mermaid_block:
                        in_mermaid_block = False
                        new_lines.append(line)
                        in_map_section = False
                        continue
                    elif line.strip().startswith("##") and not in_mermaid_block:
                        # Hit next section, end map section
                        in_map_section = False
                        new_lines.append(line)
                        continue
                    elif not in_mermaid_block:
                        # Skip old map content outside mermaid block
                        continue
                    else:
                        # Skip old mermaid content
                        continue
                else:
                    new_lines.append(line)
                    
            return '\n'.join(new_lines)
        else:
            # Add new map section at the end
            if existing_knowledge.strip():
                return existing_knowledge.rstrip() + map_section
            else:
                # Create minimal knowledge base with just the map
                return f"""# Zork Strategy Guide for AI Agent

## BASIC STRATEGY
- Always begin each location with 'look' to gather environmental data
- Use systematic exploration patterns with cardinal directions (north, south, east, west)
- Execute 'take' commands for all portable items - inventory constraints are minimal
- Parse all text output for puzzle-solving information and command hints
- Prioritize information extraction over rapid action execution
{map_section}"""

    def _build_map_from_logs(self, episode_id: str) -> Optional[str]:
        """
        Build a mermaid map from log data for a specific episode.
        
        Args:
            episode_id: Episode ID to extract map data for
            
        Returns:
            Mermaid diagram string or None if failed
        """
        try:
            # Create a temporary MapGraph to build from logs
            temp_map = MapGraph()
            
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # Skip entries not from this episode
                        if log_entry.get("episode_id") != episode_id:
                            continue
                            
                        event_type = log_entry.get("event_type", "")
                        
                        if event_type == "extracted_info":
                            extracted_info = log_entry.get("extracted_info", {})
                            location_name = extracted_info.get("current_location_name", "")
                            exits = extracted_info.get("exits", [])
                            
                            if location_name and location_name != "Unknown Location":
                                # Add room and exits
                                temp_map.add_room(location_name)
                                temp_map.update_room_exits(location_name, exits)
                                
                        elif event_type == "movement_connection_created":
                            from_room = log_entry.get("from_room", "")
                            to_room = log_entry.get("to_room", "")
                            action = log_entry.get("action", "")
                            
                            if from_room and to_room and action:
                                temp_map.add_connection(from_room, action, to_room)
                                
                    except json.JSONDecodeError:
                        continue
                        
            # Generate mermaid representation
            mermaid_map = temp_map.render_mermaid()
            return mermaid_map if mermaid_map and mermaid_map.strip() else None
            
        except Exception as e:
            print(f"  ⚠️ Failed to build map from logs: {e}")
            return None
