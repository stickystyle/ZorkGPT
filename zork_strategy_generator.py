"""
ZorkStrategyGenerator module for generating strategic guides from episode data.

This module was previously named LLMStrategyGenerator and has been renamed for consistency.
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
        
        # Load agent instructions to avoid duplication
        self.agent_instructions = self._load_agent_instructions()

    def _load_agent_instructions(self) -> str:
        """Load the agent.md prompt to understand what's already covered."""
        try:
            with open("agent.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"  ⚠️ Could not load agent.md: {e}")
            return ""

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

        # Check if this is the very first knowledge update (no knowledge base exists)
        is_first_update = (
            not os.path.exists(self.output_file)
            or os.path.getsize(self.output_file) == 0
        )

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
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at evaluating gameplay data quality for knowledge extraction. Be honest about data that would produce low-quality or repetitive insights.",
                    },
                    {"role": "user", "content": prompt},
                ],
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
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at determining optimal knowledge extraction strategies for interactive fiction gameplay.",
                    },
                    {"role": "user", "content": prompt},
                ],
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
Be specific about locations, items, and sequences when relevant.

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
                current_knowledge = f.read()
        except:
            return None

        prompt = f"""Review and consolidate this Zork strategy guide to improve its quality and organization.

CURRENT KNOWLEDGE BASE:
{current_knowledge}

Tasks:
1. Remove any contradictory or outdated information
2. Consolidate redundant advice into clearer statements
3. Improve organization and clarity
4. Ensure all advice is actionable and specific

Maintain the same general structure but improve the content quality.
Do not add new information - only reorganize and clarify existing knowledge."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at consolidating and organizing strategic knowledge for interactive fiction games.",
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
            # Replace entirely with consolidated version
            merged_knowledge = new_insights
        else:
            # Intelligent merge
            merged_knowledge = self._merge_insights_with_existing(
                existing_knowledge, new_insights, strategy
            )

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

        prompt = f"""Merge these new strategic insights with the existing Zork knowledge base.

EXISTING KNOWLEDGE BASE:
{existing}

NEW INSIGHTS ({strategy}):
{new_insights}

Merge guidelines:
1. Preserve all valuable information from both sources
2. Resolve contradictions by favoring more specific/recent insights
3. Organize information clearly with consistent structure
4. Remove redundancy while maintaining completeness
5. Update outdated information with newer insights

The merged guide should be more comprehensive and accurate than either individual source.
Maintain the existing structure but enhance it with the new insights."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at merging and consolidating strategic knowledge to create comprehensive guides.",
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

INSIGHTS:
{insights}

Structure the guide with these sections:
## PRIORITY OBJECTIVES
## NAVIGATION STRATEGY  
## ITEM COLLECTION & USAGE
## COMBAT & DANGER MANAGEMENT
## COMMON MISTAKES TO AVOID
## ADVANCED TACTICS

Make the guide practical and actionable. Use specific locations, items, and commands when relevant."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert strategy guide writer for interactive fiction games.",
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


class ZorkStrategyGenerator:
    """
    Generates strategy guides using LLM analysis of raw episode data.
    """

    def __init__(self):
        self.log_file = "zork_episode_log.jsonl"
        self.output_file = "knowledgebase.md"

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=env.str("CLIENT_BASE_URL", None),
            api_key=env.str("CLIENT_API_KEY", None),
        )

        # Model for analysis
        self.analysis_model = env.str("ANALYSIS_MODEL", "gpt-4")

        # Sliding window configuration
        self.sliding_window_size = env.int("EPISODE_SLIDING_WINDOW_SIZE", 10)

    def generate_strategy_guide(self) -> str:
        """
        Generate a strategy guide using LLM analysis of episode logs.
        Uses cumulative knowledge merging when episodes exceed sliding window size.

        Returns:
            Complete strategy guide as markdown string
        """
        print("🧠 Generating LLM-based strategy guide...")
        print(f"  ⚙️ Sliding window size: {self.sliding_window_size} episodes")

        # Parse episodes from logs
        print("  📚 Parsing episode logs...")
        episodes = self._parse_episodes_from_logs()

        if not episodes:
            print("  ⚠️ No episodes found in logs")
            return self._generate_empty_guide("No episodes found to analyze")

        print(f"  📖 Found {len(episodes)} episodes to analyze")

        # Analyze recent episodes with LLM (sliding window)
        print("  🔍 Analyzing individual episodes...")
        episode_analyses = []
        for i, episode in enumerate(
            episodes[-self.sliding_window_size :], 1
        ):  # Analyze recent episodes
            print(f"    Episode {i}/{min(self.sliding_window_size, len(episodes))}...")
            analysis = self._analyze_episode_with_llm(episode, i)
            if analysis:
                episode_analyses.append(analysis)
            else:
                print(f"    ⚠️ Episode {i} analysis failed")

        if not episode_analyses:
            print("  ⚠️ No successful episode analyses")
            return self._generate_empty_guide(
                "Episode analysis failed - check API configuration and episode data"
            )

        # Generate strategic guide from recent episodes
        print("  📋 Generating strategic guide from recent episodes...")
        new_strategic_guide = self._generate_overall_guide(episode_analyses)

        # If we have more episodes than sliding window and existing knowledge, merge with previous knowledge
        if len(episodes) >= self.sliding_window_size + 1:
            print("  🔄 Attempting knowledge merging with existing guide...")
            merged_guide = self._merge_with_existing_knowledge(new_strategic_guide)
            if merged_guide:
                strategic_guide = merged_guide
            else:
                print("  ⚠️ Knowledge merging failed, using new guide only")
                strategic_guide = new_strategic_guide
        else:
            strategic_guide = new_strategic_guide

        # Build and add Mermaid map at the end (using sliding window)
        mermaid_map = self._build_map_from_episodes(
            episodes[-self.sliding_window_size :]
        )
        if mermaid_map:
            strategic_guide += (
                f"\n\n## CURRENT WORLD MAP\n\n```mermaid\n{mermaid_map}\n```\n"
            )

        return strategic_guide

    def _parse_episodes_from_logs(self) -> List[List[Dict]]:
        """Parse episodes from JSON log file."""
        episodes = []
        current_episode = []
        current_episode_start_time = None

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        episode_id = log_entry.get("episode_id")
                        event_type = log_entry.get("event_type", "")

                        # Skip entries with null episode_id
                        if episode_id is None:
                            continue

                        # Normalize episode_id to handle format inconsistencies
                        normalized_episode_id = self._normalize_episode_id(episode_id)

                        # Detect new episode by episode_start event or significant time gap
                        if event_type == "episode_start":
                            # Start of a new episode
                            if current_episode:
                                episodes.append(current_episode)
                            current_episode = []
                            current_episode_start_time = normalized_episode_id

                        # Only include entries that belong to the current episode
                        if (
                            current_episode_start_time
                            and normalized_episode_id == current_episode_start_time
                        ):
                            current_episode.append(log_entry)

                    except json.JSONDecodeError:
                        continue

            if current_episode:
                episodes.append(current_episode)

        except FileNotFoundError:
            print(f"  ⚠️ Log file {self.log_file} not found")
            return []

        return episodes

    def _normalize_episode_id(self, episode_id: str) -> str:
        """
        Normalize episode_id to handle format inconsistencies.

        Converts both "2025-05-25T07:47:56" and "20250525_074756"
        to the same normalized format.
        """
        if not episode_id:
            return ""

        # Handle ISO format: "2025-05-25T07:47:56"
        if "T" in episode_id and "-" in episode_id:
            # Extract date and time parts
            date_part, time_part = episode_id.split("T")
            date_clean = date_part.replace("-", "")
            time_clean = time_part.replace(":", "")
            return f"{date_clean}_{time_clean}"

        # Handle underscore format: "20250525_074756"
        if "_" in episode_id and len(episode_id) == 15:
            return episode_id

        # For any other format, return as-is
        return episode_id

    def _analyze_episode_with_llm(
        self, episode_logs: List[Dict], episode_num: int
    ) -> Optional[str]:
        """
        Analyze a single episode using LLM.

        Args:
            episode_logs: List of log entries for the episode
            episode_num: Episode number for reference

        Returns:
            LLM analysis of the episode, or None if failed
        """
        # Extract the action-response sequence
        episode_data = self._extract_episode_sequence(episode_logs)

        print(f"    Debug - Episode {episode_num} data:")
        print(f"      Episode ID: {episode_data['episode_id']}")
        print(
            f"      Action-response pairs: {len(episode_data['actions_and_responses'])}"
        )
        print(f"      Final score: {episode_data['final_score']}")
        print(f"      Outcome: {episode_data['outcome']}")

        if not episode_data["actions_and_responses"]:
            print(f"    ⚠️ Episode {episode_num} has no action-response pairs")
            return None

        # Create prompt for episode analysis
        prompt = self._create_episode_analysis_prompt(episode_data, episode_num)

        try:
            print(f"    🔗 Making API call for episode {episode_num} analysis...")
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

            analysis = response.choices[0].message.content.strip()
            print(
                f"    ✅ Episode {episode_num} analysis complete ({len(analysis)} chars)"
            )
            return analysis

        except Exception as e:
            print(f"    ⚠️ Failed to analyze episode {episode_num}: {e}")
            return None

    def _extract_episode_sequence(self, episode_logs: List[Dict]) -> Dict:
        """Extract the action-response sequence from episode logs."""
        episode_data = {
            "episode_id": "",
            "final_score": 0,
            "max_score": 585,
            "turn_count": 0,
            "actions_and_responses": [],
            "outcome": "unknown",
        }

        # Extract basic episode info
        for log_entry in episode_logs:
            event_type = log_entry.get("event_type", "")

            if event_type == "episode_start":
                episode_data["episode_id"] = log_entry.get("episode_id", "")
            elif event_type == "episode_end":
                episode_data["final_score"] = log_entry.get("zork_score", 0)
                episode_data["turn_count"] = log_entry.get("turn_count", 0)
                # Determine outcome based on final score
                if episode_data["final_score"] >= 350:
                    episode_data["outcome"] = "excellent"
                elif episode_data["final_score"] >= 200:
                    episode_data["outcome"] = "good"
                elif episode_data["final_score"] >= 100:
                    episode_data["outcome"] = "moderate"
                elif episode_data["final_score"] >= 50:
                    episode_data["outcome"] = "poor"
                else:
                    episode_data["outcome"] = "very_poor"

        # Extract action-response pairs
        for log_entry in episode_logs:
            event_type = log_entry.get("event_type", "")

            if event_type == "final_action_selection":
                action = log_entry.get("agent_action", "")
                if action:
                    episode_data["actions_and_responses"].append(
                        {
                            "action": action,
                            "response": "",
                            "critic_score": log_entry.get("critic_score", 0),
                        }
                    )

            elif event_type == "zork_response":
                response = log_entry.get("zork_response", "")
                if response and episode_data["actions_and_responses"]:
                    # Update the last action with its response
                    episode_data["actions_and_responses"][-1]["response"] = response

        return episode_data

    def _create_episode_analysis_prompt(
        self, episode_data: Dict, episode_num: int
    ) -> str:
        """Create a prompt for LLM to analyze an episode."""
        actions_text = ""
        for i, pair in enumerate(
            episode_data["actions_and_responses"][:50], 1
        ):  # Limit to first 50 actions
            action = pair["action"]
            response = (
                pair["response"][:200] + "..."
                if len(pair["response"]) > 200
                else pair["response"]
            )
            critic_score = pair.get("critic_score", 0)
            actions_text += (
                f"Turn {i}: {action} -> {response} (Critic: {critic_score:.2f})\n"
            )

        prompt = f"""Analyze this Zork episode and provide strategic insights:

**Episode {episode_num} Summary:**
- Episode ID: {episode_data["episode_id"]}
- Turns: {episode_data["turn_count"]}
- Final Score: {episode_data["final_score"]}/{episode_data["max_score"]}
- Outcome: {episode_data["outcome"]}

**Action Sequence (first 50 turns):**
{actions_text}

Please analyze this episode and provide:
1. **Key Successful Strategies**: What actions or patterns led to progress?
2. **Critical Mistakes**: What actions hindered progress or caused setbacks?
3. **Navigation Insights**: How effectively did the agent navigate the world?
4. **Item Management**: Were items collected and used effectively?
5. **Combat/Danger Handling**: How well were threats managed?
6. **Learning Opportunities**: What should be done differently next time?

Focus on actionable insights that would help improve future gameplay. Be specific about locations, items, and sequences when relevant.
"""
        return prompt

    def _generate_overall_guide(self, episode_analyses: List[str]) -> str:
        """Generate an overall strategic guide from multiple episode analyses."""
        combined_analyses = "\n\n---\n\n".join(episode_analyses)

        prompt = f"""Based on the following episode analyses, create a comprehensive strategic guide for playing Zork.

**Episode Analyses:**
{combined_analyses}

Output ONLY the strategic guide content in markdown format, starting directly with the first section header. Do not include any conversational text, introductions, or explanations about the guide itself.

Use exactly this structure:

## PRIORITY OBJECTIVES
- Most important goals to pursue first
- Critical items to find early

## NAVIGATION STRATEGY  
- Efficient exploration patterns
- Key areas to prioritize
- Areas to avoid or approach carefully

## ITEM COLLECTION & USAGE
- Essential items and their locations
- Proper usage sequences
- Items to prioritize vs. items to ignore

## COMBAT & DANGER MANAGEMENT
- How to handle threats effectively
- When to fight vs. when to flee
- Preparation strategies

## COMMON MISTAKES TO AVOID
- Repeated errors seen across episodes
- Actions that consistently fail
- Time-wasting behaviors

## ADVANCED TACTICS
- Sophisticated strategies for experienced play
- Sequence optimizations
- Hidden or non-obvious solutions

Make this guide practical and actionable. Use specific locations, items, and commands when relevant. Focus on insights that will lead to higher scores and more efficient gameplay.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert strategy guide writer for interactive fiction games. Create clear, actionable guides that help players improve their performance.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ Failed to generate overall guide: {e}")
            return self._generate_empty_guide("Failed to generate strategic guide")

    def _build_map_from_episodes(self, episodes: List[List[Dict]]) -> Optional[str]:
        """
        Build Mermaid diagram from episode data.

        Uses Mermaid instead of ASCII because:
        - LLMs can parse structured Mermaid syntax more reliably
        - Easier to merge multiple maps programmatically
        - Standard syntax that LLMs are trained on
        - Can easily extend with metadata (room types, items, etc.)
        """
        # Build individual episode maps
        episode_maps = self._build_individual_episode_maps(episodes)

        if not episode_maps:
            return None

        # If only one map, return it directly
        if len(episode_maps) == 1:
            return episode_maps[0]

        # Otherwise, build consensus map
        return self._build_consensus_map_with_llm(episode_maps)

    def _build_individual_episode_maps(self, episodes: List[List[Dict]]) -> List[str]:
        """Build maps for individual episodes."""
        episode_maps = []

        for episode_idx, episode_logs in enumerate(episodes):
            try:
                # Create a MapGraph for this episode
                episode_map = MapGraph()

                # Track current room for connections
                current_room = None

                for log_entry in episode_logs:
                    event_type = log_entry.get("event_type", "")

                    if event_type == "extracted_info":
                        extracted_info = log_entry.get("extracted_info", {})
                        location_name = extracted_info.get("current_location_name", "")
                        exits = extracted_info.get("exits", [])

                        if location_name and location_name != "Unknown Location":
                            # Add room and exits
                            episode_map.add_room(location_name)
                            episode_map.update_room_exits(location_name, exits)
                            current_room = location_name

                    elif event_type == "movement_connection_created":
                        from_room = log_entry.get("from_room", "")
                        to_room = log_entry.get("to_room", "")
                        action = log_entry.get("action", "")

                        if from_room and to_room and action:
                            episode_map.add_connection(from_room, action, to_room)

                # Generate Mermaid representation (easier for LLMs to parse)
                mermaid_map = episode_map.render_mermaid()
                if mermaid_map and mermaid_map.strip():
                    episode_maps.append(mermaid_map)

            except Exception as e:
                print(f"    ⚠️ Failed to build map for episode {episode_idx}: {e}")
                continue

        return episode_maps

    def _build_consensus_map_with_llm(self, episode_maps: List[str]) -> str:
        """Build a consensus map using LLM analysis of multiple episode maps."""
        maps_text = self._format_episode_maps_for_prompt(episode_maps)

        prompt = f"""Based on these Mermaid diagrams from different Zork episodes, create a single consensus map that represents the most accurate and complete world layout:

{maps_text}

Rules for creating the consensus map:
1. Rooms that appear in multiple maps are definitely real
2. Connections that appear consistently should be included
3. If maps conflict, prefer the more detailed/complete version
4. Use proper Mermaid syntax: graph TD with nodes like R1["Room Name"] and connections like R1 -->|"direction"| R2
5. Keep node IDs consistent and meaningful
6. Include all important rooms discovered across episodes
7. Use solid arrows (-->) for confirmed connections and dotted arrows (-.->)  for uncertain ones

Create a single, well-formatted Mermaid diagram using 'graph TD' format:
"""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing and combining map data to create accurate world representations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ Failed to build consensus map: {e}")
            # Return the most detailed map as fallback
            return max(episode_maps, key=len) if episode_maps else ""

    def _format_episode_maps_for_prompt(self, episode_maps: List[str]) -> str:
        """Format episode maps for LLM prompt."""
        formatted_maps = []
        for i, episode_map in enumerate(episode_maps, 1):
            formatted_maps.append(
                f"**Episode {i} Map:**\n```mermaid\n{episode_map}\n```\n"
            )
        return "\n".join(formatted_maps)

    def _merge_with_existing_knowledge(self, new_guide: str) -> Optional[str]:
        """Merge new strategic guide with existing knowledge base."""
        if not os.path.exists(self.output_file):
            return new_guide

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                existing_guide = f.read()

            if not existing_guide.strip():
                return new_guide

            prompt = self._create_knowledge_merging_prompt(existing_guide, new_guide)

            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at merging and consolidating strategic knowledge to create comprehensive guides.",
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

    def _create_knowledge_merging_prompt(
        self, existing_guide: str, new_guide: str
    ) -> str:
        """Create prompt for merging existing and new knowledge."""
        return f"""Merge these two Zork strategy guides into a single, comprehensive guide:

**EXISTING KNOWLEDGE BASE:**
{existing_guide}

**NEW STRATEGIC INSIGHTS:**
{new_guide}

Create a merged guide that:
1. Preserves all valuable information from both guides
2. Resolves any contradictions by favoring more specific/recent insights
3. Organizes information clearly with consistent structure
4. Removes redundancy while maintaining completeness
5. Updates outdated information with newer insights

The merged guide should be more comprehensive and accurate than either individual guide.
"""

    def _generate_empty_guide(self, reason: str) -> str:
        """Generate a minimal guide when analysis fails."""
        return f"""# Zork Strategy Guide

## Analysis Status
{reason}

## Basic Strategy
- Start by examining your surroundings with 'look'
- Explore systematically using cardinal directions
- Take all items you find - inventory space is usually not limited
- Read any text carefully for clues about puzzles

*This guide will be updated as more episode data becomes available.*
"""

    def save_strategy_guide(self) -> str:
        """Generate and save the strategy guide."""
        guide_content = self.generate_strategy_guide()

        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(guide_content)

        print(f"📝 Strategy guide saved to {self.output_file}")
        return self.output_file


if __name__ == "__main__":
    system = ZorkStrategyGenerator()
    output_file = system.save_strategy_guide()
    print(f"\n✅ Strategy guide generation complete!")
    print(f"📄 Output: {output_file}")
