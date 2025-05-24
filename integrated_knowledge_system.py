#!/usr/bin/env python3
"""
LLM-Based Strategic Knowledge System for ZorkGPT

This module uses LLMs to directly analyze episode logs and generate strategic guides,
rather than pre-processing the data. Much simpler and more effective approach.
"""

import json
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

from openai import OpenAI
import environs

# Import the proper map and movement analysis tools
from map_graph import MapGraph
from movement_analyzer import MovementAnalyzer, create_movement_context

# Load environment variables
env = environs.Env()
env.read_env()


class LLMStrategyGenerator:
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

        # Build and add ASCII map at the end (using sliding window)
        ascii_map = self._build_map_from_episodes(episodes[-self.sliding_window_size :])
        if ascii_map:
            strategic_guide += f"\n\n## CURRENT WORLD MAP\n\n```\n{ascii_map}\n```\n"

        return strategic_guide

    def _parse_episodes_from_logs(self) -> List[List[Dict]]:
        """Parse episodes from JSON log file."""
        episodes = []
        current_episode = []
        current_episode_id = None

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        episode_id = log_entry.get("episode_id")

                        # Skip entries with null episode_id
                        if episode_id is None:
                            continue

                        if episode_id != current_episode_id:
                            if current_episode:
                                episodes.append(current_episode)
                            current_episode = []
                            current_episode_id = episode_id

                        current_episode.append(log_entry)

                    except json.JSONDecodeError:
                        continue

            if current_episode:
                episodes.append(current_episode)

        except FileNotFoundError:
            print(f"  ⚠️ Log file {self.log_file} not found")
            return []

        return episodes

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
                episode_data["max_score"] = log_entry.get("max_score", 585)
                episode_data["turn_count"] = log_entry.get("turn_count", 0)
            elif event_type == "episode_end_debug":
                # Extract end reason from debug event
                reasons = log_entry.get("reasons", [])
                if reasons:
                    primary_reason = reasons[0]
                    if "max_turns_reached" in primary_reason:
                        episode_data["outcome"] = "turn_limit"
                    elif "zork_process_not_running" in primary_reason:
                        episode_data["outcome"] = "process_ended"
                    else:
                        episode_data["outcome"] = "other_end"
            elif event_type == "game_over":
                reason = log_entry.get("reason", "")
                if "died" in reason.lower():
                    episode_data["outcome"] = "died"
                elif "victory" in reason.lower():
                    episode_data["outcome"] = "victory"
                else:
                    episode_data["outcome"] = "game_over"

        # Extract action-response pairs
        current_action = None
        for log_entry in episode_logs:
            event_type = log_entry.get("event_type", "")

            if event_type == "final_action_selection":
                current_action = log_entry.get("agent_action", "")
            elif event_type == "zork_response" and current_action:
                response = log_entry.get("zork_response", "")
                episode_data["actions_and_responses"].append(
                    {"action": current_action, "response": response}
                )
                current_action = None

        return episode_data

    def _create_episode_analysis_prompt(
        self, episode_data: Dict, episode_num: int
    ) -> str:
        """Create a prompt for analyzing a single episode."""

        # Limit to key parts of the episode to avoid token limits
        actions_responses = episode_data["actions_and_responses"]

        # Take first 5, last 5, and some middle actions if episode is long
        if len(actions_responses) <= 15:
            selected_actions = actions_responses
        else:
            # Take first 5, middle 5, last 5
            middle_start = len(actions_responses) // 2 - 2
            middle_end = middle_start + 5
            selected_actions = (
                actions_responses[:5]
                + [{"action": "... (middle actions omitted) ...", "response": ""}]
                + actions_responses[middle_start:middle_end]
                + [{"action": "... (more actions omitted) ...", "response": ""}]
                + actions_responses[-5:]
            )

        action_sequence = ""
        for i, turn in enumerate(selected_actions, 1):
            if (
                turn["action"] == "... (middle actions omitted) ..."
                or turn["action"] == "... (more actions omitted) ..."
            ):
                action_sequence += f"\n{turn['action']}\n"
            else:
                action_sequence += f"\nTurn {i}:\nAction: {turn['action']}\nResult: {turn['response']}\n"

        prompt = f"""Analyze this Zork gameplay episode and provide strategic insights:

**Episode {episode_num} Summary:**
- Final Score: {episode_data["final_score"]}/{episode_data["max_score"]}
- Total Turns: {episode_data["turn_count"]}
- Outcome: {episode_data["outcome"]}

**Action Sequence:**{action_sequence}

**Analysis Instructions:**
Please analyze this episode and identify:

1. **What Worked Well:** Actions, strategies, or decisions that led to progress, points, or successful exploration
2. **What Failed:** Actions that were unsuccessful, led to danger, or wasted turns  
3. **Key Items Found:** Important items discovered and where they were located
4. **Important Locations:** Significant rooms or areas discovered
5. **Dangerous Situations:** Any encounters with death, combat, or hazardous areas
6. **Strategic Lessons:** Key takeaways that would help in future episodes

Focus on actionable insights that would help improve future gameplay. Be specific about item locations, successful action sequences, and things to avoid.

Provide your analysis in clear sections using the headers above."""

        return prompt

    def _generate_overall_guide(self, episode_analyses: List[str]) -> str:
        """
        Generate overall strategic guide from individual episode analyses.

        Args:
            episode_analyses: List of LLM analyses from individual episodes

        Returns:
            Overall strategic guide
        """
        # Combine all episode analyses
        combined_analyses = "\n\n---\n\n".join(
            [
                f"**Episode Analysis {i + 1}:**\n{analysis}"
                for i, analysis in enumerate(episode_analyses)
            ]
        )

        prompt = f"""Based on the following analyses of multiple Zork gameplay episodes, create a comprehensive strategic guide for playing Zork effectively:

{combined_analyses}

**Instructions:**
Create a strategic guide with the following sections:

# ZORK STRATEGIC GUIDE
*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## PRIORITY OBJECTIVES
List the most important goals and items to pursue early in the game.

## NAVIGATION & EXPLORATION
Provide guidance on movement, key locations to visit, and exploration strategies.

## ESSENTIAL ITEMS
List crucial items, where to find them, and how to obtain them.

## DANGERS TO AVOID
Highlight dangerous situations, areas, or actions that commonly lead to death.

## SUCCESSFUL STRATEGIES
Document proven tactics and action sequences that work well.

## COMMON MISTAKES
List actions and approaches that typically fail or waste time.

Focus on actionable advice that will directly improve gameplay performance. Be specific about locations, items, and action sequences. Prioritize information that appears consistently across multiple episodes.
**IMPORTANT:** This guide is for an LLM playing Zork, not a human player. Use clear, concise language and avoid unnecessary details."""

        try:
            print("  🔗 Making API call for overall guide generation...")
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

            guide = response.choices[0].message.content.strip()
            print(f"  ✅ Overall guide generated ({len(guide)} chars)")
            return guide

        except Exception as e:
            print(f"  ⚠️ Failed to generate overall guide: {e}")
            return self._generate_empty_guide(f"Overall guide generation failed: {e}")

    def _build_map_from_episodes(self, episodes: List[List[Dict]]) -> Optional[str]:
        """
        Build consensus map using LLM analysis of individual episode maps.

        Args:
            episodes: List of episodes to analyze for movement patterns

        Returns:
            ASCII consensus map string or None if map building failed
        """
        try:
            print("  🗺️ Building individual episode maps...")
            episode_maps = self._build_individual_episode_maps(episodes)

            if not episode_maps:
                print("  ⚠️ No valid episode maps generated")
                return None

            print(
                f"  🤖 Building LLM consensus map from {len(episode_maps)} episode maps..."
            )
            consensus_map = self._build_consensus_map_with_llm(episode_maps)

            print("  ✅ LLM consensus map generated")
            return consensus_map

        except Exception as e:
            print(f"  ⚠️ Failed to build consensus map: {e}")
            return None

    def _build_individual_episode_maps(self, episodes: List[List[Dict]]) -> List[str]:
        """
        Build ASCII map for each episode separately.

        Args:
            episodes: List of episodes to process

        Returns:
            List of ASCII map strings, one per episode
        """
        episode_maps = []

        for episode_idx, episode_logs in enumerate(episodes):
            try:
                print(
                    f"    Building map for episode {episode_idx + 1}/{len(episodes)}..."
                )

                # Create fresh MapGraph for this episode only
                game_map = MapGraph()
                movement_analyzer = MovementAnalyzer()

                # Track location state through the episode
                current_location = None
                previous_location = None
                turn_number = 0

                for log_entry in episode_logs:
                    event_type = log_entry.get("event_type", "")

                    # Update turn counter
                    if event_type == "turn_start":
                        turn_number = log_entry.get("turn", turn_number + 1)

                    # Track location changes from extracted info
                    elif event_type == "extracted_info":
                        extracted_info = log_entry.get("extracted_info", {})
                        new_location = extracted_info.get("current_location_name")
                        if new_location:
                            previous_location = current_location
                            current_location = new_location

                            # Add room to map
                            game_map.add_room(current_location)

                    # Analyze movement when we have action-response pairs
                    elif event_type == "zork_response":
                        action = log_entry.get("action", "")
                        response = log_entry.get("zork_response", "")

                        if action and current_location and previous_location:
                            # Create movement context
                            context = create_movement_context(
                                current_location=current_location,
                                previous_location=previous_location,
                                action=action,
                                game_response=response,
                                turn_number=turn_number,
                            )

                            # Analyze movement
                            result = movement_analyzer.analyze_movement(context)

                            # Add connection to map if movement was successful
                            if (
                                result.connection_created
                                and result.from_location
                                and result.to_location
                            ):
                                game_map.add_connection(
                                    from_room_name=result.from_location,
                                    exit_taken=result.action,
                                    to_room_name=result.to_location,
                                )

                # Generate ASCII map for this episode
                connection_count = sum(
                    len(room_connections)
                    for room_connections in game_map.connections.values()
                )
                if len(game_map.rooms) > 0:  # Only include maps with actual rooms
                    ascii_map = game_map.render_ascii()
                    episode_maps.append(ascii_map)
                    print(
                        f"      Episode {episode_idx + 1}: {len(game_map.rooms)} rooms, {connection_count} connections"
                    )
                else:
                    print(f"      Episode {episode_idx + 1}: No rooms found, skipping")

            except Exception as e:
                print(f"      Episode {episode_idx + 1}: Failed to build map - {e}")
                continue

        return episode_maps

    def _build_consensus_map_with_llm(self, episode_maps: List[str]) -> str:
        """
        Use LLM to build consensus map from individual episode maps.

        Args:
            episode_maps: List of ASCII map strings from individual episodes

        Returns:
            Consensus ASCII map string
        """
        # Format episode maps for the prompt
        formatted_maps = self._format_episode_maps_for_prompt(episode_maps)

        prompt = f"""You are analyzing {len(episode_maps)} different maps of the same game world, each built from a separate gameplay session. Your task is to create a single, accurate consensus map that includes only the connections that appear consistently across multiple episodes.

INDIVIDUAL EPISODE MAPS:
{formatted_maps}

INSTRUCTIONS:
1. Identify rooms that appear in multiple episodes with the same or very similar names
2. Identify connections (room A -> room B via direction X) that appear consistently
3. DISCARD connections that only appear in one episode or seem erroneous
4. DISCARD rooms that have inconsistent connections across episodes
5. Normalize room names to the most common/clear version
6. Create a final ASCII map using the same format as the input maps

CRITERIA FOR INCLUSION:
- Rooms: Must appear in at least 2 episodes OR have multiple consistent connections
- Connections: Must appear in at least 2 episodes with the same direction
- Names: Use the most frequently occurring or clearest room name variant

OUTPUT FORMAT:
Return only the final ASCII map in the exact same format as the input maps, starting with "--- ASCII Map State ---" and ending with "--- End of Map State ---".
"""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing and merging spatial/geographic data. You identify patterns and inconsistencies to create accurate composite maps.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000,
            )

            consensus_map = response.choices[0].message.content.strip()
            return consensus_map

        except Exception as e:
            print(f"  ⚠️ Failed to generate LLM consensus map: {e}")
            # Fallback: return the first episode map if available
            if episode_maps:
                print("  📋 Using first episode map as fallback")
                return episode_maps[0]
            return "-- No consensus map available --"

    def _format_episode_maps_for_prompt(self, episode_maps: List[str]) -> str:
        """
        Format episode maps for inclusion in LLM prompt.

        Args:
            episode_maps: List of ASCII map strings

        Returns:
            Formatted string with numbered episode maps
        """
        formatted_parts = []

        for i, episode_map in enumerate(episode_maps, 1):
            formatted_parts.append(f"=== EPISODE {i} MAP ===")
            formatted_parts.append(episode_map)
            formatted_parts.append("")  # Empty line separator

        return "\n".join(formatted_parts)

    def _merge_with_existing_knowledge(self, new_guide: str) -> Optional[str]:
        """
        Merge new strategic guide with existing knowledgebase.md.

        Args:
            new_guide: Newly generated strategic guide

        Returns:
            Merged strategic guide, or None if merging failed
        """
        existing_guide_path = Path(self.output_file)

        if not existing_guide_path.exists():
            print("    📝 No existing knowledgebase found, using new guide")
            return new_guide

        try:
            # Read existing guide
            with open(existing_guide_path, "r", encoding="utf-8") as f:
                existing_guide = f.read()

            print(f"    📖 Loaded existing guide ({len(existing_guide)} chars)")
            print(f"    🆕 New guide ({len(new_guide)} chars)")

            # Create merging prompt
            prompt = self._create_knowledge_merging_prompt(existing_guide, new_guide)

            print("    🔗 Making API call for knowledge merging...")
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at merging and consolidating strategic knowledge. You identify reinforced patterns, add new insights, and remove outdated information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,  # Lower temperature for consistency
                max_tokens=4000,
            )

            merged_guide = response.choices[0].message.content.strip()
            print(f"    ✅ Knowledge merged successfully ({len(merged_guide)} chars)")
            return merged_guide

        except Exception as e:
            print(f"    ⚠️ Failed to merge knowledge: {e}")
            return None

    def _create_knowledge_merging_prompt(
        self, existing_guide: str, new_guide: str
    ) -> str:
        """Create prompt for merging existing and new knowledge."""

        prompt = f"""You are merging two strategic guides for playing Zork. The EXISTING guide contains accumulated knowledge from previous episodes, while the NEW guide contains insights from the most recent {self.sliding_window_size} episodes.

Your task is to create a consolidated guide that:
1. **REINFORCES** strategies/knowledge that appear in both guides
2. **ADDS** new insights from the recent episodes
3. **UPDATES** outdated information if recent episodes show better approaches
4. **MAINTAINS** valuable knowledge from the existing guide that doesn't conflict with new findings

EXISTING STRATEGIC GUIDE:
{existing_guide}

NEW STRATEGIC GUIDE (from recent episodes):
{new_guide}

MERGING INSTRUCTIONS:
- Keep the same section structure as the guides
- When knowledge appears in both guides, emphasize it as "proven/reinforced"
- Add new discoveries from recent episodes
- If recent episodes contradict old knowledge, update with the newer information
- Preserve the timestamp and indicate this is a "merged/updated" version
- Keep the guide practical and actionable
- Do NOT include the map section - that will be added separately

Return the complete merged strategic guide in the same markdown format."""

        return prompt

    def _generate_empty_guide(self, reason: str) -> str:
        """Generate an empty guide when no knowledge is available."""
        return f"""# ZORK STRATEGIC GUIDE
*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Status
{reason}

To build a strategic guide, play more episodes and ensure the LLM analysis system is properly configured.

*This guide contains no pre-existing knowledge - all strategy must be learned through gameplay.*
"""

    def save_strategy_guide(self) -> str:
        """
        Generate and save the strategy guide.

        Returns:
            Path to the saved strategy guide file
        """
        strategy_guide = self.generate_strategy_guide()

        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(strategy_guide)

        print(f"✅ LLM-based strategy guide saved: {self.output_file}")
        return self.output_file


def create_integrated_knowledge_base(log_file: str = "zork_episode_log.jsonl") -> str:
    """
    Create a strategy guide using LLM analysis of episode logs.

    Args:
        log_file: Path to the episode log file

    Returns:
        Path to the strategy guide file
    """
    system = LLMStrategyGenerator()
    system.log_file = log_file
    return system.save_strategy_guide()


if __name__ == "__main__":
    # Generate strategy guide
    output_file = create_integrated_knowledge_base()
    print(f"🎯 LLM-based strategy guide generated: {output_file}")
