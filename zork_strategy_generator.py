"""
AdaptiveKnowledgeManager module for turn-based knowledge management.

This module provides LLM-first adaptive knowledge management for ZorkGPT,
using turn-based sliding windows instead of episode-based analysis.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from map_graph import MapGraph
from movement_analyzer import MovementAnalyzer, create_movement_context
from llm_client import LLMClientWrapper
from config import get_config, get_client_api_key
import re
from pathlib import Path



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

        # Initialize LLM client
        config = get_config()
        self.client = LLMClientWrapper(
            base_url=config.llm.client_base_url,
            api_key=get_client_api_key(),
        )

        # Model for analysis
        self.analysis_model = config.llm.analysis_model

        # Load analysis sampling parameters from configuration
        self.analysis_sampling = config.analysis_sampling

        # Turn-based configuration
        self.turn_window_size = config.gameplay.turn_window_size
        self.min_quality_threshold = config.gameplay.min_knowledge_quality
        
        # Prompt logging counter for temporary evaluation
        self.prompt_counter = 0
        self.enable_prompt_logging = config.logging.enable_prompt_logging
        
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
            "death_events": [],  # Track death events for knowledge base
            "game_over_events": [],  # Track all game over events
        }

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                current_turn = 0
                current_score = 0
                current_location = ""
                current_inventory = []
                
                # Store death messages temporarily for proper association
                death_messages_by_turn = {}

                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # Skip entries not from this episode
                        if log_entry.get("episode_id") != episode_id:
                            continue

                        event_type = log_entry.get("event_type", "")

                        # Track turn progression - always update current_turn for this episode
                        if event_type == "turn_start":
                            current_turn = log_entry.get("turn", 0)

                        # Collect action-response pairs - but only within our turn window
                        if event_type == "final_action_selection" and (start_turn <= current_turn <= end_turn):
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
                            and (start_turn <= current_turn <= end_turn)
                        ):
                            # Update the last action with its response
                            response = log_entry.get("zork_response", "")
                            turn_data["actions_and_responses"][-1]["response"] = response
                            
                            # Check if this zork response contains death information and store it
                            if any(death_indicator in response.lower() for death_indicator in [
                                "you have died", "you are dead", "slavering fangs", "eaten by a grue",
                                "you have been killed", "****  you have died  ****", "fatal"
                            ]):
                                action = log_entry.get("action", "")
                                # Create contextual description instead of bare action
                                death_context = f"{action} from {current_location}" if current_location else action
                                death_messages_by_turn[current_turn] = {
                                    "detailed_death_message": response,
                                    "death_context": death_context,
                                    "death_location": current_location,
                                    "fatal_action": action  # Keep raw action for reference
                                }

                        # Only collect data within our turn window for other events
                        if not (start_turn <= current_turn <= end_turn):
                            continue

                        # Track death and game over events
                        if event_type in ["game_over", "game_over_final", "death_during_inventory"]:
                            death_event = {
                                "turn": current_turn,
                                "event_type": event_type,
                                "reason": log_entry.get("reason", ""),
                                "action_taken": log_entry.get("action_taken", ""),
                                "final_score": log_entry.get("final_score", current_score),
                                "death_count": log_entry.get("death_count", 0),
                            }
                            
                            # Add to both death_events and game_over_events for different analysis purposes
                            turn_data["game_over_events"].append(death_event)
                            
                            # Check if this is specifically a death (vs victory)
                            reason = log_entry.get("reason", "").lower()
                            death_indicators = ["died", "death", "eaten", "grue", "killed", "fall", "crushed"]
                            if any(indicator in reason for indicator in death_indicators):
                                turn_data["death_events"].append(death_event)

                        # Track death state extraction for context
                        elif event_type == "death_state_extracted":
                            extracted_info = log_entry.get("extracted_info", {})
                            if extracted_info and turn_data["death_events"]:
                                # Add extraction details to the most recent death event
                                turn_data["death_events"][-1]["death_location"] = extracted_info.get("current_location_name", "")
                                turn_data["death_events"][-1]["death_objects"] = extracted_info.get("visible_objects", [])
                                turn_data["death_events"][-1]["death_messages"] = extracted_info.get("important_messages", [])

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
        
        # Apply stored death messages to death events
        for turn_num, death_info in death_messages_by_turn.items():
            # Apply to death events
            for death_event in turn_data["death_events"]:
                if death_event["turn"] == turn_num:
                    death_event.update(death_info)
            
            # Apply to game over events  
            for game_over_event in turn_data["game_over_events"]:
                if game_over_event["turn"] == turn_num:
                    game_over_event.update(death_info)

        return turn_data if turn_data["actions_and_responses"] else None

    def _assess_knowledge_update_quality(
        self, turn_data: Dict, is_final_update: bool = False
    ) -> Tuple[float, str]:
        """Let LLM assess if this turn window would produce useful knowledge."""

        # Prepare summary of turn data
        num_actions = len(turn_data["actions_and_responses"])
        num_score_changes = len(turn_data["score_changes"])
        num_location_changes = len(turn_data["location_changes"])
        num_death_events = len(turn_data.get("death_events", []))
        num_game_over_events = len(turn_data.get("game_over_events", []))

        # Sample some actions for context (limit to avoid token overflow)
        sample_actions = turn_data["actions_and_responses"][:50]
        actions_summary = "\n".join(
            [
                f"Turn {action['turn']}: {action['action']} -> {action['response'][:100]}..."
                if len(action["response"]) > 100
                else f"Turn {action['turn']}: {action['action']} -> {action['response']}"
                for action in sample_actions
            ]
        )

        # Add death event summary if any occurred
        death_summary = ""
        if num_death_events > 0:
            death_summary = "\n\nDEATH EVENTS:\n" + "\n".join([
                f"Turn {event['turn']}: {event['reason']} ({'Context: ' + event.get('death_context', 'N/A') if event.get('death_context') else 'Action: ' + event.get('action_taken', 'N/A')})"
                for event in turn_data["death_events"]
            ])
        elif num_game_over_events > 0:
            death_summary = "\n\nGAME OVER EVENTS:\n" + "\n".join([
                f"Turn {event['turn']}: {event['reason']} ({'Context: ' + event.get('death_context', 'N/A') if event.get('death_context') else 'Action: ' + event.get('action_taken', 'N/A')})"
                for event in turn_data["game_over_events"]
            ])

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
- Death events: {num_death_events}
- Game over events: {num_game_over_events}

SAMPLE ACTIONS:
{actions_summary}{death_summary}

Before doing the full analysis, assess the potential value of this data:

1. How much NEW strategic value could be extracted from this data?
2. Are there repetitive patterns that would dilute useful insights?
3. Would analyzing this data improve or degrade the knowledge base?
4. What type of gameplay situation does this represent?
5. {"Are there any dangers, deaths, or new discoveries that should be documented?" if is_final_update else ""}

**IMPORTANT**: Death events are typically HIGH VALUE for learning - they teach about dangers to avoid.
New locations and score changes also indicate valuable discoveries.

Rate the potential knowledge value from 0-10 and provide a brief explanation.
{"For final updates: Focus on ANY useful insights, especially dangers or new areas." if is_final_update else "Focus on whether this data would generate actionable, non-repetitive insights."}

Format your response as:
SCORE: [0-10]
REASON: [brief explanation]"""

        # Incase using Qwen qwen3-30b-a3b
        prompt = r"\no_think " + prompt
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
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 500,
            )

            content = response.content.strip()

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
- Death events: {len(turn_data.get("death_events", []))}

Choose the most appropriate strategy based on what will generate the most valuable game world knowledge:

1. FULL_UPDATE: Comprehensive analysis focusing on items, puzzles, dangers, and strategic discoveries
2. SELECTIVE_UPDATE: Focus on specific game world aspects (items, deaths, new areas, puzzle clues)
3. CONSOLIDATION_ONLY: Reorganize existing knowledge without adding new information
4. ESCAPE_ANALYSIS: Focus on navigation and movement strategies for stuck situations

**PRIORITIZE GAME WORLD CONTENT**: Focus on strategies that will extract:
- Specific item locations, uses, and combinations
- Puzzle mechanics and solution patterns  
- Danger identification and avoidance strategies
- Room-specific secrets and interactive elements
- Combat/interaction strategies with NPCs or creatures

Consider:
- What specific game world discoveries could be documented?
- Are there deaths that reveal important danger patterns?
- Do score changes indicate successful puzzle solving or item acquisition?
- Are there new areas with unique properties to document?

Respond with just the strategy name: FULL_UPDATE, SELECTIVE_UPDATE, CONSOLIDATION_ONLY, or ESCAPE_ANALYSIS"""

        # Incase using Qwen qwen3-30b-a3b
        prompt = r"\no_think " + prompt

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at determining optimal knowledge extraction strategies for interactive fiction gameplay. Focus on strategies that will generate valuable game world knowledge about items, puzzles, dangers, and strategic discoveries rather than just navigation patterns.",
                },
                {"role": "user", "content": prompt},
            ]
            
            # Log the full prompt for evaluation
            self._log_prompt_to_file(messages, "knowledge_strategy")
            
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=messages,
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 100,
            )

            strategy = response.content.strip().upper()

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
            :50
        ]:  # Limit to avoid token overflow
            actions_text += f"Turn {action['turn']}: {action['action']} -> {action['response'][:150]}...\n"

        # Prepare death event details if any
        death_analysis = ""
        if turn_data.get("death_events"):
            death_analysis = "\n\nDEATH EVENT DETAILS:\n"
            for event in turn_data["death_events"]:
                death_analysis += f"Turn {event['turn']}: {event['reason']}\n"
                # Use contextual death information if available
                if event.get('death_context'):
                    death_analysis += f"- Dangerous action/location: {event['death_context']}\n"
                else:
                    death_analysis += f"- Action leading to death: {event.get('action_taken', 'Unknown')}\n"
                if event.get('death_location'):
                    death_analysis += f"- Location: {event['death_location']}\n"
                if event.get('death_objects'):
                    death_analysis += f"- Objects present: {', '.join(event['death_objects'])}\n"
                if event.get('death_messages'):
                    death_analysis += f"- Key messages: {', '.join(event['death_messages'])}\n"
                death_analysis += "\n"

        # Include agent instructions context if available
        agent_context = ""
        if self.agent_instructions:
            # Use actual agent instructions content instead of hardcoded summary
            agent_context = f"""
**IMPORTANT - EXISTING AGENT INSTRUCTIONS CONTEXT:**
The agent already has detailed instructions that include the following guidelines:

```
{self.agent_instructions}...
```

Focus your analysis on DISCOVERED KNOWLEDGE that goes BEYOND these existing instructions:
- Specific location details and secrets not covered in basic instructions
- Exact item locations and usage sequences discovered through play
- Specific puzzle solutions and tricks found during gameplay
- Particular dangers and how to handle them (beyond general combat advice)
- Advanced tactics discovered through actual experience
- Specific sequences that work well in practice

DO NOT repeat basic information already covered in the agent instructions above."""

        prompt = f"""Analyze this Zork gameplay data and extract the most valuable strategic insights about the game world.{agent_context}

TURN RANGE: {turn_data["start_turn"]}-{turn_data["end_turn"]}
SCORE CHANGES: {turn_data["score_changes"]}
LOCATION CHANGES: {turn_data["location_changes"]}
DEATH EVENTS: {len(turn_data.get("death_events", []))} death(s) occurred

ACTION SEQUENCE:
{actions_text}{death_analysis}

**FOCUS ON GAME WORLD CONTENT**: Extract insights that reveal specific information about:

1. **Items and Objects**: 
   - Specific items found and their exact locations
   - How items can be used or combined
   - Which items are essential vs. optional
   - Item interaction patterns and effects

2. **Puzzles and Mechanisms**:
   - Puzzle solutions or partial solutions discovered
   - Interactive elements in rooms (doors, switches, containers)
   - Sequence requirements for complex actions
   - Clues found about how to solve problems

3. **Dangers and Death Prevention**:
   - Specific threats and how they manifest
   - Warning signs that precede danger
   - Exact conditions that lead to death
   - Safe vs. dangerous actions in specific locations

4. **Location-Specific Knowledge**:
   - Unique properties of rooms or areas
   - Special interactions available in certain locations
   - Hidden or non-obvious features of places
   - Room-specific commands that work

5. **Character/Creature Interactions**:
   - NPCs encountered and how to interact with them
   - Combat or negotiation strategies
   - Behavior patterns of game entities

6. **Strategic Discoveries**:
   - Successful action sequences for achieving goals
   - Time-sensitive events or conditions
   - Resource management insights (light, health, etc.)

**PRIORITIZE SPECIFIC, ACTIONABLE DISCOVERIES**: Focus on concrete information that would help an AI agent make better decisions about:
- Which specific items to prioritize taking
- How to solve particular puzzles step-by-step
- Which locations or actions to avoid and why
- Optimal sequences for achieving objectives

Skip basic navigation advice (which is already covered) unless it reveals something specific about the game world (like secret passages or special movement requirements).

{death_analysis if turn_data.get("death_events") else ""}

Be specific about locations, items, commands, and sequences. Provide insights that go beyond general gameplay mechanics. **Aim for conciseness in these new discoveries, presenting them clearly and avoiding redundancy if similar points arise from this specific turn window's data.**"""

        # Incase using Qwen qwen3-30b-a3b
        prompt = r"\no_think " + prompt

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting specific, actionable game world knowledge from interactive fiction gameplay. Focus on concrete discoveries about items, puzzles, dangers, and strategic elements rather than general navigation or basic gameplay mechanics.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 1500,
            )

            return response.content.strip()

        except Exception as e:
            print(f"  ⚠️ Selective analysis failed: {e}")
            return None

    def _analyze_escape_strategies(self, turn_data: Dict) -> Optional[str]:
        """Analyze data to identify escape strategies from stuck situations."""

        # Focus on recent actions and any changes
        recent_actions = turn_data["actions_and_responses"][-30:]
        actions_text = "\n".join(
            [
                f"Turn {action['turn']}: {action['action']} -> {action['response'][:100]}..."
                for action in recent_actions
            ]
        )

        prompt = f"""This gameplay data appears to show a stuck or repetitive situation. Analyze it to identify escape strategies.

**CRITICAL FOCUS**: The agent already has comprehensive loop detection and navigation instructions. Focus this analysis on SPECIFIC DISCOVERIES about escaping from stuck situations in this particular gameplay:

RECENT ACTIONS:
{actions_text}

LOCATION CHANGES: {turn_data["location_changes"]}
SCORE CHANGES: {turn_data["score_changes"]}

Analyze this specific stuck situation for NEWLY DISCOVERED escape insights:

1. **Situation-Specific Patterns**: What specific stuck situation was encountered here?
   - Which location(s) caused the loop?
   - What particular objects or features were involved?
   - Were there any missed opportunities or overlooked elements?

2. **Successful Escape Actions**: What actions (if any) successfully broke out of this loop?
   - Which specific commands worked to escape?
   - Were there any discoveries about special movement commands?
   - Did examining specific objects reveal new information?

3. **Game World Insights**: What does this situation reveal about the game world?
   - Are there location-specific navigation requirements?
   - Were there hidden exits or special commands discovered?
   - Any items or interactive elements that were overlooked?

4. **Context-Specific Learnings**: What would help in similar future situations?
   - Location-specific warnings or guidance
   - Object interaction patterns unique to this area
   - Special command sequences that work in this context

Focus on CONCRETE DISCOVERIES from this specific gameplay session rather than general navigation principles. What specific insights about items, locations, commands, or game mechanics were revealed by this stuck situation?"""
    
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
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 1000,
            )

            return response.content.strip()

        except Exception as e:
            print(f"  ⚠️ Escape analysis failed: {e}")
            return None

    def _analyze_full_insights(self, turn_data: Dict) -> Optional[str]:
        """Perform comprehensive analysis of all aspects."""

        # This is similar to the original episode analysis but for turn ranges
        actions_text = ""
        for action in turn_data["actions_and_responses"][:60]:
            actions_text += f"Turn {action['turn']}: {action['action']} -> {action['response'][:150]}...\n"

        # Prepare death event analysis
        death_analysis = ""
        if turn_data.get("death_events"):
            death_analysis = "\n\nDEATH EVENT ANALYSIS:\n"
            for event in turn_data["death_events"]:
                death_analysis += f"Turn {event['turn']}: {event['reason']}\n"
                # Use contextual death information if available
                if event.get('death_context'):
                    death_analysis += f"- Dangerous action/location: {event['death_context']}\n"
                else:
                    death_analysis += f"- Fatal action: {event.get('action_taken', 'Unknown')}\n"
                death_analysis += f"- Final score: {event.get('final_score', 'Unknown')}\n"
                if event.get('death_location'):
                    death_analysis += f"- Death location: {event['death_location']}\n"
                if event.get('death_objects'):
                    death_analysis += f"- Objects at death scene: {', '.join(event['death_objects'])}\n"
                if event.get('death_messages'):
                    death_analysis += f"- Death messages: {', '.join(event['death_messages'])}\n"
                death_analysis += "\n"

        prompt = f"""Analyze this Zork gameplay data and provide comprehensive strategic insights.

TURN RANGE: {turn_data["start_turn"]}-{turn_data["end_turn"]}
SCORE CHANGES: {turn_data["score_changes"]}
LOCATION CHANGES: {turn_data["location_changes"]}
DEATH EVENTS: {len(turn_data.get("death_events", []))} death(s) occurred

ACTION SEQUENCE:
{actions_text}{death_analysis}

Provide insights in these categories:
1. **Key Successful Strategies**: What actions or patterns led to progress?
2. **Critical Mistakes**: What actions hindered progress or caused setbacks?
3. **Navigation Insights**: How effectively was the world navigated?
4. **Item Management**: Were items collected and used effectively?
5. **Combat/Danger Handling**: How well were threats managed?
6. **Death Prevention**: If deaths occurred, what specific strategies would prevent them?
7. **Learning Opportunities**: What should be done differently?

**DEATH ANALYSIS PRIORITY**: If deaths occurred, prioritize understanding:
- The exact sequence of events leading to death
- Warning signs that should have been recognized
- Alternative actions that could have been taken
- How to recognize similar dangerous situations in the future

Focus on actionable insights that would help improve future gameplay. Be specific about locations, items, and sequences when relevant. **Prioritize novel strategies and observations from this specific turn window that are unlikely to be covered in general Zork advice. Aim for conciseness and avoid repeating information that would already be known to an experienced player or present in a general strategy guide.**"""

        # Incase using Qwen qwen3-30b-a3b
        prompt = r"\no_think " + prompt

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
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 2000,
            )

            return response.content.strip()

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

        # Incase using Qwen qwen3-30b-a3b
        prompt = r"\no_think " + prompt
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
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 3000,
            )

            return response.content.strip()

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
        # Incase using Qwen qwen3-30b-a3b
        prompt = r"\no_think " + prompt
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
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 4000,
            )

            return response.content.strip()

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

**FOCUS ON DISCOVERED GAME WORLD CONTENT**: The agent already has comprehensive movement and loop detection instructions. This knowledge base should focus exclusively on discoveries about the game world itself:

- **Items and Objects**: Specific item locations, uses, combinations, and properties
- **Puzzles and Mechanisms**: Puzzle solutions, interactive elements, sequence requirements
- **Dangers and Threats**: Specific dangers, warning signs, death prevention strategies  
- **Location Properties**: Unique room features, special interactions, hidden elements
- **NPC/Creature Interactions**: Character behaviors, combat strategies, interaction patterns
- **Strategic Discoveries**: Successful action sequences, resource management, timing insights

**AVOID BASIC NAVIGATION CONTENT**: Do not include general movement instructions, loop detection patterns, or basic directional commands - these are handled elsewhere.

INSIGHTS TO ANALYZE:
{insights}

Create a focused strategy guide about discovered game world content that incorporates these insights. Emphasize specific, actionable knowledge about items, puzzles, dangers, and locations discovered through gameplay."""
        # Incase using Qwen qwen3-30b-a3b
        prompt = r"\no_think " + prompt
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
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 3000,
            )

            return response.content.strip()

        except Exception as e:
            print(f"  ⚠️ New knowledge base creation failed: {e}")
            return f"# Zork Strategy Guide\n\n{insights}"

    def _trim_map_section(self, knowledge_content: str) -> str:
        """Remove the map section from knowledge content for LLM processing."""
        if not knowledge_content or "## CURRENT WORLD MAP" not in knowledge_content:
            return knowledge_content
            
        # Remove the mermaid diagram section more precisely
        # Look for the pattern: ## CURRENT WORLD MAP followed by ```mermaid...```
        pattern = r'## CURRENT WORLD MAP\s*\n\s*```mermaid\s*\n.*?\n```'
        
        # Remove the mermaid diagram section while preserving other content
        knowledge_only = re.sub(pattern, '', knowledge_content, flags=re.DOTALL)
        
        # Clean up any extra whitespace that might be left
        knowledge_only = re.sub(r'\n\s*\n\s*\n', '\n\n', knowledge_only)
        
        return knowledge_only.strip()

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
                return f"""# Zork Game World Knowledge Base

This knowledge base contains discovered information about the Zork game world, including specific items, puzzles, dangers, and strategic insights learned through gameplay.

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
