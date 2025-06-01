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

# Import shared utilities
from shared_utils import estimate_tokens



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
            base_url=config.llm.get_base_url_for_model('analysis'),
            api_key=get_client_api_key(),
        )

        # Models for different tasks
        self.analysis_model = config.llm.analysis_model
        self.info_ext_model = config.llm.info_ext_model

        # Load sampling parameters from configuration
        self.analysis_sampling = config.analysis_sampling
        self.extractor_sampling = config.extractor_sampling

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

        # Method 2 Optimization: Always use comprehensive analysis since we have full episode context
        # The strategy determination system was designed for incremental windows with limited context
        print(f"  🎯 Using comprehensive analysis (Method 2: full episode context)")

        # Step 2: Perform comprehensive analysis with full episode context
        new_insights = self._analyze_full_insights(turn_data)
        if not new_insights:
            print("  ⚠️ Comprehensive analysis failed to generate insights")
            return False

        # Step 3: Intelligent knowledge merging with comprehensive strategy
        success = self._intelligent_knowledge_merge(new_insights, "FULL_UPDATE")
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
        # prompt = r"\no_think " + prompt
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

**IMPORTANT - COORDINATE WITH DISCOVERED OBJECTIVES SYSTEM**: 
The agent has a separate real-time objective tracking system that maintains current goals every 20 turns. Your knowledge base should COMPLEMENT this system by focusing on LONG-TERM strategic insights rather than current objectives.

Provide insights in these categories focused on strategic patterns and game world knowledge:

1. **Game World Mechanics**: What specific game rules, item behaviors, or location properties were discovered?
2. **Strategic Patterns**: What types of actions consistently lead to progress vs. setbacks across different situations?
3. **Environmental Knowledge**: How do different locations behave? What objects are consistently significant?
4. **Danger Recognition**: What specific threats, traps, or failure patterns should be avoided based on experience?
5. **Efficiency Insights**: What meta-strategies help approach different types of situations more effectively?
6. **Problem-Solving Patterns**: What general approaches work well for different categories of challenges?
7. **Learning from Experience**: What insights about the game world emerged from this gameplay session?

**AVOID (Handled by Objectives System)**:
- Specific current objectives or immediate tactical goals
- Real-time action prioritization advice
- "What should I do next" guidance
- Current situation analysis

Focus on actionable insights that help the agent become better at recognizing opportunities, avoiding dangers, and understanding the game world. Be specific about locations, items, commands, and game mechanics discovered through actual gameplay experience."""

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

        # Load persistent wisdom from previous episodes for context
        persistent_wisdom = ""
        try:
            from config import get_config
            config = get_config()
            persistent_wisdom_file = config.orchestrator.persistent_wisdom_file
            
            with open(persistent_wisdom_file, "r", encoding="utf-8") as f:
                persistent_wisdom = f.read().strip()
                
            if persistent_wisdom:
                persistent_wisdom = f"\n\n**PERSISTENT WISDOM FROM PREVIOUS EPISODES:**\n{persistent_wisdom}\n"
        except FileNotFoundError:
            # No persistent wisdom file yet - this is fine for early episodes
            persistent_wisdom = ""
        except Exception as e:
            print(f"  ⚠️ Could not load persistent wisdom for analysis: {e}")
            persistent_wisdom = ""

        prompt = f"""Analyze this Zork gameplay data and provide comprehensive strategic insights.

TURN RANGE: {turn_data["start_turn"]}-{turn_data["end_turn"]}
SCORE CHANGES: {turn_data["score_changes"]}
LOCATION CHANGES: {turn_data["location_changes"]}
DEATH EVENTS: {len(turn_data.get("death_events", []))} death(s) occurred

ACTION SEQUENCE:
{actions_text}{death_analysis}{persistent_wisdom}

**IMPORTANT - COORDINATE WITH DISCOVERED OBJECTIVES SYSTEM**: 
The agent has a separate real-time objective tracking system that maintains current goals every 20 turns. Your knowledge base should COMPLEMENT this system by focusing on LONG-TERM strategic insights rather than current objectives.

**LEVERAGE PERSISTENT WISDOM**: 
Use the persistent wisdom from previous episodes to inform your analysis. Look for patterns that confirm, contradict, or extend the existing cross-episode knowledge.

Provide insights in these categories focused on strategic patterns and game world knowledge:

1. **Game World Mechanics**: What specific game rules, item behaviors, or location properties were discovered?
2. **Strategic Patterns**: What types of actions consistently lead to progress vs. setbacks across different situations?
3. **Environmental Knowledge**: How do different locations behave? What objects are consistently significant?
4. **Danger Recognition**: What specific threats, traps, or failure patterns should be avoided based on experience?
5. **Efficiency Insights**: What meta-strategies help approach different types of situations more effectively?
6. **Problem-Solving Patterns**: What general approaches work well for different categories of challenges?
7. **Learning from Experience**: What insights about the game world emerged from this gameplay session?
8. **Cross-Episode Validation**: How do these discoveries relate to patterns from previous episodes?

**AVOID (Handled by Objectives System)**:
- Specific current objectives or immediate tactical goals
- Real-time action prioritization advice
- "What should I do next" guidance
- Current situation analysis

Focus on actionable insights that help the agent become better at recognizing opportunities, avoiding dangers, and understanding the game world. Be specific about locations, items, commands, and game mechanics discovered through actual gameplay experience."""

        # Incase using Qwen qwen3-30b-a3b
        # prompt = r"\no_think " + prompt

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

        # Check if condensation is needed based on size threshold
        # Remove map section for size checking since it's handled separately
        knowledge_without_map = self._trim_map_section(merged_knowledge)
        
        if (self.enable_condensation and 
            len(knowledge_without_map) > self.condensation_threshold):
            print(f"  📏 Knowledge base size ({len(knowledge_without_map)} chars) exceeds threshold ({self.condensation_threshold}), triggering condensation...")
            
            # Apply condensation to the knowledge content (without map)
            condensed_knowledge = self._condense_knowledge_base(knowledge_without_map)
            
            if condensed_knowledge and condensed_knowledge != knowledge_without_map:
                # Condensation was successful, restore map section
                merged_knowledge = self._preserve_map_section(existing_knowledge, condensed_knowledge)
                print(f"  ✨ Condensation complete: {len(knowledge_without_map)} -> {len(condensed_knowledge)} chars")
            else:
                print(f"  ⚠️ Condensation failed or unnecessary, keeping original content")
        elif not self.enable_condensation and len(knowledge_without_map) > self.condensation_threshold:
            print(f"  ℹ️ Knowledge base size ({len(knowledge_without_map)} chars) exceeds threshold but condensation is disabled")

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

**FOCUS ON STRATEGIC DISCOVERY FRAMEWORKS**: The agent already has comprehensive movement and loop detection instructions. This knowledge base should focus on strategic frameworks to help the agent discover and maintain objectives through gameplay:

**PRIMARY STRATEGIC FRAMEWORKS**:
- **OBJECTIVE DISCOVERY**: How to recognize meaningful goals through gameplay patterns and responses
- **PROGRESS RECOGNITION**: How to identify when actions lead to advancement vs. unproductive exploration  
- **STRATEGIC PRIORITIZATION**: How to choose between multiple possible actions based on demonstrated value
- **GOAL MAINTENANCE**: How to stay focused on discovered objectives rather than getting distracted
- **LEARNING PATTERNS**: How to build strategic knowledge from gameplay experience

**STRATEGIC CONTENT AREAS TO EMPHASIZE**:
- **Discovery-Driven Navigation**: Movement strategies that maximize meaningful discoveries
- **Value Recognition**: How to identify important elements through game responses and patterns
- **Strategic Assessment**: Methods for evaluating the importance of discoveries and obstacles
- **Objective Development**: How to evolve from exploration to focused goal pursuit
- **Progress Measurement**: How to recognize meaningful advancement through gameplay feedback
- **Efficiency Patterns**: Decision-making frameworks that promote goal-directed behavior

**AVOID BASIC NAVIGATION CONTENT**: Do not include general movement instructions, loop detection patterns, or basic directional commands - these are handled elsewhere.

INSIGHTS TO ANALYZE:
{insights}

Create a strategy guide that prioritizes strategic discovery frameworks, objective development through gameplay, and pattern recognition for meaningful progress. Focus on strategic insights that help the agent develop its own goals through play rather than pursue predetermined objectives. Emphasize actionable guidance for "How can I recognize and develop meaningful objectives through gameplay?" rather than "What specific things should I do in this game?"""""
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

    def _condense_knowledge_base(self, verbose_knowledge: str) -> Optional[str]:
        """
        Use the info_ext_model to condense a knowledge base into a more concise format.
        
        This step focuses purely on reformatting and removing redundancy without
        adding new strategies or losing critical information.
        
        Args:
            verbose_knowledge: The full knowledge base content (without map section)
            
        Returns:
            Condensed knowledge base or None if condensation failed
        """
        
        if not verbose_knowledge or len(verbose_knowledge) < 1000:
            # Don't condense if content is already short
            return verbose_knowledge
            
        prompt = f"""You are tasked with condensing this Zork strategy guide into a more concise format while preserving ALL critical information.

**CRITICAL REQUIREMENTS**:
1. **NO NEW STRATEGIES**: Only reformat existing content - never invent or add new strategic advice
2. **PRESERVE ALL KEY INFORMATION**: Every important insight, danger warning, item detail, and strategic pattern must be retained
3. **REMOVE REDUNDANCY**: Eliminate repetitive statements and merge similar advice
4. **MAINTAIN STRUCTURE**: Keep the logical organization but make it more compact
5. **AI-FOCUSED LANGUAGE**: Use direct, actionable instructions for an AI language model
6. **CONSOLIDATE EXAMPLES**: Merge similar examples or scenarios into representative cases

**CURRENT KNOWLEDGE BASE TO CONDENSE**:
{verbose_knowledge}

**CONDENSATION GUIDELINES**:
- Merge repetitive advice into single, comprehensive statements
- Combine similar examples or scenarios into representative cases  
- Use bullet points and concise formatting for better readability
- Eliminate verbose explanations while keeping essential details
- Maintain all specific game elements (locations, items, commands, dangers)
- Preserve the strategic frameworks and decision-making patterns
- Keep all unique insights and specialized knowledge

**OUTPUT FORMAT**: Provide a condensed version that is 50-70% of the original length while maintaining 100% of the strategic value.

Focus on creating a guide that is information-dense but highly readable for an AI agent during gameplay."""

        try:
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert technical writer specializing in condensing strategic guides for AI systems. Your goal is to maximize information density while preserving completeness and accuracy. Never add new information - only reorganize and consolidate existing content."
                },
                {"role": "user", "content": prompt}
            ]
            
            # Log the condensation prompt if enabled
            self._log_prompt_to_file(messages, "knowledge_condensation")
            
            response = self.client.chat.completions.create(
                model=self.info_ext_model,
                messages=messages,
                temperature=self.extractor_sampling.temperature,
                top_p=getattr(self.extractor_sampling, 'top_p', None),
                top_k=getattr(self.extractor_sampling, 'top_k', None), 
                min_p=getattr(self.extractor_sampling, 'min_p', None),
                max_tokens=self.analysis_sampling.max_tokens or 5000,
            )
            
            condensed_content = response.content.strip()
            
            # Validate that condensation was successful and actually shorter
            if condensed_content and len(condensed_content) < len(verbose_knowledge):
                # Provide both character and token estimates for better feedback
                original_tokens = estimate_tokens(verbose_knowledge)
                condensed_tokens = estimate_tokens(condensed_content)
                
                print(f"  📝 Knowledge condensed: {len(verbose_knowledge)} -> {len(condensed_content)} characters ({len(condensed_content)/len(verbose_knowledge)*100:.1f}%)")
                print(f"      Token estimate: {original_tokens} -> {condensed_tokens} tokens ({condensed_tokens/original_tokens*100:.1f}%)")
                return condensed_content
            else:
                print(f"  ⚠️ Condensation failed or didn't reduce size - keeping original")
                return verbose_knowledge
                
        except Exception as e:
            print(f"  ⚠️ Knowledge condensation failed: {e}")
            return verbose_knowledge  # Return original on failure

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

    def update_knowledge_section(self, section_id: str, content: str, quality_score: float = None) -> bool:
        """
        Update a specific section of the knowledge base without affecting other sections.
        
        This enables granular updates similar to the Pokemon agent's sectioned approach,
        while maintaining ZorkGPT's quality assessment principles.
        
        Args:
            section_id: The section to update (e.g., "items", "locations", "dangers")
            content: The new content for this section
            quality_score: Optional quality score for immediate updates
            
        Returns:
            True if the section was updated, False otherwise
        """
        if not os.path.exists(self.output_file):
            # If no knowledge base exists, create with this section
            self._create_sectioned_knowledge_base(section_id, content)
            return True
            
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                existing_content = f.read()
                
            # Parse existing sections
            sections = self._parse_knowledge_sections(existing_content)
            
            # Update the specific section
            sections[section_id] = content
            
            # Reassemble knowledge base
            updated_knowledge = self._reassemble_knowledge_sections(sections)
            
            # Preserve map section if it exists
            if "## CURRENT WORLD MAP" in existing_content:
                updated_knowledge = self._preserve_map_section(existing_content, updated_knowledge)
            
            # Write updated knowledge base
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(updated_knowledge)
                
            print(f"  ✅ Updated knowledge section: {section_id}")
            return True
            
        except Exception as e:
            print(f"  ⚠️ Failed to update knowledge section {section_id}: {e}")
            return False

    def _parse_knowledge_sections(self, content: str) -> Dict[str, str]:
        """Parse existing knowledge base into sections."""
        sections = {}
        
        # Define section patterns
        section_patterns = {
            "strategies": r"## 1\. \*\*Key Successful Strategies\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "mistakes": r"## 2\. \*\*Critical Mistakes\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "navigation": r"## 3\. \*\*Navigation Insights\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "items": r"## 4\. \*\*Item Management\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "combat": r"## 5\. \*\*Combat/Danger Handling\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "death_prevention": r"## 6\. \*\*Death Prevention\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
            "learning": r"## 7\. \*\*Learning Opportunities\*\*(.*?)(?=## \d+\.|\n## CURRENT WORLD MAP|$)",
        }
        
        for section_id, pattern in section_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[section_id] = match.group(1).strip()
                
        return sections

    def _reassemble_knowledge_sections(self, sections: Dict[str, str]) -> str:
        """Reassemble sections into a complete knowledge base."""
        header = """# **Zork Game World Knowledge Base (Merged and Enhanced)**

This knowledge base contains discovered information about the Zork game world, including specific items, puzzles, dangers, and strategic insights learned through gameplay. The guide is structured to reflect algorithmic decision-making patterns, precise command syntax, and computational approaches for optimal performance.

---

"""
        
        section_templates = {
            "strategies": "## 1. **Key Successful Strategies**\n\n{content}\n\n---\n\n",
            "mistakes": "## 2. **Critical Mistakes**\n\n{content}\n\n---\n\n",
            "navigation": "## 3. **Navigation Insights**\n\n{content}\n\n---\n\n",
            "items": "## 4. **Item Management**\n\n{content}\n\n---\n\n",
            "combat": "## 5. **Combat/Danger Handling**\n\n{content}\n\n---\n\n",
            "death_prevention": "## 6. **Death Prevention**\n\n{content}\n\n---\n\n",
            "learning": "## 7. **Learning Opportunities**\n\n{content}\n\n---\n\n",
        }
        
        assembled = header
        for section_id, template in section_templates.items():
            if section_id in sections:
                assembled += template.format(content=sections[section_id])
                
        return assembled

    def _create_sectioned_knowledge_base(self, initial_section: str, content: str) -> None:
        """Create a new knowledge base with sections, starting with the given section."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(f"# Zork Strategy Guide\n\n## {initial_section}\n{content}\n")

    def synthesize_inter_episode_wisdom(self, episode_data: Dict) -> bool:
        """
        Synthesize persistent wisdom from episode completion that should carry forward
        to future episodes. Focuses on deaths, major discoveries, and cross-episode patterns.
        
        Args:
            episode_data: Dictionary containing episode summary information
            
        Returns:
            True if synthesis was performed and wisdom was updated, False if skipped
        """
        from config import get_config
        config = get_config()
        
        persistent_wisdom_file = config.orchestrator.persistent_wisdom_file
        
        print(f"🔄 Synthesizing inter-episode wisdom from episode {episode_data['episode_id']}...")
        
        # Extract key episode data for synthesis
        episode_id = episode_data['episode_id']
        turn_count = episode_data['turn_count']
        final_score = episode_data['final_score']
        death_count = episode_data['death_count']
        episode_ended_in_death = episode_data['episode_ended_in_death']
        avg_critic_score = episode_data['avg_critic_score']
        
        # Always synthesize if episode ended in death (critical learning event)
        # or if significant progress was made (score > 50 or many turns)
        should_synthesize = (
            episode_ended_in_death or 
            final_score >= 50 or 
            turn_count >= 100 or
            avg_critic_score >= 0.3
        )
        
        if not should_synthesize:
            print(f"  ⚠️ Episode not significant enough for wisdom synthesis")
            print(f"     - Death: {episode_ended_in_death}, Score: {final_score}, Turns: {turn_count}, Avg Critic: {avg_critic_score:.2f}")
            return False
        
        # Extract turn-by-turn data for death analysis and major discoveries
        turn_data = self._extract_turn_window_data(episode_id, 1, turn_count)
        if not turn_data:
            print(f"  ⚠️ Could not extract turn data for wisdom synthesis")
            return False
        
        # Load existing persistent wisdom
        existing_wisdom = ""
        try:
            with open(persistent_wisdom_file, "r", encoding="utf-8") as f:
                existing_wisdom = f.read()
        except FileNotFoundError:
            # No existing wisdom file - this is fine for first episode
            existing_wisdom = ""
        except Exception as e:
            print(f"  ⚠️ Could not load existing wisdom: {e}")
        
        # Prepare death event analysis if applicable
        death_analysis = ""
        if episode_ended_in_death or turn_data.get("death_events"):
            death_analysis = "\n\nDEATH EVENT ANALYSIS:\n"
            for event in turn_data.get("death_events", []):
                death_analysis += f"Episode {episode_id}, Turn {event['turn']}: {event['reason']}\n"
                if event.get('death_context'):
                    death_analysis += f"- Context: {event['death_context']}\n"
                if event.get('death_location'):
                    death_analysis += f"- Location: {event['death_location']}\n"
                if event.get('action_taken'):
                    death_analysis += f"- Fatal action: {event['action_taken']}\n"
                death_analysis += "\n"
        
        # Create synthesis prompt
        prompt = f"""Analyze this completed Zork episode and update the persistent wisdom base with key learnings that should carry forward to future episodes.

**CURRENT EPISODE SUMMARY:**
- Episode ID: {episode_id}
- Total turns: {turn_count}
- Final score: {final_score}
- Deaths this episode: {death_count}
- Episode ended in death: {episode_ended_in_death}
- Average critic score: {avg_critic_score:.2f}
- Discovered objectives: {len(episode_data.get('discovered_objectives', []))}
- Completed objectives: {len(episode_data.get('completed_objectives', []))}

**EPISODE ACTIONS SUMMARY:**
{turn_data['actions_and_responses'][:10] if turn_data.get('actions_and_responses') else 'No action data available'}

{death_analysis}

**EXISTING PERSISTENT WISDOM:**
{existing_wisdom if existing_wisdom else "No previous wisdom recorded."}

**SYNTHESIS TASK:**

Extract and synthesize the most critical learnings from this episode that should persist across future episodes. Focus on:

1. **Death Pattern Analysis**: If deaths occurred, what specific patterns, locations, or actions consistently lead to death? What environmental cues signal danger?

2. **Critical Environmental Knowledge**: What persistent facts about the game world were discovered? (Dangerous locations, item behaviors, puzzle mechanics)

3. **Strategic Patterns**: What meta-strategies or approaches proved consistently effective or ineffective across different situations?

4. **Discovery Insights**: What major discoveries about game mechanics, hidden areas, or puzzle solutions should be remembered?

5. **Cross-Episode Learning**: How does this episode's experience relate to patterns from previous episodes?

**REQUIREMENTS:**
- Focus on persistent, reusable knowledge rather than episode-specific details
- Emphasize death avoidance and danger recognition patterns
- Maintain existing wisdom while adding new insights
- Remove outdated or contradicted information
- Keep wisdom concise but actionable for an AI agent

**OUTPUT FORMAT:**
Provide the updated persistent wisdom as a well-organized markdown document. If no significant new wisdom emerged, return "NO_SIGNIFICANT_WISDOM" instead."""

        try:
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting persistent strategic wisdom from interactive fiction gameplay that can help AI agents improve across multiple game sessions. Focus on actionable patterns, danger recognition, and cross-episode learning."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.analysis_sampling.temperature,
                top_p=self.analysis_sampling.top_p,
                top_k=self.analysis_sampling.top_k,
                min_p=self.analysis_sampling.min_p,
                max_tokens=self.analysis_sampling.max_tokens or 2000,
            )

            wisdom_response = response.content.strip()
            
            if wisdom_response == "NO_SIGNIFICANT_WISDOM":
                print(f"  ⚠️ No significant wisdom to synthesize from this episode")
                return False
            
            # Save the updated persistent wisdom
            try:
                with open(persistent_wisdom_file, "w", encoding="utf-8") as f:
                    f.write(wisdom_response)
                
                print(f"  ✅ Persistent wisdom updated and saved to {persistent_wisdom_file}")
                print(f"     - Synthesized from episode with {turn_count} turns, score {final_score}")
                
                return True
                
            except Exception as e:
                print(f"  ⚠️ Failed to save persistent wisdom: {e}")
                return False
            
        except Exception as e:
            print(f"  ⚠️ Inter-episode wisdom synthesis failed: {e}")
            return False
