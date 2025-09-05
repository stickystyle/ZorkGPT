"""
ContextManager for ZorkGPT orchestration.

Handles all context management responsibilities:
- Agent context preparation and assembly
- Prompt context building from various sources
- Memory and history filtering for context
- Context formatting for different agent operations
- Context data aggregation from managers
- Context overflow protection and management
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from managers.base_manager import BaseManager
from session.game_state import GameState
from session.game_configuration import GameConfiguration


class ContextManager(BaseManager):
    """
    Manages all context-related functionality for ZorkGPT.

    Responsibilities:
    - Agent context preparation and assembly
    - Context overflow detection and management
    - Memory and history filtering for relevance
    - Prompt building for different operations
    - Context data aggregation from multiple sources
    - Context formatting and optimization
    """

    def __init__(self, logger, config: GameConfiguration, game_state: GameState):
        super().__init__(logger, config, game_state, "context_manager")

    def reset_episode(self) -> None:
        """Reset context manager state for a new episode."""
        self.log_debug("Context manager reset for new episode")

    def process_turn(self) -> None:
        """Process context management for the current turn."""
        # Context processing is handled via explicit calls
        pass

    def should_process_turn(self) -> bool:
        """Check if context needs processing this turn."""
        return False  # Context management is event-driven

    def add_memory(self, extracted_info: Any) -> None:
        """Add extracted information to memory log history."""
        try:
            self.game_state.memory_log_history.append(extracted_info)

            self.log_debug(
                f"Added memory entry: turn {self.game_state.turn_count}",
                details=f"Memory entry added for turn {self.game_state.turn_count}",
            )

        except Exception as e:
            self.log_error(f"Failed to add memory: {e}")

    def add_action(self, action: str, response: str) -> None:
        """Add action and response to action history."""
        try:
            self.game_state.action_history.append((action, response))

            self.log_debug(
                f"Added action to history: {action[:50]}...",
                details=f"Action: {action}, Response length: {len(response)}",
            )

        except Exception as e:
            self.log_error(f"Failed to add action: {e}")

    def add_reasoning(self, reasoning: str, agent_action: str = "") -> None:
        """Add agent reasoning to reasoning history."""
        try:
            reasoning_entry = {
                "turn": self.game_state.turn_count,
                "reasoning": reasoning,
                "action": agent_action,
                "timestamp": datetime.now().isoformat(),
            }

            self.game_state.action_reasoning_history.append(reasoning_entry)

            self.log_debug(
                f"Added reasoning entry: turn {self.game_state.turn_count}",
                details=f"Reasoning length: {len(reasoning)}",
            )

        except Exception as e:
            self.log_error(f"Failed to add reasoning: {e}")

    def get_agent_context(
        self,
        current_state: str,
        inventory: List[str],
        location: str,
        game_map=None,
        in_combat: bool = False,
        failed_actions: List[str] = None,
        discovered_objectives: List[str] = None,
    ) -> Dict[str, Any]:
        """Assemble comprehensive context for agent action generation."""
        try:
            context = {
                "game_state": current_state,
                "current_location": location,
                "inventory": inventory,
                "in_combat": in_combat,
                "recent_actions": self.get_recent_actions(5),
                "recent_memories": self.get_recent_memories(10),
                "recent_reasoning": self.get_recent_reasoning(3),
                "failed_actions_here": failed_actions or [],
                "discovered_objectives": discovered_objectives or [],
                "action_counts": dict(self.game_state.action_counts),
            }

            # Add map context if available
            if game_map and hasattr(game_map, "get_context_for_prompt"):
                try:
                    context["map_context"] = game_map.get_context_for_prompt(location)
                except Exception as e:
                    self.log_warning(f"Failed to get map context: {e}")
                    context["map_context"] = ""

            # Add navigation suggestions
            if game_map and hasattr(game_map, "get_available_exits"):
                try:
                    context["available_exits"] = game_map.get_available_exits(location)
                except Exception as e:
                    context["available_exits"] = []

            # Add room transition context
            context["previous_room"] = self.game_state.prev_room_for_prompt_context
            context["action_to_current_room"] = (
                self.game_state.action_leading_to_current_room_for_prompt_context
            )

            self.log_debug(
                f"Assembled agent context with {len(context['recent_actions'])} actions, "
                f"{len(context['recent_memories'])} memories",
                details=f"Context keys: {list(context.keys())}",
            )

            return context

        except Exception as e:
            self.log_error(f"Failed to assemble agent context: {e}")
            return {}

    def get_critic_context(
        self,
        current_state: str,
        proposed_action: str,
        location: str,
        available_exits: List[str] = None,
        failed_actions: List[str] = None,
    ) -> Dict[str, Any]:
        """Assemble context for critic evaluation."""
        try:
            context = {
                "game_state": current_state,
                "proposed_action": proposed_action,
                "current_location": location,
                "available_exits": available_exits or [],
                "failed_actions_here": failed_actions or [],
                "recent_actions": self.get_recent_actions(3),
                "recent_outcomes": self.get_recent_action_outcomes(3),
                "inventory": self.game_state.current_inventory,
                "score": self.game_state.previous_zork_score,
            }

            self.log_debug(
                f"Assembled critic context for action: {proposed_action[:50]}...",
                details=f"Context includes {len(context['recent_actions'])} recent actions",
            )

            return context

        except Exception as e:
            self.log_error(f"Failed to assemble critic context: {e}")
            return {}

    def get_objective_analysis_context(self, current_reasoning: str = "") -> str:
        """Prepare gameplay context for objective analysis."""
        try:
            context_parts = []

            # Add recent actions and responses
            recent_actions = self.get_recent_actions(5)
            if recent_actions:
                context_parts.append("RECENT ACTIONS:")
                for action, response in recent_actions:
                    context_parts.append(f"Action: {action}")
                    context_parts.append(
                        f"Response: {response[:200]}..."
                    )  # Truncate long responses
                    context_parts.append("")

            # Add current agent reasoning
            if current_reasoning:
                context_parts.append("CURRENT AGENT REASONING:")
                context_parts.append(current_reasoning)
                context_parts.append("")

            # Add recent memory highlights
            recent_memories = self.get_recent_memories(3)
            if recent_memories:
                context_parts.append("RECENT MEMORY HIGHLIGHTS:")
                for memory in recent_memories:
                    if isinstance(memory, dict):
                        context_parts.append(
                            f"Turn {memory.get('turn', '?')}: {memory.get('summary', str(memory))}"
                        )
                context_parts.append("")

            return "\n".join(context_parts)

        except Exception as e:
            self.log_error(f"Failed to prepare objective analysis context: {e}")
            return ""

    def get_recent_actions(self, n: int = 5) -> List[Tuple[str, str]]:
        """Get last n actions with responses."""
        try:
            return (
                self.game_state.action_history[-n:]
                if self.game_state.action_history
                else []
            )
        except Exception as e:
            self.log_error(f"Failed to get recent actions: {e}")
            return []

    def get_recent_memories(self, n: int = 10) -> List[Any]:
        """Get last n memory entries."""
        try:
            return (
                self.game_state.memory_log_history[-n:]
                if self.game_state.memory_log_history
                else []
            )
        except Exception as e:
            self.log_error(f"Failed to get recent memories: {e}")
            return []

    def get_recent_reasoning(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get last n reasoning entries."""
        try:
            return (
                self.game_state.action_reasoning_history[-n:]
                if self.game_state.action_reasoning_history
                else []
            )
        except Exception as e:
            self.log_error(f"Failed to get recent reasoning: {e}")
            return []

    def get_recent_action_outcomes(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get recent action outcomes for critic context."""
        try:
            outcomes = []
            recent_actions = self.get_recent_actions(n)

            for action, response in recent_actions:
                # Analyze the outcome
                outcome = {
                    "action": action,
                    "success": self.is_successful_action(response),
                    "response_summary": response[:100],  # Truncated response
                }
                outcomes.append(outcome)

            return outcomes

        except Exception as e:
            self.log_error(f"Failed to get recent action outcomes: {e}")
            return []

    def is_successful_action(self, response: str) -> bool:
        """Determine if an action was successful based on response."""
        try:
            response_lower = response.lower()

            # Failure indicators
            failure_indicators = [
                "you can't",
                "impossible",
                "don't understand",
                "nothing happens",
                "no such",
                "not here",
                "can't see",
                "don't have",
                "already",
            ]

            if any(indicator in response_lower for indicator in failure_indicators):
                return False

            # Success indicators
            success_indicators = [
                "taken",
                "dropped",
                "opened",
                "closed",
                "moved",
                "went",
                "points",
                "score",
                "treasure",
                "got",
            ]

            if any(indicator in response_lower for indicator in success_indicators):
                return True

            # Default: assume neutral/successful if no clear failure
            return True

        except Exception as e:
            return True  # Default to success if analysis fails

    def update_location_context(
        self, from_room: str, to_room: str, action: str
    ) -> None:
        """Update location tracking for context."""
        try:
            self.game_state.prev_room_for_prompt_context = from_room
            self.game_state.action_leading_to_current_room_for_prompt_context = action

            self.log_debug(
                f"Updated location context: {from_room} --({action})--> {to_room}",
                details=f"Location context updated for navigation",
            )

        except Exception as e:
            self.log_error(f"Failed to update location context: {e}")

    def get_formatted_agent_prompt_context(self, context: Dict[str, Any]) -> str:
        """Format agent context into a readable prompt format."""
        try:
            formatted_parts = []

            # Current situation
            formatted_parts.append(
                f"CURRENT LOCATION: {context.get('current_location', 'Unknown')}"
            )
            formatted_parts.append(
                f"INVENTORY: {', '.join(context.get('inventory', [])) or 'empty'}"
            )
            formatted_parts.append(f"SCORE: {self.game_state.previous_zork_score}")

            if context.get("in_combat"):
                formatted_parts.append("STATUS: IN COMBAT")

            # Recent actions
            recent_actions = context.get("recent_actions", [])
            if recent_actions:
                formatted_parts.append("\nRECENT ACTIONS:")
                for action, response in recent_actions[-3:]:  # Last 3 actions
                    formatted_parts.append(f"  > {action}")
                    formatted_parts.append(f"    {response[:100]}...")

            # Failed actions at this location
            failed_actions = context.get("failed_actions_here", [])
            if failed_actions:
                formatted_parts.append(
                    f"\nFAILED ACTIONS HERE: {', '.join(failed_actions)}"
                )

            # Available exits
            exits = context.get("available_exits", [])
            if exits:
                formatted_parts.append(f"\nAVAILABLE EXITS: {', '.join(exits)}")

            # Objectives
            objectives = context.get("discovered_objectives", [])
            if objectives:
                formatted_parts.append("\nCURRENT OBJECTIVES:")
                for obj in objectives[:5]:  # Limit to top 5
                    formatted_parts.append(f"  - {obj}")

            return "\n".join(formatted_parts)

        except Exception as e:
            self.log_error(f"Failed to format agent prompt context: {e}")
            return str(context)  # Fallback to string representation

    def get_context_summary_for_export(self) -> Dict[str, Any]:
        """Get context summary for state export."""
        try:
            return {
                "memory_entries": len(self.game_state.memory_log_history),
                "action_history_length": len(self.game_state.action_history),
                "reasoning_history_length": len(
                    self.game_state.action_reasoning_history
                ),
                "recent_locations": [
                    self.game_state.prev_room_for_prompt_context,
                    self.game_state.current_room_name_for_map,
                ],
                "action_counts": dict(self.game_state.action_counts),
            }

        except Exception as e:
            self.log_error(f"Failed to get context summary: {e}")
            return {}

    def detect_loops_in_recent_actions(self, n: int = 10) -> bool:
        """Detect if agent is stuck in a loop based on recent actions."""
        try:
            recent_actions = self.get_recent_actions(n)
            if len(recent_actions) < 4:
                return False

            # Check for repeated action sequences
            actions_only = [action for action, _ in recent_actions]

            # Look for patterns of length 2-4
            for pattern_length in range(2, 5):
                if len(actions_only) >= pattern_length * 2:
                    # Check if the last pattern_length actions repeat
                    recent_pattern = actions_only[-pattern_length:]
                    previous_pattern = actions_only[
                        -pattern_length * 2 : -pattern_length
                    ]

                    if recent_pattern == previous_pattern:
                        self.log_warning(
                            f"Loop detected: pattern of length {pattern_length} repeated",
                            details=f"Pattern: {recent_pattern}",
                        )
                        return True

            return False

        except Exception as e:
            self.log_error(f"Failed to detect loops: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current context manager status."""
        status = super().get_status()
        status.update(
            {
                "memory_entries": len(self.game_state.memory_log_history),
                "action_history_length": len(self.game_state.action_history),
                "reasoning_history_length": len(
                    self.game_state.action_reasoning_history
                ),
                "current_location": self.game_state.current_room_name_for_map,
                "previous_location": self.game_state.prev_room_for_prompt_context,
            }
        )
        return status
