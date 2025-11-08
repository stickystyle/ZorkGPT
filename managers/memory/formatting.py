"""
ABOUTME: History formatting helpers for ZorkGPT memory synthesis prompts.
ABOUTME: Provides HistoryFormatter class for action and reasoning history markdown generation.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from session.game_state import ActionHistoryEntry


class HistoryFormatter:
    """Formats action and reasoning history for memory synthesis prompts."""

    def format_recent_actions(
        self,
        actions: List[Union[ActionHistoryEntry, Tuple[str, str]]],
        start_turn: int
    ) -> str:
        """
        Format recent action/response pairs into markdown for multi-step synthesis.

        Part of Phase 2 helpers used by Phase 3's multi-step procedure detection.
        Matches ContextManager formatting conventions for consistency across systems.

        Args:
            actions: List of ActionHistoryEntry or (action, response) tuples from game_state.action_history
            start_turn: Turn number of the first action in the list

        Returns:
            Formatted markdown string with turn context, empty string if no actions

        Example output:
            Turn 47: go north
            Response: You are in a forest clearing.

            Turn 48: examine trees
            Response: The trees are ordinary pine trees.

        Usage:
            Injected into synthesis prompt's "RECENT ACTION SEQUENCE" section to give
            LLM temporal context for detecting prerequisites and delayed consequences.
        """
        # Handle empty list
        if not actions:
            return ""

        lines = []
        for i, entry in enumerate(actions):
            turn_num = start_turn + i
            # Handle both ActionHistoryEntry and tuple formats
            if isinstance(entry, ActionHistoryEntry):
                action = entry.action
                response = entry.response
            else:
                action, response = entry
            lines.append(f"Turn {turn_num}: {action}")
            lines.append(f"Response: {response}")
            # Add blank line between entries (except after last entry)
            if i < len(actions) - 1:
                lines.append("")

        return "\n".join(lines)

    def format_recent_reasoning(
        self,
        reasoning_entries: List[Dict[str, Any]],
        action_history: Optional[List[Union[ActionHistoryEntry, Tuple[str, str]]]] = None
    ) -> str:
        """
        Format recent reasoning history into markdown for multi-step synthesis.

        Part of Phase 2 helpers used by Phase 3's multi-step procedure detection.
        Matches ContextManager formatting conventions (Turn → Reasoning → Action → Response).
        Uses reverse iteration through action_history to match actions to responses.

        Args:
            reasoning_entries: List of reasoning history dicts from game_state.action_reasoning_history
                Each dict has: turn, reasoning, action, timestamp
            action_history: Optional list of ActionHistoryEntry or (action, response) tuples for response lookup
                Uses reverse iteration to handle duplicate actions correctly

        Returns:
            Formatted markdown string with reasoning context, empty string if no entries

        Example output:
            Turn 47:
            Reasoning: I need to explore north systematically.
            Action: go north
            Response: You are in a forest clearing.

            Turn 48:
            Reasoning: Will examine objects before moving on.
            Action: examine trees
            Response: The trees are ordinary pine trees.

        Usage:
            Injected into synthesis prompt's "AGENT'S REASONING" section to help LLM
            understand strategic intent behind multi-step procedures.
        """
        # Handle empty list
        if not reasoning_entries:
            return ""

        lines = []
        for i, entry in enumerate(reasoning_entries):
            # Skip non-dict entries gracefully
            if not isinstance(entry, dict):
                continue

            # Extract fields with fallbacks
            turn = entry.get("turn")
            if turn is None:
                turn = "?"
            reasoning = entry.get("reasoning", "(No reasoning recorded)")
            action = entry.get("action", "(No action recorded)")

            # Find matching game response from action_history
            # Iterate in reverse to match the most recent occurrence
            response = "(Response not recorded)"
            if action_history:
                for hist_entry in reversed(action_history):
                    # Handle both ActionHistoryEntry and tuple formats
                    if isinstance(hist_entry, ActionHistoryEntry):
                        hist_action = hist_entry.action
                        hist_response = hist_entry.response
                    else:
                        hist_action, hist_response = hist_entry

                    if hist_action == action:
                        response = hist_response
                        break

            # Format this entry
            lines.append(f"Turn {turn}:")
            lines.append(f"Reasoning: {reasoning}")
            lines.append(f"Action: {action}")
            lines.append(f"Response: {response}")
            # Add blank line between entries (except after last entry)
            if i < len(reasoning_entries) - 1:
                lines.append("")

        return "\n".join(lines)
