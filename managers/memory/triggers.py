"""
ABOUTME: SynthesisTrigger - Detects when memory synthesis should be triggered based on Z-machine context.
ABOUTME: Pure boolean logic using ground truth state changes (no LLM calls) for fast synthesis gating.
"""

from typing import Dict, Any, Optional
import logging


class SynthesisTrigger:
    """
    Detects when memory synthesis should be triggered based on Z-machine context.

    Uses Z-machine ground truth data to make fast boolean decision without LLM calls.
    Pure boolean logic based on state changes (score, location, inventory, death, etc).

    Trigger conditions:
    1. Score changed (any non-zero delta)
    2. Location changed (moved to new room)
    3. Inventory changed (item added or removed)
    4. Death occurred (fatal outcome)
    5. First visit to location (new discovery)
    6. Substantial response (>100 characters, indicates detailed outcome)
    """

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Initialize with game configuration for trigger settings.

        Args:
            config: GameConfiguration instance for trigger settings
            logger: Optional logger for debugging trigger decisions
        """
        self.config = config
        self.logger = logger

    def should_synthesize(self, z_machine_context: Dict[str, Any]) -> bool:
        """
        Main entry point - determines if any trigger condition is met.

        Args:
            z_machine_context: Dict with keys:
                - score_before, score_after, score_delta
                - location_before, location_after, location_changed
                - inventory_before, inventory_after, inventory_changed
                - died (bool)
                - response_length (int)
                - first_visit (bool)

        Returns:
            True if LLM synthesis should be invoked, False otherwise
        """
        # Check all individual triggers in priority order
        if self._check_score_change(z_machine_context):
            return True

        if self._check_location_change(z_machine_context):
            return True

        if self._check_inventory_change(z_machine_context):
            return True

        if self._check_death(z_machine_context):
            return True

        if self._check_first_visit(z_machine_context):
            return True

        if self._check_substantial_response(z_machine_context):
            return True

        # No triggers fired
        return False

    def _check_score_change(self, context: Dict) -> bool:
        """
        Check if score changed.

        Args:
            context: Z-machine context dict

        Returns:
            True if score delta is non-zero
        """
        score_delta = context.get('score_delta', 0)
        if score_delta != 0:
            self._log_debug(
                "Trigger: Score changed",
                score_delta=score_delta
            )
            return True
        return False

    def _check_location_change(self, context: Dict) -> bool:
        """
        Check if location changed.

        Args:
            context: Z-machine context dict

        Returns:
            True if location changed
        """
        location_changed = context.get('location_changed', False)
        if location_changed:
            self._log_debug(
                "Trigger: Location changed",
                location_before=context.get('location_before'),
                location_after=context.get('location_after')
            )
            return True
        return False

    def _check_inventory_change(self, context: Dict) -> bool:
        """
        Check if inventory changed.

        Args:
            context: Z-machine context dict

        Returns:
            True if inventory changed
        """
        inventory_changed = context.get('inventory_changed', False)
        if inventory_changed:
            self._log_debug(
                "Trigger: Inventory changed",
                inventory_before=context.get('inventory_before'),
                inventory_after=context.get('inventory_after')
            )
            return True
        return False

    def _check_death(self, context: Dict) -> bool:
        """
        Check if death occurred.

        Args:
            context: Z-machine context dict

        Returns:
            True if agent died
        """
        died = context.get('died', False)
        if died:
            self._log_debug("Trigger: Death occurred")
            return True
        return False

    def _check_first_visit(self, context: Dict) -> bool:
        """
        Check if this is first visit to location.

        Args:
            context: Z-machine context dict

        Returns:
            True if first visit to location
        """
        first_visit = context.get('first_visit', False)
        if first_visit:
            self._log_debug(
                "Trigger: First visit to location",
                location=context.get('location_after')
            )
            return True
        return False

    def _check_substantial_response(self, context: Dict) -> bool:
        """
        Check if response is substantial (>100 characters).

        Args:
            context: Z-machine context dict

        Returns:
            True if response length > 100 characters
        """
        response_length = context.get('response_length', 0)
        if response_length > 100:
            self._log_debug(
                "Trigger: Substantial response",
                response_length=response_length
            )
            return True
        return False

    def _log_debug(self, message: str, **kwargs) -> None:
        """
        Log debug message if logger is available.

        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        if self.logger:
            self.logger.debug(
                message,
                extra={
                    "event_type": "debug",
                    "component": "synthesis_trigger",
                    **kwargs,
                },
            )
