"""
ABOUTME: Unit tests for SynthesisTrigger detection logic.
ABOUTME: Tests when memory synthesis should/should not be triggered.
"""

import pytest


class TestTriggerDetectionPositiveCases:
    """Test cases where SynthesisTrigger.should_synthesize() should return True."""

    def test_trigger_on_positive_score_change(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when score increases."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Score increases by 5
        context = base_context.copy()
        context['score_after'] = 55
        context['score_delta'] = 5

        result = trigger.should_synthesize(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("score" in call.lower() for call in debug_calls)

    def test_trigger_on_negative_score_change(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when score decreases."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Score decreases by 10
        context = base_context.copy()
        context['score_after'] = 40
        context['score_delta'] = -10

        result = trigger.should_synthesize(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("score" in call.lower() for call in debug_calls)

    def test_trigger_on_location_change(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when location changes."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Location changes from 15 to 23
        context = base_context.copy()
        context['location_after'] = 23
        context['location_changed'] = True

        result = trigger.should_synthesize(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("location" in call.lower() for call in debug_calls)

    def test_trigger_on_inventory_item_added(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when inventory gains an item."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Inventory gains 'key'
        context = base_context.copy()
        context['inventory_after'] = ['lamp', 'sword', 'key']
        context['inventory_changed'] = True

        result = trigger.should_synthesize(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("inventory" in call.lower() for call in debug_calls)

    def test_trigger_on_inventory_item_removed(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when inventory loses an item."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Inventory loses 'sword'
        context = base_context.copy()
        context['inventory_after'] = ['lamp']
        context['inventory_changed'] = True

        result = trigger.should_synthesize(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("inventory" in call.lower() for call in debug_calls)

    def test_trigger_on_death(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when death occurs."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Death occurred
        context = base_context.copy()
        context['died'] = True

        result = trigger.should_synthesize(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("death" in call.lower() for call in debug_calls)

    def test_trigger_on_first_visit(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires on first visit to location."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # First visit to location
        context = base_context.copy()
        context['first_visit'] = True

        result = trigger.should_synthesize(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("first visit" in call.lower() for call in debug_calls)

    def test_trigger_on_substantial_response(self, mock_logger, game_config, game_state, base_context):
        """Test trigger fires when response length exceeds 100 characters."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Substantial response (>100 chars)
        context = base_context.copy()
        context['response_length'] = 150

        result = trigger.should_synthesize(context)

        # Should trigger
        assert result is True

        # Logger should have logged reason
        mock_logger.debug.assert_called()
        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        assert any("substantial" in call.lower() or "response" in call.lower() for call in debug_calls)


class TestTriggerDetectionNegativeCases:
    """Test cases where SynthesisTrigger.should_synthesize() should return False."""

    def test_no_trigger_on_trivial_action(self, mock_logger, game_config, game_state, base_context):
        """Test no trigger when nothing significant happens."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # No changes at all
        context = base_context.copy()

        result = trigger.should_synthesize(context)

        # Should NOT trigger
        assert result is False

    def test_no_trigger_when_multiple_conditions_false(self, mock_logger, game_config, game_state):
        """Test no trigger when all conditions are false."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Explicitly false conditions
        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 20,
            'first_visit': False
        }

        result = trigger.should_synthesize(context)

        # Should NOT trigger
        assert result is False

    def test_no_trigger_on_short_response_only(self, mock_logger, game_config, game_state, base_context):
        """Test no trigger when response is short and nothing else changes."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        # Short response with no other changes
        context = base_context.copy()
        context['response_length'] = 30

        result = trigger.should_synthesize(context)

        # Should NOT trigger
        assert result is False


class TestTriggerDetectionEdgeCases:
    """Test edge cases for trigger detection."""

    def test_edge_case_score_change_of_zero(self, mock_logger, game_config, game_state):
        """Test that score delta of 0 does not trigger."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,  # Explicit zero delta
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        result = trigger.should_synthesize(context)

        # Should NOT trigger (no actual change)
        assert result is False

    def test_edge_case_location_change_to_same_location(self, mock_logger, game_config, game_state):
        """Test that location staying the same does not trigger."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,  # Same location
            'location_changed': False,  # Explicit false
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        result = trigger.should_synthesize(context)

        # Should NOT trigger (no actual change)
        assert result is False

    def test_edge_case_inventory_change_with_same_items(self, mock_logger, game_config, game_state):
        """Test that inventory with same items does not trigger."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp', 'sword'],
            'inventory_after': ['lamp', 'sword'],  # Same items
            'inventory_changed': False,  # Explicit false
            'died': False,
            'response_length': 50,
            'first_visit': False
        }

        result = trigger.should_synthesize(context)

        # Should NOT trigger (no actual change)
        assert result is False

    def test_edge_case_response_exactly_100_chars(self, mock_logger, game_config, game_state):
        """Test boundary condition: exactly 100 characters should NOT trigger."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 100,  # Exactly 100
            'first_visit': False
        }

        result = trigger.should_synthesize(context)

        # Should NOT trigger (must be > 100, not >= 100)
        assert result is False

    def test_edge_case_response_exactly_101_chars(self, mock_logger, game_config, game_state):
        """Test boundary condition: 101 characters should trigger."""
        from managers.memory import SynthesisTrigger

        trigger = SynthesisTrigger(config=game_config, logger=mock_logger)

        context = {
            'score_before': 50,
            'score_after': 50,
            'score_delta': 0,
            'location_before': 15,
            'location_after': 15,
            'location_changed': False,
            'inventory_before': ['lamp'],
            'inventory_after': ['lamp'],
            'inventory_changed': False,
            'died': False,
            'response_length': 101,  # Just over threshold
            'first_visit': False
        }

        result = trigger.should_synthesize(context)

        # Should trigger (> 100)
        assert result is True
