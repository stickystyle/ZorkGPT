"""
Test that inventory sync from Z-machine is working correctly and not being overwritten.

This test verifies the fix for the critical data race where extractor was
overwriting the Z-machine inventory sync.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2
from session.game_configuration import GameConfiguration
from hybrid_zork_extractor import ExtractorResponse


class TestInventorySyncFix:
    """Test inventory sync from Z-machine is authoritative."""

    @pytest.fixture
    def mock_config(self):
        """Create a test configuration."""
        config = GameConfiguration.from_toml()
        return config

    def test_inventory_sync_not_overwritten_by_extractor(self, mock_config):
        """Test that Z-machine inventory sync is not overwritten by extractor."""
        # Create orchestrator
        orchestrator = ZorkOrchestratorV2(episode_id="test-inventory-sync")

        # Mock the Jericho interface to return specific inventory
        # Create Mock objects with 'name' as an attribute, not constructor arg
        lantern = Mock()
        lantern.name = "brass lantern"
        leaflet = Mock()
        leaflet.name = "leaflet"
        sword = Mock()
        sword.name = "sword"

        mock_zmachine_inventory = [lantern, leaflet, sword]

        with patch.object(
            orchestrator.jericho_interface,
            'get_inventory_structured',
            return_value=mock_zmachine_inventory
        ):
            # Call the sync method
            orchestrator._sync_inventory_from_z_machine()

            # Verify inventory was set from Z-machine
            assert orchestrator.game_state.current_inventory == [
                "brass lantern", "leaflet", "sword"
            ]

            # Now simulate extractor returning different inventory
            extractor_response = ExtractorResponse(
                current_location_name="Test Room",
                inventory=["wrong item 1", "wrong item 2"],  # Extractor parsed wrong items
                exits=[],
                visible_objects=[],
                visible_characters=[],
                important_messages=[],
                in_combat=False,
            )

            # Process extraction (this should NOT overwrite inventory)
            orchestrator._process_extraction(
                extractor_response,
                action="test action",
                response="test response"
            )

            # Verify inventory is STILL the Z-machine version, not extractor version
            assert orchestrator.game_state.current_inventory == [
                "brass lantern", "leaflet", "sword"
            ], "Extractor should not overwrite Z-machine inventory"

            assert orchestrator.game_state.current_inventory != [
                "wrong item 1", "wrong item 2"
            ], "Extractor inventory should be ignored"

    def test_inventory_sync_handles_jericho_failure(self, mock_config):
        """Test that inventory sync handles Jericho failures gracefully."""
        orchestrator = ZorkOrchestratorV2(episode_id="test-inventory-sync-error")

        # Set initial inventory
        orchestrator.game_state.current_inventory = ["initial item"]

        # Mock Jericho to raise RuntimeError
        with patch.object(
            orchestrator.jericho_interface,
            'get_inventory_structured',
            side_effect=RuntimeError("Z-machine error")
        ):
            # Call sync - should not crash
            orchestrator._sync_inventory_from_z_machine()

            # Verify inventory is unchanged (previous state preserved)
            assert orchestrator.game_state.current_inventory == ["initial item"]

    def test_inventory_sync_called_after_action(self, mock_config):
        """Test that inventory sync is called after every action execution."""
        orchestrator = ZorkOrchestratorV2(episode_id="test-inventory-sync-timing")

        # Track whether sync was called
        sync_called = []

        def track_sync():
            sync_called.append(True)
            return []  # Return empty inventory

        # Setup mocks for full turn execution
        with patch.object(orchestrator, '_sync_inventory_from_z_machine', side_effect=track_sync):
            with patch.object(orchestrator.jericho_interface, 'send_command', return_value="Test response"):
                with patch.object(orchestrator.jericho_interface, 'is_game_over', return_value=(False, "")):
                    with patch.object(orchestrator.jericho_interface, 'get_score', return_value=(0, 0)):
                        with patch.object(orchestrator.jericho_interface, 'get_location_structured') as mock_location:
                            mock_location.return_value = Mock(num=1, name="Test Room")
                            with patch.object(orchestrator.jericho_interface, 'get_inventory_structured', return_value=[]):
                                with patch.object(orchestrator.agent, 'get_action_with_reasoning', return_value={"action": "look", "reasoning": "test"}):
                                    with patch.object(orchestrator.critic, 'evaluate_action') as mock_critic:
                                        mock_critic_result = Mock(score=0.8, justification="test", confidence=0.9)
                                        mock_critic.return_value = mock_critic_result
                                        with patch.object(orchestrator.extractor, 'extract_info', return_value=Mock()):
                                            with patch.object(orchestrator.extractor, 'get_clean_game_text', return_value="Test"):
                                                # Execute a turn
                                                orchestrator._execute_turn_logic("Test state")

        # Verify sync was called during turn execution
        assert len(sync_called) > 0, "Inventory sync should be called after action execution"
