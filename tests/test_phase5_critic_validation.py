"""
Tests for Phase 5.3: ZorkCritic Object Tree Validation.

This module tests the integration of Z-machine object tree validation
into the critic evaluation process. Tests verify that:
- Invalid actions are rejected with high confidence based on Z-machine data
- Valid actions are allowed to proceed to LLM evaluation
- Validation doesn't interfere with normal critic operation
- Error handling works properly when Jericho methods fail
"""

import pytest
from unittest.mock import Mock
from zork_critic import ZorkCritic, ValidationResult


class TestValidationMethod:
    """Test the _validate_against_object_tree method directly."""

    @pytest.fixture
    def critic(self):
        """Create a minimal ZorkCritic instance for testing validation."""
        # Create instance without calling __init__ to avoid config loading
        critic = object.__new__(ZorkCritic)
        critic.logger = Mock()
        critic.episode_id = "test-episode"
        return critic

    @pytest.fixture
    def mock_jericho(self):
        """Create a mock JerichoInterface."""
        return Mock()

    def test_rejects_take_action_when_object_not_visible(self, critic, mock_jericho):
        """Test that validation rejects 'take lamp' when lamp is not in room."""
        # Setup: lamp not in visible objects
        mock_jericho.get_visible_objects_in_location.return_value = []

        # Execute
        result = critic._validate_against_object_tree("take lamp", mock_jericho)

        # Verify
        assert result.valid is False
        assert result.confidence == 0.9
        assert "Object 'lamp' is not visible" in result.reason

    def test_allows_take_action_when_object_is_takeable(self, critic, mock_jericho):
        """Test that validation allows 'take lamp' when lamp IS in room and takeable."""
        # Setup: lamp is visible and takeable
        mock_lamp = Mock()
        mock_lamp.name = "brass lamp"
        mock_jericho.get_visible_objects_in_location.return_value = [mock_lamp]
        mock_jericho.get_object_attributes.return_value = {
            'takeable': True,
            'portable': False,
            'openable': False
        }

        # Execute
        result = critic._validate_against_object_tree("take lamp", mock_jericho)

        # Verify - should pass validation
        assert result.valid is True

    def test_take_action_only_validates_visibility(self, critic, mock_jericho):
        """Test that 'take' validation only checks visibility, not takeable attribute.

        The takeable attribute in Jericho is unreliable (e.g., lunch has takeable=False
        but can actually be taken), so validation only checks if the object is visible.
        The game engine will provide the authoritative response about whether the object
        can be taken.
        """
        # Setup: mailbox is visible but has takeable=False
        mock_mailbox = Mock()
        mock_mailbox.name = "small mailbox"
        mock_jericho.get_visible_objects_in_location.return_value = [mock_mailbox]
        mock_jericho.get_object_attributes.return_value = {
            'takeable': False,
            'portable': False,
            'openable': True
        }

        # Execute
        result = critic._validate_against_object_tree("take mailbox", mock_jericho)

        # Verify - should pass because object is visible, regardless of takeable attribute
        assert result.valid is True

    def test_rejects_open_action_when_object_not_present(self, critic, mock_jericho):
        """Test that validation rejects 'open door' when door is not present."""
        # Setup: no door visible
        mock_jericho.get_visible_objects_in_location.return_value = []

        # Execute
        result = critic._validate_against_object_tree("open door", mock_jericho)

        # Verify
        assert result.valid is False
        assert result.confidence == 0.9
        assert "Object 'door' is not present" in result.reason

    def test_allows_open_action_when_object_is_openable(self, critic, mock_jericho):
        """Test that validation allows 'open mailbox' when mailbox is openable."""
        # Setup: mailbox is visible and openable
        mock_mailbox = Mock()
        mock_mailbox.name = "small mailbox"
        mock_jericho.get_visible_objects_in_location.return_value = [mock_mailbox]
        mock_jericho.get_object_attributes.return_value = {
            'takeable': False,
            'portable': False,
            'openable': True
        }

        # Execute
        result = critic._validate_against_object_tree("open mailbox", mock_jericho)

        # Verify - should pass validation
        assert result.valid is True

    def test_allows_open_action_for_containers(self, critic, mock_jericho):
        """Test that validation allows opening containers even without openable bit.

        The mailbox in Zork is a container without the openable bit set,
        but can still be opened. This test verifies that containers are
        treated as openable regardless of the openable attribute.
        """
        # Setup: mailbox is a container but openable=False (matches real Zork behavior)
        mock_mailbox = Mock()
        mock_mailbox.name = "small mailbox"
        mock_jericho.get_visible_objects_in_location.return_value = [mock_mailbox]
        mock_jericho.get_object_attributes.return_value = {
            'takeable': False,
            'container': True,  # Container bit set
            'openable': False,  # Openable bit NOT set (like real mailbox)
            'transparent': True
        }

        # Execute
        result = critic._validate_against_object_tree("open mailbox", mock_jericho)

        # Verify - should pass validation because it's a container
        assert result.valid is True

    def test_rejects_open_action_when_object_not_openable(self, critic, mock_jericho):
        """Test that validation rejects 'open lamp' when lamp is not openable."""
        # Setup: lamp is visible but not openable and not a container
        mock_lamp = Mock()
        mock_lamp.name = "brass lamp"
        mock_jericho.get_visible_objects_in_location.return_value = [mock_lamp]
        mock_jericho.get_object_attributes.return_value = {
            'takeable': True,
            'portable': True,
            'openable': False,
            'container': False  # Not a container, so should be rejected
        }

        # Execute
        result = critic._validate_against_object_tree("open lamp", mock_jericho)

        # Verify
        assert result.valid is False
        assert result.confidence == 0.9
        assert "Object 'lamp' cannot be opened/closed" in result.reason

    def test_validation_doesnt_interfere_with_other_actions(self, critic, mock_jericho):
        """Test that validation allows unvalidated actions (like movement) to pass."""
        # Setup: movement action that isn't validated by object tree
        mock_jericho.get_visible_objects_in_location.return_value = []

        # Execute
        result = critic._validate_against_object_tree("go north", mock_jericho)

        # Verify - should pass validation (let LLM handle it)
        assert result.valid is True
        assert "not validated by object tree" in result.reason

    def test_error_handling_when_jericho_fails(self, critic, mock_jericho):
        """Test that validation fails gracefully when Jericho methods raise errors."""
        # Setup: Jericho method raises an exception
        mock_jericho.get_visible_objects_in_location.side_effect = RuntimeError("Jericho error")

        # Execute
        result = critic._validate_against_object_tree("take lamp", mock_jericho)

        # Verify - should fall back to allowing action
        assert result.valid is True
        assert "Validation error" in result.reason
        # Should have logged warning
        assert critic.logger.warning.called

    def test_single_word_commands_pass_validation(self, critic, mock_jericho):
        """Test that single-word commands (look, inventory) pass validation."""
        # Execute
        result = critic._validate_against_object_tree("look", mock_jericho)

        # Verify - should pass validation
        assert result.valid is True
        assert "Single word command" in result.reason
        # Should not have called Jericho methods
        assert not mock_jericho.get_visible_objects_in_location.called

    def test_portable_attribute_makes_object_takeable(self, critic, mock_jericho):
        """Test that objects with 'portable' attribute are considered takeable."""
        # Setup: object has portable=True but takeable=False
        mock_item = Mock()
        mock_item.name = "leaflet"
        mock_jericho.get_visible_objects_in_location.return_value = [mock_item]
        mock_jericho.get_object_attributes.return_value = {
            'takeable': False,
            'portable': True,  # Should be sufficient
            'openable': False
        }

        # Execute
        result = critic._validate_against_object_tree("take leaflet", mock_jericho)

        # Verify - portable should be treated as takeable
        assert result.valid is True

    def test_validation_with_alternate_verbs(self, critic, mock_jericho):
        """Test validation works with alternate verb forms (get, grab, pick)."""
        # Setup: lamp not visible
        mock_jericho.get_visible_objects_in_location.return_value = []

        # Test "get"
        result = critic._validate_against_object_tree("get lamp", mock_jericho)
        assert result.valid is False
        assert "not visible" in result.reason

        # Test "grab"
        result = critic._validate_against_object_tree("grab lamp", mock_jericho)
        assert result.valid is False
        assert "not visible" in result.reason

        # Test "pick"
        result = critic._validate_against_object_tree("pick lamp", mock_jericho)
        assert result.valid is False
        assert "not visible" in result.reason

    def test_close_action_validation(self, critic, mock_jericho):
        """Test that 'close' action is validated like 'open'."""
        # Setup: lamp visible but not openable and not a container
        mock_lamp = Mock()
        mock_lamp.name = "lamp"
        mock_jericho.get_visible_objects_in_location.return_value = [mock_lamp]
        mock_jericho.get_object_attributes.return_value = {
            'openable': False,
            'container': False
        }

        # Execute
        result = critic._validate_against_object_tree("close lamp", mock_jericho)

        # Verify
        assert result.valid is False
        assert "cannot be opened/closed" in result.reason

    def test_validation_result_dataclass(self):
        """Test the ValidationResult dataclass."""
        # Test valid result
        valid = ValidationResult(valid=True, reason="Action is valid")
        assert valid.valid is True
        assert valid.reason == "Action is valid"
        assert valid.confidence == 0.9  # Default

        # Test invalid result with custom confidence
        invalid = ValidationResult(
            valid=False,
            reason="Object not found",
            confidence=0.95
        )
        assert invalid.valid is False
        assert invalid.reason == "Object not found"
        assert invalid.confidence == 0.95


class TestIntegrationWithOrchestrator:
    """Test integration of validation with the orchestrator."""

    def test_orchestrator_passes_jericho_interface_to_critic(self):
        """Test that the orchestrator passes jericho_interface to critic.evaluate_action()."""
        # This is a structural test - verify the orchestrator code has the right parameters
        orchestrator_file = "/Volumes/workingfolder/ZorkGPT/orchestration/zork_orchestrator_v2.py"
        with open(orchestrator_file, 'r') as f:
            orchestrator_code = f.read()

        # Verify both critic.evaluate_action calls include jericho_interface
        assert "jericho_interface=self.jericho_interface" in orchestrator_code
        # Should appear twice (initial evaluation and re-evaluation in rejection loop)
        assert orchestrator_code.count("jericho_interface=self.jericho_interface") >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
