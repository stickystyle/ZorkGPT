# ABOUTME: Integration test verifying JerichoInterface works with session manager patterns
# ABOUTME: Tests that all three new methods (trigger_zork_save, trigger_zork_restore, is_game_over) work in realistic scenarios

import pytest
import tempfile
from pathlib import Path
from game_interface.core.jericho_interface import JerichoInterface


@pytest.fixture
def zork_game_path():
    """Get the path to the Zork I game file."""
    game_path = Path(__file__).parent.parent / "infrastructure" / "zork.z5"
    if game_path.exists():
        return str(game_path)
    pytest.skip("Zork I game file not found")


def test_session_manager_workflow(zork_game_path):
    """
    Test a realistic session manager workflow:
    1. Start a game
    2. Make some moves
    3. Save the session
    4. Make more moves
    5. Restore to the saved point
    6. Verify state is correctly restored
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "session_save.pkl"

        # Start interface
        interface = JerichoInterface(zork_game_path)
        intro = interface.start()

        try:
            # Verify we got intro text
            assert len(intro) > 0, "Should get intro text"

            # Make initial moves (session start)
            initial_location = interface.get_location_text()
            response1 = interface.send_command("open mailbox")

            # Check for game over (should be False)
            is_over, reason = interface.is_game_over(response1)
            assert is_over is False, "Game should not be over after opening mailbox"
            assert reason is None, "Reason should be None when game is not over"

            # Save the session at this point
            save_success = interface.trigger_zork_save(str(save_path))
            assert save_success is True, "Save should succeed"

            # Continue playing - take leaflet and read it
            response2 = interface.send_command("take leaflet")
            response3 = interface.send_command("read leaflet")

            # Verify inventory has changed
            current_inventory = interface.get_inventory_text()
            assert len(current_inventory) > 0, "Should have items after taking leaflet"

            # Check for game over (should still be False)
            is_over, reason = interface.is_game_over(response3)
            assert is_over is False, "Game should not be over after reading leaflet"

            # Now restore to the saved point
            restore_success = interface.trigger_zork_restore(str(save_path))
            assert restore_success is True, "Restore should succeed"

            # Verify we're back to the state right after opening mailbox
            restored_inventory = interface.get_inventory_text()
            assert len(restored_inventory) == 0, "Inventory should be empty after restore"

            # Verify location is still the same
            restored_location = interface.get_location_text()
            assert restored_location == initial_location, "Location should match initial location"

        finally:
            interface.close()


def test_game_over_detection_with_actual_game(zork_game_path):
    """
    Test game-over detection by creating a scenario that could lead to death.

    Note: We simulate the death message since actually dying in Zork requires
    specific scenarios that are hard to reproduce in a test.
    """
    interface = JerichoInterface(zork_game_path)
    interface.start()

    try:
        # Test with simulated death messages
        death_messages = [
            "You fall into a pit and die. You have died.",
            "The troll swings his axe and cleaves your skull. You are dead.",
            "****  You have won  **** Your score is 350 of 350.",
        ]

        for msg in death_messages:
            is_over, reason = interface.is_game_over(msg)
            assert is_over is True, f"Should detect game over in: {msg}"
            assert reason is not None, f"Should have a reason for: {msg}"

        # Test with normal game text
        normal_response = interface.send_command("look")
        is_over, reason = interface.is_game_over(normal_response)
        assert is_over is False, "Normal game text should not trigger game over"

    finally:
        interface.close()


def test_multiple_save_restore_cycles(zork_game_path):
    """
    Test multiple save/restore cycles to ensure state management is robust.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        save1_path = Path(tmpdir) / "save1.pkl"
        save2_path = Path(tmpdir) / "save2.pkl"

        interface = JerichoInterface(zork_game_path)
        interface.start()

        try:
            # Save point 1: Initial state
            interface.trigger_zork_save(str(save1_path))

            # Make moves
            interface.send_command("open mailbox")
            interface.send_command("take leaflet")

            # Save point 2: After taking leaflet
            interface.trigger_zork_save(str(save2_path))

            # Make more moves
            interface.send_command("south")

            # Restore to save point 2
            success = interface.trigger_zork_restore(str(save2_path))
            assert success is True, "Should restore to save point 2"

            inventory_at_save2 = interface.get_inventory_text()
            assert "leaflet" in str(inventory_at_save2).lower(), "Should have leaflet at save point 2"

            # Restore to save point 1
            success = interface.trigger_zork_restore(str(save1_path))
            assert success is True, "Should restore to save point 1"

            inventory_at_save1 = interface.get_inventory_text()
            assert len(inventory_at_save1) == 0, "Should have empty inventory at save point 1"

        finally:
            interface.close()


def test_save_directory_creation(zork_game_path):
    """
    Test that save operation creates directories if they don't exist.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a nested path that doesn't exist yet
        nested_save_path = Path(tmpdir) / "nested" / "directories" / "save.pkl"

        interface = JerichoInterface(zork_game_path)
        interface.start()

        try:
            # Should create directories as needed
            success = interface.trigger_zork_save(str(nested_save_path))
            assert success is True, "Should create directories and save"
            assert nested_save_path.exists(), "Save file should exist"

        finally:
            interface.close()
