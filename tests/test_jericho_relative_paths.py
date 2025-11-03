# ABOUTME: Test that JerichoInterface handles relative save paths correctly
# ABOUTME: Ensures compatibility with session_manager's path handling

import pytest
import tempfile
import os
from pathlib import Path
from game_interface.core.jericho_interface import JerichoInterface


@pytest.fixture
def zork_game_path():
    """Get the path to the Zork I game file."""
    game_path = Path(__file__).parent.parent / "infrastructure" / "zork.z5"
    if game_path.exists():
        return str(game_path)
    pytest.skip("Zork I game file not found")


def test_relative_path_save_restore(zork_game_path):
    """
    Test that save/restore work with relative paths like session_manager uses.

    The session_manager passes just a filename like "autosave_session123" and
    expects it to be saved in the current working directory or a configured
    working directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to the temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            interface = JerichoInterface(zork_game_path)
            interface.start()

            # Make a move
            interface.send_command("open mailbox")

            # Save with just a filename (relative path)
            save_filename = "autosave_test_session"
            success = interface.trigger_zork_save(save_filename)
            assert success is True, "Save with relative path should succeed"

            # Check that file was created in current directory
            save_path = Path(tmpdir) / save_filename
            assert save_path.exists(), f"Save file should exist at {save_path}"

            # Make more moves
            interface.send_command("take leaflet")
            inventory = interface.get_inventory_text()
            assert len(inventory) > 0, "Should have items after taking leaflet"

            # Restore with just the filename
            success = interface.trigger_zork_restore(save_filename)
            assert success is True, "Restore with relative path should succeed"

            # Verify state was restored
            inventory = interface.get_inventory_text()
            assert len(inventory) == 0, "Inventory should be empty after restore"

            interface.close()

        finally:
            os.chdir(original_cwd)


def test_absolute_path_save_restore(zork_game_path):
    """
    Test that save/restore work with absolute paths.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        interface = JerichoInterface(zork_game_path)
        interface.start()

        # Make a move
        interface.send_command("open mailbox")

        # Save with absolute path
        save_path = Path(tmpdir) / "autosave_absolute"
        success = interface.trigger_zork_save(str(save_path))
        assert success is True, "Save with absolute path should succeed"
        assert save_path.exists(), "Save file should exist"

        # Make more moves
        interface.send_command("take leaflet")

        # Restore with absolute path
        success = interface.trigger_zork_restore(str(save_path))
        assert success is True, "Restore with absolute path should succeed"

        # Verify state was restored
        inventory = interface.get_inventory_text()
        assert len(inventory) == 0, "Inventory should be empty after restore"

        interface.close()


def test_working_directory_path_handling(zork_game_path):
    """
    Test that save files are created relative to the working directory,
    mimicking how session_manager uses it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir) / "sessions"
        working_dir.mkdir(parents=True, exist_ok=True)

        # Change to working directory
        original_cwd = os.getcwd()
        try:
            os.chdir(working_dir)

            interface = JerichoInterface(zork_game_path)
            interface.start()

            # Save with just filename (will save to working_dir)
            session_id = "2024-01-01T12:00:00"
            save_filename = f"autosave_{session_id}"

            success = interface.trigger_zork_save(save_filename)
            assert success is True, "Save should succeed"

            # Verify file exists in working directory
            save_path = working_dir / save_filename
            assert save_path.exists(), f"Save file should exist in working dir: {save_path}"

            interface.close()

        finally:
            os.chdir(original_cwd)
