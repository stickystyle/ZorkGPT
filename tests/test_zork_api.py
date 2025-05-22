import unittest
from unittest.mock import MagicMock
from zork_api import ZorkInterface


class TestZorkInterfaceInventory(unittest.TestCase):
    def setUp(self):
        # Mock the ZorkInterface for testing inventory without a live game
        self.zork_interface = ZorkInterface()
        # We need to mock send_command as inventory() calls it
        self.zork_interface.send_command = MagicMock()

    def test_inventory_with_containers(self):
        # Mock the response from Zork for an inventory with a container
        mock_inventory_output = """You are carrying:
  A glass bottle.
  A lamp.
The glass bottle contains:
  A quantity of water.
  """
        self.zork_interface.send_command.return_value = mock_inventory_output

        expected_inventory = [
            "A glass bottle: Containing A quantity of water",
            "A lamp",
        ]
        actual_inventory = self.zork_interface.inventory()
        self.assertEqual(actual_inventory, expected_inventory)

    def test_inventory_without_containers(self):
        # Mock the response from Zork for an inventory with simple items
        mock_inventory_output = """You are carrying:
  A pile of leaves.
  A sword."""
        self.zork_interface.send_command.return_value = mock_inventory_output

        expected_inventory = ["A pile of leaves", "A sword"]
        actual_inventory = self.zork_interface.inventory()
        self.assertEqual(actual_inventory, expected_inventory)

    def test_inventory_empty(self):
        # Mock the response from Zork for an empty inventory
        mock_inventory_output = "You are empty handed."
        self.zork_interface.send_command.return_value = mock_inventory_output

        expected_inventory = []
        actual_inventory = self.zork_interface.inventory()
        self.assertEqual(actual_inventory, expected_inventory)

    def test_inventory_item_not_ending_with_period_in_container(self):
        mock_inventory_output = """You are carrying:
  An elvish sword.
  A small box.
The small box contains:
  A strange coin
  """
        self.zork_interface.send_command.return_value = mock_inventory_output
        expected_inventory = [
            "An elvish sword",
            "A small box: Containing A strange coin",
        ]
        actual_inventory = self.zork_interface.inventory()
        self.assertEqual(actual_inventory, expected_inventory)

    def test_inventory_only_item_in_container_not_ending_with_period(self):
        mock_inventory_output = """You are carrying:
  A small box.
The small box contains:
  A strange coin"""
        self.zork_interface.send_command.return_value = mock_inventory_output
        expected_inventory = ["A small box: Containing A strange coin"]
        actual_inventory = self.zork_interface.inventory()
        self.assertEqual(actual_inventory, expected_inventory)

    def test_inventory_only_items_no_period(self):
        mock_inventory_output = """  A leaflet
  A sword"""  # Items might not always have periods, especially if that's the only thing
        self.zork_interface.send_command.return_value = mock_inventory_output
        expected_inventory = ["A leaflet", "A sword"]
        actual_inventory = self.zork_interface.inventory()
        self.assertEqual(actual_inventory, expected_inventory)


if __name__ == "__main__":
    unittest.main()
