# ABOUTME: Production interface for interacting with Zork I via Jericho library
# ABOUTME: Provides both structured (ZObject) and text-based access to game state

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
from jericho import FrotzEnv
from jericho.util import clean


class JerichoInterface:
    """
    Interface for interacting with Zork I using the Jericho library.

    Provides two types of access:
    1. Structured: Direct access to Z-machine objects (inventory, location, object tree)
    2. Text-based: Compatible text representations for backward compatibility

    The structured methods eliminate the need for brittle regex parsing by accessing
    the Z-machine object tree directly. Jericho's ZObject provides:
        - num: Object ID number
        - name: Object name/description
        - parent: Parent object ID
        - sibling: Sibling object ID
        - child: Child object ID
        - attr: Attribute flags (as bytes)
        - properties: Object properties
    """

    def __init__(self, game_file_path: str, logger=None):
        """
        Initialize the Jericho interface.

        Args:
            game_file_path: Absolute path to the .z5 game file
            logger: Optional logger instance for debugging
        """
        if not game_file_path.endswith(".z5"):
            raise ValueError(f"Game file must be a .z5 file, got: {game_file_path}")

        self.game_file_path = game_file_path
        self.logger = logger
        self.env: Optional[FrotzEnv] = None

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting the context manager."""
        self.close()
        return False  # Don't suppress exceptions

    def start(self) -> str:
        """
        Initialize the Frotz environment and return the game intro text.

        Returns:
            The initial game text/introduction

        Raises:
            RuntimeError: If the environment cannot be initialized
            FileNotFoundError: If the game file doesn't exist
        """
        if not os.path.exists(self.game_file_path):
            raise FileNotFoundError(f"Game file not found: {self.game_file_path}")

        try:
            self.env = FrotzEnv(self.game_file_path)
            intro, _ = self.env.reset()
            intro_text = clean(intro)

            if self.logger:
                self.logger.info(
                    f"Jericho environment started successfully - intro length: {len(intro_text)}"
                )

            return intro_text
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start Jericho environment: {e}")
            raise RuntimeError(f"Failed to initialize Jericho: {e}") from e

    def send_command(self, cmd: str) -> str:
        """
        Execute a game command and return the text response.

        Args:
            cmd: Command to send to the game

        Returns:
            The game's text response

        Raises:
            RuntimeError: If the environment is not initialized
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        observation, reward, done, info = self.env.step(cmd)
        observation_text = clean(observation)

        if self.logger:
            self.logger.debug(
                f"Command '{cmd}' executed - response length: {len(observation_text)}"
            )

        return observation_text

    def get_inventory_structured(self) -> List[Any]:
        """
        Get the player's inventory as a list of ZObjects.

        This provides direct access to the Z-machine object tree, eliminating
        the need for regex parsing of inventory text. Uses Jericho's built-in
        get_inventory() method.

        Returns:
            List of Jericho ZObject instances representing items in inventory.
            Each ZObject has attributes: num, name, parent, sibling, child, attr, properties

        Raises:
            RuntimeError: If the environment is not initialized
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        return self.env.get_inventory()

    def get_inventory_text(self) -> List[str]:
        """
        Get the player's inventory as a list of text strings.

        This provides backward compatibility with text-based inventory systems.

        Returns:
            List of item names as strings
        """
        inventory_objects = self.get_inventory_structured()
        return [obj.name for obj in inventory_objects]

    def get_location_structured(self) -> Any:
        """
        Get the current location as a ZObject.

        This provides direct access to the location object in the Z-machine,
        eliminating the need for text parsing. Uses Jericho's built-in
        get_player_location() method.

        Returns:
            Jericho ZObject representing the current location.
            ZObject has attributes: num, name, parent, sibling, child, attr, properties

        Raises:
            RuntimeError: If the environment is not initialized
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        return self.env.get_player_location()

    def get_location_text(self) -> str:
        """
        Get the current location name as a string.

        This provides backward compatibility with text-based location systems.

        Returns:
            The location name
        """
        location = self.get_location_structured()
        return location.name if location else ""

    def get_player_object(self) -> Any:
        """
        Get the player object from the Z-machine.

        Returns:
            Jericho ZObject representing the player

        Raises:
            RuntimeError: If the environment is not initialized
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        return self.env.get_player_object()

    def get_all_objects(self) -> List[Any]:
        """
        Get the entire object tree from the Z-machine.

        This is useful for exploration and understanding the game state structure.

        Returns:
            List of all Jericho ZObjects in the game.
            Each ZObject has attributes: num, name, parent, sibling, child, attr, properties

        Raises:
            RuntimeError: If the environment is not initialized
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        return self.env.get_world_objects()

    def save_state(self) -> tuple:
        """
        Get the current game state for later restoration.

        Returns:
            Tuple containing the internal game state

        Raises:
            RuntimeError: If the environment is not initialized or state cannot be saved
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        state = self.env.get_state()
        if state is None:
            raise RuntimeError("Failed to save state - get_state() returned None")

        if self.logger:
            self.logger.debug("Game state saved successfully")

        return state

    def restore_state(self, state: tuple) -> None:
        """
        Restore the game to a previously saved state.

        Args:
            state: State tuple from save_state()

        Raises:
            RuntimeError: If the environment is not initialized
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        self.env.set_state(state)

    def get_score(self) -> Tuple[int, int]:
        """
        Get the current score and maximum possible score.

        Returns:
            Tuple of (current_score, max_score)

        Raises:
            RuntimeError: If the environment is not initialized
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        current_score = self.env.get_score()
        max_score = self.env.get_max_score()
        return (current_score, max_score)

    def score(self) -> Tuple[int, int]:
        """Alias for get_score() to match ZorkInterface API."""
        return self.get_score()

    def inventory(self) -> List[str]:
        """Alias for get_inventory_text() to match ZorkInterface API."""
        return self.get_inventory_text()

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        status = "running" if self.env is not None else "not started"
        return f"JerichoInterface(game_file='{self.game_file_path}', status='{status}')"

    def __str__(self) -> str:
        """User-friendly representation."""
        if self.env is None:
            return f"Jericho interface (not started): {self.game_file_path}"

        try:
            location = self.get_location_text()
            inv_count = len(self.get_inventory_structured())
            score, max_score = self.get_score()
            return f"Jericho interface at {location} - Score: {score}/{max_score} - Items: {inv_count}"
        except Exception:
            return f"Jericho interface (error reading state): {self.game_file_path}"

    def trigger_zork_save(self, save_filename: str) -> bool:
        """
        Save the current game state to a file for later restoration.

        This method provides session manager compatibility by saving Jericho's
        internal state tuple to a pickle file. Unlike the dfrotz-based ZorkInterface
        which uses in-game save commands, this uses Jericho's get_state() method.

        Args:
            save_filename: Path to save file (absolute or relative path)

        Returns:
            True if save was successful, False otherwise
        """
        if self.env is None:
            if self.logger:
                self.logger.error("Save failed: Environment not started")
            return False

        try:
            # Get the state tuple from Jericho
            state = self.env.get_state()
            if state is None:
                if self.logger:
                    self.logger.error("Save failed: get_state() returned None")
                return False

            # Ensure directory exists
            save_path = Path(save_filename)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize state to file using pickle
            with open(save_path, "wb") as f:
                pickle.dump(state, f)

            if self.logger:
                self.logger.info(f"Successfully saved game state to {save_filename}")
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save game state: {e}")
            return False

    def trigger_zork_restore(self, save_filename: str) -> bool:
        """
        Restore game state from a previously saved file.

        This method provides session manager compatibility by loading a pickled
        state tuple and applying it using Jericho's set_state() method.

        Args:
            save_filename: Path to save file (absolute or relative path)

        Returns:
            True if restore was successful, False otherwise
        """
        if self.env is None:
            if self.logger:
                self.logger.error("Restore failed: Environment not started")
            return False

        try:
            save_path = Path(save_filename)

            if not save_path.exists():
                if self.logger:
                    self.logger.error(
                        f"Restore failed: Save file not found: {save_filename}"
                    )
                return False

            # Load state tuple from pickle file
            with open(save_path, "rb") as f:
                state = pickle.load(f)

            # Restore the state using Jericho
            self.env.set_state(state)

            if self.logger:
                self.logger.info(
                    f"Successfully restored game state from {save_filename}"
                )
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to restore game state: {e}")
            return False

    def is_game_over(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the game has ended based on response text.

        This method provides backward compatibility with the session manager
        by checking for common Zork death/victory phrases. Jericho also provides
        a 'done' flag from env.step(), but this method allows checking any text.

        Args:
            text: Game response text to check for game-over conditions

        Returns:
            Tuple of (is_over: bool, reason: Optional[str])
            - is_over: True if game has ended
            - reason: Description of why the game ended, or None if not ended
        """
        # Common Zork game-over phrases
        game_ending_phrases = {
            "you have died": "Player death",
            "you are dead": "Player death",
            "game over": "Game over",
            "****  you have won  ****": "Victory",
            "your score is now": None,  # Score changes don't mean game over
        }

        text_lower = text.lower()

        # Check for death/victory phrases
        for phrase, reason in game_ending_phrases.items():
            if phrase in text_lower:
                # "your score is now" is common and doesn't mean game over
                if phrase == "your score is now":
                    continue

                if self.logger:
                    self.logger.info(
                        f"Game over detected: {reason} (phrase: '{phrase}')"
                    )
                return True, reason

        # Not game over
        return False, None

    # ABOUTME: Object attribute helpers for enhanced context (Phase 5)
    # ABOUTME: Provide structured access to Z-machine object attributes and vocabulary

    def _check_attribute(self, obj: Any, bit: int) -> bool:
        """
        Check if a specific attribute bit is set on a Z-machine object.

        Jericho exposes object attributes as a numpy array where each element
        corresponds to one attribute bit. A value of 1 means the attribute is set.

        Args:
            obj: ZObject instance from Jericho
            bit: Attribute bit number (0-31 for Jericho's representation)

        Returns:
            True if the bit is set, False otherwise

        Example:
            >>> leaflet = game.get_inventory_structured()[0]
            >>> is_takeable = self._check_attribute(leaflet, 10)
        """
        if obj is None or not hasattr(obj, 'attr'):
            return False

        try:
            # Get the attribute array
            # In Jericho, obj.attr is a numpy array where attr[i] = 1 if bit i is set
            attr_array = obj.attr
            if attr_array is None or len(attr_array) == 0:
                return False

            # Check bounds
            if bit < 0 or bit >= len(attr_array):
                return False

            # Check if the attribute is set (array value is 1)
            return bool(attr_array[bit] != 0)

        except (AttributeError, IndexError, TypeError):
            return False

    def get_object_attributes(self, obj: Any) -> Dict[str, bool]:
        """
        Extract useful attributes from a Z-machine object.

        This method checks common attribute bits used in Zork I. Note that
        attribute bit positions are game-dependent and assigned by the ZIL
        compiler at compile time. The mappings below are based on empirical
        testing with Zork I using Jericho.

        Zork I attribute bits (empirically determined through testing):
        - Bit 3: touched/manipulated (set when object is interacted with)
        - Bit 13: container (mailbox has this)
        - Bit 14: openable (door, mailbox, window)
        - Bit 16, 17: related to takeable items (leaflet has these)
        - Bit 19: possibly transparent or secondary container flag
        - Bit 26: possibly takeable (leaflet has this)

        NOTE: These bit positions are specific to Zork I and may differ in
        other Z-machine games. The ZIL compiler assigns attributes to free
        slots during compilation. Some mappings are tentative and based on
        limited testing.

        Args:
            obj: ZObject instance from Jericho

        Returns:
            Dictionary mapping attribute names to boolean values

        Example:
            >>> visible = game.get_visible_objects_in_location()
            >>> mailbox = visible[0]  # Or find by name
            >>> attrs = game.get_object_attributes(mailbox)
            >>> if attrs.get('openable'):
            ...     print("You can open this!")
        """
        if obj is None:
            return {}

        try:
            # Check common Zork I attributes
            # Note: Bit positions are empirically determined for Zork I
            # Some mappings are tentative and may need refinement
            attributes = {
                'touched': self._check_attribute(obj, 3),
                'container': self._check_attribute(obj, 13),
                'openable': self._check_attribute(obj, 14),
                'takeable': self._check_attribute(obj, 26),  # Tentative
                'transparent': self._check_attribute(obj, 19),  # Tentative
                'portable': self._check_attribute(obj, 16),  # Tentative - leaflet has this
                'readable': self._check_attribute(obj, 17),  # Tentative - leaflet has this
            }

            return attributes

        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to extract attributes from object: {e}"
                )
            return {}

    def get_visible_objects_in_location(self) -> List[Any]:
        """
        Get all objects visible in the current location.

        This method traverses the Z-machine object tree to find objects
        whose parent is the current location and filters for visible objects.
        The player object itself is excluded from the results.

        Returns:
            List of ZObject instances that are visible in the current location

        Raises:
            RuntimeError: If the environment is not initialized

        Example:
            >>> visible = game.get_visible_objects_in_location()
            >>> for obj in visible:
            ...     print(f"You can see: {obj.name}")
        """
        if self.env is None:
            raise RuntimeError("Environment not started. Call start() first.")

        try:
            current_location = self.get_location_structured()
            if current_location is None:
                return []

            location_id = current_location.num
            visible_objects = []

            # Get the player object to exclude it from visible objects
            player = self.get_player_object()
            player_id = player.num if player else None

            # Get all objects in the world
            all_objects = self.env.get_world_objects()

            # Find objects whose parent is the current location
            for obj in all_objects:
                if obj.parent == location_id:
                    # Exclude the player object - you can't interact with yourself
                    if player_id is not None and obj.num == player_id:
                        continue

                    # Include all other objects whose parent is the current location
                    # In Z-machine, parent relationship determines visibility
                    visible_objects.append(obj)

            return visible_objects

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Failed to get visible objects in location: {e}"
                )
            return []

    def get_valid_verbs(self) -> List[str]:
        """
        Get a list of valid action verbs recognized by Zork I.

        This returns a hardcoded list of common Zork verbs. In future
        enhancements, this could be extracted from the Z-machine dictionary
        table dynamically.

        Returns:
            List of valid verb strings

        Example:
            >>> verbs = game.get_valid_verbs()
            >>> print(f"You can use these verbs: {', '.join(verbs[:10])}")

        TODO: Extract verbs dynamically from Z-machine dictionary table
        """
        # Common Zork I verbs (alphabetically organized for readability)
        # Categories: movement, interaction, combat, communication, meta
        return [
            # Movement
            'go', 'north', 'south', 'east', 'west', 'up', 'down',
            'northeast', 'northwest', 'southeast', 'southwest',
            'enter', 'exit', 'climb', 'jump',

            # Object interaction
            'take', 'get', 'drop', 'put', 'place',
            'open', 'close', 'lock', 'unlock',
            'examine', 'look', 'read', 'search',
            'move', 'push', 'pull', 'turn', 'press',
            'touch', 'rub', 'shake', 'wave',
            'throw', 'tie', 'attach',

            # Container/inventory management
            'inventory', 'i',

            # Light sources
            'light', 'extinguish',

            # Combat
            'attack', 'kill', 'fight', 'hit', 'strike',

            # Communication
            'say', 'tell', 'ask', 'answer', 'shout', 'yell',

            # Consumption
            'eat', 'drink',

            # Equipment
            'wear', 'remove',

            # Magic/special
            'pray', 'chant', 'incant',
            'ring', 'blow', 'play',

            # Meta commands
            'wait', 'sleep', 'wake',
            'save', 'restore', 'quit', 'restart',
            'score', 'brief', 'verbose', 'superbrief',
            'diagnose', 'version',
        ]

    def close(self) -> None:
        """Cleanup and close the environment."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error closing Jericho environment: {e}")
            finally:
                self.env = None
                if self.logger:
                    self.logger.debug("Jericho environment closed")
