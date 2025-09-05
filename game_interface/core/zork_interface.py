"""
ABOUTME: Core interface for managing the Zork game process through dfrotz
ABOUTME: Provides low-level communication with the text adventure game engine
"""

import subprocess
import threading
import queue
import time
import re
from typing import List
import os


class ZorkInterface:
    """A Python interface for interacting with the Zork text adventure game."""

    def __init__(self, timeout=0.1, working_directory=None, logger=None):
        """Initialize the Zork interface.

        Args:
            timeout (float): Time to wait for responses after sending a command (in seconds)
            working_directory (str): Optional working directory for the Zork process (for save files)
            logger: Optional logger instance for enhanced debugging
        """
        self.timeout = timeout
        self.working_directory = working_directory
        self.logger = logger
        self.process = None
        self.response_queue = queue.Queue()
        self.reader_thread = None
        self.running = False
        self.current_score = 0
        self.max_score = 0
        # Add synchronization lock to prevent race conditions between save and game commands
        self.command_lock = threading.Lock()

    def __enter__(self):
        """Support for context manager protocol."""
        # Don't call start here, let the user call it explicitly
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting the context manager."""
        self.close()

    def start(self):
        """Start the Zork game process."""
        if self.process is not None:
            raise RuntimeError("Zork process is already running")

        # Get absolute path to zork.z5 file to ensure it works regardless of working directory
        # Get project root (two levels up from game_interface/core)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        zork_file_path = os.path.join(project_root, "infrastructure", "zork.z5")

        self.process = subprocess.Popen(
            ["dfrotz", zork_file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            cwd=self.working_directory,  # Set working directory for save files
        )

        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_thread)
        self.reader_thread.daemon = True
        self.reader_thread.start()

        # Wait longer for the initial game output and give it time to fully load
        time.sleep(2)
        intro_text = self.get_response()

        # If the intro text doesn't contain the welcome message, it may not have been captured yet
        if "Welcome to Dungeon" not in intro_text:
            time.sleep(1)  # Wait a bit more
            additional_text = self.get_response()
            intro_text = (
                intro_text + "\n" + additional_text if intro_text else additional_text
            )

        return intro_text

    def _reader_thread(self):
        """Background thread to read output from the Zork process."""
        while self.running:
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    self.running = False
                    break
                continue
            self.response_queue.put(line)

    def get_response(self) -> str:
        """Get the full response since the last command."""
        response = ""
        time.sleep(self.timeout)  # Give the game time to respond

        while not self.response_queue.empty():
            response += self.response_queue.get()

        # Clean up the response (remove extra whitespace and game prompt)
        response = response.strip()
        return response

    def clear_response_queue(self):
        """Clear any pending responses from the queue."""
        # First clear what's currently in the queue
        while not self.response_queue.empty():
            self.response_queue.get()

        # Wait a bit for any remaining output to arrive from the reader thread
        time.sleep(0.1)

        # Clear again to catch any new output that arrived during the delay
        while not self.response_queue.empty():
            self.response_queue.get()

    def _drain_queue_completely(self) -> str:
        """Drain the response queue completely, waiting for all output to arrive.

        This method ensures we capture ALL output from dfrotz, including any delayed
        output that arrives after an initial queue check. Critical for save operations.

        Returns:
            str: All output that was in the queue
        """
        response = ""

        # First pass - get everything currently in queue
        while not self.response_queue.empty():
            response += self.response_queue.get()

        # Wait for any delayed output from dfrotz and check again
        time.sleep(0.3)
        while not self.response_queue.empty():
            response += self.response_queue.get()

        # One more shorter wait to catch any final stragglers
        time.sleep(0.1)
        while not self.response_queue.empty():
            response += self.response_queue.get()

        return response.strip()

    def is_running(self) -> bool:
        """Check if the Zork process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None and self.running

    def send_command(self, command) -> str:
        """Send a command to the Zork game and return the response.

        Args:
            command (str): The command to send to the game

        Returns:
            str: The game's response to the command
        """
        if not self.is_running():
            raise RuntimeError("Zork process is not running")

        # Use lock to prevent race conditions with save operations
        with self.command_lock:
            # Send the command to the game
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

            # Get the response
            response = self.get_response().strip()

        # Enhanced logging for debug - capture all game responses
        if self.logger:
            self.logger.debug(
                "Game command and response",
                extra={
                    "event_type": "game_command_response_debug",
                    "command": command,
                    "response": response,
                    "response_length": len(response),
                },
            )

        return response

    def send_interactive_command(
        self,
        initial_command: str,
        follow_up_input: str,
        prompt_keyword: str,
        success_keyword: str,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        """Send a command that requires interactive input (like save/restore).

        Args:
            initial_command: The first command to send (e.g., "save")
            follow_up_input: The response to the prompt (e.g., filename)
            prompt_keyword: Text to look for in Zork's prompt (e.g., "Please enter a filename")
            success_keyword: Text indicating success (e.g., "Ok.")
            timeout: Maximum time to wait for responses

        Returns:
            tuple: (success: bool, full_response: str)
        """
        if not self.is_running():
            raise RuntimeError("Zork process is not running")

        full_response = ""

        try:
            # Send the initial command and filename together for save/restore
            # Zork's save/restore commands don't produce output until filename is provided
            self.process.stdin.write(initial_command + "\n")
            self.process.stdin.flush()

            # Send the follow-up input immediately (for save/restore this is the filename)
            self.process.stdin.write(follow_up_input + "\n")
            self.process.stdin.flush()

            # Wait for the combined response (prompt + result)
            time.sleep(1.0)  # Give Zork time to process both commands

            response = self.get_response()
            full_response = response

            # Check if we got both the prompt and success message
            has_prompt = prompt_keyword.lower() in response.lower()
            has_success = success_keyword.lower() in response.lower()

            # For save/restore, success means we got both prompt and "Ok." in the response
            success_found = has_prompt and has_success

            return success_found, full_response.strip()

        except Exception as e:
            return (
                False,
                f"Error during interactive command: {e}. Response: {full_response}",
            )

    def trigger_zork_save(self, filename: str) -> bool:
        """Save the current Zork game state to a file.

        Args:
            filename: The filename to save to (relative to Zork's working directory)
                     Note: Zork will automatically add .qzl extension

        Returns:
            bool: True if save was successful
        """
        if not self.is_running():
            if self.logger:
                self.logger.error("Save failed: Zork process not running")
            return False

        # Use the same lock as send_command to ensure complete synchronization
        with self.command_lock:
            try:
                if self.logger:
                    self.logger.debug(
                        f"Starting save operation with filename: {filename}"
                    )

                # Clear queue first to start fresh
                self.clear_response_queue()

                # Send save command
                self.process.stdin.write("save\n")
                self.process.stdin.flush()

                # Wait and collect initial response
                time.sleep(0.8)
                initial_response = self._drain_queue_completely()

                if self.logger:
                    self.logger.debug(
                        f"Save initial response: {repr(initial_response)}"
                    )

                # Send filename
                self.process.stdin.write(filename + "\n")
                self.process.stdin.flush()

                # Wait and collect response after filename
                # Increased wait time to ensure dfrotz has time to present overwrite prompt
                time.sleep(3.0)
                filename_response = self._drain_queue_completely()

                if self.logger:
                    self.logger.debug(
                        f"Save filename response: {repr(filename_response)}"
                    )

                # Check if we got an overwrite prompt
                combined_response = initial_response + " " + filename_response
                if (
                    "overwrite" in combined_response.lower()
                    and "?" in combined_response
                ):
                    if self.logger:
                        self.logger.debug("Got overwrite prompt, sending 'y'")

                    # Send yes to overwrite
                    self.process.stdin.write("y\n")
                    self.process.stdin.flush()

                    # Wait and collect final response
                    time.sleep(1.5)
                    final_response = self._drain_queue_completely()

                    if self.logger:
                        self.logger.debug(
                            f"Save final response after overwrite: {repr(final_response)}"
                        )

                    combined_response += " " + final_response

                # Send a marker command to ensure we're completely done with save
                # Use "score" as it's harmless and gives us a clean response
                self.process.stdin.write("score\n")
                self.process.stdin.flush()

                # Wait for marker response
                time.sleep(0.5)
                marker_response = self._drain_queue_completely()

                if self.logger:
                    self.logger.debug(f"Save marker response: {repr(marker_response)}")

                # Analyze the save response (excluding marker)
                save_response = combined_response.lower()

                # Check for success indicators
                success_indicators = [
                    "ok.",
                    "saved",
                    "done",
                    ".qzl",
                ]

                # Check for failure indicators
                failure_indicators = [
                    "can't",
                    "cannot",
                    "unable",
                    "error",
                    "failed",
                    "invalid",
                ]

                has_success = any(
                    indicator in save_response for indicator in success_indicators
                )
                has_failure = any(
                    indicator in save_response for indicator in failure_indicators
                )

                if has_failure:
                    if self.logger:
                        self.logger.error(
                            f"Save failed - failure indicator found: {combined_response}"
                        )
                    return False

                if has_success:
                    if self.logger:
                        self.logger.info(
                            f"Save succeeded - success indicator found: {combined_response}"
                        )
                    return True

                # If response is very short and no failure indicators, assume success
                if len(combined_response.strip()) < 50 and not has_failure:
                    if self.logger:
                        self.logger.info(
                            f"Save likely succeeded - minimal response: {combined_response}"
                        )
                    return True

                # Default to failure if unclear
                if self.logger:
                    self.logger.warning(
                        f"Save status unclear - defaulting to failure: {combined_response}"
                    )
                return False

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Save failed with exception: {e}")
                return False

    def trigger_zork_restore(self, filename: str) -> bool:
        """Restore a Zork game state from a file.

        Args:
            filename: The filename to restore from (relative to Zork's working directory)
                     Note: Should match the name used in save (Zork adds .qzl extension)

        Returns:
            bool: True if restore was successful
        """
        success, response = self.send_interactive_command(
            "restore", filename, "Please enter a filename", "Ok."
        )

        if not success:
            if self.logger:
                self.logger.error(f"Restore failed: {response}")

        return success

    def close(self):
        """Terminate the Zork game process."""
        if self.process is not None:
            self.running = False
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None

    def score(self, score_text=None) -> tuple[int, int]:
        if not score_text:
            score_text = self.send_command("score").strip()
        current_score, max_score = 0, 0  # Default if parsing fails

        # Try structured format first: "> Room Name ... Score: X ... Moves: Y"
        structured_match = re.search(
            r">\s*(.+?)\s+Score:\s*(\d+)\s+Moves:\s*(\d+)", score_text, re.MULTILINE
        )
        if structured_match:
            current_score = int(structured_match.group(2))
            max_score = 585  # Default max score for Zork I when not specified
            return current_score, max_score

        # "Your score is 0 [total of 350 points], in 1 moves."
        match = re.search(
            r"Your score is (\d+)\s*\[total of (\d+) points], in \d+ moves.", score_text
        )
        if match:
            current_score = int(match.group(1))
            max_score = int(match.group(2))
        else:
            # Alternative format, e.g. from 'score' command directly if not in room description
            match = re.search(
                r"Your score is (\d+) of a possible (\d+), in \d+ moves\.", score_text
            )
            if match:
                current_score = int(match.group(1))
                max_score = int(match.group(2))
            else:  # "Score: 5 points out of 350, in 8 turns." (Frotz sometimes)
                match = re.search(r"Score:\s*(\d+)\s*points out of\s*(\d+)", score_text)
                if match:
                    current_score = int(match.group(1))
                    max_score = int(match.group(2))

        return current_score, max_score

    def is_game_over(self, text) -> tuple[bool, str]:
        """
        Check if the game has ended based on the response text.

        Args:
            text (str): The text output from the game to check

        Returns:
            tuple: (bool, str) where the boolean indicates if the game is over,
                  and the string provides a reason (or None if game is not over)
        """
        # Common game-ending phrases
        game_ending_phrases = [
            # Death messages
            "you have died",
            "oh, no! you walked into the slavering fangs",
            "you clearly are a suicidal maniac",
            "your adventure ends here",
            "I'm afraid, that you are dead",
            "your head is taken off by the axe",
            "the troll swings his axe and it cuts your head off",
            "with his final blow, the troll kills you",
            "the troll takes a mighty swing and cleaves you",
            "you are dead",
            "game over",
            # Victory/ending messages
            "the dungeon master appears",
            "****  you have won  ****",
            "your score would be",
            "this gives you the rank of",
        ]

        if text:
            text_lower = text.lower()
            for phrase in game_ending_phrases:
                if phrase.lower() in text_lower:
                    # Find the score if present in the ending message
                    score_match = re.search(
                        r"your score (would be|is) (\d+).*?in (\d+) moves", text_lower
                    )
                    if score_match:
                        reason = f"Game over - Score: {score_match.group(2)} in {score_match.group(3)} moves"
                    else:
                        reason = f"Game over - {phrase}"
                    return True, reason

        return False, None

    def inventory(self) -> List[str]:
        """Get the current inventory of the player.
        Parses Zork's output to handle items, items in containers, and empty inventory.
        """
        inv_text = self.send_command("inventory")
        return self._parse_inventory(inv_text)

    def inventory_with_response(self) -> tuple[List[str], str]:
        """Get the current inventory and the raw response text.
        Returns both the parsed inventory and the raw response for game-over checking.
        """
        inv_text = self.send_command("inventory")
        return self._parse_inventory(inv_text), inv_text

    def _parse_inventory(self, inv_text: str) -> List[str]:
        """Parse inventory text into a list of items."""
        # Enhanced logging for debug - capture full response text
        if self.logger:
            self.logger.debug(
                "Inventory parsing input",
                extra={
                    "event_type": "inventory_parse_debug",
                    "raw_inventory_text": inv_text,
                    "text_length": len(inv_text),
                },
            )

        # Check for death messages that shouldn't be parsed as inventory
        death_indicators = [
            "you have died",
            "you are dead",
            "slavering fangs",
            "eaten by a grue",
            "you have been killed",
            "****  you have died  ****",
            "fatal",
            "troll",
            "axe hits you",
            "puts you to death",
            "last blow was too much",
            "i'm afraid you are dead",
            "conquering his fears",
            "flat of the troll's axe",
        ]

        inv_text_lower = inv_text.lower()
        for indicator in death_indicators:
            if indicator in inv_text_lower:
                if self.logger:
                    self.logger.warning(
                        "Death text detected in inventory response",
                        extra={
                            "event_type": "death_text_in_inventory",
                            "death_indicator": indicator,
                            "full_response": inv_text,
                        },
                    )
                return []  # Return empty inventory if death text detected

        # Check for empty inventory (case insensitive)
        if "empty-handed" in inv_text.lower() or "empty handed" in inv_text.lower():
            return []

        lines = inv_text.split("\n")
        result = []
        skip_lines = set()

        # First pass: identify structure and collect all lines that should be skipped
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                skip_lines.add(i)
                continue

            # Skip game status lines
            if stripped.startswith(">") and (
                "Score:" in stripped or "Moves:" in stripped
            ):
                skip_lines.add(i)
                continue

            # Skip "You are carrying:" header
            if stripped == "You are carrying:":
                skip_lines.add(i)
                continue

            # Container header line - mark it and its contents for special processing
            if stripped.startswith("The") and "contains:" in stripped:
                skip_lines.add(i)
                # Mark following indented lines as container contents
                j = i + 1
                while j < len(lines) and (
                    lines[j].startswith("  ") or lines[j].strip() == ""
                ):
                    if lines[j].strip():  # Non-empty indented line
                        skip_lines.add(j)
                    j += 1

        # Second pass: collect items and handle containers
        for i, line in enumerate(lines):
            stripped = line.strip()

            if i in skip_lines or not stripped:
                continue

            # This is a regular item
            if stripped.endswith("."):
                stripped = stripped[:-1]
            result.append(stripped)

        # Third pass: find containers and their contents
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("The") and "contains:" in stripped:
                # Extract container name
                container_match = re.search(r"The\s+([^:]+)\s+contains:", stripped)
                if container_match:
                    container_name = container_match.group(1).strip()

                    # Find first content item
                    first_content = None
                    j = i + 1
                    while j < len(lines):
                        content_line = lines[j]
                        if content_line.startswith("  ") and content_line.strip():
                            first_content = content_line.strip()
                            if first_content.endswith("."):
                                first_content = first_content[:-1]
                            break
                        elif content_line.strip() == "":
                            j += 1
                        else:
                            break
                        j += 1

                    # Check if this container is already in our result (top-level item)
                    found_in_result = False
                    if first_content:
                        for idx, item in enumerate(result):
                            if container_name.lower() in item.lower():
                                result[idx] = f"{item}: Containing {first_content}"
                                found_in_result = True
                                break

                    # If not found in result, it might be a nested container - add it as a separate item
                    if not found_in_result and first_content:
                        result.append(f"A {container_name}: Containing {first_content}")

        return result


if __name__ == "__main__":
    # Example usage
    with ZorkInterface() as zork:
        print("Starting Zork...")
        initial_text = zork.start()
        print("Game introduction:")
        print(initial_text)

        print("\nSending 'look' command...")
        response = zork.send_command("look")
        print(response)

        print("\nSending 'inventory' command...")
        response = zork.inventory()
        print(response)

        print("\nchecking score")
        response = zork.score()
        print(response)

        print("\nZork session ended.")
