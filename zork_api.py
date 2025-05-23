import subprocess
import threading
import queue
import time
import re
from typing import List


class ZorkInterface:
    """A Python interface for interacting with the Zork text adventure game."""

    def __init__(self, timeout=0.1):
        """Initialize the Zork interface.

        Args:
            timeout (float): Time to wait for responses after sending a command (in seconds)
        """
        self.timeout = timeout
        self.process = None
        self.response_queue = queue.Queue()
        self.reader_thread = None
        self.running = False
        self.current_score = 0
        self.max_score = 0

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

        self.process = subprocess.Popen(
            ["zork"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
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

        # Send the command to the game
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

        # Get the response
        return self.get_response().strip()

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
        if "empty handed" in inv_text:
            return []

        # Handling items without periods
        if "\n" in inv_text and not any("." in line for line in inv_text.split("\n")):
            return [line.strip() for line in inv_text.split("\n") if line.strip()]

        # Extract container and content relationships
        container_pattern = r"The\s+([A-Za-z0-9\s\-]+)\s+contains:"
        container_matches = re.findall(container_pattern, inv_text)

        all_items = []
        containers = {}
        lines = inv_text.split("\n")

        # Skip "You are carrying:" if present
        start_idx = 0
        if lines and "You are carrying:" in lines[0]:
            start_idx = 1

        # First pass to collect all items in order
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            # Skip container header lines
            if line.startswith("The") and "contains:" in line:
                continue

            # Regular item, remove trailing period if present
            if line.endswith("."):
                line = line[:-1]
            all_items.append(line)

        # Second pass to identify container->content relationships
        for container_name in container_matches:
            # Find the container line
            for i in range(len(lines)):
                if f"The {container_name} contains:" in lines[i]:
                    # The next line should have the content
                    if i + 1 < len(lines) and lines[i + 1].strip():
                        content = lines[i + 1].strip()
                        if content.endswith("."):
                            content = content[:-1]
                        containers[container_name] = content
                        break

        # Build the final inventory list, preserving original item order
        result = []
        processed = set()

        # Process each item in the original order
        for item in all_items:
            if item in processed:
                continue

            # Check if this item is a container
            is_container = False
            for container in containers:
                # Check if this item contains the container name
                if container.lower() in item.lower() or item.lower().endswith(
                    container.lower()
                ):
                    result.append(f"{item}: Containing {containers[container]}")
                    processed.add(item)
                    processed.add(containers[container])
                    is_container = True
                    break

            # If it's not a container and not contained in another container
            if not is_container and not any(item == containers[c] for c in containers):
                result.append(item)
                processed.add(item)

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
