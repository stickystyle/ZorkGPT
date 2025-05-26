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
            ["dfrotz", "./infrastructure/zork.z5"],
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
            if stripped.startswith(">") and ("Score:" in stripped or "Moves:" in stripped):
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
                while j < len(lines) and (lines[j].startswith("  ") or lines[j].strip() == ""):
                    if lines[j].strip():  # Non-empty indented line
                        skip_lines.add(j)
                    j += 1
        
        # Second pass: collect items and handle containers
        containers = {}
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
