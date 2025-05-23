import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON objects."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add extra fields if they exist
        if hasattr(record, "extras"):
            log_data.update(record.extras)

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """Custom formatter for human-readable console output."""

    def format(self, record):
        message = record.getMessage()

        # Format differently based on log extras
        if hasattr(record, "extras"):
            extras = record.extras
            if "turn" in extras:
                return f"\n--- Turn {extras['turn']} ---\n{message}"
            elif "agent_action" in extras:
                return f"Agent proposes: {extras['agent_action']}"
            elif "critic_score" in extras and "critic_justification" in extras:
                return f"Critic evaluation: Score={extras['critic_score']:.2f}, Justification='{extras['critic_justification']}'"
            elif "zork_response" in extras:
                return f"Zork response:\n{extras['zork_response']}"
            elif "reward" in extras and "total_reward" in extras:
                return f"Turn reward: {extras['reward']:.2f}, Total episode reward: {extras['total_reward']:.2f}"
            elif "extracted_info" in extras:
                info = extras["extracted_info"]
                return (
                    f"Extracted Info: Current Location='{info.get('current_location_name', 'Unknown')}' "
                    f"Exits='{', '.join(info.get('exits', []))}', "
                    f"Visible Objects='{', '.join(info.get('visible_objects', []))}', "
                    f"Visible Characters='{', '.join(info.get('visible_characters', []))}', "
                    f"Important Messages='{', '.join(info.get('important_messages', []))}'"
                )

        # Default formatting
        return message


def setup_logging(
    episode_log_file: str, json_log_file: str, log_level: int = logging.INFO
):
    """
    Set up logging with console and file handlers.

    Args:
        episode_log_file: Path to the human-readable log file
        json_log_file: Path to the JSON log file
        log_level: Logging level (default: INFO)
    """
    # Create logger
    logger = logging.getLogger("zorkgpt")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers

    # Console handler with human-readable formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(HumanReadableFormatter())
    logger.addHandler(console_handler)

    # File handler with human-readable formatter
    file_handler = logging.FileHandler(episode_log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(HumanReadableFormatter())
    logger.addHandler(file_handler)

    # JSON file handler
    json_handler = logging.FileHandler(json_log_file, mode="a", encoding="utf-8")
    json_handler.setLevel(log_level)
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)

    return logger


class ZorkExperienceTracker:
    """Class for tracking experiences for reinforcement learning in ZorkGPT."""

    def __init__(self):
        self.experiences = []

    def add_experience(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool,
        critic_score: float = 0,
        critic_justification: str = None,
        zork_score: int = 0,
    ):
        """Add an experience for RL."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "critic_score": critic_score,
            "critic_justification": critic_justification,
            "zork_score": zork_score,
        }
        self.experiences.append(experience)
        return experience

    def get_experiences(self) -> List[Dict[str, Any]]:
        """Get all recorded experiences."""
        return self.experiences

    def save_experiences(self, filename: str):
        """Save experiences to a JSON file for RL."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.experiences, f, indent=2)


# Create a global instance that can be imported directly
def create_zork_logger(
    episode_log_file: str = "zork_episode_log.txt",
    json_log_file: str = "zork_episode_log.jsonl",
):
    """Create and return a logger for ZorkGPT."""
    return setup_logging(episode_log_file, json_log_file)


# Experience tracker is now managed within ZorkAgent class instances


# Utility functions for parsing and rendering logs


def parse_json_logs(json_log_file: str) -> List[Dict[str, Any]]:
    """Parse a JSON log file into a list of log entries."""
    logs = []
    with open(json_log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return logs


def render_logs_as_text(logs: List[Dict[str, Any]]) -> str:
    """Render JSON logs as human-readable text."""
    output = []
    for log in logs:
        event_type = log.get("event_type", "unknown")

        if event_type == "episode_start":
            episode_id = log.get("episode_id", "unknown")
            output.append(f"\n--- NEW EPISODE: {episode_id} ({log['timestamp']}) ---")
            output.append(f"Using agent model: {log.get('agent_model', 'unknown')}")
            output.append(f"Using critic model: {log.get('critic_model', 'unknown')}")
            output.append(
                f"Using info ext model: {log.get('info_ext_model', 'unknown')}"
            )

        elif event_type == "initial_state":
            output.append(f"INITIAL STATE:\n{log.get('game_state', '')}\n")

        elif event_type == "turn_start":
            output.append(f"\n--- Turn {log.get('turn', '?')} ---")

        elif event_type == "agent_action":
            output.append(f"Agent proposes: {log.get('agent_action', '')}")

        elif event_type == "critic_evaluation":
            output.append(
                f"Critic evaluation: Score={log.get('critic_score', 0.0):.2f}, "
                f"Justification='{log.get('critic_justification', '')}'"
            )

        elif event_type == "zork_response":
            output.append(
                f"ZORK RESPONSE for '{log.get('action', '')}':\n{log.get('zork_response', '')}\n"
            )

        elif event_type == "reward":
            output.append(
                f"Turn reward: {log.get('reward', 0.0):.2f}, "
                f"Total episode reward: {log.get('total_reward', 0.0):.2f}"
            )

        elif event_type == "episode_end":
            output.append(f"\nEpisode finished in {log.get('turn_count', 0)} turns.")
            output.append(
                f"Final Zork Score: {log.get('zork_score', 0)} / {log.get('max_score', 'N/A')}"
            )
            output.append(
                f"Total accumulated reward for episode: {log.get('total_reward', 0.0):.2f}"
            )
        elif event_type == "unknown":
            # Fallback for simpler log entries
            output.append(
                f"{log.get('timestamp', '')}: {log.get('level', '')} - {log.get('message', '')}"
            )

    return "\n".join(output)


def format_experiences_for_rl(experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format experiences for reinforcement learning frameworks."""
    # This can be customized based on the specific RL framework being used
    # For example, formatting for PyTorch, TensorFlow, or other RL libraries
    formatted_data = {
        "states": [exp["state"] for exp in experiences],
        "actions": [exp["action"] for exp in experiences],
        "rewards": [exp["reward"] for exp in experiences],
        "next_states": [exp["next_state"] for exp in experiences],
        "dones": [exp["done"] for exp in experiences],
        "metadata": {
            "critic_scores": [exp.get("critic_score", 0) for exp in experiences],
            "zork_scores": [exp.get("zork_score", 0) for exp in experiences],
        },
    }
    return formatted_data
