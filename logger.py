import json
import logging
from datetime import datetime
from typing import Any, Dict, List


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON objects."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add any extra attributes that were passed via extra={}
        # This excludes standard logging attributes
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "exc_info",
            "exc_text",
            "stack_info",
            "getMessage",
        }

        for attr_name, attr_value in record.__dict__.items():
            if attr_name not in standard_attrs and not attr_name.startswith("_"):
                log_data[attr_name] = attr_value

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """Custom formatter for human-readable console output focused on gameplay."""

    def format(self, record):
        message = record.getMessage()

        # Skip debug messages for console output unless it's an error/warning
        if record.levelname == "DEBUG":
            return None  # Don't display debug messages on console

        # Always show errors and warnings, regardless of event type
        if record.levelname in ["ERROR", "WARNING"]:
            return f"{record.levelname}: {message}"

        # Check for structured event types and format accordingly
        if hasattr(record, "event_type"):
            event_type = record.event_type

            # Key gameplay events to show
            if event_type == "episode_initialized":
                episode_id = getattr(record, "episode_id", "unknown")
                return f"\n🎮 NEW EPISODE: {episode_id}"

            elif event_type == "turn_completed":
                turn = getattr(record, "turn", "?")
                action = getattr(record, "action", "unknown")
                score = getattr(record, "score", 0)
                location = getattr(record, "location", "unknown")
                confidence = getattr(record, "confidence", 0)
                return f"Turn {turn}: '{action}' → Score: {score}, Location: {location}, Confidence: {confidence:.2f}"

            elif event_type == "episode_completed" or event_type == "episode_finalized":
                turn = getattr(record, "turn", "?")
                final_score = getattr(record, "final_score", 0)
                reason = getattr(record, "reason", "completed")
                return f"🏁 Episode completed after {turn} turns - Final Score: {final_score} ({reason})"

            elif event_type == "agent_action_parsed":
                action = getattr(record, "action", "unknown")
                return f"  Agent: '{action}'"

            elif event_type == "critic_evaluation":
                confidence = getattr(record, "confidence", 0)
                return f"  Critic: confidence {confidence:.2f}"

            elif event_type == "objective_update" and hasattr(record, "status"):
                status = record.status
                details = getattr(record, "details", "")
                return f"📋 Objective [{status}]: {details}"

            elif event_type == "knowledge_update":
                return f"📚 Knowledge: {message}"

            # Hide low-value progress messages unless they're important
            elif event_type == "progress":
                stage = getattr(record, "stage", "")
                # Only show key progress events
                if stage in [
                    "episode_initialization",
                    "episode_finalization",
                    "inter_episode_synthesis",
                ]:
                    details = getattr(record, "details", "")
                    return f"⚙️  {stage.replace('_', ' ').title()}: {details if details else message}"
                else:
                    return None  # Hide routine progress messages

            # Hide other low-value events
            elif event_type in [
                "agent_raw_response_debug",
                "reasoning_extraction_debug",
                "fallback_reasoning_debug",
                "final_reasoning_debug",
                "map_consolidation",
                "agent_llm_response",
            ]:
                return None  # Don't display these on console

        # For non-structured messages, only show if they're important
        if record.levelname == "INFO" and any(
            keyword in message.lower()
            for keyword in [
                "error",
                "failed",
                "exception",
                "warning",
                "completed",
                "initialized",
            ]
        ):
            return message

        # Hide everything else to keep console clean
        return None


class FilteringStreamHandler(logging.StreamHandler):
    """Stream handler that filters out None messages from formatter."""

    def emit(self, record):
        try:
            msg = self.format(record)
            if msg is not None:  # Only emit if formatter didn't return None
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


class FilteringFileHandler(logging.FileHandler):
    """File handler that filters out None messages from formatter."""

    def emit(self, record):
        try:
            msg = self.format(record)
            if msg is not None:  # Only emit if formatter didn't return None
                if self.stream is None:
                    self.stream = self._open()
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


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

    # Console handler with human-readable formatter (filtered)
    console_handler = FilteringStreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(HumanReadableFormatter())
    logger.addHandler(console_handler)

    # File handler with human-readable formatter (filtered)
    file_handler = FilteringFileHandler(episode_log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(HumanReadableFormatter())
    logger.addHandler(file_handler)

    # JSON file handler
    json_handler = logging.FileHandler(json_log_file, mode="a", encoding="utf-8")
    json_handler.setLevel(log_level)
    json_handler.setFormatter(JSONFormatter())
    json_handler.is_json_handler = True  # Mark for identification
    logger.addHandler(json_handler)

    return logger


def setup_episode_logging(
    episode_id: str, workdir: str = "game_files", log_level: int = logging.INFO
):
    """
    Setup logging for a specific episode.
    Creates episode directory and configures JSON handler for episode-specific logging.

    Args:
        episode_id: The episode identifier
        workdir: Working directory for game files
        log_level: Logging level (default: INFO)

    Returns:
        Path to the episode log file
    """
    from pathlib import Path

    # Create episode directory
    episode_dir = Path(workdir) / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Episode-specific log file
    episode_log_file = episode_dir / "episode_log.jsonl"

    # Get existing logger
    logger = logging.getLogger("zorkgpt")
    logger.setLevel(log_level)

    # Remove existing JSON handler if present
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and hasattr(
            handler, "is_json_handler"
        ):
            logger.removeHandler(handler)
            handler.close()

    # Create episode-specific JSON handler
    json_handler = logging.FileHandler(episode_log_file, mode="a", encoding="utf-8")
    json_handler.setFormatter(JSONFormatter())
    json_handler.setLevel(log_level)
    json_handler.is_json_handler = True  # Mark for identification

    logger.addHandler(json_handler)

    return str(episode_log_file)


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


def parse_episode_logs(
    episode_id: str, workdir: str = "game_files"
) -> List[Dict[str, Any]]:
    """
    Parse logs from a specific episode.

    Args:
        episode_id: The episode identifier
        workdir: Working directory for game files

    Returns:
        List of log entries for the episode
    """
    from pathlib import Path

    episode_log_file = Path(workdir) / "episodes" / episode_id / "episode_log.jsonl"

    if not episode_log_file.exists():
        return []

    logs = []
    try:
        with open(episode_log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass

    return logs
