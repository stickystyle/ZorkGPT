"""
Game configuration management for ZorkGPT orchestration.

This module provides a clean, typed interface to game configuration,
loading directly from pyproject.toml and environment variables.
"""

import os
import tomllib
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class GameConfiguration:
    """
    Typed configuration object for ZorkGPT orchestration.

    Loads configuration directly from pyproject.toml and environment variables.
    All required values must be present in TOML or the application will crash.
    """

    # Required fields (no defaults)
    # Core game settings
    max_turns_per_episode: int
    turn_delay_seconds: float

    # File paths
    episode_log_file: str
    json_log_file: str
    state_export_file: str
    zork_game_workdir: str
    game_file_path: str

    # LLM client settings
    client_base_url: str

    # Model specifications
    agent_model: str
    critic_model: str
    info_ext_model: str
    analysis_model: str

    # Update intervals
    knowledge_update_interval: int
    map_update_interval: int
    objective_update_interval: int

    # Objective refinement
    enable_objective_refinement: bool
    objective_refinement_interval: int
    max_objectives_before_forced_refinement: int
    refined_objectives_target_count: int

    # Context management
    max_context_tokens: int
    context_overflow_threshold: float

    # State export
    enable_state_export: bool
    s3_key_prefix: str

    # Gameplay settings
    critic_rejection_threshold: float

    # Optional fields (with defaults)
    # LLM client settings
    client_api_key: Optional[str] = None

    # Per-model base URLs (optional)
    agent_base_url: Optional[str] = None
    info_ext_base_url: Optional[str] = None
    critic_base_url: Optional[str] = None
    analysis_base_url: Optional[str] = None

    # State export
    s3_bucket: Optional[str] = None

    def __post_init__(self):
        """Load environment variables after initialization."""
        # Load environment variables
        load_dotenv()

        # Override client_api_key from environment if not set
        if self.client_api_key is None:
            self.client_api_key = os.environ.get("CLIENT_API_KEY")

        # Override s3_bucket from environment if not set
        if self.s3_bucket is None:
            self.s3_bucket = os.environ.get("ZORK_S3_BUCKET")

    @classmethod
    def from_toml(cls, config_file: Optional[Path] = None) -> "GameConfiguration":
        """
        Create GameConfiguration by loading directly from pyproject.toml.

        Args:
            config_file: Path to TOML file (defaults to pyproject.toml)

        Returns:
            GameConfiguration instance loaded from TOML

        Raises:
            FileNotFoundError: If config file doesn't exist
            KeyError: If required configuration sections are missing
        """
        config_file = config_file or Path("pyproject.toml")

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "rb") as f:
            toml_data = tomllib.load(f)

        try:
            zorkgpt_config = toml_data["tool"]["zorkgpt"]
        except KeyError:
            raise KeyError("Missing [tool.zorkgpt] section in pyproject.toml")

        # Extract configuration sections with validation
        llm_config = zorkgpt_config.get("llm", {})
        orchestrator_config = zorkgpt_config.get("orchestrator", {})
        files_config = zorkgpt_config.get("files", {})
        gameplay_config = zorkgpt_config.get("gameplay", {})
        aws_config = zorkgpt_config.get("aws", {})

        def require_key(config_dict: dict, key: str, section: str) -> any:
            """Helper to get required config values with clear error messages."""
            if key not in config_dict:
                raise KeyError(
                    f"Missing required key '{key}' in [tool.zorkgpt.{section}] section"
                )
            return config_dict[key]

        return cls(
            # Core game settings
            max_turns_per_episode=require_key(
                orchestrator_config, "max_turns_per_episode", "orchestrator"
            ),
            turn_delay_seconds=require_key(
                gameplay_config, "turn_delay_seconds", "gameplay"
            ),
            # File paths
            episode_log_file=require_key(files_config, "episode_log_file", "files"),
            json_log_file=require_key(files_config, "json_log_file", "files"),
            state_export_file=require_key(files_config, "state_export_file", "files"),
            zork_game_workdir=require_key(
                gameplay_config, "zork_game_workdir", "gameplay"
            ),
            game_file_path=files_config.get("game_file_path", "infrastructure/zork.z5"),
            # LLM client settings
            client_base_url=require_key(llm_config, "client_base_url", "llm"),
            # Model specifications
            agent_model=require_key(llm_config, "agent_model", "llm"),
            critic_model=require_key(llm_config, "critic_model", "llm"),
            info_ext_model=require_key(llm_config, "info_ext_model", "llm"),
            analysis_model=require_key(llm_config, "analysis_model", "llm"),
            # Per-model base URLs (optional)
            agent_base_url=llm_config.get("agent_base_url"),
            info_ext_base_url=llm_config.get("info_ext_base_url"),
            critic_base_url=llm_config.get("critic_base_url"),
            analysis_base_url=llm_config.get("analysis_base_url"),
            # Update intervals
            knowledge_update_interval=require_key(
                orchestrator_config, "knowledge_update_interval", "orchestrator"
            ),
            map_update_interval=require_key(
                orchestrator_config, "map_update_interval", "orchestrator"
            ),
            objective_update_interval=require_key(
                orchestrator_config, "objective_update_interval", "orchestrator"
            ),
            # Objective refinement
            enable_objective_refinement=require_key(
                orchestrator_config, "enable_objective_refinement", "orchestrator"
            ),
            objective_refinement_interval=require_key(
                orchestrator_config, "objective_refinement_interval", "orchestrator"
            ),
            max_objectives_before_forced_refinement=require_key(
                orchestrator_config,
                "max_objectives_before_forced_refinement",
                "orchestrator",
            ),
            refined_objectives_target_count=require_key(
                orchestrator_config, "refined_objectives_target_count", "orchestrator"
            ),
            # Context management
            max_context_tokens=require_key(
                orchestrator_config, "max_context_tokens", "orchestrator"
            ),
            context_overflow_threshold=require_key(
                orchestrator_config, "context_overflow_threshold", "orchestrator"
            ),
            # State export
            enable_state_export=require_key(
                orchestrator_config, "enable_state_export", "orchestrator"
            ),
            s3_key_prefix=require_key(aws_config, "s3_key_prefix", "aws"),
            # Gameplay settings
            critic_rejection_threshold=require_key(
                gameplay_config, "critic_rejection_threshold", "gameplay"
            ),
        )

    def get_effective_api_key(self) -> Optional[str]:
        """Get the effective API key from environment or instance."""
        return self.client_api_key

    def get_llm_base_url_for_model(self, model_type: str) -> str:
        """
        Get the effective base URL for a specific model type.

        Args:
            model_type: One of 'agent', 'critic', 'info_ext', 'analysis'

        Returns:
            Base URL for the model type
        """
        base_url_map = {
            "agent": self.agent_base_url,
            "critic": self.critic_base_url,
            "info_ext": self.info_ext_base_url,
            "analysis": self.analysis_base_url,
        }

        # Return model-specific URL if available, otherwise fall back to client_base_url
        return base_url_map.get(model_type) or self.client_base_url
