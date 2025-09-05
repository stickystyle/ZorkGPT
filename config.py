"""
Configuration module for ZorkGPT.

Loads application settings from pyproject.toml while keeping sensitive
data like API keys in environment variables.
"""

import os
import tomllib
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv


class LLMConfig(BaseModel):
    """LLM configuration settings."""

    client_base_url: str = "http://localhost:1234"

    # Model configurations
    agent_model: str = "qwen3-30b-a3b-mlx"
    info_ext_model: str = "qwen3-30b-a3b-mlx"
    critic_model: str = "qwen3-30b-a3b-mlx"
    analysis_model: str = (
        "meta-llama/llama-4-scout"  # Should be configured in pyproject.toml
    )

    # Per-model base URLs (optional, falls back to client_base_url if not specified)
    agent_base_url: Optional[str] = None
    info_ext_base_url: Optional[str] = None
    critic_base_url: Optional[str] = None
    analysis_base_url: Optional[str] = None

    def get_base_url_for_model(self, model_type: str) -> str:
        """
        Get the appropriate base URL for a specific model type.

        Args:
            model_type: One of 'agent', 'info_ext', 'critic', 'analysis'

        Returns:
            The base URL to use for that model type
        """
        base_url_map = {
            "agent": self.agent_base_url,
            "info_ext": self.info_ext_base_url,
            "critic": self.critic_base_url,
            "analysis": self.analysis_base_url,
        }

        # Return model-specific URL if available, otherwise fall back to client_base_url
        return base_url_map.get(model_type) or self.client_base_url


class AgentSamplingConfig(BaseModel):
    """Sampling parameters for the agent."""

    temperature: float = 0.5
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: Optional[int] = None


class CriticSamplingConfig(BaseModel):
    """Sampling parameters for the critic."""

    temperature: float = 0.2
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: int = 100


class ExtractorSamplingConfig(BaseModel):
    """Sampling parameters for the information extractor."""

    temperature: float = 0.1
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: int = 300


class AnalysisSamplingConfig(BaseModel):
    """Sampling parameters for the analysis model used in knowledge generation."""

    temperature: float = 0.3
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None


class GameplayConfig(BaseModel):
    """Gameplay configuration settings."""

    turn_delay_seconds: float = 0.0
    turn_window_size: int = 100
    min_knowledge_quality: float = 6.0
    critic_rejection_threshold: float = -0.05
    # Exit pruning configuration
    enable_exit_pruning: bool = True
    exit_failure_threshold: int = 3
    # Knowledge base condensation configuration
    enable_knowledge_condensation: bool = True
    knowledge_condensation_threshold: int = 15000
    # Save/restore configuration
    zork_save_filename_template: str = "zorkgpt_save_{timestamp}.sav"
    zork_game_workdir: str = "game_files"
    save_signal_filename: str = ".SAVE_REQUESTED_BY_SYSTEM"


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    enable_prompt_logging: bool = False


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration settings."""

    max_turns_per_episode: int = 200
    knowledge_update_interval: int = 100  # Every 100 turns
    map_update_interval: int = 25  # Every 25 turns, more frequent than full knowledge
    objective_update_interval: int = 20  # Every 20 turns for objective discovery
    enable_state_export: bool = True
    # Context management settings - adjusted for 40K token models
    max_context_tokens: int = 150000  # Max context tokens for LLM calls
    context_overflow_threshold: float = (
        0.8  # Trigger summarization at 80% of max_context_tokens
    )
    enable_objective_refinement: bool = True
    objective_refinement_interval: int = 75
    max_objectives_before_forced_refinement: int = 20
    refined_objectives_target_count: int = 10
    # Inter-episode synthesis configuration
    enable_inter_episode_synthesis: bool = True
    persistent_wisdom_file: str = "persistent_wisdom.md"


class FilesConfig(BaseModel):
    """File configuration settings."""

    episode_log_file: str = "zork_episode_log.txt"
    json_log_file: str = "zork_episode_log.jsonl"
    state_export_file: str = "current_state.json"


class AWSConfig(BaseModel):
    """AWS configuration settings."""

    s3_bucket: Optional[str] = None
    s3_key_prefix: str = "zorkgpt/"

    def __init__(self, **data):
        """Initialize with environment variable for s3_bucket."""
        # Override s3_bucket with environment variable if available
        env_s3_bucket = os.environ.get("ZORK_S3_BUCKET")
        if env_s3_bucket:
            data["s3_bucket"] = env_s3_bucket
        super().__init__(**data)


class RetryConfig(BaseModel):
    """Retry and exponential backoff configuration settings."""

    max_retries: int = 5
    initial_delay: float = 1.0  # Initial retry delay in seconds
    max_delay: float = 60.0  # Maximum retry delay in seconds
    exponential_base: float = 2.0  # Multiplier for exponential backoff
    jitter_factor: float = 0.1  # Random jitter to prevent thundering herd (0.0 to 1.0)
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True  # 5xx errors
    timeout_seconds: float = 120.0
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = (
        10  # Number of failures before opening circuit
    )
    circuit_breaker_recovery_timeout: float = (
        300.0  # Seconds before trying to close circuit
    )
    circuit_breaker_success_threshold: int = (
        3  # Consecutive successes needed to close circuit
    )


class ZorkGPTConfig(BaseModel):
    """Complete ZorkGPT configuration."""

    llm: LLMConfig = LLMConfig()
    retry: RetryConfig = RetryConfig()
    agent_sampling: AgentSamplingConfig = AgentSamplingConfig()
    critic_sampling: CriticSamplingConfig = CriticSamplingConfig()
    extractor_sampling: ExtractorSamplingConfig = ExtractorSamplingConfig()
    analysis_sampling: AnalysisSamplingConfig = AnalysisSamplingConfig()
    gameplay: GameplayConfig = GameplayConfig()
    logging: LoggingConfig = LoggingConfig()
    orchestrator: OrchestratorConfig = OrchestratorConfig()
    files: FilesConfig = FilesConfig()
    aws: AWSConfig = AWSConfig()


class ConfigLoader:
    """Loads configuration from pyproject.toml and environment variables."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("pyproject.toml")
        self._config: Optional[ZorkGPTConfig] = None
        # Load environment variables from .env file if it exists
        load_dotenv()

    def load_config(self) -> ZorkGPTConfig:
        """Load configuration from pyproject.toml."""
        if self._config is not None:
            return self._config

        config_data = {}

        if self.config_file.exists():
            with open(self.config_file, "rb") as f:
                toml_data = tomllib.load(f)
                config_data = toml_data.get("tool", {}).get("zorkgpt", {})

        self._config = ZorkGPTConfig(**config_data)
        return self._config

    def get_client_api_key(self) -> Optional[str]:
        """Get the CLIENT_API_KEY from environment variables."""
        return os.environ.get("CLIENT_API_KEY")


# Global configuration instance
_config_loader = ConfigLoader()


def get_config() -> ZorkGPTConfig:
    """Get the global configuration instance."""
    return _config_loader.load_config()


def get_client_api_key() -> Optional[str]:
    """Get the CLIENT_API_KEY from environment variables."""
    return _config_loader.get_client_api_key()
