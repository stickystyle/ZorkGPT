"""
Game configuration management for ZorkGPT orchestration.

This module provides a clean, typed interface to game configuration,
loading directly from pyproject.toml and environment variables.
"""

import tomllib
import warnings
from typing import Optional
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


def _default_retry_config() -> dict:
    """
    Provide default retry configuration values.

    These defaults are used when retry config is not explicitly provided
    (e.g., in tests or when creating GameConfiguration instances directly).
    """
    return {
        "max_retries": 5,
        "initial_delay": 1.0,
        "max_delay": 60.0,
        "exponential_base": 2.0,
        "jitter_factor": 0.1,
        "retry_on_timeout": True,
        "retry_on_rate_limit": True,
        "retry_on_server_error": True,
        "timeout_seconds": 120.0,
        "circuit_breaker_enabled": True,
        "circuit_breaker_failure_threshold": 10,
        "circuit_breaker_recovery_timeout": 300.0,
        "circuit_breaker_success_threshold": 3,
    }


class GameConfiguration(BaseSettings):
    """
    Typed configuration object for ZorkGPT orchestration.

    Loads configuration directly from pyproject.toml and environment variables.
    All required values must be present in TOML or the application will crash.
    """

    # Core game settings
    max_turns_per_episode: int = Field(
        description="Maximum number of turns allowed per episode"
    )
    turn_delay_seconds: float = Field(
        default=3.0, description="Delay in seconds between turns"
    )
    turn_window_size: int = Field(
        default=100, description="Number of recent turns to consider for context"
    )

    # File paths
    episode_log_file: str = Field(
        default="zork_episode_log.txt", description="Path to episode log file"
    )
    json_log_file: str = Field(
        default="zork_episode_log.jsonl", description="Path to JSON log file"
    )
    state_export_file: str = Field(
        default="current_state.json", description="Path to state export file"
    )
    map_state_file: str = Field(
        default="map_state.json", description="Path to map state file"
    )
    knowledge_file: str = Field(
        default="knowledgebase.md", description="Path to knowledge base file"
    )
    zork_game_workdir: str = Field(
        default="game_files", description="Working directory for game files"
    )
    game_file_path: str = Field(
        default="jericho-game-suite/zork1.z5", description="Path to game ROM file"
    )

    # LLM client settings
    client_base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="Base URL for LLM client"
    )
    client_api_key: Optional[str] = Field(
        default=None, description="API key for LLM client"
    )

    # Model specifications
    agent_model: str = Field(
        default="deepseek/deepseek-r1-0528", description="Model name for agent"
    )
    critic_model: str = Field(
        default="google/gemma-3-27b-it", description="Model name for critic"
    )
    info_ext_model: str = Field(
        default="google/gemma-3-12b-it", description="Model name for information extractor"
    )
    analysis_model: str = Field(
        default="deepseek/deepseek-r1-0528", description="Model name for analysis"
    )
    memory_model: str = Field(
        default="qwen/qwq-32b", description="Model name for memory synthesis"
    )

    # Per-model base URLs (optional)
    agent_base_url: Optional[str] = Field(
        default=None, description="Optional base URL for agent model"
    )
    info_ext_base_url: Optional[str] = Field(
        default=None, description="Optional base URL for info extractor model"
    )
    critic_base_url: Optional[str] = Field(
        default=None, description="Optional base URL for critic model"
    )
    analysis_base_url: Optional[str] = Field(
        default=None, description="Optional base URL for analysis model"
    )
    memory_base_url: Optional[str] = Field(
        default=None, description="Optional base URL for memory model"
    )

    # Retry configuration
    retry: dict = Field(
        default_factory=_default_retry_config,
        description="Retry and exponential backoff configuration",
    )

    # Update intervals
    knowledge_update_interval: int = Field(
        default=100, description="Interval for knowledge base updates"
    )
    objective_update_interval: int = Field(
        default=25, description="Interval for objective discovery updates"
    )

    # Objective refinement
    enable_objective_refinement: bool = Field(
        default=True, description="Enable objective refinement"
    )
    objective_refinement_interval: int = Field(
        default=50, description="Interval for objective refinement"
    )
    max_objectives_before_forced_refinement: int = Field(
        default=15, description="Maximum objectives before forced refinement"
    )
    refined_objectives_target_count: int = Field(
        default=8, description="Target count for refined objectives"
    )

    # Objective completion checking
    enable_objective_completion_llm_check: bool = Field(
        default=True, description="Enable LLM-based objective completion validation"
    )
    completion_check_interval: int = Field(
        default=1, description="Check completion every N turns (1 = every turn)"
    )
    completion_history_window: int = Field(
        default=3, description="Number of recent turns to include in completion context"
    )
    completion_include_memories: bool = Field(
        default=True, description="Include location-specific memories in completion context"
    )

    # State export
    enable_state_export: bool = Field(
        default=True, description="Enable state export functionality"
    )
    s3_bucket: Optional[str] = Field(
        default=None, description="S3 bucket for state export"
    )
    s3_key_prefix: str = Field(
        default="zorkgpt/", description="S3 key prefix for exports"
    )

    # Gameplay settings
    critic_rejection_threshold: float = Field(
        default=-0.2, description="Threshold for critic to reject actions"
    )
    min_knowledge_quality: float = Field(
        default=6.0, description="Minimum quality score for knowledge entries"
    )
    enable_exit_pruning: bool = Field(
        default=True, description="Enable pruning of failed exits from map"
    )
    exit_failure_threshold: int = Field(
        default=2, description="Number of failures before pruning an exit"
    )
    zork_save_filename_template: str = Field(
        default="zorkgpt_save_{timestamp}", description="Template for save file names"
    )

    # Orchestrator settings
    enable_inter_episode_synthesis: bool = Field(
        default=True, description="Enable inter-episode wisdom synthesis"
    )

    # Simple memory settings
    simple_memory_file: str = Field(
        default="Memories.md", description="Path to simple memory file"
    )
    simple_memory_max_shown: int = Field(
        default=10, description="Maximum number of memories to show"
    )

    # Room description settings
    room_description_age_window: int = Field(
        default=10, description="Number of turns before room descriptions age out of context"
    )

    # Critic configuration
    enable_critic: bool = Field(
        default=True,
        description="Enable LLM-based critic evaluation (object tree validation always runs)"
    )

    # Progress Velocity Detection
    max_turns_stuck: int = Field(
        default=40,
        description="Maximum turns without score change before terminating episode",
        ge=1,  # Must be at least 1
        le=1000  # Sanity check upper bound
    )
    stuck_check_interval: int = Field(
        default=10,
        description="Check for stuck behavior every N turns",
        ge=1,  # Must be at least 1
        le=100  # Sanity check
    )
    enable_objective_based_progress: bool = Field(
        default=True,
        description="Enable objective completion as a form of progress (alongside score changes)"
    )

    # Location Revisit Penalty (Anti-Loop)
    enable_location_penalty: bool = Field(
        default=True,
        description="Enable location revisit penalty to discourage loops"
    )
    location_revisit_penalty: float = Field(
        default=-0.2,
        description="Penalty applied per recent revisit to same location",
        ge=-1.0,  # Can't reduce score below 0
        le=0.0    # Must be negative (it's a penalty)
    )
    location_revisit_window: int = Field(
        default=5,
        description="Number of recent locations to check for revisits",
        ge=2,  # Need at least 2 to detect revisit
        le=20  # Max history window
    )

    # Exploration Context Guidance
    enable_exploration_hints: bool = Field(
        default=True,
        description="Enable context hints for action novelty and unexplored exits"
    )
    action_novelty_window: int = Field(
        default=15,
        description="Number of recent actions to check for novelty",
        ge=5,
        le=50
    )

    # Stuck Countdown Warnings
    enable_stuck_warnings: bool = Field(
        default=True,
        description="Enable countdown warnings when score stagnates"
    )
    stuck_warning_threshold: int = Field(
        default=20,
        description="Turns stuck before warnings start (should be < max_turns_stuck)",
        ge=5,
        le=100
    )

    # MCP Configuration
    mcp_enabled: bool = Field(
        default=False, description="Enable MCP tool calling integration"
    )
    mcp_config_file: str = Field(
        default="mcp_config.json", description="Path to MCP server configuration file"
    )
    mcp_max_tool_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum tool-calling iterations per turn",
    )
    mcp_tool_call_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout for individual tool calls in seconds",
    )
    mcp_server_startup_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Timeout for MCP server startup in seconds",
    )
    mcp_force_tool_support: bool = Field(
        default=False,
        description="Override model compatibility auto-detection for tool calling",
    )

    # Sampling parameters (loaded from TOML)
    agent_sampling: dict = Field(
        default_factory=dict, description="Sampling parameters for agent"
    )
    critic_sampling: dict = Field(
        default_factory=dict, description="Sampling parameters for critic"
    )
    extractor_sampling: dict = Field(
        default_factory=dict, description="Sampling parameters for extractor"
    )
    analysis_sampling: dict = Field(
        default_factory=dict, description="Sampling parameters for analysis"
    )
    memory_sampling: dict = Field(
        default_factory=dict, description="Sampling parameters for memory"
    )

    model_config = SettingsConfigDict(
        env_prefix="ZORKGPT_",
        env_file=None,  # Don't auto-load .env to avoid validation errors from unrelated vars
        case_sensitive=False,
        extra="forbid",  # Catch typos in config early
    )

    @model_validator(mode='after')
    def validate_stuck_detection(self) -> 'GameConfiguration':
        """Validate stuck detection configuration."""
        if self.max_turns_stuck < self.stuck_check_interval:
            raise ValueError(
                f"max_turns_stuck ({self.max_turns_stuck}) must be >= "
                f"stuck_check_interval ({self.stuck_check_interval})"
            )
        return self

    @model_validator(mode='after')
    def validate_warning_threshold(self) -> 'GameConfiguration':
        """Ensure warning threshold is less than termination threshold."""
        if self.stuck_warning_threshold >= self.max_turns_stuck:
            raise ValueError(
                f"stuck_warning_threshold ({self.stuck_warning_threshold}) must be < "
                f"max_turns_stuck ({self.max_turns_stuck})"
            )
        return self

    def model_post_init(self, __context) -> None:
        """Ensure game working directory exists and handle legacy env vars."""
        import os

        # Backward compatibility: Fall back to old env var names
        # Legacy infrastructure uses CLIENT_API_KEY and ZORK_S3_BUCKET
        if self.client_api_key is None:
            self.client_api_key = os.environ.get("CLIENT_API_KEY")

        if self.s3_bucket is None:
            self.s3_bucket = os.environ.get("ZORK_S3_BUCKET")

        # Create workdir
        workdir = Path(self.zork_game_workdir)
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)

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
        # Load .env file if it exists to populate environment variables
        load_dotenv()

        config_file = config_file or Path("pyproject.toml")

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "rb") as f:
            toml_data = tomllib.load(f)

        try:
            zorkgpt_config = toml_data["tool"]["zorkgpt"]
        except KeyError:
            raise KeyError("Missing [tool.zorkgpt] section in pyproject.toml")

        # Extract configuration sections
        llm_config = zorkgpt_config.get("llm", {})
        orchestrator_config = zorkgpt_config.get("orchestrator", {})
        files_config = zorkgpt_config.get("files", {})
        gameplay_config = zorkgpt_config.get("gameplay", {})
        aws_config = zorkgpt_config.get("aws", {})
        simple_memory_config = zorkgpt_config.get("simple_memory", {})
        retry_config = zorkgpt_config.get("retry", {})
        objective_completion_config = zorkgpt_config.get("objective_completion", {})
        loop_break_config = zorkgpt_config.get("loop_break", {})
        mcp_config = zorkgpt_config.get("mcp", {})

        # Extract sampling parameter sections
        agent_sampling = zorkgpt_config.get("agent_sampling", {})
        critic_sampling = zorkgpt_config.get("critic_sampling", {})
        extractor_sampling = zorkgpt_config.get("extractor_sampling", {})
        analysis_sampling = zorkgpt_config.get("analysis_sampling", {})
        memory_sampling = zorkgpt_config.get("memory_sampling", {})

        # Build flat configuration dictionary for Pydantic validation
        config_dict = {
            # Core game settings
            "max_turns_per_episode": orchestrator_config.get("max_turns_per_episode"),
            "turn_delay_seconds": gameplay_config.get("turn_delay_seconds"),
            "turn_window_size": gameplay_config.get("turn_window_size"),
            # File paths
            "episode_log_file": files_config.get("episode_log_file"),
            "json_log_file": files_config.get("json_log_file"),
            "state_export_file": files_config.get("state_export_file"),
            "map_state_file": files_config.get("map_state_file"),
            "knowledge_file": files_config.get("knowledge_file"),
            "zork_game_workdir": gameplay_config.get("zork_game_workdir"),
            "game_file_path": files_config.get("game_file_path"),
            # LLM client settings
            "client_base_url": llm_config.get("client_base_url"),
            # Model specifications
            "agent_model": llm_config.get("agent_model"),
            "critic_model": llm_config.get("critic_model"),
            "info_ext_model": llm_config.get("info_ext_model"),
            "analysis_model": llm_config.get("analysis_model"),
            "memory_model": llm_config.get("memory_model"),
            # Per-model base URLs (optional)
            "agent_base_url": llm_config.get("agent_base_url"),
            "info_ext_base_url": llm_config.get("info_ext_base_url"),
            "critic_base_url": llm_config.get("critic_base_url"),
            "analysis_base_url": llm_config.get("analysis_base_url"),
            "memory_base_url": llm_config.get("memory_base_url"),
            # Retry configuration
            "retry": retry_config,
            # Update intervals
            "knowledge_update_interval": orchestrator_config.get("knowledge_update_interval"),
            "objective_update_interval": orchestrator_config.get("objective_update_interval"),
            # Objective refinement
            "enable_objective_refinement": orchestrator_config.get("enable_objective_refinement"),
            "objective_refinement_interval": orchestrator_config.get("objective_refinement_interval"),
            "max_objectives_before_forced_refinement": orchestrator_config.get("max_objectives_before_forced_refinement"),
            "refined_objectives_target_count": orchestrator_config.get("refined_objectives_target_count"),
            # Objective completion checking
            "enable_objective_completion_llm_check": objective_completion_config.get("enable_llm_check", True),
            "completion_check_interval": objective_completion_config.get("check_interval", 1),
            "completion_history_window": objective_completion_config.get("history_window", 3),
            "completion_include_memories": objective_completion_config.get("include_memories", True),
            # State export
            "enable_state_export": orchestrator_config.get("enable_state_export"),
            "s3_key_prefix": aws_config.get("s3_key_prefix"),
            # Gameplay settings
            "critic_rejection_threshold": gameplay_config.get("critic_rejection_threshold"),
            "enable_critic": gameplay_config.get("enable_critic", True),
            "min_knowledge_quality": gameplay_config.get("min_knowledge_quality"),
            "enable_exit_pruning": gameplay_config.get("enable_exit_pruning"),
            "exit_failure_threshold": gameplay_config.get("exit_failure_threshold"),
            "zork_save_filename_template": gameplay_config.get("zork_save_filename_template"),
            # Orchestrator settings
            "enable_inter_episode_synthesis": orchestrator_config.get("enable_inter_episode_synthesis"),
            # Simple memory settings
            "simple_memory_file": simple_memory_config.get("memory_file"),
            "simple_memory_max_shown": simple_memory_config.get("max_memories_shown"),
            # Sampling parameters
            "agent_sampling": agent_sampling,
            "critic_sampling": critic_sampling,
            "extractor_sampling": extractor_sampling,
            "analysis_sampling": analysis_sampling,
            "memory_sampling": memory_sampling,
        }

        # Add optional progress velocity detection settings (only if present in TOML)
        if loop_break_config.get("max_turns_stuck") is not None:
            config_dict["max_turns_stuck"] = loop_break_config.get("max_turns_stuck")
        if loop_break_config.get("stuck_check_interval") is not None:
            config_dict["stuck_check_interval"] = loop_break_config.get("stuck_check_interval")
        if loop_break_config.get("enable_objective_based_progress") is not None:
            config_dict["enable_objective_based_progress"] = loop_break_config.get("enable_objective_based_progress")

        # Add optional location revisit penalty settings (only if present in TOML)
        if loop_break_config.get("enable_location_penalty") is not None:
            config_dict["enable_location_penalty"] = loop_break_config.get("enable_location_penalty")
        if loop_break_config.get("location_revisit_penalty") is not None:
            config_dict["location_revisit_penalty"] = loop_break_config.get("location_revisit_penalty")
        if loop_break_config.get("location_revisit_window") is not None:
            config_dict["location_revisit_window"] = loop_break_config.get("location_revisit_window")

        # Add optional exploration context guidance settings (only if present in TOML)
        if loop_break_config.get("enable_exploration_hints") is not None:
            config_dict["enable_exploration_hints"] = loop_break_config.get("enable_exploration_hints")
        if loop_break_config.get("action_novelty_window") is not None:
            config_dict["action_novelty_window"] = loop_break_config.get("action_novelty_window")

        # Add optional stuck countdown warning settings (only if present in TOML)
        if loop_break_config.get("enable_stuck_warnings") is not None:
            config_dict["enable_stuck_warnings"] = loop_break_config.get("enable_stuck_warnings")
        if loop_break_config.get("stuck_warning_threshold") is not None:
            config_dict["stuck_warning_threshold"] = loop_break_config.get("stuck_warning_threshold")

        # Add optional MCP settings (only if present in TOML)
        if mcp_config.get("enabled") is not None:
            config_dict["mcp_enabled"] = mcp_config.get("enabled")
        if mcp_config.get("config_file") is not None:
            config_dict["mcp_config_file"] = mcp_config.get("config_file")
        if mcp_config.get("max_tool_iterations") is not None:
            config_dict["mcp_max_tool_iterations"] = mcp_config.get("max_tool_iterations")
        if mcp_config.get("tool_call_timeout_seconds") is not None:
            config_dict["mcp_tool_call_timeout_seconds"] = mcp_config.get("tool_call_timeout_seconds")
        if mcp_config.get("server_startup_timeout_seconds") is not None:
            config_dict["mcp_server_startup_timeout_seconds"] = mcp_config.get("server_startup_timeout_seconds")
        if mcp_config.get("force_tool_support") is not None:
            config_dict["mcp_force_tool_support"] = mcp_config.get("force_tool_support")

        # Use Pydantic's model_validate to create the instance
        return cls.model_validate(config_dict)

    def get_effective_api_key(self) -> Optional[str]:
        """Get the effective API key from environment or instance."""
        return self.client_api_key

    def get_llm_base_url_for_model(self, model_type: str) -> str:
        """
        Get the effective base URL for a specific model type.

        Args:
            model_type: One of 'agent', 'critic', 'info_ext', 'analysis', 'memory'

        Returns:
            Base URL for the model type
        """
        base_url_map = {
            "agent": self.agent_base_url,
            "critic": self.critic_base_url,
            "info_ext": self.info_ext_base_url,
            "analysis": self.analysis_base_url,
            "memory": self.memory_base_url,
        }

        # Return model-specific URL if available, otherwise fall back to client_base_url
        return base_url_map.get(model_type) or self.client_base_url

    def get_memory_history_window(self) -> int:
        """
        Get the number of recent turns to include in memory synthesis context.

        This value controls how many recent actions/reasoning entries are included
        in memory synthesis prompts. Higher values provide more context but may
        use excessive tokens.

        Returns:
            Number of recent turns (validated to be >= 1)

        Warns:
            If value is > 10 (may use excessive context)
        """
        window = self.memory_sampling.get("memory_history_window", 3)

        # Validation: must be at least 1
        if window < 1:
            warnings.warn(
                f"memory_history_window must be >= 1, got {window}. Using default: 3",
                UserWarning,
                stacklevel=2
            )
            return 3

        # Warning: recommended range is 3-8
        if window > 10:
            warnings.warn(
                f"memory_history_window = {window} may use excessive context. "
                f"Recommended range: 3-8 actions.",
                UserWarning,
                stacklevel=2
            )

        return window
