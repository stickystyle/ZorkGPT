"""
ABOUTME: Data models for ZorkGPT memory system - Memory, MemoryStatus, and synthesis response types.
ABOUTME: Defines the core data structures used by SimpleMemoryManager for memory persistence and LLM synthesis.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Set
from pydantic import BaseModel, Field, model_validator


# Type alias for valid status values
MemoryStatusType = Literal["ACTIVE", "TENTATIVE", "SUPERSEDED"]

class MemoryStatus:
    """
    Memory status constants.

    ACTIVE: Confirmed reliable memory (default)
    TENTATIVE: Appears true but may be invalidated by future evidence
    SUPERSEDED: Proven wrong and replaced by newer memory
    """
    ACTIVE: MemoryStatusType = "ACTIVE"
    TENTATIVE: MemoryStatusType = "TENTATIVE"
    SUPERSEDED: MemoryStatusType = "SUPERSEDED"

# Invalidation marker (not a status, used in superseded_by field)
INVALIDATION_MARKER: str = "INVALIDATED"  # Used in superseded_by when memory invalidated without replacement


@dataclass
class Memory:
    """
    Represents a single location memory entry.

    Usage patterns:
    1. Active/Tentative memories (not superseded):
       - status = ACTIVE or TENTATIVE
       - superseded_by = None, superseded_at_turn = None, invalidation_reason = None

    2. Supersession (memory replaced by better version):
       - status = SUPERSEDED
       - superseded_by = <new_memory_title>
       - superseded_at_turn = <turn>
       - invalidation_reason = None

    3. Standalone invalidation (memory proven wrong, no replacement):
       - status = SUPERSEDED
       - superseded_by = INVALIDATION_MARKER
       - superseded_at_turn = <turn>
       - invalidation_reason = <explanation>

    Persistence levels:
    - "core": Fundamental game mechanics (never expire)
    - "permanent": Validated successful strategies (long-lasting)
    - "ephemeral": Temporary situational insights (expire quickly)
    """
    category: str              # SUCCESS, FAILURE, DISCOVERY, DANGER, NOTE
    title: str                 # Short title of the memory
    episode: int               # Episode number
    turns: str                 # Turn range (e.g., "23-24" or "23")
    score_change: Optional[int]  # Score change (+5, +0, None if not specified)
    text: str                  # 1-2 sentence synthesized insight
    persistence: str           # "core" | "permanent" | "ephemeral" - REQUIRED, no default
    status: MemoryStatusType = MemoryStatus.ACTIVE  # Memory status
    superseded_by: Optional[str] = None  # Title of superseding memory or INVALIDATION_MARKER
    superseded_at_turn: Optional[int] = None  # Turn when superseded or invalidated
    invalidation_reason: Optional[str] = None  # Reason for standalone invalidation (only with INVALIDATION_MARKER)

    def __post_init__(self):
        """Validate persistence field after initialization."""
        valid_values = ["core", "permanent", "ephemeral"]
        if self.persistence not in valid_values:
            raise ValueError(
                f"Invalid persistence value: '{self.persistence}'. "
                f"Must be one of: {', '.join(valid_values)}"
            )


class MemorySynthesisResponse(BaseModel):
    """LLM response for memory synthesis."""
    model_config = {"strict": True}

    should_remember: bool
    category: Optional[str] = None  # Only required if should_remember=true
    memory_title: Optional[str] = None  # Only required if should_remember=true
    memory_text: Optional[str] = None  # Only required if should_remember=true
    persistence: Optional[str] = None  # "core" | "permanent" | "ephemeral", required if should_remember=true
    status: MemoryStatusType = Field(default=MemoryStatus.ACTIVE)  # Default to ACTIVE
    supersedes_memory_titles: Set[str] = Field(
        default_factory=set,
        max_length=3,  # CRITICAL: Max 3 items to prevent hallucination
        description="Titles to mark as superseded (MAX 3 items)"
    )
    invalidate_memory_titles: Set[str] = Field(
        default_factory=set,
        max_length=3,  # CRITICAL: Max 3 items to prevent hallucination
        description="Titles to invalidate without replacement (MAX 3 items)"
    )
    invalidation_reason: Optional[str] = None  # Reason for invalidation
    reasoning: str = Field(
        default="",
        max_length=500,  # Limit reasoning to prevent token bloat
        description="Brief reasoning for debugging (max 500 chars)"
    )

    @model_validator(mode='after')
    def validate_remember_fields(self) -> 'MemorySynthesisResponse':
        """Ensure required fields are present when should_remember=true."""
        if self.should_remember:
            if not self.category:
                raise ValueError("category is required when should_remember=true")
            if not self.memory_title:
                raise ValueError("memory_title is required when should_remember=true")
            if not self.memory_text:
                raise ValueError("memory_text is required when should_remember=true")

            # Validate persistence field
            valid_persistence = ["core", "permanent", "ephemeral"]
            if not self.persistence:
                raise ValueError(
                    f"persistence required when should_remember=true. "
                    f"Must be one of {valid_persistence}"
                )

            # Validate persistence value
            if self.persistence not in valid_persistence:
                raise ValueError(
                    f"persistence must be one of {valid_persistence}, got: {self.persistence}"
                )

        # Validate invalidation fields
        if self.invalidate_memory_titles:
            if not self.invalidation_reason or not self.invalidation_reason.strip():
                raise ValueError("invalidation_reason must be non-empty when invalidate_memory_titles is not empty")

        # Validate mutual exclusivity of supersession and invalidation for same titles
        if self.supersedes_memory_titles and self.invalidate_memory_titles:
            overlap = self.supersedes_memory_titles & self.invalidate_memory_titles
            if overlap:
                raise ValueError(
                    f"Memory titles cannot be both superseded and invalidated: {overlap}"
                )

        return self
