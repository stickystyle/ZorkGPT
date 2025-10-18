# ABOUTME: Test fixtures module for deterministic Zork I testing
# ABOUTME: Provides walkthrough data and replay utilities via Jericho

from .walkthrough import (
    get_zork1_walkthrough,
    get_walkthrough_slice,
    get_walkthrough_until_lamp,
    get_walkthrough_dark_sequence,
    replay_walkthrough,
)

__all__ = [
    "get_zork1_walkthrough",
    "get_walkthrough_slice",
    "get_walkthrough_until_lamp",
    "get_walkthrough_dark_sequence",
    "replay_walkthrough",
]
