# ABOUTME: Walkthrough testing infrastructure for deterministic Zork I testing
# ABOUTME: Provides access to Jericho's built-in walkthrough and replay functionality

"""
Walkthrough testing infrastructure for Zork I.

This module provides utilities for accessing and replaying the canonical Jericho
walkthrough for Zork I. It supports both full walkthrough retrieval and partial
sequences for targeted testing scenarios.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from jericho import FrotzEnv


# Use relative path from this file's location to find the game file
GAME_FILE_PATH = str(Path(__file__).parent.parent.parent / "infrastructure" / "zork.z5")


@lru_cache(maxsize=1)
def get_zork1_walkthrough() -> List[str]:
    """
    Returns the complete Zork I walkthrough from Jericho.

    This function creates a fresh FrotzEnv instance and retrieves the
    built-in walkthrough that comes with Jericho. The walkthrough contains
    all actions needed to complete Zork I.

    The result is cached since the walkthrough is immutable and expensive to retrieve.

    Returns:
        List of action strings representing the complete walkthrough.
        Each string is a valid game command that can be executed via env.step().

    Raises:
        FileNotFoundError: If the zork1.z5 game file is not found at the expected path.
        RuntimeError: If Jericho fails to load the game or retrieve the walkthrough.

    Example:
        >>> walkthrough = get_zork1_walkthrough()
        >>> print(f"Walkthrough has {len(walkthrough)} steps")
        >>> print(f"First move: {walkthrough[0]}")
    """
    if not Path(GAME_FILE_PATH).exists():
        raise FileNotFoundError(
            f"Game file not found at {GAME_FILE_PATH}. "
            "Please ensure zork.z5 is in the infrastructure directory."
        )

    env = None
    try:
        env = FrotzEnv(GAME_FILE_PATH)
        walkthrough = env.get_walkthrough()

        if not walkthrough:
            raise RuntimeError("Walkthrough retrieval returned empty list")

        return walkthrough
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve walkthrough from Jericho: {e}") from e
    finally:
        if env is not None:
            env.close()


def get_walkthrough_slice(start: int = 0, end: Optional[int] = None) -> List[str]:
    """
    Returns a slice of the Zork I walkthrough for targeted testing.

    This is useful for testing specific game sequences without executing
    the entire walkthrough. Slicing follows Python's standard slice semantics.

    Args:
        start: Starting index (inclusive). Defaults to 0 (beginning of walkthrough).
        end: Ending index (exclusive). If None, returns all steps from start to end.

    Returns:
        List of action strings representing the requested walkthrough slice.

    Raises:
        FileNotFoundError: If the zork1.z5 game file is not found.
        RuntimeError: If Jericho fails to retrieve the walkthrough.
        ValueError: If start is negative or start > end.

    Example:
        >>> # Get first 10 steps
        >>> early_game = get_walkthrough_slice(0, 10)
        >>>
        >>> # Get steps 20-30
        >>> mid_game = get_walkthrough_slice(20, 30)
        >>>
        >>> # Get everything from step 50 onwards
        >>> late_game = get_walkthrough_slice(50)
    """
    if start < 0:
        raise ValueError(f"Start index must be non-negative, got {start}")

    if end is not None and end < start:
        raise ValueError(f"End index ({end}) must be >= start index ({start})")

    walkthrough = get_zork1_walkthrough()

    return walkthrough[start:end]


def get_walkthrough_until_lamp() -> List[str]:
    """
    Returns walkthrough actions until the lamp is acquired.

    This provides a commonly-needed test sequence for early game testing.
    The lamp is typically acquired within the first ~15 actions of the
    optimal walkthrough and is essential for exploring dark areas.

    Returns:
        List of action strings from game start until lamp acquisition.
        Currently returns the first 15 steps of the walkthrough.

    Raises:
        FileNotFoundError: If the zork1.z5 game file is not found.
        RuntimeError: If Jericho fails to retrieve the walkthrough.

    Example:
        >>> actions = get_walkthrough_until_lamp()
        >>> # Use these actions to test early game mechanics
        >>> for action in actions:
        ...     observation, reward, done, info = env.step(action)

    Note:
        The slice endpoint (15) is based on the canonical Jericho walkthrough
        for Zork I. This may need adjustment if the walkthrough changes.
    """
    return get_walkthrough_slice(0, 15)


def get_walkthrough_dark_sequence() -> List[str]:
    """
    Returns a walkthrough sequence that navigates dark areas.

    This provides a test sequence for dark area navigation, which is a
    critical game mechanic in Zork I. The sequence assumes the lamp has
    already been acquired in prior steps.

    Returns:
        List of action strings for navigating dark areas.
        Currently returns steps 15-30 of the walkthrough.

    Raises:
        FileNotFoundError: If the zork1.z5 game file is not found.
        RuntimeError: If Jericho fails to retrieve the walkthrough.

    Example:
        >>> # First get lamp
        >>> lamp_actions = get_walkthrough_until_lamp()
        >>> for action in lamp_actions:
        ...     env.step(action)
        >>>
        >>> # Then navigate dark areas
        >>> dark_actions = get_walkthrough_dark_sequence()
        >>> for action in dark_actions:
        ...     env.step(action)

    Note:
        The slice range (15:30) is based on the canonical Jericho walkthrough.
        These steps typically include navigation through the cellar and other
        dark locations. May need adjustment if walkthrough changes.
    """
    return get_walkthrough_slice(15, 30)


def replay_walkthrough(
    env: FrotzEnv,
    actions: List[str]
) -> List[Tuple[str, int, bool, Dict[str, Any]]]:
    """
    Replays a sequence of walkthrough actions through a Jericho environment.

    This function executes each action in sequence and captures the results.
    It does NOT reset the environment - actions are executed from whatever
    state the environment is currently in.

    Args:
        env: Initialized FrotzEnv instance (must be already started/reset).
        actions: List of action strings to execute sequentially.

    Returns:
        List of tuples, one per action, each containing:
            - observation (str): Game text response
            - score (int): Current score after the action
            - done (bool): Whether the game has ended
            - info (dict): Additional metadata from Jericho

    Raises:
        ValueError: If env is None or actions is empty.
        RuntimeError: If any action execution fails.

    Example:
        >>> from jericho import FrotzEnv
        >>>
        >>> # Create and initialize environment
        >>> env = FrotzEnv("zork1.z5")
        >>> initial_obs, info = env.reset()
        >>>
        >>> # Get some actions
        >>> actions = get_walkthrough_until_lamp()
        >>>
        >>> # Replay them
        >>> results = replay_walkthrough(env, actions)
        >>>
        >>> # Analyze results
        >>> for i, (obs, score, done, info) in enumerate(results):
        ...     print(f"Step {i}: Score={score}, Done={done}")
        ...     if done:
        ...         print("Game ended!")
        ...         break

    Note:
        - The environment must be initialized before calling this function
        - Each action is executed with env.step(action)
        - The function does not handle environment cleanup (call env.close() separately)
        - If 'done' becomes True, subsequent actions will still be attempted
    """
    if env is None:
        raise ValueError("Environment cannot be None")

    if not actions:
        raise ValueError("Actions list cannot be empty")

    results = []

    for i, action in enumerate(actions):
        try:
            observation, reward, done, info = env.step(action)

            # Note: reward from env.step() is the score delta, not absolute score
            # We need to get the absolute score from the environment
            current_score = env.get_score()

            results.append((observation, current_score, done, info))

            if done:
                # Game ended - stop executing remaining actions
                break

        except Exception as e:
            raise RuntimeError(
                f"Failed to execute action {i} ('{action}'): {e}"
            ) from e

    return results
