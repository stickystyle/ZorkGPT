# Specification: Objective-Based Progress Tracking for Loop Break System

**Version:** 1.1
**Date:** 2025-11-10
**Status:** Ready for Implementation (Validated & Corrected)

---

## Executive Summary

Enhance the loop break system's progress detection (Phase 1A) to recognize objective completion as valid progress alongside score changes. This reduces false terminations when agents are making meaningful progress (completing objectives) without immediate score gains.

**Current Behavior:** Episode terminates after 40 turns without score change.

**New Behavior:** Episode terminates after 40 turns without (score change OR objective completion).

**Validation Status:** This specification has been validated against 75 real episodes (see `SPEC_VALIDATION.md`) and corrected based on comprehensive codebase research. All variable names, data structures, and implementation details now match the actual ZorkGPT codebase.

---

## Problem Statement

The current loop break system uses score changes as the sole measure of progress. This causes premature terminations when:

1. **Agent completes objectives without score change**: Many exploratory or preparatory objectives (e.g., "explore north to Location 81", "open the window") don't immediately change score but represent meaningful progress.

2. **Agent panics unnecessarily**: Warnings trigger based purely on score, even when the agent is successfully working through objectives.

3. **Progress metric is too narrow**: Score is the ultimate goal, but intermediate achievements (objectives) are also valid indicators that the agent isn't stuck.

**Goal:** Broaden the definition of "progress" to include both score changes and objective completions, giving agents more opportunities to demonstrate forward movement before termination.

**Note:** Current implementation tracks ANY score change (increase or decrease) as progress, as even score penalties represent learning/discovery.

---

## Solution Overview

### Core Logic Change

**Current Progress Definition:**
```
progress = score changed within last 40 turns
```

**New Progress Definition:**
```
progress = (score changed in last 40 turns) OR (objective completed in last 40 turns)
```

**Note:** Current implementation tracks ANY score change (increase or decrease) as progress, resetting the stuck timer when score changes.

### Progress Tracking Flow

1. **Each turn after action execution:**
   - Check if score increased (existing logic)
   - Check if any objective was completed (new logic)
   - If either condition is true → progress was made → reset stuck timer
   - If neither condition is true → increment stuck timer

2. **Termination condition:**
   - If stuck timer reaches 40 turns → terminate episode
   - Same threshold, broader definition of progress

3. **Fallback behavior:**
   - If no objectives exist (empty list, disabled, or all completed) → fall back to score-only tracking (backward compatible)

---

## Detailed Specification

### 1. Progress Calculation

#### Location
`orchestration/zork_orchestrator_v2.py`, Phase 1A (lines 347-380)

#### Algorithm

**Input:**
- `current_turn`: Current turn number
- `_last_score_change_turn`: Turn when score last changed (existing instance variable)
- `completed_objectives`: List of completed objective records from GameState

**Data Structures:**

GameState stores two separate objective lists (`session/game_state.py`, lines 74-75):
- `discovered_objectives: List[str]` - Active goals the agent is pursuing (simple strings)
- `completed_objectives: List[Dict[str, Any]]` - Achievement records with metadata

Each completed objective dict has 6 keys:
```python
{
    "objective": str,              # The objective text
    "completed_turn": int,         # Turn number when completed
    "completion_action": str,      # Action that completed it
    "completion_response": str,    # Game response text
    "completion_location": str,    # Room name where completed
    "completion_score": int        # Score at completion time
}
```

**Note:** Completed objectives are NEVER removed (persist for entire episode). Only `discovered_objectives` (active goals) are removed via staleness (30 turns without progress).

**Steps:**

1. **Calculate score-based progress (window-based):**
   ```python
   # Use existing _last_score_change_turn variable (updated when score changes)
   score_progress = (current_turn - self._last_score_change_turn) < self.config.max_turns_stuck
   ```

2. **Calculate objective-based progress (window-based):**
   ```python
   if self.game_state.completed_objectives:
       last_objective_turn = max(
           obj["completed_turn"] for obj in self.game_state.completed_objectives
       )
       objective_progress = (current_turn - last_objective_turn) < self.config.max_turns_stuck
   else:
       objective_progress = False  # No objectives → fall back to score-only
   ```

3. **Determine overall progress:**
   ```python
   progress_made = score_progress OR objective_progress
   ```

4. **Update stuck timer:**
   ```python
   # Note: Current implementation calculates stuck duration via helper function
   # _get_turns_since_score_change() rather than maintaining explicit counter.
   # New implementation should maintain this approach for consistency.
   if progress_made:
       # Reset: Update _last_score_change_turn if score changed
       if score_changed_this_turn:
           self._last_score_change_turn = current_turn
   ```

#### Edge Cases

| Scenario | Behavior |
|----------|----------|
| No objectives exist (empty `completed_objectives` list) | Fall back to score-only tracking |
| Objectives disabled in config | Fall back to score-only tracking |
| All objectives completed | Completed objectives persist in `completed_objectives` list and continue counting as progress within 40-turn window |
| Objective removed via staleness | Only affects `discovered_objectives` (active goals). Completed objectives are never removed, so staleness doesn't affect progress tracking |
| Objective added by agent mid-episode | No effect on progress tracking until objective is completed and added to `completed_objectives` |

---

### 2. Configuration

#### Feature Flag

**Section:** `[tool.zorkgpt.loop_break]` in `pyproject.toml`

**New Field:**
```toml
[tool.zorkgpt.loop_break]
enable_objective_based_progress = true  # Default: enabled
```

**Field Documentation:**
```toml
# Enable objective completion as a form of progress (alongside score increases)
# When true: Progress = score increase OR objective completion
# When false: Progress = score increase only (legacy behavior)
# Default: true
enable_objective_based_progress = true
```

#### Config Loading

**Location:** `session/game_configuration.py`, lines 220-282

**Add to GameConfiguration dataclass:**
```python
@dataclass
class GameConfiguration:
    # ... existing fields ...

    # Loop break configuration
    enable_objective_based_progress: bool = True
```

**Add to config loading from pyproject.toml:**
```python
loop_break_config = config.get("tool", {}).get("zorkgpt", {}).get("loop_break", {})
self.enable_objective_based_progress = loop_break_config.get(
    "enable_objective_based_progress", True
)
```

---

### 3. Logging and Observability

#### Log Format

**Style:** Combined metrics in single statement (easier to parse and analyze)

**Example Log Statements:**

**Debug logging (each turn):**
```python
self.logger.debug(
    f"Progress check: score_increase={score_progress}, "
    f"objective_completed={objective_progress}, "
    f"progress_made={progress_made}, turns_stuck={turns_stuck}"
)
```

**Info logging (when progress made):**
```python
self.logger.info(
    "Progress detected - resetting stuck timer",
    extra={
        "event_type": "progress_detected",
        "episode_id": self.game_state.episode_id,
        "turn": current_turn,
        "score_progress": score_progress,
        "objective_progress": objective_progress,
        "last_score_change_turn": self._last_score_change_turn,
        "last_objective_completion_turn": last_objective_turn if self.game_state.completed_objectives else None,
        "turns_stuck_before_reset": self._get_turns_since_score_change()
    }
)
```

**Warning logging (stuck threshold reached):**
```python
turns_stuck = self._get_turns_since_score_change()
self.logger.warning(
    f"No progress for {turns_stuck} turns - approaching termination",
    extra={
        "event_type": "loop_break_warning",
        "episode_id": self.game_state.episode_id,
        "turn": current_turn,
        "turns_stuck": turns_stuck,
        "last_score_change_turn": self._last_score_change_turn,
        "last_objective_completion_turn": last_objective_turn if self.game_state.completed_objectives else None,
        "score_progress": score_progress,
        "objective_progress": objective_progress
    }
)
```

---

### 4. Warning Messages (Phase 1C)

#### Current Warnings

Phase 1C currently shows warnings starting at 20 turns stuck with:
- "DIE in {y} turns" countdown
- Action novelty hints
- Unexplored exit suggestions

**Location:** `orchestration/zork_orchestrator_v2.py`, lines 483-522 (warnings) and 849-866 (critic integration)

#### Updated Warning Message

**When objectives exist:**
```
⚠️ WARNING: No progress for {turns_stuck} turns!
You will DIE in {turns_remaining} turns unless you increase your score or complete an objective.

CURRENT OBJECTIVES:
- {objective_1}
- {objective_2}
- {objective_3}

SUGGESTIONS:
- Try working on the objectives listed above
- Prioritize actions that might increase your score
- {existing action novelty hints}
- {existing unexplored exit suggestions}
```

**When no objectives exist:**
```
⚠️ WARNING: No progress for {turns_stuck} turns!
You will DIE in {turns_remaining} turns unless you increase your score.

SUGGESTIONS:
- {existing action novelty hints}
- {existing unexplored exit suggestions}
```

#### Implementation Changes

**Modify warning generation in orchestrator:**

```python
def _generate_stuck_warning(self, turns_stuck: int) -> str:
    """Generate warning message for stuck agent."""
    turns_remaining = 40 - turns_stuck

    # Base warning
    warning_parts = [
        f"⚠️ WARNING: No progress for {turns_stuck} turns!"
    ]

    # Termination message (adapt based on objective existence)
    if self.game_state.discovered_objectives:
        warning_parts.append(
            f"You will DIE in {turns_remaining} turns unless you "
            f"increase your score or complete an objective."
        )

        # Add current objectives
        warning_parts.append("\nCURRENT OBJECTIVES:")
        for obj in self.game_state.discovered_objectives[:5]:  # Limit to 5
            warning_parts.append(f"- {obj}")
        warning_parts.append("")
    else:
        warning_parts.append(
            f"You will DIE in {turns_remaining} turns unless you "
            f"increase your score."
        )

    # Suggestions section
    warning_parts.append("SUGGESTIONS:")

    # Add objective-specific hint if objectives exist
    if self.game_state.discovered_objectives:
        warning_parts.append("- Try working on the objectives listed above")
        warning_parts.append("- Prioritize actions that might increase your score")

    # Add existing hints (action novelty, unexplored exits)
    warning_parts.extend(self._get_existing_stuck_hints())

    return "\n".join(warning_parts)
```

---

### 5. Implementation Changes

#### Files to Modify

1. **`orchestration/zork_orchestrator_v2.py`**
   - Modify Phase 1A progress tracking (lines 347-380)
   - Update Phase 1C warning generation (lines 483-522)
   - Add objective-based progress calculation logic
   - Update logging statements

2. **`session/game_configuration.py`**
   - Add `enable_objective_based_progress` field to GameConfiguration
   - Add config loading from pyproject.toml

3. **`pyproject.toml`**
   - Add feature flag to `[tool.zorkgpt.loop_break]` section

4. **`loop_break.md`** (documentation)
   - Update Phase 1A documentation to reflect new progress definition
   - Add configuration documentation for new flag
   - Update examples and monitoring guidance

#### Code Changes - Progress Tracking

**Current code (lines 354-393 in orchestrator):**
```python
def _track_score_for_progress_detection(self) -> None:
    """Track score changes for progress detection."""
    if not hasattr(self, '_last_score_change_turn'):
        self._last_score_change_turn = 0
        self._last_tracked_score = self.game_state.previous_zork_score
        return

    current_score = self.game_state.previous_zork_score

    # Detect any score change (increase or decrease)
    if current_score != self._last_tracked_score:
        self._last_score_change_turn = self.game_state.turn_count
        self._last_tracked_score = current_score
```

**Note:** Current implementation uses point-in-time comparison (did score change THIS turn?) but tracks the turn when it last changed, allowing window-based progress checks.

**New code:**
```python
def _track_score_for_progress_detection(self) -> None:
    """Track score changes for progress detection (with objective support)."""
    # Initialize on first call
    if not hasattr(self, '_last_score_change_turn'):
        self._last_score_change_turn = 0
        self._last_tracked_score = self.game_state.previous_zork_score
        return

    current_score = self.game_state.previous_zork_score
    current_turn = self.game_state.turn_count

    # Track if score changed THIS turn (for updating _last_score_change_turn)
    score_changed_this_turn = (current_score != self._last_tracked_score)

    # Calculate window-based progress checks (both use same window)
    score_progress = (current_turn - self._last_score_change_turn) < self.config.max_turns_stuck
    objective_progress = False
    last_objective_turn = None

    # Check objective completion if feature enabled
    if (
        self.config.enable_objective_based_progress
        and self.game_state.completed_objectives
    ):
        last_objective_turn = max(
            obj["completed_turn"] for obj in self.game_state.completed_objectives
        )
        objective_progress = (current_turn - last_objective_turn) < self.config.max_turns_stuck

    # Determine if progress was made (window-based OR logic)
    progress_made = score_progress or objective_progress

    # Calculate turns stuck for logging
    turns_stuck = self._get_turns_since_score_change()

    # Log progress check with both metrics
    self.logger.debug(
        f"Progress check: score_progress={score_progress}, "
        f"objective_progress={objective_progress}, "
        f"progress_made={progress_made}, turns_stuck={turns_stuck}",
        extra={
            "event_type": "progress_check",
            "episode_id": self.game_state.episode_id,
            "turn": current_turn,
            "score_progress": score_progress,
            "objective_progress": objective_progress,
            "last_score_change_turn": self._last_score_change_turn,
            "last_objective_completion_turn": last_objective_turn,
            "turns_stuck": turns_stuck
        }
    )

    # Update _last_score_change_turn based on progress
    if score_changed_this_turn:
        # Score changed this turn - update tracking
        old_turn = self._last_score_change_turn
        self._last_score_change_turn = current_turn
        self._last_tracked_score = current_score

        if turns_stuck > 0:
            self.logger.info(
                "Progress detected (score change) - resetting stuck timer",
                extra={
                    "event_type": "progress_detected",
                    "episode_id": self.game_state.episode_id,
                    "turn": current_turn,
                    "score_progress": True,
                    "objective_progress": objective_progress,
                    "turns_stuck_before_reset": turns_stuck
                }
            )
    elif objective_progress and not score_progress:
        # Only objective progress (no recent score change) - update tracking to prevent termination
        self._last_score_change_turn = current_turn

        self.logger.info(
            "Progress detected (objective completion) - resetting stuck timer",
            extra={
                "event_type": "progress_detected",
                "episode_id": self.game_state.episode_id,
                "turn": current_turn,
                "score_progress": False,
                "objective_progress": True,
                "last_objective_completion_turn": last_objective_turn,
                "turns_stuck_before_reset": turns_stuck
            }
        )
```

#### Code Changes - Warning Generation

**Update existing `_generate_stuck_warning()` method (lines 483-522):**

Add objective-aware warning content as specified in Section 4.

**Changes needed:**
1. Check if `self.game_state.discovered_objectives` exists and is non-empty
2. Adjust termination message to mention "score or objective"
3. Add "CURRENT OBJECTIVES:" section listing objectives (max 5)
4. Add objective-focused suggestions to existing hints

---

### 6. Testing Strategy

#### Unit Tests

**File:** `tests/test_loop_break_objective_progress.py` (new file)

**Test Cases:**

1. **`test_score_increase_resets_timer`**
   - Score increases → stuck timer resets to 0
   - Validates existing behavior still works

2. **`test_objective_completion_resets_timer`**
   - Objective completes → stuck timer resets to 0
   - Validates new behavior

3. **`test_both_score_and_objective_reset_timer`**
   - Both occur in same turn → stuck timer resets to 0
   - Validates OR logic

4. **`test_no_progress_increments_timer`**
   - No score increase, no objective completion → stuck timer increments
   - Validates stuck detection

5. **`test_fallback_to_score_only_when_no_objectives`**
   - Empty objectives list → only score affects stuck timer
   - Validates backward compatibility

6. **`test_feature_flag_disabled_ignores_objectives`**
   - `enable_objective_based_progress=False` → only score affects timer
   - Validates feature flag

7. **`test_objective_removal_does_not_count_as_progress`**
   - Objectives removed via staleness/refinement → stuck timer continues
   - Validates removal doesn't reset timer

8. **`test_objective_completion_within_window`**
   - Objective completed 30 turns ago → objective_progress=True (within window)
   - Objective completed 39 turns ago → objective_progress=True (edge of window)
   - Objective completed at turn 1, checked at turn 42 → objective_progress=False (outside window)
   - Validates 40-turn window with realistic scenarios

9. **`test_logging_includes_both_metrics`**
   - Verify log statements include score_increased and objective_completed flags
   - Validates observability

#### Integration Tests

**File:** `tests/test_loop_break_objective_integration.py` (new file at root level)

**Test Cases:**

1. **`test_episode_continues_with_objective_completions`**
   - Setup: 30 turns without score increase
   - Action: Complete an objective at turn 31
   - Expected: Episode continues (stuck timer resets)
   - Action: Continue for 20 more turns with no progress
   - Expected: Warnings start appearing (turns_stuck=20)

2. **`test_episode_terminates_without_any_progress`**
   - Setup: 40 turns without score increase or objective completion
   - Expected: Episode terminates at turn 40

3. **`test_warnings_mention_objectives_when_present`**
   - Setup: 20 turns stuck with 3 active objectives
   - Expected: Warning includes "score or complete an objective" and lists objectives

4. **`test_warnings_omit_objectives_when_absent`**
   - Setup: 20 turns stuck with no objectives
   - Expected: Warning includes "increase your score" only (no objective mention)

5. **`test_mixed_progress_scenario`**
   - Turn 10: Score increase (reset)
   - Turn 30: No progress for 20 turns (warning appears)
   - Turn 35: Objective completion (reset)
   - Turn 55: No progress for 20 turns (warning appears)
   - Turn 75: Score increase (reset)
   - Expected: Episode continues past turn 75

6. **`test_objective_completion_after_score_increase`**
   - Turn 5: Score increase
   - Turn 10: Objective completion
   - Expected: Both events reset timer independently

7. **`test_phase_1a_1b_1c_integration`**
   - Verify all three phases work together without conflicts
   - Phase 1A uses new progress metric (score OR objectives)
   - Phase 1B applies location penalties independently
   - Phase 1C warnings mention objectives when present
   - Validates that phases don't interfere with each other

---

### 7. Backward Compatibility

#### Compatibility Requirements

1. **Config file compatibility:**
   - Missing `enable_objective_based_progress` → defaults to `true`
   - Existing configs work without modification

2. **Behavior with objectives disabled:**
   - If objective system disabled in config → falls back to score-only tracking
   - No breaking changes to existing score-only behavior

3. **Logs and monitoring:**
   - New log fields added, but existing fields unchanged
   - Monitoring dashboards will see new fields (backward compatible)

#### Migration Path

**For users upgrading:**

1. **No action required** - feature enabled by default
2. **To disable** (revert to old behavior): Add to `pyproject.toml`:
   ```toml
   [tool.zorkgpt.loop_break]
   enable_objective_based_progress = false
   ```

**For testing/comparison:**

Run episodes with both settings to compare behavior:
```bash
# Test with new behavior (default)
uv run python main.py

# Test with old behavior (score-only)
# (temporarily set enable_objective_based_progress = false in config)
uv run python main.py
```

---

### 8. Success Metrics

#### Before Implementation (Baseline)

Measure on 10 episodes:
1. Number of episodes terminated by loop break
2. Average turns stuck before termination
3. Number of objective completions in stuck episodes
4. False termination rate (terminated despite making objective progress)

**Note on measuring false terminations:** Requires manual episode review to determine if agent was making meaningful progress (completing objectives) when terminated. See `SPEC_VALIDATION.md` for methodology used with 75-episode dataset.

#### After Implementation (Expected)

1. **Reduced false terminations:** Episodes that complete objectives within 40 turns should not terminate
2. **Increased episode length:** Episodes with objective progress should run longer (until score progress occurs or truly stuck)
3. **Better agent experience:** Warnings should be more relevant (mentioning both progress paths)
4. **Same termination for truly stuck:** Episodes with no progress (score OR objectives) still terminate at 40 turns

#### Metrics to Monitor

| Metric | Pre-Implementation | Post-Implementation | Expected Change |
|--------|-------------------|-------------------|-----------------|
| False termination rate | Baseline | Lower | -30% to -50% |
| Avg turns per episode | Baseline | Higher | +10% to +20% |
| Episodes with objective completions | Baseline | Same | No change |
| Truly stuck terminations | Baseline | Same | No change |

---

### 9. Risks and Mitigations

#### Risk 1: Episodes run too long

**Description:** Agent completes trivial objectives repeatedly without meaningful progress, avoiding termination.

**Likelihood:** Low - objective completion requires LLM validation and staleness checks already prevent objective accumulation.

**Mitigation:**
- 40-turn window applies to both metrics equally (same threshold)
- Objective staleness system removes stale DISCOVERED objectives after 30 turns without progress (location/score change)
- Completed objectives are never removed and persist for entire episode
- Feature flag allows disabling if problematic

#### Risk 2: Objective spam gaming the system

**Description:** Agent creates many easy objectives to reset timer without real progress.

**Likelihood:** Very low - agents don't directly create objectives (LLM discovery system controls this).

**Mitigation:**
- Objectives created by periodic LLM discovery (agent doesn't control timing)
- Objective refinement reduces count when too many accumulate
- Completion requires LLM validation (not agent self-declaration)

#### Risk 3: Performance degradation

**Description:** Scanning completed_objectives list every turn adds overhead.

**Likelihood:** Low - list is small (< 50 elements expected).

**Mitigation:**
- O(n) scan is acceptable for small lists
- Early exit when objectives list is empty (no scan)
- Feature flag allows disabling if performance issue observed

#### Risk 4: Logging noise

**Description:** Adding fields to every-turn logs increases log volume.

**Likelihood:** Medium - debug logs will be more verbose.

**Mitigation:**
- Use debug level for per-turn checks (filterable)
- Use info level only when progress state changes
- Combined format reduces line count vs separate statements

---

### 10. Future Enhancements (Out of Scope)

These ideas are NOT part of this spec but could be considered later:

1. **Weighted progress:** Score increases worth more than objective completions (e.g., score reset = 40 turns, objective = 20 turns)
2. **Objective quality scoring:** High-value objectives count more than trivial ones
3. **Separate thresholds:** Different stuck windows for score vs objectives
4. **Dynamic thresholds:** Adjust window based on episode performance
5. **Progress velocity:** Track rate of objective completion, not just binary completion

---

## Implementation Checklist

**For the developer implementing this spec:**

- [ ] Add `enable_objective_based_progress` field to `GameConfiguration` dataclass
- [ ] Add config loading from `pyproject.toml` in `GameConfiguration.__init__`
- [ ] Update default `pyproject.toml` with new flag in `[tool.zorkgpt.loop_break]`
- [ ] Modify progress tracking in `orchestration/zork_orchestrator_v2.py` (lines 347-380)
  - [ ] Add objective completion scan logic
  - [ ] Implement OR logic for progress detection
  - [ ] Add feature flag check
  - [ ] Update stuck timer logic
- [ ] Update logging statements with combined metrics format
  - [ ] Debug logs for every turn
  - [ ] Info logs when progress detected
  - [ ] Warning logs when stuck threshold reached
- [ ] Update warning generation (lines 483-522)
  - [ ] Add objective-aware message variants
  - [ ] List current objectives when present
  - [ ] Add objective-focused suggestions
- [ ] Write unit tests in `tests/test_loop_break_objective_progress.py`
  - [ ] 9 test cases as specified in Section 6
- [ ] Write integration tests in `tests/test_loop_break_objective_integration.py` (root level)
  - [ ] 7 test cases as specified in Section 6
- [ ] Update `loop_break.md` documentation
  - [ ] Update Phase 1A description
  - [ ] Add configuration documentation
  - [ ] Add examples with objective completion
  - [ ] Update monitoring guidance
- [ ] Run full test suite to verify no regressions
  - [ ] `uv run pytest tests/ -xvs`
  - [ ] Verify existing loop break tests still pass
- [ ] Test with feature flag disabled (backward compatibility)
- [ ] Test with objectives disabled in config (fallback behavior)
- [ ] Collect baseline metrics on 10 episodes before implementation
- [ ] Collect post-implementation metrics on 10 episodes
- [ ] Compare metrics against success criteria

---

## Questions for Reviewer

**Before implementation, please confirm:**

1. Is the 40-turn window for objective completion acceptable? (Same as score threshold)
2. Should objective completion detection use a different interval than score checking? (Currently: every turn)
3. Are there any other forms of "progress" we should consider besides score and objectives?
4. Should we log when the feature flag is disabled and objectives are being ignored?
5. Any concerns about the O(n) scan of completed_objectives list each turn?

---

## Appendix A: Example Scenarios

### Scenario 1: Objective Completion Prevents False Termination

**Setup:**
- Turn 1-30: Agent explores, no score increase
- Turn 31: Agent completes objective "Explore north to Location 81"
- Turn 32-50: Agent continues exploring, no score increase

**Current Behavior (score-only):**
- Turn 21: 20 turns stuck → warnings start
- Turn 41: 40 turns stuck → episode terminates (FALSE TERMINATION)

**New Behavior (score OR objective):**
- Turn 31: Objective completion → stuck timer resets to 0
- Turn 51: 20 turns since last progress → warnings start
- Turn 71: 40 turns since last progress → episode terminates (CORRECT - truly stuck)

**Result:** Episode runs 30 turns longer, giving agent more time to make progress.

---

### Scenario 2: Mixed Progress Pattern

**Setup:**
- Turn 1-10: Agent explores
- Turn 11: Score increases by 5 (picked up treasure)
- Turn 12-30: Agent explores, no score
- Turn 31: Agent completes objective "Open the trap door"
- Turn 32-50: Agent explores, no score
- Turn 51: Score increases by 10 (solved puzzle)

**Progress Timeline:**

| Turn | Event | Stuck Timer | Action |
|------|-------|-------------|--------|
| 11 | Score +5 | 0 (reset) | Continue |
| 31 | Objective complete | 0 (reset) | Continue |
| 51 | Score +10 | 0 (reset) | Continue |
| 52-71 | No progress | 0→20 | Warning starts at turn 71 |
| 92 | No progress | 40 | Terminate |

**Result:** Episode continues as long as either form of progress occurs within 40-turn windows.

---

### Scenario 3: Fallback When No Objectives

**Setup:**
- Objectives disabled in config OR all objectives completed and list empty
- Turn 1-40: Agent explores, no score increase

**Current Behavior:**
- Turn 41: Terminate (40 turns no score)

**New Behavior:**
- Turn 41: Terminate (40 turns no score OR objective)
- No change - falls back to score-only tracking

**Result:** Backward compatible behavior when objectives unavailable.

---

## Appendix B: Configuration Examples

### Enable Objective-Based Progress (Default)

```toml
[tool.zorkgpt.loop_break]
# Progress velocity detection
max_turns_stuck = 40              # Termination threshold
stuck_check_interval = 10         # Check every 10 turns

# Location revisit penalty
enable_location_penalty = true
location_revisit_penalty = -0.2
location_revisit_window = 5

# Exploration guidance
enable_exploration_hints = true
action_novelty_window = 15

# Stuck warnings
enable_stuck_warnings = true
stuck_warning_threshold = 20

# Objective-based progress (NEW)
enable_objective_based_progress = true  # Default: enabled
```

### Disable Objective-Based Progress (Legacy Behavior)

```toml
[tool.zorkgpt.loop_break]
max_turns_stuck = 40
enable_objective_based_progress = false  # Revert to score-only
# ... other settings as above ...
```

### Disable Loop Break Entirely

```toml
[tool.zorkgpt.loop_break]
enable_velocity_detection = false  # Disables all loop break phases
# Other fields ignored when velocity detection disabled
```

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-10 | Initial | Complete specification based on requirements gathering |
| 1.1 | 2025-11-10 | Corrected | Fixed based on codebase research: (1) Corrected config field names (`stuck_turn_threshold` → `max_turns_stuck`), (2) Added data structure documentation for GameState objectives, (3) Fixed algorithm to use window-based checking for both score and objectives (symmetric), (4) Corrected variable names to match existing implementation (`_last_score_change_turn`, `_last_tracked_score`), (5) Updated implementation code to use window-based progress detection, (6) Clarified that completed objectives persist forever (only discovered objectives have staleness), (7) Fixed test file paths (no integration/ subdirectory), (8) Updated Scenario 1 warning threshold from 30 to 20 turns, (9) Fixed config examples to match actual pyproject.toml structure |

---

**End of Specification**
