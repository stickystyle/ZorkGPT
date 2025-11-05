# Loop Break System - Design Specification

## Problem Statement

During episode `2025-11-03T12:38:53`, the agent exhibited severe stuck behavior:
- **341 total turns** before hitting `max_turns_per_episode`
- **Score stuck at 40** from turn 105 to turn 340 (235 turns with zero progress)
- **Location oscillation**: Dam (139 visits), Dam Lobby (26 visits), Maintenance (37 visits)
- **Action patterns**: Repeatedly trying failed puzzle actions, inventory manipulation loops

### Current Loop Detection Failures

Three existing loop detection mechanisms all failed:

1. **StateManager.track_state_hash()** (Z-machine state comparison)
   - **Why it failed**: Requires EXACT state match, but inventory constantly changes
   - Detection triggered once but had no intervention mechanism
   - Game continued despite detection

2. **ContextManager.detect_loops_in_recent_actions()** (Action sequence patterns)
   - **Why it failed**: Only checks 2-4 action sequences
   - Agent varied actions enough to avoid pattern detection
   - No repeated sequences despite obvious behavioral loop

3. **RejectionManager.should_override_rejection()** (Oscillation detection)
   - **Why it failed**: Only runs when critic REJECTS actions
   - Critic approved oscillating movements with high confidence (0.9-1.0)
   - Detection code never executed because actions were accepted
   - **Design flaw**: Loop detection is reactive (only on rejections) instead of proactive

### Cost Analysis

Episode ran for 235+ wasted turns after getting stuck:
- API costs for ~940+ LLM calls (agent + critic + extractor + synthesis)
- Zero gameplay value delivered
- No score improvement or knowledge gained

## Proposed Solution

Implement three complementary mechanisms:

### 1. Progress Velocity Detection (Score Stagnation)
**Approach**: Terminate episode when score unchanged for configurable threshold

### 2. Location Loop Detection (A-B-A-B Oscillation + Location Camping)
**Approach**: Detect two patterns of stuck behavior:
- **A-B-A-B Oscillation**: Alternating between two locations
- **Location Camping**: Staying at same location repeatedly (5+ visits in 10-turn window)

### 3. Programmatic Score Modification (Enforce Loop Breaking)
**Approach**: Apply penalties/bonuses to critic scores programmatically when loops detected
- Critic evaluates action → base score
- Orchestrator predicts destination from action
- Apply penalties for returning to loop locations (-0.6 to -0.8)
- Apply bonuses for exploring away (+0.5)
- Adjusted score determines acceptance/rejection

## Rationale for Chosen Direction

### Why Terminate Instead of Warning? (Progress Velocity)

**Rejected approach**: Inject context warning like "You've made no progress for 20 turns. Abandon this area."

**Problems with warnings:**
1. **Vague guidance**: Agent has no concept of:
   - How far away is "elsewhere"?
   - What counts as "this area"?
   - How to systematically escape?

2. **Oscillation amplification risk**: Warning could trigger:
   ```
   Turn 155: "Abandon this area" → Agent goes to Dam Base
   Turn 175: "Abandon this area" → Agent returns to Dam Lobby
   Turn 195: "Abandon this area" → Agent goes to Maintenance
   → Oscillation intensifies
   ```

3. **Token waste**: Continuing a stuck episode burns tokens trying to "unstick" an unsalvageable agent state

**Chosen approach rationale:**
- **Clean failure mode**: If stuck 30+ turns, agent is unsalvageable - cut losses
- **Cross-episode learning**: Memories from successful portion still valuable
- **Token efficiency**: Episode 2025-11-03T12:38:53 would have terminated at turn ~135 instead of 340 (save 205 turns)
- **Definitive signal**: Score stagnation = objective evidence of no progress

### Why Programmatic Score Modification? (Loop Detection)

**Rejected approaches:**

1. **Text guidance to critic**: "Apply -0.8 penalty for returns"
   - Problem: LLM may not follow numeric instructions reliably
   - Problem statement shows critic approved oscillations with 0.9-1.0 confidence
   - Unvalidated assumption that text changes LLM behavior

2. **Hard constraint**: "You cannot return to Dam Lobby for 3 turns"
   - Problem: Agent might not know which actions avoid forbidden location
   - Still relies on LLM following instructions

3. **Geographic distance mandate**: "Move 3 rooms away from stuck zone"
   - Problem: Complex to implement, agent lacks "distance" concept
   - Requires pathfinding logic in orchestrator

**Chosen approach rationale:**
- **Deterministic enforcement**: Penalties always applied in code, not relying on LLM interpretation
- **Debuggable**: Log exact adjustments made (base_score → adjusted_score)
- **Tunable**: All penalty/bonus values in configuration
- **Testable**: Can unit test penalty logic without LLM calls
- **Soft constraint**: Adjusts scores probabilistically rather than hard-blocking actions
- **Override with context**: Critic still informed about patterns for reasoning, but orchestrator enforces numerics
- **Graceful degradation**: If destination prediction fails, no penalty applied (safe fallback)

### Why All Three Mechanisms?

**Complementary coverage:**
- **Progress velocity** catches ALL stuck patterns (location loops, puzzle loops, combat loops) - safety net
- **A-B-A-B oscillation** detects simple back-and-forth movement between two locations
- **Location camping** detects repeated returns to same location (catches Dam: 139 visits pattern)
- **Programmatic penalties** enforce loop breaking deterministically

**Together**: Early intervention (loop detection + penalties) with definitive safety net (progress velocity)

## Implementation Specification

### 1. Progress Velocity Detection

**Location**: `orchestration/zork_orchestrator_v2.py` in `_run_game_loop()`

**Mechanism**:
```python
# Check every N turns (configurable, default 10)
if self.game_state.turn_count % self.config.stuck_check_interval == 0:
    turns_stuck = self._get_turns_since_score_change()

    if turns_stuck >= self.config.max_turns_stuck:
        self.logger.warning(
            f"Terminating episode: no progress for {turns_stuck} turns (score stuck at {self.game_state.previous_zork_score})",
            extra={
                "event_type": "stuck_termination",
                "episode_id": self.game_state.episode_id,
                "turn": self.game_state.turn_count,
                "score": self.game_state.previous_zork_score,
                "turns_stuck": turns_stuck,
            }
        )
        self.game_state.game_over_flag = True
        self.game_state.termination_reason = "stuck_no_progress"
        return self.game_state.previous_zork_score
```

**Helper methods**:
```python
def _get_turns_since_score_change(self) -> int:
    """Calculate turns since last score change."""
    if not hasattr(self, '_last_score_change_turn'):
        # First turn, no score change yet
        return self.game_state.turn_count

    return self.game_state.turn_count - self._last_score_change_turn

def _track_score_for_progress_detection(self):
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

        self.logger.info(
            f"Score changed: {self._last_tracked_score} → {current_score}",
            extra={
                "event_type": "score_change",
                "turn": self.game_state.turn_count,
                "old_score": self._last_tracked_score,
                "new_score": current_score,
            }
        )
```

**Benefits**:
- O(1) calculation instead of O(n) history search
- No sliding window issues - always accurate
- Handles score decreases correctly
- No memory growth issues

**Integration point**:
```python
def _run_game_loop(self, initial_state: str) -> int:
    """Run the main game loop."""
    current_game_state = initial_state

    while (
        not self.game_state.game_over_flag
        and self.game_state.turn_count < self.config.max_turns_per_episode
    ):
        self.game_state.turn_count += 1

        # Run turn
        action_taken, next_game_state = self._run_turn(current_game_state)

        # Track score for progress detection
        self._track_score_for_progress_detection()

        # Check for stuck behavior (every N turns)
        if self.game_state.turn_count % self.config.stuck_check_interval == 0:
            turns_stuck = self._get_turns_since_score_change()
            if turns_stuck >= self.config.max_turns_stuck:
                # Termination logic (see above)
                ...

        # Continue with periodic updates, state export, etc.
```

### 2. Location Loop Detection (Oscillation + Camping)

**Location**: `orchestration/zork_orchestrator_v2.py` in `_execute_turn_logic()`

**Pattern 1: A-B-A-B Oscillation Detection**:
```python
def _detect_oscillation_pattern(self) -> dict:
    """
    Detect A-B-A-B oscillation pattern in recent locations.

    Returns:
        dict with keys:
        - detected (bool): Whether oscillation found
        - pattern_ids (list): Location IDs in oscillation
        - pattern_names (list): Location names for display
        - recent_locations (list): Recent location IDs
    """
    if not hasattr(self, '_location_id_history'):
        self._location_id_history = []

    # Need at least 4 locations for A-B-A-B pattern
    if len(self._location_id_history) < 4:
        return {
            "detected": False,
            "pattern_ids": [],
            "pattern_names": [],
            "recent_locations": []
        }

    recent_ids = self._location_id_history[-6:]  # Last 6 for context

    # Check A-B-A-B pattern
    if (
        len(recent_ids) >= 4
        and recent_ids[-4] == recent_ids[-2]  # A matches A
        and recent_ids[-3] == recent_ids[-1]  # B matches B
        and recent_ids[-4] != recent_ids[-3]  # A != B
    ):
        pattern_ids = [recent_ids[-4], recent_ids[-3]]  # [A, B]

        # Map IDs to names for display
        pattern_names = [
            self._get_location_name_from_id(loc_id)
            for loc_id in pattern_ids
        ]

        return {
            "detected": True,
            "pattern_ids": pattern_ids,
            "pattern_names": pattern_names,
            "recent_locations": recent_ids,
        }

    return {
        "detected": False,
        "pattern_ids": [],
        "pattern_names": [],
        "recent_locations": recent_ids
    }
```

**Pattern 2: Location Camping Detection**:
```python
def _detect_location_camping(self) -> dict:
    """
    Detect location camping (staying in same location repeatedly).

    Returns:
        dict with keys:
        - detected (bool): Whether camping found
        - camped_location_id (int): Location ID being camped
        - camped_location_name (str): Location name for display
        - visit_count (int): Number of visits in window
        - window_size (int): Number of turns examined
    """
    if not hasattr(self, '_location_id_history'):
        self._location_id_history = []

    window_size = min(
        self.config.camping_window,
        len(self._location_id_history)
    )

    if window_size < self.config.camping_threshold:
        return {
            "detected": False,
            "camped_location_id": None,
            "camped_location_name": "",
            "visit_count": 0,
            "window_size": window_size
        }

    # Count visits to each location in recent window
    recent_window = self._location_id_history[-window_size:]
    location_counts = {}

    for loc_id in recent_window:
        location_counts[loc_id] = location_counts.get(loc_id, 0) + 1

    # Find most visited location
    max_visits = max(location_counts.values())
    most_visited_id = max(location_counts.items(), key=lambda x: x[1])[0]

    # Check if camping threshold exceeded
    if max_visits >= self.config.camping_threshold:
        return {
            "detected": True,
            "camped_location_id": most_visited_id,
            "camped_location_name": self._get_location_name_from_id(most_visited_id),
            "visit_count": max_visits,
            "window_size": window_size
        }

    return {
        "detected": False,
        "camped_location_id": None,
        "camped_location_name": "",
        "visit_count": 0,
        "window_size": window_size
    }
```

**Location Tracking (uses IDs per architectural constraints)**:
```python
def _track_location_history(self):
    """Track location at each turn for loop detection."""
    if not hasattr(self, '_location_id_history'):
        self._location_id_history = []

    # Use Z-machine location ID (integer) - NOT room names
    current_location_id = self.jericho_interface.get_location_id()
    self._location_id_history.append(current_location_id)

    # Keep sliding window
    if len(self._location_id_history) > 20:
        self._location_id_history.pop(0)

def _get_location_name_from_id(self, location_id: int) -> str:
    """Get location name from ID for display purposes."""
    # Try map first
    if location_id in self.map_manager.map_graph.rooms:
        return self.map_manager.map_graph.rooms[location_id].name

    # Fallback to Jericho
    loc_data = self.jericho_interface.get_location_structured()
    if loc_data and loc_data.num == location_id:
        return loc_data.name

    return f"Location_{location_id}"
```

### 3. Programmatic Score Modification

**Location**: `orchestration/zork_orchestrator_v2.py` in `_execute_turn_logic()`

**Mechanism**: Apply penalties/bonuses to critic scores programmatically

```python
def _apply_loop_penalties(
    self,
    base_score: float,
    action: str,
    oscillation_info: dict,
    camping_info: dict
) -> Tuple[float, str]:
    """
    Apply programmatic penalties/bonuses for loop behavior.

    Returns:
        (adjusted_score, reason_string)
    """
    adjusted_score = base_score
    reasons = []

    # Predict where action will lead (simple heuristic)
    predicted_destination = self._predict_action_destination(action)

    # Oscillation penalty
    if oscillation_info["detected"] and predicted_destination:
        if predicted_destination in oscillation_info["pattern_ids"]:
            penalty = self.config.oscillation_return_penalty  # e.g., -0.8
            adjusted_score = max(0.0, adjusted_score + penalty)
            pattern_str = " → ".join(oscillation_info["pattern_names"])
            reasons.append(f"Oscillation penalty {penalty} (pattern: {pattern_str})")
        else:
            # Reward exploration away from pattern
            bonus = self.config.oscillation_exploration_bonus  # e.g., +0.5
            adjusted_score = min(1.0, adjusted_score + bonus)
            reasons.append(f"Exploration bonus {bonus}")

    # Location camping penalty
    if camping_info["detected"] and predicted_destination:
        if predicted_destination == camping_info["camped_location_id"]:
            penalty = self.config.camping_return_penalty  # e.g., -0.6
            adjusted_score = max(0.0, adjusted_score + penalty)
            loc_name = camping_info["camped_location_name"]
            reasons.append(f"Camping penalty {penalty} (returning to {loc_name})")

    # Ensure score stays in valid range
    adjusted_score = max(0.0, min(1.0, adjusted_score))

    reason_string = "; ".join(reasons) if reasons else ""
    return adjusted_score, reason_string

def _predict_action_destination(self, action: str) -> Optional[int]:
    """
    Predict where an action will lead (location ID).

    Returns:
        Location ID if action is a known movement, None otherwise.
    """
    # For movement actions, check map graph
    action_lower = action.lower().strip()

    # Check if it's a direction
    directions = ['north', 'south', 'east', 'west', 'ne', 'nw', 'se', 'sw',
                  'up', 'down', 'in', 'out']

    for direction in directions:
        if action_lower == direction or action_lower.startswith(f"go {direction}"):
            current_loc_id = self.jericho_interface.get_location_id()

            # Look up in map graph
            if current_loc_id in self.map_manager.map_graph.rooms:
                room = self.map_manager.map_graph.rooms[current_loc_id]
                if direction in room.exits:
                    return room.exits[direction]
            break

    # Can't predict - non-movement action or unknown destination
    return None
```

**Integration point**:
```python
def _execute_turn_logic(self, current_state: str) -> Tuple[str, str]:
    """Execute a single turn's logic with loop detection."""

    # Track location for loop detection
    self._track_location_history()

    # Detect both oscillation and camping
    oscillation_info = self._detect_oscillation_pattern()
    camping_info = self._detect_location_camping()

    # Build contexts
    agent_context = self.context_manager.get_agent_context(...)

    # Get agent action
    action, reasoning = self.agent.get_action_with_reasoning(agent_context)

    # Build critic context (informational only - no numeric guidance)
    critic_context = self.context_manager.get_critic_context(...)

    if oscillation_info["detected"] or camping_info["detected"]:
        # Add informational context (no numeric penalties)
        critic_context += self._build_loop_context_info(
            oscillation_info,
            camping_info
        )

    # Critic evaluation (base score)
    critic_result = self.critic.evaluate_action(action, critic_context, ...)
    base_score = critic_result.confidence

    # Apply programmatic adjustments
    adjusted_score, adjustment_reason = self._apply_loop_penalties(
        base_score=base_score,
        action=action,
        oscillation_info=oscillation_info,
        camping_info=camping_info
    )

    # Update critic result with adjusted score
    critic_result.confidence = adjusted_score
    if adjustment_reason:
        critic_result.reasoning += f"\n[Loop Detection Override: {adjustment_reason}]"

    # Log adjustment if applied
    if adjustment_reason:
        self.logger.info(
            f"Applied loop penalty: {base_score:.2f} → {adjusted_score:.2f}",
            extra={
                "event_type": "loop_penalty_applied",
                "turn": self.game_state.turn_count,
                "action": action,
                "base_score": base_score,
                "adjusted_score": adjusted_score,
                "reason": adjustment_reason,
            }
        )

    # Continue with rejection loop using adjusted score...
```

**Helper for informational context**:
```python
def _build_loop_context_info(
    self,
    oscillation_info: dict,
    camping_info: dict
) -> str:
    """
    Build informational context about detected loop behavior.
    This is for the critic's awareness, not numeric guidance.
    """
    context_parts = []

    if oscillation_info["detected"]:
        pattern_str = " → ".join(oscillation_info["pattern_names"])
        context_parts.append(
            f"⚠️ Oscillation Pattern: {pattern_str}"
        )

    if camping_info["detected"]:
        context_parts.append(
            f"⚠️ Location Camping: {camping_info['camped_location_name']} "
            f"({camping_info['visit_count']} visits in {camping_info['window_size']} turns)"
        )

    if context_parts:
        return "\n\n" + "\n".join(context_parts) + "\n"

    return ""
```

## Configuration

**Add to `session/game_configuration.py`**:

```python
class GameConfiguration(BaseModel):
    """Game configuration with validation."""

    # Existing fields...

    # Loop Break Configuration - Progress Velocity
    max_turns_stuck: int = Field(
        default=30,
        description="Maximum turns without score change before terminating episode"
    )

    stuck_check_interval: int = Field(
        default=10,
        description="Check for stuck behavior every N turns"
    )

    # Loop Break Configuration - Loop Detection
    enable_loop_detection: bool = Field(
        default=True,
        description="Enable loop detection and programmatic score penalties"
    )

    oscillation_return_penalty: float = Field(
        default=-0.8,
        description="Penalty applied when action returns to oscillation pattern location"
    )

    oscillation_exploration_bonus: float = Field(
        default=0.5,
        description="Bonus applied when action explores away from oscillation pattern"
    )

    camping_return_penalty: float = Field(
        default=-0.6,
        description="Penalty applied when action returns to camped location"
    )

    camping_threshold: int = Field(
        default=5,
        description="Number of visits to same location in window to trigger camping detection"
    )

    camping_window: int = Field(
        default=10,
        description="Turn window for detecting location camping"
    )
```

**Add to `pyproject.toml`**:

```toml
[tool.zorkgpt.loop_break]
# Progress velocity detection
max_turns_stuck = 30              # Terminate if no progress for 30 turns
stuck_check_interval = 10         # Check every 10 turns

# Loop detection and penalties
enable_loop_detection = true      # Enable loop detection
oscillation_return_penalty = -0.8 # Penalty for returning to oscillation pattern
oscillation_exploration_bonus = 0.5  # Bonus for exploring away
camping_return_penalty = -0.6     # Penalty for returning to camped location
camping_threshold = 5             # Visits needed to trigger camping
camping_window = 10               # Turn window for camping detection
```

**Configuration loading in `GameConfiguration.from_toml()`**:

```python
@classmethod
def from_toml(cls, config_path: str = "pyproject.toml") -> "GameConfiguration":
    """Load configuration from TOML file."""
    try:
        with open(config_path, "rb") as f:
            data = tomli.load(f)

        zork_config = data.get("tool", {}).get("zorkgpt", {})

        # Existing configuration loading...

        # Loop break configuration
        loop_break_config = zork_config.get("loop_break", {})

        return cls(
            # Existing fields...
            max_turns_stuck=loop_break_config.get("max_turns_stuck", 30),
            stuck_check_interval=loop_break_config.get("stuck_check_interval", 10),
            enable_loop_detection=loop_break_config.get("enable_loop_detection", True),
            oscillation_return_penalty=loop_break_config.get("oscillation_return_penalty", -0.8),
            oscillation_exploration_bonus=loop_break_config.get("oscillation_exploration_bonus", 0.5),
            camping_return_penalty=loop_break_config.get("camping_return_penalty", -0.6),
            camping_threshold=loop_break_config.get("camping_threshold", 5),
            camping_window=loop_break_config.get("camping_window", 10),
        )
```

## Expected Behavior

### Scenario 1: Progress Velocity Termination

```
Turn 105: Score = 40
Turn 115: Score = 40 (10 turns stuck, logged at INFO level)
Turn 125: Score = 40 (20 turns stuck, logged at WARNING level)
Turn 135: Score = 40 (30 turns stuck, TERMINATE)

Log output:
[WARNING] Terminating episode: no progress for 30 turns (score stuck at 40)
[INFO] Episode completed after 135 turns - Final Score: 40 (stuck_no_progress)
```

**Savings for episode 2025-11-03T12:38:53**:
- Would terminate at turn ~135 instead of 340
- Saves 205 turns ≈ 820 LLM calls ≈ significant API cost

### Scenario 2: Oscillation Detection + Programmatic Penalties

```
Turn 108: Dam Lobby (ID: 15)
Turn 111: Maintenance (ID: 18) - A-B pattern starts
Turn 114: Dam Lobby (ID: 15) - A-B-A detected, need 4 for pattern
Turn 117: Maintenance (ID: 18) - A-B-A-B detected!

Informational context added to critic:
"⚠️ Oscillation Pattern: Dam Lobby → Maintenance"

Turn 117 agent proposes: "north" (would return to Maintenance)
Turn 117 critic evaluates: base_score = 0.9
Turn 117 orchestrator predicts destination: ID 18 (Maintenance)
Turn 117 orchestrator detects: destination in pattern_ids [15, 18]
Turn 117 programmatic penalty: 0.9 + (-0.8) = 0.1
Turn 117 adjusted_score = 0.1 < threshold (0.5) → REJECTED

Agent regenerates:
Turn 117 agent proposes: "south" (to Dam, ID: 20)
Turn 117 critic evaluates: base_score = 0.6
Turn 117 orchestrator predicts destination: ID 20 (Dam)
Turn 117 orchestrator detects: destination NOT in pattern_ids
Turn 117 programmatic bonus: 0.6 + 0.5 = 1.0 (capped at 1.0)
Turn 117 adjusted_score = 1.0 → ACCEPTED
→ Oscillation broken, agent explores new location
```

### Scenario 3: Location Camping Detection (Actual 2025-11-03 Pattern)

```
Turn 108-117: Agent visits Dam (ID: 20) 6 times in 10-turn window
→ Location camping detected (threshold: 5 visits)

Informational context added to critic:
"⚠️ Location Camping: Dam (6 visits in 10 turns)"

Turn 118 agent proposes: "east" (would return to Dam)
Turn 118 critic evaluates: base_score = 0.85
Turn 118 orchestrator predicts destination: ID 20 (Dam)
Turn 118 orchestrator detects: destination == camped_location_id (20)
Turn 118 programmatic penalty: 0.85 + (-0.6) = 0.25
Turn 118 adjusted_score = 0.25 < threshold (0.5) → REJECTED

Agent regenerates:
Turn 118 agent proposes: "south" (to Deep Canyon)
Turn 118 critic evaluates: base_score = 0.7
Turn 118 orchestrator: no camping penalty (different location)
Turn 118 adjusted_score = 0.7 → ACCEPTED
→ Camping broken, agent explores away from Dam
```

### Scenario 4: Combined Protection (All Mechanisms)

```
Turn 108-120: Location camping at Dam
- Camping detection applies penalties
- Agent forced to explore Dam Lobby, Maintenance

Turn 121-128: Oscillates between Dam Lobby ↔ Maintenance
- Oscillation detection applies penalties
- Agent explores Dam Base, Deep Canyon

Turn 129-135: Explores new locations but score still stuck at 40
- No more loops, but puzzle unsolvable
- Progress velocity check: 30 turns without score change
- Episode TERMINATES (stuck_no_progress)

Result:
- Loops broken by programmatic penalties (turns 108-128)
- Progress velocity provides safety net (turn 135)
- Saves 205 turns vs. original 341-turn episode
```

## Testing Strategy

### Unit Tests

**Test progress velocity detection:**
```python
def test_progress_velocity_terminates_after_threshold(orchestrator):
    """Episode should terminate after max_turns_stuck without score change."""
    # Set score to 40
    # Run 30 turns without score change
    # Assert game_over_flag = True
    # Assert termination_reason = "stuck_no_progress"

def test_progress_velocity_resets_on_score_change(orchestrator):
    """Turns stuck counter should reset when score increases."""
    # Set score to 40
    # Run 20 turns
    # Increase score to 50
    # Run 20 more turns (total 40, but only 20 since last change)
    # Assert game_over_flag = False

def test_configurable_stuck_threshold(orchestrator):
    """max_turns_stuck should be configurable."""
    # Set config.max_turns_stuck = 50
    # Run 49 turns without score change → not terminated
    # Run 50 turns without score change → terminated
```

**Test oscillation detection:**
```python
def test_oscillation_detection_abab_pattern(orchestrator):
    """Should detect A-B-A-B location pattern using location IDs."""
    # Visit: Dam (15) → Lobby (18) → Dam (15) → Lobby (18)
    # Assert oscillation_info["detected"] = True
    # Assert oscillation_info["pattern_ids"] = [15, 18]
    # Assert oscillation_info["pattern_names"] = ["Dam", "Lobby"]

def test_oscillation_no_false_positive(orchestrator):
    """Should not detect oscillation in normal exploration."""
    # Visit: Dam (15) → Lobby (18) → Maintenance (20) → Dam Base (22)
    # Assert oscillation_info["detected"] = False

def test_oscillation_uses_location_ids(orchestrator):
    """Oscillation detection must use location IDs, not names."""
    # Verify _location_id_history contains integers
    # Verify pattern_ids contains integers
    # Architectural constraint compliance
```

**Test location camping detection:**
```python
def test_camping_detection_threshold(orchestrator):
    """Should detect camping when threshold exceeded."""
    # Visit Dam (ID 20) 6 times in 10-turn window
    # Assert camping_info["detected"] = True
    # Assert camping_info["camped_location_id"] = 20
    # Assert camping_info["visit_count"] = 6

def test_camping_no_false_positive(orchestrator):
    """Should not detect camping below threshold."""
    # Visit Dam 3 times, Lobby 2 times in 10-turn window
    # Assert camping_info["detected"] = False

def test_camping_window_sliding(orchestrator):
    """Camping should only count visits in recent window."""
    # Visit Dam 10 times in turns 1-10
    # Visit other locations for turns 11-20
    # At turn 20, Dam visits outside window
    # Assert camping_info["detected"] = False
```

**Test programmatic penalties:**
```python
def test_penalty_applied_on_oscillation_return(orchestrator):
    """Penalty should be applied when returning to oscillation pattern."""
    # Setup: A-B-A-B pattern detected
    # Action: "north" (destination in pattern_ids)
    # Base score: 0.9
    # Assert adjusted_score = 0.9 + (-0.8) = 0.1

def test_bonus_applied_on_exploration(orchestrator):
    """Bonus should be applied when exploring away from pattern."""
    # Setup: A-B-A-B pattern detected
    # Action: "south" (destination NOT in pattern_ids)
    # Base score: 0.6
    # Assert adjusted_score = min(1.0, 0.6 + 0.5) = 1.0

def test_penalty_not_applied_without_prediction(orchestrator):
    """No penalty if destination cannot be predicted."""
    # Setup: Pattern detected
    # Action: "examine lamp" (non-movement)
    # Base score: 0.8
    # Assert adjusted_score = 0.8 (unchanged)

def test_score_clamping(orchestrator):
    """Adjusted scores should be clamped to [0.0, 1.0]."""
    # Test: 0.2 + (-0.8) = -0.6 → clamped to 0.0
    # Test: 0.9 + 0.5 = 1.4 → clamped to 1.0
```

### Integration Tests

**Test full loop break workflow:**
```python
def test_stuck_episode_terminates_early(zorkgpt_system):
    """Full episode should terminate when stuck, not hit max_turns."""
    # Configure: max_turns_stuck=20, max_turns_per_episode=200
    # Mock agent to always propose failed actions
    # Run episode
    # Assert final turn < 200 (terminated early)
    # Assert termination_reason = "stuck_no_progress"

def test_loop_breaks_via_programmatic_penalties(zorkgpt_system):
    """Loops should be broken by programmatic score penalties."""
    # Mock agent to oscillate between two locations (IDs 15, 18)
    # Run 10 turns with loop detection enabled
    # Verify penalties are applied when returning to pattern
    # Assert unique locations > 2 (oscillation broken)

def test_camping_breaks_via_penalties(zorkgpt_system):
    """Location camping should be broken by penalties."""
    # Mock agent to camp at location ID 20
    # Run 15 turns
    # Verify camping detected after 5 visits
    # Verify penalties force exploration
    # Assert visit count to location 20 decreases after detection
```

### Walkthrough Tests

**Test against known stuck scenario:**
```python
def test_2025_11_03_stuck_scenario(walkthrough_fixture):
    """Episode 2025-11-03T12:38:53 scenario should terminate early with loop break."""
    # Load walkthrough actions from turns 105-135
    # Run with loop break enabled
    # Assert episode terminates before turn 200
    # Compare to baseline (341 turns without loop break)
```

## Rollout Plan

### Phase 1: Implement Progress Velocity (High ROI, Low Risk)
1. Add configuration fields to `GameConfiguration` (max_turns_stuck, stuck_check_interval)
2. Implement simplified score tracking (_track_score_for_progress_detection)
3. Add termination check in game loop
4. Write unit tests (score tracking, termination logic)
5. Test with problematic episode logs

**Estimated effort**: 4-6 hours

### Phase 2: Implement Loop Detection (High ROI, Medium Risk)
1. Add location tracking using IDs in orchestrator (_track_location_history with IDs)
2. Implement A-B-A-B oscillation detection (_detect_oscillation_pattern)
3. Implement location camping detection (_detect_location_camping)
4. Add destination prediction (_predict_action_destination)
5. Write unit tests (oscillation, camping, prediction)
6. Test with known patterns

**Estimated effort**: 6-8 hours

### Phase 3: Implement Programmatic Penalties (Medium ROI, Medium Risk)
1. Add configuration fields (oscillation_return_penalty, camping_return_penalty, etc.)
2. Implement penalty application logic (_apply_loop_penalties)
3. Integrate into _execute_turn_logic (apply after critic evaluation)
4. Add informational context builder (_build_loop_context_info)
5. Write unit tests (penalty math, score clamping, logging)
6. Test with mocked critic scores

**Estimated effort**: 4-6 hours

### Phase 4: Integration Testing and Tuning
1. Run integration tests with all three mechanisms
2. Tune `max_turns_stuck` threshold (start conservative at 30)
3. Tune penalty/bonus values (start with -0.8, +0.5, -0.6)
4. Monitor false positive rate (premature terminations)
5. Test against episode 2025-11-03T12:38:53 scenario
6. Verify architectural compliance (location IDs, not names)

**Estimated effort**: 6-8 hours

### Phase 5: Production Deployment
1. Enable by default in `pyproject.toml`
2. Document configuration in README
3. Add logging for loop detection events
4. Monitor episode analytics for:
   - Termination rate by reason (stuck_no_progress)
   - Loop penalty application frequency
   - Average episode length
   - Score distribution
   - False positive reviews (manual sampling)

**Total estimated effort**: 20-28 hours

## Success Metrics

**Primary:**
- Episode length reduction for stuck scenarios (target: 50%+ reduction)
- API cost savings (measured in LLM calls avoided)
- Savings for episode 2025-11-03T12:38:53 type scenarios: ~205 turns (60% reduction)

**Secondary:**
- False positive rate (premature terminations via stuck_no_progress) < 5%
- Loop break success rate (oscillation + camping) > 80%
- Programmatic penalty application rate when loops detected > 90%
- No regression in successful episode completion rates
- No regression in max scores achieved

**Monitoring:**
- Loop detection event frequency (oscillation vs. camping)
- Penalty effectiveness (does rejection → regeneration lead to exploration?)
- Average turns before loop detection triggers
- Distribution of adjusted_scores vs. base_scores

## Future Enhancements

**Possible improvements (out of scope for initial implementation):**

1. **Adaptive thresholds**: Adjust `max_turns_stuck` based on current score
   - Early game (score < 20): More lenient (40 turns)
   - Mid game (score 20-100): Standard (30 turns)
   - Late game (score > 100): Stricter (20 turns)

2. **Progressive penalties**: Increase penalty severity with pattern duration
   - First 5 occurrences: -0.6 penalty
   - Next 5 occurrences: -0.8 penalty
   - After 10 occurrences: -1.0 penalty (guaranteed rejection)

3. **Semantic location clustering**: Group related locations (Dam area) for smarter loop detection
   - Current: Only detects exact location ID repeats
   - Enhanced: Detect oscillation within geographic clusters (e.g., "Dam area" = Dam + Dam Lobby + Dam Base)
   - Prevents "micro-exploration" within stuck zone

4. **Advanced destination prediction**: Improve movement prediction accuracy
   - Current: Only handles directional movement via map graph
   - Enhanced: Predict "enter building", "climb tree", context-dependent verbs
   - Higher penalty accuracy → better loop breaking

5. **Cross-episode stuck pattern learning**: Track which locations/puzzles cause stuck behavior
   - Synthesize into knowledge base
   - Pre-emptively apply light penalties to known stuck zones
   - Learn which puzzle sequences lead to loops

6. **Action pattern detection**: Extend beyond location loops
   - Detect repeated action sequences ("examine lamp" → "take lamp" → "drop lamp" loop)
   - Apply penalties to action repetition, not just location repetition
   - Catches puzzle-interaction loops that don't change location
