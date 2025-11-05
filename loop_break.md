# Loop Break System - Implementation Complete

## Status: ✅ DEPLOYED (Phase 1A-1C Complete)

### Quick Summary

The ZorkGPT loop break system prevents wasted tokens on stuck episodes through three complementary mechanisms:

1. **Progress Velocity Detection** - Terminates episodes after 40 turns without score change
2. **Location Revisit Penalty** - Applies -0.2 penalty per recent revisit to discourage loops
3. **Stuck Countdown Warnings** - Shows explicit "you will DIE in {y} turns" warnings to agent

### Real-World Results

**Episode 2025-11-04T09:35:24** (validated with `/tmp/claude/analyze_episode.py`):
- **Before**: 273 turns total, stuck at various scores for extended periods
- **Would terminate at**: Turn 60 (39 turns stuck at score 10)
- **Savings**: 213 turns (78% reduction), approximately 852 LLM calls saved

Analysis showed 14 potential termination points where the system would have intervened, with the first occurring at turn 60.

### How It Works

#### Phase 1A: Progress Velocity Detection (Programmatic Termination)

Tracks score changes and terminates when no progress is made:

```python
# Checks every 10 turns (configurable)
if turn_count % stuck_check_interval == 0:
    turns_stuck = _get_turns_since_score_change()
    if turns_stuck >= max_turns_stuck:
        terminate_episode()  # Hard stop after 40 turns stuck
```

**Key Features**:
- O(1) performance using simple counter
- Checks at configurable intervals (default: every 10 turns)
- Terminates after configurable threshold (default: 40 turns)
- Score change resets the counter (any increase or decrease)
- Logs `stuck_termination` event with metrics

**Implementation**: See `/Volumes/workingfolder/ZorkGPT/orchestration/zork_orchestrator_v2.py` lines 347-380 (tracking) and 526-547 (termination)

#### Phase 1B: Location Revisit Penalty (Programmatic Scoring)

Discourages location loops through penalty system:

```python
# Tracks last 20 locations using Z-machine IDs
recent_window = location_history[-5:]  # Last 5 locations
visits = recent_window.count(current_location_id)

if visits > 0:
    penalty = -0.2 * visits  # -0.2 per revisit
    adjusted_score = base_critic_score + penalty
```

**Key Features**:
- Uses Z-machine location IDs (not names) for stability
- Sliding window of last 5 locations (configurable)
- Penalty stacks: -0.2, -0.4, -0.6, etc.
- Applied to critic confidence scores (not agent context)
- Can be disabled via config flag

**Implementation**: See `/Volumes/workingfolder/ZorkGPT/orchestration/zork_orchestrator_v2.py` lines 394-504

#### Phase 1C: Exploration Guidance + Stuck Countdown Warnings (Context-Based Hints)

Helps agent escape stuck behavior through explicit warnings and hints:

```python
# Countdown warnings (shown in agent context)
if turns_stuck >= stuck_warning_threshold:
    turns_until_death = max_turns_stuck - turns_stuck
    warning = f"""
    ⚠️ SCORE STAGNATION DETECTED
    You have made NO SCORE PROGRESS for {turns_stuck} turns.
    You will DIE in {turns_until_death} turns unless you INCREASE YOUR SCORE.
    """

# Action novelty hints
if proposed_action in recent_action_history:
    hint = "You have tried this action recently with no progress."

# Unexplored exit hints
if current_room has unexplored_exits:
    hint = f"Unexplored directions: {', '.join(unexplored_exits)}"
```

**Key Features**:
- Escalating urgency levels (IMPORTANT → URGENT → CRITICAL)
- Warnings start at 20 turns stuck (configurable)
- Action novelty detection (tracks last 15 actions)
- Unexplored exit suggestions from map data
- All hints added to agent/critic context

**Implementation**: See `/Volumes/workingfolder/ZorkGPT/orchestration/zork_orchestrator_v2.py` lines 483-522 (warnings) and 849-866 (critic integration)

### Configuration

All settings in `pyproject.toml` under `[tool.zorkgpt.loop_break]`:

```toml
[tool.zorkgpt.loop_break]
# Phase 1A: Progress velocity detection
max_turns_stuck = 40          # Turns without score change before termination
stuck_check_interval = 10     # Check interval (every N turns)

# Phase 1B: Location revisit penalty
enable_location_penalty = true
location_revisit_penalty = -0.2  # Penalty per revisit (must be negative)
location_revisit_window = 5      # Number of recent locations to check

# Phase 1C: Stuck warnings & exploration hints
enable_stuck_warnings = true
stuck_warning_threshold = 20     # Turns stuck before warnings start
enable_exploration_hints = true
action_novelty_window = 15       # Number of recent actions to track
```

**Validation Rules**:
- `max_turns_stuck >= stuck_check_interval` (must check before terminating)
- `stuck_warning_threshold < max_turns_stuck` (warnings must appear before death)
- `location_revisit_penalty < 0` (must be negative penalty)

**Loading Configuration**:
```python
from session.game_configuration import GameConfiguration
config = GameConfiguration.from_toml(Path("pyproject.toml"))
```

### Monitoring & Metrics

Track these events in episode logs:

1. **stuck_termination** - Episode terminated by progress velocity
   ```json
   {
     "event_type": "stuck_termination",
     "turn": 60,
     "turns_stuck": 40,
     "score": 10,
     "max_turns_stuck": 40
   }
   ```

2. **score_change** - Score increased/decreased (resets stuck counter)
   ```json
   {
     "event_type": "score_change",
     "turn": 25,
     "old_score": 10,
     "new_score": 35,
     "was_stuck_for": 15
   }
   ```

3. **location_penalty_applied** - Revisit penalty applied to action
   ```json
   {
     "event_type": "location_penalty_applied",
     "turn": 45,
     "location_id": 20,
     "location_name": "Kitchen",
     "recent_visits": 3,
     "base_score": 0.9,
     "adjusted_score": 0.3,
     "penalty": -0.6
   }
   ```

### Testing

**Test Coverage**: 66 tests across 4 test files

```bash
# Run all loop break tests
uv run pytest tests/test_progress_velocity.py tests/test_location_revisit.py \
             tests/test_exploration_guidance.py tests/test_loop_break_integration.py -v

# Phase 1A tests (14 tests)
uv run pytest tests/test_progress_velocity.py -v

# Phase 1B tests (19 tests)
uv run pytest tests/test_location_revisit.py -v

# Phase 1C tests (24 tests)
uv run pytest tests/test_exploration_guidance.py -v

# Integration tests (9 tests)
uv run pytest tests/test_loop_break_integration.py -v

# Verify configuration loading
uv run python /tmp/claude/verify_config.py
```

**Integration Test Coverage**:
- All three phases active simultaneously
- Stuck episode with all mechanisms working together
- Location penalties and stuck warnings coexisting
- Warnings disabled doesn't break velocity detection
- Penalties disabled doesn't break velocity detection
- Configuration loads correctly from TOML
- All mechanisms work independently
- Event structure validation

### Episode Analysis Tool

Analyze historical episodes to see where loop break would have intervened:

```bash
python3 /tmp/claude/analyze_episode.py game_files/episodes/EPISODE_ID/episode_log.jsonl
```

Output shows:
- Score progression timeline
- Stuck periods identified
- First termination point
- Total turns that would be saved
- All check points that exceeded threshold

### Implementation Timeline

- **2025-11-05**: Phase 1A (Progress Velocity) - Score tracking and termination
- **2025-11-05**: Phase 1B (Location Revisit Penalty) - Programmatic penalty system
- **2025-11-05**: Phase 1C (Exploration Guidance) - Context-based warnings and hints
- **2025-11-05**: Integration testing and documentation complete

### Design Decisions

#### Why Three Complementary Mechanisms?

1. **Phase 1A (Velocity)** - Hard safety net. Guarantees termination when completely stuck.
2. **Phase 1B (Penalty)** - Soft discouragement. Makes revisiting less appealing without preventing it.
3. **Phase 1C (Warnings)** - Explicit guidance. Helps agent understand the problem and find solutions.

Each mechanism operates independently:
- Can be enabled/disabled separately
- Different failure modes (velocity catches everything eventually)
- Layered defense against different stuck patterns

#### Why Z-Machine Location IDs?

Location IDs from the Z-machine are stable integers that don't fragment:
- Same location always has same ID
- No "West of House" vs "west of house" issues
- O(1) comparison and counting
- Enables cross-episode location tracking

#### Why Window-Based Revisit Detection?

Looking at last N locations (not all history):
- Prevents penalizing legitimate backtracking
- Focuses on recent looping behavior
- Bounded memory usage (deque with maxlen=20)
- Configurable sensitivity via window size

#### Why Context-Based Warnings Instead of Penalties?

Phase 1C adds to context, not scores:
- Preserves critic's ability to judge actions independently
- Warnings are explicit and interpretable
- Agent sees the problem and can reason about solutions
- Complements programmatic penalty system

### Future Enhancements (Not Implemented)

These were considered but deprioritized:

- **Phase 2A**: Action loop detection (e.g., "open door, close door" cycles)
- **Phase 2B**: Dead-end room detection (rooms with no score progress)
- **Phase 2C**: Inventory thrashing detection (repeated take/drop)
- **Phase 2D**: Command history compression for memory

**Rationale for deferral**: Phase 1 mechanisms should catch most stuck patterns. Deploy and collect data before adding complexity.

### Troubleshooting

**Issue: Agent terminated too early**
- Check `max_turns_stuck` setting (default 40)
- Verify score is actually changing (check episode log)
- Look for `score_change` events in logs

**Issue: Agent still stuck in loops**
- Check if `enable_location_penalty` is true
- Verify `location_revisit_window` is appropriate (default 5)
- Look at `location_penalty_applied` events to see if penalties are triggering

**Issue: Warnings not appearing**
- Check if `enable_stuck_warnings` is true
- Verify `stuck_warning_threshold < max_turns_stuck`
- Check agent context in episode log to see if warnings are present

**Issue: Configuration not loading**
- Verify pyproject.toml syntax (must be valid TOML)
- Check that all required fields have valid values
- Run `/tmp/claude/verify_config.py` to diagnose

### References

**Implementation Files**:
- Core logic: `/Volumes/workingfolder/ZorkGPT/orchestration/zork_orchestrator_v2.py`
- Configuration: `/Volumes/workingfolder/ZorkGPT/session/game_configuration.py`
- Tests: `/Volumes/workingfolder/ZorkGPT/tests/test_*loop*.py`

**Configuration**:
- Settings: `/Volumes/workingfolder/ZorkGPT/pyproject.toml` (section `[tool.zorkgpt.loop_break]`)

**Analysis Tools**:
- Episode analyzer: `/tmp/claude/analyze_episode.py`
- Config verifier: `/tmp/claude/verify_config.py`

---

## Appendix: Technical Details

### Score Tracking Algorithm

```python
class ProgressTracker:
    def __init__(self):
        self._last_score_change_turn = 0
        self._last_known_score = 0

    def track(self, current_turn, current_score):
        if current_score != self._last_known_score:
            self._last_score_change_turn = current_turn
            self._last_known_score = current_score
            log_event("score_change", ...)

    def get_turns_stuck(self, current_turn):
        return current_turn - self._last_score_change_turn
```

### Location Revisit Detection

```python
def detect_revisit(location_history, current_id, window_size=5):
    """
    Checks if current location was recently visited.

    Window excludes current location:
    - History: [A, B, C, D, E]  (E is current)
    - Check: [A, B, C, D]  (last 4)
    - Returns: count of current_id in window
    """
    if len(location_history) <= 1:
        return 0

    recent = location_history[-(window_size + 1):-1]
    return recent.count(current_id)
```

### Warning Escalation Logic

```python
def get_urgency_level(turns_until_death):
    if turns_until_death <= 5:
        return "🚨 CRITICAL EMERGENCY", "IMMEDIATE"
    elif turns_until_death <= 10:
        return "⚠️ URGENT WARNING", "URGENT"
    else:
        return "⚠️ SCORE STAGNATION DETECTED", "IMPORTANT"
```

---

**Last Updated**: 2025-11-05
**Status**: Production Ready
**Test Coverage**: 66/66 passing
**Documentation**: Complete
