# Game Files Directory Guide

This directory contains episode logs and game memories for ZorkGPT gameplay sessions.

## Directory Structure

```
game_files/
├── episodes/              # Episode-specific logs
│   ├── 2025-11-01T21:48:15/    # Production episode (ISO 8601 timestamp)
│   ├── 2025-11-01T08:51:46/    # Production episode
│   ├── test_integration/        # Test episode (IGNORE)
│   └── test-episode/            # Test episode (IGNORE)
├── Memories.md            # Persistent memory store
├── Memories.md.backup     # Memory backup
└── Memories.md.lock       # Memory lock file
```

## Episode Directory Naming

**Production Episodes**: Named with ISO 8601 timestamps (`YYYY-MM-DDTHH:MM:SS`)
- Example: `2025-11-01T21:48:15`
- These contain real gameplay sessions

**Test Episodes**: Prefixed with `test_` or `test-`
- Example: `test_integration`, `test-episode`, `test_episode_1762090936`
- **IMPORTANT**: Always ignore these when analyzing gameplay data
- They are generated during automated testing and do not represent actual gameplay

## Episode Log Structure

Each episode directory contains:
- `episode_log.jsonl` - JSON Lines format log file (one JSON object per line)
- Optional: `.claude/settings.local.json` - Episode-specific settings

### JSONL Log Format

Each line in `episode_log.jsonl` is a JSON object with this structure:

```json
{
  "timestamp": "2025-11-01T21:48:15.986572",
  "level": "INFO",
  "message": "Human-readable message",
  "event_type": "orchestrator_init",
  "component": "map_manager",
  "turn": 0,
  // ... additional event-specific fields
}
```

**Standard Fields:**
- `timestamp` - ISO 8601 timestamp with microseconds
- `level` - Log level (INFO, WARNING, ERROR, DEBUG)
- `message` - Human-readable description
- `event_type` - Event category (optional)
- `component` - Component name (optional)
- `turn` - Turn number (optional)

**Common Event Types:**
- `orchestrator_init` - Episode initialization
- `episode_initialized` - Episode setup complete
- `map_restoration` - Map state loaded
- `progress` - Turn progress updates
- `info` - General informational events

## Parsing Logs with jq

### Basic Queries

**List all production episodes (exclude test episodes):**
```bash
ls game_files/episodes/ | grep -v "^test"
```

**View first 10 log entries:**
```bash
head -10 game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | jq '.'
```

**View last 5 log entries:**
```bash
tail -5 game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | jq '.'
```

### Event Type Filtering

**List all unique event types in an episode:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq -r '.event_type' | sort -u
```

**Filter by specific event type:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.event_type == "orchestrator_init")'
```

**Show only initialization events:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.event_type == "episode_initialized" or .event_type == "orchestrator_init")'
```

### Component Filtering

**List all unique components:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq -r '.component // empty' | sort -u
```

**Filter by component:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.component == "map_manager")'
```

**Show map_manager events only:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.component == "map_manager") | {turn, message}'
```

### Turn-Based Analysis

**Filter by turn number:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.turn == 5)'
```

**Show all events with turn numbers:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.turn) | {turn, event_type, message}'
```

**Get turn range (turns 10-20):**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.turn >= 10 and .turn <= 20)'
```

**Count events per turn:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq -r '.turn // empty' | sort -n | uniq -c
```

### Field Extraction

**Extract specific fields only:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq '{timestamp, level, message}'
```

**Extract nested fields (if present):**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.event_type == "map_restoration") | {version, rooms, connections}'
```

**Extract messages only:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq -r '.message'
```

### Complex Queries

**Find episodes with errors:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.level == "ERROR")'
```

**Count events by type:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq -r '.event_type // "none"' | sort | uniq -c | sort -rn
```

**Find memory-related events:**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.component == "simple_memory" or .message | contains("memory"))'
```

**Timeline of episode (timestamp + message):**
```bash
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq -r '[.timestamp, .message] | @tsv'
```

### Multi-Episode Analysis

**Find all production episodes sorted by date:**
```bash
ls -1 game_files/episodes/ | grep -v "^test" | sort
```

**Count total production episodes:**
```bash
ls -1 game_files/episodes/ | grep -v "^test" | wc -l
```

**Find most recent production episode:**
```bash
ls -1t game_files/episodes/ | grep -v "^test" | head -1
```

**Analyze all production episodes (example: count map restorations):**
```bash
for episode in game_files/episodes/2025-*/; do
  echo "$episode:"
  cat "$episode/episode_log.jsonl" | \
    jq 'select(.event_type == "map_restoration")' | wc -l
done
```

## Common Analysis Patterns

### Episode Summary
```bash
# Get episode metadata
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.event_type == "orchestrator_init") |
      {episode_id, agent_model, critic_model, max_turns}'
```

### Map State Tracking
```bash
# Find map restoration events
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.event_type == "map_restoration") |
      {rooms, connections, version}'
```

### Memory Loading
```bash
# Check memory initialization
cat game_files/episodes/2025-11-01T21:48:15/episode_log.jsonl | \
  jq 'select(.component == "simple_memory" and .turn == 0) |
      {locations, total_memories}'
```

## Best Practices

1. **Always filter out test episodes** when analyzing gameplay:
   ```bash
   ls game_files/episodes/ | grep -v "^test"
   ```

2. **Use compact output for exploration**:
   ```bash
   cat episode_log.jsonl | jq -c '.' | head -20
   ```

3. **Use pretty output for detailed analysis**:
   ```bash
   cat episode_log.jsonl | jq '.' | less
   ```

4. **Check for errors first** when debugging:
   ```bash
   cat episode_log.jsonl | jq 'select(.level == "ERROR" or .level == "WARNING")'
   ```

5. **Use field extraction for summaries**:
   ```bash
   cat episode_log.jsonl | jq '{turn, event_type, component, message}' | head -50
   ```

## Notes

- All timestamps are in ISO 8601 format with microsecond precision
- JSONL format = one JSON object per line (newline-delimited)
- Use `jq -c` for compact output, `jq '.'` for pretty output
- The `// empty` pattern in jq filters out null values
- Test episodes are automatically created by the test suite and should be ignored for gameplay analysis
