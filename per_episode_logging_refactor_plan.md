# Per-Episode Logging Refactor Implementation Plan

## Instructions
Check off each checkbox `[ ]` → `[x]` as you complete the corresponding stage or sub-stage in this implementation plan. This will help track progress and ensure no steps are missed during the refactor process.

---

## Project Overview

### Current Problems

1. **Growing Monolithic Log File**: The `zork_episode_log.jsonl` file grows continuously as episodes accumulate, containing data from all episodes in a single file. This creates:
   - Performance degradation when reading logs (must filter entire file for single episode)
   - Scalability issues as file size grows unbounded
   - Difficulty in analyzing individual episodes
   - Backup and management challenges

2. **Episode ID Initialization Timing Issue**: Currently, `episode_id` is generated during `play_episode()` execution, but logging begins during orchestrator initialization. This creates:
   - Early logs (orchestrator init, component init) have empty `episode_id` (`""`)
   - Inconsistent log context during startup phase
   - Need for complex "update episode_id" propagation logic

### Current Architecture Analysis

#### Episode ID Generation Flow
- **Location**: `zork_orchestrator_v2.py:229` in `play_episode()` method
- **Format**: ISO8601 timestamp (`YYYY-MM-DDTHH:MM:SS`)
- **Propagation**: Manual update to agent, critic, extractor via `update_episode_id()` calls
- **Timing**: After orchestrator initialization, during episode execution

#### JSON Logging System
- **Current File**: Single monolithic `zork_episode_log.jsonl` (configured in `config.py:134`)
- **Format**: JSONL (JSON Lines) - one JSON object per line
- **Content**: All episode data mixed together, filtered by `episode_id` field
- **Writer**: `JSONFormatter` in `logger.py` with shared file handler
- **Primary Reader**: `zork_strategy_generator.py:_extract_turn_window_data()` method

#### Key Components Affected
- **ZorkOrchestratorV2**: `/Volumes/workingfolder/ZorkGPT/orchestration/zork_orchestrator_v2.py`
- **Logger Setup**: `/Volumes/workingfolder/ZorkGPT/logger.py`
- **Knowledge Synthesis**: `/Volumes/workingfolder/ZorkGPT/zork_strategy_generator.py`
- **Main Entry**: `/Volumes/workingfolder/ZorkGPT/main.py`
- **Configuration**: `/Volumes/workingfolder/ZorkGPT/config.py`

## Desired End State

### Episode ID Architecture
- `episode_id` generated in `main.py` and passed as mandatory constructor parameter
- `ZorkOrchestratorV2(episode_id=episode_id)` - clean dependency injection
- All components receive proper `episode_id` from initialization
- No need for `update_episode_id()` propagation methods

### Logging Architecture
- Per-episode log files: `{workdir}/episodes/{episode_id}/episode_log.jsonl`
- Directory structure mirrors episode organization
- Knowledge synthesis reads episodes chronologically from individual files
- Clean separation enables individual episode analysis

### Directory Structure
```
game_files/
├── episodes/
│   ├── 2024-01-10T14:30:00/
│   │   └── episode_log.jsonl
│   ├── 2024-01-10T15:45:00/
│   │   └── episode_log.jsonl
│   └── 2024-01-11T09:15:30/
│       └── episode_log.jsonl
├── autosave_* files
└── zork_episode_log.jsonl (legacy, post-migration)
```

## Implementation Plan

### Stage 0: Add Required CLI Arguments
- [x] Add `--max-turns` argument for limiting episode length
- [x] Add `--episodes` argument for running multiple episodes
- [x] Update argument parsing and main logic

**Files to Modify**: `main.py`

**Changes Required**:
1. Add `--max-turns` argument for limiting episode length
2. Add `--episodes` argument for running multiple episodes
3. Update argument parsing and main logic

**Implementation Details**:
```python
def main():
    parser = argparse.ArgumentParser(description="ZorkGPT Agent")
    # ... existing arguments ...
    
    # Add new arguments
    parser.add_argument("--max-turns", type=int, default=None, 
                       help="Maximum number of turns per episode")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run")
    
    args = parser.parse_args()
    
    # Update logic to use max_turns and episodes
    if args.episodes > 1:
        run_multiple_episodes(args.episodes, max_turns=args.max_turns)
    else:
        run_episode(max_turns=args.max_turns, episode_id=args.episode)
```

### Stage 1: Episode ID Refactor
- [x] Complete all Stage 1 sub-tasks

#### 1.1 Update ZorkOrchestratorV2 Constructor
- [x] Add `episode_id: str` as mandatory constructor parameter
- [x] Remove episode_id generation from `play_episode()` method (line ~229)
- [x] Set `self.game_state.episode_id = episode_id` immediately after GameState creation
- [x] Remove `update_episode_id()` calls from `initialize_episode()`

**Files to Modify**: `orchestration/zork_orchestrator_v2.py`

**Changes Required**:
1. Add `episode_id: str` as mandatory constructor parameter
2. Remove episode_id generation from `play_episode()` method (line ~229)
3. Set `self.game_state.episode_id = episode_id` immediately after GameState creation
4. Remove `update_episode_id()` calls from `initialize_episode()` since components get correct ID from start

**Implementation Details**:
```python
# Current constructor signature:
def __init__(self, **kwargs):

# New constructor signature:
def __init__(self, episode_id: str, **kwargs):
    # ... existing initialization ...
    self.game_state = GameState()
    self.game_state.episode_id = episode_id  # Set immediately
    
    # Components now get correct episode_id from game_state
    self.agent = Agent(episode_id=episode_id, ...)
    self.critic = Critic(episode_id=episode_id, ...)
    self.extractor = Extractor(episode_id=episode_id, ...)
```

**Validation Steps**:
- Verify orchestrator initialization completes without errors
- Check that `game_state.episode_id` is set correctly
- Confirm all components receive the episode_id during initialization

#### 1.2 Update Main Entry Point
- [x] Generate `episode_id` before orchestrator creation
- [x] Pass `episode_id` to `ZorkOrchestratorV2` constructor
- [x] Update episode restoration logic to pass restored `episode_id`
- [x] Update `run_multiple_episodes()` to create new orchestrator per episode

**Files to Modify**: `main.py`

**Changes Required**:
1. Generate `episode_id` before orchestrator creation
2. Pass `episode_id` to `ZorkOrchestratorV2` constructor
3. Update episode restoration logic to pass restored `episode_id`
4. Update `run_multiple_episodes()` to create new orchestrator per episode

**Implementation Details**:
```python
# New episode creation
def run_episode(max_turns=None, episode_id=None):
    if episode_id is None:
        episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    orchestrator = ZorkOrchestratorV2(episode_id=episode_id)
    # Pass max_turns to orchestrator or episode configuration
    # ... rest of logic

# Multiple episodes - create new orchestrator per episode
def run_multiple_episodes(num_episodes, max_turns=None):
    for i in range(num_episodes):
        episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        orchestrator = ZorkOrchestratorV2(episode_id=episode_id)
        # ... run episode with max_turns
        # orchestrator goes out of scope, new one created for next episode
```

#### 1.3 Update Test Files
- [x] Add `episode_id` parameter to all `ZorkOrchestratorV2` instantiations
- [x] Use test-appropriate episode IDs (e.g., `"test_episode_001"`)

**Files to Modify**: `tests/test_integration.py`, `demo_refactored_system.py`

**Changes Required**:
1. Add `episode_id` parameter to all `ZorkOrchestratorV2` instantiations
2. Use test-appropriate episode IDs (e.g., `"test_episode_001"`)

**Implementation Details**:
```python
# Test fixture update
def create_test_orchestrator():
    episode_id = f"test_episode_{int(time.time())}"
    return ZorkOrchestratorV2(
        episode_id=episode_id,
        episode_log_file=temp_files["episode_log"],
        # ... other config
    )
```

#### 1.4 Testing Stage 1
- [x] Run unit tests and verify they pass
- [x] Run demo to verify end-to-end flow
- [x] Run single episode test with live game server
- [x] Verify all log entries contain consistent episode_id

**Test Strategy**:
1. **Unit Tests**: Verify orchestrator constructor accepts episode_id
2. **Integration Tests**: Run full episode with game server
3. **Validation**: Confirm episode_id consistency across all components

**Test Commands**:
```bash
# Run unit tests
python -m pytest tests/test_integration.py -v

# Run demo to verify end-to-end flow
python demo_refactored_system.py

# Run single episode test with live game server
python main.py --max-turns 5
```

**Success Criteria**:
- All tests pass
- Episode runs successfully from start to finish
- All log entries contain consistent episode_id
- No "update_episode_id" calls needed

### Stage 2: Logging Refactor
- [x] Complete all Stage 2 sub-tasks

#### 2.1 Update Logger Setup
- [x] Add `setup_episode_logging(episode_id: str, workdir: str)` function
- [x] Create episode directory structure
- [x] Switch JSON handler to episode-specific file
- [x] Maintain backward compatibility for existing `setup_logging()`

**Files to Modify**: `logger.py`

**Changes Required**:
1. Add `setup_episode_logging(episode_id: str, workdir: str)` function
2. Create episode directory structure
3. Switch JSON handler to episode-specific file
4. Maintain backward compatibility for existing `setup_logging()`

**Implementation Details**:
```python
def setup_episode_logging(episode_id: str, workdir: str = "game_files"):
    """
    Setup logging for a specific episode.
    Creates episode directory and configures JSON handler for episode-specific logging.
    """
    import os
    from pathlib import Path
    
    # Create episode directory
    episode_dir = Path(workdir) / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Episode-specific log file
    episode_log_file = episode_dir / "episode_log.jsonl"
    
    # Update existing JSON handler or create new one
    logger = logging.getLogger()
    
    # Remove existing JSON handler if present
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and hasattr(handler, 'is_json_handler'):
            logger.removeHandler(handler)
    
    # Create episode-specific JSON handler
    json_handler = logging.FileHandler(episode_log_file, mode='a', encoding='utf-8')
    json_handler.setFormatter(JSONFormatter())
    json_handler.setLevel(logging.INFO)
    json_handler.is_json_handler = True  # Mark for identification
    
    logger.addHandler(json_handler)
    
    return str(episode_log_file)
```

#### 2.2 Integration with Orchestrator
- [x] Call `setup_episode_logging()` after episode_id is set
- [x] Store episode log file path for reference
- [x] Update configuration handling

**Files to Modify**: `orchestration/zork_orchestrator_v2.py`

**Changes Required**:
1. Call `setup_episode_logging()` after episode_id is set
2. Store episode log file path for reference
3. Update configuration handling

**Implementation Details**:
```python
def __init__(self, episode_id: str, **kwargs):
    # ... existing initialization ...
    self.game_state = GameState()
    self.game_state.episode_id = episode_id
    
    # Setup episode-specific logging
    workdir = self.config.files.zork_game_workdir
    self.episode_log_file = setup_episode_logging(episode_id, workdir)
    
    # Log orchestrator initialization with proper episode context
    logger.info("ZorkOrchestrator v2 initialized", extra={
        "event_type": "orchestrator_initialized",
        "episode_id": episode_id,
        "episode_log_file": self.episode_log_file
    })
```

#### 2.3 Update Knowledge Synthesis
- [x] Update `_extract_turn_window_data()` to read from episode-specific file
- [x] Add `_get_all_episode_ids()` function to scan episodes directory
- [x] Add wrapper function for processing all episodes chronologically
- [x] Remove episode_id filtering logic (no longer needed)

**Files to Modify**: `zork_strategy_generator.py`

**Changes Required**:
1. Update `_extract_turn_window_data()` to read from episode-specific file
2. Add `_get_all_episode_ids()` function to scan episodes directory
3. Add wrapper function for processing all episodes chronologically
4. Remove episode_id filtering logic (no longer needed)

**Implementation Details**:
```python
def _get_all_episode_ids(self, workdir: str = "game_files") -> List[str]:
    """
    Scan episodes directory and return all episode IDs in chronological order.
    """
    from pathlib import Path
    
    episodes_dir = Path(workdir) / "episodes"
    if not episodes_dir.exists():
        return []
    
    # Get all episode directories
    episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
    
    # Sort chronologically (episode IDs are ISO8601 timestamps)
    episode_ids = sorted([d.name for d in episode_dirs])
    
    return episode_ids

def _extract_turn_window_data(self, episode_id: str, workdir: str = "game_files") -> Optional[Dict]:
    """
    Extract turn window data from specific episode file.
    No longer needs episode_id filtering since each file contains only one episode.
    """
    from pathlib import Path
    import json
    
    episode_log_file = Path(workdir) / "episodes" / episode_id / "episode_log.jsonl"
    
    if not episode_log_file.exists():
        logger.warning(f"Episode log file not found: {episode_log_file}")
        return None
    
    try:
        with open(episode_log_file, "r", encoding="utf-8") as f:
            for line in f:
                log_entry = json.loads(line.strip())
                # Process log entry - no episode_id filtering needed
                # ... existing processing logic ...
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"Error reading episode log {episode_log_file}: {e}")
        return None

def process_all_episodes_chronologically(self) -> List[Dict]:
    """
    Process all episodes in chronological order.
    """
    all_episode_data = []
    episode_ids = self._get_all_episode_ids()
    
    for episode_id in episode_ids:
        episode_data = self._extract_turn_window_data(episode_id)
        if episode_data:
            all_episode_data.append(episode_data)
    
    return all_episode_data
```

#### 2.4 Update Utility Functions
- [x] Add `parse_episode_logs()` function for single episode
- [x] Add `parse_all_episode_logs()` function for multiple episodes
- [x] Maintain existing `parse_json_logs()` for backward compatibility

**Files to Modify**: `logger.py`

**Changes Required**:
1. Add `parse_episode_logs()` function for single episode
2. Add `parse_all_episode_logs()` function for multiple episodes
3. Maintain existing `parse_json_logs()` for backward compatibility

**Implementation Details**:
```python
def parse_episode_logs(episode_id: str, workdir: str = "game_files") -> List[Dict[str, Any]]:
    """
    Parse logs from a specific episode.
    """
    from pathlib import Path
    import json
    
    episode_log_file = Path(workdir) / "episodes" / episode_id / "episode_log.jsonl"
    
    if not episode_log_file.exists():
        return []
    
    logs = []
    try:
        with open(episode_log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass
    
    return logs

def parse_all_episode_logs(workdir: str = "game_files") -> List[Dict[str, Any]]:
    """
    Parse logs from all episodes in chronological order.
    """
    from pathlib import Path
    
    episodes_dir = Path(workdir) / "episodes"
    if not episodes_dir.exists():
        return []
    
    # Get all episode IDs in chronological order
    episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
    episode_ids = sorted([d.name for d in episode_dirs])
    
    all_logs = []
    for episode_id in episode_ids:
        episode_logs = parse_episode_logs(episode_id, workdir)
        all_logs.extend(episode_logs)
    
    return all_logs
```

#### 2.5 Testing Stage 2
- [x] Verify episode directories are created
- [x] Confirm logs go to correct episode-specific file
- [x] Test reading from episode-specific files
- [x] Run complete episode with game server

**Test Strategy**:
1. **Directory Creation**: Verify episode directories are created
2. **Log File Writing**: Confirm logs go to correct episode-specific file
3. **Knowledge Synthesis**: Test reading from episode-specific files
4. **Full Integration**: Run complete episode with game server

**Test Commands**:
```bash
# Test episode directory creation
python -c "
from logger import setup_episode_logging
setup_episode_logging('test_episode_001')
import os
assert os.path.exists('game_files/episodes/test_episode_001/episode_log.jsonl')
print('Episode directory creation: PASS')
"

# Test full episode with logging
python main.py --max-turns 10

# Verify episode log exists and contains data
EPISODE_ID=$(ls game_files/episodes/ | head -1)
echo "Testing episode: $EPISODE_ID"

# Check log file exists and has content
wc -l "game_files/episodes/$EPISODE_ID/episode_log.jsonl"

# Test episode restoration
python main.py --episode "$EPISODE_ID" --max-turns 5

# Test multiple episodes
python main.py --episodes 3 --max-turns 10

# Test knowledge synthesis reading
python -c "
from zork_strategy_generator import AdaptiveKnowledgeManager
# Test with actual episode file
"
```

**Success Criteria**:
- Episode directories created automatically
- Logs written to episode-specific files
- Knowledge synthesis reads from individual files successfully
- All episode data preserved and accessible

### Stage 3: Migration Strategy
- [x] Complete all Stage 3 sub-tasks

#### 3.1 Create Migration Script
- [x] Create `migrate_episode_logs.py` script
- [x] Implement log splitting functionality
- [x] Add dry-run capability
- [x] Add proper error handling and logging

**New File**: `migrate_episode_logs.py`

**Purpose**: Split existing monolithic `zork_episode_log.jsonl` into per-episode files

**Implementation**:
```python
#!/usr/bin/env python3
"""
Migration script to split monolithic episode log into per-episode files.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import logging

def migrate_episode_logs(input_file: str, workdir: str = "game_files"):
    """
    Split monolithic episode log into per-episode files.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        return False
    
    # Group log entries by episode_id
    episode_logs = defaultdict(list)
    total_entries = 0
    
    logger.info(f"Reading logs from {input_file}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    log_entry = json.loads(line)
                    episode_id = log_entry.get('episode_id', 'unknown')
                    
                    # Skip entries without episode_id or with empty episode_id
                    if not episode_id or episode_id == '':
                        logger.warning(f"Line {line_num}: No episode_id, skipping")
                        continue
                    
                    episode_logs[episode_id].append(log_entry)
                    total_entries += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error: {e}")
                    continue
    
    except IOError as e:
        logger.error(f"Error reading input file: {e}")
        return False
    
    logger.info(f"Processed {total_entries} log entries across {len(episode_logs)} episodes")
    
    # Create episode directories and write files
    episodes_dir = Path(workdir) / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    migrated_episodes = 0
    
    for episode_id, logs in episode_logs.items():
        episode_dir = episodes_dir / episode_id
        episode_dir.mkdir(exist_ok=True)
        
        episode_log_file = episode_dir / "episode_log.jsonl"
        
        try:
            with open(episode_log_file, 'w', encoding='utf-8') as f:
                for log_entry in logs:
                    f.write(json.dumps(log_entry) + '\n')
            
            logger.info(f"Migrated episode {episode_id}: {len(logs)} entries")
            migrated_episodes += 1
            
        except IOError as e:
            logger.error(f"Error writing episode file {episode_log_file}: {e}")
            continue
    
    logger.info(f"Migration complete: {migrated_episodes} episodes migrated")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate monolithic episode log to per-episode files")
    parser.add_argument("input_file", help="Path to monolithic episode log file")
    parser.add_argument("--workdir", default="game_files", help="Working directory for episodes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without doing it")
    
    args = parser.parse_args()
    
    if args.dry_run:
        # Implement dry-run logic
        print(f"DRY RUN: Would migrate {args.input_file} to {args.workdir}/episodes/")
    else:
        success = migrate_episode_logs(args.input_file, args.workdir)
        exit(0 if success else 1)
```

#### 3.2 Migration Testing
- [x] Create backup of original log
- [x] Run migration script
- [x] Verify episode files created correctly
- [x] Validate data integrity

**Test Strategy**:
1. Create test episode log with multiple episodes
2. Run migration script
3. Verify episode files created correctly
4. Validate data integrity

**Test Commands**:
```bash
# Create backup of original log
cp zork_episode_log.jsonl zork_episode_log.jsonl.backup

# Run migration
python migrate_episode_logs.py zork_episode_log.jsonl

# Verify migration
ls -la game_files/episodes/

# Test parsing migrated files
python -c "
from logger import parse_all_episode_logs
logs = parse_all_episode_logs()
print(f'Total migrated logs: {len(logs)}')
"
```

### Stage 4: Validation and Testing
- [x] Complete all Stage 4 sub-tasks

#### 4.1 Comprehensive Integration Testing
- [x] Test new episode creation and logging
- [x] Test episode restoration with log continuity
- [x] Test multiple episodes with separate logs
- [x] Test knowledge synthesis with new structure
- [x] Test error handling for missing files and corrupted logs

**Test Scenarios**:
1. **New Episode**: Start fresh episode, verify logging
2. **Episode Restoration**: Restore existing episode, verify log continuity
3. **Multiple Episodes**: Run multiple episodes, verify separate logs
4. **Knowledge Synthesis**: Verify adaptive learning works with new structure
5. **Error Handling**: Test missing files, corrupted logs

#### 4.2 Live Game Server Testing
- [x] Start game server and run test episodes
- [x] Verify game state persistence and logging
- [x] Test episode restoration functionality
- [x] Test multiple episode scenarios
- [x] Test knowledge synthesis integration

**Setup Required**:
1. Start game server: `python game_server.py`
2. Run test episodes with various configurations
3. Verify game state persistence and logging

**Test Commands**:
```bash
# Start game server in background
python game_server.py &
GAME_SERVER_PID=$!

# Run single episode test
python main.py --max-turns 20

# Verify episode log created
EPISODE_ID=$(ls game_files/episodes/ | head -1)
echo "Testing episode: $EPISODE_ID"

# Check log file exists and has content
wc -l "game_files/episodes/$EPISODE_ID/episode_log.jsonl"

# Test episode restoration
python main.py --episode "$EPISODE_ID" --max-turns 5

# Test multiple episodes
python main.py --episodes 3 --max-turns 10

# Test knowledge synthesis
python -c "
from zork_strategy_generator import AdaptiveKnowledgeManager
manager = AdaptiveKnowledgeManager()
data = manager.process_all_episodes_chronologically()
print(f'Processed {len(data)} episodes')
"

# Cleanup
kill $GAME_SERVER_PID
```

#### 4.3 Performance Validation
- [x] Measure log file creation time
- [x] Compare knowledge synthesis reading time (before vs after)
- [x] Monitor memory usage during log processing
- [x] Track disk space usage patterns

**Metrics to Measure**:
1. Log file creation time
2. Knowledge synthesis reading time (before vs after)
3. Memory usage during log processing
4. Disk space usage patterns

**Performance Test**:
```bash
# Time knowledge synthesis before migration
time python -c "
from zork_strategy_generator import AdaptiveKnowledgeManager
manager = AdaptiveKnowledgeManager()
# Use old monolithic file method
"

# Time knowledge synthesis after migration
time python -c "
from zork_strategy_generator import AdaptiveKnowledgeManager  
manager = AdaptiveKnowledgeManager()
data = manager.process_all_episodes_chronologically()
"
```

## Risk Mitigation

### Rollback Strategy

1. **Keep Original Log File**: Migration script preserves original monolithic file
2. **Backward Compatibility**: Old code can still read monolithic file if needed
3. **Configuration Toggle**: Add config option to switch between old/new logging

### Error Handling

1. **Missing Episode Directories**: Create automatically during logging
2. **Corrupted Episode Files**: Skip with warning, continue processing
3. **Disk Space Issues**: Monitor and alert on episode directory growth
4. **Permission Issues**: Clear error messages for file system problems

### Monitoring

1. **Episode Directory Growth**: Track number of episode directories
2. **Log File Sizes**: Monitor individual episode log file sizes
3. **Knowledge Synthesis Performance**: Track processing time trends
4. **Error Rates**: Monitor failed episode processing

## Success Criteria

### Functional Requirements
- [x] Episode ID available from orchestrator initialization
- [x] All logs written to episode-specific files
- [x] Knowledge synthesis reads from individual episode files
- [x] Episode restoration works with new logging structure
- [x] Multiple episodes create separate log files

### Quality Requirements
- [x] All existing functionality preserved
- [x] No data loss during migration
- [x] Error handling for edge cases
- [x] Clear logging for debugging

## Post-Implementation Tasks

- [ ] **Update Documentation**: Update code comments
- [ ] **Cleanup Old Code**: Remove unused episode_id update methods
- [ ] **Archive Old Logs**: Move monolithic files to archive directory

---

## Implementation Notes for Future Claude

### Context You'll Need
- This refactor addresses scalability issues with growing log files
- Episode IDs are ISO8601 timestamps used for chronological ordering
- The game server runs separately - ensure it's running for integration tests
- Knowledge synthesis is critical - it must work with new file structure
- Backward compatibility maintained through migration script only

### Critical Implementation Details
1. **Episode ID Generation**: Must happen in main.py before orchestrator creation
2. **Directory Creation**: Use `mkdir(parents=True, exist_ok=True)` for safety
3. **JSONL Format**: Maintain exact format - one JSON object per line
4. **Error Handling**: Log warnings but continue processing for missing files
5. **Chronological Order**: Sort episode IDs as strings (ISO8601 sorts correctly)

### Testing Strategy
- Always test with live game server for true fidelity
- Verify log continuity during episode restoration
- Test edge cases like missing files, corrupted JSON
- Measure performance before/after for validation

### Common Pitfalls to Avoid
- Don't break existing episode restoration functionality
- Ensure episode directories are created atomically
- Handle empty or malformed episode IDs gracefully
- Maintain exact JSON format compatibility
- Test multiple episode scenarios thoroughly

### Required CLI Arguments to Add
- `--max-turns`: Add to main.py argument parser and pass to orchestrator/episode logic
- `--episodes`: Add to main.py argument parser for multiple episode runs
- Note: `--episode` already exists for episode restoration

This plan provides comprehensive guidance for implementing per-episode logging while maintaining system reliability and performance.