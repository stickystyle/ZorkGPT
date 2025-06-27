# Autosave Backend Process Leak Fix Documentation

## Problem Description

The ZorkGPT system was experiencing an issue where autosave backend prompts were leaking into normal game responses. This was observed in episode `2025-06-12T09:44:10` at turn 20, where instead of a normal game response to the command "south", the system returned:

```
"Please enter a filename [autosave_2025-06-12T09:44:10]: Overwrite existing file? Failed."
```

### Root Cause Analysis

1. **Multi-Step Save Process**: The Zork `save` command is not a simple request/response operation. It involves multiple interactive prompts:
   - Initial command: `save`
   - Zork prompts: "Please enter a filename [default.qzl]: "
   - User provides filename
   - If file exists, Zork prompts: "Overwrite existing file? "
   - User responds with "y" or "n"
   - Zork responds with "Ok." or error message

2. **Autosave Timing**: The game server (`game_server.py`) triggers autosaves:
   - Every 10 turns (line 134)
   - When the score changes (line 160)
   - When a session is closed

3. **Race Condition**: The `trigger_zork_save()` method in `zork_api.py` was clearing the response queue immediately after detecting success, but:
   - The reader thread continues to run asynchronously
   - New output from dfrotz could still be arriving after the queue was cleared
   - These late-arriving save prompts would then appear in the next game command's response

## Investigation Steps

1. **Examined Episode Log**: Found the leak at turn 20 in `game_files/episodes/2025-06-12T09:44:10/episode_log.jsonl`

2. **Traced Autosave Flow**:
   - `game_server.py`: `_trigger_save()` method calls `zork.trigger_zork_save()`
   - `zork_api.py`: `trigger_zork_save()` sends save commands and attempts to clear queue
   - Issue: `clear_response_queue()` only empties the queue at that instant

3. **Identified the Gap**: The reader thread (`_reader()` method) continuously reads from the dfrotz process and adds lines to the queue. There was no synchronization to ensure all save-related output had been consumed.

## Solution Implementation

### 1. Enhanced `clear_response_queue()` Method

**Original Implementation**:
```python
def clear_response_queue(self):
    """Clear any pending responses from the queue."""
    while not self.response_queue.empty():
        self.response_queue.get()
```

**Fixed Implementation**:
```python
def clear_response_queue(self):
    """Clear any pending responses from the queue."""
    # First clear what's currently in the queue
    while not self.response_queue.empty():
        self.response_queue.get()
    
    # Wait a bit for any remaining output to arrive from the reader thread
    time.sleep(0.1)
    
    # Clear again to catch any new output that arrived during the delay
    while not self.response_queue.empty():
        self.response_queue.get()
```

### 2. More Aggressive Clearing After Successful Saves

In the `trigger_zork_save()` method, after detecting success indicators:

**Original**:
```python
# Clear any remaining save-related output from the queue
self.clear_response_queue()
return True
```

**Fixed**:
```python
# Clear any remaining save-related output from the queue more aggressively
self.clear_response_queue()
# Wait a bit longer to ensure all save-related output has been processed
time.sleep(0.2)
self.clear_response_queue()
return True
```

## Reasoning for the Fix

1. **Double-Clear Pattern**: By clearing the queue twice with a delay between, we catch output that arrives after the first clear but before we return control to the game flow.

2. **Time Delays**: 
   - 0.1s in `clear_response_queue()`: Short enough to not impact game performance, long enough for the reader thread to capture trailing output
   - 0.2s additional delay after save success: Provides extra insurance for the more complex save operation

3. **Minimal Impact**: The delays only occur during autosaves (every 10 turns or on score changes), not during normal gameplay.

## Files Modified

- `/Volumes/workingfolder/ZorkGPT/zork_api.py`:
  - Modified `clear_response_queue()` method (lines 101-112)
  - Modified `trigger_zork_save()` method (lines 291-305)

## Testing Recommendations

1. Run a game session and verify autosaves occur at turn 10, 20, 30, etc.
2. Check that save prompts no longer appear in game responses
3. Verify that legitimate game responses are not being lost
4. Test with rapid command sequences to ensure the delays don't cause issues

## Alternative Solutions Considered

1. **Marker-based approach**: Insert a unique marker after save operations and read until that marker is seen
   - Rejected: More complex and could interfere with legitimate game output

2. **Disable output during saves**: Temporarily stop the reader thread
   - Rejected: Could miss important game state changes

3. **Separate save process**: Use a different dfrotz instance for saves
   - Rejected: Would require significant architectural changes

## Future Improvements

1. Consider implementing a more sophisticated queue management system that can distinguish between save-related output and game output
2. Add logging to track when save output leaks occur for monitoring
3. Consider making the delay times configurable for different system performances

## Related Code Context

- Autosave triggers: `game_server.py` lines 131-197
- Save implementation: `zork_api.py` lines 194-300
- Response queue management: `zork_api.py` lines 42-65 (reader thread)
- Episode logging: Used to diagnose the issue in production