# Critic Multi-Object Validation Fix

## Issue Summary

In episode `2025-10-30T11:42:07`, turn 34, the critic incorrectly rejected the action `"take lunch, garlic, bottle"` with the error message:

```
[Object Tree Validation] Object 'lunch, garlic, bottle' is not visible in current location
```

However, the action actually succeeded in the game:
```
lunch: Taken. clove of garlic: Taken. glass bottle: Taken.
```

## Root Causes

### 1. Multi-Object Command Parsing
The validator treated "lunch, garlic, bottle" as a single object name instead of three separate objects. Zork supports comma-separated multi-object commands like `take X, Y, Z`, but the validation code wasn't handling this pattern.

### 2. Container Visibility
Objects inside open/transparent containers weren't being included in the visibility check. The validator only checked objects directly in the location, not objects in containers (even when opened).

### 3. Unreliable Takeable Attributes
Jericho's Z-machine attributes for `takeable` and `portable` don't always match what's actually possible in the game. For example, the `lunch` object has `takeable: False` but can actually be taken.

## Changes Made

### `zork_critic.py:627-729` - `_validate_against_object_tree()`

1. **Multi-Object Parsing** (Line 670)
   - Split target string on commas: `targets = [t.strip() for t in target.split(',')]`
   - Validate each object individually

2. **Recursive Container Visibility** (Lines 675-699)
   - Check objects directly in location
   - Recursively add children of transparent objects (open containers)
   - Handles nested containers (e.g., table → sack → lunch)

3. **Removed Unreliable Takeable Check** (Lines 701-718)
   - Only validate object visibility, not takeability
   - Jericho's `takeable` attribute is unreliable
   - Let the game itself reject truly un-takeable objects

## Code Example

**Before:**
```python
# Failed on "take lunch, garlic, bottle" because:
target = "lunch, garlic, bottle"  # Treated as one object name
# Looked for object named "lunch, garlic, bottle" (doesn't exist)
```

**After:**
```python
# Correctly handles multi-object commands:
targets = ["lunch", "garlic", "bottle"]  # Split by comma
# Validates each object individually

# Recursively checks transparent containers:
for obj in visible_objects:
    if attrs.get('transparent'):
        # Add children (objects inside open container)
        add_accessible_children(obj)
```

## Testing

Created comprehensive test suite in `tests/test_critic_multiobject.py`:

- ✅ Single object commands
- ✅ Multi-object comma-separated commands
- ✅ Invalid objects in multi-object commands
- ✅ Objects in open containers
- ✅ Commands with extra whitespace

All 6 tests passing.

## Impact

- **Reduces false negatives**: Valid multi-object commands no longer incorrectly rejected
- **Better container handling**: Objects in open containers properly validated
- **More lenient validation**: Only checks visibility (fast, reliable) not takeability (unreliable)
- **Maintains performance**: Still provides fast Z-machine validation before expensive LLM calls

## Files Changed

- `zork_critic.py` - Updated `_validate_against_object_tree()` method
- `tests/test_critic_multiobject.py` - New comprehensive test suite (6 tests)
