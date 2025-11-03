# Testing Guide

This guide covers testing patterns, fixtures, and best practices for ZorkGPT testing.

## Testing Philosophy

**Test Determinism**: All tests should be deterministic and reproducible. Use walkthrough fixtures for game state testing rather than random exploration.

**Integration Focus**: Prefer integration tests that validate multi-component behavior over isolated unit tests, especially for LLM-driven systems.

**Fast Feedback**: Tests should run quickly. Use `pytest -k "not slow"` to skip long-running tests during development.

## Walkthrough Fixtures

The most important testing tool for ZorkGPT. Located in `fixtures/walkthrough.py`.

### Available Fixtures

```python
from tests.fixtures.walkthrough import (
    get_zork1_walkthrough,
    get_walkthrough_slice,
    get_walkthrough_until_lamp,
    get_walkthrough_dark_sequence,
    replay_walkthrough
)

# Full walkthrough from Jericho (350/350 score)
full_walkthrough = get_zork1_walkthrough()

# Subset of actions
first_20_actions = get_walkthrough_slice(0, 20)

# Pre-defined useful sequences
lamp_sequence = get_walkthrough_until_lamp()  # First ~15 actions
dark_sequence = get_walkthrough_dark_sequence()  # Dark room navigation

# Execute sequence and collect results
results = replay_walkthrough(env, actions)
```

### Example Test Pattern

```python
from tests.fixtures.walkthrough import get_walkthrough_slice
from game_interface.core.jericho_interface import JerichoInterface

def test_location_id_stability():
    """Verify location IDs are deterministic across replays."""
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    walkthrough = get_walkthrough_slice(0, 20)

    # First run
    location_ids = []
    for action in walkthrough:
        interface.send_command(action)
        loc = interface.get_location_structured()
        location_ids.append(loc.num)

    # Second run should match exactly
    interface2 = JerichoInterface(rom_path="infrastructure/zork.z5")
    replay_ids = []
    for action in walkthrough:
        interface2.send_command(action)
        loc = interface2.get_location_structured()
        replay_ids.append(loc.num)

    assert location_ids == replay_ids, "Location IDs must be deterministic"
```

## Test Organization

### Directory Structure

```
tests/
├── CLAUDE.md                              # This file
├── conftest.py                            # Shared fixtures
├── fixtures/
│   └── walkthrough.py                     # Walkthrough fixtures
├── simple_memory/                         # SimpleMemoryManager tests
│   ├── conftest.py                        # Memory-specific fixtures
│   ├── test_simple_memory_synthesis.py    # Synthesis logic
│   ├── test_simple_memory_formatting.py   # Formatting helpers
│   ├── test_movement_memory_location.py   # Source location storage
│   └── test_simple_memory_status.py       # Status transitions
├── test_map_persistence.py                # MapGraph save/load
├── test_phase5_enhanced_context.py        # Context management
├── test_multi_step_window_sequence.py     # Multi-step prerequisites
└── test_multi_step_delayed_consequence.py # Delayed consequences
```

### Naming Conventions

- **Unit tests**: `test_<component>_<feature>.py`
- **Integration tests**: `test_<feature>_integration.py`
- **Phase tests**: `test_phase<N>_<feature>.py` (tracking refactoring phases)
- **Test functions**: `test_<what_it_validates>`

## Running Tests

### Quick Commands

```bash
# Run all tests (fast)
uv run pytest tests/ -k "not slow" -q

# Run specific test file
uv run pytest tests/test_map_persistence.py -v

# Run tests matching pattern
uv run pytest tests/ -k "memory" -v

# Run with detailed output
uv run pytest tests/test_simple_memory_integration.py -xvs

# Stop on first failure
uv run pytest tests/ --maxfail=1

# Run only failed tests from last run
uv run pytest --lf
```

### Performance Validation

```bash
# Run benchmarks
uv run python benchmarks/comparison_report.py

# Individual benchmark
uv run python benchmarks/performance_metrics.py
```

## Test Coverage Requirements

**Critical paths that MUST have tests:**
1. Memory synthesis (multi-step, supersession, status transitions)
2. Map persistence (save/load, room deduplication, connection tracking)
3. Movement detection (ID comparison, dark rooms, teleportation)
4. Manager lifecycle (init, reset, processing)
5. Context assembly (reasoning history, game state, objectives)

## Debugging Failed Tests

### Common Issues

**Flaky tests due to LLM variation:**
- Use deterministic fixtures (walkthrough) instead of live LLM calls
- Mock LLM responses for unit tests
- Use temperature=0 for reproducibility in integration tests

**File persistence issues:**
- Use temp directories with pytest's `tmp_path` fixture
- Clean up files in teardown or use context managers
- Don't assume files exist from previous test runs

**Game state pollution:**
- Always create fresh JerichoInterface per test
- Reset game_state between tests
- Use pytest fixtures for proper cleanup

### Debug Commands

```bash
# Print full output
uv run pytest tests/test_failing.py -xvs --tb=short

# Show local variables on failure
uv run pytest tests/test_failing.py -l

# Enter debugger on failure
uv run pytest tests/test_failing.py --pdb

# Run with logging output
uv run pytest tests/test_failing.py -v --log-cli-level=DEBUG
```

## Writing New Tests

### Checklist

- [ ] Test is deterministic (no random behavior)
- [ ] Test uses walkthrough fixtures for game state
- [ ] Test has clear assertion messages
- [ ] Test cleans up any created files
- [ ] Test follows naming conventions
- [ ] Test is in appropriate subdirectory
- [ ] Test validates both success and failure cases
- [ ] Test runs quickly (< 1 second for unit tests)

### Example Template

```python
def test_feature_description():
    """
    Clear description of what this test validates.

    Test approach:
    1. Setup initial state
    2. Execute operation
    3. Verify expected outcome
    """
    # Arrange
    setup_data = create_test_data()

    # Act
    result = function_under_test(setup_data)

    # Assert
    assert result.success, "Operation should succeed"
    assert result.value == expected_value, f"Expected {expected_value}, got {result.value}"
```