# Testing Guide

This guide describes the testing strategy and infrastructure for ZorkGPT, with a focus on deterministic walkthrough-based testing enabled by the Jericho integration.

## Table of Contents

1. [Overview](#overview)
2. [Walkthrough-Based Testing](#walkthrough-based-testing)
3. [Test Organization](#test-organization)
4. [Writing New Tests](#writing-new-tests)
5. [Performance Benchmarking](#performance-benchmarking)
6. [Testing Best Practices](#testing-best-practices)

---

## Overview

ZorkGPT uses a multi-layered testing strategy:

**Unit Tests**: Test individual components in isolation (e.g., `map_graph.py`, `jericho_interface.py`)

**Integration Tests**: Test component interactions (e.g., orchestrator + managers)

**Walkthrough Tests**: Test full system behavior using deterministic game sequences

**Performance Benchmarks**: Measure and validate performance improvements

### Testing Philosophy

1. **Determinism**: Tests must be reproducible and not rely on randomness
2. **Fast Feedback**: Unit tests run in milliseconds, integration tests in seconds
3. **Comprehensive**: Cover happy paths, edge cases, and error conditions
4. **Maintainable**: Tests should be clear, focused, and easy to update
5. **Validate Assumptions**: Especially for Z-machine behavior (empirical validation)

---

## Walkthrough-Based Testing

The Jericho library provides built-in walkthroughs for supported games. These are deterministic sequences of actions that complete the game perfectly. We use these for:

- **Regression Testing**: Ensure changes don't break core functionality
- **Performance Validation**: Benchmark turn processing speed
- **Integration Validation**: Test full system over extended gameplay
- **Attribute Validation**: Empirically verify Z-machine behavior

### Walkthrough Fixtures

**Location**: `tests/fixtures/walkthrough.py`

This module provides utilities for working with Jericho walkthroughs:

```python
from tests.fixtures.walkthrough import (
    get_zork1_walkthrough,
    get_walkthrough_slice,
    get_walkthrough_until_lamp,
    get_walkthrough_dark_sequence,
    replay_walkthrough
)
```

**Available Functions:**

#### `get_zork1_walkthrough() -> List[str]`

Returns the complete Zork I walkthrough from Jericho (396 actions).

```python
walkthrough = get_zork1_walkthrough()
print(f"Total actions: {len(walkthrough)}")  # 396
print(f"First action: {walkthrough[0]}")     # "open mailbox"
```

#### `get_walkthrough_slice(start: int = 0, end: int = None) -> List[str]`

Returns a subset of the walkthrough for targeted testing.

```python
# First 20 actions
opening = get_walkthrough_slice(0, 20)

# Actions 50-100
mid_game = get_walkthrough_slice(50, 100)

# Last 10 actions
ending = get_walkthrough_slice(-10)
```

#### `get_walkthrough_until_lamp() -> List[str]`

Returns actions until the brass lamp is acquired (~15 actions).

```python
lamp_sequence = get_walkthrough_until_lamp()
# Useful for quick tests that need inventory items
```

#### `get_walkthrough_dark_sequence() -> List[str]`

Returns actions that navigate dark areas (tests dark room handling).

```python
dark_sequence = get_walkthrough_dark_sequence()
# Useful for testing movement detection in darkness
```

#### `replay_walkthrough(env: jericho.FrotzEnv, actions: List[str]) -> List[Tuple]`

Executes a sequence of actions and collects results.

```python
from game_interface.core.jericho_interface import JerichoInterface

interface = JerichoInterface(rom_path="infrastructure/zork.z5")
walkthrough = get_walkthrough_slice(0, 20)

# Replay and collect (observation, score, done, info) tuples
results = replay_walkthrough(interface.env, walkthrough)

for obs, score, done, info in results:
    print(f"Score: {score}, Done: {done}")
```

### Example Walkthrough Tests

#### Test 1: Location ID Stability

Verify that location IDs are deterministic across multiple replays.

```python
def test_location_id_stability():
    """Location IDs must be identical across replays."""
    from tests.fixtures.walkthrough import get_walkthrough_slice
    from game_interface.core.jericho_interface import JerichoInterface

    walkthrough = get_walkthrough_slice(0, 20)

    # First run
    interface1 = JerichoInterface(rom_path="infrastructure/zork.z5")
    ids1 = []
    for action in walkthrough:
        interface1.send_command(action)
        loc = interface1.get_location_structured()
        ids1.append(loc.num)

    # Second run
    interface2 = JerichoInterface(rom_path="infrastructure/zork.z5")
    ids2 = []
    for action in walkthrough:
        interface2.send_command(action)
        loc = interface2.get_location_structured()
        ids2.append(loc.num)

    # Must match exactly
    assert ids1 == ids2, "Location IDs must be deterministic"
```

#### Test 2: Zero Fragmentation

Verify that each location ID maps to exactly one room name.

```python
def test_no_room_fragmentation():
    """Each location ID must map to exactly one room name."""
    from tests.fixtures.walkthrough import get_walkthrough_slice
    from game_interface.core.jericho_interface import JerichoInterface

    walkthrough = get_walkthrough_slice(0, 50)
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")

    visited = {}  # id -> name mapping

    for action in walkthrough:
        interface.send_command(action)
        loc = interface.get_location_structured()

        if loc.num in visited:
            # Revisiting a room - name must match exactly
            assert visited[loc.num] == loc.name, \
                f"Fragmentation: ID {loc.num} has multiple names"
        else:
            visited[loc.num] = loc.name

    # Each ID should map to exactly one name
    assert len(visited) >= 10, "Should visit at least 10 unique rooms"
```

#### Test 3: Perfect Movement Detection

Verify movement detection works in all scenarios (including dark rooms).

```python
def test_movement_detection_accuracy():
    """Movement detection must be 100% accurate."""
    from tests.fixtures.walkthrough import get_walkthrough_slice
    from game_interface.core.jericho_interface import JerichoInterface

    walkthrough = get_walkthrough_slice(0, 100)
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")

    movement_actions = {'north', 'south', 'east', 'west', 'up', 'down',
                        'ne', 'nw', 'se', 'sw', 'in', 'out', 'enter', 'climb'}

    for action in walkthrough:
        before_loc = interface.get_location_structured()
        before_id = before_loc.num

        interface.send_command(action)

        after_loc = interface.get_location_structured()
        after_id = after_loc.num

        # Movement occurred if ID changed
        moved = (before_id != after_id)

        # Should only move with movement actions
        is_movement_action = any(cmd in action.lower() for cmd in movement_actions)

        if moved and not is_movement_action:
            # Unexpected movement (teleportation, etc.)
            print(f"Non-movement action caused movement: {action}")
```

#### Test 4: Inventory Tracking

Verify inventory tracking through item acquisition sequence.

```python
def test_inventory_tracking():
    """Inventory must track item acquisitions correctly."""
    from tests.fixtures.walkthrough import get_walkthrough_until_lamp
    from game_interface.core.jericho_interface import JerichoInterface

    walkthrough = get_walkthrough_until_lamp()
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")

    inventory_sizes = []

    for action in walkthrough:
        interface.send_command(action)
        inventory = interface.get_inventory_structured()
        inventory_sizes.append(len(inventory))

    # Inventory should grow as items are acquired
    assert max(inventory_sizes) > min(inventory_sizes), \
        "Inventory should change during walkthrough"

    # Final inventory should contain lamp
    final_inventory = interface.get_inventory_structured()
    lamp_acquired = any('lamp' in item.name.lower() for item in final_inventory)
    assert lamp_acquired, "Should have lamp after acquisition sequence"
```

---

## Test Organization

### Directory Structure

```
tests/
├── fixtures/
│   ├── __init__.py
│   └── walkthrough.py              # Walkthrough utilities
│
├── test_jericho_interface.py       # JerichoInterface unit tests
├── test_jericho_interface_session_methods.py  # Session management tests
├── test_map_graph.py                # MapGraph unit tests
├── test_movement_analyzer.py        # Movement detection tests
│
├── test_phase5_object_attributes.py    # Object attribute tests (29 tests)
├── test_phase5_enhanced_context.py     # Context manager tests (12 tests)
├── test_phase5_critic_validation.py    # Critic validation tests (14 tests)
├── test_phase5_integration.py          # Phase 5 integration (19 tests)
│
├── test_phase6_state_loop_detection.py # State loop tests (20 tests)
├── test_phase6_object_tracking.py      # Object event tests (18 tests)
│
└── test_phase7_walkthrough_integration.py  # Walkthrough tests (7 tests)
```

### Test Naming Conventions

**Unit Tests**: `test_<component>.py`
- Example: `test_map_graph.py`

**Phase Tests**: `test_phase<N>_<feature>.py`
- Example: `test_phase5_object_attributes.py`

**Integration Tests**: `test_<scope>_integration.py`
- Example: `test_phase5_integration.py`

**Test Functions**: `test_<what_is_tested>()`
- Good: `test_location_id_stability()`
- Bad: `test_1()`, `test_stuff()`

---

## Writing New Tests

### Unit Test Template

```python
"""Tests for <component>."""
import pytest
from <module> import <ComponentUnderTest>


class Test<Component>:
    """Test suite for <Component>."""

    def setup_method(self):
        """Set up test fixtures."""
        self.component = <ComponentUnderTest>()

    def test_basic_functionality(self):
        """Test that basic operation works."""
        # Arrange
        input_data = "test"

        # Act
        result = self.component.process(input_data)

        # Assert
        assert result == expected_value

    def test_edge_case_empty_input(self):
        """Test behavior with empty input."""
        result = self.component.process("")
        assert result is not None

    def test_error_handling(self):
        """Test that errors are handled gracefully."""
        with pytest.raises(ValueError):
            self.component.process(None)
```

### Integration Test Template

```python
"""Integration tests for <feature>."""
import pytest
from game_interface.core.jericho_interface import JerichoInterface
from tests.fixtures.walkthrough import get_walkthrough_slice


class Test<Feature>Integration:
    """Integration tests for <feature> with Jericho."""

    @pytest.fixture
    def interface(self):
        """Provide fresh Jericho interface for each test."""
        return JerichoInterface(rom_path="infrastructure/zork.z5")

    def test_feature_with_walkthrough(self, interface):
        """Test <feature> through deterministic walkthrough."""
        walkthrough = get_walkthrough_slice(0, 20)

        for action in walkthrough:
            # Execute action
            response = interface.send_command(action)

            # Verify expected behavior
            assert response is not None
            # ... more assertions
```

### Walkthrough Test Template

```python
"""Walkthrough-based tests for <system>."""
from tests.fixtures.walkthrough import (
    get_walkthrough_slice,
    get_walkthrough_until_lamp,
    replay_walkthrough
)
from game_interface.core.jericho_interface import JerichoInterface


def test_system_behavior_over_walkthrough():
    """Test <system> behavior through extended walkthrough."""
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    walkthrough = get_walkthrough_slice(0, 50)

    metrics = []

    for action in walkthrough:
        # Execute action
        before_state = interface.get_location_structured()
        response = interface.send_command(action)
        after_state = interface.get_location_structured()

        # Collect metrics
        metric = {
            "action": action,
            "before_id": before_state.num,
            "after_id": after_state.num,
            "response_length": len(response)
        }
        metrics.append(metric)

    # Verify metrics
    assert len(metrics) == 50
    assert all(m["response_length"] > 0 for m in metrics)
```

### Empirical Validation Template

Use this pattern to validate assumptions about Z-machine behavior:

```python
"""Empirical validation of Z-machine behavior."""
from game_interface.core.jericho_interface import JerichoInterface


def test_empirical_attribute_validation():
    """Empirically validate that object attributes match expected values."""
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")

    # Start game
    interface.send_command("open mailbox")
    interface.send_command("take leaflet")

    # Get leaflet object
    inventory = interface.get_inventory_structured()
    leaflet = next(obj for obj in inventory if "leaflet" in obj.name.lower())

    # Check attributes
    attrs = interface.get_object_attributes(leaflet)

    # Empirical expectations (validated via testing)
    assert attrs["takeable"] is True, "Leaflet should be takeable"
    assert attrs["readable"] is True, "Leaflet should be readable"
    assert attrs["portable"] is True, "Leaflet should be portable"
```

---

## Performance Benchmarking

### Running Benchmarks

**Full Comparison Report:**

```bash
uv run python benchmarks/comparison_report.py
```

**Individual Benchmarks:**

```bash
uv run python benchmarks/performance_metrics.py
```

### Writing New Benchmarks

**Location**: `benchmarks/`

**Template:**

```python
"""Benchmark for <feature>."""
import time
from game_interface.core.jericho_interface import JerichoInterface
from tests.fixtures.walkthrough import get_walkthrough_slice


def benchmark_feature_performance():
    """
    Measure performance of <feature>.

    Returns:
        dict: Benchmark results with metrics
    """
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    walkthrough = get_walkthrough_slice(0, 100)

    start_time = time.time()
    iterations = 0

    for action in walkthrough:
        # Measure feature
        interface.send_command(action)
        # ... perform measurement
        iterations += 1

    elapsed = time.time() - start_time
    throughput = iterations / elapsed

    return {
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "throughput": throughput,
        "avg_time_per_iteration": elapsed / iterations
    }


if __name__ == "__main__":
    results = benchmark_feature_performance()
    print(f"Throughput: {results['throughput']:.2f} iterations/sec")
    print(f"Avg time: {results['avg_time_per_iteration']*1000:.2f}ms")
```

### Benchmark Best Practices

1. **Use Deterministic Input**: Use walkthrough fixtures for reproducibility
2. **Warm-Up Phase**: Run a few iterations before timing to warm caches
3. **Multiple Runs**: Average results over multiple runs for stability
4. **Measure What Matters**: Focus on user-facing metrics (turn processing time)
5. **Document Baselines**: Record baseline performance for comparison

**Example:**

```python
def benchmark_with_warmup():
    """Benchmark with proper warm-up."""
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    walkthrough = get_walkthrough_slice(0, 100)

    # Warm-up phase (not timed)
    for action in get_walkthrough_slice(0, 10):
        interface.send_command(action)

    # Actual benchmark (timed)
    start = time.time()
    for action in walkthrough:
        interface.send_command(action)
    elapsed = time.time() - start

    return elapsed
```

---

## Testing Best Practices

### 1. Use Fixtures Appropriately

**Good:**

```python
@pytest.fixture
def jericho_interface():
    """Provide fresh interface for each test."""
    return JerichoInterface(rom_path="infrastructure/zork.z5")

def test_with_fixture(jericho_interface):
    loc = jericho_interface.get_location_structured()
    assert loc.num > 0
```

**Bad:**

```python
# Creating interface in every test (not reusable)
def test_without_fixture():
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    loc = interface.get_location_structured()
    assert loc.num > 0
```

### 2. Test One Thing Per Test

**Good:**

```python
def test_location_id_is_integer():
    """Test that location ID is an integer."""
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    loc = interface.get_location_structured()
    assert isinstance(loc.num, int)

def test_location_name_is_string():
    """Test that location name is a string."""
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    loc = interface.get_location_structured()
    assert isinstance(loc.name, str)
```

**Bad:**

```python
def test_location_stuff():
    """Test various location properties."""  # Too vague!
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    loc = interface.get_location_structured()
    assert isinstance(loc.num, int)
    assert isinstance(loc.name, str)
    assert loc.num > 0
    assert len(loc.name) > 0
    # Too many assertions - hard to debug failures
```

### 3. Use Descriptive Assertions

**Good:**

```python
assert location_id > 0, f"Location ID must be positive, got {location_id}"
assert room_name != "", "Room name cannot be empty"
```

**Bad:**

```python
assert location_id > 0  # No message - unclear what failed
assert room_name       # No message - unclear expectation
```

### 4. Test Error Cases

**Good:**

```python
def test_invalid_command_handling():
    """Test that invalid commands are handled gracefully."""
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")

    # Invalid command should not crash
    response = interface.send_command("xyzzy123nonsense")

    # Should receive error message from game
    assert response is not None
    assert len(response) > 0
```

### 5. Use Parametrized Tests for Variations

**Good:**

```python
@pytest.mark.parametrize("action,expected_direction", [
    ("north", "north"),
    ("n", "north"),
    ("go north", "north"),
    ("walk north", "north"),
])
def test_direction_normalization(action, expected_direction):
    """Test that various direction formats normalize correctly."""
    result = normalize_direction(action)
    assert result == expected_direction
```

**Bad:**

```python
def test_north():
    assert normalize_direction("north") == "north"

def test_n():
    assert normalize_direction("n") == "north"

def test_go_north():
    assert normalize_direction("go north") == "north"
# Repetitive - use parametrize instead
```

### 6. Clean Up Resources

**Good:**

```python
@pytest.fixture
def interface():
    """Provide interface with automatic cleanup."""
    iface = JerichoInterface(rom_path="infrastructure/zork.z5")
    yield iface
    # Cleanup happens automatically when test ends
    iface.env.close()
```

### 7. Skip Tests Appropriately

**Good:**

```python
@pytest.mark.skip(reason="Waiting for Jericho 3.0 release")
def test_future_feature():
    """Test feature that requires Jericho 3.0."""
    pass

@pytest.mark.skipif(not has_game_file(), reason="Game file not found")
def test_requires_game_file():
    """Test that requires game file."""
    pass
```

---

## Running Tests

### Run All Tests

```bash
uv run pytest tests/ -v
```

### Run Specific Test File

```bash
uv run pytest tests/test_jericho_interface.py -v
```

### Run Specific Test

```bash
uv run pytest tests/test_jericho_interface.py::test_location_structured -v
```

### Run Tests by Marker

```bash
# Run only integration tests
uv run pytest tests/ -v -m integration

# Run only unit tests
uv run pytest tests/ -v -m unit

# Skip slow tests
uv run pytest tests/ -v -m "not slow"
```

### Run Tests with Coverage

```bash
uv run pytest tests/ --cov=. --cov-report=html
```

### Run Tests in Parallel

```bash
uv run pytest tests/ -n auto
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync

      - name: Run tests
        run: uv run pytest tests/ -v

      - name: Run benchmarks
        run: uv run python benchmarks/comparison_report.py
```

---

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "Game file not found"
**Solution**: Ensure `infrastructure/zork.z5` exists

**Issue**: Tests are slow
**Solution**: Use `get_walkthrough_slice()` for faster tests with fewer actions

**Issue**: Tests are flaky
**Solution**: Avoid randomness - use walkthrough fixtures for determinism

**Issue**: Fixture import errors
**Solution**: Ensure `tests/fixtures/__init__.py` exists

### Debug Tips

**Print Game State:**

```python
def test_with_debug():
    interface = JerichoInterface(rom_path="infrastructure/zork.z5")
    loc = interface.get_location_structured()

    # Debug output
    print(f"\nLocation ID: {loc.num}")
    print(f"Location Name: {loc.name}")
    print(f"Inventory: {interface.get_inventory_structured()}")

    assert loc.num > 0
```

**Use pytest's `-s` flag to see print output:**

```bash
uv run pytest tests/test_my_test.py -v -s
```

---

## Additional Resources

- **Pytest Documentation**: [docs.pytest.org](https://docs.pytest.org/)
- **Jericho Documentation**: [github.com/microsoft/jericho](https://github.com/microsoft/jericho)
- **Walkthrough Fixtures**: `tests/fixtures/walkthrough.py`
- **Benchmark Examples**: `benchmarks/performance_metrics.py`
- **Phase Test Examples**: `tests/test_phase5_*.py`, `tests/test_phase6_*.py`

---

## Summary

**Key Takeaways:**

1. **Use walkthrough fixtures** for deterministic, reproducible tests
2. **Test one thing per test** for clarity and debuggability
3. **Validate empirically** - don't assume Z-machine behavior
4. **Benchmark performance** - measure before/after changes
5. **Write descriptive test names** - make failures obvious
6. **Clean up resources** - use fixtures with teardown
7. **Run tests often** - fast feedback prevents bugs

**Test Coverage Goals:**

- Unit tests: 80%+ coverage of core components
- Integration tests: Cover all major workflows
- Walkthrough tests: Validate full system over extended gameplay
- Benchmarks: Validate all performance claims

Happy testing! 🧪
