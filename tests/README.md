# Tests

This directory contains unit tests for the ZorkGPT project.

## Test Structure

- `test_combat_detection.py` - Tests for combat detection functionality in the extractor
- `test_combat_scenario.py` - Integration tests for full combat scenarios and protection features
- `test_game_over_detection.py` - Tests for game over detection in the Zork API
- `test_inventory_skip.py` - Tests for inventory skipping logic during combat
- `test_map_graph.py` - Tests for the map graph functionality and navigation
- `test_zork_api.py` - Tests for the Zork API interface, particularly inventory parsing

## Running Tests

### Using unittest (built-in)

Run all tests:
```bash
python -m unittest discover tests
```

Run a specific test file:
```bash
python -m unittest tests.test_combat_detection
```

Run a specific test class:
```bash
python -m unittest tests.test_combat_detection.TestCombatDetection
```

Run a specific test method:
```bash
python -m unittest tests.test_combat_detection.TestCombatDetection.test_combat_scenario_detection
```

### Using pytest (recommended)

First install test dependencies:
```bash
pip install -e ".[test]"
```

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=.
```

Run a specific test file:
```bash
pytest tests/test_combat_detection.py
```

Run tests matching a pattern:
```bash
pytest -k "combat"
```

## Test Categories

- **Unit Tests**: Test individual functions and methods in isolation
- **Integration Tests**: Test how different components work together
- **Functional Tests**: Test complete workflows and scenarios

Each test file focuses on a specific area of functionality and includes both positive and negative test cases to ensure robust error handling. 