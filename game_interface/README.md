# Game Interface Layer

The Game Interface Layer provides a clean separation between the ZorkGPT AI orchestration logic and the underlying game mechanics using the Jericho library for direct Z-machine access.

## Architecture

```
game_interface/
├── __init__.py              # Main exports for easy importing
└── core/                    # Core game interface
    ├── __init__.py
    └── jericho_interface.py # Jericho-based Z-machine interface
```

## Components

### Core Package (`game_interface.core`)

The core package contains the Jericho-based game interface:

- **`jericho_interface.py`**: `JerichoInterface` class for direct Z-machine access via Jericho

#### Key Features
- Direct Z-machine memory access (no text parsing needed)
- Structured access to game objects, inventory, location, and score
- Built-in save/restore functionality
- Game over detection
- Location ID tracking for perfect movement detection
- Object attribute inspection

## Usage Examples

### Using the Jericho Interface

```python
from game_interface.core.jericho_interface import JerichoInterface

# Initialize with game file path
jericho = JerichoInterface(
    game_file_path="/path/to/zork1.z5",
    logger=your_logger
)

# Start the game
intro = jericho.start()

# Send commands
response = jericho.send_command("look")

# Access structured Z-machine data
location = jericho.get_location_structured()
print(f"Location ID: {location.num}, Name: {location.name}")

inventory = jericho.get_inventory_structured()
for item in inventory:
    print(f"Item ID: {item.num}, Name: {item.name}")

score = jericho.get_score()
print(f"Score: {score}")

# Check game over
is_over, reason = jericho.is_game_over(response)

# Save/restore game state
jericho.trigger_zork_save()
jericho.trigger_zork_restore()

# Clean up
jericho.close()
```

## Integration with ZorkGPT

The game interface layer integrates seamlessly with the ZorkGPT orchestration system:

```python
from game_interface.core.jericho_interface import JerichoInterface
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2

# Initialize orchestrator (it creates JerichoInterface internally)
orchestrator = ZorkOrchestratorV2(
    episode_id="episode_001",
    max_turns_per_episode=100
)
final_score = orchestrator.play_episode()
```

## Z-Machine Object Model

Jericho provides structured access to Z-machine objects:

### Location Object
```python
location = jericho.get_location_structured()
# location.num - Unique location ID (integer)
# location.name - Location name (string)
```

### Inventory Objects
```python
inventory = jericho.get_inventory_structured()
# Returns list of ZObject instances
# Each object has: .num (ID), .name (name), .attr (attributes)
```

### Visible Objects
```python
visible = jericho.get_visible_objects()
# Returns list of objects visible in current location
# Filtered to exclude unwanted objects (YOU, rooms, etc.)
```

## Benefits

1. **Clean Separation**: AI logic is completely separate from game mechanics
2. **Direct Z-Machine Access**: No text parsing needed for inventory, location, score
3. **Stable Object IDs**: Integer-based location IDs eliminate room fragmentation
4. **Perfect Movement Detection**: ID comparison replaces complex heuristics
5. **Testability**: Easy to mock and test individual components
6. **Maintainability**: Focused responsibilities and clear interfaces
7. **Performance**: ~40% reduction in LLM calls by using Z-machine data directly

## Jericho vs dfrotz

The migration from dfrotz to Jericho provides:

- **No subprocess management**: Jericho runs in-process
- **No Docker required**: Direct Python library integration
- **Structured data access**: Objects, inventory, location via Z-machine memory
- **Stable location IDs**: Integer IDs prevent duplicate rooms in map
- **Object inspection**: Access to Z-machine object attributes
- **Built-in save/restore**: Native Z-machine save state management

## Development

When modifying the game interface:

1. Update `jericho_interface.py` for interface changes
2. Run tests to verify compatibility: `uv run pytest tests/`
3. Update imports in dependent modules if APIs change
4. Update this README if new features are added

The game interface layer follows the principle that all game reasoning must originate from LLMs - no hardcoded solutions or predetermined game mechanics are embedded in the interface itself.