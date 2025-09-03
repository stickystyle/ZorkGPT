# Game Interface Layer

The Game Interface Layer provides a clean separation between the ZorkGPT AI orchestration logic and the underlying game mechanics. This modular architecture enables hot-reloading of AI code without losing game state and supports multiple orchestrators connecting to the same game.

## Architecture

```
game_interface/
├── __init__.py              # Main exports for easy importing
├── server/                  # Game server components
│   ├── __init__.py
│   ├── game_server.py       # FastAPI server managing dfrotz processes
│   ├── models.py            # Pydantic models for API requests/responses
│   └── session_manager.py   # GameSession class for session lifecycle
├── client/                  # REST API client
│   ├── __init__.py
│   └── game_server_client.py # HTTP client for connecting to game server
├── core/                    # Core game interface
│   ├── __init__.py
│   ├── zork_interface.py    # Direct dfrotz process interface
│   └── structured_parser.py # Parser for Zork's structured output
└── README.md               # This file
```

## Components

### Server Package (`game_interface.server`)

The server package contains FastAPI-based components for running the game server:

- **`game_server.py`**: Main FastAPI application with HTTP endpoints for session management
- **`session_manager.py`**: `GameSession` class managing individual game instances with save/restore
- **`models.py`**: Pydantic models for type-safe API requests and responses

#### Key Features
- Automatic save/restore functionality
- Session persistence across server restarts
- RESTful API for game commands and state queries
- Concurrent session support

### Client Package (`game_interface.client`)

Contains the REST API client for connecting to the game server:

- **`game_server_client.py`**: `GameServerClient` class providing ZorkInterface-compatible API

#### Key Features  
- Compatible with existing ZorkInterface API
- HTTP-based communication with game server
- Automatic session management and restoration
- Command history and state tracking

### Core Package (`game_interface.core`)

Core game interface classes and parsers:

- **`zork_interface.py`**: `ZorkInterface` class for direct dfrotz process management
- **`structured_parser.py`**: `StructuredZorkParser` for parsing Zork's structured output format

#### Key Features
- Direct subprocess management of dfrotz
- Robust save/restore operations with race condition handling
- Inventory parsing with container support
- Game over detection
- Score and move tracking

## Usage Examples

### Using the Complete Game Interface

```python
from game_interface import ZorkInterface, GameServerClient, StructuredZorkParser

# Direct interface (local dfrotz process)
with ZorkInterface() as zork:
    intro = zork.start()
    response = zork.send_command("look")

# Client-server interface (remote game server)
with GameServerClient("http://localhost:8000") as client:
    intro = client.start("my_session_id")
    response = client.send_command("look")

# Structured parsing
parser = StructuredZorkParser()
parsed = parser.parse_response(response)
print(f"Room: {parsed.room_name}, Score: {parsed.score}")
```

### Running the Game Server

```bash
# Start the server directly
python -m game_interface.server.game_server

# Or using Docker
docker build -f Dockerfile.game_server -t zorkgpt-server .
docker run -p 8000:8000 zorkgpt-server
```

### Server API Endpoints

- `POST /sessions/{session_id}` - Create or restore a session
- `POST /sessions/{session_id}/command` - Send a command  
- `GET /sessions/{session_id}/history` - Get command history
- `GET /sessions/{session_id}/state` - Get session state
- `POST /sessions/{session_id}/save` - Force immediate save
- `DELETE /sessions/{session_id}` - Close session
- `GET /health` - Health check

## Integration with ZorkGPT

The game interface layer integrates seamlessly with the ZorkGPT orchestration system:

```python
from game_interface.client import GameServerClient
from orchestration.zork_orchestrator_v2 import ZorkOrchestratorV2

# Initialize orchestrator with game client
client = GameServerClient("http://localhost:8000")
orchestrator = ZorkOrchestratorV2(game_client=client)
orchestrator.run_episode()
```

## Data Models

### CommandRequest/Response
```python
class CommandRequest(BaseModel):
    command: str

class CommandResponse(BaseModel):
    session_id: str
    turn_number: int  
    score: Optional[int]
    raw_response: str
    parsed: dict
    game_over: bool
    game_over_reason: Optional[str]
```

### Session Management
```python
class SessionState(BaseModel):
    session_id: str
    turn_number: int
    last_score: int
    last_save_turn: int
    active: bool
    start_time: str
    last_command_time: str
```

## Benefits

1. **Clean Separation**: AI logic is completely separate from game mechanics
2. **Hot Reloading**: Restart orchestrators without losing game progress  
3. **Scalability**: Multiple AI instances can connect to the same game
4. **Testability**: Easy to mock and test individual components
5. **Maintainability**: Focused responsibilities and clear interfaces
6. **Robustness**: Automatic save/restore with comprehensive error handling

## Development

When modifying the game interface:

1. Update the appropriate package (server, client, or core)
2. Run tests to verify compatibility: `pytest tests/`
3. Update imports in dependent modules if APIs change
4. Update this README if new features are added

The game interface layer follows the principle that all game reasoning must originate from LLMs - no hardcoded solutions or predetermined game mechanics are embedded in the interface itself.