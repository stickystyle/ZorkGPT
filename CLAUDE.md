# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Do not make any changes until you have 95% confidence in the change you need to make. Ask me questions until you reach that confidence

## Project Overview

ZorkGPT is an AI agent system that plays the classic text adventure game "Zork" using Large Language Models. The system uses a modular architecture with specialized LLM-driven components for action generation, information extraction, action evaluation, and adaptive learning.

**Key Principle**: All game reasoning must originate from LLMs - no hardcoded solutions or predetermined game mechanics are allowed.

## Refactored Architecture (2025)

The system has been completely refactored from a monolithic design into a clean, modular architecture following the Single Responsibility Principle and orchestration patterns.

### Client-Server Architecture
- **Game Server** (`game_server.py`): Runs dfrotz in a Docker container and exposes a REST API
- **Game Client** (`game_server_client.py`): Provides standardized interface between orchestrator and game server
- **Orchestrator** (`orchestration/zork_orchestrator_v2.py`): Streamlined coordination layer

This architecture enables:
- Hot-reloading of AI code without losing game state
- Automatic save/restore functionality
- Clean separation of concerns
- Multiple orchestrators connecting to the same game

### Manager-Based Architecture

The refactored system follows a **manager pattern** where specialized managers handle distinct responsibilities:

#### Core Session Management
- **GameState** (`session/game_state.py`): Centralized shared state using dataclass pattern
- **GameConfiguration** (`session/game_configuration.py`): Configuration management with proper precedence

#### Specialized Managers
- **ObjectiveManager** (`managers/objective_manager.py`): Objective discovery, tracking, completion, and refinement
- **KnowledgeManager** (`managers/knowledge_manager.py`): Knowledge updates, synthesis, and learning integration
- **MapManager** (`managers/map_manager.py`): Map building, navigation, and spatial intelligence
- **StateManager** (`managers/state_manager.py`): State export, context management, and memory tracking
- **ContextManager** (`managers/context_manager.py`): Context assembly and prompt preparation
- **EpisodeSynthesizer** (`managers/episode_synthesizer.py`): Episode lifecycle and synthesis coordination

#### LLM-Powered Components
- **Agent** (`zork_agent.py`): Generates game actions based on current state and context
- **Extractor** (`hybrid_zork_extractor.py`): Parses raw game text into structured information  
- **Critic** (`zork_critic.py`): Evaluates proposed actions before execution with confidence scoring
- **Strategy Generator** (`zork_strategy_generator.py`): Manages adaptive knowledge and continuous learning

#### Supporting Systems
- **Map Graph** (`map_graph.py`): Builds and maintains spatial understanding with confidence tracking
- **Movement Analyzer** (`movement_analyzer.py`): Analyzes movement patterns and spatial relationships
- **Logger** (`logger.py`): Comprehensive logging for analysis and debugging
- **LLM Client** (`llm_client.py`): Custom LLM client with advanced sampling parameters and standardized response handling

### Manager Lifecycle and Dependencies

Managers follow a standardized lifecycle:
1. **Initialization**: Dependency injection with logger, config, and game state
2. **Reset**: Episode-specific state cleanup for new episodes
3. **Processing**: Turn-based and periodic update processing
4. **Status**: Comprehensive status reporting for monitoring

Dependency flow:
- MapManager → no dependencies
- ContextManager → no dependencies  
- StateManager → needs LLM client
- KnowledgeManager → needs agent and map references
- ObjectiveManager → needs knowledge manager reference
- EpisodeSynthesizer → needs knowledge and state managers

## Development Commands

### Starting the System

1. **Start the game server** (required first):
```bash
docker-compose up -d
```

2. **Run a gameplay episode**:
```bash
python main.py
```

3. **Start the web viewer** for live monitoring:
```bash
python start_viewer.py
```

### Testing
```bash
# Run all tests (requires game server running)
pytest

# Run unit tests for managers
pytest tests/test_managers.py -v

# Run integration tests for full system
pytest tests/test_integration.py -v

# Run specific test file
pytest tests/test_zork_api.py

# Run tests with coverage
pytest --cov
```

### Dependency Management
This project uses `uv` for dependency management. Install dependencies with:
```bash
uv sync
```

## Configuration

The system is configured via `pyproject.toml` under `[tool.zorkgpt.*]` sections:

- **LLM models**: Different models for agent, extractor, critic, and analysis tasks
- **Gameplay settings**: Turn delays, update intervals, thresholds
- **File paths**: Log files, knowledge base, state exports
- **AWS/S3**: Optional cloud storage integration

Key files generated during gameplay:
- `knowledgebase.md`: Dynamic knowledge base updated during play
- `persistent_wisdom.md`: Cross-episode learning insights
- `zork_episode_log.txt`: Detailed gameplay logs
- `current_state.json`: Exportable game state

## Key Data Flows

1. **Game Loop**: Observation → Extraction → Context Assembly → Action Generation → Evaluation → Execution
2. **Manager Coordination**: Orchestrator delegates to specialized managers for each domain
3. **Adaptive Learning**: Periodic analysis of recent gameplay (every 100 turns) to update knowledge base
4. **Spatial Intelligence**: Movement tracking and map building with confidence scoring
5. **Memory Management**: Context overflow protection with LLM-powered summarization
6. **Episode Synthesis**: Cross-episode learning and wisdom synthesis

## File Organization

### Core Architecture
- `orchestration/` - Streamlined orchestrator v2
- `managers/` - Specialized manager classes
- `session/` - Shared state and configuration management

### Supporting Components  
- `tests/` - Comprehensive unit and integration tests
- `infrastructure/` - Deployment and monitoring code
- `game_files/` - Game saves and state files
- Component-specific prompts: `agent.md`, `critic.md`, `extractor.md`

## Critical Design Constraints

### Episode ID Format
**MANDATORY**: All episode IDs MUST use ISO8601 format: `YYYY-MM-DDTHH:MM:SS`

```python
# CORRECT - ISO8601 format (orchestrator only)
episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
# Example: "2025-06-08T14:30:25"

# INCORRECT - Do not use any other format
episode_id = datetime.now().strftime("episode_%Y%m%d_%H%M%S")  # ❌ Wrong!
```

**Rationale**: 
- Ensures consistency across all components (GameState, EpisodeSynthesizer, GameServerClient, etc.)
- ISO8601 is internationally standardized and sortable
- Compatible with existing logging and monitoring systems
- Prevents integration issues between components

**Single Source of Truth for Episode IDs**:
**MANDATORY**: Only the orchestrator generates episode IDs. All other components receive and use them.

```python
# ✅ CORRECT FLOW - Orchestrator owns episode lifecycle
ZorkOrchestratorV2.play_episode()
├── 1. Generates episode ID: datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
├── 2. Calls: episode_synthesizer.initialize_episode(episode_id=episode_id)
│   └── 3. Calls: game_state.reset_episode(episode_id=episode_id) 
├── 4. Calls: game_interface.start_session(session_id=episode_id)
└── 5. All components use SAME episode ID ✅

# ❌ WRONG - No other component should generate episode IDs
def some_manager_method(self):
    episode_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # ❌ Forbidden!
```

### LLM Response Handling Architecture

**MANDATORY**: The LLMClient handles all response format standardization internally. Components should only use `.content` property.

```python
# ✅ CORRECT - Clean interface, implementation details hidden
response = llm_client.chat.completions.create(...)
content = response.content  # Always works regardless of backend format

# ❌ WRONG - Do not handle response format extraction in components  
response = llm_client.chat.completions.create(...)
if hasattr(response, 'choices'):
    content = response.choices[0].message.content  # ❌ Implementation detail leaked!
elif hasattr(response, 'content'):
    content = response.content  # ❌ Component handling format differences!
```

**Component Responsibilities**:
- `ZorkOrchestratorV2.play_episode()` - **ONLY** component that generates episode IDs (owns episode lifecycle)
- `EpisodeSynthesizer.initialize_episode(episode_id)` - Receives episode ID from orchestrator, coordinates initialization  
- `GameState.reset_episode(episode_id)` - Accepts episode ID as parameter, resets state
- `GameServerClient.start_session(session_id)` - Uses provided episode ID as session ID
- `All Managers.reset_episode()` - Reset manager-specific state only (no episode ID involvement)

### LLM Response Handling
**MANDATORY**: All LLM response handling MUST use the standardized utility:

```python
from utils.llm_utils import extract_llm_content

# CORRECT - Standardized handling
response_content = extract_llm_content(llm_response)

# INCORRECT - Direct access
content = llm_response.choices[0].message.content  # ❌ Breaks with different formats!
```

This utility handles multiple response formats (OpenAI, direct content, dict, etc.) with proper error handling.

### Manager Pattern
**MANDATORY**: All domain logic must be implemented in specialized managers, not in the orchestrator.

- Orchestrator is for **coordination only**
- Managers handle **domain-specific logic**
- No business logic in the orchestrator
- Use dependency injection for manager relationships

## Important Notes

- This is a research project exploring LLM capabilities in interactive environments
- The system is designed for extended sessions (up to 5000 turns)
- All reasoning must emerge from LLMs - avoid adding hardcoded game logic
- The system emphasizes continuous learning and adaptation during gameplay
- All unit tests and integration tests must pass before committing changes