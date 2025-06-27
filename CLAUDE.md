# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: Do not make any changes until you have 95% confidence in the change you need to make. Ask me questions until you reach that confidence

## Project Overview

ZorkGPT is an AI agent system that plays the classic text adventure game "Zork" using Large Language Models. The system uses a modular architecture with specialized LLM-driven components for action generation, information extraction, action evaluation, and adaptive learning.

**Key Principle**: All game reasoning must originate from LLMs - no hardcoded solutions or predetermined game mechanics are allowed.

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

## Memories and Principles

- The state should be exported at the end of every turn.

## Development Commands

[Rest of the file remains the same as in the original content]