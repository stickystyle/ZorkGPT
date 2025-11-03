# ZorkGPT: An LLM-Powered Agent for Interactive Fiction

**🎮 Watch ZorkGPT play live at [https://zorkgpt.com](https://zorkgpt.com)**

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Mission and Design Philosophy](#core-mission-and-design-philosophy)
3. [System Architecture](#system-architecture)
    * [Central Coordinator](#central-coordinator)
    * [Core LLM-Powered Modules](#core-llm-powered-modules)
    * [Supporting Systems](#supporting-systems)
4. [Key Features](#key-features)
    * [Extended Gameplay Sessions](#extended-gameplay-sessions)
    * [Adaptive Knowledge System](#adaptive-knowledge-system)
    * [Spatial Intelligence System](#spatial-intelligence-system)
5. [Data Management](#data-management)
6. [System Flow](#system-flow)
7. [Logging Infrastructure](#logging-infrastructure)
8. [Architecture Diagram](#architecture-diagram)
9. [Turn-by-Turn Flow with Adaptive Learning](#turn-by-turn-flow-with-adaptive-learning)

## Project Overview

ZorkGPT is an AI agent system designed to play the classic interactive fiction game "[Zork](https://en.wikipedia.org/wiki/Zork)" using Large Language Models (LLMs). The project explores how modern language models can understand, navigate, and solve complex interactive environments through natural language reasoning. It aims to achieve genuine AI-driven gameplay by relying on LLM capabilities for all aspects of decision-making and learning, rather than pre-programmed logic or solutions.

The system employs a modular architecture where specialized LLM-driven components handle different aspects of gameplay: action generation, information extraction, action evaluation, and strategic learning. This design allows each component to focus on its specific task while adhering to the core principle that all reasoning originates from the language models. Gameplay occurs over extended sessions, during which the system continuously accumulates knowledge and refines its strategies in real-time.

## Core Mission and Design Philosophy

ZorkGPT operates under several fundamental principles:

* **LLM-First Design**: All game reasoning, decision-making, and understanding must originate from language models. The system deliberately avoids hardcoded game mechanics, location databases, or predetermined solutions.
* **No Hardcoded Solutions**: The only acceptable hardcoded logic is for validating parser acceptance (i.e., did the Zork game engine understand the command). Puzzle solutions, navigation choices, and strategic decisions must emerge from LLM reasoning.
* **Adaptive Learning**: The system implements continuous knowledge extraction and strategy refinement *during* gameplay. The AI analyzes its experiences in real-time, updating its strategic understanding at regular intervals (e.g., every 100 turns). This allows for the immediate incorporation of new insights.
* **Genuine AI Play**: The objective is to have the LLMs genuinely "play" the game, demonstrating authentic language model capabilities in a complex, interactive environment. When LLMs encounter challenges, the approach is to improve prompts, models, or context rather than introducing fallback mechanisms.

## System Architecture

ZorkGPT utilizes a multi-component architecture, coordinated by a central orchestrator, with distinct LLM-powered modules for different cognitive functions, supported by specialized systems for game interaction and data management.

### Jericho Integration

The system leverages the **Jericho** library for direct Z-machine memory access, providing:

* **Direct Z-machine Access**: Instant retrieval of inventory, location, score, and object data without text parsing
* **Integer-based Location IDs**: Stable room identification eliminating fragmentation issues
* **Object Tree Visibility**: Access to game object attributes and valid action verbs
* **Built-in Save/Restore**: Leverages Z-machine state management for session handling
* **Perfect Movement Detection**: Location ID comparison replaces error-prone heuristics

### Central Coordinator

A **ZorkOrchestrator** class serves as the primary coordinator. It manages extended gameplay sessions (potentially thousands of turns), coordinates the interactions between all other system components, handles the overall game loop, and orchestrates the periodic adaptive knowledge updates.

### Core LLM-Powered Modules

These components form the cognitive core of the agent:

* **Agent LM**: Responsible for generating game actions. It analyzes the current game state, integrates relevant information from session memory, spatial knowledge, and strategic guidance from the knowledge base to decide the next action. Receives structured Z-machine object data for enhanced reasoning.
* **Extractor LM**: Parses raw game text output into structured information. With Jericho integration, extraction is now hybrid: inventory, location, score, and objects come directly from Z-machine memory, while LLM parsing focuses on exits, combat, and important messages. This reduces LLM calls by 40% per turn.
* **Critic LM**: Evaluates proposed actions before execution. It assesses actions based on relevance, potential for progress, risk, and alignment with current strategy, providing a confidence score and justification. This helps filter out suboptimal actions and guides the agent towards more effective gameplay. It also incorporates a trust mechanism that can adapt its strictness based on recent performance. Performs fast object tree validation before expensive LLM calls (83.3% LLM reduction for invalid actions).
* **Adaptive Knowledge Manager (Strategy LM)**: Drives the continuous learning process. This component analyzes gameplay data in real-time (e.g., within 100-turn windows) to identify patterns, successful tactics, and areas for improvement. It assesses the quality of new information and intelligently merges new insights into the existing knowledge base, ensuring that learning is productive and avoids knowledge degradation.

### Supporting Systems

Several systems support the core LLM modules:

* **Game Interface (JerichoInterface)**: Manages low-level interaction with the Z-machine game engine via the Jericho library, providing instant access to game state, inventory, location data, object attributes, and valid action vocabulary.
* **Spatial Intelligence System**: Builds and maintains a dynamic understanding of the game world's layout using integer-based location IDs from the Z-machine, eliminating room fragmentation and enabling perfect movement detection.
* **Logging System**: Captures detailed data from all parts of the system for analysis, debugging, and to provide the raw material for the adaptive learning process.

## Key Features

### Extended Gameplay Sessions

ZorkGPT is designed for multi-thousand turn gameplay sessions. Knowledge and memory persist and evolve throughout these long sessions, allowing for cumulative learning and adaptation over extended periods of interaction with the game world.

### Adaptive Knowledge System

The system features a sophisticated continuous learning pipeline:

* **Real-Time Analysis**: Gameplay data is analyzed during play. For instance, every 100 turns, recent experiences are processed to extract strategic insights.
* **LLM-Driven Assessment**: The determination of what constitutes valuable knowledge and how it should be integrated is itself an LLM-driven process, adhering to the LLM-first principle.
* **Intelligent Merging**: New insights are merged into the existing knowledge base in a way that aims to enhance, rather than overwrite or degrade, prior learning.
* **Dynamic Strategy Guides**: The output of this system is a dynamically updated knowledge base that provides strategic guidance to the Agent LM, influencing its decisions.
* **Dynamic Objective Formulation**: The learning system contributes to identifying and prioritizing high-level objectives. These emergent goals, derived from gameplay analysis and strategic understanding, help guide the agent's long-term decision-making and focus its exploration and actions towards achieving meaningful progress in the game.

### Spatial Intelligence System

Understanding and navigating the game's geography is crucial:

* **Graph-Based Mapping**: A dynamic graph represents the game world, with locations as nodes and connections as edges. Confidence scores are associated with connections, reflecting the agent's certainty about them.
* **Movement Analysis**: The system analyzes movement patterns to verify connections, identify efficient routes, and recognize important spatial features like hubs or dead ends.
* **Contextual Navigation**: Spatial information (current location, known exits, map-based relationships) is provided to the Agent LM to inform its navigation decisions.
* **Intelligent Exit Validation**: Failed movement attempts are tracked. Exits that consistently fail are eventually pruned from the agent's map to prevent repeated errors, based on empirical evidence from gameplay.

## Data Management

Effective gameplay relies on robust data management:

* **Session Memory**: Structured records of turn-by-turn observations, including game text, extracted information, and actions taken, are maintained throughout a session. This provides historical context for decision-making.
* **Dynamic Knowledge Base**: A continuously updated repository of strategic insights, learned heuristics, and important facts about the game world. This knowledge base is curated by the Adaptive Knowledge Manager and directly informs the Agent LM.

## System Flow

A typical gameplay loop involves several stages:

1. **Observation**: The system receives the latest game text from the Z-machine.
2. **Extraction (Hybrid)**:
   - **Instant Z-machine Access**: Inventory, location, score, and visible objects retrieved directly from game memory
   - **LLM Parsing**: Extractor LM processes text for exits, combat, and important messages only
3. **State Update**: Session memory and the spatial map are updated with the new information. Movement detection uses location ID comparison (perfect accuracy).
4. **Contextualization**: Relevant memories, spatial data, strategic knowledge, and structured Z-machine object data are assembled for the Agent LM.
5. **Action Generation**: The Agent LM proposes an action, informed by structured object attributes and valid action vocabulary.
6. **Evaluation**: The Critic performs fast object tree validation first (microseconds), then LLM evaluation if needed. If the action is deemed suboptimal, the Agent LM may be prompted for alternatives. Object tree validation reduces LLM calls by 83.3% for invalid actions.
7. **Execution**: The chosen action is sent to the Z-machine game engine.
8. **Learning**: Periodically (e.g., every 25 turns for map-focused updates, every 100 turns for broader strategic updates), the Adaptive Knowledge Manager analyzes recent gameplay to refine the knowledge base. State loop detection alerts when exact game states repeat.

This cycle repeats, allowing the agent to explore, interact, and learn from its experiences in the game world over thousands of turns.

## Logging Infrastructure

Comprehensive logging captures all significant events, LLM interactions, state changes, and decisions. This structured log data is crucial for:

* **Debugging and Analysis**: Understanding system behavior and diagnosing issues.
* **Adaptive Learning**: Providing the raw data (e.g., turn-window information) that the Adaptive Knowledge Manager uses to extract insights and generate new strategies.

The design of the logging system is specifically tailored to support the turn-based analysis required for continuous learning.

## Architecture Diagram

```mermaid
graph TB
    subgraph "ZorkGPT Manager-Based Architecture"
        subgraph "Central Coordination"
            Orchestrator[ZorkOrchestratorV2<br/>Episode Lifecycle Management<br/>Turn-by-Turn Coordination<br/>Manager Orchestration]
        end

        subgraph "Core Session Management"
            GameState[GameState<br/>Centralized Shared State<br/>Episode & Turn Tracking<br/>Action & Memory History]
            GameConfig[GameConfiguration<br/>TOML + Env Config<br/>Model Settings<br/>Update Intervals]
        end

        subgraph "Specialized Managers"
            ObjectiveMgr[ObjectiveManager<br/>Objective Discovery<br/>Completion Tracking<br/>Staleness Management]
            KnowledgeMgr[KnowledgeManager<br/>Periodic Updates<br/>Episode Synthesis<br/>Inter-Episode Wisdom]
            MapMgr[MapManager<br/>Integer-Based Mapping<br/>Connection Tracking<br/>Map Persistence]
            StateMgr[StateManager<br/>State Export & S3<br/>Context Overflow<br/>Loop Detection]
            ContextMgr[ContextManager<br/>Agent Context Assembly<br/>Critic Context Prep<br/>Memory Filtering]
            EpisodeSynth[EpisodeSynthesizer<br/>Episode Lifecycle<br/>Final Synthesis<br/>Episode Summaries]
            RejectionMgr[RejectionManager<br/>Trust Calibration<br/>Override Decisions<br/>Rejection Tracking]
            MemoryMgr[SimpleMemoryManager<br/>Location Memories<br/>Memory Synthesis<br/>Deduplication]
        end

        subgraph "LLM-Powered Components"
            Agent[Agent<br/>Action Generation<br/>Knowledge Integration<br/>Loop Detection]
            Extractor[HybridExtractor<br/>Z-machine + LLM Parsing<br/>Object Tree Access<br/>State Structuring]
            Critic[Critic<br/>Object Tree Validation<br/>LLM Evaluation<br/>Trust Tracking]
            StrategyGen[StrategyGenerator<br/>Turn-Window Analysis<br/>Knowledge Synthesis<br/>Cross-Episode Learning]
        end

        subgraph "Supporting Systems"
            JerichoIF[JerichoInterface<br/>Direct Z-machine Access<br/>Object Tree Queries<br/>Save/Restore]
            MapGraph[MapGraph<br/>Integer-Based Rooms<br/>Confidence Scoring<br/>Exit Pruning]
            MovementAnalyzer[MovementAnalyzer<br/>ID-Based Detection<br/>Perfect Accuracy]
            Logger[Logger<br/>Triple-Output System<br/>Structured Events<br/>Episode Logs]
            LLMClient[LLMClient<br/>Advanced Sampling<br/>Retry + Circuit Breaker<br/>Langfuse Integration]
        end

        subgraph "Persistent Data"
            KnowledgeBase[knowledgebase.md<br/>Strategic Insights<br/>Cross-Episode Wisdom<br/>World Map]
            MapState[map_state.json<br/>Room Graph<br/>Connection Confidence<br/>Pruned Exits]
            MemoryFile[Memories.md<br/>Location Memories<br/>Status Tracking<br/>Supersession]
            EpisodeLogs[Episode JSON Logs<br/>Turn-by-Turn Data<br/>Analysis Substrate]
        end

        subgraph "External"
            ZorkGame[Zork I Z-machine<br/>via Jericho Library<br/>Direct Memory Access]
        end
    end

    %% Orchestrator connections
    Orchestrator --> GameState
    Orchestrator --> GameConfig
    Orchestrator --> Agent
    Orchestrator --> Extractor
    Orchestrator --> Critic
    Orchestrator --> JerichoIF
    Orchestrator --> Logger

    %% Manager dependencies
    Orchestrator --> ObjectiveMgr
    Orchestrator --> KnowledgeMgr
    Orchestrator --> MapMgr
    Orchestrator --> StateMgr
    Orchestrator --> ContextMgr
    Orchestrator --> EpisodeSynth
    Orchestrator --> RejectionMgr
    Orchestrator --> MemoryMgr

    %% Manager interactions
    KnowledgeMgr --> Agent
    KnowledgeMgr --> MapMgr
    KnowledgeMgr --> StrategyGen
    ObjectiveMgr --> KnowledgeMgr
    EpisodeSynth --> KnowledgeMgr
    EpisodeSynth --> StateMgr
    ContextMgr --> MapMgr
    ContextMgr --> MemoryMgr

    %% LLM component dependencies
    Agent --> LLMClient
    Agent --> KnowledgeBase
    Extractor --> LLMClient
    Extractor --> JerichoIF
    Critic --> LLMClient
    Critic --> JerichoIF
    StrategyGen --> LLMClient
    StrategyGen --> EpisodeLogs

    %% Supporting system connections
    MapMgr --> MapGraph
    MapMgr --> MovementAnalyzer
    StateMgr --> LLMClient
    MemoryMgr --> LLMClient
    JerichoIF --> ZorkGame

    %% Data persistence
    MapMgr -.->|Load/Save| MapState
    KnowledgeMgr -.->|Read/Write| KnowledgeBase
    MemoryMgr -.->|Read/Write| MemoryFile
    Logger -.->|Write| EpisodeLogs

    %% GameState access (all managers read/write)
    GameState -.->|Shared State| ObjectiveMgr
    GameState -.->|Shared State| KnowledgeMgr
    GameState -.->|Shared State| MapMgr
    GameState -.->|Shared State| StateMgr
    GameState -.->|Shared State| ContextMgr
    GameState -.->|Shared State| RejectionMgr
    GameState -.->|Shared State| MemoryMgr

    classDef coordinator fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef session fill:#ffecb3,stroke:#f57f17,stroke-width:2px
    classDef manager fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef llm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef support fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef data fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef external fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class Orchestrator coordinator
    class GameState,GameConfig session
    class ObjectiveMgr,KnowledgeMgr,MapMgr,StateMgr,ContextMgr,EpisodeSynth,RejectionMgr,MemoryMgr manager
    class Agent,Extractor,Critic,StrategyGen llm
    class JerichoIF,MapGraph,MovementAnalyzer,Logger,LLMClient support
    class KnowledgeBase,MapState,MemoryFile,EpisodeLogs data
    class ZorkGame external
```

## Turn-by-Turn Flow with Manager-Based Architecture

```mermaid
sequenceDiagram
    participant UserMain as User/Main
    participant Orch as ZorkOrchestratorV2
    participant EpisodeSynth as EpisodeSynthesizer
    participant Jericho as JerichoInterface
    participant HybridExt as HybridExtractor
    participant ContextMgr as ContextManager
    participant MemoryMgr as SimpleMemoryManager
    participant Agent as Agent LM
    participant Critic as Critic LM
    participant RejectionMgr as RejectionManager
    participant MapMgr as MapManager
    participant StateMgr as StateManager
    participant ObjectiveMgr as ObjectiveManager
    participant KnowledgeMgr as KnowledgeManager

    UserMain->>Orch: Start Episode
    activate Orch

    Orch->>EpisodeSynth: initialize_episode()
    activate EpisodeSynth
    EpisodeSynth->>EpisodeSynth: Reset GameState, set episode_id
    EpisodeSynth-->>Orch: Episode initialized
    deactivate EpisodeSynth

    Orch->>Jericho: Start game
    activate Jericho
    Jericho-->>Orch: Initial game text
    deactivate Jericho

    Orch->>HybridExt: extract_info(game_text)
    activate HybridExt
    HybridExt->>Jericho: get_location_structured()
    Jericho-->>HybridExt: ZObject(num=ID, name)
    HybridExt->>Jericho: get_inventory_structured()
    Jericho-->>HybridExt: [ZObjects]
    HybridExt->>Jericho: get_visible_objects_in_location()
    Jericho-->>HybridExt: [ZObjects from object tree]
    HybridExt->>Jericho: get_score()
    Jericho-->>HybridExt: (score, moves)
    HybridExt->>HybridExt: LLM extract exits/combat/messages
    HybridExt-->>Orch: ExtractorResponse (structured data)
    deactivate HybridExt

    Orch->>MapMgr: add_initial_room(room_id, room_name)
    activate MapMgr
    MapMgr-->>Orch: Initial room added
    deactivate MapMgr

    loop Extended Gameplay (Episode)
        Orch->>RejectionMgr: start_new_turn()

        Orch->>ContextMgr: get_agent_context()
        activate ContextMgr
        ContextMgr->>MapMgr: get_context_for_prompt()
        MapMgr-->>ContextMgr: Map context
        ContextMgr->>MemoryMgr: get_location_memory()
        MemoryMgr-->>ContextMgr: Location memory
        ContextMgr->>Jericho: get_inventory_structured(), get_visible_objects_in_location()
        Jericho-->>ContextMgr: Structured objects + attributes
        ContextMgr-->>Orch: Formatted agent context
        deactivate ContextMgr

        Orch->>Agent: get_action_with_reasoning(context)
        activate Agent
        Agent-->>Orch: Proposed action + reasoning
        deactivate Agent

        loop Critic Evaluation (max 3 attempts)
            Orch->>ContextMgr: get_critic_context()
            activate ContextMgr
            ContextMgr->>Jericho: get_valid_exits()
            Jericho-->>ContextMgr: Ground-truth exits
            ContextMgr-->>Orch: Critic context
            deactivate ContextMgr

            Orch->>Critic: evaluate_action(action, context, jericho_interface)
            activate Critic
            Critic->>Critic: Fast object tree validation (<1ms)
            alt Object Tree Validation Failed
                Critic-->>Orch: score=0.0, high confidence rejection
            else Object Tree Validation Passed
                Critic->>Critic: LLM evaluation (~800ms)
                Critic-->>Orch: LLM score + justification
            end
            deactivate Critic

            alt Action Rejected (score < threshold)
                Orch->>RejectionMgr: should_override_rejection()
                activate RejectionMgr
                RejectionMgr-->>Orch: Override decision
                deactivate RejectionMgr

                alt No Override - Request New Action
                    Orch->>Agent: get_action_with_reasoning(rejection feedback)
                    Agent-->>Orch: Alternative action
                else Override - Accept Action
                    Note over Orch: Break evaluation loop
                end
            else Action Accepted
                Note over Orch: Break evaluation loop
            end
        end

        Orch->>Jericho: send_command(chosen_action)
        activate Jericho
        Jericho-->>Orch: New game text
        deactivate Jericho

        Orch->>HybridExt: extract_info(game_text)
        activate HybridExt
        HybridExt->>Jericho: Z-machine direct access (location, inventory, objects, score)
        Jericho-->>HybridExt: Structured data
        HybridExt->>HybridExt: LLM parse (exits, combat, messages)
        HybridExt-->>Orch: ExtractorResponse
        deactivate HybridExt

        Orch->>StateMgr: track_state_hash()
        activate StateMgr
        StateMgr->>Jericho: get_state_hash()
        Jericho-->>StateMgr: Z-machine state hash
        StateMgr-->>Orch: Loop detection result
        deactivate StateMgr

        Orch->>MapMgr: update_from_movement(old_id, new_id, action, exits)
        activate MapMgr
        MapMgr->>MapMgr: Compare location IDs (perfect movement detection)
        MapMgr->>MapMgr: Update connections, track failures
        MapMgr-->>Orch: Map updated
        deactivate MapMgr

        Orch->>MemoryMgr: record_action_outcome()
        activate MemoryMgr
        MemoryMgr->>MemoryMgr: Check synthesis triggers (score/location/death)
        opt Trigger Met
            MemoryMgr->>MemoryMgr: LLM synthesize memory
            MemoryMgr->>MemoryMgr: Update Memories.md with file locking
        end
        MemoryMgr-->>Orch: Memory recorded
        deactivate MemoryMgr

        Orch->>ContextMgr: add_action/memory/reasoning()
        Orch->>KnowledgeMgr: detect_object_events()

        Orch->>ObjectiveMgr: check_objective_completion()
        activate ObjectiveMgr
        ObjectiveMgr-->>Orch: Completion status
        deactivate ObjectiveMgr

        alt Every 20 turns (Objective Updates)
            Orch->>ObjectiveMgr: check_and_update_objectives()
            activate ObjectiveMgr
            ObjectiveMgr->>ObjectiveMgr: LLM discover new objectives
            ObjectiveMgr->>ObjectiveMgr: Check staleness, refinement
            ObjectiveMgr-->>Orch: Updated objectives
            deactivate ObjectiveMgr
        end

        alt Every 50 turns (Knowledge Updates)
            Orch->>KnowledgeMgr: check_periodic_update()
            activate KnowledgeMgr
            KnowledgeMgr->>KnowledgeMgr: Analyze full episode history
            KnowledgeMgr->>KnowledgeMgr: Update knowledgebase.md sections
            KnowledgeMgr->>Agent: Reload knowledge base
            KnowledgeMgr-->>Orch: Knowledge updated
            deactivate KnowledgeMgr
        end

        Orch->>StateMgr: export_current_state()
        activate StateMgr
        StateMgr->>StateMgr: Check context overflow (>80% tokens)
        opt Context Overflow
            StateMgr->>StateMgr: LLM summarize history, prune memories
        end
        StateMgr->>StateMgr: Write state JSON (local + S3)
        StateMgr-->>Orch: State exported
        deactivate StateMgr

        Orch->>RejectionMgr: update_movement_tracking()

        alt Death or Max Turns
            Note over Orch: End gameplay loop
        end
    end

    Orch->>EpisodeSynth: finalize_episode()
    activate EpisodeSynth
    EpisodeSynth->>KnowledgeMgr: perform_final_update()
    activate KnowledgeMgr
    KnowledgeMgr-->>EpisodeSynth: Final update complete
    deactivate KnowledgeMgr

    opt Death or High Score/Turns
        EpisodeSynth->>KnowledgeMgr: perform_inter_episode_synthesis()
        activate KnowledgeMgr
        KnowledgeMgr->>KnowledgeMgr: Synthesize CROSS-EPISODE INSIGHTS
        KnowledgeMgr-->>EpisodeSynth: Inter-episode wisdom updated
        deactivate KnowledgeMgr
    end

    EpisodeSynth-->>Orch: Episode finalized
    deactivate EpisodeSynth

    Orch->>MapMgr: save_map_state()
    activate MapMgr
    MapMgr->>MapMgr: Persist to map_state.json
    MapMgr-->>Orch: Map persisted
    deactivate MapMgr

    Orch-->>UserMain: Episode complete (score, summary)
    deactivate Orch
```
