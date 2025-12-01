# ZorkGPT: An LLM-Powered Agent for Interactive Fiction

**🎮 Watch ZorkGPT play live at [https://zorkgpt.com](https://zorkgpt.com)**

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Core Research Approach](#core-research-approach)
4. [System Architecture](#system-architecture)
5. [Subsystems](#subsystems)
6. [System Execution Flow](#system-execution-flow)
7. [Architecture Diagrams](#architecture-diagrams)
8. [MCP Integration](#mcp-integration)

## Project Overview

Interactive fiction games like Zork present a compelling challenge for artificial intelligence: they require long-horizon planning, spatial reasoning, puzzle solving, and the ability to learn from failure—all through natural language interaction. Traditional approaches rely on reinforcement learning with carefully shaped rewards or extensive offline training. Can large language models achieve genuine competence in these complex environments through pure reasoning and real-time adaptive learning, without any hardcoded solutions?

ZorkGPT explores this question by demonstrating an AI agent system that plays the classic text adventure game "[Zork](https://en.wikipedia.org/wiki/Zork)" using only LLM-driven reasoning and continuous in-game learning. The system achieves extended gameplay sessions spanning thousands of turns, during which it builds spatial knowledge, discovers objectives, synthesizes strategic insights, and accumulates cross-episode wisdom—all without predetermined solutions or game-specific programming.

The research contribution lies in demonstrating that LLMs can develop authentic gameplay competence through a combination of specialized cognitive modules (action generation, evaluation, knowledge synthesis), direct game state access via the Z-machine, and a sophisticated multi-step learning pipeline that continuously refines strategic understanding during play.

## Quick Start

```bash
# Prerequisites: Python 3.11+, uv installed

# 1. Clone and setup
git clone https://github.com/stickystyle/ZorkGPT
cd ZorkGPT
uv sync

# 2. Configure API keys
cp .env.example .env
# Edit .env with your LLM API keys (OpenAI, Anthropic, etc.)

# 3. Run a gameplay episode
uv run python main.py

# Watch the agent play live or review episode logs in game_files/episodes/
```

## Core Research Approach

ZorkGPT operates under four fundamental research principles:

### LLM-First Design

All game reasoning, decision-making, and understanding must originate from language models. The system deliberately avoids hardcoded game mechanics, location databases, or predetermined solutions. This constraint forces the agent to rely on genuine language model capabilities rather than falling back on programmatic shortcuts.

### No Hardcoded Solutions

The only acceptable hardcoded logic validates whether the game engine accepted a command. Puzzle solutions, navigation choices, and strategic decisions must emerge entirely from LLM reasoning. This ensures that observed competence reflects authentic AI capabilities, not disguised rule-based systems.

### Adaptive Learning During Gameplay

The system implements continuous knowledge extraction and strategy refinement *during* gameplay sessions, not just between them. The agent analyzes its experiences in real-time, updating its strategic understanding at regular intervals (every 100 turns by default). This allows immediate incorporation of new insights and demonstrates online learning capabilities.

### Genuine AI Play

The objective is to have LLMs genuinely "play" the game, demonstrating authentic language model capabilities in complex, interactive environments. When the agent encounters challenges, the solution is to improve prompts, models, or context—never to introduce fallback mechanisms or hardcoded assistance.

## System Architecture

ZorkGPT employs a modular architecture coordinated by a central orchestrator, with specialized LLM-powered components handling distinct cognitive functions and supporting systems managing game interaction and data persistence.

### Central Coordinator

The **ZorkOrchestratorV2** serves as the primary coordination layer, managing extended gameplay sessions that can span thousands of turns. It orchestrates interactions between all system components, handles the main game loop, coordinates periodic knowledge updates, and manages episode lifecycle from initialization through finalization and cross-episode synthesis.

### LLM-Powered Cognitive Modules

Four specialized LLM components form the cognitive core of the agent:

**Agent LM** generates actions by analyzing current game state, integrating memories from previous turns, consulting spatial knowledge from the map system, and following strategic guidance from the knowledge base. It receives structured Z-machine object data including attributes and valid action verbs to inform its reasoning.

**Extractor LM** operates as a hybrid system: it retrieves inventory, location, score, and visible objects directly from Z-machine memory (bypassing text parsing), while using LLM reasoning to extract exits, combat status, and important narrative messages from game text. This hybrid approach reduces LLM calls significantly per turn while maintaining reasoning quality.

**Critic LM** evaluates proposed actions before execution through a two-stage process. It first performs fast object tree validation (microseconds) to catch impossible actions, then conducts LLM-based evaluation assessing relevance, progress potential, risk, and strategic alignment. This reduces expensive LLM calls significantly for invalid actions. The Critic incorporates a trust calibration mechanism that adapts its strictness based on recent agent performance.

**Strategy Generator LM** drives the continuous learning process by analyzing gameplay data within turn windows (typically 100 turns), identifying successful tactics and patterns, and synthesizing strategic insights. It assesses the quality of new information and intelligently merges insights into the existing knowledge base, ensuring productive learning without knowledge degradation.

### Supporting Systems

Several specialized systems support the LLM cognitive modules:

**Jericho Interface** manages low-level interaction with the Z-machine game engine through the Jericho library, providing direct memory access for instant retrieval of inventory, location, score, and object data without text parsing. It uses integer-based location IDs for stable room identification, eliminating fragmentation issues that plague text-based parsing. The interface exposes the object tree for validation and provides built-in save/restore capabilities, enabling perfect movement detection through location ID comparison.

**Map System** builds and maintains a dynamic graph-based representation of the game world using integer location IDs from the Z-machine. It tracks connection confidence scores, analyzes movement patterns, and prunes consistently failing exits based on empirical evidence from gameplay.

**Memory System** implements a multi-step synthesis pipeline that transforms raw action history into location-specific memories and ultimately strategic knowledge. Memories are stored at source locations (not destinations) to enable effective cross-episode learning and are deduplicated to prevent redundancy.

**State Manager** handles game state persistence, context overflow detection (triggering LLM-based summarization when token limits approach), state loop detection (alerting when exact game states repeat), and exports state to local storage and S3.

## Subsystems

### Adaptive Knowledge System

The adaptive knowledge system enables ZorkGPT to improve continuously during gameplay through real-time analysis and knowledge synthesis. Rather than learning only between episodes, the agent analyzes its recent experiences every 100 turns (by default) to extract strategic insights.

The system operates through an LLM-driven assessment process that determines what constitutes valuable knowledge and how to integrate it. The Strategy Generator analyzes turn windows to identify successful tactics, failed approaches, and emerging patterns. New insights are intelligently merged into the existing knowledge base in a way that enhances rather than overwrites prior learning, preventing knowledge degradation.

The output is a dynamically updated knowledge base that provides strategic guidance to the Agent LM, influencing its decisions. The system also contributes to objective discovery and prioritization—high-level goals that emerge from gameplay analysis and help guide the agent's long-term decision-making and exploration focus.

At episode boundaries, particularly after significant progress or failure (death), the system performs cross-episode synthesis to distill validated wisdom that persists across gameplay sessions.

### Spatial Intelligence System

Understanding and navigating Zork's complex geography is crucial for progress. ZorkGPT builds a dynamic graph-based map where locations are nodes identified by stable integer IDs from the Z-machine, and connections are edges with associated confidence scores.

The system achieves perfect movement detection by comparing location IDs before and after actions, eliminating the ambiguity inherent in text-based approaches. Movement patterns are analyzed to verify connections, identify efficient routes, and recognize important spatial features like hub rooms or dead ends.

Spatial information—current location, known exits, map-based relationships—is provided to the Agent LM to inform navigation decisions. The system tracks failed movement attempts and prunes exits that consistently fail, preventing repeated errors based on empirical evidence from gameplay.

### Memory System

ZorkGPT implements a multi-step memory hierarchy that converts raw experiences into strategic understanding:

**Action History** → **Location Memories** → **Strategic Knowledge** → **Cross-Episode Wisdom**

Location-specific memories are synthesized when certain triggers occur: score changes, location changes, deaths, or manual triggers. The Memory Manager uses LLM reasoning to analyze recent actions and outcomes, creating concise memories that capture what happened and why it matters.

Memories are stored at the *source* location where they were created, not at destination locations. This design enables cross-episode learning: when revisiting a location, the agent receives memories from all previous episodes at that location, regardless of when they occurred.

The system handles supersession, allowing new memories to replace or refine earlier ones when better understanding emerges. Deduplication prevents redundant memories from cluttering the knowledge base.

## System Execution Flow

A typical gameplay turn proceeds through these stages:

1. **Observation**: The system receives game text from the Z-machine after the previous action.

2. **Hybrid Extraction**: Inventory, location ID, score, and visible objects are retrieved instantly from Z-machine memory. The Extractor LM parses game text only for exits, combat status, and room description detection.

3. **State Update**: Session memory is updated with new information. The map system compares location IDs to detect movement with perfect accuracy. State hash tracking identifies exact state loops.

4. **Context Assembly**: The Context Manager gathers relevant memories from the current location, spatial data from the map, strategic knowledge from the knowledge base, and structured object data from the Z-machine, assembling a comprehensive context for the Agent LM.

5. **Action Generation**: The Agent LM analyzes the context and proposes an action, informed by object attributes, valid action vocabulary, spatial relationships, and strategic guidance.

6. **Action Evaluation**: The Critic performs fast object tree validation (microseconds). If the action passes, it conducts LLM-based evaluation (~800ms). Low-scoring actions may trigger re-generation with feedback, subject to trust calibration and rejection override logic.

7. **Execution**: The chosen action is sent to the Z-machine game engine.

8. **Periodic Learning**: At regular intervals, specialized updates occur:
   - Every 25 turns: Objective discovery and completion tracking
   - Every 100 turns: Strategic knowledge synthesis and knowledge base updates
   - Every turn: Memory synthesis triggers (score change, location change, death)
   - Continuous: State export, context overflow detection, map updates

This cycle repeats over extended sessions, allowing the agent to explore, learn, and improve over thousands of turns.

## Architecture Diagrams

### Component Architecture

The following diagram illustrates the relationships between ZorkGPT's components in its manager-based architecture. Solid lines show direct dependencies between components, while dashed lines indicate data persistence relationships. The architecture separates concerns into coordination (orchestrator), session management (shared state), specialized managers (objectives, knowledge, map, memory, state, context, episodes, rejection), LLM-powered components (agent, extractor, critic, strategy), supporting systems (Jericho, map graph, logger, LLM client), and persistent data stores.

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

### Turn-by-Turn Execution Flow

This sequence diagram details the execution flow for a single gameplay turn, showing how the orchestrator coordinates managers, LLM components, the Jericho interface, and persistent data stores. Note the two-stage critic evaluation (fast object tree validation followed by LLM evaluation only when needed), the hybrid extraction process (Z-machine direct access + selective LLM parsing), and periodic update triggers based on turn counts.

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

        alt Every 25 turns (Objective Updates)
            Orch->>ObjectiveMgr: check_and_update_objectives()
            activate ObjectiveMgr
            ObjectiveMgr->>ObjectiveMgr: LLM discover new objectives
            ObjectiveMgr->>ObjectiveMgr: Check staleness, refinement
            ObjectiveMgr-->>Orch: Updated objectives
            deactivate ObjectiveMgr
        end

        alt Every 100 turns (Knowledge Updates)
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

## MCP Integration

ZorkGPT supports Model Context Protocol (MCP) to give the agent access to external reasoning tools during gameplay. When enabled, the agent can use [thoughtbox](https://github.com/kastalien-research/thoughtbox) - a structured meta-cognitive reasoning tool that helps with complex puzzle solving.

### Enabling MCP

1. Install thoughtbox locally:
   ```bash
   npm install @kastalien-research/thoughtbox
   ```
   Note: The thoughtbox project recommends npx, but local installation works more reliably.

2. Create `mcp_config.json` in project root:
   ```json
   {
     "mcpServers": {
       "thoughtbox": {
         "command": "node",
         "args": ["node_modules/@kastalien-research/thoughtbox/dist/index.js"],
         "env": {
           "DISABLE_THOUGHT_LOGGING": "true"
         }
       }
     }
   }
   ```

3. Enable in `pyproject.toml`:
   ```toml
   [tool.zorkgpt.mcp]
   enabled = true
   ```

### Disabling MCP

Set `enabled = false` in the `[tool.zorkgpt.mcp]` section of `pyproject.toml`, or remove the section entirely.

### Requirements

- Node.js 22+ (for thoughtbox server)
- Python 3.11+ (for async support)
- mcp package >= 1.22.0 (included in dependencies)
