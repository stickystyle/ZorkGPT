# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: Do not make any changes until you have 95% confidence in the change you need to make. Ask me questions until you reach that confidence.

## Project Overview

ZorkGPT is an AI agent system that plays the classic text adventure game "Zork" using Large Language Models. The system uses a modular architecture with specialized LLM-driven components for action generation, information extraction, action evaluation, and adaptive learning.

**Key Principle**: All game reasoning must originate from LLMs - no hardcoded solutions or predetermined game mechanics are allowed.

## Architecture Overview

### Core Components

- **JerichoInterface** (`game_interface/core/jericho_interface.py`): Direct Z-machine access via Jericho library
- **Orchestrator** (`orchestration/zork_orchestrator_v2.py`): Streamlined coordination layer using JerichoInterface
- **Managers** (`managers/`): Specialized components for objectives, knowledge, map, memory, context, and state
- **LLM Components** (root): Agent, Critic, Extractor - LLM-powered decision making

### Key Architectural Benefits

- Direct Z-machine memory access (no text parsing for inventory, location, score)
- Stable integer-based location IDs (eliminates room fragmentation)
- Perfect movement detection via ID comparison
- Multi-step memory synthesis across turns
- Cross-episode knowledge accumulation

## Critical Architectural Constraints

**These are invariants that prevent breaking changes. Always follow these rules:**

1. **All game reasoning from LLMs** - No hardcoded solutions or predetermined game mechanics
2. **Use `location.num` for room IDs** - NEVER use room names as primary keys
3. **Store memories at SOURCE location** - Not destination (enables cross-episode learning)
4. **Use Z-machine data directly** - Don't parse text when structured data is available
5. **Validate with object tree before LLM calls** - Fast rejection before expensive evaluation

## Subsystem Documentation

**Each major subsystem has its own CLAUDE.md with detailed patterns and examples. Consult these when working in that area:**

### Game Interface & Jericho Integration
**Working with game state, Z-machine, movement detection, or object tree validation?**
→ See `game_interface/CLAUDE.md`

Key topics: Z-machine data access, location IDs, movement detection, object tree validation, performance metrics

### Managers & Memory System
**Adding/modifying managers, working with memory synthesis, or reasoning history?**
→ See `managers/CLAUDE.md`

Key topics: Manager pattern, lifecycle, dependencies, multi-step memory synthesis, supersession workflow, reasoning history, source location storage

### Testing & Quality
**Writing tests, using walkthrough fixtures, or running benchmarks?**
→ See `tests/CLAUDE.md`

Key topics: Walkthrough fixtures, test patterns, deterministic testing, integration tests, debugging failed tests

### Knowledge System
**Working with knowledgebase, cross-episode learning, or strategic wisdom?**
→ See `knowledge/CLAUDE.md`

Key topics: Knowledge base structure, cross-episode insights, synthesis triggers, knowledge vs memory distinction

### Game Configuration
**Working with game files, prompts, or agent configuration?**
→ See `game_files/CLAUDE.md` (already exists)

## Quick Start

### Run the Agent

```bash
# Run single episode
uv run python main.py

# Run with specific config
uv run python main.py --config custom_config.toml
```

### Run Tests

```bash
# Fast test suite (skip slow tests)
uv run pytest tests/ -k "not slow" -q

# Run specific test file
uv run pytest tests/test_map_persistence.py -v

# Run with detailed output
uv run pytest tests/ -xvs --tb=short

# Run benchmarks
uv run python benchmarks/comparison_report.py
```

## Key Design Patterns

### Manager Pattern
All managers follow standardized lifecycle: initialization → reset → processing → status. See `managers/CLAUDE.md` for details.

### Integer-Based Maps
MapGraph uses `Dict[int, Room]` with location IDs from Z-machine. No consolidation needed - IDs are unique by design.

### Memory Hierarchy
```
Action → Memory (location-specific) → Knowledge (strategic) → Cross-Episode Wisdom (validated)
```

### Z-Machine First
Always prefer Z-machine structured data over text parsing:
- Location: `get_location_structured()` → `location.num`
- Inventory: `get_inventory_structured()` → `List[ZObject]`
- Movement: Compare `before_id != after_id`
- Objects: `get_visible_objects_in_location()` → `List[ZObject]`

### Loop Break System
ZorkGPT includes a three-phase loop break system to prevent stuck episodes and token waste:

**Phase 1A - Progress Velocity Detection**: Terminates episodes after 40 turns without score change. Programmatic hard stop using O(1) score tracking. See `orchestration/zork_orchestrator_v2.py` lines 347-380 (tracking) and 526-547 (termination).

**Phase 1B - Location Revisit Penalty**: Applies programmatic -0.2 penalty per recent location revisit to discourage loops. Uses Z-machine location IDs in sliding window (last 5 locations). Modifies critic confidence scores, not context. See lines 394-504.

**Phase 1C - Stuck Countdown Warnings**: Shows explicit "DIE in {y} turns" warnings in agent context starting at 20 turns stuck. Includes action novelty hints and unexplored exit suggestions. Context-based guidance, not penalties. See lines 483-522 (warnings) and 849-866 (critic integration).

**Configuration**: All settings in `pyproject.toml` under `[tool.zorkgpt.loop_break]` section. Loaded via `session/game_configuration.py` lines 220-282.

**Testing**: 66 tests across 4 files validate all three phases independently and together. Run: `uv run pytest tests/test_*loop*.py -v`

**For complete details**: See `loop_break.md` for design rationale, configuration options, monitoring, troubleshooting, and episode analysis tools.

## File Organization

```
ZorkGPT/
├── CLAUDE.md                    # This file (routing hub)
├── game_interface/
│   ├── CLAUDE.md                # Jericho/Z-machine patterns
│   └── core/
│       └── jericho_interface.py # Z-machine access layer
├── managers/
│   ├── CLAUDE.md                # Manager patterns & memory system
│   ├── simple_memory_manager.py # Location-specific memory
│   ├── knowledge_manager.py     # Strategic knowledge
│   ├── map_manager.py           # Spatial navigation
│   ├── objective_manager.py     # Goal tracking
│   ├── context_manager.py       # Prompt assembly
│   └── state_manager.py         # State export
├── orchestration/
│   └── zork_orchestrator_v2.py  # Main coordination layer
├── tests/
│   ├── CLAUDE.md                # Testing patterns
│   └── fixtures/
│       └── walkthrough.py       # Deterministic test fixtures
├── knowledge/
│   └── CLAUDE.md                # Knowledge base system
├── game_files/
│   ├── CLAUDE.md                # Game configuration
│   ├── knowledgebase.md         # Strategic wisdom
│   └── memories.md              # Location-specific memories
├── zork_agent.py                # LLM action generation
├── zork_critic.py               # LLM action evaluation
├── hybrid_zork_extractor.py     # LLM text parsing
└── map_graph.py                 # Map data structure
```

## Common Questions

**Q: How do I add a new manager?**
A: See `managers/CLAUDE.md` for manager pattern, lifecycle, and dependencies.

**Q: Why are tests failing after my changes?**
A: See `tests/CLAUDE.md` for debugging patterns and common issues.

**Q: How do I work with game state?**
A: See `game_interface/CLAUDE.md` for Z-machine data access patterns.

**Q: How does the knowledge system work?**
A: See `knowledge/CLAUDE.md` for knowledge base structure and synthesis.

**Q: How does room description extraction work?**
A: See `managers/CLAUDE.md` (Room Description Extraction section) for full details. Brief: Extractor flags room descriptions (boolean), orchestrator stores original text, ContextManager adds to prompts when recent and location-matched. Configurable aging window (default: 10 turns).

---

**Remember**: Each subsystem's CLAUDE.md contains detailed patterns, examples, and common pitfalls. This file is your routing hub - consult the specific subsystem documentation when working in that area.
