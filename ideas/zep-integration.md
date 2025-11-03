# Zep Integration Specification for ZorkGPT

## Executive Summary

This document specifies the integration of Zep, a temporal knowledge graph-based memory system, into ZorkGPT. Zep offers superior temporal tracking, automatic fact extraction, and dramatic token reduction (98% efficiency) compared to traditional approaches. This integration will provide ZorkGPT with sophisticated memory capabilities while maintaining compatibility with the existing manager-based architecture.

## Why Zep for ZorkGPT

### Key Advantages

1. **Temporal Intelligence**: Zep's bi-temporal knowledge graph tracks how game state evolves over time, perfect for understanding changing game conditions
2. **Token Efficiency**: Uses <2% of baseline tokens while improving accuracy to 94.8%
3. **Automatic Learning**: Extracts facts and patterns without predefined schemas
4. **Performance**: 90% latency reduction with 18.5% accuracy improvements
5. **Enterprise-Ready**: Battle-tested in production environments

### ZorkGPT-Specific Benefits

- **Episode Memory**: Track what worked/failed across multiple game sessions
- **Dynamic Strategy Evolution**: Facts automatically update as new strategies are discovered
- **Critic Cost Reduction**: Token efficiency dramatically reduces critic LLM costs
- **Intent Detection**: Built-in dialog classification helps understand game responses

## Architecture Overview

### Hybrid Approach: Zep + SQLite

Given Zep's lack of native spatial features, we propose a hybrid architecture:

```
┌─────────────────────────────────────────────────────┐
│                Agent Decision Layer                  │
└─────────────┬──────────────────────────┬────────────┘
              │                          │
    ┌─────────▼────────┐       ┌────────▼────────┐
    │  Working Memory  │       │  Context Query   │
    │  (Last 20 turns) │       │    Interface     │
    └─────────┬────────┘       └────────┬────────┘
              │                          │
    ┌─────────▼──────────────────────────▼────────┐
    │            Hybrid Memory System              │
    │                                              │
    │  ┌──────────────┐    ┌─────────────────┐   │
    │  │  Zep Cloud   │    │  SQLite Spatial │   │
    │  │              │    │                 │   │
    │  │  • Episodes  │    │  • Room Graph   │   │
    │  │  • Facts     │    │  • Connections  │   │
    │  │  • Patterns  │    │  • Navigation   │   │
    │  │  • Wisdom    │    │  • Local Items  │   │
    │  └──────────────┘    └─────────────────┘   │
    └───────────────────────────────────────────────┘
```

## Core Components

### 1. ZepMemoryManager Class

```python
# managers/zep_memory_manager.py
from typing import List, Dict, Any, Optional
from zep_cloud import Zep, Memory, Message, Session, Fact
from managers.base_manager import BaseManager
import json

class ZepMemoryManager(BaseManager):
    """
    Manages Zep integration for episodic memory and learning.
    
    Responsibilities:
    - Store turn-by-turn experiences with temporal context
    - Extract facts and patterns automatically
    - Retrieve relevant memories based on context
    - Handle memory persistence across episodes
    """
    
    def __init__(self, logger, config, game_state):
        super().__init__(logger, config, game_state, "zep_memory_manager")
        
        # Initialize Zep client
        self.zep = Zep(
            api_key=config.zep_api_key,
            base_url=config.zep_base_url if hasattr(config, 'zep_base_url') else None
        )
        
        # Create or get session for this episode
        self.session = self._init_session()
        
        # Track current context
        self.current_context = {}
        
    def _init_session(self) -> Session:
        """Initialize or retrieve Zep session for this episode."""
        session_id = f"zorkgpt_{self.game_state.episode_id}"
        
        try:
            # Try to get existing session
            session = self.zep.memory.get_session(session_id)
        except:
            # Create new session
            session = self.zep.memory.add_session(
                session_id=session_id,
                metadata={
                    "game": "zork",
                    "episode_id": self.game_state.episode_id,
                    "start_time": self.game_state.start_time,
                    "agent_version": self.config.agent_version
                }
            )
        
        return session
    
    def store_turn_experience(self, action: str, response: str, analysis: Dict[str, Any]) -> None:
        """Store a game turn as a conversation in Zep."""
        
        # Create messages for this turn
        messages = [
            Message(
                role="Ryan",  # Player (agent)
                role_type="human",
                content=action,
                metadata={
                    "turn": self.game_state.turn_count,
                    "location": self.game_state.current_location,
                    "score": self.game_state.score,
                    "inventory": json.dumps(self.game_state.current_inventory)
                }
            ),
            Message(
                role="Zork",  # Game
                role_type="assistant", 
                content=response,
                metadata={
                    "success": analysis.get("success", False),
                    "moved": analysis.get("moved", False),
                    "score_change": analysis.get("score_change", 0),
                    "new_items": json.dumps(analysis.get("new_items", [])),
                    "combat": analysis.get("combat", False)
                }
            )
        ]
        
        # Add to Zep memory
        self.zep.memory.add(
            session_id=self.session.session_id,
            messages=messages
        )
        
        # Log the storage
        self.logger.debug(
            f"Stored turn {self.game_state.turn_count} to Zep",
            extra={
                "event_type": "zep_store",
                "turn": self.game_state.turn_count,
                "action": action[:50],
                "location": self.game_state.current_location
            }
        )
    
    def get_relevant_memories(self, context_query: str = None, limit: int = 5) -> Dict[str, Any]:
        """Retrieve relevant memories using Zep's semantic search."""
        
        # Use Zep's high-level memory.get() API
        memory_response = self.zep.memory.get(
            session_id=self.session.session_id,
            limit=limit
        )
        
        # Extract different memory types
        memories = {
            "recent": [],
            "facts": [],
            "relevant": [],
            "patterns": []
        }
        
        # Process recent messages
        if memory_response.messages:
            for msg in memory_response.messages[-10:]:  # Last 10 messages
                memories["recent"].append({
                    "content": msg.content,
                    "role": msg.role,
                    "metadata": msg.metadata
                })
        
        # Process extracted facts
        if memory_response.facts:
            for fact in memory_response.facts:
                memories["facts"].append({
                    "fact": fact.fact,
                    "confidence": fact.confidence,
                    "created_at": fact.created_at
                })
        
        # Search for similar experiences if context provided
        if context_query:
            search_results = self.zep.memory.search(
                session_id=self.session.session_id,
                text=context_query,
                limit=limit
            )
            
            for result in search_results:
                memories["relevant"].append({
                    "content": result.message.content,
                    "similarity": result.score,
                    "metadata": result.message.metadata
                })
        
        return memories
    
    def extract_patterns(self) -> List[Dict[str, Any]]:
        """Let Zep automatically extract patterns from conversation."""
        
        # Get all facts from session
        facts = self.zep.memory.get_facts(
            session_id=self.session.session_id
        )
        
        patterns = []
        for fact in facts:
            # Categorize facts into patterns
            if any(keyword in fact.fact.lower() for keyword in ["always", "never", "every time", "whenever"]):
                patterns.append({
                    "pattern": fact.fact,
                    "confidence": fact.confidence,
                    "type": "universal",
                    "discovered_at": fact.created_at
                })
            elif any(keyword in fact.fact.lower() for keyword in ["usually", "often", "sometimes"]):
                patterns.append({
                    "pattern": fact.fact,
                    "confidence": fact.confidence,
                    "type": "probabilistic",
                    "discovered_at": fact.created_at
                })
        
        return patterns
    
    def get_location_memories(self, location: str) -> Dict[str, Any]:
        """Get memories specific to a location (using metadata search)."""
        
        # Search messages with location metadata
        search_query = f"location: {location}"
        results = self.zep.memory.search(
            session_id=self.session.session_id,
            text=search_query,
            search_scope="metadata",
            limit=10
        )
        
        location_data = {
            "experiences": [],
            "dangers": [],
            "items_found": [],
            "successful_actions": []
        }
        
        for result in results:
            metadata = result.message.metadata
            
            # Categorize by outcome
            if metadata.get("combat"):
                location_data["dangers"].append(result.message.content)
            elif metadata.get("new_items"):
                location_data["items_found"].extend(json.loads(metadata["new_items"]))
            elif metadata.get("success"):
                location_data["successful_actions"].append(result.message.content)
            
            location_data["experiences"].append({
                "action": result.message.content,
                "turn": metadata.get("turn"),
                "outcome": metadata
            })
        
        return location_data
```

### 2. Spatial Memory Bridge (SQLite)

```python
# managers/spatial_memory_bridge.py
import sqlite3
from typing import List, Tuple, Optional, Dict
from pathlib import Path

class SpatialMemoryBridge:
    """
    Local SQLite database for spatial/navigation data.
    Complements Zep with fast local spatial queries.
    """
    
    def __init__(self, db_path: str = "zorkgpt_spatial.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        """Create spatial database schema."""
        cursor = self.conn.cursor()
        
        # Spatial graph table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS room_graph (
                from_room TEXT,
                to_room TEXT,
                direction TEXT,
                confidence REAL DEFAULT 1.0,
                verified_count INTEGER DEFAULT 0,
                last_verified INTEGER,
                PRIMARY KEY (from_room, to_room, direction)
            )
        """)
        
        # Room metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                name TEXT PRIMARY KEY,
                base_name TEXT,
                description TEXT,
                visits INTEGER DEFAULT 0,
                last_visit INTEGER,
                danger_level REAL DEFAULT 0.0,
                items TEXT,  -- JSON array
                npcs TEXT    -- JSON array
            )
        """)
        
        # Failed exits tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failed_exits (
                room TEXT,
                direction TEXT,
                attempts INTEGER DEFAULT 0,
                last_attempt INTEGER,
                PRIMARY KEY (room, direction)
            )
        """)
        
        self.conn.commit()
    
    def add_room_connection(self, from_room: str, to_room: str, direction: str, confidence: float = 1.0):
        """Add or update a room connection."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO room_graph 
            (from_room, to_room, direction, confidence, verified_count, last_verified)
            VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT verified_count FROM room_graph 
                             WHERE from_room=? AND to_room=? AND direction=?), 0) + 1,
                    ?)
        """, (from_room, to_room, direction, confidence, from_room, to_room, direction, 
              self._current_timestamp()))
        self.conn.commit()
    
    def find_path(self, from_room: str, to_room: str, max_depth: int = 10) -> Optional[List[str]]:
        """Find shortest path between rooms using BFS."""
        # Implement BFS for pathfinding
        visited = set()
        queue = [(from_room, [])]
        
        while queue and len(queue[0][1]) < max_depth:
            current_room, path = queue.pop(0)
            
            if current_room == to_room:
                return path
            
            if current_room in visited:
                continue
                
            visited.add(current_room)
            
            # Get neighbors
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT to_room, direction FROM room_graph 
                WHERE from_room = ? AND confidence > 0.5
                ORDER BY confidence DESC
            """, (current_room,))
            
            for neighbor, direction in cursor.fetchall():
                if neighbor not in visited:
                    queue.append((neighbor, path + [direction]))
        
        return None
    
    def get_nearby_rooms(self, room: str, radius: int = 2) -> List[Tuple[str, int]]:
        """Get rooms within radius moves of current room."""
        visited = {}
        queue = [(room, 0)]
        
        while queue:
            current_room, distance = queue.pop(0)
            
            if distance > radius:
                continue
                
            if current_room in visited:
                continue
                
            visited[current_room] = distance
            
            # Get neighbors
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT to_room FROM room_graph 
                WHERE from_room = ? AND confidence > 0.5
            """, (current_room,))
            
            for (neighbor,) in cursor.fetchall():
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        return [(r, d) for r, d in visited.items() if r != room]
```

### 3. Enhanced Memory-Informed Critic

```python
# zork_memory_critic_zep.py
from typing import Optional, Dict, Any
from zep_cloud import Zep
from managers.zep_memory_manager import ZepMemoryManager
from managers.spatial_memory_bridge import SpatialMemoryBridge

class ZepMemoryInformedCritic:
    """
    Critic system using Zep's efficient memory retrieval.
    Leverages Zep's <2% token usage for dramatic cost reduction.
    """
    
    def __init__(self, zep_manager: ZepMemoryManager, spatial_bridge: SpatialMemoryBridge, 
                 base_critic, config, logger):
        self.zep_manager = zep_manager
        self.spatial_bridge = spatial_bridge
        self.base_critic = base_critic
        self.config = config
        self.logger = logger
        
        self.stats = {
            "total_evaluations": 0,
            "memory_bypasses": 0,
            "llm_invocations": 0,
            "tokens_saved": 0
        }
    
    def evaluate_action(self, game_state_text: str, proposed_action: str, 
                        current_location: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate action using Zep's efficient memory retrieval."""
        
        self.stats["total_evaluations"] += 1
        
        # Build minimal context query using Zep
        query = f"Action: {proposed_action} at {current_location}"
        
        # Get relevant memories (Zep uses <2% of tokens)
        memories = self.zep_manager.get_relevant_memories(
            context_query=query,
            limit=3  # Zep is so efficient we can check more
        )
        
        # Check for similar past experiences
        if memories["relevant"]:
            best_match = memories["relevant"][0]
            if best_match["similarity"] > 0.85:
                # High confidence match - use cached decision
                self.stats["memory_bypasses"] += 1
                self.stats["tokens_saved"] += 15000  # Approximate full context size
                
                return {
                    "score": best_match["metadata"].get("success", 0) * 0.8 - 0.2,
                    "justification": f"Based on similar past experience (confidence: {best_match['similarity']:.2f})",
                    "source": "zep_memory",
                    "llm_invoked": False
                }
        
        # Check facts for universal patterns
        for fact in memories.get("facts", []):
            if proposed_action.lower() in fact["fact"].lower():
                if fact["confidence"] > 0.8:
                    self.stats["memory_bypasses"] += 1
                    self.stats["tokens_saved"] += 15000
                    
                    # Determine score based on fact sentiment
                    is_negative = any(word in fact["fact"].lower() 
                                    for word in ["never", "don't", "avoid", "dangerous"])
                    score = -0.7 if is_negative else 0.6
                    
                    return {
                        "score": score,
                        "justification": f"Known fact: {fact['fact']} (confidence: {fact['confidence']:.2f})",
                        "source": "zep_fact",
                        "llm_invoked": False
                    }
        
        # Check spatial context for navigation
        if any(direction in proposed_action.lower() 
               for direction in ["north", "south", "east", "west", "up", "down"]):
            
            # Quick spatial check from local SQLite
            direction = proposed_action.lower().split()[0]
            path_exists = self.spatial_bridge.check_exit(current_location, direction)
            
            if not path_exists:
                self.stats["memory_bypasses"] += 1
                self.stats["tokens_saved"] += 15000
                
                return {
                    "score": -0.9,
                    "justification": f"No known exit {direction} from {current_location}",
                    "source": "spatial_memory",
                    "llm_invoked": False
                }
        
        # Fall back to LLM critic with minimal context
        self.stats["llm_invocations"] += 1
        
        # Build ultra-minimal context from Zep (< 1K tokens)
        minimal_context = {
            "recent_facts": [f["fact"][:100] for f in memories.get("facts", [])[:2]],
            "location_hints": memories.get("recent", [])[-2:] if memories.get("recent") else [],
            "proposed_action": proposed_action,
            "location": current_location
        }
        
        return self.base_critic.evaluate_action(
            game_state_text=game_state_text,
            proposed_action=proposed_action,
            context=minimal_context
        )
```

### 4. Integration with Orchestrator

```python
# In orchestration/zork_orchestrator_v2.py modifications

def __init__(self, episode_id: str = None):
    # ... existing initialization ...
    
    # Initialize Zep memory system
    self.zep_memory = ZepMemoryManager(
        logger=self.logger,
        config=self.config,
        game_state=self.game_state
    )
    
    # Initialize spatial bridge
    self.spatial_bridge = SpatialMemoryBridge(
        db_path=f"spatial_{episode_id}.db"
    )
    
    # Create Zep-informed critic
    self.memory_critic = ZepMemoryInformedCritic(
        zep_manager=self.zep_memory,
        spatial_bridge=self.spatial_bridge,
        base_critic=self.critic,
        config=self.config,
        logger=self.logger
    )

def _run_turn(self, game_interface: GameServerClient, current_state: str) -> Tuple[str, str]:
    """Enhanced turn processing with Zep memory storage."""
    
    # ... existing turn logic ...
    
    # Store turn experience in Zep
    self.zep_memory.store_turn_experience(
        action=action,
        response=response,
        analysis={
            "success": self.extractor.last_extraction.get("success", False),
            "moved": self.extractor.last_extraction.get("moved", False),
            "score_change": score_delta,
            "new_items": self.extractor.last_extraction.get("new_items", []),
            "combat": self.extractor.last_extraction.get("combat", False)
        }
    )
    
    # Update spatial bridge if moved
    if self.extractor.last_extraction.get("moved"):
        self.spatial_bridge.add_room_connection(
            from_room=previous_location,
            to_room=self.game_state.current_location,
            direction=action.split()[0].lower(),
            confidence=0.9
        )
    
    # ... rest of turn processing ...
```

## Configuration

### pyproject.toml Updates

```toml
[tool.zorkgpt.memory]
# Memory system selection
provider = "zep"  # Options: "zep", "mem0", "legacy"

[tool.zorkgpt.memory.zep]
# Zep Cloud configuration
api_key = "${ZEP_API_KEY}"
base_url = "https://api.getzep.com"  # Optional, for self-hosted

# Memory settings
session_prefix = "zorkgpt"
fact_confidence_threshold = 0.7
max_messages_per_turn = 2
enable_auto_facts = true

# Token optimization
use_perpetual_memory = true
memory_window = 50  # Recent messages to keep

[tool.zorkgpt.memory.spatial]
# Local spatial database (SQLite)
enabled = true
db_path = "./spatial_memory.db"
cache_paths = true
max_path_length = 15

[tool.zorkgpt.critic]
# Memory-informed critic settings
memory_confidence_threshold = 0.85
pattern_confidence_threshold = 0.7
estimated_token_savings = 15000  # Per bypass
```

## Implementation Phases

### Phase 1: Core Zep Integration (Days 1-2)
- [ ] Set up Zep Cloud account and API keys
- [ ] Implement ZepMemoryManager class
- [ ] Create basic store/retrieve operations
- [ ] Test fact extraction capabilities

### Phase 2: Spatial Bridge (Day 3)
- [ ] Implement SpatialMemoryBridge with SQLite
- [ ] Create pathfinding algorithms
- [ ] Integrate with MapManager
- [ ] Test navigation queries

### Phase 3: Memory-Informed Critic (Days 4-5)
- [ ] Implement ZepMemoryInformedCritic
- [ ] Integrate with orchestrator
- [ ] Test token reduction metrics
- [ ] Validate bypass accuracy

### Phase 4: Advanced Features (Days 6-7)
- [ ] Implement cross-episode learning
- [ ] Add pattern synthesis from facts
- [ ] Create memory visualization dashboard
- [ ] Performance optimization

### Phase 5: Testing & Refinement (Days 8-9)
- [ ] End-to-end testing with full episodes
- [ ] A/B testing vs current system
- [ ] Performance benchmarking
- [ ] Documentation and examples

## Migration Strategy

### Gradual Rollout
1. **Parallel Operation**: Run Zep alongside existing system for 5 episodes
2. **Validation**: Compare memory retrieval quality and token usage
3. **Cutover**: Switch primary system to Zep after validation
4. **Fallback**: Keep legacy system available via config flag

### Data Migration
```python
# scripts/migrate_to_zep.py
def migrate_existing_knowledge():
    """Migrate knowledgebase.md and persistent_wisdom.md to Zep."""
    
    # Read existing knowledge files
    with open("knowledgebase.md", "r") as f:
        knowledge = f.read()
    
    with open("persistent_wisdom.md", "r") as f:
        wisdom = f.read()
    
    # Convert to Zep facts
    zep = Zep(api_key=config.zep_api_key)
    session = zep.memory.add_session(
        session_id="zorkgpt_migration",
        metadata={"type": "knowledge_import"}
    )
    
    # Parse and add as messages/facts
    for section in parse_knowledge_sections(knowledge):
        zep.memory.add(
            session_id=session.session_id,
            messages=[
                Message(
                    role="System",
                    role_type="system",
                    content=section.content,
                    metadata={"source": "knowledgebase", "section": section.title}
                )
            ]
        )
```

## Performance Expectations

### Token Usage
- **Current System**: 15,000-20,000 tokens per turn
- **Zep System**: 300-500 tokens per turn (98% reduction)
- **Critic Bypasses**: 85-90% using Zep memories

### Latency
- **Memory Retrieval**: <50ms (Zep Cloud)
- **Spatial Queries**: <1ms (local SQLite)
- **Fact Extraction**: Automatic, async

### Accuracy
- **Memory Recall**: 94.8% relevance (Zep benchmark)
- **Pattern Detection**: Automatic via facts
- **Critic Decisions**: 18.5% improvement in accuracy

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Zep Cloud downtime | Local SQLite fallback for critical data |
| API rate limits | Batch operations, local caching |
| Fact extraction quality | Confidence thresholds, manual validation |
| Integration complexity | Phased rollout, extensive testing |
| Cost concerns | Monitor API usage, optimize calls |

## Success Metrics

1. **Token Reduction**: Achieve >95% reduction in tokens per turn
2. **Critic Efficiency**: >80% decisions from memory (no LLM)
3. **Learning Quality**: Automatic extraction of >50 facts per episode
4. **Performance**: <100ms total latency for memory operations
5. **Cost Savings**: >70% reduction in API costs

## Future Enhancements

### Phase 6+: Advanced Capabilities
- **Multi-Agent Learning**: Share facts across different agent instances
- **Meta-Learning**: Analyze patterns across all episodes
- **Predictive Actions**: Suggest next actions based on facts
- **Real-time Adaptation**: Update strategies mid-episode based on facts

## Conclusion

Zep provides ZorkGPT with state-of-the-art temporal memory capabilities, dramatic token efficiency, and automatic learning extraction. The hybrid approach with SQLite for spatial data combines Zep's strengths in temporal tracking with the spatial requirements specific to Zork navigation. This integration will significantly reduce costs while improving agent performance through better memory utilization.

The phased implementation approach ensures low risk while the parallel testing strategy validates improvements before full deployment. With Zep's proven 98% token reduction and 94.8% accuracy, this integration represents a major advancement for ZorkGPT's cognitive capabilities.