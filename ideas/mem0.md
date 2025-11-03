# Mem0 Integration Specification for ZorkGPT

## Executive Summary

This document specifies the integration of Mem0, a universal memory layer for AI agents, into the ZorkGPT system. Mem0 will unify and replace the current two-tier knowledge system (knowledgebase.md and persistent_wisdom.md) with a single, more efficient semantic memory system using vector similarity search and graph-based spatial relationships. The system will also absorb and enhance the current MapGraph functionality, creating a unified knowledge graph that combines episodic memories with spatial navigation data.

## Architecture Overview

### Unified Memory & Spatial Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Decision Layer                  │
└─────────────┬──────────────────────────┬────────────────┘
              │                          │
    ┌─────────▼────────┐       ┌────────▼────────┐
    │  Working Memory  │       │  Context Query   │
    │  (Last 20 turns) │       │    Interface     │
    └─────────┬────────┘       └────────┬────────┘
              │                          │
    ┌─────────▼──────────────────────────▼────────┐
    │     Unified Knowledge Graph (Mem0 + Neo4j)  │
    │                                              │
    │  MEMORY NODES                                │
    │  ├─ Experience (importance: 0.1-0.6)        │
    │  ├─ Pattern (importance: 0.6-0.8)           │
    │  └─ Wisdom (importance: 0.8-1.0)            │
    │                                              │
    │  SPATIAL NODES                               │
    │  ├─ Room (name, base_name, visits)          │
    │  ├─ Item (name, properties)                 │
    │  └─ Entity (NPCs, objects)                  │
    │                                              │
    │  RELATIONSHIPS                               │
    │  ├─ CONNECTS_TO (rooms → rooms)             │
    │  ├─ OCCURRED_AT (memories → rooms)          │
    │  ├─ INVOLVES (memories → entities)          │
    │  └─ LOCATED_IN (items → rooms)              │
    └───────────────────────────────────────────────┘
```

### Memory Types and Migration

1. **Working Memory** (Unchanged)
   - Scope: Current episode, last 20-50 turns
   - Storage: In-memory list in GameState
   - Purpose: Immediate context for agent decisions

2. **Unified Memory (Mem0)** (Replaces knowledgebase.md and persistent_wisdom.md)
   - **Experience Memories** (importance: 0.1-0.6)
     - Raw turn-by-turn experiences
     - Location-specific interactions
     - Failed/successful action records
   
   - **Pattern Memories** (importance: 0.6-0.8)
     - Migrated from knowledgebase.md patterns
     - Synthesized strategies from multiple experiences
     - Objective completion patterns
   
   - **Wisdom Memories** (importance: 0.8-1.0)
     - Migrated from persistent_wisdom.md
     - Death avoidance patterns
     - Universal game mechanics
     - Verified safe/dangerous actions

## Graph Database Schema

### Node Types

```cypher
// Memory Nodes
(:Memory {
    id: String,           // Unique memory ID
    content: String,      // Memory content
    importance: Float,    // 0.0 to 1.0
    episode_id: String,   // Episode where created
    timestamp: DateTime,  // When created
    turn: Integer,        // Turn number
    type: String,         // 'experience' | 'pattern' | 'wisdom'
    tags: [String],       // Categorization tags
    confidence: Float     // Confidence score
})

// Spatial Nodes  
(:Room {
    name: String,         // Unique room identifier
    base_name: String,    // Canonical name without suffixes
    description: String,  // Room description
    visits: Integer,      // Visit count
    last_visit: DateTime, // Last visit timestamp
    exits: [String]       // Available exits
})

(:Item {
    name: String,         // Item name
    description: String,  // Item description
    takeable: Boolean,    // Can be picked up
    properties: Map       // Additional properties
})

(:Entity {
    name: String,         // Entity name (NPC, object)
    type: String,         // 'npc' | 'object' | 'puzzle'
    properties: Map       // Entity-specific data
})
```

### Relationship Types

```cypher
// Spatial Relationships
(:Room)-[:CONNECTS_TO {
    direction: String,    // 'north', 'south', etc.
    confidence: Float,    // Connection confidence
    verified_count: Int,  // Times verified
    last_verified: DateTime,
    bidirectional: Boolean
}]->(:Room)

// Memory-Location Relationships
(:Memory)-[:OCCURRED_AT {
    primary: Boolean      // Primary location for this memory
}]->(:Room)

// Memory-Entity Relationships
(:Memory)-[:INVOLVES {
    role: String          // 'subject' | 'object' | 'tool'
}]->(:Entity)

// Item Location
(:Item)-[:LOCATED_IN {
    since: DateTime,      // When item was placed
    container: String     // Optional container name
}]->(:Room)

// Failed Connections
(:Room)-[:FAILED_EXIT {
    direction: String,    // Attempted direction
    attempts: Integer,    // Number of attempts
    last_attempt: DateTime
}]->(:Room)
```

## Integration Points

### 1. Unified Memory-Map Manager (Replaces MemoryManager and MapManager)

Create a new `MemoryManager` class that extends `BaseManager`:

```python
# managers/unified_memory_manager.py
class UnifiedMemoryManager(BaseManager):
    """
    Manages Mem0 integration for both episodic memory and spatial mapping.
    
    Responsibilities:
    - Store experiences with rich metadata and spatial context
    - Manage room graph and navigation paths
    - Retrieve memories based on semantic and spatial queries
    - Handle both memory and map persistence through Neo4j
    """
    
    def __init__(self, logger, config, game_state):
        super().__init__(logger, config, game_state, "unified_memory_manager")
        
        # Initialize Mem0 with graph support
        self.mem0_config = {
            "graph_store": {
                "provider": config.graph_provider,  # 'neo4j' or 'kuzu'
                "config": config.graph_config
            },
            "vector_store": {
                "provider": config.vector_provider,
                "config": config.vector_config
            }
        }
        self.mem0 = Memory(self.mem0_config)
        
        # Track current map state locally for performance
        self.current_room = None
        self.room_cache = {}  # Cache frequent room lookups
        
    # Memory Operations
    def store_experience(self, experience: ExperienceData) -> str:
        """Store an experience with spatial and semantic metadata."""
        memory_id = self.mem0.add(
            experience.to_mem0_format(),
            user_id=self.game_state.episode_id
        )
        
        # Link to spatial context
        if experience.location:
            self._link_memory_to_room(memory_id, experience.location)
        
        return memory_id
    
    # Spatial Operations (replacing MapGraph)
    def add_room_connection(self, from_room: str, direction: str, to_room: str, confidence: float = 0.8):
        """Add or update a connection between rooms in the graph."""
        cypher = """
        MERGE (from:Room {name: $from_room})
        MERGE (to:Room {name: $to_room})
        MERGE (from)-[r:CONNECTS_TO {direction: $direction}]->(to)
        SET r.confidence = $confidence,
            r.last_verified = timestamp(),
            r.verified_count = coalesce(r.verified_count, 0) + 1
        """
        self._execute_graph_query(cypher, {
            'from_room': from_room,
            'to_room': to_room,
            'direction': direction,
            'confidence': confidence
        })
        
    def find_navigation_path(self, from_room: str, to_room: str) -> List[str]:
        """Find shortest path between rooms using graph algorithms."""
        cypher = """
        MATCH path = shortestPath(
            (from:Room {name: $from_room})-[:CONNECTS_TO*]-(to:Room {name: $to_room})
        )
        RETURN [r in relationships(path) | r.direction] as directions
        """
        result = self._execute_graph_query(cypher, {
            'from_room': from_room,
            'to_room': to_room
        })
        return result.get('directions', [])
    
    # Hybrid Queries
    def recall_by_location(self, location: str, limit: int = 10) -> List[Memory]:
        """Retrieve memories that occurred at a specific location."""
        cypher = """
        MATCH (m:Memory)-[:OCCURRED_AT]->(r:Room {name: $location})
        RETURN m.id, m.content, m.importance, m.timestamp
        ORDER BY m.importance DESC, m.timestamp DESC
        LIMIT $limit
        """
        results = self._execute_graph_query(cypher, {
            'location': location,
            'limit': limit
        })
        
        # Enrich with vector similarity if needed
        return self._enrich_graph_memories(results)
    
    def get_dangerous_paths(self) -> List[Dict]:
        """Find paths where negative experiences occurred."""
        cypher = """
        MATCH (m:Memory {type: 'death'})-[:OCCURRED_AT]->(r:Room)
        MATCH (r)-[c:CONNECTS_TO]->(:Room)
        RETURN r.name as room, c.direction as exit, 
               count(m) as death_count
        ORDER BY death_count DESC
        """
        return self._execute_graph_query(cypher, {})
```

### 2. Experience Data Model

```python
# models/experience.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ExperienceData:
    """Represents a single experience to be stored in memory."""
    
    # Core data
    action: str
    response: str
    location: str
    turn: int
    episode_id: str
    timestamp: str
    
    # Context
    inventory: List[str]
    score: int
    score_delta: int
    
    # Outcomes
    success: bool
    moved: bool
    combat: bool
    puzzle_interaction: bool
    item_interaction: bool
    
    # Analysis
    importance: float  # 0.0 to 1.0
    novelty: float    # 0.0 to 1.0
    tags: List[str]   # ["combat", "puzzle", "exploration", etc.]
    
    def to_mem0_format(self) -> Dict[str, Any]:
        """Convert to Mem0 storage format."""
        return {
            "content": f"At {self.location}: {self.action} -> {self.response[:200]}",
            "metadata": {
                "location": self.location,
                "turn": self.turn,
                "episode_id": self.episode_id,
                "score": self.score,
                "success": self.success,
                "tags": self.tags,
                "importance": self.importance
            }
        }
```

### 3. Integration Changes for Unified Memory

#### Agent Prompt Enhancement (Replaces knowledgebase.md loading)

```python
# In zork_agent.py - replace _enhance_prompt_with_knowledge()
def _enhance_prompt_with_memories(self, base_prompt: str, memory_manager) -> str:
    """Enhance agent prompt with relevant memories from Mem0."""
    
    # Get high-importance wisdom memories (replacing persistent_wisdom.md)
    wisdom_memories = memory_manager.recall_by_importance(
        min_importance=0.8,
        limit=10
    )
    
    # Format memories for prompt
    memory_section = """
**STRATEGIC MEMORIES FROM PREVIOUS EPISODES:**

FUNDAMENTAL WISDOM (Importance 0.8-1.0):
{wisdom_content}

STRATEGIC PATTERNS (Importance 0.6-0.8):
{pattern_content}

**END OF MEMORIES**
"""
    
    # Insert before Output Format section
    return self._insert_section(base_prompt, memory_section, "**Output Format")
```

#### Knowledge Manager Becomes Memory Synthesis Manager

```python
# managers/memory_synthesis_manager.py (replaces knowledge_manager.py)
class MemorySynthesisManager(BaseManager):
    """
    Manages memory importance scoring and tier promotion.
    Replaces AdaptiveKnowledgeManager and knowledge file management.
    """
    
    def process_turn(self):
        # Store current turn as experience memory
        experience = ExperienceData(
            action=self.game_state.last_action,
            response=self.game_state.last_response,
            location=self.game_state.current_location,
            importance=self._calculate_importance(),
            # ... other metadata
        )
        
        memory_id = self.memory_manager.store_experience(experience)
        
        # Check if we should synthesize patterns
        if self.should_synthesize():
            self.synthesize_patterns()
    
    def synthesize_patterns(self):
        # Get recent low-importance memories
        experiences = self.memory_manager.get_memories(
            importance_range=(0.1, 0.6),
            limit=100
        )
        
        # Use LLM to identify patterns
        patterns = self._identify_patterns_with_llm(experiences)
        
        # Store as pattern-tier memories
        for pattern in patterns:
            self.memory_manager.store_memory(
                content=pattern.description,
                importance=0.7,  # Pattern tier
                metadata={
                    "type": "pattern",
                    "source_experiences": pattern.source_ids,
                    "confidence": pattern.confidence
                }
            )
    
    def promote_validated_patterns(self):
        # Find patterns that have proven reliable
        patterns = self.memory_manager.get_memories(
            importance_range=(0.6, 0.8),
            metadata_filter={"confidence": {"$gte": 0.9}}
        )
        
        # Promote to wisdom tier
        for pattern in patterns:
            if self._validate_pattern(pattern):
                self.memory_manager.update_importance(
                    memory_id=pattern.id,
                    new_importance=0.9
                )
```

## Configuration

Add to `pyproject.toml`:

```toml
[tool.zorkgpt.memory]
# Mem0 configuration
enabled = true
recall_limit = 5
importance_threshold = 0.3

# Vector database settings
[tool.zorkgpt.memory.vector_db]
provider = "qdrant"  # Options: qdrant, chroma, pinecone, weaviate
collection_name = "zork_experiences"
embedding_model = "text-embedding-ada-002"
dimension = 1536

# Graph database (REQUIRED for unified memory-map system)
[tool.zorkgpt.memory.graph_db]
enabled = true  # Must be true for spatial mapping
provider = "neo4j"  # Options: neo4j, kuzu (local), memgraph, neptune
uri = "bolt://localhost:7687"
username = "neo4j"
password = "your-password"

# Graph-specific settings
[tool.zorkgpt.memory.graph_settings]
enable_spatial_index = true  # Index rooms for fast lookup
enable_memory_linking = true  # Link memories to locations
cache_navigation_paths = true  # Cache common navigation queries
max_path_length = 20  # Max rooms in navigation path

# Memory management
[tool.zorkgpt.memory.management]
max_memories_per_episode = 10000
ttl_days = 0  # 0 = no expiration
deduplication = true
compression_threshold = 100  # Compress similar memories after N occurrences
```

## Implementation Phases

### Phase 1: Basic Memory Integration (Week 1)
- [x] Install Mem0: `pip install mem0ai`
- [ ] Set up Neo4j database instance
- [ ] Create UnifiedMemoryManager class
- [ ] Implement basic memory store/recall operations
- [ ] Add memory storage to turn processing
- [ ] Test with single episode (memory only, no map)

### Phase 2: Graph Database Setup (Week 2)
- [ ] Configure Neo4j connection and authentication
- [ ] Create graph schema (nodes and relationships)
- [ ] Implement graph query wrapper methods
- [ ] Add spatial node creation for rooms
- [ ] Test basic room creation and connection
- [ ] Create migration script for existing map data

### Phase 3: Unified Memory-Map Integration (Week 3)
- [ ] Implement room connection management
- [ ] Add memory-to-location linking
- [ ] Create navigation path finding
- [ ] Implement hybrid queries (memory + spatial)
- [ ] Add caching layer for frequent queries
- [ ] Parallel operation with existing MapGraph

### Phase 4: Map Migration & Enhancement (Week 4)
- [ ] Migrate MapGraph consolidation logic to Cypher
- [ ] Implement confidence scoring in graph
- [ ] Add failed exit tracking
- [ ] Create map visualization from graph
- [ ] Performance testing and optimization
- [ ] Gradual cutover from MapGraph to graph DB

### Phase 5: Advanced Features (Week 5)
- [ ] Pattern detection using graph algorithms
- [ ] Memory compression for repeated experiences
- [ ] Cross-episode spatial analysis
- [ ] Dangerous path identification
- [ ] Item and entity tracking
- [ ] Integration with knowledge synthesis

### Phase 6: Complete Migration (Week 6)
- [ ] Remove dependency on MapGraph class
- [ ] Remove knowledge base file dependencies
- [ ] Update all managers to use unified system
- [ ] Final performance tuning
- [ ] Documentation and monitoring dashboard

## API Examples

### Storing an Experience with Spatial Context

```python
# In orchestrator during turn processing
experience = ExperienceData(
    action="attack troll with sword",
    response="The troll grabs your sword and breaks it!",
    location="Bridge",
    turn=45,
    episode_id=self.game_state.episode_id,
    timestamp=datetime.now().isoformat(),
    inventory=["lamp", "rope"],
    score=25,
    score_delta=0,
    success=False,
    moved=False,
    combat=True,
    puzzle_interaction=False,
    item_interaction=True,
    importance=0.8,  # High importance: lost weapon
    novelty=0.9,     # First time this happened
    tags=["combat", "item_loss", "troll", "bridge"]
)

# Store memory and automatically link to Bridge room node
memory_id = self.unified_memory_manager.store_experience(experience)

# Also update map if we discovered new connections
if moved_to_new_room:
    self.unified_memory_manager.add_room_connection(
        from_room="Troll Room",
        direction="north",
        to_room="Bridge",
        confidence=0.9
    )
```

### Retrieving Relevant Memories with Spatial Context

```python
# In agent context building
# Semantic search (unchanged, but now includes spatial context)
memories = unified_memory_manager.recall_similar(
    "How do I defeat the troll at the bridge?",
    limit=5
)

# Location-based search with graph traversal
bridge_memories = unified_memory_manager.recall_by_location("Bridge", limit=10)

# Find navigation path using graph
path = unified_memory_manager.find_navigation_path(
    from_room="West of House",
    to_room="Bridge"
)
# Returns: ["north", "east", "north", "up"]

# Get dangerous paths to avoid
dangerous_exits = unified_memory_manager.get_dangerous_paths()
# Returns: [{"room": "Bridge", "exit": "jump", "death_count": 3}, ...]

# Hybrid query: memories near current location
nearby_memories = unified_memory_manager.get_nearby_memories(
    current_room="Troll Room",
    max_distance=2,  # Within 2 rooms
    importance_min=0.7
)

# Spatial pattern detection
troll_locations = unified_memory_manager.find_entity_locations("troll")
# Returns all rooms where troll encounters occurred
```

### Memory-Enhanced Agent Prompt

```python
# Context provided to agent
context = {
    "current_situation": "You are at the Bridge. A troll blocks your path.",
    "episodic_memories": [
        {
            "memory": "Previously at Bridge: Showing the lamp made the troll nervous",
            "episode": "2024-01-15T10:30:00",
            "turns_ago": 1523,
            "relevance": 0.92
        },
        {
            "memory": "At Troll Room: Giving treasure satisfied a different troll",
            "episode": "2024-01-14T15:22:00", 
            "turns_ago": 2341,
            "relevance": 0.78
        }
    ],
    "semantic_knowledge": "Trolls are generally hostile but can be bypassed with light or treasure"
}
```

## Performance Considerations

### Memory Storage Limits
- Max 10,000 memories per episode (configurable)
- Automatic compression of similar memories
- Optional TTL for old memories

### Query Performance
- Vector similarity search: ~50ms for 1M memories
- Caching of frequent queries
- Batch processing for multiple recalls

### Memory Importance Scoring
```python
def calculate_importance(experience: ExperienceData) -> float:
    """Calculate importance score for memory retention."""
    importance = 0.0
    
    # Score changes are important
    if abs(experience.score_delta) > 0:
        importance += 0.3 * min(abs(experience.score_delta) / 10, 1.0)
    
    # Novel experiences are important
    importance += 0.3 * experience.novelty
    
    # Combat and puzzle interactions are important
    if experience.combat or experience.puzzle_interaction:
        importance += 0.2
    
    # Failed actions in new locations are important
    if not experience.success and experience.novelty > 0.5:
        importance += 0.2
    
    return min(importance, 1.0)
```

## Success Metrics

### Key Performance Indicators
1. **Memory Recall Precision**: % of recalled memories that influence decisions
2. **Memory Coverage**: % of important events that are stored
3. **Query Latency**: Average time to retrieve relevant memories
4. **Decision Impact**: % improvement in agent performance with memories

### Monitoring Dashboard
```python
{
    "memory_stats": {
        "total_memories": 45234,
        "episodes": 89,
        "avg_memories_per_episode": 508,
        "unique_locations": 67,
        "unique_actions": 234
    },
    "performance": {
        "avg_recall_time_ms": 47,
        "cache_hit_rate": 0.73,
        "memory_usage_mb": 124
    },
    "effectiveness": {
        "memories_used_per_turn": 3.2,
        "decision_influence_rate": 0.41,
        "pattern_detection_rate": 0.67
    }
}
```

## Migration from Current System

### MapGraph to Neo4j Migration Strategy

The migration from the in-memory MapGraph to Neo4j-backed spatial graph will be gradual and non-disruptive:

#### Phase A: Parallel Map Systems (Week 1)
```python
class HybridMapManager:
    """Temporary manager that writes to both systems during migration."""
    
    def __init__(self, map_graph, unified_memory_manager):
        self.legacy_map = map_graph  # Existing MapGraph
        self.unified = unified_memory_manager  # New Neo4j-backed system
        
    def add_connection(self, from_room, direction, to_room, confidence):
        # Write to both systems
        self.legacy_map.add_connection(from_room, direction, to_room, confidence)
        self.unified.add_room_connection(from_room, direction, to_room, confidence)
        
    def get_navigation_path(self, from_room, to_room):
        # Read from legacy, validate with new
        legacy_path = self.legacy_map.find_path(from_room, to_room)
        graph_path = self.unified.find_navigation_path(from_room, to_room)
        
        # Log discrepancies for debugging
        if legacy_path != graph_path:
            self.logger.warning(f"Path mismatch: {legacy_path} vs {graph_path}")
        
        return legacy_path  # Use legacy until validated
```

#### Phase B: Migrate Existing Map Data (Week 2)
```python
def migrate_map_to_graph():
    """One-time migration of existing MapGraph data to Neo4j."""
    
    # Load existing map data
    map_graph = load_existing_map()
    unified = UnifiedMemoryManager(config)
    
    # Migrate rooms
    for room_name, room_obj in map_graph.rooms.items():
        cypher = """
        MERGE (r:Room {name: $name})
        SET r.base_name = $base_name,
            r.exits = $exits,
            r.migrated_at = timestamp()
        """
        unified.execute_graph_query(cypher, {
            'name': room_name,
            'base_name': room_obj.base_name,
            'exits': list(room_obj.exits)
        })
    
    # Migrate connections with confidence scores
    for from_room, connections in map_graph.connections.items():
        for direction, to_room in connections.items():
            confidence_key = (from_room, direction)
            confidence = map_graph.connection_confidence.get(confidence_key, 0.5)
            verifications = map_graph.connection_verifications.get(confidence_key, 1)
            
            cypher = """
            MATCH (from:Room {name: $from_room})
            MATCH (to:Room {name: $to_room})
            MERGE (from)-[r:CONNECTS_TO {direction: $direction}]->(to)
            SET r.confidence = $confidence,
                r.verified_count = $verifications,
                r.migrated = true
            """
            unified.execute_graph_query(cypher, {
                'from_room': from_room,
                'to_room': to_room,
                'direction': direction,
                'confidence': confidence,
                'verifications': verifications
            })
    
    # Migrate failed exits
    for (room, exit), failure_count in map_graph.exit_failure_counts.items():
        cypher = """
        MATCH (r:Room {name: $room})
        MERGE (r)-[f:FAILED_EXIT {direction: $exit}]->(r)
        SET f.attempts = $attempts
        """
        unified.execute_graph_query(cypher, {
            'room': room,
            'exit': exit,
            'attempts': failure_count
        })
```

#### Phase C: Gradual Cutover (Week 3)
- Start with read operations (pathfinding, room lookups)
- Monitor performance and accuracy
- Gradually move write operations
- Maintain rollback capability

### Memory System Migration Strategy

1. **Phase 1: Parallel Operation (Week 1-2)**
   - Deploy Mem0 alongside existing systems
   - Mirror all new experiences to both systems
   - Compare retrieval quality between systems

2. **Phase 2: Knowledge Base Migration (Week 3)**
   ```python
   # scripts/migrate_knowledge_to_mem0.py
   def migrate_knowledge_base():
       """Convert knowledgebase.md to pattern-tier memories."""
       with open("knowledgebase.md", "r") as f:
           content = f.read()
       
       # Parse sections
       sections = parse_knowledge_sections(content)
       
       for section in sections:
           # Determine importance based on section type
           if "critical" in section.title.lower():
               importance = 0.75
           elif "strategy" in section.title.lower():
               importance = 0.7
           else:
               importance = 0.65
           
           memory_manager.store_memory(
               content=section.content,
               importance=importance,
               metadata={
                   "source": "knowledgebase.md",
                   "section": section.title,
                   "migrated_at": datetime.now().isoformat()
               }
           )
   ```

3. **Phase 3: Persistent Wisdom Migration (Week 3)**
   ```python
   def migrate_persistent_wisdom():
       """Convert persistent_wisdom.md to wisdom-tier memories."""
       with open("persistent_wisdom.md", "r") as f:
           content = f.read()
       
       # Parse wisdom entries
       wisdom_entries = parse_wisdom_entries(content)
       
       for entry in wisdom_entries:
           # Wisdom gets high importance
           importance = 0.9 if "death" in entry.lower() else 0.85
           
           memory_manager.store_memory(
               content=entry,
               importance=importance,
               metadata={
                   "source": "persistent_wisdom.md",
                   "type": "wisdom",
                   "migrated_at": datetime.now().isoformat(),
                   "ttl": None  # Permanent
               }
           )
   ```

4. **Phase 4: Cutover (Week 4)**
   - Disable file-based knowledge systems
   - Remove knowledgebase.md and persistent_wisdom.md dependencies
   - Update all managers to use Mem0 exclusively

## Testing Strategy

### Unit Tests
- Memory storage and retrieval
- Similarity search accuracy
- Pattern detection algorithms
- Performance benchmarks

### Integration Tests
- Memory recall during gameplay
- Cross-episode memory access
- Knowledge synthesis with memories
- Save/restore with memories

### A/B Testing
- Run episodes with/without memory
- Compare scores and completion rates
- Measure decision quality improvements

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory bloat | High storage costs | Implement compression and TTL |
| Slow recall | Poor agent performance | Use caching and indexing |
| Irrelevant memories | Confusion in decisions | Tune similarity thresholds |
| Vector DB failure | System downtime | Fallback to knowledge base only |

## Future Enhancements

### Phase 5+ Possibilities
1. **Graph relationships**: Use Neo4j for entity relationships
2. **Emotional tagging**: Track emotional valence of memories
3. **Procedural memory**: Learn action sequences that work
4. **Memory consolidation**: Sleep-like memory reorganization
5. **Adversarial memories**: Learn from other agents' plays

## Key Advantages of Unified Memory-Map System

### Why Unification with Graph Database is Superior

1. **Single Knowledge Graph**
   - Memories and spatial data in one queryable system
   - Rich relationships between experiences and locations
   - No synchronization between separate map and memory systems
   - Unified query interface for all knowledge types

2. **Spatial-Semantic Integration**
   - "What happened in this room?" queries
   - "Where did I encounter trolls?" spatial patterns
   - "Avoid paths where I died" navigation safety
   - Location-aware memory retrieval

3. **Graph Algorithm Power**
   - Shortest path navigation with Dijkstra/A*
   - Centrality analysis for important rooms
   - Community detection for map regions
   - Pattern detection across spatial-temporal dimensions

4. **Dynamic Map Evolution**
   - Confidence scoring on connections
   - Automatic consolidation of duplicate rooms
   - Progressive discovery and verification
   - Failed path tracking and pruning

5. **Efficient Operations**
   - Native graph traversal beats dictionary lookups
   - Indexed spatial queries
   - Cached navigation paths
   - Parallel query execution

6. **Rich Query Capabilities**
   ```cypher
   // Find memories in dangerous areas
   MATCH (m:Memory)-[:OCCURRED_AT]->(r:Room)
   WHERE m.tags CONTAINS 'death'
   RETURN r.name, count(m) as danger_level
   
   // Find items near current location
   MATCH path = (current:Room {name: $current})-[:CONNECTS_TO*1..2]-(r:Room)
   MATCH (i:Item)-[:LOCATED_IN]->(r)
   RETURN i.name, r.name, length(path) as distance
   ```

7. **Better Observability**
   - Visualize memory-location relationships
   - Track navigation efficiency over time
   - Analyze spatial memory distribution
   - Monitor map quality and fragmentation

## Conclusion

The unified Mem0 integration will replace ZorkGPT's current three separate systems (knowledgebase.md, persistent_wisdom.md, and MapGraph) with a single, powerful knowledge graph that combines semantic memory with spatial relationships. By leveraging Neo4j's graph database capabilities, the system will enable rich queries that span both episodic memories and map navigation, creating a truly integrated cognitive architecture for the Zork agent.

This unification eliminates redundancy, improves retrieval performance, and enables sophisticated spatial-semantic reasoning. The graph-based approach allows the agent to not just remember what happened, but understand where it happened and how locations relate to each other. The importance-based tier system (experiences → patterns → wisdom) combined with spatial graph traversal provides a natural knowledge evolution pathway that mirrors how humans build mental maps and associate memories with places.

The phased implementation approach ensures a smooth transition from the current architecture while maintaining system stability and allowing for performance validation at each step. The end result will be a more intelligent, spatially-aware agent that can leverage both its memories and its understanding of the game world's geography to make better decisions.