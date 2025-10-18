# Spatial-Semantic Memory Refactor Specification

## Executive Summary

This specification proposes a unified memory system for ZorkGPT that combines spatial (map/location) knowledge with semantic (game mechanics/strategies) knowledge in a single, token-efficient system. The solution uses SQLite with local embeddings to provide fast, contextual memory retrieval while reducing token usage by ~95%.

**Key Insight**: The current system already has sophisticated adaptive learning via AdaptiveKnowledgeManager (updates knowledgebase.md every 100 turns using LLM analysis). The problem isn't lack of learning - it's the lack of spatial correlation and inefficient retrieval.

## Problem Statement

### Current Issues

1. **Fragmented Knowledge Systems**
   - Game knowledge in `knowledgebase.md` (LLM-updated every 100 turns via AdaptiveKnowledgeManager)
   - Death patterns in `persistent_wisdom.md` (episode-end synthesis)
   - Spatial data in `MapGraph` class (in-memory)
   - **No connection between "what happened" and "where it happened"**

2. **Token Explosion**
   - Current system sends 15-20K tokens per turn:
     - Full knowledgebase.md (~7KB, grows over time)
     - Relevant persistent_wisdom.md (~4KB)
     - Complete map state
     - Recent action history
   - High API costs and latency
   - Context window limitations
   - **Knowledge files sent in full regardless of relevance**

3. **Lack of Spatial-Semantic Correlation**
   - Cannot answer: "What dangers are near my current location?"
   - Cannot retrieve: "What items are 2 rooms away?"
   - No way to link "use lantern in dark" with "dark cellar is east"
   - **AdaptiveKnowledgeManager updates lack location context**

4. **Inefficient Knowledge Retrieval**
   - Entire knowledge base sent every turn
   - No query-based retrieval
   - Cannot filter knowledge by location or relevance
   - LLM must process all knowledge even when only 5% is relevant

### What's Working Well

- AdaptiveKnowledgeManager provides sophisticated LLM-driven knowledge synthesis
- Knowledge base evolves during gameplay (every 100 turns)
- MapGraph tracks spatial relationships with confidence scoring
- Manager-based architecture is clean and modular

## Goals

### Primary Objectives

1. **Unified Memory System**
   - Single storage for all game knowledge
   - Automatic linking of experiences to locations
   - Queryable relationships between spatial and semantic data
   - **Preserve AdaptiveKnowledgeManager's learning capabilities**

2. **Token Efficiency**
   - Reduce context from 15K to <1K tokens per turn
   - Smart retrieval of only relevant memories
   - Compressed, actionable context formatting
   - Location-aware filtering

3. **Spatial Awareness**
   - Associate memories with locations
   - Track what happened where
   - Understand spatial relationships in queries
   - Enhance AdaptiveKnowledgeManager with location data

4. **Performance**
   - Sub-millisecond query times
   - No external dependencies
   - Run entirely in-process
   - Compatible with existing learning systems

## Proposed Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Agent Decision Layer                │
└────────────┬──────────────────────┬─────────────────┘
             │                      │
    ┌────────▼────────┐    ┌───────▼────────┐
    │ Context Manager │    │  Map Manager   │
    └────────┬────────┘    └───────┬────────┘
             │                      │
    ┌────────▼──────────────────────▼────────┐
    │     Spatial-Semantic Memory (SQLite)   │
    │                                         │
    │  - Memory Table (experiences + embed)  │
    │  - Spatial Links (room connections)    │
    │  - Knowledge Patterns (from AKM)       │
    │  - FTS5 Index (full-text search)       │
    │  - FAISS Index (vector similarity)     │
    └──────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │ AdaptiveKnowledge  │
         │     Manager        │
         │ (Continues to run) │
         └────────────────────┘
```

### Integration Strategy

**Key Principle**: Enhance, don't replace. The AdaptiveKnowledgeManager continues to perform its sophisticated analysis, but now:
1. Stores outputs in SQLite with location tags
2. Knowledge retrieval becomes query-based
3. Context assembly uses spatial filtering

### Core Components

#### 1. SpatialSemanticMemory Class

```python
class SpatialSemanticMemory:
    """
    Unified memory system combining spatial and semantic knowledge.
    Works alongside AdaptiveKnowledgeManager, not replacing it.
    
    Features:
    - SQLite for structured queries and relationships
    - Sentence-transformers for local embeddings (no API calls)
    - FAISS for fast vector similarity search
    - Full-text search for keyword matching
    - Spatial graph queries for navigation context
    - Integration with AdaptiveKnowledgeManager outputs
    """
```

#### 2. Database Schema

```sql
-- Core memory storage
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    content TEXT,           -- What happened
    location TEXT,          -- Where it happened
    importance REAL,        -- 0.0 to 1.0 score
    episode_id TEXT,        -- Episode identifier
    turn INTEGER,           -- Turn number
    tags TEXT,              -- JSON array of tags
    embedding BLOB,         -- Vector embedding (384 dims)
    source TEXT,            -- 'experience' | 'knowledge_update' | 'wisdom'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge patterns from AdaptiveKnowledgeManager
CREATE TABLE knowledge_patterns (
    id INTEGER PRIMARY KEY,
    pattern TEXT,           -- The pattern/strategy
    locations TEXT,         -- JSON array of relevant locations
    confidence REAL,        -- Pattern confidence
    last_updated INTEGER,   -- Turn when last updated
    source_turns TEXT,      -- JSON array of source turn numbers
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Spatial relationships
CREATE TABLE spatial_links (
    from_location TEXT,
    to_location TEXT,
    direction TEXT,         -- north, south, etc.
    distance INTEGER,       -- Rooms away
    object_found TEXT,      -- Items/NPCs at destination
    confidence REAL,        -- Link confidence
    PRIMARY KEY (from_location, to_location)
);

-- Full-text search index
CREATE VIRTUAL TABLE memory_fts 
USING fts5(content, location, tags);
```

#### 3. Enhanced AdaptiveKnowledgeManager Integration

```python
class EnhancedKnowledgeManager(KnowledgeManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_memory = SpatialSemanticMemory()
    
    def check_periodic_update(self, current_agent_reasoning: str = "") -> None:
        """Enhanced periodic update that stores to spatial memory."""
        
        # Original knowledge update
        super().check_periodic_update(current_agent_reasoning)
        
        # Parse the updated knowledge base and store with location tags
        if self.last_knowledge_update_turn == self.game_state.turn_count:
            self._store_knowledge_patterns_to_spatial_memory()
    
    def _store_knowledge_patterns_to_spatial_memory(self):
        """Parse knowledgebase.md and store patterns with spatial context."""
        
        # Read the newly updated knowledge base
        with open("knowledgebase.md", "r") as f:
            content = f.read()
        
        # Extract patterns and their associated locations from recent turns
        patterns = self._extract_patterns_with_locations(content)
        
        for pattern in patterns:
            # Store in spatial memory with location context
            self.spatial_memory.store_knowledge_pattern(
                pattern=pattern['content'],
                locations=pattern.get('locations', []),
                confidence=pattern.get('confidence', 0.7),
                turn=self.game_state.turn_count
            )
```

#### 4. Token-Efficient Context Retrieval

```python
def build_context(self):
    """Build minimal, relevant context (<1K tokens)."""
    
    current_loc = self.game_state.current_location
    last_action = self.game_state.last_action
    
    # Get ONLY location-relevant memories (400 tokens)
    memory_context = self.spatial_memory.recall_for_location(
        current_loc, 
        last_action,
        radius=2  # Include memories from nearby rooms
    )
    
    # Get ONLY relevant patterns from knowledge base (300 tokens)
    knowledge_patterns = self.spatial_memory.get_relevant_patterns(
        location=current_loc,
        action_context=last_action,
        limit=3
    )
    
    # Get ONLY local map (200 tokens)
    map_context = self._get_minimal_map(current_loc, radius=2)
    
    # Get ONLY critical wisdom (100 tokens)
    wisdom = self.spatial_memory.get_critical_wisdom(
        location=current_loc,
        limit=2
    )
    
    # Total: ~1000 tokens instead of 15,000+
    return {
        'memories': memory_context,
        'knowledge': knowledge_patterns,
        'map': map_context,
        'wisdom': wisdom
    }
```

### Retrieval Strategy

The system uses a multi-tiered retrieval approach:

1. **Location-specific memories** (highest priority)
   - Direct experiences at current location
   - Warnings and dangers for this room
   - Patterns discovered here

2. **Spatial context** (nearby locations)
   - Memories from adjacent rooms
   - Objects and paths within 2-3 moves
   - Navigation warnings

3. **Semantic similarity** (action-relevant)
   - Vector search for similar situations
   - Pattern matching for current action
   - Knowledge base patterns matching context

4. **Critical wisdom** (universal truths)
   - High-importance memories (>0.9)
   - Death patterns from persistent_wisdom.md
   - Proven strategies

## Implementation Plan

### Phase 1: Core Infrastructure (Days 1-2)
- [ ] Create `spatial_semantic_memory.py`
- [ ] Set up SQLite schema
- [ ] Implement basic store/retrieve operations
- [ ] Add sentence-transformers integration
- [ ] Set up FAISS indexing

### Phase 2: Integration with Existing Systems (Days 3-4)
- [ ] Create `EnhancedKnowledgeManager` that extends current KnowledgeManager
- [ ] Hook into AdaptiveKnowledgeManager's update cycle
- [ ] Store knowledge patterns with location tags
- [ ] Modify `ContextManager.build_context()` for efficient retrieval

### Phase 3: Migration and Compatibility (Day 5)
- [ ] Import existing knowledge files into SQLite
- [ ] Ensure AdaptiveKnowledgeManager continues to work unchanged
- [ ] Add fallback to file-based system if needed
- [ ] Create migration scripts for existing episodes

### Phase 4: Optimization (Days 6-7)
- [ ] Token counting and dynamic budgets
- [ ] Query performance tuning
- [ ] Context compression algorithms
- [ ] Caching layer for frequent queries

## Expected Outcomes

### Performance Metrics

| Metric | Current System | New System | Improvement |
|--------|---------------|------------|-------------|
| Tokens per turn | 15,000-20,000 | 800-1,000 | 95% reduction |
| Query latency | 10-50ms | <1ms | 10-50x faster |
| Memory types | 3 separate | 1 unified | 67% simpler |
| Spatial awareness | None | Full | ∞ |
| Knowledge updates | Every 100 turns | Every 100 turns + spatial | Enhanced |

### Functional Improvements

1. **Contextual Intelligence**
   - "The troll is 2 rooms east, get the sword first"
   - "Last knowledge update: lantern needed in dark areas near here"
   - "Pattern discovered at turn 450: this room has hidden passages"

2. **Preserved Learning Capabilities**
   - AdaptiveKnowledgeManager continues unchanged
   - Knowledge synthesis still happens every 100 turns
   - Now enhanced with spatial correlation

3. **Developer Benefits**
   - Minimal changes to existing code
   - AdaptiveKnowledgeManager untouched
   - SQL queries for debugging
   - Gradual migration path

## Potential Issues and Mitigations

### Risk: Integration with AdaptiveKnowledgeManager
- **Issue**: Parsing LLM-generated knowledge for location tags
- **Mitigation**: 
  - Use regex patterns to identify location references
  - Fall back to general storage if location unclear
  - Log unparseable patterns for analysis

### Risk: Embedding Model Size
- **Issue**: sentence-transformers models can be 100-500MB
- **Mitigation**: Use `all-MiniLM-L6-v2` (80MB), cache on first run
- **Alternative**: Start with keyword search only, add embeddings later

### Risk: SQLite Performance at Scale
- **Issue**: May slow down after 1M+ memories
- **Mitigation**: 
  - Implement importance-based pruning
  - Archive old episodes to separate databases
  - Use WAL mode for better concurrency

### Risk: Knowledge Update Timing
- **Issue**: Spatial memory might be out of sync with knowledge updates
- **Mitigation**:
  - Hook directly into AdaptiveKnowledgeManager's update cycle
  - Store update timestamps
  - Refresh spatial indices after each update

## Testing Strategy

### Unit Tests
```python
def test_knowledge_manager_integration():
    """Ensure AdaptiveKnowledgeManager still works unchanged."""
    
def test_spatial_retrieval():
    """Test location-based memory retrieval."""
    
def test_token_budget_compliance():
    """Ensure context stays within 1K tokens."""
```

### Integration Tests
- Knowledge updates trigger spatial memory storage
- Context retrieval during live gameplay
- Fallback to file-based system works
- Performance under 1000-turn episodes

### Backward Compatibility Tests
- Existing episodes can still run
- AdaptiveKnowledgeManager produces same outputs
- Knowledge files still generated for debugging

## Success Criteria

1. **Token Reduction**: Achieve <1,000 tokens per turn average
2. **Performance**: All queries complete in <10ms
3. **Compatibility**: AdaptiveKnowledgeManager works unchanged
4. **Spatial Correlation**: 80%+ of knowledge linked to locations
5. **Zero Regressions**: Existing functionality preserved

## Conclusion

This refactor enhances the existing sophisticated learning system with spatial awareness and efficient retrieval, without disrupting the proven AdaptiveKnowledgeManager. By adding a SQLite layer for spatial-semantic correlation and query-based retrieval, we achieve dramatic token reduction while preserving and enhancing the current adaptive learning capabilities.

The key insight is that we don't need to replace the existing learning system - we need to make it spatially aware and more efficient in how it delivers knowledge to the agent. This approach minimizes risk while maximizing value delivery.