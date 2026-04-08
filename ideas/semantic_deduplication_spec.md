# Semantic Deduplication System - Design Specification

## 1. Overview

### Purpose
Prevent duplicate memories from accumulating in the memory system by using semantic similarity detection (embeddings) to identify and merge similar content before creating new memory entries.

### Motivation
**Current Problem:**
- `game_files/Memories.md` contains 9 duplicate "trap door bars" memories at Location 193
- Duplicate "window ajar" entries at Location 81
- No pre-creation duplicate detection → memory bloat over time

**Impact:**
- Agent context pollution (duplicate information)
- Wasted LLM synthesis calls
- Degraded memory file readability
- Increased token costs

### Success Criteria
1. Prevent creation of semantically identical memories (>95% similarity)
2. Merge near-duplicates (85-95% similarity) when appropriate
3. Preserve genuine variations in memory content
4. Minimal performance overhead (<100ms per memory operation)
5. Zero false positives on distinct memories

---

## 2. Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleMemoryManager                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  record_action_outcome()                                        │
│       │                                                         │
│       ├─> LLM Memory Synthesis                                 │
│       │       │                                                 │
│       │       └─> MemorySynthesisResponse                      │
│       │                                                         │
│       ├─> [NEW] check_semantic_duplicate()  ◄──────────┐      │
│       │       │                                         │      │
│       │       ├─> get_location_embeddings()            │      │
│       │       │       │                                 │      │
│       │       │       └─> embedding_cache: Dict        │      │
│       │       │                                         │      │
│       │       ├─> calculate_embedding()                │      │
│       │       │       │                                 │      │
│       │       │       └─> EmbeddingProvider ───────────┘      │
│       │       │                                                │
│       │       └─> cosine_similarity() > threshold?            │
│       │               │                                        │
│       │               ├─> Yes: merge/update existing          │
│       │               └─> No: proceed with creation           │
│       │                                                        │
│       └─> add_memory() / update_memory_metadata()             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    EmbeddingProvider (NEW)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Strategy Pattern:                                              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ SentenceTransformerProvider (Primary)                │     │
│  │   - Model: all-MiniLM-L6-v2 (384 dims)               │     │
│  │   - Fast: ~50ms per embedding                         │     │
│  │   - Free: No API costs                                │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ LLMEmbeddingProvider (Fallback)                      │     │
│  │   - Uses existing LLM client                          │     │
│  │   - Model: text-embedding-3-small                     │     │
│  │   - Cost: ~$0.00002 per memory                        │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  Interface:                                                     │
│    - embed(text: str) -> np.ndarray                            │
│    - embed_batch(texts: List[str]) -> List[np.ndarray]        │
│    - get_dimension() -> int                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Normal Flow (No Duplicate):**
```
1. LLM synthesis creates MemorySynthesisResponse
2. check_semantic_duplicate(memory_text, location_id)
   a. Get existing memories at location from cache
   b. Calculate embedding for candidate memory text
   c. Compare to existing memory embeddings (cosine similarity)
   d. All similarities < threshold (0.85)
3. Return None (no duplicate found)
4. Proceed with add_memory() → write to file + cache
```

**Duplicate Detected (Exact Match, >0.95 similarity):**
```
1. LLM synthesis creates MemorySynthesisResponse
2. check_semantic_duplicate(memory_text, location_id)
   a. Get existing memories at location
   b. Calculate embedding for candidate text
   c. Find existing memory with similarity > 0.95
3. Return existing memory (exact duplicate)
4. update_memory_metadata(existing_memory, current_turn, current_episode)
   - Update turn to latest
   - Update episode if newer
   - Increment observation count
5. Skip add_memory() call (no new entry created)
```

**Near-Duplicate (0.85-0.95 similarity):**
```
1. LLM synthesis creates MemorySynthesisResponse
2. check_semantic_duplicate(memory_text, location_id)
   a. Get existing memories at location
   b. Calculate embedding for candidate text
   c. Find existing memory with similarity 0.85-0.95
3. Return existing memory (near-duplicate)
4. Decision logic:
   - If new memory has higher confidence → supersede old with new
   - If new memory has more detail → merge details into old
   - If new memory has same info → update metadata only
5. Call appropriate action (supersede_memory / update_memory_metadata)
```

---

## 3. Integration Points

### 3.1 SimpleMemoryManager Changes

**File:** `managers/simple_memory_manager.py`

**New Dependencies:**
```python
import numpy as np
from typing import Optional, Tuple
from embedding_provider import EmbeddingProvider, SentenceTransformerProvider, LLMEmbeddingProvider
```

**New Fields:**
```python
class SimpleMemoryManager:
    def __init__(self, ...):
        # Existing fields...

        # NEW: Embedding deduplication
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.embedding_cache: Dict[int, List[Tuple[str, np.ndarray]]] = {}
        # Structure: {location_id: [(memory_title, embedding_vector), ...]}

        self.enable_deduplication: bool = config.enable_semantic_deduplication
        self.similarity_threshold: float = config.deduplication_similarity_threshold

        # Initialize embedding provider
        if self.enable_deduplication:
            self._initialize_embedding_provider()
```

**New Methods:**
```python
def _initialize_embedding_provider(self) -> None:
    """Initialize embedding provider based on configuration."""

def check_semantic_duplicate(
    self,
    candidate_text: str,
    location_id: int
) -> Optional[Tuple[Memory, float]]:
    """
    Check if candidate memory text is semantically duplicate of existing memory.

    Returns:
        (existing_memory, similarity_score) if duplicate found, else None
    """

def calculate_embedding(self, text: str) -> np.ndarray:
    """Calculate embedding vector for text using provider."""

def get_location_embeddings(self, location_id: int) -> List[Tuple[str, np.ndarray]]:
    """Get cached embeddings for all memories at location."""

def update_memory_metadata(
    self,
    memory: Memory,
    location_id: int,
    location_name: str,
    new_turn: int,
    new_episode: str
) -> bool:
    """Update metadata for existing memory (turn, episode, observation_count)."""

def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two embedding vectors."""

def rebuild_embedding_cache(self) -> None:
    """Rebuild embedding cache from current memory_cache contents."""
```

**Modified Method:**
```python
def record_action_outcome(
    self,
    location_id: int,
    location_name: str,
    action: str,
    response: str,
    z_machine_context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record action outcome and synthesize memory if warranted.

    CHANGES:
    - After LLM synthesis, before add_memory()
    - Call check_semantic_duplicate()
    - If duplicate found, update existing instead of creating new
    """
    # ... existing synthesis logic ...

    if synthesis_response.should_remember:
        # NEW: Check for semantic duplicate
        if self.enable_deduplication:
            duplicate_result = self.check_semantic_duplicate(
                synthesis_response.memory_text,
                location_id
            )

            if duplicate_result:
                existing_memory, similarity = duplicate_result

                # Handle based on similarity level
                if similarity >= 0.95:
                    # Exact duplicate: update metadata only
                    self._handle_exact_duplicate(
                        existing_memory, location_id, location_name
                    )
                    return
                elif similarity >= self.similarity_threshold:
                    # Near-duplicate: merge or supersede
                    self._handle_near_duplicate(
                        existing_memory, synthesis_response,
                        location_id, location_name
                    )
                    return

        # No duplicate: proceed with normal creation
        # ... existing add_memory() logic ...
```

### 3.2 New Module: EmbeddingProvider

**File:** `embedding_provider.py` (new file in project root)

**Abstract Base Class:**
```python
from abc import ABC, abstractmethod
import numpy as np
from typing import List

class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts (optional optimization)."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding vector dimension."""
        pass
```

**Implementation 1: SentenceTransformerProvider (Primary)**
```python
class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embedding provider using sentence-transformers library.
    Fast, local, no API costs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            self.available = False
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: uv add sentence-transformers"
            )

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32)

    def get_dimension(self) -> int:
        return 384  # all-MiniLM-L6-v2 dimension
```

**Implementation 2: LLMEmbeddingProvider (Fallback)**
```python
class LLMEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using LLM client API.
    Fallback when sentence-transformers unavailable.
    """

    def __init__(self, llm_client, model: str = "text-embedding-3-small"):
        self.client = llm_client
        self.model = model
        self.available = True

    def embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [np.array(item.embedding) for item in response.data]

    def get_dimension(self) -> int:
        return 1536  # text-embedding-3-small dimension
```

### 3.3 Configuration Changes

**File:** `pyproject.toml`

**New Section:**
```toml
[tool.zorkgpt.memory_deduplication]
# Enable semantic deduplication of memories
enable = true

# Similarity threshold for duplicate detection (0.0-1.0)
# 0.95-1.0: Exact duplicates only
# 0.85-0.95: Near-duplicates (similar meaning)
# 0.75-0.85: Loose similarity (risky, may merge distinct memories)
similarity_threshold = 0.85

# Embedding provider: "sentence-transformers" or "llm"
provider = "sentence-transformers"

# Model name for sentence-transformers
# Options: "all-MiniLM-L6-v2" (fast, 384d), "all-mpnet-base-v2" (better, 768d)
sentence_transformer_model = "all-MiniLM-L6-v2"

# Model name for LLM embeddings (fallback)
llm_embedding_model = "text-embedding-3-small"

# Cache embeddings in memory (recommended)
cache_embeddings = true

# Batch size for embedding generation (optimization)
batch_size = 32
```

**File:** `session/game_configuration.py`

**New Fields:**
```python
@dataclass
class GameConfiguration:
    # ... existing fields ...

    # Memory deduplication settings
    enable_semantic_deduplication: bool = True
    deduplication_similarity_threshold: float = 0.85
    embedding_provider: str = "sentence-transformers"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    llm_embedding_model: str = "text-embedding-3-small"
    cache_embeddings: bool = True
    embedding_batch_size: int = 32

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "GameConfiguration":
        # ... existing parsing ...

        # Parse deduplication config
        dedup_config = config_dict.get("memory_deduplication", {})
        enable_semantic_deduplication = dedup_config.get("enable", True)
        deduplication_similarity_threshold = dedup_config.get("similarity_threshold", 0.85)
        embedding_provider = dedup_config.get("provider", "sentence-transformers")
        sentence_transformer_model = dedup_config.get("sentence_transformer_model", "all-MiniLM-L6-v2")
        llm_embedding_model = dedup_config.get("llm_embedding_model", "text-embedding-3-small")
        cache_embeddings = dedup_config.get("cache_embeddings", True)
        embedding_batch_size = dedup_config.get("batch_size", 32)

        # ... rest of initialization ...
```

---

## 4. Implementation Details

### 4.1 Deduplication Logic

**Similarity Ranges and Actions:**

| Similarity | Category | Action |
|------------|----------|--------|
| 0.95 - 1.0 | Exact Duplicate | Update metadata (turn, episode, observation_count) |
| 0.85 - 0.95 | Near-Duplicate | Compare quality: supersede if new is better, else update metadata |
| 0.75 - 0.85 | Loose Similarity | Log warning, create new (distinct enough) |
| 0.0 - 0.75 | Distinct | Create new memory normally |

**Quality Comparison Logic (for near-duplicates):**
```python
def is_new_memory_better(
    existing: Memory,
    new_synthesis: MemorySynthesisResponse
) -> bool:
    """Determine if new memory is higher quality than existing."""

    # Prefer ACTIVE over TENTATIVE
    if new_synthesis.status == "ACTIVE" and existing.status == "TENTATIVE":
        return True
    if new_synthesis.status == "TENTATIVE" and existing.status == "ACTIVE":
        return False

    # Prefer more detailed (longer text)
    new_length = len(new_synthesis.memory_text)
    existing_length = len(existing.text)
    if new_length > existing_length * 1.5:  # 50% more detail
        return True

    # Prefer PERMANENT over EPHEMERAL
    if new_synthesis.persistence == "permanent" and existing.persistence == "ephemeral":
        return True

    # Default: keep existing
    return False
```

### 4.2 Embedding Cache Management

**Cache Structure:**
```python
# In-memory cache: {location_id: [(memory_title, embedding_vector), ...]}
embedding_cache: Dict[int, List[Tuple[str, np.ndarray]]] = {}
```

**Cache Lifecycle:**

1. **Initialization (on manager creation):**
   ```python
   def __init__(self, ...):
       # Load memories from file
       self._load_memories_from_file()

       # Build embedding cache if deduplication enabled
       if self.enable_deduplication:
           self.rebuild_embedding_cache()
   ```

2. **Addition (when new memory created):**
   ```python
   def add_memory(self, location_id: int, location_name: str, memory: Memory):
       # ... existing logic to write to file and memory_cache ...

       # Update embedding cache
       if self.enable_deduplication and memory.persistence != "ephemeral":
           embedding = self.calculate_embedding(memory.text)
           if location_id not in self.embedding_cache:
               self.embedding_cache[location_id] = []
           self.embedding_cache[location_id].append((memory.title, embedding))
   ```

3. **Supersession (when memory superseded):**
   ```python
   def supersede_memory(self, ...):
       # ... existing supersession logic ...

       # Remove old embedding from cache
       if self.enable_deduplication:
           self._remove_embedding_from_cache(location_id, old_memory_title)

           # Add new embedding if new memory is persistent
           if new_memory.persistence != "ephemeral":
               embedding = self.calculate_embedding(new_memory.text)
               self.embedding_cache[location_id].append((new_memory.title, embedding))
   ```

4. **Episode Reset (clear ephemeral only):**
   ```python
   def reset_episode(self):
       # Existing: clear ephemeral_cache
       self.ephemeral_cache.clear()

       # NEW: Embedding cache contains only persistent memories
       # No need to clear (ephemeral memories never added to embedding_cache)
   ```

**Cache Rebuild (on load or corruption):**
```python
def rebuild_embedding_cache(self) -> None:
    """Rebuild embedding cache from memory_cache contents."""
    self.log_info("Rebuilding embedding cache...")

    self.embedding_cache.clear()

    # Collect all memory texts for batch embedding
    batch_data: List[Tuple[int, str, str]] = []  # (location_id, title, text)

    for location_id, memories in self.memory_cache.items():
        for memory in memories:
            if memory.status != "SUPERSEDED":  # Skip superseded
                batch_data.append((location_id, memory.title, memory.text))

    # Generate embeddings in batch
    texts = [text for _, _, text in batch_data]
    embeddings = self.embedding_provider.embed_batch(texts)

    # Populate cache
    for (location_id, title, _), embedding in zip(batch_data, embeddings):
        if location_id not in self.embedding_cache:
            self.embedding_cache[location_id] = []
        self.embedding_cache[location_id].append((title, embedding))

    self.log_info(f"Rebuilt embedding cache: {len(batch_data)} embeddings")
```

### 4.3 Metadata Update Mechanism

**New Memory Field:**
```python
@dataclass
class Memory:
    # ... existing fields ...

    observation_count: int = 1  # NEW: Number of times this memory was re-observed
```

**Update Logic:**
```python
def update_memory_metadata(
    self,
    memory: Memory,
    location_id: int,
    location_name: str,
    new_turn: int,
    new_episode: str
) -> bool:
    """
    Update metadata for existing memory when duplicate detected.

    Updates:
    - turns: Update to latest turn
    - episode: Update if newer episode
    - observation_count: Increment

    File format change:
    OLD: **[CATEGORY - PERSISTENCE] Title** *(Ep01, T25, +5)*
    NEW: **[CATEGORY - PERSISTENCE] Title** *(Ep01, T25→T48, +5, seen 3x)*
    """
    try:
        # Find memory in cache
        if location_id not in self.memory_cache:
            return False

        memories = self.memory_cache[location_id]
        mem_index = next(
            (i for i, m in enumerate(memories) if m.title == memory.title),
            None
        )

        if mem_index is None:
            return False

        # Update fields
        old_turns = memories[mem_index].turns
        memories[mem_index].turns = f"{old_turns}→{new_turn}"  # Show progression
        memories[mem_index].observation_count += 1

        # Update episode if newer
        if new_episode > memories[mem_index].episode:
            memories[mem_index].episode = new_episode

        # Rewrite to file
        self._rewrite_location_memories(location_id, location_name)

        self.log_info(
            f"Updated memory metadata: '{memory.title}' (observed {memories[mem_index].observation_count}x)",
            location_id=location_id,
            turn=new_turn
        )

        return True

    except Exception as e:
        self.log_error(f"Failed to update memory metadata: {e}")
        return False
```

---

## 5. Performance Considerations

### 5.1 Benchmarks and Targets

**Target Performance:**
- Embedding calculation: < 50ms per memory (sentence-transformers)
- Similarity comparison: < 10ms for 50 existing memories
- Total overhead per memory creation: < 100ms

**Embedding Model Comparison:**

| Model | Dimension | Speed | Size | Quality |
|-------|-----------|-------|------|---------|
| all-MiniLM-L6-v2 | 384 | ~50ms | 80MB | Good |
| all-mpnet-base-v2 | 768 | ~120ms | 420MB | Better |
| text-embedding-3-small (API) | 1536 | ~200ms* | N/A | Best |

*Network latency dependent

**Recommendation:** Use `all-MiniLM-L6-v2` for balance of speed and quality.

### 5.2 Optimization Strategies

**1. Batch Embedding on Cache Rebuild:**
```python
# Instead of:
for memory in all_memories:
    embedding = provider.embed(memory.text)  # N API calls

# Do:
all_texts = [m.text for m in all_memories]
embeddings = provider.embed_batch(all_texts)  # 1 batched call
```

**2. Early Exit on Empty Location:**
```python
def check_semantic_duplicate(self, candidate_text: str, location_id: int):
    # Early exit: no existing memories at location
    if location_id not in self.embedding_cache:
        return None

    if len(self.embedding_cache[location_id]) == 0:
        return None

    # ... proceed with similarity check ...
```

**3. Lazy Embedding Calculation:**
```python
# Only calculate candidate embedding if location has existing memories
if location_id in self.embedding_cache and self.embedding_cache[location_id]:
    candidate_embedding = self.calculate_embedding(candidate_text)
    # ... compare ...
else:
    return None  # No comparison needed
```

**4. Vectorized Similarity Computation:**
```python
def find_most_similar(
    self,
    candidate_embedding: np.ndarray,
    location_embeddings: List[Tuple[str, np.ndarray]]
) -> Tuple[str, float]:
    """Find most similar embedding using vectorized operations."""

    if not location_embeddings:
        return None, 0.0

    # Stack all embeddings into matrix
    titles = [title for title, _ in location_embeddings]
    embeddings_matrix = np.stack([emb for _, emb in location_embeddings])

    # Vectorized cosine similarity (much faster than loop)
    similarities = np.dot(embeddings_matrix, candidate_embedding) / (
        np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(candidate_embedding)
    )

    max_idx = np.argmax(similarities)
    return titles[max_idx], float(similarities[max_idx])
```

### 5.3 Memory Footprint

**Embedding Cache Size Estimation:**
```
Assumptions:
- Average 200 locations visited per game
- Average 10 memories per location = 2000 total memories
- Embedding dimension: 384 (all-MiniLM-L6-v2)
- Float32: 4 bytes per dimension

Memory per embedding: 384 * 4 = 1.5 KB
Total cache size: 2000 * 1.5 KB = 3 MB

Conclusion: Negligible memory footprint, safe to cache in-memory
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**File:** `tests/test_semantic_deduplication.py`

**Test Coverage:**

1. **Embedding Provider Tests (8 tests):**
   ```python
   def test_sentence_transformer_initialization()
   def test_sentence_transformer_embed_single()
   def test_sentence_transformer_embed_batch()
   def test_llm_embedding_provider_fallback()
   def test_embedding_dimension_consistency()
   def test_embedding_provider_error_handling()
   def test_batch_vs_single_embedding_consistency()
   def test_embedding_determinism()  # Same text → same embedding
   ```

2. **Similarity Detection Tests (12 tests):**
   ```python
   def test_exact_duplicate_detection_0_95_similarity()
   def test_near_duplicate_detection_0_85_similarity()
   def test_distinct_memory_creation_below_threshold()
   def test_similarity_threshold_configuration()
   def test_empty_location_no_duplicate()
   def test_multiple_memories_find_most_similar()
   def test_case_insensitive_similarity()  # "Trap door bars" vs "trap door bars"
   def test_paraphrase_similarity()  # "Window is ajar" vs "Window slightly open"
   def test_different_context_distinct()  # Same action, different location
   def test_cosine_similarity_calculation()
   def test_vectorized_similarity_performance()
   def test_edge_case_identical_strings()
   ```

3. **Cache Management Tests (10 tests):**
   ```python
   def test_cache_initialization_on_load()
   def test_cache_update_on_new_memory()
   def test_cache_removal_on_supersession()
   def test_cache_persistence_through_episode_reset()
   def test_rebuild_cache_from_scratch()
   def test_cache_excludes_superseded_memories()
   def test_cache_excludes_ephemeral_memories()
   def test_batch_embedding_on_rebuild()
   def test_cache_corruption_recovery()
   def test_cache_memory_footprint_reasonable()
   ```

4. **Integration Tests (15 tests):**
   ```python
   def test_end_to_end_duplicate_prevention()
   def test_metadata_update_on_exact_duplicate()
   def test_observation_count_increment()
   def test_supersession_on_better_near_duplicate()
   def test_quality_comparison_logic()
   def test_deduplication_disabled_via_config()
   def test_similarity_threshold_tuning()
   def test_provider_fallback_sentence_to_llm()
   def test_file_format_with_observation_count()
   def test_concurrent_memory_creation()  # Thread safety
   def test_large_memory_set_performance()  # 1000+ memories
   def test_cross_location_no_false_positives()
   def test_real_zork_duplicate_scenarios()  # 9 trap door memories
   def test_deduplication_logging()
   def test_error_handling_embedding_failure()
   ```

### 6.2 Integration Test Scenarios

**Scenario 1: Exact Duplicate Prevention**
```python
def test_exact_duplicate_prevention():
    """
    GIVEN: Memory "Trap door bars after descending" exists at Location 193
    WHEN: Agent re-discovers same fact at Location 193
    THEN:
      - No new memory created
      - Existing memory metadata updated (turn, observation_count)
      - observation_count incremented to 2
      - File shows "seen 2x"
    """
```

**Scenario 2: Near-Duplicate Supersession**
```python
def test_near_duplicate_supersession():
    """
    GIVEN: TENTATIVE memory "Troll might be friendly" (similarity 0.88)
    WHEN: Agent discovers "Troll attacks unprovoked" (more specific)
    THEN:
      - Old memory superseded (status → SUPERSEDED)
      - New ACTIVE memory created
      - Embedding cache updated with new embedding
    """
```

**Scenario 3: Cross-Location Distinct Memories**
```python
def test_cross_location_distinct_memories():
    """
    GIVEN: Memory "Window entry leads to Kitchen" at Location 79
    WHEN: Agent discovers "Window entry leads to Chamber" at Location 134
    THEN:
      - Two distinct memories created (different locations)
      - Similarity check scoped to same location only
      - No false positive duplicate detection
    """
```

### 6.3 Performance Regression Tests

```python
def test_deduplication_overhead_acceptable():
    """
    GIVEN: 100 existing memories at location
    WHEN: Creating new memory with deduplication check
    THEN: Total time < 100ms (target)
    """

def test_cache_rebuild_performance():
    """
    GIVEN: 2000 total memories across 200 locations
    WHEN: Rebuilding embedding cache from scratch
    THEN: Total time < 5 seconds (batch embedding)
    """

def test_vectorized_similarity_faster_than_loop():
    """
    GIVEN: 50 existing embeddings at location
    WHEN: Finding most similar using vectorized vs loop
    THEN: Vectorized is 10x+ faster
    """
```

---

## 7. Edge Cases and Error Handling

### 7.1 Edge Cases

**1. Provider Unavailable:**
```python
# Graceful fallback
if sentence_transformers not available:
    try LLM embedding provider
    if LLM provider fails:
        disable deduplication, log warning, proceed without it
```

**2. Empty Candidate Text:**
```python
def check_semantic_duplicate(self, candidate_text: str, location_id: int):
    if not candidate_text or candidate_text.strip() == "":
        self.log_warning("Cannot deduplicate empty text")
        return None  # Proceed with creation
```

**3. Corrupted Embedding Cache:**
```python
try:
    embedding = self.calculate_embedding(text)
except Exception as e:
    self.log_error(f"Embedding calculation failed: {e}")
    # Disable deduplication for this memory
    return None
```

**4. Cache Rebuild Failure:**
```python
def rebuild_embedding_cache(self):
    try:
        # ... rebuild logic ...
    except Exception as e:
        self.log_error(f"Cache rebuild failed: {e}")
        self.embedding_cache.clear()
        self.enable_deduplication = False  # Disable temporarily
        # System continues without deduplication
```

**5. Very Long Memory Text:**
```python
# Embedding models have token limits (e.g., 512 tokens for sentence-transformers)
def calculate_embedding(self, text: str) -> np.ndarray:
    # Truncate if too long
    max_length = 500  # Conservative for 512 token models
    if len(text) > max_length:
        text = text[:max_length]
        self.log_warning(f"Truncated long memory text: {len(text)} chars")

    return self.embedding_provider.embed(text)
```

### 7.2 Error Recovery

**Strategy: Fail gracefully, never crash memory creation**

```python
def record_action_outcome(self, ...):
    # ... synthesis logic ...

    if synthesis_response.should_remember:
        duplicate_result = None

        # Try deduplication, but don't fail if it errors
        if self.enable_deduplication:
            try:
                duplicate_result = self.check_semantic_duplicate(
                    synthesis_response.memory_text,
                    location_id
                )
            except Exception as e:
                self.log_error(f"Deduplication check failed: {e}")
                duplicate_result = None  # Proceed without deduplication

        # Always proceed with memory creation if deduplication failed
        if duplicate_result is None:
            # Normal memory creation path
            self.add_memory(location_id, location_name, memory)
```

---

## 8. Configuration Tuning Guide

### 8.1 Similarity Threshold Selection

**Conservative (0.90-0.95):**
- Only catches near-exact duplicates
- Low risk of false positives
- May allow some similar memories through
- **Use when:** High precision required, memory bloat acceptable

**Balanced (0.85-0.90) [RECOMMENDED]:**
- Catches exact and near-duplicates
- Good balance of precision/recall
- Minimal false positives on distinct memories
- **Use when:** General gameplay (default)

**Aggressive (0.75-0.85):**
- Catches loosely similar memories
- Higher risk of merging distinct memories
- More aggressive deduplication
- **Use when:** Memory bloat is severe, willing to accept some false merges

### 8.2 Provider Selection

**sentence-transformers (Recommended):**
- **Pros:** Fast (~50ms), free, offline
- **Cons:** Requires installation, 80MB model download
- **When:** Default choice for all users

**LLM embeddings:**
- **Pros:** No installation, highest quality
- **Cons:** API costs (~$0.02 per 1000 memories), network latency
- **When:** sentence-transformers unavailable, or quality critical

### 8.3 Performance Tuning

**High-Performance Mode:**
```toml
[tool.zorkgpt.memory_deduplication]
enable = true
provider = "sentence-transformers"
sentence_transformer_model = "all-MiniLM-L6-v2"  # Fastest
cache_embeddings = true
batch_size = 64  # Larger batches
```

**High-Quality Mode:**
```toml
[tool.zorkgpt.memory_deduplication]
enable = true
provider = "sentence-transformers"
sentence_transformer_model = "all-mpnet-base-v2"  # Better embeddings
cache_embeddings = true
batch_size = 16  # Smaller batches for memory
```

**Cost-Optimized Mode (No sentence-transformers):**
```toml
[tool.zorkgpt.memory_deduplication]
enable = true
provider = "llm"
llm_embedding_model = "text-embedding-3-small"  # Cheap API
cache_embeddings = true
batch_size = 100  # Batch API calls
```

---

## 9. Migration Path

### 9.1 Existing Memory File Compatibility

**No migration required:**
- Existing `Memories.md` files work unchanged
- Deduplication only affects new memory creation
- Embedding cache built lazily from existing memories

**Optional cleanup:**
```bash
# After deploying deduplication, optionally clean existing duplicates
uv run python scripts/deduplicate_existing_memories.py \
  --threshold 0.85 \
  --dry-run  # Preview changes first
```

### 9.2 Rollback Plan

**To disable deduplication:**
```toml
[tool.zorkgpt.memory_deduplication]
enable = false  # Disable feature
```

**System continues working normally:**
- No code changes required
- Embedding cache not built
- Memory creation follows original logic
- Zero performance impact

---

## 10. Monitoring and Metrics

### 10.1 Logging

**Key Log Messages:**

```python
# Successful duplicate detection
self.log_info(
    f"Duplicate detected: '{candidate_title}' matches '{existing_title}' ({similarity:.2f})",
    location_id=location_id,
    similarity=similarity,
    action="metadata_update"
)

# Supersession due to near-duplicate
self.log_info(
    f"Near-duplicate superseded: '{old_title}' → '{new_title}' ({similarity:.2f})",
    location_id=location_id,
    similarity=similarity,
    action="supersession"
)

# Cache rebuild
self.log_info(
    f"Rebuilt embedding cache: {total_embeddings} embeddings in {duration:.2f}s",
    total_embeddings=total_embeddings,
    duration=duration
)

# Performance metrics
self.log_debug(
    f"Deduplication check: {duration_ms:.1f}ms for {num_existing} memories",
    duration_ms=duration_ms,
    num_existing=num_existing
)
```

### 10.2 Metrics to Track

**Episode-Level:**
- `duplicates_prevented`: Count of duplicate memories blocked
- `metadata_updates`: Count of observation_count increments
- `supersessions_from_near_duplicates`: Count of near-duplicate supersessions
- `avg_deduplication_time_ms`: Average time per deduplication check

**System-Level:**
- `embedding_cache_size`: Total embeddings cached
- `embedding_cache_rebuild_count`: Number of cache rebuilds
- `deduplication_error_count`: Number of deduplication failures

**Add to episode_log.jsonl:**
```json
{
  "timestamp": "2025-11-25T...",
  "event_type": "deduplication_metrics",
  "episode_id": "2025-11-25T14:30:00",
  "duplicates_prevented": 3,
  "metadata_updates": 3,
  "supersessions": 1,
  "avg_check_time_ms": 45.2
}
```

---

## 11. Future Enhancements

### 11.1 Phase 2: Cross-Location Pattern Extraction

After basic deduplication working:
- Detect similar memories **across locations**
- Extract universal patterns → promote to knowledge
- Example: "Open X, enter X" pattern detected at 5+ locations → knowledgebase

### 11.2 Phase 3: Periodic Batch Consolidation

Scheduled cleanup job:
- Run every 5 episodes
- Re-embed all memories (catch model improvements)
- Find and merge duplicates that slipped through
- Generate consolidation report

### 11.3 Phase 4: Semantic Search

Leverage embedding infrastructure:
- Query: "What do we know about dark areas?"
- Semantic search across all memories
- Return most relevant memories regardless of keyword match

---

## 12. Success Metrics

### 12.1 Quantitative Goals

**Primary Metrics:**
1. **Duplicate Prevention Rate**: > 95% of exact duplicates caught
2. **False Positive Rate**: < 2% of distinct memories incorrectly merged
3. **Performance Overhead**: < 100ms average per memory creation
4. **Memory File Size Reduction**: 20-30% reduction over 10 episodes

**Secondary Metrics:**
1. Context token savings: ~10-15% fewer memory tokens in agent context
2. LLM synthesis call reduction: ~5% fewer synthesis calls (caught early)
3. Zero system crashes from deduplication errors (graceful degradation)

### 12.2 Qualitative Goals

1. **Memory file readability**: Easier for humans to review Memories.md
2. **Agent performance**: No degradation in gameplay quality
3. **Developer experience**: Simple configuration, clear logging
4. **User trust**: Transparent duplicate handling, explainable merges

---

## 13. Dependencies

### 13.1 New Python Packages

**Primary:**
```toml
[tool.uv]
dependencies = [
    # ... existing ...
    "sentence-transformers>=2.2.0",  # NEW
    "numpy>=1.24.0",  # Already present
]
```

**Installation:**
```bash
uv add sentence-transformers
```

**Model Download (automatic on first run):**
- Model: `all-MiniLM-L6-v2`
- Size: ~80MB
- Location: `~/.cache/torch/sentence_transformers/`

### 13.2 Existing Dependencies (no changes)

- numpy (already in project)
- LLM client (fallback provider)
- File I/O (existing memory system)

---

## 14. Timeline and Phases

### Phase 1: Core Implementation (Week 1)
- [ ] Create `embedding_provider.py` module
- [ ] Add SentenceTransformerProvider and LLMEmbeddingProvider
- [ ] Add deduplication logic to SimpleMemoryManager
- [ ] Add configuration to GameConfiguration
- [ ] Basic unit tests (embedding, similarity)

### Phase 2: Integration (Week 2)
- [ ] Integrate into `record_action_outcome()` workflow
- [ ] Implement metadata update mechanism
- [ ] Add embedding cache management
- [ ] Integration tests (end-to-end scenarios)

### Phase 3: Optimization (Week 3)
- [ ] Vectorized similarity computation
- [ ] Batch embedding on cache rebuild
- [ ] Performance benchmarks
- [ ] Memory footprint analysis

### Phase 4: Testing & Polish (Week 4)
- [ ] Comprehensive test suite (45 tests)
- [ ] Real Zork duplicate scenarios validation
- [ ] Documentation and examples
- [ ] Configuration tuning guide

---

## 15. Open Questions

1. **Should we deduplicate across TENTATIVE and ACTIVE memories?**
   - Current spec: Yes, compare regardless of status
   - Alternative: Only compare within same status
   - **Decision needed:** Confirm approach

2. **How to handle EPHEMERAL memories?**
   - Current spec: Never add to embedding cache (ephemeral never duplicates)
   - Alternative: Add to cache but clear on reset
   - **Decision needed:** Confirm ephemeral handling

3. **Should observation_count affect memory ranking in context?**
   - Current spec: Just metadata tracking
   - Alternative: Show high-observation memories first in context
   - **Decision needed:** Future enhancement or now?

4. **Embedding model selection flexibility?**
   - Current spec: Config allows model selection
   - Alternative: Auto-select based on available memory
   - **Decision needed:** Keep configurable or auto-detect?

5. **Cross-episode embedding cache persistence?**
   - Current spec: Rebuild from file on each episode start
   - Alternative: Pickle cache to disk for faster startup
   - **Decision needed:** Worth complexity? (~2-3s rebuild time)

---

## 16. References

- ACE Paper: Section 3.2 "Semantic Deduplication"
- sentence-transformers docs: https://www.sbert.net/
- Cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- Memory system: `managers/CLAUDE.md`
- Existing memory synthesis: `managers/simple_memory_manager.py:482-600`
