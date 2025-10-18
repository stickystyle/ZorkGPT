# Mem0-Enhanced Critic System Specification

## Executive Summary

This document specifies a memory-informed critic system that dramatically reduces LLM costs by leveraging Mem0's semantic memory to bypass unnecessary critic evaluations. The system maintains decision quality while reducing critic invocations by 60-90% through experience-based pattern matching and predictive scoring.

## Problem Statement

Current critic system issues:
- **Cost**: Critic evaluates EVERY action, with multiple calls per rejection
- **Redundancy**: Re-evaluates identical situations across episodes
- **Statelessness**: No learning from past successes/failures
- **Loop Detection**: Expensive LLM calls for algorithmic pattern detection

## Solution Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│                   Proposed Action                        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   Memory-Based Prescreening │
         │   (Mem0 Similarity Search)  │
         └────────────┬────────────────┘
                      │
              ┌───────▼───────┐
              │ Confidence > θ │
              └───┬───────┬───┘
                Yes│       │No
                   │       │
         ┌─────────▼───┐   │
         │Return Cached│   │
         │   Score     │   │
         └─────────────┘   │
                           │
                   ┌───────▼───────┐
                   │  Loop Pattern  │
                   │   Detection    │
                   └───┬───────┬───┘
                     Yes│       │No
                        │       │
              ┌─────────▼───┐   │
              │Return Loop  │   │
              │Override Score│   │
              └─────────────┘   │
                                │
                        ┌───────▼───────┐
                        │  Safe Pattern  │
                        │   Matching     │
                        └───┬───────┬───┘
                          Yes│       │No
                             │       │
                   ┌─────────▼───┐   │
                   │Return Safe  │   │
                   │Pattern Score│   │
                   └─────────────┘   │
                                     │
                           ┌─────────▼─────────┐
                           │   Invoke Critic   │
                           │   (LLM Call)      │
                           └─────────┬─────────┘
                                     │
                           ┌─────────▼─────────┐
                           │  Store Experience  │
                           │    in Mem0        │
                           └───────────────────┘
```

## Core Components

### 1. MemoryInformedCritic Class

```python
# zork_memory_critic.py
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from mem0 import Memory
from zork_critic import ZorkCritic, CriticResponse

class DecisionSource(Enum):
    """Track how the critic decision was made."""
    MEMORY_EXACT = "memory_exact_match"
    MEMORY_SIMILAR = "memory_similar_situation"
    PATTERN_SAFE = "pattern_safe_action"
    PATTERN_LOOP = "pattern_loop_detection"
    LLM_CRITIC = "llm_critic_evaluation"
    FALLBACK = "fallback_default"

@dataclass
class CriticDecision:
    """Enhanced critic response with decision metadata."""
    score: float
    justification: str
    confidence: float
    source: DecisionSource
    memory_references: List[str] = None
    llm_invoked: bool = False
    cost_saved: float = 0.0  # Estimated cost saved by avoiding LLM

class MemoryInformedCritic:
    """
    Critic system that uses Mem0 memories to avoid unnecessary LLM calls.
    
    This class wraps the existing ZorkCritic and adds memory-based
    prescreening to dramatically reduce LLM invocations.
    """
    
    def __init__(
        self,
        memory_manager,
        base_critic: Optional[ZorkCritic] = None,
        config: Dict[str, Any] = None,
        logger=None,
        episode_id: str = "unknown"
    ):
        """
        Initialize the memory-informed critic.
        
        Args:
            memory_manager: MemoryManager instance with Mem0 integration
            base_critic: Existing ZorkCritic instance (optional)
            config: Configuration dictionary
            logger: Logger instance
            episode_id: Current episode ID
        """
        self.memory_manager = memory_manager
        self.base_critic = base_critic or ZorkCritic(
            logger=logger,
            episode_id=episode_id
        )
        self.config = config or self._default_config()
        self.logger = logger
        self.episode_id = episode_id
        
        # Statistics tracking
        self.stats = {
            "total_evaluations": 0,
            "memory_bypasses": 0,
            "pattern_bypasses": 0,
            "llm_invocations": 0,
            "cost_saved": 0.0
        }
        
        # Cache for recent decisions
        self.decision_cache = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for memory-informed critic."""
        return {
            "memory_confidence_threshold": 0.85,
            "pattern_confidence_threshold": 0.7,
            "loop_detection_window": 10,
            "cache_ttl_turns": 5,
            "enable_safe_patterns": True,
            "enable_loop_detection": True,
            "estimated_critic_cost": 0.0001  # Per call
        }
    
    def evaluate_action(
        self,
        game_state_text: str,
        proposed_action: str,
        current_location: str,
        context: Dict[str, Any] = None
    ) -> CriticDecision:
        """
        Evaluate an action using memory-informed decision making.
        
        Args:
            game_state_text: Current game state
            proposed_action: Action to evaluate
            current_location: Current location name
            context: Additional context (inventory, recent_actions, etc.)
            
        Returns:
            CriticDecision with score, justification, and metadata
        """
        self.stats["total_evaluations"] += 1
        
        # 1. Check cache first
        cache_key = self._get_cache_key(current_location, proposed_action)
        if cache_key in self.decision_cache:
            cached = self.decision_cache[cache_key]
            if self._is_cache_valid(cached):
                return cached["decision"]
        
        # 2. Memory-based evaluation
        memory_decision = self._evaluate_from_memory(
            proposed_action,
            current_location,
            game_state_text,
            context
        )
        
        if memory_decision and memory_decision.confidence >= self.config["memory_confidence_threshold"]:
            self.stats["memory_bypasses"] += 1
            self.stats["cost_saved"] += self.config["estimated_critic_cost"]
            self._cache_decision(cache_key, memory_decision)
            return memory_decision
        
        # 3. Pattern-based evaluation
        if self.config["enable_loop_detection"]:
            loop_decision = self._detect_loop_pattern(
                proposed_action,
                context.get("recent_actions", []),
                context.get("recent_locations", [])
            )
            if loop_decision:
                self.stats["pattern_bypasses"] += 1
                self.stats["cost_saved"] += self.config["estimated_critic_cost"]
                self._cache_decision(cache_key, loop_decision)
                return loop_decision
        
        if self.config["enable_safe_patterns"]:
            safe_decision = self._check_safe_patterns(
                proposed_action,
                current_location,
                context
            )
            if safe_decision:
                self.stats["pattern_bypasses"] += 1
                self.stats["cost_saved"] += self.config["estimated_critic_cost"]
                self._cache_decision(cache_key, safe_decision)
                return safe_decision
        
        # 4. Fall back to LLM critic
        llm_decision = self._invoke_llm_critic(
            game_state_text,
            proposed_action,
            current_location,
            context,
            memory_hint=memory_decision  # Pass memory context to LLM
        )
        
        # 5. Store this experience for future use
        self._store_experience(
            proposed_action,
            current_location,
            llm_decision,
            game_state_text,
            context
        )
        
        self._cache_decision(cache_key, llm_decision)
        return llm_decision
    
    def _evaluate_from_memory(
        self,
        action: str,
        location: str,
        game_state: str,
        context: Dict[str, Any]
    ) -> Optional[CriticDecision]:
        """
        Evaluate action based on similar past experiences.
        
        Returns None if no confident memory match found.
        """
        # Query for similar experiences
        query = f"Location: {location}, Action: {action}"
        memories = self.memory_manager.recall_similar(
            query=query,
            limit=5,
            filters={
                "location": location,  # Prioritize same location
                "action_type": self._classify_action(action)
            }
        )
        
        if not memories:
            return None
        
        # Analyze memory matches
        best_match = memories[0]
        
        # Calculate confidence based on similarity and recency
        confidence = self._calculate_memory_confidence(
            best_match,
            action,
            location,
            context
        )
        
        if confidence < 0.5:  # Too different to use
            return None
        
        # Build decision from memory
        return CriticDecision(
            score=best_match.metadata.get("outcome_score", 0.0),
            justification=f"Based on similar past experience: {best_match.metadata.get('outcome_summary', 'action previously attempted')}",
            confidence=confidence,
            source=DecisionSource.MEMORY_EXACT if confidence > 0.95 else DecisionSource.MEMORY_SIMILAR,
            memory_references=[best_match.id],
            llm_invoked=False,
            cost_saved=self.config["estimated_critic_cost"]
        )
    
    def _detect_loop_pattern(
        self,
        action: str,
        recent_actions: List[str],
        recent_locations: List[str]
    ) -> Optional[CriticDecision]:
        """
        Detect loop patterns using memory and recent history.
        """
        if len(recent_actions) < 3:
            return None
        
        # Query memories for loop patterns at this location
        loop_memories = self.memory_manager.get_pattern(
            pattern_type="action_loop",
            location=recent_locations[-1] if recent_locations else None,
            time_window=self.config["loop_detection_window"]
        )
        
        # Check for immediate repetition
        if len(set(recent_actions[-3:])) == 1 and recent_actions[-1] == action:
            return CriticDecision(
                score=-0.8,
                justification="Detected immediate action repetition (loop pattern)",
                confidence=0.95,
                source=DecisionSource.PATTERN_LOOP,
                llm_invoked=False,
                cost_saved=self.config["estimated_critic_cost"]
            )
        
        # Check for oscillation pattern
        if len(recent_actions) >= 4:
            if (recent_actions[-4] == recent_actions[-2] == action and
                recent_actions[-3] == recent_actions[-1]):
                return CriticDecision(
                    score=-0.7,
                    justification="Detected oscillation pattern between two actions",
                    confidence=0.9,
                    source=DecisionSource.PATTERN_LOOP,
                    llm_invoked=False,
                    cost_saved=self.config["estimated_critic_cost"]
                )
        
        # Check memory for similar loop patterns
        if loop_memories and loop_memories.confidence > 0.8:
            return CriticDecision(
                score=-0.6,
                justification=f"Memory indicates loop pattern: {loop_memories.description}",
                confidence=loop_memories.confidence,
                source=DecisionSource.PATTERN_LOOP,
                memory_references=loop_memories.memory_ids,
                llm_invoked=False,
                cost_saved=self.config["estimated_critic_cost"]
            )
        
        return None
    
    def _check_safe_patterns(
        self,
        action: str,
        location: str,
        context: Dict[str, Any]
    ) -> Optional[CriticDecision]:
        """
        Check if action matches known safe patterns.
        """
        # Define safe patterns that rarely need criticism
        SAFE_PATTERNS = {
            "inventory": {
                "condition": lambda ctx: not ctx.get("in_combat", False),
                "score": 0.3,
                "justification": "Checking inventory is generally safe"
            },
            "look": {
                "condition": lambda ctx: True,
                "score": 0.5,
                "justification": "Looking around is always safe and informative"
            },
            "examine": {
                "condition": lambda ctx: not ctx.get("in_combat", False),
                "score": 0.4,
                "justification": "Examining objects is safe exploration"
            },
            "save": {
                "condition": lambda ctx: not ctx.get("in_combat", False),
                "score": 0.8,
                "justification": "Saving game is always beneficial"
            }
        }
        
        # Check if action matches a safe pattern
        action_verb = action.lower().split()[0] if action else ""
        
        if action_verb in SAFE_PATTERNS:
            pattern = SAFE_PATTERNS[action_verb]
            if pattern["condition"](context or {}):
                # Query memory for any negative experiences with this pattern
                negative_memories = self.memory_manager.recall_similar(
                    query=f"Failed {action_verb} at {location}",
                    limit=3,
                    filters={"success": False, "location": location}
                )
                
                # Adjust confidence based on negative memories
                confidence = 0.8
                if negative_memories:
                    confidence -= 0.1 * len(negative_memories)
                
                if confidence > 0.5:
                    return CriticDecision(
                        score=pattern["score"],
                        justification=pattern["justification"],
                        confidence=confidence,
                        source=DecisionSource.PATTERN_SAFE,
                        llm_invoked=False,
                        cost_saved=self.config["estimated_critic_cost"]
                    )
        
        return None
    
    def _invoke_llm_critic(
        self,
        game_state_text: str,
        proposed_action: str,
        current_location: str,
        context: Dict[str, Any],
        memory_hint: Optional[CriticDecision] = None
    ) -> CriticDecision:
        """
        Invoke the actual LLM critic as a fallback.
        """
        self.stats["llm_invocations"] += 1
        
        # Enhance context with memory hint if available
        enhanced_context = context.copy() if context else {}
        if memory_hint:
            enhanced_context["memory_suggestion"] = {
                "score": memory_hint.score,
                "confidence": memory_hint.confidence,
                "reference": memory_hint.justification
            }
        
        # Call the base critic
        critic_response = self.base_critic.evaluate_action(
            game_state_text=game_state_text,
            proposed_action=proposed_action,
            available_exits=enhanced_context.get("available_exits", []),
            action_counts=enhanced_context.get("action_counts"),
            previous_actions_and_responses=enhanced_context.get("recent_actions", []),
            current_location_name=current_location,
            failed_actions_by_location=enhanced_context.get("failed_actions_by_location", {})
        )
        
        # Convert to CriticDecision
        return CriticDecision(
            score=critic_response.score,
            justification=critic_response.justification,
            confidence=critic_response.confidence,
            source=DecisionSource.LLM_CRITIC,
            llm_invoked=True,
            cost_saved=0.0
        )
    
    def _store_experience(
        self,
        action: str,
        location: str,
        decision: CriticDecision,
        game_state: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Store the critic decision as an experience in Mem0.
        """
        experience = {
            "action": action,
            "location": location,
            "critic_score": decision.score,
            "critic_justification": decision.justification,
            "game_state_summary": game_state[:200],  # First 200 chars
            "timestamp": context.get("timestamp"),
            "turn": context.get("turn"),
            "episode_id": self.episode_id,
            "action_type": self._classify_action(action),
            "outcome_score": decision.score,
            "outcome_summary": decision.justification[:100]
        }
        
        # Store in Mem0
        self.memory_manager.store_experience(experience)
    
    def _calculate_memory_confidence(
        self,
        memory: Any,
        action: str,
        location: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score for using a memory as basis for decision.
        
        Factors:
        - Semantic similarity of action
        - Location match
        - Recency of memory
        - Success rate of similar memories
        - Context similarity (inventory, game state)
        """
        confidence = 0.0
        
        # Base similarity from Mem0
        confidence += memory.relevance * 0.4
        
        # Location match bonus
        if memory.metadata.get("location") == location:
            confidence += 0.2
        
        # Recency factor (memories decay in relevance)
        turns_ago = context.get("turn", 0) - memory.metadata.get("turn", 0)
        if turns_ago < 100:
            confidence += 0.1
        elif turns_ago < 500:
            confidence += 0.05
        
        # Action similarity
        if memory.metadata.get("action") == action:
            confidence += 0.2
        elif self._classify_action(memory.metadata.get("action")) == self._classify_action(action):
            confidence += 0.1
        
        # Success history
        if memory.metadata.get("outcome_score", 0) > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _classify_action(self, action: str) -> str:
        """Classify action into categories for pattern matching."""
        if not action:
            return "unknown"
        
        action_lower = action.lower()
        first_word = action_lower.split()[0] if action_lower else ""
        
        # Movement actions
        if first_word in ["go", "walk", "move", "enter", "exit", "climb", "descend"]:
            return "movement"
        elif first_word in ["north", "south", "east", "west", "up", "down", "ne", "nw", "se", "sw"]:
            return "movement"
        
        # Interaction actions
        elif first_word in ["take", "get", "drop", "put", "give", "throw"]:
            return "item_interaction"
        elif first_word in ["open", "close", "unlock", "lock", "push", "pull", "turn"]:
            return "object_interaction"
        
        # Information actions
        elif first_word in ["look", "examine", "read", "search", "inspect"]:
            return "information"
        elif first_word in ["inventory", "score", "diagnostic", "save", "restore"]:
            return "meta"
        
        # Combat actions
        elif first_word in ["attack", "hit", "kill", "fight", "strike"]:
            return "combat"
        
        return "other"
    
    def _get_cache_key(self, location: str, action: str) -> str:
        """Generate cache key for decision caching."""
        return f"{location}|{action.lower()}"
    
    def _cache_decision(self, key: str, decision: CriticDecision) -> None:
        """Cache a decision with TTL."""
        self.decision_cache[key] = {
            "decision": decision,
            "turn": self.stats["total_evaluations"],
            "ttl": self.config["cache_ttl_turns"]
        }
        
        # Clean old cache entries
        self._clean_cache()
    
    def _is_cache_valid(self, cached: Dict) -> bool:
        """Check if cached decision is still valid."""
        turns_elapsed = self.stats["total_evaluations"] - cached["turn"]
        return turns_elapsed < cached["ttl"]
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        current_turn = self.stats["total_evaluations"]
        expired_keys = [
            key for key, value in self.decision_cache.items()
            if current_turn - value["turn"] >= value["ttl"]
        ]
        for key in expired_keys:
            del self.decision_cache[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about critic performance and cost savings.
        """
        total = self.stats["total_evaluations"] or 1  # Avoid division by zero
        
        return {
            "total_evaluations": self.stats["total_evaluations"],
            "llm_invocations": self.stats["llm_invocations"],
            "memory_bypasses": self.stats["memory_bypasses"],
            "pattern_bypasses": self.stats["pattern_bypasses"],
            "bypass_rate": 1 - (self.stats["llm_invocations"] / total),
            "cost_saved": self.stats["cost_saved"],
            "cache_size": len(self.decision_cache),
            "average_cost_per_evaluation": (
                self.stats["llm_invocations"] * self.config["estimated_critic_cost"]
            ) / total
        }
    
    def reset_statistics(self) -> None:
        """Reset statistics for new episode."""
        self.stats = {
            "total_evaluations": 0,
            "memory_bypasses": 0,
            "pattern_bypasses": 0,
            "llm_invocations": 0,
            "cost_saved": 0.0
        }
        self.decision_cache.clear()
```

## Integration with Orchestrator

### Modified Turn Processing

```python
# In orchestration/zork_orchestrator_v2.py

def _run_turn(self, game_interface: GameServerClient, current_state: str) -> Tuple[str, str]:
    """Run a single game turn with memory-informed critic."""
    
    # ... existing agent action generation ...
    
    # Build context for critic
    critic_context = {
        "available_exits": self.map_manager.game_map.get_available_exits(
            self.game_state.current_room_name_for_map
        ),
        "action_counts": self.game_state.action_counts,
        "recent_actions": [a for a, _ in self.game_state.action_history[-10:]],
        "recent_locations": self.game_state.recent_locations[-10:],
        "failed_actions_by_location": self.game_state.failed_actions_by_location,
        "in_combat": self.state_manager.get_combat_status(),
        "inventory": self.game_state.current_inventory,
        "turn": self.game_state.turn_count,
        "timestamp": datetime.now().isoformat()
    }
    
    # Use memory-informed critic
    critic_decision = self.memory_critic.evaluate_action(
        game_state_text=current_state,
        proposed_action=proposed_action,
        current_location=self.game_state.current_room_name_for_map,
        context=critic_context
    )
    
    # Log decision source for monitoring
    self.logger.info(
        f"Critic decision via {critic_decision.source.value}",
        extra={
            "event_type": "critic_decision",
            "source": critic_decision.source.value,
            "llm_invoked": critic_decision.llm_invoked,
            "score": critic_decision.score,
            "confidence": critic_decision.confidence,
            "cost_saved": critic_decision.cost_saved
        }
    )
    
    # ... rest of turn processing ...
```

## Configuration Updates

### Add to pyproject.toml

```toml
[tool.zorkgpt.memory_critic]
# Memory-informed critic configuration
enabled = true
memory_confidence_threshold = 0.85
pattern_confidence_threshold = 0.7
loop_detection_window = 10
cache_ttl_turns = 5
enable_safe_patterns = true
enable_loop_detection = true

# Cost tracking
estimated_critic_cost_per_call = 0.0001
track_cost_savings = true

# Fallback behavior
fallback_to_llm_threshold = 0.5  # Min confidence to avoid LLM
max_memory_lookups = 5  # Max memories to consider

# Pattern library
[tool.zorkgpt.memory_critic.safe_patterns]
inventory = { score = 0.3, combat_allowed = false }
look = { score = 0.5, combat_allowed = true }
examine = { score = 0.4, combat_allowed = false }
save = { score = 0.8, combat_allowed = false }
```

## Migration Strategy

### Phase 1: Parallel Testing (Week 1)
```python
# Run both critics in parallel for comparison
def evaluate_with_comparison(self, ...):
    memory_decision = self.memory_critic.evaluate_action(...)
    llm_decision = self.base_critic.evaluate_action(...)
    
    # Log divergence for analysis
    if abs(memory_decision.score - llm_decision.score) > 0.3:
        self.logger.info("Critic divergence detected", ...)
    
    # Use LLM decision but track memory performance
    return llm_decision
```

### Phase 2: Gradual Rollout (Week 2)
- Start with 10% memory-based decisions
- Increase by 10% daily if metrics are stable
- Monitor score progression and completion rates

### Phase 3: Full Deployment (Week 3)
- 100% memory-informed critic
- LLM only for low-confidence situations
- Continuous monitoring and tuning

## Performance Metrics

### Key Performance Indicators

```python
{
    "cost_metrics": {
        "llm_calls_per_turn": 0.15,  # Target: < 0.2
        "cost_per_episode": 0.45,     # Target: < $0.50
        "bypass_rate": 0.85,          # Target: > 0.8
        "cost_reduction": 0.75        # Target: > 70%
    },
    "quality_metrics": {
        "score_progression": 125,      # Points per episode
        "bad_action_rate": 0.02,      # Failed actions
        "loop_detection_rate": 0.95,  # Caught loops
        "novel_situation_handling": 0.88  # Success on new scenarios
    },
    "performance_metrics": {
        "memory_lookup_time_ms": 12,
        "cache_hit_rate": 0.65,
        "decision_latency_ms": 45
    }
}
```

### Monitoring Dashboard

```python
def get_critic_dashboard(self) -> Dict[str, Any]:
    """Real-time critic performance dashboard."""
    stats = self.memory_critic.get_statistics()
    
    return {
        "current_episode": {
            "total_turns": self.game_state.turn_count,
            "critic_calls": stats["total_evaluations"],
            "llm_calls": stats["llm_invocations"],
            "memory_hits": stats["memory_bypasses"],
            "pattern_hits": stats["pattern_bypasses"],
            "current_bypass_rate": stats["bypass_rate"],
            "cost_saved_usd": stats["cost_saved"]
        },
        "decision_sources": {
            "memory": stats["memory_bypasses"] / max(stats["total_evaluations"], 1),
            "patterns": stats["pattern_bypasses"] / max(stats["total_evaluations"], 1),
            "llm": stats["llm_invocations"] / max(stats["total_evaluations"], 1)
        },
        "cache_performance": {
            "size": stats["cache_size"],
            "hit_rate": self._calculate_cache_hit_rate()
        }
    }
```

## Testing Strategy

### Unit Tests

```python
# tests/test_memory_critic.py

def test_memory_bypass_high_confidence():
    """Test that high-confidence memories bypass LLM."""
    memory_manager = MockMemoryManager()
    memory_manager.add_memory(
        action="take lamp",
        location="West of House",
        score=0.8,
        relevance=0.95
    )
    
    critic = MemoryInformedCritic(memory_manager)
    decision = critic.evaluate_action(
        game_state_text="West of House...",
        proposed_action="take lamp",
        current_location="West of House"
    )
    
    assert decision.source == DecisionSource.MEMORY_EXACT
    assert not decision.llm_invoked
    assert decision.score == 0.8

def test_loop_detection():
    """Test that loops are detected without LLM."""
    critic = MemoryInformedCritic(MockMemoryManager())
    
    decision = critic.evaluate_action(
        game_state_text="...",
        proposed_action="go north",
        current_location="Forest",
        context={
            "recent_actions": ["go north", "go south", "go north", "go south"],
            "recent_locations": ["Forest", "Path", "Forest", "Path"]
        }
    )
    
    assert decision.source == DecisionSource.PATTERN_LOOP
    assert not decision.llm_invoked
    assert decision.score < 0

def test_fallback_to_llm():
    """Test fallback to LLM for novel situations."""
    critic = MemoryInformedCritic(
        MockMemoryManager(),  # Empty memories
        base_critic=MockCritic()
    )
    
    decision = critic.evaluate_action(
        game_state_text="Strange alien room...",
        proposed_action="activate quantum manipulator",
        current_location="Unknown Dimension"
    )
    
    assert decision.source == DecisionSource.LLM_CRITIC
    assert decision.llm_invoked
```

### Integration Tests

```python
def test_episode_cost_reduction():
    """Test cost reduction over a full episode."""
    orchestrator = ZorkOrchestratorV2(episode_id="test")
    orchestrator.memory_critic = MemoryInformedCritic(...)
    
    # Run episode
    score = orchestrator.play_episode(game_interface)
    
    # Check cost metrics
    stats = orchestrator.memory_critic.get_statistics()
    assert stats["bypass_rate"] > 0.7
    assert stats["cost_saved"] > 0.0
```

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| Memory returns wrong decision | Confidence thresholds + similarity validation |
| Pattern matching too aggressive | Conservative patterns + gradual rollout |
| Cache becomes stale | TTL-based expiration + turn-based invalidation |
| Memory lookup latency | Indexed queries + local caching |
| Catastrophic failure | Fallback to base critic + monitoring alerts |

## Future Enhancements

### Advanced Pattern Learning
- Train lightweight ML model on critic decisions
- Learn location-specific action patterns
- Discover emergent safe/unsafe patterns

### Predictive Scoring
- Predict critic scores before evaluation
- Pre-filter obviously bad actions
- Suggest alternatives based on memories

### Cross-Episode Learning
- Share learned patterns across episodes
- Build global action success rates
- Identify universally safe/unsafe actions

## Conclusion

The memory-informed critic system leverages Mem0's semantic memory to dramatically reduce LLM costs while maintaining decision quality. By checking memories and patterns before invoking the expensive critic, we can achieve 70-90% cost reduction after initial learning phases. The system is designed to fail gracefully, with fallback to the traditional critic when confidence is low, ensuring that decision quality is maintained even in novel situations.

## Appendix: Quick Start Guide

```python
# 1. Install dependencies
pip install mem0ai

# 2. Initialize memory-informed critic
from zork_memory_critic import MemoryInformedCritic

memory_critic = MemoryInformedCritic(
    memory_manager=self.memory_manager,
    base_critic=self.critic,
    config=self.config.memory_critic,
    logger=self.logger,
    episode_id=episode_id
)

# 3. Replace critic calls in orchestrator
# Old:
critic_result = self.critic.evaluate_action(...)

# New:
critic_decision = self.memory_critic.evaluate_action(...)

# 4. Monitor performance
stats = self.memory_critic.get_statistics()
print(f"Cost saved: ${stats['cost_saved']:.4f}")
print(f"Bypass rate: {stats['bypass_rate']:.2%}")
```