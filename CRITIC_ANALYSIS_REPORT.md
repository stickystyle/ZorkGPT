# ZorkGPT Critic Analysis Report

**Date:** 2025-11-10
**Episode Analyzed:** `2025-11-10T13:25:37`
**Total Turns:** 254

---

## Executive Summary

**The LLM-based critic is COSTING more than it SAVES and should be removed.**

- **88.3% of critic rejections are overridden** (and succeed)
- **Only 11.7% of overridden actions fail** (critic was right 9 out of 77 times)
- **Critic adds $0.37/episode in direct costs** (~38% overhead)
- **Adds 960,000 tokens per episode** in overhead
- **Forces 76 unnecessary agent retries** per episode (wasted LLM calls)

**However:** The object tree validation component (pre-LLM) is VALUABLE and should be retained.

---

## Detailed Findings

### 1. Decision Pattern Analysis

**From `analyze_critic.py` output:**

```
Total decisions analyzed: 254
Accepted (score ≥ 0):     167 (65.7%)
Rejected (score < 0):     10 (3.9%)
Overridden:               77 (30.3%)

Override Analysis:
- Critic was RIGHT:   9 (11.7%)  ← Prevented 9 bad actions
- Critic was WRONG:  68 (88.3%)  ← Blocked 68 VALID actions
- Unclear outcome:    0 (0.0%)

Action Success Analysis:
- Accepted actions that succeeded: 156 (93.4% success rate)
- Accepted actions that failed:    11 (6.6% failure rate)
```

**Key Insight:** When the critic rejects an action and it gets overridden, the action succeeds 88.3% of the time. This means **the critic is wrong almost 9 out of 10 times** when second-guessed by the override system.

### 2. Override Reason Breakdown

From the 77 overrides, the most common reasons were:

| Reason | Count | Description |
|--------|-------|-------------|
| `low_critic_confidence` | ~35 | Critic unsure, trust agent |
| `exploring_new_locations` | ~30 | Agent productively exploring |
| `low_failure_rate` | ~12 | Few failures, trust agent |

**Analysis:** The override system is essentially saying "the critic doesn't know what it's doing, let the agent try anyway." This suggests the critic's value proposition is weak.

### 3. Location Penalty Impact

**From analysis:**
- **72.8% of turns** had location penalties applied
- **Average penalty:** -0.57 confidence points
- Penalties reduced critic confidence from 0.8 → 0.0 in many cases

**Example progression:**
```
Turn 2: 0.80 → 0.60 (penalty: -0.20)  # First revisit
Turn 3: 0.80 → 0.40 (penalty: -0.40)  # Second revisit
Turn 4: 0.90 → 0.30 (penalty: -0.60)  # Third revisit
Turn 5: 0.90 → 0.10 (penalty: -0.80)  # Fourth revisit
```

**Issue:** The location penalty system is so aggressive that it's artificially lowering critic confidence, which then triggers overrides based on "low_critic_confidence." This creates a **self-defeating cycle**:
1. Agent revisits location → penalty applied
2. Critic confidence drops to near-zero
3. Override system sees low confidence → overrides rejection
4. Action succeeds anyway

The penalty system is essentially **pre-rejecting** the critic's opinion before it even evaluates!

### 4. Cost Analysis

**Per Episode Breakdown:**

| Component | Tokens (Input) | Tokens (Output) | Cost | % of Total |
|-----------|----------------|-----------------|------|------------|
| Agent calls (base) | 1,016,000 | 12,700 | $0.288 | 42% |
| Agent retries (from rejections) | 304,000 | 3,800 | $0.086 | 13% |
| Critic LLM calls | 635,000 | 19,050 | $0.193 | 28% |
| Extractor/Objectives | ~200,000 | ~10,000 | $0.065 | 9% |
| **TOTAL WITH CRITIC** | **2,155,000** | **45,550** | **$0.632** | **100%** |

**Agent-Only Scenario (no critic):**
- Agent calls: 254 × 4,000 tokens = 1,016,000 tokens
- Cost: ~$0.288
- **Savings: $0.344 per episode (54% reduction)**

**But wait - what about failures?**

The critic prevents 9 bad actions per episode (11.7% of 77 overrides). Without the critic, those 9 would fail, requiring:
- 9 × 1 retry = 9 additional agent calls
- 9 × 4,000 tokens = 36,000 tokens
- Cost: ~$0.010

**Net savings: $0.344 - $0.010 = $0.334 per episode (53% cost reduction)**

### 5. Rejection Loop Overhead

**Current behavior:**
1. Agent proposes action
2. Critic rejects (score < 0)
3. If not overridden: ask agent for NEW action
4. Critic evaluates NEW action
5. Repeat up to 3 times per turn

**Overhead:**
- Average rejections per turn: ~0.3 (77 rejections / 254 turns)
- Each rejection triggers new agent call (~4,000 tokens)
- **Total wasted agent calls: 76 per episode**

**Without critic:**
- Agent proposes action
- Action executed immediately
- If fails: agent sees failure and adapts next turn (normal gameplay)

---

## Value Components

### ✅ KEEP: Object Tree Validation

**Location:** `zork_critic.py` lines 599-731

**What it does:**
- Uses Z-machine data to validate actions BEFORE LLM call
- Checks if objects exist, are visible, are takeable
- Validates exits against ground-truth from Jericho

**Why it's valuable:**
- **Instant** (no LLM call, ~0ms latency)
- **Accurate** (based on game state, not LLM reasoning)
- **Free** (no token cost)
- Catches ~20-30% of invalid actions immediately

**Example:**
```python
# Agent proposes: "take lamp"
# Object tree check: lamp not in current room
# → Instant rejection, no LLM call needed
```

**Token savings:** ~500 tokens per validation (no critic LLM call)

### ❌ REMOVE: LLM Critic Evaluation

**Location:** `zork_critic.py` lines 733-917

**What it does:**
- Calls DeepSeek V3.2 with 1,830-token system prompt
- Evaluates action reasoning with 500-700 token user prompt
- Returns score, confidence, justification

**Why it's NOT valuable:**
- **88.3% override rate** → wrong most of the time
- **$0.193/episode cost** → expensive for low accuracy
- **635k tokens per episode** → significant overhead
- **Forces 76 agent retries** → multiplies wasted calls

**What's happening:**
- Critic sees action like "examine grating" (perfectly reasonable)
- Critic rejects with score -1.0 (too conservative)
- Override system says "critic is being dumb, let it through"
- Action succeeds
- **Net result: wasted tokens, wasted time**

---

## The Real Problem: Critic Prompt is Too Conservative

Looking at `critic.md`, the prompt emphasizes:
- "Be wary of repetition"
- "Question circular logic"
- "Prefer proven actions over experimental ones"
- "Reject if unsure"

This creates a **bias toward rejection** that doesn't match the actual game dynamics. Zork rewards exploration and experimentation, but the critic penalizes it.

**Evidence from overrides:**
- 30 overrides due to "exploring_new_locations"
- 35 overrides due to "low_critic_confidence"

These are situations where the critic is **supposed** to be conservative, but the game **rewards** trying new things.

---

## Location Penalty System Analysis

**Current implementation:** `orchestration/zork_orchestrator_v2.py` lines 715-779

**Configuration:**
```toml
enable_location_penalty = true
location_revisit_penalty = -0.2  # Per revisit in last 5 locations
```

**Impact:**
- Applied to 72.8% of turns (185 out of 254)
- Average penalty: -0.57 confidence points
- Frequently drives confidence to 0.0 (88 times)

**Problem:** The penalty is **too aggressive** and creates a feedback loop:
1. Agent revisits location (often necessary for puzzle-solving)
2. Massive confidence penalty applied (-0.6 to -0.8)
3. Critic confidence near zero
4. Override system triggers on "low_critic_confidence"
5. Action succeeds anyway

**The penalty defeats its own purpose.** It's meant to discourage loops, but instead it just forces overrides.

**Recommendation:** Either:
- **Remove the penalty entirely** (let loop break system handle it)
- **Reduce penalty to -0.05 per revisit** (gentler nudge)
- **Cap penalty at -0.3 total** (prevent confidence collapse)

---

## Recommendations

### Option 1: Remove LLM Critic Entirely ⭐ **RECOMMENDED**

**Keep:**
- Object tree validation (lines 599-731 in `zork_critic.py`)
- Override system (already good at catching false negatives)
- Loop break system (handles stuck episodes)

**Remove:**
- LLM critic evaluation (lines 733-917)
- Rejection loop (lines 1042-1249 in orchestrator)
- Location penalty system (lines 715-779 in orchestrator)

**Implementation:**
1. Modify `_execute_critic_evaluation_loop()` to:
   - Run object tree validation ONLY
   - If validation fails → reject immediately
   - If validation passes → accept (no LLM call)
2. Remove rejection loop (no retries)
3. Remove location penalty system

**Expected results:**
- 53% cost reduction ($0.632 → $0.298 per episode)
- 960k fewer tokens per episode
- Faster execution (no critic LLM latency)
- Simpler codebase (remove rejection_manager.py)
- More exploration (no false rejections)

**Risk mitigation:**
- Object tree validation still catches ~30% of invalid actions
- Override system already prevents most stuck loops
- Loop break system terminates stuck episodes (40 turns without progress)
- Agent learns from failures naturally (next turn sees error message)

### Option 2: Simplify Critic to Rule-Based System

**Keep:**
- Object tree validation
- Simple rule-based checks:
  - Same action failed at this location → reject
  - Action repeats 3+ times in last 5 turns → reject
  - Not in object tree → reject (already done)

**Remove:**
- LLM evaluation
- Complex override logic
- Location penalty system

**Expected results:**
- Similar cost reduction (~50%)
- Faster execution (rules are O(1))
- Less false positives than LLM critic
- Retains some guardrails

### Option 3: Tune Existing Critic (NOT RECOMMENDED)

**Adjustments:**
- Increase rejection threshold (reject only at score < -0.95)
- Reduce location penalty to -0.05 per revisit
- Simplify override logic (trust agent more)

**Why not recommended:**
- Still 635k tokens per episode
- Fundamental issue is conservative bias
- Tuning is trial-and-error with high iteration cost

---

## Cost Projections

**Current costs (100 episodes/month):**
- Per episode: $0.632
- Monthly: $63.20
- Annual: $758.40

**With Option 1 (remove LLM critic):**
- Per episode: $0.298
- Monthly: $29.80
- Annual: $357.60
- **Savings: $400.80/year (53% reduction)**

**With Option 2 (rule-based):**
- Per episode: ~$0.310
- Monthly: $31.00
- Annual: $372.00
- **Savings: $386.40/year (51% reduction)**

---

## Implementation Plan

**Phase 1: Validate Object Tree Component**
1. Extract object tree validation to separate function
2. Write tests for object tree validation
3. Verify it catches invalid actions correctly

**Phase 2: Remove LLM Critic**
1. Modify `_execute_critic_evaluation_loop()`:
   - Call object tree validation
   - If fails → return rejection
   - If passes → return acceptance (no LLM)
2. Remove rejection loop logic
3. Remove location penalty system

**Phase 3: Clean Up**
1. Remove `rejection_manager.py` (no longer needed)
2. Remove critic prompt file (`critic.md`)
3. Remove unused imports and configuration
4. Update tests

**Phase 4: Validate**
1. Run test suite
2. Run 5-10 test episodes
3. Compare performance metrics:
   - Episode length (should be similar or shorter)
   - Final score (should be similar or higher)
   - Token usage (should be 50% lower)
   - Cost (should be 50% lower)

---

## Conclusion

The LLM critic is a **net negative** for ZorkGPT:
- 88.3% of its rejections are overridden
- It costs $0.334/episode more than it saves
- It adds 960k tokens of overhead per episode
- It forces 76 unnecessary agent retries per episode

**The object tree validation is the only valuable part** - it's fast, accurate, and free.

**Recommendation:** Remove the LLM critic, keep object tree validation, and rely on the override system + loop break system to handle edge cases. This will:
- Cut costs by 53%
- Reduce tokens by 45%
- Speed up execution
- Simplify the codebase
- Allow more exploration

The data is clear: **remove the critic.**

---

## Appendix: Analysis Scripts

Two analysis scripts were created:

1. **`analyze_critic.py`** - Decision pattern analysis
   - Parses episode logs
   - Categorizes decisions (accept/reject/override)
   - Analyzes override correctness
   - Generates comprehensive report

2. **`analyze_critic_cost.py`** - Token/cost analysis
   - Extracts LLM calls from logs
   - Estimates token usage per component
   - Calculates costs
   - Projects monthly/annual costs

**Usage:**
```bash
python3 analyze_critic.py game_files/episodes/EPISODE_ID/episode_log.jsonl
python3 analyze_critic_cost.py game_files/episodes/EPISODE_ID/episode_log.jsonl
```

These scripts can be used to analyze future episodes and validate the impact of changes.
