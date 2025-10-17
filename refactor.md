# Jericho Integration Refactoring Plan

**Document Version**: 1.0
**Date**: 2025-01-17
**Status**: Phase 3 Ready to Start

## Executive Summary

This plan implements **complete migration from dfrotz to Jericho** with zero backwards compatibility. We eliminate ~660-690 lines of brittle text parsing and map consolidation code (11% of codebase) while gaining:

- ✅ **Phase 1 COMPLETE**: JerichoInterface replaces dfrotz entirely
- ✅ **Phase 2 COMPLETE**: Direct Z-machine access for inventory, location, objects, score
- ✅ **Phase 3 COMPLETE**: Integer-based map with stable location IDs (~317 lines to delete)
- 📋 **Phase 4-7**: Movement detection, context enhancement, knowledge tracking, testing

**Key Metrics**:
- Code reduction: ~660-690 lines (11% of codebase)
- LLM call reduction: ~40% (instant Z-machine data access)
- Zero room fragmentation: Location IDs guarantee uniqueness
- Perfect movement detection: ID comparison replaces heuristics

---

## Phase Status Report

### ✅ Phase 1: Foundation (COMPLETE)

**Implemented**:
- ✅ JerichoInterface fully implemented at `game_interface/core/jericho_interface.py`
- ✅ Orchestrator uses JerichoInterface directly (line 29, 78-80 of `zork_orchestrator_v2.py`)
- ✅ dfrotz completely removed from codebase
- ✅ Tests exist: `tests/test_jericho_interface.py`, `tests/test_jericho_interface_session_methods.py`
- ✅ Session methods implemented: `trigger_zork_save()`, `trigger_zork_restore()`, `is_game_over()`

**Verification**:
```bash
# Confirm no dfrotz references remain
grep -r "dfrotz" --exclude-dir=".git" --exclude="*.md" --exclude="repomix-output.xml"
# Should only find references in documentation/history
```

---

### ✅ Phase 2: Extractor - Direct Z-Machine Access (COMPLETE)

**Implemented**:
- ✅ `_get_location_from_jericho()` at line 248 of `hybrid_zork_extractor.py`
- ✅ `_get_inventory_from_jericho()` at line 261
- ✅ `_get_visible_objects_from_jericho()` at line 274
- ✅ `_get_visible_characters_from_jericho()` at line 306
- ✅ `_get_score_from_jericho()` at line 350
- ✅ LLM only used for exits, combat, and important messages (line 364)

**Results**:
- Zero regex parsing for inventory/location/score
- Extractor response includes structured Jericho data
- ~100 lines of regex parsing eliminated

**Verification**:
```python
# Confirm extractor uses only Jericho methods
grep -n "get_.*_from_jericho" hybrid_zork_extractor.py
```

---

## 🔜 Phase 3: Map Intelligence - Location ID Migration (READY TO START)

**Status**: NOT STARTED
**Estimated Impact**: ~317 lines deleted, ~150 lines modified
**Risk Level**: MEDIUM (core data structure change)

### Current State Analysis

**Files Requiring Changes**:
1. `session/game_state.py` (line 26): `current_room_name_for_map: str` → needs ID
2. `map_graph.py` (line 219): `self.rooms: Dict[str, Room]` → needs `Dict[int, Room]`
3. `map_graph.py` (lines 1028-1135): `_create_unique_location_id()` → DELETE (107 lines)
4. `map_graph.py` (lines 1363-1499): `consolidate_base_name_variants()` → DELETE (136 lines)
5. `map_graph.py` (lines 1501-1575): `_choose_best_base_name_variant()` → DELETE (74 lines)

**Total Code to Delete**: ~317 lines of consolidation logic

### Phase 3 Implementation Plan

#### 3.1: Update GameState Data Model

**File**: `session/game_state.py`

**Changes**:
```python
# BEFORE (line 26):
current_room_name_for_map: str = ""

# AFTER:
current_room_id: int = 0  # Primary identifier for map operations
current_room_name: str = ""  # Display name only
```

**Additional Updates**:
- Line 141: Update export_data to include both ID and name
- Update all docstrings mentioning room names

**Validation**:
- All tests pass after data model change
- State export includes both `current_room_id` and `current_room_name`

---

#### 3.2: Refactor Room Class for Integer IDs

**File**: `map_graph.py`

**Changes to Room class** (line 201-213):
```python
# BEFORE:
class Room:
    def __init__(self, name: str, base_name: str = None):
        self.name: str = name
        self.base_name: str = base_name or name
        self.exits: Set[str] = set()

# AFTER:
class Room:
    def __init__(self, room_id: int, name: str):
        self.id: int = room_id  # PRIMARY KEY - Z-machine object ID
        self.name: str = name    # Display name only
        self.exits: Set[str] = set()

    def __repr__(self) -> str:
        return f"Room(id={self.id}, name='{self.name}', exits={self.exits})"
```

**Rationale**: Eliminate base_name complexity, use Z-machine IDs directly

---

#### 3.3: Refactor MapGraph to Integer Keys

**File**: `map_graph.py`

**Changes to MapGraph.__init__** (line 217-233):
```python
# BEFORE (line 219):
self.rooms: Dict[str, Room] = {}

# AFTER:
self.rooms: Dict[int, Room] = {}  # Integer keys = location IDs
self.room_names: Dict[int, str] = {}  # ID -> name mapping for display
```

**Update all MapGraph methods**:

1. **`add_room()`** (line 361-368):
```python
# BEFORE:
def add_room(self, room_name: str, base_name: str = None) -> Room:
    room_key = room_name
    if room_key not in self.rooms:
        self.rooms[room_key] = Room(name=room_key, base_name=base_name)
        self.has_new_rooms_since_consolidation = True
    return self.rooms[room_key]

# AFTER:
def add_room(self, room_id: int, room_name: str) -> Room:
    if room_id not in self.rooms:
        self.rooms[room_id] = Room(room_id=room_id, name=room_name)
        self.room_names[room_id] = room_name
    return self.rooms[room_id]
```

2. **`add_connection()`** (line 448-612):
```python
# BEFORE signature:
def add_connection(self, from_room_name: str, exit_taken: str, to_room_name: str, confidence: float = 1.0)

# AFTER signature:
def add_connection(self, from_room_id: int, exit_taken: str, to_room_id: int, confidence: float = 1.0)

# Implementation changes:
# - Replace all string-based room lookups with integer lookups
# - Update self.connections to use integer keys
# - Update confidence tracking to use integer keys
```

3. **`update_room_exits()`** (line 370-410):
```python
# BEFORE:
def update_room_exits(self, room_name: str, new_exits: List[str]):

# AFTER:
def update_room_exits(self, room_id: int, new_exits: List[str]):
    if room_id not in self.rooms:
        # Cannot add exits for non-existent room
        return
    # Rest of logic remains the same, just use room_id instead of room_name
```

4. **`get_room_info()`** (line 614-650):
```python
# BEFORE:
def get_room_info(self, room_name: str) -> str:
    normalized_name = self._normalize_room_name(room_name)
    if normalized_name not in self.rooms:
        return f"Room '{room_name}' is unknown."

# AFTER:
def get_room_info(self, room_id: int) -> str:
    if room_id not in self.rooms:
        return f"Room ID {room_id} is unknown."

    room = self.rooms[room_id]
    # Build info string using room.id and room.name
```

5. **`get_context_for_prompt()`** (line 652-740):
```python
# BEFORE:
def get_context_for_prompt(
    self,
    current_room_name: str,
    previous_room_name: str = None,
    action_taken_to_current: str = None,
) -> str:

# AFTER:
def get_context_for_prompt(
    self,
    current_room_id: int,
    current_room_name: str,  # For display only
    previous_room_id: int = None,
    previous_room_name: str = None,  # For display only
    action_taken_to_current: str = None,
) -> str:
    # Use IDs for all lookups, names only for display strings
```

---

#### 3.4: DELETE Consolidation Methods

**File**: `map_graph.py`

**DELETE ENTIRELY** (no replacement needed):

1. **`_create_unique_location_id()`** (lines 1028-1135) - **107 lines**
   - Reason: Z-machine provides stable IDs, no need to generate unique identifiers

2. **`consolidate_base_name_variants()`** (lines 1363-1499) - **136 lines**
   - Reason: Integer IDs prevent fragmentation, consolidation not needed

3. **`_choose_best_base_name_variant()`** (lines 1501-1575) - **74 lines**
   - Reason: Supporting method for consolidation, no longer needed

4. **`needs_consolidation()`** (lines 1137-1166) - **29 lines**
   - Reason: Consolidation logic removed

5. **`consolidate_similar_locations()`** (lines 1168-1291) - **123 lines**
   - Reason: Case variation consolidation not needed with integer IDs

6. **`_choose_best_variant()`** (lines 1293-1336) - **43 lines**
   - Reason: Supporting method for consolidation

**Total Deleted**: ~512 lines (more than initially estimated!)

**Methods to KEEP** (still useful):
- `_get_opposite_direction()` (line 235) - Still useful for reverse connections
- `normalize_direction()` (line 174) - Still useful for exit normalization
- `get_navigation_suggestions()` (line 988) - Useful with integer-based lookups
- `render_ascii()` (line 742) - Visualization still needed
- `render_mermaid()` (line 797) - Visualization still needed

---

#### 3.5: Update get_or_create_node_id Logic

**File**: `map_graph.py`

**REPLACE** `get_or_create_node_id()` (lines 283-359):

```python
# BEFORE (77 lines of complex exit compatibility logic):
def get_or_create_node_id(
    self, base_location_name: str, current_exits: List[str], description: str = ""
) -> str:
    # ... 77 lines of string-based compatibility checking ...

# AFTER (5-10 lines - trivial ID pass-through):
def get_or_create_room(self, location_id: int, location_name: str) -> int:
    """
    Get or create a room using Z-machine location ID.

    With Jericho, location IDs come from the Z-machine and are guaranteed
    unique. This method simply ensures the room exists in our graph.

    Args:
        location_id: Z-machine object ID (from location.num)
        location_name: Display name (from location.name)

    Returns:
        The location ID (same as input)
    """
    if location_id not in self.rooms:
        self.add_room(location_id, location_name)
    return location_id
```

**Impact**: Eliminates 77 lines of complex exit compatibility logic

---

#### 3.6: Update MapManager Integration

**File**: `managers/map_manager.py`

**Key Changes**:

1. Update `update_from_movement()` to accept and use location IDs
2. Update `add_initial_room()` to use location ID
3. Update all calls to `game_map` methods to use IDs

**Example**:
```python
# BEFORE:
def update_from_movement(
    self,
    action_taken: str,
    new_room_name: str,
    previous_room_name: str,
    game_response: str
):
    # ... uses room names as keys ...

# AFTER:
def update_from_movement(
    self,
    action_taken: str,
    new_room_id: int,
    new_room_name: str,
    previous_room_id: int,
    previous_room_name: str,
    game_response: str
):
    # ... uses room IDs as keys ...
```

---

#### 3.7: Update Orchestrator to Extract Location IDs

**File**: `orchestration/zork_orchestrator_v2.py`

**Changes in `_process_extraction()`** (line 621):

```python
# BEFORE (line 639-658):
if (
    hasattr(extracted_info, "current_location_name")
    and extracted_info.current_location_name
):
    new_location = extracted_info.current_location_name
    self.game_state.visited_locations.add(new_location)

    if action and self.game_state.current_room_name_for_map:
        self.map_manager.update_from_movement(
            action_taken=action,
            new_room_name=new_location,
            previous_room_name=self.game_state.current_room_name_for_map,
            game_response=response,
        )

# AFTER:
if (
    hasattr(extracted_info, "current_location_name")
    and extracted_info.current_location_name
):
    # Get location ID from Jericho
    location_obj = self.jericho_interface.get_location_structured()
    new_location_id = location_obj.num
    new_location_name = location_obj.name

    # Update game state with both ID and name
    self.game_state.current_room_id = new_location_id
    self.game_state.current_room_name = new_location_name
    self.game_state.visited_locations.add(new_location_name)

    # Update map using location IDs
    if action and self.game_state.current_room_id != 0:
        self.map_manager.update_from_movement(
            action_taken=action,
            new_room_id=new_location_id,
            new_room_name=new_location_name,
            previous_room_id=self.game_state.current_room_id,
            previous_room_name=self.game_state.current_room_name,
            game_response=response,
        )
    elif self.game_state.current_room_id == 0:
        # Initial room
        self.map_manager.add_initial_room(new_location_id, new_location_name)
```

---

#### 3.8: Update Tests

**Files**: `tests/test_map_graph.py`, `tests/test_integration.py`

**Changes**:
- Update all test assertions to use integer IDs
- Update test fixtures to use integer IDs
- Remove tests for consolidation methods (no longer exist)
- Add new tests for ID-based room operations

**New Test Cases**:
```python
def test_map_handles_multiple_rooms_same_name():
    """Test that rooms with same name but different IDs are distinct."""
    map_graph = MapGraph()

    # Two clearings with same name, different IDs
    map_graph.add_room(42, "Clearing")
    map_graph.add_room(78, "Clearing")

    assert len(map_graph.rooms) == 2
    assert 42 in map_graph.rooms
    assert 78 in map_graph.rooms
    assert map_graph.rooms[42].name == "Clearing"
    assert map_graph.rooms[78].name == "Clearing"
```

---

### Phase 3 Deliverables

**Code Changes**:
- ✅ GameState uses `current_room_id: int` as primary key
- ✅ MapGraph uses `Dict[int, Room]` with integer keys
- ✅ Room class uses `id: int` as primary key
- ✅ ~512 lines of consolidation code DELETED
- ✅ MapManager integration updated for IDs
- ✅ Orchestrator extracts location IDs from Jericho

**Tests**:
- ✅ All existing tests updated for integer IDs
- ✅ New tests for ID-based operations
- ✅ Test for multiple rooms with same name (different IDs)
- ✅ No tests for deleted consolidation methods

**Validation Criteria**:
```bash
# 1. No consolidation code remains
grep -n "consolidate" map_graph.py
# Should return zero results (or only comments)

# 2. No _create_unique_location_id remains
grep -n "_create_unique_location_id" map_graph.py
# Should return zero results

# 3. Map uses integer keys
grep -n "self.rooms: Dict\[str" map_graph.py
# Should return zero results

# 4. GameState uses room ID
grep -n "current_room_id" session/game_state.py
# Should show new field definition

# 5. All tests pass
uv run pytest tests/ -v
```

**Success Metrics**:
- Zero room fragmentation (guaranteed by Z-machine IDs)
- ~512 lines deleted (11% of map_graph.py)
- All tests passing with ID-based operations
- Map correctly handles 5 rooms named "Forest" with distinct IDs

---

## 📋 Phase 4: Movement - Perfect Detection via ID Comparison

**Status**: NOT STARTED
**Estimated Impact**: ~150 lines deleted/simplified
**Dependencies**: Phase 3 complete (location IDs available)

### Current State

**File**: `movement_analyzer.py`

**Code to Simplify/Delete**:
- `PendingConnection` class (lines 49-69) - **20 lines** → DELETE
- `_indicates_pending_movement()` (lines 251-287) - **36 lines** → DELETE
- `_should_resolve_pending()` (lines 223-243) - **20 lines** → DELETE
- `_check_pending_resolution()` (lines 106-129) - **23 lines** → DELETE
- Dark room heuristics → REPLACE with ID comparison

**Total Simplification**: ~100 lines deleted, ~50 lines simplified

### Implementation Plan

#### 4.1: Simplify MovementResult

**File**: `movement_analyzer.py`

```python
# BEFORE (lines 28-46):
@dataclass
class MovementResult:
    movement_occurred: bool
    from_location: Optional[str]
    to_location: Optional[str]
    action: str
    is_pending: bool
    environmental_factors: List[str]
    requires_resolution: bool
    connection_created: bool = False
    from_description: str = ""
    from_objects: List[str] = None
    from_exits: List[str] = None
    to_description: str = ""
    to_objects: List[str] = None
    to_exits: List[str] = None

# AFTER:
@dataclass
class MovementResult:
    movement_occurred: bool
    from_location_id: int
    to_location_id: int
    action: str
    # All pending-related fields REMOVED
    # All description/object fields REMOVED (not needed with IDs)
```

---

#### 4.2: Simplify MovementAnalyzer

**File**: `movement_analyzer.py`

**DELETE**:
- `pending_connections` list (line 84)
- `max_pending_turns` (line 85)
- `_check_pending_resolution()` method (lines 106-129)
- `_should_resolve_pending()` method (lines 223-243)
- `_indicates_pending_movement()` method (lines 251-287)
- `add_intermediate_action_to_pending()` method (lines 367-371)
- `cleanup_expired_pending()` method (lines 373-385)
- `get_pending_connections()` method (lines 387-389)
- `has_pending_connections()` method (lines 391-393)
- `clear_pending_connections()` method (lines 395-397)

**REPLACE** `analyze_movement()` (lines 87-104):

```python
# BEFORE (complex pending connection logic):
def analyze_movement(self, context: MovementContext) -> MovementResult:
    # First, check if this resolves any pending connections
    resolved_connection = self._check_pending_resolution(context)
    if resolved_connection:
        return resolved_connection

    # Then check if this creates a new movement
    return self._analyze_new_movement(context)

# AFTER (simple ID comparison):
def analyze_movement(
    self,
    before_location_id: int,
    after_location_id: int,
    action: str
) -> MovementResult:
    """
    Analyze movement by comparing location IDs.

    With Jericho, movement detection is trivial: if the location ID
    changed, movement occurred. No heuristics needed.

    Args:
        before_location_id: Location ID before action
        after_location_id: Location ID after action
        action: Action taken

    Returns:
        MovementResult indicating if movement occurred
    """
    movement_occurred = (before_location_id != after_location_id)

    return MovementResult(
        movement_occurred=movement_occurred,
        from_location_id=before_location_id,
        to_location_id=after_location_id,
        action=action,
    )
```

**Impact**: Replaces ~300 lines of heuristics with ~20 lines of ID comparison

---

#### 4.3: Update MapManager Integration

**File**: `managers/map_manager.py`

**Changes**:
```python
# BEFORE:
def update_from_movement(self, ...):
    # Complex movement analysis with text comparison
    movement_result = self.movement_analyzer.analyze_movement(context)
    if movement_result.is_pending:
        # Handle pending connections...

# AFTER:
def update_from_movement(
    self,
    action_taken: str,
    before_location_id: int,
    after_location_id: int,
    new_room_name: str,
    previous_room_name: str,
):
    # Simple ID comparison
    if before_location_id != after_location_id:
        # Movement occurred - add connection
        self.game_map.add_connection(
            before_location_id,
            action_taken,
            after_location_id
        )
```

---

#### 4.4: Update Orchestrator

**File**: `orchestration/zork_orchestrator_v2.py`

**Changes in `_process_extraction()`**:

```python
# Get location ID before action
before_location_id = self.game_state.current_room_id

# Execute action
next_game_state = self.jericho_interface.send_command(action_to_take)

# Get location ID after action
location_obj = self.jericho_interface.get_location_structured()
after_location_id = location_obj.num

# Detect movement via ID comparison
if before_location_id != after_location_id:
    # Movement occurred
    self.map_manager.update_from_movement(
        action_taken=action_to_take,
        before_location_id=before_location_id,
        after_location_id=after_location_id,
        new_room_name=location_obj.name,
        previous_room_name=self.game_state.current_room_name,
    )
```

---

### Phase 4 Deliverables

**Code Changes**:
- ✅ PendingConnection class DELETED
- ✅ All pending connection methods DELETED (~100 lines)
- ✅ Movement detection simplified to ID comparison (~20 lines)
- ✅ Dark room handling automatic (IDs work regardless of visibility)
- ✅ MapManager uses simple ID comparison
- ✅ Orchestrator tracks before/after location IDs

**Tests**:
- ✅ Test movement detection in dark rooms
- ✅ Test movement detection with same-named rooms
- ✅ Test no-movement actions (ID stays same)
- ✅ Remove all pending connection tests

**Validation Criteria**:
```bash
# 1. No pending connection code remains
grep -n "PendingConnection" movement_analyzer.py
# Should return zero results

# 2. No dark room heuristics remain
grep -n "_indicates_pending" movement_analyzer.py
# Should return zero results

# 3. Movement detection is simple
wc -l movement_analyzer.py
# Should be ~200 lines (down from ~400)
```

**Success Metrics**:
- Movement correctly detected in pitch dark rooms
- Zero false positives/negatives
- No pending connections needed
- ~150-200 lines deleted from movement_analyzer.py

---

## 📋 Phase 5: Enhanced Context - Object Tree Integration

**Status**: NOT STARTED
**Estimated Impact**: ~100 lines added (context enhancement)
**Dependencies**: Phases 3-4 complete

### Goal

Provide Agent and Critic with direct Z-machine object data for better reasoning.

### Implementation Plan

#### 5.1: Enhance Agent Context

**File**: `managers/context_manager.py`

**Add structured world snapshot**:
```python
def get_agent_context(self, ...):
    # ... existing context ...

    # NEW: Add structured Z-machine data
    context["current_location_id"] = current_room_id
    context["inventory_objects"] = [
        {
            "id": obj.num,
            "name": obj.name,
            "attributes": self._get_object_attributes(obj)
        }
        for obj in jericho_interface.get_inventory_structured()
    ]
    context["visible_objects"] = [
        {
            "id": obj.num,
            "name": obj.name,
            "attributes": self._get_object_attributes(obj)
        }
        for obj in self._get_visible_objects()
    ]
    context["action_vocabulary"] = self._get_action_vocabulary()

    return context
```

---

#### 5.2: Enhance Critic Validation

**File**: `zork_critic.py`

**Add object tree validation**:
```python
def evaluate_action(self, proposed_action, ...):
    # ... existing evaluation ...

    # NEW: Validate against Z-machine object tree
    validation_result = self._validate_against_object_tree(
        proposed_action,
        jericho_interface
    )

    if not validation_result.valid:
        return CriticResult(
            score=0.2,
            justification=validation_result.reason,
            confidence=0.9  # High confidence in Z-machine data
        )
```

---

#### 5.3: Add Object Attribute Helper

**File**: `game_interface/core/jericho_interface.py`

**Add utility method**:
```python
def get_object_attributes(self, obj) -> Dict[str, bool]:
    """
    Extract useful attributes from a ZObject.

    Returns:
        Dictionary of attribute flags (takeable, openable, etc.)
    """
    # Z-machine attribute bits (game-specific)
    return {
        "takeable": self._check_attribute(obj, 10),  # Example bit
        "openable": self._check_attribute(obj, 12),
        "container": self._check_attribute(obj, 5),
        # ... more as needed
    }

def _check_attribute(self, obj, bit: int) -> bool:
    """Check if a specific attribute bit is set."""
    if len(obj.attr) > bit // 8:
        byte_index = bit // 8
        bit_index = bit % 8
        return bool(obj.attr[byte_index] & (1 << bit_index))
    return False
```

---

### Phase 5 Deliverables

**Code Changes**:
- ✅ Agent receives structured world snapshot
- ✅ Critic validates against object tree
- ✅ Action vocabulary included in context
- ✅ Object attributes extracted from Z-machine

**Tests**:
- ✅ Test agent context includes object IDs
- ✅ Test critic rejects "take lamp" when lamp not in room
- ✅ Test object attribute extraction

**Success Metrics**:
- Agent sees takeable items with attributes
- Critic precisely validates object presence
- Fewer "I don't see that here" failures

---

## 📋 Phase 6: Knowledge & State - Object Tracking

**Status**: NOT STARTED
**Estimated Impact**: ~50 lines added
**Dependencies**: Phases 3-5 complete

### Goal

Track object state changes and detect exact game state loops.

### Implementation Plan

#### 6.1: Add State Hash Tracking

**File**: `managers/state_manager.py`

```python
def process_turn(self):
    # ... existing logic ...

    # NEW: Track state hash
    state_hash = self.jericho_interface.env.get_world_state_hash()

    if state_hash in self.state_history:
        self.logger.warning(
            "Exact game state loop detected",
            extra={"state_hash": state_hash}
        )
        # Trigger intervention logic

    self.state_history.append(state_hash)
```

---

#### 6.2: Track Object Events

**File**: `managers/knowledge_manager.py`

```python
def track_object_event(self, event_type: str, obj_id: int, obj_name: str, turn: int):
    """
    Track object-related events for knowledge synthesis.

    Args:
        event_type: "acquired", "dropped", "opened", "relocated"
        obj_id: Z-machine object ID
        obj_name: Object name
        turn: Turn number
    """
    event = {
        "turn": turn,
        "event_type": event_type,
        "object_id": obj_id,
        "object_name": obj_name,
        "timestamp": datetime.now().isoformat()
    }

    self.object_events.append(event)
```

---

### Phase 6 Deliverables

**Code Changes**:
- ✅ State hash tracking implemented
- ✅ Object event tracking implemented
- ✅ Loop detection logic added

**Success Metrics**:
- Detects exact state loops
- Tracks object acquisitions/relocations
- Knowledge synthesis includes object events

---

## 📋 Phase 7: Testing, Documentation & Deployment

**Status**: NOT STARTED
**Estimated Impact**: Comprehensive testing and validation
**Dependencies**: Phases 3-6 complete

### Implementation Plan

#### 7.1: Integration Tests

**New Test File**: `tests/test_jericho_integration.py`

```python
def test_full_game_session_with_location_ids():
    """Test complete game session using location IDs."""
    # ... 100-turn game session
    # Verify map uses integer IDs
    # Verify no room fragmentation
    # Verify movement detection works

def test_multiple_rooms_same_name():
    """Test system handles duplicate room names with different IDs."""
    # ... create 5 rooms named "Forest"
    # Verify all are distinct in map
    # Verify connections correct

def test_dark_room_movement():
    """Test movement detection in pitch dark rooms."""
    # ... move in dark
    # Verify movement detected via ID change
    # Verify connection created
```

---

#### 7.2: Performance Benchmarking

**Script**: `benchmarks/jericho_performance.py`

```python
def benchmark_llm_call_reduction():
    """Measure LLM call reduction with Jericho."""
    # Run same episode with old vs new system
    # Count LLM calls
    # Verify ~40% reduction

def benchmark_turn_processing_speed():
    """Measure turn processing speed improvement."""
    # Time 100 turns
    # Compare before/after Jericho
    # Verify faster processing
```

---

#### 7.3: Documentation Updates

**Files to Update**:
- `README.md`: Update architecture section
- `CLAUDE.md`: Add Jericho architecture notes
- `game_interface/README.md`: Document Jericho interface
- `jericho/README.md`: Keep as reference

---

#### 7.4: Code Cleanup

**Delete**:
- Any remaining dfrotz references
- Old consolidation tests
- Legacy text parsing code

**Verify**:
```bash
# No dfrotz remaining
grep -r "dfrotz" --exclude-dir=".git" --exclude="*.md"

# No consolidation code remaining
grep -r "consolidate" map_graph.py

# No pending connection code remaining
grep -r "PendingConnection" movement_analyzer.py
```

---

### Phase 7 Deliverables

**Tests**:
- ✅ Full integration test suite
- ✅ Performance benchmarks
- ✅ Dark room movement tests
- ✅ Duplicate room name tests

**Documentation**:
- ✅ Updated README
- ✅ Updated CLAUDE.md
- ✅ Jericho architecture documented

**Validation**:
- ✅ All tests pass
- ✅ ~40% LLM reduction achieved
- ✅ ~660 lines deleted total
- ✅ Zero dfrotz artifacts

---

## Success Criteria

### Overall Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| Code Reduction | ~660-690 lines | `git diff --stat` |
| LLM Call Reduction | ~40% | Benchmark script |
| Room Fragmentation | Zero | Test suite |
| Test Pass Rate | 100% | `pytest tests/` |
| Movement Detection | 100% accuracy | Integration tests |

### Per-Phase Success

- ✅ **Phase 1**: JerichoInterface only, zero dfrotz
- ✅ **Phase 2**: Zero regex parsing for core data
- 🔜 **Phase 3**: Integer-based map, ~512 lines deleted
- 📋 **Phase 4**: ~150 lines deleted, perfect movement
- 📋 **Phase 5**: Agent/Critic use object tree
- 📋 **Phase 6**: Object tracking, loop detection
- 📋 **Phase 7**: All tests pass, docs updated

---

## Implementation Strategy

### Execution Per Phase

1. **Plan Review**: Review this plan with `@agent-python-engineer`
2. **Implementation**: Execute phase with `@agent-python-engineer`
3. **Code Review**: Review with `@agent-code-reviewer`
4. **Fix Issues**: Address any issues with `@agent-python-engineer`
5. **Validate**: Run tests, verify success criteria
6. **Next Phase**: Only proceed after current phase validated

### Validation Before Next Phase

**Required for Phase 3 → Phase 4**:
```bash
# All Phase 3 validations pass
uv run pytest tests/test_map_graph.py -v
uv run pytest tests/test_integration.py -v
grep -c "consolidate" map_graph.py  # Should be 0
grep -c "Dict\[str, Room\]" map_graph.py  # Should be 0
```

**Required for Phase 4 → Phase 5**:
```bash
# All Phase 4 validations pass
uv run pytest tests/test_movement_analyzer.py -v
grep -c "PendingConnection" movement_analyzer.py  # Should be 0
wc -l movement_analyzer.py  # Should be ~200 lines
```

---

## Risk Management

### Medium Risks

**Risk**: Breaking existing map functionality during ID migration
**Mitigation**:
- Comprehensive test coverage before changes
- Incremental updates with validation at each step
- Keep git commits small and reversible

**Risk**: Missing edge cases in movement detection
**Mitigation**:
- Test dark rooms, teleportation, same-named rooms
- Add integration tests covering 100+ turns

### Low Risks

**Risk**: Object attribute extraction complexity
**Mitigation**:
- Start with basic attributes only
- Expand as needed based on actual requirements

---

## Current Status: Phase 3 Ready

**Next Steps**:
1. Review this plan with Ryan
2. Execute Phase 3.1 (Update GameState)
3. Execute Phase 3.2 (Update Room class)
4. Execute Phase 3.3 (Update MapGraph)
5. Execute Phase 3.4 (DELETE consolidation code)
6. Execute Phase 3.5-3.8 (Integration updates)
7. Validate Phase 3 complete before Phase 4

**Estimated Timeline**:
- Phase 3: 3-4 hours (data model changes + deletions)
- Phase 4: 2-3 hours (movement simplification)
- Phase 5: 2-3 hours (context enhancement)
- Phase 6: 1-2 hours (object tracking)
- Phase 7: 2-3 hours (testing & docs)
- **Total**: 10-15 hours

**Ready to proceed with Phase 3?**
