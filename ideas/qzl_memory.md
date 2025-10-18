# Z-Machine Memory Extraction via Quetzal Save Files

## Executive Summary

This document outlines a feasible approach to extract structured game metadata (current room, inventory, score) directly from Z-machine memory by parsing Quetzal save files, rather than relying solely on text parsing. This will make ZorkGPT more reliable by separating narrative text from structured state information.

**Status:** Research complete, ready for implementation
**Recommendation:** Parse Quetzal (.qzl) save files after each turn
**Use Case:** Augment text parsing - narrative via text channel, metadata via memory inspection

---

## Background: Why Memory Extraction?

### Current System
- dfrotz outputs text responses to commands
- Text parsing extracts room, inventory, score from natural language
- Prone to ambiguity, special cases, and parsing failures

### Proposed Enhancement
- Continue using text for narrative/story content
- Extract structured metadata out-of-band from memory
- Canonical, unambiguous game state
- Parse at end of every turn using existing save files

---

## Z-Machine Memory Architecture

### Memory Layout (Version 3 - Zork I)

The Z-machine stores game state in a byte-addressable memory array with specific regions:

```
Memory Map:
0x0000 - 0x001F: Header (contains pointers to other structures)
0x0020+       : Dynamic memory (modifiable game state)
              : Static memory (read-only data)
              : High memory (program code)
```

### Header Structure (Key Fields for Version 3)

| Offset | Size | Field | Purpose |
|--------|------|-------|---------|
| 0x00 | 1 byte | Version | Z-machine version (3 for Zork I) |
| 0x0A | 2 bytes | Object table address | Pointer to object table start |
| 0x0C | 2 bytes | Global variables | Pointer to global variable table |
| 0x0E | 2 bytes | Static memory base | Start of read-only memory |

All multi-byte values are **big-endian**.

### Object Table Structure

The object table contains all game entities (rooms, items, NPCs). For Version 3:

**Table Layout:**
```
[Property Defaults: 31 words = 62 bytes]
[Object 1: 9 bytes]
[Object 2: 9 bytes]
...
[Object 255: 9 bytes]
```

**Object Entry Format (9 bytes):**
```
Byte 0-3:   Attributes (32 boolean flags)
Byte 4:     Parent object number
Byte 5:     Sibling object number
Byte 6:     Child object number
Byte 7-8:   Properties pointer (big-endian word)
```

### Object Relationships (The Key Insight!)

Objects form a tree using parent/sibling/child pointers:

- **Parent:** What contains this object
- **Sibling:** Next object in same container
- **Child:** First object contained by this object

**Example: Player in "West of House" with lamp and sword**
```
Player Object (#20):
  Parent: 68 (West of House room)
  Sibling: 0 (no sibling)
  Child: 39 (lamp)

Lamp Object (#39):
  Parent: 20 (player)
  Sibling: 45 (sword)
  Child: 0

Sword Object (#45):
  Parent: 20 (player)
  Sibling: 0
  Child: 0
```

**To extract current room:** Read player object's parent byte
**To extract inventory:** Follow player object's child → sibling → sibling chain

### Object Properties (For Names)

Each object has a properties pointer leading to:

```
[Text-length byte: high 7 bits = word count]
[Short name: ZSCII encoded text]
[Property entries...]
```

The short name is what we want to extract (e.g., "West of House", "brass lantern").

---

## Quetzal Save File Format

Quetzal is the standardized IFF (Interchange File Format) for Z-machine saved games.

### IFF Structure

```
'FORM' <length> 'IFZS'
    'IFhd' <length> [header chunk data]
    'CMem' <length> [compressed memory data]
    'Stks' <length> [stack data]
    ['IntD' <length> [interpreter data]]
```

Each chunk has:
- 4-byte ID (e.g., 'CMem')
- 4-byte big-endian length
- N bytes of data
- Optional padding byte if length is odd

### Required Chunks

#### IFhd - Story File Identification (13 bytes)
```
Bytes 0-1:  Release number
Bytes 2-7:  Serial number (6 ASCII chars)
Bytes 8-9:  Checksum
Bytes 10:   Program counter (high byte)
Bytes 11-12: Program counter (low word)
```

#### CMem - Compressed Memory
**Most important chunk for our purposes!**

Contains the current state of dynamic memory, compressed via:
1. **XOR compression:** Current memory ⊕ Original story file memory
2. **Run-length encoding:**
   - Non-zero byte → literal byte value
   - Zero byte followed by N → (N+1) zero bytes

**Decompression Algorithm:**
```python
def decompress_cmem(compressed_data, original_memory):
    # Step 1: RLE decode
    decoded = []
    i = 0
    while i < len(compressed_data):
        if compressed_data[i] == 0:
            # Zero followed by count
            count = compressed_data[i + 1] + 1
            decoded.extend([0] * count)
            i += 2
        else:
            decoded.append(compressed_data[i])
            i += 1

    # Step 2: XOR with original
    result = bytes(a ^ b for a, b in zip(decoded, original_memory))
    return result
```

#### UMem - Uncompressed Memory (Alternative)
Some interpreters use uncompressed memory instead of CMem. This is just a direct dump of dynamic memory (simpler but larger files).

#### Stks - Stack Data
Contains the Z-machine call stack and evaluation stack. Not needed for room/inventory extraction.

---

## Evaluated Approaches

### ✅ Approach 1: Quetzal Save File Parsing (RECOMMENDED)

**How it works:**
1. Read existing `.qzl` save files (already generated after each turn)
2. Parse IFF/FORM structure to extract CMem or UMem chunk
3. If CMem: decompress using XOR + RLE algorithm
4. Read object table from decompressed memory
5. Find player object (search for object with specific attributes/properties)
6. Extract room (parent) and inventory (child chain)
7. Decode object names from properties (ZSCII → UTF-8)

**Pros:**
- ✅ No modifications to dfrotz needed
- ✅ Works with existing save file infrastructure
- ✅ Save files accessible via Docker volume mount (`./game_files`)
- ✅ Well-documented Quetzal format (official standard)
- ✅ Canonical game state - no parsing ambiguity
- ✅ Can be done entirely in Python

**Cons:**
- ⚠️ Moderate complexity (IFF parsing, RLE decompression, ZSCII decoding)
- ⚠️ Need to read original zork.z5 file for XOR decompression
- ⚠️ Some implementation work required (~500 lines of code)

**Implementation Effort:** 2-3 days

---

### ❌ Approach 2: Direct Process Memory Inspection

**How it works:**
- Attach to dfrotz process using ptrace/gdb
- Locate Z-machine memory array in process heap
- Read object table directly from RAM

**Pros:**
- ✅ Real-time access, no save/load overhead
- ✅ No file I/O

**Cons:**
- ❌ Very complex - need to locate Z-machine memory in process heap
- ❌ Docker container complicates process attachment (need --cap-add=SYS_PTRACE)
- ❌ Platform-specific (Linux ptrace vs macOS task_for_pid)
- ❌ Security/permissions issues
- ❌ Fragile - memory location changes between runs
- ❌ Requires C extensions or ctypes
- ❌ May interfere with dfrotz operation

**Implementation Effort:** 1-2 weeks + ongoing maintenance

---

### ❌ Approach 3: Modified Z-Machine Interpreter

**How it works:**
- Fork dfrotz or switch to a different interpreter (e.g., jzip)
- Add REST API or IPC endpoint to expose memory state
- Query object table on demand

**Pros:**
- ✅ Clean API design
- ✅ Full control over implementation
- ✅ Efficient real-time access

**Cons:**
- ❌ Need to maintain a fork
- ❌ Significant development effort (understanding interpreter internals)
- ❌ Deployment complexity (custom builds, Docker images)
- ❌ Must keep up with upstream dfrotz changes
- ❌ Risk of introducing bugs into interpreter

**Implementation Effort:** 2-4 weeks + ongoing maintenance

---

## Recommended Implementation Plan

### Phase 1: Core Parsing Library

Create `game_interface/memory/` package with four modules:

#### 1. `quetzal_parser.py`
**Purpose:** Parse IFF/FORM structure and extract chunks

```python
class QuetzalParser:
    def __init__(self, file_path: str):
        """Load and parse a Quetzal save file."""

    def get_chunk(self, chunk_id: str) -> bytes:
        """Get data for a specific chunk (e.g., 'CMem', 'IFhd')."""

    def get_ifhd_info(self) -> dict:
        """Extract story file identification info."""

    def has_compressed_memory(self) -> bool:
        """Check if file uses CMem (compressed) or UMem (uncompressed)."""
```

**Key Implementation Details:**
- Read file in binary mode
- Verify FORM type is 'IFZS'
- Parse chunk structure with proper endianness
- Handle odd-length chunks with padding

#### 2. `zmachine_memory.py`
**Purpose:** Decompress memory and provide byte-level access

```python
class ZMachineMemory:
    def __init__(self, quetzal_file: str, story_file: str):
        """Load memory from Quetzal file, decompress if needed."""

    def read_byte(self, address: int) -> int:
        """Read a single byte from memory."""

    def read_word(self, address: int) -> int:
        """Read a big-endian 16-bit word from memory."""

    def get_header_field(self, offset: int, size: int) -> int:
        """Read a field from the header."""

    def get_object_table_address(self) -> int:
        """Get address of object table from header."""
```

**Key Implementation Details:**
- Implement CMem decompression (RLE + XOR)
- Handle UMem (uncompressed) as well
- Load original story file for XOR base
- Bounds checking on all memory access

#### 3. `zmachine_text.py`
**Purpose:** Decode ZSCII text to Python strings

```python
class ZSCIIDecoder:
    def __init__(self, memory: ZMachineMemory):
        """Initialize with memory access for alphabet table."""

    def decode_object_name(self, prop_address: int) -> str:
        """Decode object's short name from properties."""

    def decode_zstring(self, address: int, max_words: int = None) -> str:
        """Decode a ZSCII-encoded string."""
```

**Key Implementation Details:**
- ZSCII uses 5-bit character encoding packed into 16-bit words
- Three alphabets (A0, A1, A2) with shift codes
- Special handling for abbreviations
- Text terminated by high bit of word = 1

#### 4. `object_table_reader.py`
**Purpose:** High-level API for game state extraction

```python
class ObjectTableReader:
    def __init__(self, save_file: str, story_file: str):
        """Initialize with paths to save and story files."""

    def find_player_object(self) -> int:
        """Find player object number (heuristic or fixed)."""

    def get_current_room(self) -> tuple[int, str]:
        """Get current room object number and name."""

    def get_inventory(self) -> list[tuple[int, str]]:
        """Get list of (object_num, name) for all inventory items."""

    def get_object_parent(self, obj_num: int) -> int:
        """Get parent of an object."""

    def get_object_name(self, obj_num: int) -> str:
        """Get decoded name of an object."""

    def get_object_children(self, obj_num: int) -> list[int]:
        """Get all children of an object (follows sibling chain)."""
```

### Phase 2: Integration with Existing System

#### Modify `ZorkInterface` (`game_interface/core/zork_interface.py`)

Add new method:

```python
def get_memory_state(self, save_filename: str = None) -> dict:
    """Extract structured state from most recent save file.

    Returns:
        {
            'room_number': int,
            'room_name': str,
            'inventory': [
                {'object_number': int, 'name': str},
                ...
            ],
            'score': int,  # if available from globals
            'moves': int,  # if available from globals
        }
    """
```

#### Modify `GameSession` (`game_interface/server/session_manager.py`)

Update `execute_command()` to extract memory state after each command:

```python
def execute_command(self, command: str) -> CommandResponse:
    # ... existing code ...

    # After command execution, extract memory state
    save_filename = f"autosave_{self.session_id}"
    memory_state = self.zork.get_memory_state(save_filename)

    # Include in response
    return CommandResponse(
        # ... existing fields ...
        memory_state=memory_state,  # Add new field
    )
```

### Phase 3: Testing & Validation

#### Test Cases

1. **Basic Parsing Test**
   - Parse `game_files/autosave_2025-09-03T14:04:02`
   - Verify IFF structure is valid
   - Extract and decompress CMem chunk
   - Validate memory size matches expected dynamic memory size

2. **Object Table Test**
   - Read object table address from header
   - Parse object entries
   - Verify player object can be found
   - Check object relationships (parent/sibling/child pointers)

3. **Text Decoding Test**
   - Extract known object names (e.g., "brass lantern", "mailbox")
   - Verify ZSCII decoding produces correct UTF-8
   - Handle special characters and abbreviations

4. **Integration Test**
   - Start game, move to different room
   - Pick up item
   - Save game
   - Parse save file
   - Verify extracted state matches game text

5. **Edge Cases**
   - Empty inventory
   - Room with no objects
   - Container objects
   - Very long object names

---

## Technical Specifications

### File Locations

```
game_interface/
├── memory/
│   ├── __init__.py
│   ├── quetzal_parser.py      # IFF/FORM parsing
│   ├── zmachine_memory.py     # Memory decompression & access
│   ├── zmachine_text.py       # ZSCII decoding
│   └── object_table_reader.py # High-level API
└── core/
    └── zork_interface.py       # Add get_memory_state() method

tests/
└── test_memory/
    ├── test_quetzal_parser.py
    ├── test_zmachine_memory.py
    ├── test_zmachine_text.py
    └── test_object_table_reader.py
```

### Dependencies

```toml
# pyproject.toml - no new dependencies needed!
# All parsing can be done with Python stdlib:
# - struct (binary parsing)
# - pathlib (file handling)
# - typing (type hints)
```

### Key Constants for Zork I

```python
# Z-machine Version 3 constants
VERSION_3_OBJECT_ENTRY_SIZE = 9
VERSION_3_MAX_OBJECTS = 255
VERSION_3_ATTRIBUTES_PER_OBJECT = 32
VERSION_3_PROPERTY_DEFAULTS_SIZE = 62  # 31 words

# Header offsets
HEADER_OBJECT_TABLE = 0x0A  # 2 bytes, big-endian
HEADER_GLOBALS_TABLE = 0x0C
HEADER_STATIC_MEMORY = 0x0E
HEADER_DYNAMIC_MEMORY_SIZE = 0x0E  # Same as static memory base

# Object entry offsets
OBJ_ATTRIBUTES = 0  # 4 bytes
OBJ_PARENT = 4      # 1 byte
OBJ_SIBLING = 5     # 1 byte
OBJ_CHILD = 6       # 1 byte
OBJ_PROPERTIES = 7  # 2 bytes, big-endian

# Zork I specific
ZORK_I_PLAYER_OBJECT = 20  # Usually object #20 or #21
```

---

## Performance Considerations

### Per-Turn Overhead

Parsing a Quetzal file involves:
1. Read file (~10-50 KB typical): **< 1ms**
2. Parse IFF structure: **< 1ms**
3. Decompress CMem (if compressed): **5-10ms** (depends on memory size)
4. Navigate object table: **< 1ms**
5. Decode 5-10 object names: **< 1ms**

**Total estimated overhead: 10-15ms per turn**

This is negligible compared to:
- LLM API calls (500-2000ms)
- Text parsing complexity
- Network round-trips

### Optimization Opportunities

1. **Cache original story file memory** (only load once at startup)
2. **Incremental parsing** (only parse if save file timestamp changed)
3. **Object name caching** (object names don't change during game)
4. **Skip decompression** (read UMem directly if we control save format)

---

## Alternative: Using `infodump` Tool

The Z-machine community has a tool called `infodump` that can dump object tables from story files. However:

❌ **Not suitable for our use case because:**
- Requires running external process (subprocess overhead)
- Designed for story files, not save files
- No API - just text output that would need parsing
- Doesn't show current game state (just static story structure)

Our custom parser is better because:
- Pure Python, no external dependencies
- Works with save files (current state)
- Programmatic API
- Can be optimized for our specific needs

---

## Risk Analysis

### Low Risk
- ✅ Quetzal format is stable and well-documented
- ✅ No changes to dfrotz (can't break interpreter)
- ✅ Existing save infrastructure already works
- ✅ Fallback to text parsing if memory parsing fails

### Medium Risk
- ⚠️ ZSCII decoding complexity (but well-specified)
- ⚠️ Finding player object requires heuristic (but consistent in Zork I)
- ⚠️ File I/O overhead (but minimal, see performance section)

### Mitigated Risks
- File not found → Fail gracefully, use text parsing
- Corrupt save file → Validate IFF structure, fall back to text parsing
- Wrong Z-machine version → Check version in header, abort if not V3

---

## Open Questions

1. **Player Object Detection**
   - Zork I uses object #20, but is this reliable?
   - Alternative: Search for object with specific attributes (animate, player-controlled)
   - Or: Include player object number in configuration

2. **Score/Moves Extraction**
   - Score stored in global variables (table at address in header)
   - Need to determine which global variable contains score
   - May require game-specific knowledge or heuristics

3. **Container Support**
   - Current spec extracts top-level inventory
   - What about items in containers (e.g., "sword in sack")?
   - Need to recursively traverse child chains?

4. **Save File Timing**
   - When exactly is save file written?
   - Is it guaranteed to be complete before next command?
   - May need file locking or timestamp checks

---

## Success Criteria

Implementation is successful when:

1. ✅ Can parse real Quetzal save files from current system
2. ✅ Accurately extracts current room name
3. ✅ Accurately extracts complete inventory list
4. ✅ Object names match what appears in game text
5. ✅ Performance overhead < 20ms per turn
6. ✅ Falls back gracefully on any errors
7. ✅ Unit tests achieve > 90% coverage
8. ✅ Integration tests pass with real game sessions

---

## Next Steps

1. **Proof of Concept** (1 day)
   - Implement basic QuetzalParser
   - Test with `game_files/autosave_2025-09-03T14:04:02`
   - Print object table to verify approach

2. **Core Implementation** (2 days)
   - Complete all four modules
   - Unit tests for each module
   - Handle edge cases

3. **Integration** (1 day)
   - Add get_memory_state() to ZorkInterface
   - Update GameSession to call memory parser
   - Update API models to include memory state

4. **Testing & Validation** (1 day)
   - End-to-end testing with real game sessions
   - Performance profiling
   - Error handling verification

5. **Documentation** (0.5 day)
   - API documentation
   - Usage examples
   - Troubleshooting guide

**Total Estimated Time: 4-5 days**

---

## References

- [Z-Machine Standards Document](https://inform-fiction.org/zmachine/standards/z1point1/)
- [Quetzal Save File Specification](https://www.inform-fiction.org/zmachine/standards/quetzal/)
- [Z-Machine Memory Architecture](https://zspec.jaredreisinger.com/002-overview)
- [dfrotz Source Code](https://github.com/DavidGriffith/frotz)

---

## Appendix: Sample Memory Dumps

### Example Object Table Entry (Brass Lantern)

```
Object #39 (Brass Lantern):
Address: 0x0347 (object table base + (39-1) * 9)

Hex dump:
00 00 00 10  14 00 00 08 93
│  │  │  │   │  │  │  │  └─> Properties: 0x0893
│  │  │  │   │  │  └─────> Child: 0 (nothing inside)
│  │  │  │   │  └────────> Sibling: 0 (no sibling)
│  │  │  │   └───────────> Parent: 20 (player)
└──┴──┴──┴──────────────> Attributes: bit 12 set (takeable)

Properties at 0x0893:
0B 42 72 61 73 73 20 6C 61 6E 74 65 72 6E ...
│  └────────────────────────────────────> "Brass lantern" (ZSCII)
└──> Text length: 11 characters (0x0B)
```

### Example Memory State After "take lamp"

```
Object #20 (Player):
  Parent: 68 (West of House)
  Child: 39 (Brass lantern)

Object #39 (Brass lantern):
  Parent: 20 (Player)  ← Changed from 68!
  Sibling: 0

⇒ get_inventory() → [(39, "Brass lantern")]
⇒ get_current_room() → (68, "West of House")
```

---

*Document Version: 1.0*
*Last Updated: 2025-10-16*
*Author: Claude Code (with Ryan)*
