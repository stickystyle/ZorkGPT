"""
ABOUTME: LLM-based memory synthesis for ZorkGPT - generates location memories from action context.
ABOUTME: Pure component with no observability - manager handles Langfuse instrumentation.
"""

from typing import Dict, List, Any, Optional
from managers.memory import Memory, MemorySynthesisResponse, HistoryFormatter
from session.game_configuration import GameConfiguration
from shared_utils import create_json_schema, extract_json_from_text
from llm_client import LLMClientWrapper


# Prompt templates as module constants
SYNTHESIS_PROMPT_HEADER = """Location: {location_name} (ID: {location_id})
{existing_section}
═══════════════════════════════════════════════════════════════
🚨 CRITICAL DEDUPLICATION CHECK 🚨

Before remembering ANYTHING, compare against existing memories above.

These are SEMANTICALLY DUPLICATE (DO NOT remember):
  ❌ "Leaflet reveals message" vs "Leaflet provides message"
  ❌ "Mailbox contains leaflet" vs "Leaflet found in mailbox"
  ❌ "Egg can be taken" vs "Taking egg succeeds"

These are NOT ACTIONABLE - handled by MapGraph (DO NOT remember):
  ❌ "Forest path leads north south" (exit information)
  ❌ "Path accessible from north house" (room connections)
  ❌ "Canyon View location discovered" (location tracking)
  ❌ "Can go west from here" (navigation)

Only remember if this provides NEW actionable information not semantically captured above.
═══════════════════════════════════════════════════════════════

CONTRADICTION CHECK:
═══════════════════════════════════════════════════════════════
Review existing memories above. Does this action outcome:

1. CONTRADICT any existing memory? (proves it wrong)
   Example: Memory says "troll accepts gifts peacefully" but troll attacks after accepting
   → Mark that memory as SUPERSEDED, create new DANGER memory

2. REVEAL DELAYED CONSEQUENCES? (success wasn't really success)
   Example: "Door opens" seemed successful but leads to death trap
   → Mark optimistic memory as SUPERSEDED, create WARNING memory

3. CLARIFY a TENTATIVE memory? (confirms or denies uncertain outcome)
   Example: TENTATIVE "troll might be friendly" → CONFIRMED as false by attack
   → Mark tentative memory as SUPERSEDED

If yes to any: list specific memory TITLES in supersedes_memory_titles field.
If contradicting multiple memories: list ALL relevant titles.
Use EXACT titles from existing memories above. If title is long, unique substring is sufficient.
═══════════════════════════════════════════════════════════════

SUPERSESSION PERSISTENCE RULES:
═══════════════════════════════════════════════════════════════
When superseding memories, persistence levels must be compatible:

✓ ALLOWED SUPERSESSIONS:
  • ephemeral → ephemeral (state update: "dropped sword" → "picked up sword")
  • ephemeral → permanent (upgrade: "door opened" → "door can be opened")
  • permanent → permanent (refinement: "troll peaceful" → "troll attacks")
  • core → core (rare: correcting spawn state observation)
  • core → permanent (confirmation: "sword here" → "sword takeable")

✗ FORBIDDEN (causes data loss after episode reset):
  • permanent → ephemeral ("troll attacks" → "dropped item near troll")
  • core → ephemeral ("mailbox here" → "opened mailbox")

**If permanent/core knowledge is wrong**: Use INVALIDATION instead of downgrade:
  1. Invalidate the wrong permanent/core memory (with reason)
  2. Create new ephemeral memory separately (if agent action needed)
  3. This preserves data integrity across episode boundaries

Example:
  ❌ Wrong: supersede "Troll attacks" (permanent) with "Dropped sword near troll" (ephemeral)
     → Would cause data loss: danger knowledge lost after episode reset

  ✓ Right approach (two separate operations):
     1. Keep "Troll attacks" (permanent) as-is (don't supersede)
     2. Create "Dropped sword near troll" (ephemeral) as NEW memory
     → Result: Both memories coexist - danger knowledge preserved, state change recorded
═══════════════════════════════════════════════════════════════

INVALIDATION CHECK (without replacement):
═══════════════════════════════════════════════════════════════
Can you DISPROVE an existing memory without creating a specific replacement?

Use INVALIDATION when:
✓ Memory proven false but no specific replacement needed
✓ Multiple memories all wrong due to core false assumption
✓ Evidence shows memory is incorrect but don't need to explain what's correct

Examples:

1. **Death invalidates TENTATIVE assumptions:**
   Existing: [TENTATIVE] "Troll might be friendly"
   Outcome: Agent died from troll attack
   → **INVALIDATE** "Troll might be friendly", reason: "Proven false by death"
   → Don't create redundant memory (agent already knows it died)
   → BUT: Do create DANGER memory about troll behavior ("Troll attacks unprovoked")

2. **Core assumption proven false:**
   Existing: [NOTE] "Door is unlocked", [NOTE] "Safe to enter"
   Outcome: Door was actually locked, entering caused trap
   → **INVALIDATE** both memories, reason: "Door was locked, not unlocked"
   → **CREATE** new memory: [DANGER] "Door locked, entering triggers trap"

   WHY CREATE HERE: The trap mechanism is new information worth remembering.
   In example 1, death itself is already known (don't duplicate death fact).

3. **Multiple related memories wrong:**
   Existing: "Troll accepts gifts", "Troll is pacified by food"
   Outcome: Troll attacks after accepting food
   → **INVALIDATE** both, reason: "Troll hostile regardless of gifts"
   → **CREATE** new memory: [DANGER] "Troll attacks immediately after accepting food"

**When to use invalidate_memory_titles vs supersedes_memory_titles:**

INVALIDATE (standalone):
- Multiple unrelated memories all wrong
- Memory proven false, no specific replacement
- Death invalidates speculative memories
- Don't want to explain the correct approach

SUPERSEDE (with replacement):
- Old memory was close but needs refinement
- Better understanding of same situation
- Specific correction or update

**Both are allowed in same response** if you're creating a new memory that supersedes
one old memory AND invalidating other unrelated wrong memories.

If invalidating: populate invalidate_memory_titles + invalidation_reason
If superseding: populate supersedes_memory_titles (and create new memory)
═══════════════════════════════════════════════════════════════

MEMORY STATUS DECISION:
═══════════════════════════════════════════════════════════════
WORKFLOW: First check duplicates → Then check contradictions → Then determine status

Choose status based on outcome certainty:

**ACTIVE** (default) - Use when:
✓ Outcome is immediate and certain
✓ Consequence is fully understood
✓ No delayed effects expected
Examples:
  • "Mailbox contains leaflet" (examined, saw leaflet, certain)
  • "Window is locked" (tried to open, failed, certain)
  • "Lamp provides light" (lit lamp, room illuminated, confirmed)

**TENTATIVE** - Use when:
⚠️  Immediate action succeeds BUT long-term consequence unclear
⚠️  Entity accepts action BUT reaction not yet known
⚠️  Effect seems positive BUT might have hidden downsides
Examples:
  • "Troll accepts lunch gift" (took it but might attack later) → TENTATIVE
  • "Door unlocked successfully" (opened but don't know what's inside) → TENTATIVE
  • "Drank mysterious potion" (consumed but effect not yet clear) → TENTATIVE

**Rule of thumb:** If you think "this worked... for now", mark it TENTATIVE.
═══════════════════════════════════════════════════════════════

MEMORY PERSISTENCE CLASSIFICATION:
═══════════════════════════════════════════════════════════════
Choose based on WHAT HAPPENED (action type), not WHEN (visit timing).

**CORE** - Spawn state from room description (FIRST VISIT ONLY):
  Definition: Items/objects/fixtures in room description on first visit
  When to use:
    ✓ ONLY on first visit to location (first_visit=true)
    ✓ ONLY for passive observations from room text
    ✓ NOT for agent actions or discoveries

  Examples:
    ✓ "Sword here" (from "Living Room. There is a sword here.") → CORE
    ✓ "Brass lantern in trophy case" (from room description) → CORE
    ✓ "Mailbox visible" (from "West of House" description) → CORE
    ✗ "Dropped sword here" (agent action, not room text) → NOT CORE
    ✗ "Sword was here" (return visit) → NOT CORE

**PERMANENT** - Game mechanics and reusable knowledge:
  Definition: How the game works; knowledge true across episodes
  When to use:
    ✓ ANY visit (first or return)
    ✓ Learning rules, mechanics, dangers, constraints
    ✓ Knowledge that stays true after episode reset

  Examples:
    ✓ "Troll attacks on sight" (danger behavior) → PERMANENT
    ✓ "Window can be opened" (game mechanic) → PERMANENT
    ✓ "Taking egg grants 5 points" (scoring rule) → PERMANENT
    ✓ "Door nailed shut" (permanent obstacle) → PERMANENT
    ✓ "Cannot climb tree from here" (constraint) → PERMANENT

**EPHEMERAL** - Agent-caused state changes:
  Definition: What agent DID that changes state temporarily
  When to use:
    ✓ ANY visit (first or return)
    ✓ Agent performed action: drop, place, open, take, move
    ✓ State change that resets on episode boundary

  Examples:
    ✓ "Dropped sword here" (agent action) → EPHEMERAL
    ✓ "Placed nest in sack" (agent organization) → EPHEMERAL
    ✓ "Opened window from outside" (agent state change) → EPHEMERAL
    ✓ "Left lantern on table" (inventory management) → EPHEMERAL

DECISION CRITERIA:
1. CORE: Room description observation on FIRST VISIT only
2. EPHEMERAL: Agent action that changes state (ANY VISIT)
3. PERMANENT: Game mechanic/rule learned (ANY VISIT)

If agent DOES something → likely EPHEMERAL
If agent LEARNS something → likely PERMANENT
If agent SEES something in room description (first visit) → likely CORE

Current visit status: {visit_status}
⚠️  CORE only allowed on first visit

Response field REQUIRED: "persistence": "core" | "permanent" | "ephemeral"
═══════════════════════════════════════════════════════════════
"""

SYNTHESIS_PROMPT_HISTORY_SECTION = """
═══════════════════════════════════════════════════════════════
RECENT ACTION SEQUENCE:
═══════════════════════════════════════════════════════════════

{actions_formatted}

{reasoning_section}
═══════════════════════════════════════════════════════════════
"""

SYNTHESIS_PROMPT_ACTION_ANALYSIS = """
ACTION ANALYSIS:
Action: {action}
Response: {response}

State Changes (ground truth from Z-machine):
• Score: {score_delta:+d} points
• Location changed: {location_changed}
• Inventory changed: {inventory_changed}
• Died: {died}
• First visit: {first_visit}

REASONING STEPS (use your reasoning capabilities):
1. Identify the KEY object/entity in this action (e.g., "leaflet", "troll", "egg")
2. Identify the KEY relationship/insight (e.g., "contains item", "blocks path", "is takeable")
3. Check existing memories: Does ANY memory mention this object + relationship?
4. Use semantic matching:
   - "reveals" = "provides" = "shows" = "contains"
   - "take" = "pick up" = "grab" = "obtain"
   - "blocks" = "prevents" = "stops"
5. If semantic match found → should_remember=false
6. If truly new insight → should_remember=true

MULTI-STEP PROCEDURE DETECTION:
═══════════════════════════════════════════════════════════════
Review the RECENT ACTION SEQUENCE above. Does the current outcome depend on previous actions?

**Look for these patterns:**

1. **Prerequisites** (action B requires action A first):
   Example: "open window" (turn N) → "enter window" (turn N+1) → success
   Memory: "To enter kitchen: (1) open window, (2) enter window"

2. **Delayed Consequences** (action seemed successful but had delayed effect):
   Example: "give lunch to troll" (turn N, seemed ok) → troll attacks (turn N+1)
   Memory: "Troll attacks after accepting gift - gift strategy fails"
   Action: Mark previous TENTATIVE memory as SUPERSEDED

3. **Progressive Discovery** (understanding deepens over multiple turns):
   Example: Turn N "examine door" (locked) → Turn N+1 "unlock with key" → Turn N+2 "open door" (success)
   Memory: "Door requires key to unlock before opening"

**How to capture multi-step procedures:**
- If outcome required previous actions: Include steps in memory_text
- Format: "To achieve X: (1) step1, (2) step2" or "After A, then B occurs"
- Don't duplicate if existing memory already captures the complete procedure

═══════════════════════════════════════════════════════════════

🚨 CRITICAL - DO NOT REMEMBER THESE (handled by MapGraph) 🚨
═══════════════════════════════════════════════════════════════
The MapGraph system ALREADY tracks all spatial navigation. DO NOT create memories for:

❌ Exits and directions
   Examples: "path leads north/south", "exits are north/east/west", "can go north"
   WHY: MapGraph tracks all room connections and exits automatically

❌ Location discovery
   Examples: "found Forest", "discovered Canyon View", "reached Kitchen", "Forest location discovered"
   WHY: MapGraph marks locations as visited automatically

❌ Room connections
   Examples: "path accessible from north house", "forest connects to clearing"
   WHY: MapGraph builds connection graph from movement

❌ Simple movement success
   Examples: "went north successfully", "moved to next room", "entered new area"
   WHY: Movement is not actionable knowledge, just navigation

❌ DUPLICATES (semantically similar to existing memories)
   Examples: "Leaflet reveals message" vs "Leaflet provides message"
   WHY: Existing memory already captures the insight
═══════════════════════════════════════════════════════════════

✅ REMEMBER (actionable game mechanics NOT handled by other systems):
═══════════════════════════════════════════════════════════════
✅ Object interactions (how to use items, what works/fails)
   WHY: MapGraph doesn't track object mechanics or puzzle solutions

✅ Dangers (death, hazards, threats)
   WHY: Critical survival information, not captured by navigation

✅ Puzzle mechanics (how things operate, constraints)
   WHY: Game rules and mechanics, not spatial data
   Example: "Window must be opened before entering" (constraint)
   NOT: "Window leads to kitchen" (navigation)

✅ Item discoveries (finding items, understanding purpose)
   WHY: Item properties and uses, not just location

✅ Score-earning actions
   WHY: Learning which actions grant points
═══════════════════════════════════════════════════════════════

**CRITICAL - OUTPUT FORMAT:**
YOU MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON. Do not include thinking tags, reasoning outside the JSON structure, or markdown fences.

If should_remember=false (duplicate/navigation/not actionable):
{{
  "should_remember": false,
  "reasoning": "explain why not remembering (semantic duplicate, navigation, etc.)"
}}

If should_remember=true (new actionable insight):
{{
  "should_remember": true,
  "category": "SUCCESS"|"FAILURE"|"DISCOVERY"|"DANGER"|"NOTE",
  "memory_title": "3-6 words, evergreen",
  "memory_text": "1-2 sentences, actionable insight",
  "persistence": "core"|"permanent"|"ephemeral",
  "status": "ACTIVE"|"TENTATIVE",
  "supersedes_memory_titles": ["Title1", "Title2"],
  "invalidate_memory_titles": ["Title3", "Title4"],
  "invalidation_reason": "explanation for why invalidated memories are wrong",
  "reasoning": "explain semantic comparison, contradiction detection, status choice, persistence choice"
}}

Example valid response for NOT remembering:
{{
  "should_remember": false,
  "reasoning": "Semantic duplicate - existing memory 'Mailbox contains leaflet' already captures this insight"
}}

Example valid response for remembering:
{{
  "should_remember": true,
  "category": "DANGER",
  "memory_title": "Troll attacks after accepting gift",
  "memory_text": "Troll accepts lunch gift but then becomes hostile and attacks. Gift strategy ineffective.",
  "persistence": "permanent",
  "status": "ACTIVE",
  "supersedes_memory_titles": ["Troll accepts lunch gift"],
  "reasoning": "Contradicts previous tentative memory - troll is not pacified by gifts. PERMANENT because this is a game mechanic that stays true across episodes."
}}

Example valid response for invalidating without new memory:
{{
  "should_remember": false,
  "invalidate_memory_titles": ["Troll is friendly", "Troll accepts gifts peacefully"],
  "invalidation_reason": "Both proven false by troll attack resulting in death",
  "reasoning": "Death proves both TENTATIVE assumptions were wrong, no new memory needed"
}}

Example valid response for creating new memory AND invalidating others:
{{
  "should_remember": true,
  "category": "DANGER",
  "memory_title": "Troll attacks after accepting gift",
  "memory_text": "Troll accepts gift but then attacks immediately. Gift strategy fails.",
  "persistence": "permanent",
  "status": "ACTIVE",
  "supersedes_memory_titles": ["Troll accepts lunch gift"],
  "invalidate_memory_titles": ["Troll is friendly"],
  "invalidation_reason": "Proven false by attack",
  "reasoning": "Superseding the direct memory about gift, invalidating unrelated assumption. PERMANENT because danger pattern persists across episodes."
}}"""


class MemorySynthesizer:
    """
    Synthesizes location memories from action context using LLM.

    This is a pure component - no Langfuse instrumentation.
    The manager (SimpleMemoryManager) handles all observability.
    """

    def __init__(
        self,
        logger,
        config: GameConfiguration,
        formatter: HistoryFormatter,
        llm_client: Optional[LLMClientWrapper] = None
    ):
        """
        Initialize synthesizer.

        Args:
            logger: Logger instance for debug/info messages
            config: Game configuration (for model, sampling, history window)
            formatter: History formatter for action/reasoning formatting
            llm_client: Optional LLM client (if None, synthesis returns None)
        """
        self.logger = logger
        self.config = config
        self.formatter = formatter
        self.llm_client = llm_client

    def synthesize_memory(
        self,
        location_id: int,
        location_name: str,
        action: str,
        response: str,
        existing_memories: List[Memory],
        z_machine_context: Dict[str, Any],
        actions_formatted: str = "",
        reasoning_formatted: str = ""
    ) -> Optional[MemorySynthesisResponse]:
        """
        Synthesize a new memory from action context.

        Returns None if:
        - LLM client not available
        - No new information to remember
        - LLM call fails

        Args:
            location_id: Current location integer ID
            location_name: Location display name
            action: Action taken by agent
            response: Game response text
            existing_memories: List of existing memories at this location
            z_machine_context: Ground truth state changes from Z-machine
            actions_formatted: Pre-formatted action history (optional)
            reasoning_formatted: Pre-formatted reasoning history (optional)

        Returns:
            MemorySynthesisResponse if LLM says to remember, None otherwise
        """
        # Early exit if no LLM client
        if not self.llm_client:
            self.logger.debug("No LLM client available, skipping synthesis")
            return None

        try:
            # Build synthesis prompt
            prompt = self._build_synthesis_prompt(
                location_id=location_id,
                location_name=location_name,
                action=action,
                response=response,
                existing_memories=existing_memories,
                z_machine_context=z_machine_context,
                actions_formatted=actions_formatted,
                reasoning_formatted=reasoning_formatted
            )

            # Call LLM with structured output
            llm_response = self.llm_client.chat.completions.create(
                model=self.config.memory_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.memory_sampling.get('temperature', 0.3),
                max_tokens=self.config.memory_sampling.get('max_tokens', 1000),
                name="SimpleMemory",
                response_format=create_json_schema(MemorySynthesisResponse)
            )

            # Check if we hit token limit (which could cause truncated JSON)
            if llm_response.usage:
                completion_tokens = llm_response.usage.get('completion_tokens', 0)
                max_tokens = self.config.memory_sampling.get('max_tokens', 1000)
                if completion_tokens >= max_tokens * 0.95:  # Within 5% of limit
                    self.logger.warning(
                        f"Memory synthesis response near token limit: {completion_tokens}/{max_tokens} tokens used",
                        extra={
                            "completion_tokens": completion_tokens,
                            "max_tokens": max_tokens,
                            "location_id": location_id
                        }
                    )

            # Extract JSON (handles markdown fences, reasoning tags, and embedded JSON)
            json_content = extract_json_from_text(llm_response.content)

            # Parse response
            synthesis = MemorySynthesisResponse.model_validate_json(json_content)

            # Check if should remember
            if not synthesis.should_remember:
                self.logger.debug(
                    "LLM decided not to remember",
                    extra={
                        "location_id": location_id,
                        "reasoning": synthesis.reasoning
                    }
                )
                return None

            self.logger.debug(
                "LLM synthesis complete",
                extra={
                    "location_id": location_id,
                    "category": synthesis.category,
                    "title": synthesis.memory_title,
                    "reasoning": synthesis.reasoning
                }
            )

            return synthesis

        except Exception as e:
            # Get response preview safely (llm_response may not be defined if error occurred during call)
            try:
                response_preview = llm_response.content[:500] if llm_response and llm_response.content else "No response content"
                response_length = len(llm_response.content) if llm_response and llm_response.content else 0
                tokens_used = llm_response.usage.get('completion_tokens', 0) if llm_response and llm_response.usage else 0
            except (NameError, AttributeError):
                response_preview = "Error occurred before LLM response received"
                response_length = 0
                tokens_used = 0

            # Check if this is a JSON truncation error (EOF while parsing)
            error_str = str(e)
            if "EOF while parsing" in error_str or "Unterminated string" in error_str:
                max_tokens = self.config.memory_sampling.get('max_tokens', 1000)
                self.logger.error(
                    f"JSON truncation detected - response likely hit token limit: {e}",
                    extra={
                        "location_id": location_id,
                        "error": error_str,
                        "response_length": response_length,
                        "tokens_used": tokens_used,
                        "max_tokens": max_tokens,
                        "suggestion": f"Consider increasing memory_sampling.max_tokens (current: {max_tokens})",
                        "response_preview": response_preview
                    }
                )
            else:
                self.logger.error(
                    f"Failed to synthesize memory: {e}",
                    extra={
                        "location_id": location_id,
                        "error": error_str,
                        "response_preview": response_preview
                    }
                )
            return None

    def _build_synthesis_prompt(
        self,
        location_id: int,
        location_name: str,
        action: str,
        response: str,
        existing_memories: List[Memory],
        z_machine_context: Dict[str, Any],
        actions_formatted: str = "",
        reasoning_formatted: str = ""
    ) -> str:
        """
        Build synthesis prompt from context.

        Args:
            location_id: Current location integer ID
            location_name: Location display name
            action: Action taken by agent
            response: Game response text
            existing_memories: List of existing memories at this location
            z_machine_context: Ground truth state changes from Z-machine
            actions_formatted: Pre-formatted action history (optional)
            reasoning_formatted: Pre-formatted reasoning history (optional)

        Returns:
            Complete synthesis prompt string
        """
        # Format existing memories - TITLES ONLY for conciseness
        existing_section = self._format_existing_memories(existing_memories)

        # Get visit status
        visit_status = "FIRST VISIT" if z_machine_context.get('first_visit', False) else "RETURN VISIT"

        # Build header with existing memories and rules
        prompt = SYNTHESIS_PROMPT_HEADER.format(
            location_name=location_name,
            location_id=location_id,
            existing_section=existing_section,
            visit_status=visit_status
        )

        # Add history sections if available (for multi-step procedure detection)
        if actions_formatted or reasoning_formatted:
            # Build reasoning section
            reasoning_section = ""
            if reasoning_formatted:
                reasoning_section = f"\nAGENT'S REASONING:\n{reasoning_formatted}\n"

            # Add history section to prompt
            prompt += SYNTHESIS_PROMPT_HISTORY_SECTION.format(
                actions_formatted=actions_formatted if actions_formatted else "(No recent actions available - this is one of the first turns)",
                reasoning_section=reasoning_section
            )

        # Add action analysis section
        prompt += SYNTHESIS_PROMPT_ACTION_ANALYSIS.format(
            action=action,
            response=response,
            score_delta=z_machine_context.get('score_delta', 0),
            location_changed=z_machine_context.get('location_changed', False),
            inventory_changed=z_machine_context.get('inventory_changed', False),
            died=z_machine_context.get('died', False),
            first_visit=z_machine_context.get('first_visit', False)
        )

        return prompt

    def _format_existing_memories(self, memories: List[Memory]) -> str:
        """
        Format existing memories for deduplication check.

        Args:
            memories: List of existing memories at location

        Returns:
            Formatted string with memory titles and categories
        """
        if memories:
            memory_titles = "\n".join(
                f"  • [{mem.category}] {mem.title}"
                for mem in memories
            )
            return f"""
EXISTING MEMORIES AT THIS LOCATION:
{memory_titles}
"""
        else:
            return "\nEXISTING MEMORIES: None (first memory for this location)\n"

    def _get_z_machine_context_text(self, context: Dict) -> str:
        """
        Format Z-machine state changes for prompt.

        Args:
            context: Dictionary with Z-machine state changes

        Returns:
            Formatted state change text
        """
        lines = []
        lines.append(f"• Score: {context.get('score_delta', 0):+d} points")
        lines.append(f"• Location changed: {context.get('location_changed', False)}")
        lines.append(f"• Inventory changed: {context.get('inventory_changed', False)}")
        lines.append(f"• Died: {context.get('died', False)}")
        lines.append(f"• First visit: {context.get('first_visit', False)}")
        return "\n".join(lines)
