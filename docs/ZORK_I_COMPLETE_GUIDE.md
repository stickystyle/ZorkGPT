# Zork I: The Great Underground Empire - Complete Game Guide

This document is a comprehensive reference for Zork I, compiled for use in AI agent development and research. It covers all puzzles, treasures, death traps, unwinnable states, and the external knowledge required to complete the game.

---

## Table of Contents

1. [Game Overview](#1-game-overview)
2. [Map Structure](#2-map-structure)
3. [Core Mechanics](#3-core-mechanics)
4. [Complete Treasure List](#4-complete-treasure-list)
5. [Puzzle Solutions by Area](#5-puzzle-solutions-by-area)
6. [External Knowledge Requirements](#6-external-knowledge-requirements)
7. [Death Traps](#7-death-traps)
8. [Unwinnable States](#8-unwinnable-states)
9. [The Thief](#9-the-thief)
10. [Optimal Walkthrough](#10-optimal-walkthrough)
11. [Common Player Frustrations](#11-common-player-frustrations)
12. [AI Agent Considerations](#12-ai-agent-considerations)

---

## 1. Game Overview

### Objective
Collect all 19 treasures (worth 350 points total) and place them in the trophy case in the Living Room of the White House. This unlocks the endgame sequence and grants the rank of "Master Adventurer."

### Setting
- **Above Ground**: The White House and surrounding forest (limited area)
- **Underground**: The Great Underground Empire - an extensive cave system with multiple zones

### Key Constraints
- **Light**: Most underground areas are dark; being in darkness leads to death by Grue
- **Inventory**: Limited carrying capacity; some items are heavy
- **Time**: Lamp battery is finite; candles burn out
- **The Thief**: A wandering NPC who steals items and attacks the player

---

## 2. Map Structure

### Above Ground Locations

```
                    [Forest]
                        |
    [Forest] -- [Clearing] -- [Forest]
                        |
    [Forest] -- [North of House] -- [Forest Path]
        |               |                   |
    [West of      [White House]      [East of House]
     House]       (inside areas)     (window entry)
        |               |
    [South          [Behind
     of House]       House]
        |
    [Forest] -- [Canyon View]
```

**White House Interior:**
- Living Room (trophy case, sword, lamp)
- Kitchen (water, food, garlic)
- Attic (rope, nasty knife) - via stairs from kitchen

### Underground Zones

**Zone 1: Cellar Area**
- Cellar (entry from trapdoor)
- Troll Room (must kill troll)
- East-West Passage

**Zone 2: Maze of Twisty Passages**
- Multiple "all alike" rooms
- Contains: Bag of coins, Skeleton key
- Leads to: Cyclops Room

**Zone 3: Dam and River**
- Flood Control Dam #3
- Dam Base, Dam Lobby
- Maintenance Room (wrench, screwdriver)
- Reservoir (North and South) - drainable
- Frigid River (boat journey)
- Aragain Falls, Rainbow End

**Zone 4: Coal Mine**
- Shaft Room (basket/rope system)
- Smelly Room, Gas Room (sapphire bracelet)
- Coal Mine (coal)
- Ladder areas

**Zone 5: Temple and Hades**
- Temple, Altar (prayer location)
- Torch Room (ivory torch)
- Egyptian Room (coffin, sceptre)
- Dome Room
- Entrance to Hades (evil spirits)
- Land of the Living Dead (skull)

**Zone 6: Atlantis**
- Atlantis Room (trident)
- Cave area

**Zone 7: Thief's Lair**
- Thief's Treasure Room
- Cyclops Room (connects to Living Room when cyclops flees)

**Special Locations:**
- Mirror Rooms (North and South) - teleportation link
- Loud Room (platinum bar puzzle)
- Machine Room (coal-to-diamond)
- Bat Room (jade figurine)
- Sandy Cave (buried scarab)

---

## 3. Core Mechanics

### Light Management

**Light Sources:**
| Source | Duration | Notes |
|--------|----------|-------|
| Brass Lantern | ~350 turns | Starts in Living Room; battery depletes |
| Ivory Torch | Unlimited | In Torch Room; essential for late game |
| Candles | ~40 turns when lit | Required for Hades puzzle; don't waste |
| Matchbook | 5 matches | From Dam Lobby |

**The Grue:**
- Appears in any dark location after 1-3 turns
- Instant death: "It is pitch black. You are likely to be eaten by a grue."
- Prevention: Always carry active light source in dark areas
- Well-warned; fair mechanic

### Combat System

**Weapons:**
- Elvish Sword (glows blue near enemies)
- Nasty Knife (slightly more effective than sword)

**Combat Factors:**
- Your score affects combat strength (higher = stronger)
- Random element in combat outcomes
- Save before combat; restore if losing

**Enemies:**
| Enemy | Location | Solution |
|-------|----------|----------|
| Troll | Troll Room | Kill with sword (blocks passage) |
| Thief | Random | Kill late game with high score |
| Cyclops | Cyclops Room | Say "ODYSSEUS" or give food |
| Spirits | Hades Entrance | Bell, book, candle ritual |

### Inventory and Weight

- Limited carrying capacity
- Some items described as "heavy" (coffin, trunk)
- Brown sack can hold items (useful for sharp objects on boat)
- Dropping items in locations is safe (except thief may steal)

### Scoring

- Points awarded for: touching treasures, placing in trophy case, solving puzzles
- 350 points = Maximum score = Master Adventurer
- Score affects combat effectiveness
- All 19 treasures in case triggers endgame

---

## 4. Complete Treasure List

| # | Treasure | Touch | Case | Total | Location | Method |
|---|----------|-------|------|-------|----------|--------|
| 1 | Jewel-encrusted Egg | 5 | 5 | 10 | Tree nest | Climb tree, get egg |
| 2 | Clockwork Canary | 6 | 4 | 10 | Inside egg | Thief must open egg |
| 3 | Brass Bauble | 1 | 1 | 2 | Forest | Wind canary in tree, bird drops it |
| 4 | Beautiful Painting | 4 | 6 | 10 | Gallery | Take from wall |
| 5 | Platinum Bar | 10 | 5 | 15 | Loud Room | Say "ECHO" or drain reservoir |
| 6 | Ivory Torch | 14 | 6 | 20 | Torch Room | Take (also serves as light) |
| 7 | Gold Coffin | 10 | 15 | 25 | Egyptian Room | PRAY at altar to escape with it |
| 8 | Egyptian Sceptre | 4 | 6 | 10 | Inside coffin | Open coffin |
| 9 | Trunk of Jewels | 15 | 5 | 20 | Reservoir South | Drain reservoir first |
| 10 | Crystal Trident | 4 | 11 | 15 | Atlantis Room | Take |
| 11 | Jade Figurine | 5 | 5 | 10 | Bat Room | Take |
| 12 | Sapphire Bracelet | 5 | 5 | 10 | Gas Room | Take |
| 13 | Huge Diamond | 10 | 10 | 20 | Machine Room | Create from coal |
| 14 | Bag of Coins | 10 | 5 | 15 | Maze | Navigate maze |
| 15 | Crystal Skull | 10 | 10 | 20 | Land of Living Dead | Complete Hades puzzle |
| 16 | Jeweled Scarab | 5 | 5 | 10 | Sandy Cave | Dig repeatedly |
| 17 | Large Emerald | 5 | 10 | 15 | Red Buoy | Get buoy on river, open it |
| 18 | Silver Chalice | 10 | 5 | 15 | Treasure Room | Kill thief |
| 19 | Pot of Gold | 10 | 10 | 20 | Rainbow End | Wave sceptre at rainbow |

**Total: 350 points**

---

## 5. Puzzle Solutions by Area

### 5.1 White House Entry

**Puzzle:** Enter the White House
- Front door is nailed shut (boarded up)
- **Solution:** Go east of house, OPEN WINDOW, ENTER

**Puzzle:** Access the underground
- **Solution:**
  1. In Living Room: MOVE RUG (reveals trapdoor)
  2. OPEN TRAPDOOR
  3. TURN ON LAMP
  4. GO DOWN

### 5.2 Troll Room

**Puzzle:** Get past the troll
- Troll guards passage, won't let you pass
- **Solution:** KILL TROLL WITH SWORD (repeat until dead)
- Combat is random; save first and restore if losing
- Once dead, passage is clear permanently

### 5.3 Maze of Twisty Passages

**Puzzle:** Navigate the maze and find treasures
- All rooms described as "maze of twisty passages, all alike"
- Directions don't work logically (north doesn't reverse south)

**Solution - Direct Path to Coins:**
From Troll Room: E, E (enter maze), then W, W, U
- Get bag of coins and skeleton key from skeleton

**Solution - Path to Cyclops Room:**
From skeleton: SW, E, S, SE

**Mapping Strategy:**
- Drop unique items in rooms to mark them
- Note: Thief may steal your markers
- Systematic exploration required

### 5.4 Cyclops Room (GREEK MYTHOLOGY PUZZLE)

**Puzzle:** The Cyclops blocks passage to the Living Room
- Cyclops is hungry and dangerous
- Combat is not effective

**Solution (Preferred):** Say "ODYSSEUS" (or "ULYSSES")
- The cyclops, terrified of his old master, flees
- Opens permanent passage to Living Room (shortcut!)

**How to Discover:**
- Black prayer book contains a poem
- First letter of each line spells "ODYSSEUS" (acrostic)
- Requires: Knowledge of Homer's Odyssey

**Alternative Solution:**
- Give LUNCH (food from kitchen)
- Give GARLIC (from kitchen)
- Cyclops falls asleep, then you can attack

**Why Players Get Stuck:**
- Acrostic pattern not obvious
- Greek mythology reference may be unknown
- Modern screens may wrap lines, hiding the pattern

### 5.5 Flood Control Dam #3

**Puzzle:** Operate the dam to drain reservoir

**Solution:**
1. Go to Maintenance Room
2. GET WRENCH, GET SCREWDRIVER
3. Go to Dam
4. PUSH YELLOW BUTTON (activates control panel; green bubble glows)
5. TURN BOLT WITH WRENCH (opens dam gates)
6. Water begins draining
7. Wait 2 turns at Reservoir South
8. Reservoir drains, revealing:
   - Trunk of jewels (Reservoir South)
   - Air pump (Reservoir North)

**Notes:**
- Yellow button must be pressed before bolt will turn
- Blue button should be avoided (causes problems)
- Draining affects Loud Room puzzle

### 5.6 Loud Room (ABSTRACT PUZZLE)

**Puzzle:** Get the platinum bar
- Room is "unbearably loud" - echoes prevent action
- Can't pick up bar normally

**Solution 1 (Unintuitive):** Type "ECHO"
- Acoustics change; room becomes quiet
- GET PLATINUM BAR

**Solution 2 (Logical):** Drain the reservoir
1. Complete dam puzzle (open gates)
2. Go to Loud Room while reservoir is empty
3. Room is now "eerily silent"
4. GET PLATINUM BAR
5. Return to dam, close gates before reservoir refills

**Why This Puzzle is Notorious:**
- "ECHO" solution has no logical connection to picking up an object
- Game doesn't explain why room is loud (water over dam)
- Considered one of the most unfair puzzles in adventure gaming
- Most players need a hint

### 5.7 Coal Mine and Diamond Machine

**Puzzle:** Create a diamond from coal

**Solution:**
1. Navigate to Coal Mine (through Gas Room, careful of smell)
2. GET COAL
3. Go to Shaft Room
4. PUT COAL IN BASKET
5. LOWER BASKET (rope and basket system)
6. Navigate to Machine Room (from below)
7. GET COAL (from basket)
8. OPEN MACHINE LID
9. PUT COAL IN MACHINE
10. CLOSE LID
11. TURN SWITCH WITH SCREWDRIVER
12. OPEN LID
13. GET DIAMOND

**Knowledge Required:**
- Real-world physics: Diamonds form from carbon under extreme pressure
- The machine is a pressure/heat device

**Notes:**
- Must close lid before turning switch
- Screwdriver from Maintenance Room required

### 5.8 Boat Journey and Frigid River

**Puzzle:** Navigate the river to get treasures

**Prerequisites:**
- Deflated plastic (from Dam Base)
- Air pump (from Reservoir North after draining)

**Solution:**
1. GET DEFLATED PLASTIC (pile of plastic at Dam Base)
2. GET PUMP (from drained Reservoir North)
3. INFLATE PLASTIC WITH PUMP (creates boat)
4. **CRITICAL:** PUT SCREWDRIVER IN SACK, PUT SCEPTRE IN SACK
   - Sharp objects will puncture the boat!
5. BOARD BOAT
6. LAUNCH
7. WAIT (4 times) - river carries you
8. During journey: GET RED BUOY (from Buoy Room)
9. At Sandy Beach: EXIT BOAT or LAND
10. OPEN RED BUOY - contains LARGE EMERALD

**Death Trap Warning:**
- Entering boat with sharp objects (screwdriver, sceptre) = boat puncture = death
- Items must be in a container (sack) to be safe
- Game warns items are "sharp" but doesn't explicitly warn about boat

**After Beach:**
- Can explore Sandy Cave (dig for scarab)
- Can return via Aragain Falls area

### 5.9 Rainbow and Pot of Gold

**Puzzle:** Cross the rainbow to get pot of gold

**Solution:**
1. Have the SCEPTRE (from coffin)
2. Go to Aragain Falls or Rainbow End
3. WAVE SCEPTRE
4. Rainbow becomes solid bridge
5. Cross rainbow
6. GET POT OF GOLD

**Knowledge Required:**
- Fantasy logic: Magic items can transform things
- Reasonably well-clued by item description

### 5.10 Temple, Altar, and Coffin

**Puzzle:** Get the coffin out of the temple

- Gold coffin is too heavy to carry through the narrow hole
- Seems impossible to remove

**Solution:**
1. GET COFFIN (you can carry it, just not through hole)
2. Go to Altar
3. PRAY
4. You are teleported above ground with all possessions!
5. Coffin and sceptre now accessible

**Knowledge Required:**
- Religious context: Prayer at altar = divine intervention
- Hint: Trying other exits gives "You haven't a prayer" (pun!)

### 5.11 Hades / Bell, Book, and Candle (RELIGIOUS PUZZLE)

**Puzzle:** Evil spirits block entrance to Land of Living Dead

**Required Items:**
- Brass bell (from Temple)
- Black prayer book (from Temple area)
- Candles (from Altar)
- Matches (from Dam Lobby)

**Solution (Must be Fast!):**
1. Go to Entrance to Hades with all items
2. RING BELL (you drop bell and candles from fright)
3. GET CANDLES (quickly!)
4. LIGHT MATCH
5. LIGHT CANDLES WITH MATCH
6. READ BOOK
7. Spirits are banished!
8. Enter Land of Living Dead, GET SKULL

**Timing is Critical:**
- Must complete sequence before spirits recover
- If you fail, pour water on bell to restore it (one retry only)

**If Candles Burned Out:**
- Game becomes unwinnable
- Don't carry lit candles; only light when ready

**Knowledge Required:**
- "Bell, book, and candle" = Catholic excommunication/exorcism rite
- Book mentions "noises, lights, and prayers"

### 5.12 The Egg, Thief, and Canary (MULTI-STEP PUZZLE)

**Puzzle:** Open the jeweled egg and get all its treasures

The egg cannot be opened by the player. It's a Faberge-style puzzle egg.

**Solution:**
1. GET EGG from tree nest
2. Drop egg somewhere the thief will find it (or give directly)
3. Thief steals and opens the egg
4. Later, kill thief (see section 9)
5. Recover OPENED EGG from Treasure Room
6. Go to tree (Up the Tree location)
7. WIND CANARY (clockwork canary inside egg)
8. Songbird in tree hears it
9. Songbird drops BRASS BAUBLE

**Critical Warning:**
- If you kill thief before he opens the egg = UNWINNABLE
- No warning; you won't know until trying to finish game
- This is considered one of the cruelest puzzles in gaming

### 5.13 Sandy Cave and Scarab

**Puzzle:** Find the buried treasure

**Solution:**
1. Go to Sandy Cave (via beach or other route)
2. DIG WITH SHOVEL (or just DIG)
3. Repeat digging until scarab appears
4. GET SCARAB

**Warning:**
- Random chance of being buried alive while digging
- SAVE before digging; restore if killed

### 5.14 Mirror Rooms (Teleportation)

**Puzzle:** Use mirrors for fast travel

**Solution:**
1. Go to Mirror Room (south area)
2. If carrying lit candles, PUT OUT CANDLES first
3. RUB MIRROR
4. Teleported to other Mirror Room (north area)
5. Works both directions

**Notes:**
- Useful shortcut between dam area and temple area
- Lit flames prevent teleportation

---

## 6. External Knowledge Requirements

### 6.1 Greek Mythology

**Required for:** Cyclops puzzle

**Knowledge Needed:**
- Homer's *Odyssey* (8th century BCE)
- Odysseus (Greek) / Ulysses (Roman) blinded the Cyclops Polyphemus
- The cyclops fears/respects Odysseus as his "master"

**In-Game Clue:**
- Prayer book contains acrostic poem spelling "ODYSSEUS"
- Requires recognizing acrostic pattern (first letter of each line)

### 6.2 Religious/Catholic Knowledge

**Required for:** Hades puzzle

**Knowledge Needed:**
- "Bell, book, and candle" is a traditional excommunication/exorcism rite
- Bell is rung, book is read, candles are extinguished
- Used to cast out evil spirits

**In-Game Clues:**
- Prayer book mentions "certain noises, lights, and prayers"
- Religious setting (temple, altar) suggests religious solutions

### 6.3 Real-World Physics

**Coal to Diamond:**
- Diamonds are carbon crystallized under extreme heat/pressure
- Coal is carbon-based

**Sharp Objects and Plastic:**
- Sharp items (screwdriver, sceptre) puncture inflatable plastic
- Items described as "sharp" in inventory

**Sound and Echoes:**
- Loud Room is loud due to water from dam (not explicitly stated)
- "Echo" command works but defies logical connection

**Hydraulics:**
- Dam controls water flow
- Opening gates drains reservoir over time
- Water levels affect other areas

### 6.4 Wordplay and Literary Devices

**Acrostics:**
- First letter of each line forms a word
- Prayer book poem spells "ODYSSEUS"

**Puns:**
- "You haven't a prayer" = both "no chance" and literal hint to pray
- "Echo" in Loud Room (self-referential word)

### 6.5 Adventure Game Conventions

**Maze Mapping:**
- Drop items to mark rooms
- Map non-Euclidean spaces systematically

**Save Scumming:**
- Save before random events (combat, digging)
- Restore unfavorable outcomes

**NPC Interactions:**
- Give items to characters
- Speak words/names found in text
- Enemies may be helpful (thief opens egg)

---

## 7. Death Traps

### Instant Deaths

| Death | Trigger | Prevention |
|-------|---------|------------|
| Grue | Dark location, 1-3 turns | Always carry lit light source |
| Boat Puncture | Board boat with sharp objects loose | Put sharp items in sack |
| Buried Alive | Random while digging | Save before digging |
| Combat Loss | Losing to thief or troll | Save first; fight when strong |
| Falling | Some climbing actions | Save before climbing |
| Dome Room | Going down without rope | Tie rope to railing first |
| Frigid River | Going over falls in boat | Exit boat at beach |
| Gas Room | Bringing flame into gas room | Extinguish light, navigate by touch |

### The Grue (Detailed)

- Iconic Zork enemy (never seen, only described)
- Lives in darkness
- "It is pitch black. You are likely to be eaten by a grue."
- After 1-3 turns in darkness: "Oh no! A lurking grue slithered into the room and ate you!"
- Only prevention is light
- Fair mechanic with clear warnings

---

## 8. Unwinnable States

These states make the game impossible to complete. No warning is given.

### 8.1 The Egg Disaster (MOST CRITICAL)

**Cause:** Killing the thief before he opens the egg

**Why Unwinnable:**
- Egg cannot be opened by player
- Canary inside is required treasure
- Bauble from canary is required treasure
- Total of 22 points lost = cannot reach 350

**Detection:** Only discovered when trying to complete game

**Prevention:**
- Let thief steal egg early
- Kill thief only after egg is opened
- Check that thief's treasure room has opened egg

### 8.2 Lamp Battery Exhaustion

**Cause:** Using lamp until battery dies before getting ivory torch

**Why Unwinnable:**
- Coal mine requires light
- Some areas have no alternative light
- Without diamond, cannot complete game

**Prevention:**
- Get ivory torch early (unlimited light)
- Turn off lamp in lit areas
- Don't waste battery on exploration

**Battery Warnings:**
- "Your lamp is getting dim"
- "Your lamp has died"

### 8.3 Candle Depletion

**Cause:** Burning candles before completing Hades puzzle

**Why Unwinnable:**
- Hades puzzle requires lit candles
- Only one set of candles in game
- Cannot banish spirits without them

**Prevention:**
- Never light candles until ready for Hades
- Don't carry lit candles around
- Complete Hades puzzle efficiently

### 8.4 Blue Button (Uncertain)

**Cause:** Pressing blue button at dam

**Effect:** Reportedly causes problems (specifics unclear)

**Prevention:** Don't press it; stick to yellow button

### 8.5 Lost Items

**Cause:** Dropping treasures where they can't be recovered

**Examples:**
- Items falling into chasms
- Items lost to failed puzzles

**Prevention:**
- Save frequently
- Be careful with item placement

---

## 9. The Thief

The thief is a unique NPC who serves multiple roles:

### Behavior

- **Wandering:** Appears randomly in underground locations
- **Stealing:** Takes items from your inventory and from rooms
- **Combat:** May attack you when encountered
- **Treasure Room:** Has a lair where he stores stolen goods
- **Egg Opening:** He can (and will) open the egg

### Strategy

**Early Game:**
- Avoid combat (you're too weak)
- Let him steal the egg (he'll open it)
- Don't carry treasures; store in trophy case quickly

**Mid Game:**
- Still avoid if possible
- Focus on exploration and puzzles
- Build up your score (makes you stronger)

**Late Game (Score 200+):**
- Hunt the thief
- Use nasty knife (slightly better than sword)
- Save before combat
- Fight repeatedly until he's dead

**After Killing Thief:**
- Go to Treasure Room
- Recover: chalice, opened egg, any stolen items
- Chalice is a unique treasure only found here

### Combat Tips

- Higher score = stronger in combat
- Nasty knife marginally better than sword
- Combat is random; save/restore as needed
- Thief becomes less aggressive at higher scores

---

## 10. Optimal Walkthrough

This walkthrough collects all 19 treasures efficiently.

### Phase 1: White House Setup

```
OPEN WINDOW
ENTER
GET LAMP, GET SWORD
MOVE RUG
OPEN TRAPDOOR
TURN ON LAMP
GO DOWN
```

### Phase 2: Kill Troll, Get Maze Treasure

```
S (to Troll Room)
KILL TROLL WITH SWORD (repeat until dead; save/restore if needed)
E, E (enter maze)
W, W, U
GET COINS, GET KEY
```

### Phase 3: Cyclops Shortcut

```
SW, E, S, SE (to Cyclops Room)
SAY ODYSSEUS (or ULYSSES)
(Cyclops flees, opens shortcut to Living Room)
```

### Phase 4: Deposit First Treasures, Get More Items

```
W (to Living Room via new shortcut)
PUT COINS IN TROPHY CASE
PUT SWORD IN TROPHY CASE (free up inventory; retrieve later if needed)
GO UP (to Attic via kitchen)
GET ROPE, GET KNIFE
GO DOWN
GET GARLIC, GET FOOD, GET WATER
```

### Phase 5: Get Egg, Set Thief Trap

```
(Go outside, to tree)
CLIMB TREE
GET EGG
(Drop egg where thief can find it, or give to him directly if encountered)
```

### Phase 6: Dam Operations

```
(Navigate to Dam area)
(Go to Maintenance Room)
GET WRENCH, GET SCREWDRIVER
(Go to Dam)
PUSH YELLOW BUTTON
TURN BOLT WITH WRENCH
(Wait for reservoir to drain)
(Go to Reservoir South)
GET TRUNK
(Go to Reservoir North)
GET PUMP
```

### Phase 7: Temple and Coffin

```
(Navigate to Temple)
GET BELL, GET BOOK, GET CANDLES
(Go to Egyptian Room)
GET COFFIN
(Open coffin)
GET SCEPTRE
(Go to Altar)
PRAY
(Teleported above ground with coffin)
(Deposit coffin and sceptre in trophy case)
```

### Phase 8: Torch and Gallery

```
(Return underground)
(Navigate to Torch Room)
GET TORCH
(Can turn off lamp now to save battery)
(Go to Gallery)
GET PAINTING
```

### Phase 9: Hades Puzzle

```
(With bell, book, candles, matches)
(Go to Entrance to Hades)
RING BELL
GET CANDLES
LIGHT MATCH
LIGHT CANDLES
READ BOOK
(Spirits banished)
(Enter Land of Living Dead)
GET SKULL
```

### Phase 10: Coal Mine and Diamond

```
(Navigate to Coal Mine)
GET COAL
(Go to Shaft Room)
PUT COAL IN BASKET
LOWER BASKET
(Navigate to Machine Room from below)
GET COAL
OPEN LID
PUT COAL IN MACHINE
CLOSE LID
TURN SWITCH WITH SCREWDRIVER
OPEN LID
GET DIAMOND
```

### Phase 11: Boat Journey

```
(Get deflated plastic from Dam Base)
INFLATE PLASTIC WITH PUMP
PUT SCREWDRIVER IN SACK
PUT SCEPTRE IN SACK (if carrying)
BOARD BOAT
LAUNCH
WAIT (repeat 4 times)
GET BUOY (during journey)
(At Sandy Beach)
EXIT BOAT
OPEN BUOY
GET EMERALD
```

### Phase 12: Sandy Cave

```
(Go to Sandy Cave)
SAVE
DIG (repeat until scarab appears; restore if buried)
GET SCARAB
```

### Phase 13: Loud Room

```
(Go to Loud Room)
ECHO
GET PLATINUM BAR
```

### Phase 14: Other Treasures

```
(Go to Atlantis Room)
GET TRIDENT
(Go to Bat Room)
GET JADE
(Go to Gas Room - no flame!)
GET BRACELET
```

### Phase 15: Rainbow

```
(Go to Aragain Falls or Rainbow End)
WAVE SCEPTRE
(Cross rainbow)
GET POT OF GOLD
```

### Phase 16: Kill Thief, Recover Treasures

```
(Build score to 200+ first)
(Hunt thief or wait in his territory)
KILL THIEF WITH KNIFE (repeat until dead; save/restore)
(Go to Treasure Room)
GET CHALICE
GET EGG (should be opened now)
(Recover any stolen items)
```

### Phase 17: Canary and Bauble

```
(Go to tree with opened egg)
CLIMB TREE
WIND CANARY
(Bird drops bauble)
GET BAUBLE
```

### Phase 18: Final Deposits

```
(Return to Living Room)
(Place all remaining treasures in trophy case)
(Score should reach 350)
(Map appears)
READ MAP
```

### Phase 19: Endgame

```
(Go southwest from White House area)
(New passage available)
ENTER STONE BARROW
(Game complete - Master Adventurer!)
```

---

## 11. Common Player Frustrations

Ranked by frequency of complaints:

### 1. The Egg Puzzle
- Undetectable unwinnable state
- Requires knowing thief can open it
- "Perhaps even crueler than usual"
- Players only discover problem at endgame

### 2. The Loud Room
- "ECHO" solution defies logic
- Alternative (drain reservoir) not obvious
- "Just doesn't make sense"
- Most walkthroughs consulted for this

### 3. The Thief
- Random appearances frustrating
- Steals items you need
- Combat outcomes random
- Must be killed but timing matters

### 4. Lamp Battery
- Silent countdown to unwinnable state
- Not obvious torch is unlimited
- Battery can run out before finding torch

### 5. Mazes
- Tedious mapping required
- Thief steals markers
- Non-logical directions
- "Tiresome" by modern standards

### 6. Parser Wrestling
- Finding exact command wording
- Synonyms not always recognized
- Verb-object combinations obscure

### 7. Timing Puzzles
- Hades ritual must be fast
- Reservoir drains on timer
- Easy to mess up sequences

### 8. Cultural References
- Odysseus/Cyclops mythology
- Bell, book, candle ritual
- Acrostic poetry form
- Players without this background stuck

### Design Philosophy Criticism

Quoted from analysis:
- "Slightly unfair, one-sided contest between smirking author and frustrated player"
- "Too difficult for someone's first introduction to adventuring"
- "Many puzzles just not well clued"
- "Random elements add frustration"

---

## 12. AI Agent Considerations

### Knowledge the Agent Must Have

**Cannot Complete Game Without:**
1. Odysseus/Cyclops mythology connection
2. Understanding thief must open egg (and keeping him alive until then)
3. Resource management (lamp battery awareness)
4. Multi-step puzzle chain awareness
5. All treasure locations
6. Unwinnable state prevention

**Very Difficult Without:**
1. Acrostic recognition capability
2. "Bell, book, candle" religious reference
3. "ECHO" solution (or reservoir timing)
4. Sharp objects + boat danger
5. Prayer as escape mechanism

### Strategic Awareness Requirements

**Unwinnable State Prevention:**
- Track egg status and thief status
- Monitor lamp battery
- Don't waste candles
- Avoid boat with loose sharp objects

**Resource Tracking:**
- Lamp turns remaining
- Candle status
- Treasure checklist
- Key items inventory

**NPC Understanding:**
- Thief behavior patterns
- Combat strength scaling with score
- Cyclops mythology weakness
- Spirit banishment requirements

### Puzzle Dependency Chains

```
Egg → Thief Opens → Canary → Wind in Tree → Bauble
         ↓
    Kill Thief → Chalice

Dam → Drain → Trunk
        ↓
      Pump → Boat → River → Emerald/Buoy
                      ↓
                   Beach → Scarab

Coffin → Sceptre → Rainbow → Pot of Gold

Coal → Machine → Diamond (requires screwdriver from dam)

Temple Items → Hades Ritual → Skull
```

### Recommended Agent Behaviors

**Exploration Phase:**
- Map systematically
- Examine everything
- Collect all items
- Identify locked/blocked passages

**Resource Phase:**
- Get torch early (unlimited light)
- Turn off lamp in lit areas
- Store treasures in trophy case quickly

**Puzzle Phase:**
- Solve dam first (enables multiple puzzles)
- Let thief steal egg before engaging
- Complete Hades before candles used
- Build score before fighting thief

**Endgame Phase:**
- Kill thief (high score = easier)
- Recover all stolen items
- Complete canary sequence
- Deposit final treasures

### Common AI Pitfalls to Avoid

1. **Killing thief too early** - Most critical error
2. **Wasting lamp battery** - Get torch early
3. **Burning candles** - Only light for Hades
4. **Boarding boat with sharp objects** - Use sack
5. **Not recognizing Odysseus clue** - Need mythology knowledge
6. **Typing commands in Loud Room** - ECHO specifically needed
7. **Not saving before random events** - Combat, digging
8. **Missing prayer escape** - Coffin seems stuck

### Hint System Design Considerations

When providing hints to the agent, consider:

**Nudge-level hints:**
- "The prayer book seems to have an unusual pattern"
- "The cyclops seems to have a troubled history"
- "The thief has nimble fingers"
- "Sharp objects and delicate materials don't mix"

**Medium hints:**
- "First letters can spell words"
- "Greek heroes had famous encounters with cyclopes"
- "Perhaps an enemy could open what you cannot"
- "Containers protect items from their surroundings"

**Direct hints (last resort):**
- "The acrostic spells ODYSSEUS"
- "Let the thief open the egg, then kill him"
- "Put sharp objects in the sack before boarding"
- "Type ECHO in the Loud Room"

---

## Appendix A: Command Reference

### Movement
`N, S, E, W, NE, NW, SE, SW, UP, DOWN, ENTER, EXIT, CLIMB, CROSS`

### Object Manipulation
`GET, DROP, PUT X IN Y, OPEN, CLOSE, EXAMINE, READ, TURN ON, TURN OFF`

### Combat
`KILL X WITH Y, ATTACK X, HIT X`

### Special Actions
`PRAY, WAVE X, RING X, LIGHT X, INFLATE X WITH Y, DIG, WIND X, LAUNCH, BOARD, EXIT`

### Communication
`SAY X, ODYSSEUS, ULYSSES, ECHO`

### System
`SAVE, RESTORE, QUIT, SCORE, INVENTORY (I), LOOK (L), AGAIN (G)`

---

## Appendix B: Map Notes

### Key Connections

- Cyclops Room ↔ Living Room (after cyclops flees)
- Mirror Room S ↔ Mirror Room N (via rubbing mirror)
- Altar → Above Ground (via PRAY)
- Trapdoor ↔ Cellar (can close behind you)

### One-Way Paths

- Dome Room → Torch Room (down, with rope)
- Frigid River → Beach (downstream only)
- Some maze exits

### Dangerous Transitions

- Gas Room with flame = explosion
- Dome Room down without rope = death
- River over falls = death

---

## Appendix C: Revision History

- Initial compilation: Based on comprehensive research of walkthroughs, guides, and community discussions
- Purpose: AI agent development reference for ZorkGPT project

---

*End of Guide*
