from typing import List, Dict, Set, Tuple

DIRECTION_MAPPING = {
    "n": "north",
    "north": "north",
    "northward": "north",
    "s": "south",
    "south": "south",
    "southward": "south",
    "e": "east",
    "east": "east",
    "eastward": "east",
    "w": "west",
    "west": "west",
    "westward": "west",
    "u": "up",
    "up": "up",
    "upward": "up",
    "climb up": "up",
    "go up": "up",
    "d": "down",
    "down": "down",
    "downward": "down",
    "climb down": "down",
    "go down": "down",
    "ne": "northeast",
    "northeast": "northeast",
    "nw": "northwest",
    "northwest": "northwest",
    "se": "southeast",
    "southeast": "southeast",
    "sw": "southwest",
    "southwest": "southwest",
    "in": "in",  # "enter" is often "in" contextually for directions
    "out": "out",
    # Specific "enter <noun>" or "exit <noun>" are usually handled as general actions if they cause room changes.
    # "climb ladder" could be "up" or "down", if more context available, it's better.
    # For now, "climb up" and "climb down" are explicit.
}

CANONICAL_DIRECTIONS = {
    "north",
    "south",
    "east",
    "west",
    "up",
    "down",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "in",
    "out",
}


def is_non_movement_command(action_str: str) -> bool:
    """
    Determines if an action is a non-movement command that should not create map connections.
    These are observation/interaction commands that don't change the player's location.

    Returns True if the action should NOT create a map connection.
    """
    if not action_str:
        return True

    action_lower = action_str.lower().strip()

    # Define non-movement command patterns
    non_movement_commands = {
        # Observation commands
        "look",
        "l",
        "examine",
        "x",
        "read",
        "search",
        "investigate",
        # Inventory commands
        "inventory",
        "i",
        "take",
        "get",
        "drop",
        "put",
        "give",
        # Interaction commands
        "open",
        "close",
        "push",
        "pull",
        "turn",
        "lift",
        "unlock",
        "lock",
        "break",
        "fix",
        "repair",
        "use",
        "activate",
        # Communication commands
        "say",
        "tell",
        "ask",
        "answer",
        "talk",
        "speak",
        # Meta commands
        "save",
        "restore",
        "quit",
        "help",
        "score",
        "time",
        "version",
        # Other interaction verbs
        "eat",
        "drink",
        "wear",
        "remove",
        "light",
        "extinguish",
        "ring",
        "knock",
        "kick",
        "hit",
        "attack",
        "kill",
        "touch",
    }

    # Check exact matches first
    if action_lower in non_movement_commands:
        return True

    # Check for commands that start with non-movement verbs
    for verb in non_movement_commands:
        if action_lower.startswith(verb + " "):
            return True

    # Check for "go" or movement patterns to explicitly allow them
    movement_patterns = [
        "go ",
        "walk ",
        "run ",
        "move ",
        "travel ",
        "head ",
        "proceed ",
        "climb ",
        "crawl ",
        "swim ",
        "fly ",
        "jump ",
        "step ",
        "enter ",
        "exit ",
        "leave ",
    ]

    for pattern in movement_patterns:
        if action_lower.startswith(pattern):
            return False  # This IS a movement command

    # If we have a recognized direction, it's movement
    if normalize_direction(action_str) is not None:
        return False  # This IS a movement command

    # Default: if unclear, assume it's NOT movement to be safe
    # This prevents spurious map connections
    return True


def normalize_direction(action_str: str) -> str | None:
    """
    Normalizes a player's action string to a canonical direction if it represents one.
    Returns the canonical direction string (e.g., "north") or None if not a clear direction.
    """
    if not action_str:
        return None

    action_lower = action_str.lower().strip()

    if action_lower in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[action_lower]

    # Handle "go <direction>" patterns and remove extra whitespace
    if action_lower.startswith("go "):
        parts = action_lower.split(" ", 1)
        if len(parts) > 1:
            potential_direction = parts[1].strip()
            if potential_direction in DIRECTION_MAPPING:
                return DIRECTION_MAPPING[potential_direction]

    # Add other common Zork phrases if necessary, e.g., "climb <object>" -> "up"/"down"
    # This can be expanded. For now, it covers many common cases.

    return None  # Not a recognized directional command


class Room:
    def __init__(self, name: str, base_name: str = None):
        self.name: str = name
        self.base_name: str = (
            base_name or name
        )  # Store the conceptual name without suffixes
        self.exits: Set[str] = set()  # Known exits from this room

    def add_exit(self, exit_name: str):
        self.exits.add(exit_name)

    def __repr__(self) -> str:
        return f"Room(name='{self.name}', base_name='{self.base_name}', exits={self.exits})"


class MapGraph:
    def __init__(self, logger=None):
        self.logger = logger
        self.rooms: Dict[str, Room] = {}
        # connections[room_name_1][exit_taken_from_room_1] = room_name_2
        self.connections: Dict[str, Dict[str, str]] = {}
        # Track confidence for each connection: (from_room, exit) -> confidence_score
        self.connection_confidence: Dict[Tuple[str, str], float] = {}
        # Track how many times each connection has been verified
        self.connection_verifications: Dict[Tuple[str, str], int] = {}
        # Track conflicts for analysis
        self.connection_conflicts: List[Dict] = []
        # Track whether new rooms have been added since last consolidation
        self.has_new_rooms_since_consolidation: bool = False
        # Track failed exit attempts: (room_name, exit) -> failure_count
        self.exit_failure_counts: Dict[Tuple[str, str], int] = {}
        # Track exits that have been permanently pruned to avoid re-adding them
        self.pruned_exits: Dict[str, Set[str]] = {}

    def _get_opposite_direction(self, direction: str) -> str:
        opposites = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
            "northeast": "southwest",
            "southwest": "northeast",
            "northwest": "southeast",
            "southeast": "northwest",
            "in": "out",
            "out": "in",
        }
        # Normalize to lower case for lookup
        normalized_direction = direction.lower()
        return opposites.get(normalized_direction)

    def _normalize_room_name(self, room_name: str) -> str:
        """
        Normalize room name for consistent storage and lookup.
        Preserves the original format but ensures consistent casing for keys.
        """
        if not room_name:
            return ""
        # Use title case for consistency (e.g., "West Of House")
        return " ".join(word.capitalize() for word in room_name.strip().split())

    def _extract_base_name(self, node_id: str) -> str:
        """
        Extract the base name from a (potentially suffixed) node_id string.

        Args:
            node_id: Node ID that may have suffixes like "(3-way: east-up-west)"

        Returns:
            The base name without any suffixes (e.g., "Kitchen Of White House")
        """
        if not node_id:
            return ""

        # Remove parenthetical suffixes
        base_name = node_id.split("(")[0].strip()

        # Apply normalization for consistency
        return self._normalize_room_name(base_name)

    def get_or_create_node_id(
        self, base_location_name: str, current_exits: List[str], description: str = ""
    ) -> str:
        """
        Get or create a node ID for a location, ensuring conceptual locations have stable IDs.

        This method attempts to find a compatible existing node based on base name matching
        and exit compatibility. If no compatible node exists, creates a new one.

        Args:
            base_location_name: The base name of the location (e.g., "Kitchen Of White House")
            current_exits: List of exits observed in the current turn
            description: Room description for generating new IDs if needed

        Returns:
            Node ID (either existing compatible one or newly created)
        """
        # Normalize the base location name
        normalized_base_name = self._normalize_room_name(base_location_name)

        # Normalize current exits into a canonical, sorted set
        normalized_current_exits = set()
        for exit_name in current_exits:
            if not exit_name or not exit_name.strip():
                continue
            norm_exit = normalize_direction(exit_name)
            if norm_exit:
                # Ensure all normalized directions are lowercase for consistency
                normalized_current_exits.add(norm_exit.lower())
            else:
                clean_exit = exit_name.strip()
                if clean_exit:
                    # Ensure non-directional exits are also lowercase for consistency
                    normalized_current_exits.add(clean_exit.lower())

        # Attempt to find a compatible existing node
        for existing_node_id, room_obj in self.rooms.items():
            # Get the base name for this existing room
            existing_base_name = (
                room_obj.base_name
                if hasattr(room_obj, "base_name") and room_obj.base_name
                else self._extract_base_name(existing_node_id)
            )

            # Check if base names match
            if normalized_base_name == existing_base_name:
                # Perform compatibility check with exits
                existing_exits = room_obj.exits

                # Check if the exits are compatible (either subset relationship or intersection)
                # This allows for progressive discovery of exits in the same room
                if (
                    normalized_current_exits == existing_exits
                    or existing_exits.issubset(normalized_current_exits)
                    or normalized_current_exits.issubset(existing_exits)
                    or (
                        normalized_current_exits
                        and existing_exits
                        and len(normalized_current_exits.intersection(existing_exits))
                        > 0
                    )
                ):
                    # Update the room's exits to include all observed exits (union)
                    union_exits = existing_exits.union(normalized_current_exits)
                    room_obj.exits = union_exits

                    return existing_node_id

        # No compatible existing node found, generate a new node ID
        new_node_id = self._create_unique_location_id(
            base_location_name, description, exits=list(normalized_current_exits)
        )

        # Create the new room with base_name stored
        self.add_room(new_node_id, base_name=normalized_base_name)

        return new_node_id

    def add_room(self, room_name: str, base_name: str = None) -> Room:
        # Use the room name as-is (it should already be a unique ID if needed)
        room_key = room_name
        if room_key not in self.rooms:
            self.rooms[room_key] = Room(name=room_key, base_name=base_name)
            # Flag that we have new rooms since last consolidation
            self.has_new_rooms_since_consolidation = True
        return self.rooms[room_key]

    def update_room_exits(self, room_name: str, new_exits: List[str]):
        # Use the room name as-is to match add_room behavior (no normalization)
        room_key = room_name
        if room_key not in self.rooms:
            self.add_room(room_name)

        # Normalize exit names for consistency
        normalized_new_exits = set()
        for exit_name in new_exits:
            if not exit_name or not exit_name.strip():
                continue  # Skip empty exits

            norm_exit = normalize_direction(
                exit_name
            )  # Try to normalize to canonical direction
            if norm_exit:
                # Use canonical direction (e.g., "north", "up") in lowercase
                normalized_new_exits.add(norm_exit.lower())
            else:
                # For non-directional exits, ensure lowercase consistency
                clean_exit = exit_name.strip()
                if clean_exit:
                    normalized_new_exits.add(clean_exit.lower())

        # Filter out exits that have been permanently pruned
        pruned_exits_for_room = self.pruned_exits.get(room_key, set())

        for exit_name in normalized_new_exits:
            # Don't re-add exits that have been pruned as invalid
            if exit_name not in pruned_exits_for_room:
                self.rooms[room_key].add_exit(exit_name)
            else:
                if self.logger:
                    self.logger.debug(
                        f"Skipping re-addition of pruned exit: {room_name} -> {exit_name}",
                        extra={
                            "event_type": "progress",
                            "stage": "map_building",
                            "details": f"Exit {exit_name} was previously pruned as invalid",
                        },
                    )

    def track_exit_failure(self, room_name: str, exit_name: str) -> int:
        """
        Track a failed exit attempt and return the current failure count.

        Args:
            room_name: The room where the exit was attempted
            exit_name: The exit that failed (will be normalized)

        Returns:
            The current failure count for this exit
        """
        # Use same normalization as other methods
        room_key = room_name
        normalized_action = normalize_direction(exit_name)
        processed_exit = (
            normalized_action if normalized_action else exit_name.lower().strip()
        )

        failure_key = (room_key, processed_exit)
        self.exit_failure_counts[failure_key] = (
            self.exit_failure_counts.get(failure_key, 0) + 1
        )

        failure_count = self.exit_failure_counts[failure_key]
        if self.logger:
            self.logger.debug(
                f"Exit failure tracked: {room_name} -> {processed_exit} (attempt #{failure_count})",
                extra={
                    "event_type": "progress",
                    "stage": "map_building",
                    "details": f"Exit failure count: {failure_count}",
                },
            )

        return failure_count

    def add_connection(
        self,
        from_room_name: str,
        exit_taken: str,
        to_room_name: str,
        confidence: float = 1.0,
    ):
        # Use the room names as-is (they should already be unique IDs if needed)
        from_room_key = from_room_name
        to_room_key = to_room_name
        self.add_room(from_room_name)
        self.add_room(to_room_name)

        # Use basic normalization for standard directions only
        # Let LLM layers handle semantic equivalence
        normalized_action = normalize_direction(exit_taken)
        processed_exit_taken = (
            normalized_action if normalized_action else exit_taken.lower().strip()
        )

        # Track confidence for this connection
        connection_key = (from_room_key, processed_exit_taken)

        # Check for existing connections and handle conflicts/verifications
        if (
            from_room_key in self.connections
            and processed_exit_taken in self.connections[from_room_key]
        ):
            existing_destination = self.connections[from_room_key][processed_exit_taken]

            if existing_destination == to_room_key:
                # Same connection verified again - increase confidence
                current_verifications = self.connection_verifications.get(
                    connection_key, 0
                )
                self.connection_verifications[connection_key] = (
                    current_verifications + 1
                )
                # Confidence increases with verifications but caps at 1.0
                self.connection_confidence[connection_key] = min(
                    1.0, self.connection_confidence.get(connection_key, 0.5) + 0.1
                )
                if self.logger:
                    self.logger.debug(
                        f"Map connection verified: {from_room_key} -> {processed_exit_taken} -> {to_room_key}",
                        extra={
                            "event_type": "progress",
                            "stage": "map_building",
                            "details": f"verifications: {self.connection_verifications[connection_key]}",
                        },
                    )
            else:
                # Conflicting connection - this is important to track
                existing_confidence = self.connection_confidence.get(
                    connection_key, 0.5
                )
                existing_verifications = self.connection_verifications.get(
                    connection_key, 0
                )

                conflict = {
                    "from_room": from_room_key,
                    "exit": processed_exit_taken,
                    "existing_destination": existing_destination,
                    "new_destination": to_room_key,
                    "existing_confidence": existing_confidence,
                    "new_confidence": confidence,
                    "existing_verifications": existing_verifications,
                    "new_verifications": 1,
                }
                self.connection_conflicts.append(conflict)

                if self.logger:
                    self.logger.warning(
                        f"Map conflict detected: {from_room_key} -> {processed_exit_taken}",
                        extra={
                            "event_type": "progress",
                            "stage": "map_building",
                            "details": f"Existing: {existing_destination} ({existing_confidence:.2f}, {existing_verifications}x) vs New: {to_room_key} ({confidence:.2f}, 1x)",
                        },
                    )

                # Enhanced conflict resolution logic
                should_update = False
                reason = ""

                if confidence > existing_confidence:
                    should_update = True
                    reason = "higher confidence"
                elif confidence == existing_confidence:
                    # When confidence is equal, prefer the connection with more verifications
                    if existing_verifications > 1:
                        should_update = False
                        reason = f"existing has more verifications ({existing_verifications} vs 1)"
                    else:
                        # Both have low verification count - this is suspicious
                        # Log this as a critical conflict that needs investigation
                        if self.logger:
                            self.logger.error(
                                "Critical map conflict: equal confidence and low verifications",
                                extra={
                                    "event_type": "progress",
                                    "stage": "map_building",
                                    "details": "Suggests inconsistent movement behavior or extraction errors",
                                },
                            )
                        should_update = False
                        reason = "keeping existing due to critical conflict (needs investigation)"
                else:
                    should_update = False
                    reason = "existing has higher confidence"

                if should_update:
                    if self.logger:
                        self.logger.info(
                            f"Using new connection ({reason})",
                            extra={
                                "event_type": "progress",
                                "stage": "map_building",
                                "details": f"{from_room_key} -> {processed_exit_taken} -> {to_room_key}",
                            },
                        )
                    self.connection_confidence[connection_key] = confidence
                    self.connection_verifications[connection_key] = 1
                else:
                    if self.logger:
                        self.logger.debug(
                            f"Keeping existing connection ({reason})",
                            extra={
                                "event_type": "progress",
                                "stage": "map_building",
                                "details": f"{from_room_key} -> {processed_exit_taken} -> {existing_destination}",
                            },
                        )
                    return  # Don't update the connection
        else:
            # New connection
            self.connection_confidence[connection_key] = confidence
            self.connection_verifications[connection_key] = 1

        # Add the forward connection
        if from_room_key not in self.connections:
            self.connections[from_room_key] = {}
        self.connections[from_room_key][processed_exit_taken] = to_room_key
        self.rooms[from_room_key].add_exit(
            processed_exit_taken
        )  # Ensure exit is recorded for the room

        # Add the reverse connection if an opposite direction exists
        opposite_exit = self._get_opposite_direction(processed_exit_taken)
        if opposite_exit:
            reverse_connection_key = (to_room_key, opposite_exit)

            if to_room_key not in self.connections:
                self.connections[to_room_key] = {}
            # Only add reverse connection if it doesn't overwrite an existing one from that direction
            # This handles cases where "north" from A leads to B, but "south" from B leads to C (unlikely but possible)
            if opposite_exit not in self.connections[to_room_key]:
                self.connections[to_room_key][opposite_exit] = from_room_key
                # Set confidence for reverse connection (slightly lower since it's inferred)
                self.connection_confidence[reverse_connection_key] = confidence * 0.9
                self.connection_verifications[reverse_connection_key] = 1
            self.rooms[to_room_key].add_exit(
                opposite_exit
            )  # Ensure reverse exit is recorded

    def get_room_info(self, room_name: str) -> str:
        normalized_name = self._normalize_room_name(room_name)
        if normalized_name not in self.rooms:
            return f"Room '{room_name}' is unknown."

        room = self.rooms[normalized_name]
        info_parts = [f"Current room: {room.name}."]

        if room.exits:
            exit_descriptions = []
            for exit_name in sorted(list(room.exits)):  # Sort for consistent output
                description = exit_name
                if (
                    normalized_name in self.connections
                    and exit_name in self.connections[normalized_name]
                ):
                    connected_room = self.connections[normalized_name][exit_name]
                    description += f" (leads to {connected_room})"
                elif (
                    self._get_opposite_direction(exit_name)
                    and self.connections.get(
                        self._get_opposite_direction(exit_name), {}
                    ).get(self._get_opposite_direction(exit_name))
                    == normalized_name
                ):
                    # This is a bit complex: trying to infer if this exit leads from somewhere else to here.
                    # Might be simpler to just list exits and let get_context_for_prompt handle "leads to"
                    pass  # Keep it simple for now
                exit_descriptions.append(description)
            if exit_descriptions:
                info_parts.append("Known exits: " + ", ".join(exit_descriptions) + ".")
            else:
                info_parts.append("No exits explicitly listed for this room yet.")
        else:
            info_parts.append("No exits known for this room yet.")

        return " ".join(info_parts)

    def get_context_for_prompt(
        self,
        current_room_name: str,
        previous_room_name: str = None,
        action_taken_to_current: str = None,
    ) -> str:
        context_parts = []

        if not current_room_name:  # Handle empty current_room_name explicitly
            context_parts.append(
                "Map: Current location name is missing. Cannot provide map context."
            )
            # Return immediately if current_room_name is missing, as no further processing is useful.
            return "--- Map Information ---\n" + "\n".join(context_parts)

        # Handle unique location IDs by extracting base name for display
        display_name = current_room_name
        if " (" in current_room_name and current_room_name.endswith(")"):
            display_name = current_room_name.split(" (")[0]

        current_room_normalized = current_room_name  # Use full unique ID for lookup
        room_known = current_room_normalized in self.rooms

        if room_known:
            room = self.rooms[current_room_normalized]
            context_parts.append(
                f"Current location: {display_name} (according to map)."
            )  # Added map context

            if previous_room_name and action_taken_to_current:
                action_desc = action_taken_to_current.lower()
                # Simple arrival string, could be enhanced by checking if previous_room_name is known
                context_parts.append(
                    f"You arrived from '{previous_room_name}' by going {action_desc}."
                )

            # Check for mapped exits and provide guidance about unmapped exits
            has_mapped_exits = (
                current_room_normalized in self.connections
                and self.connections[current_room_normalized]
            )

            has_detected_exits = room.exits and len(room.exits) > 0

            if has_mapped_exits:
                # Show confirmed working exits
                confirmed_exits = list(self.connections[current_room_normalized].keys())
                context_parts.append(
                    f"Confirmed working exits: {', '.join(sorted(confirmed_exits))}"
                )

            if has_detected_exits:
                # Show exits detected by extractor
                context_parts.append(f"Detected exits: {', '.join(sorted(room.exits))}")

            # Critical warning about unmapped exits
            if not has_mapped_exits and not has_detected_exits:
                context_parts.append(
                    "⚠️  WARNING: No exits detected, but Zork often has hidden exits!"
                )
                context_parts.append(
                    "🔍 MANDATORY: Test all cardinal directions (north, south, east, west, up, down) systematically."
                )
            elif not has_mapped_exits:
                context_parts.append(
                    "⚠️  NOTE: Exits detected but not yet confirmed by movement."
                )
                context_parts.append(
                    "🔍 RECOMMENDATION: Test detected exits AND try cardinal directions for hidden passages."
                )
            elif len(self.connections[current_room_normalized]) < 2:
                context_parts.append(
                    "🔍 TIP: Many Zork locations have additional unmapped exits. Try cardinal directions."
                )

        else:  # current_room_name is not in self.rooms
            context_parts.append(
                f"Map: Location '{display_name}' is new or not yet mapped. No detailed map data available for it yet."
            )
            context_parts.append(
                "🔍 NEW LOCATION PROTOCOL: Systematically test north, south, east, west, up, down to discover all exits."
            )

        if not context_parts:  # Should not be reached given the logic above
            # This might indicate an issue if current_room_name was provided but no conditions were met.
            # However, the current_room_name check at the start makes this unlikely.
            return "--- Map Information ---\nMap: No information available for the current context."

        return "--- Map Information ---\n" + "\n".join(context_parts)

    def render_ascii(self) -> str:
        if not self.rooms:
            return "-- Map is Empty --"

        output_lines = ["\n--- ASCII Map State ---"]
        output_lines.append("=======================")

        # Sort room names for consistent output order
        sorted_room_names = sorted(self.rooms.keys())

        for room_name in sorted_room_names:
            output_lines.append(f"\n[ {room_name} ]")

            room_obj = self.rooms.get(room_name)  # Get the Room object
            connections_exist = (
                room_name in self.connections and self.connections[room_name]
            )

            if connections_exist:
                # Sort exit actions for consistent output order
                sorted_exit_actions = sorted(self.connections[room_name].keys())
                for exit_action in sorted_exit_actions:
                    destination_room = self.connections[room_name][exit_action]
                    output_lines.append(
                        f"  --({exit_action})--> [ {destination_room} ]"
                    )

            # Also list exits known to the Room object but not yet in connections (unmapped)
            if room_obj and room_obj.exits:
                unmapped_exits = []
                for room_exit in sorted(list(room_obj.exits)):
                    # Check if this room_exit is already covered by a connection display
                    is_mapped = (
                        connections_exist and room_exit in self.connections[room_name]
                    )
                    if not is_mapped:
                        unmapped_exits.append(
                            f"  --({room_exit})--> ??? (Destination Unknown)"
                        )

                if unmapped_exits:
                    if (
                        not connections_exist
                    ):  # Avoid printing "Exits:" twice if no connections
                        # No specific header needed if only unmapped exits, they stand alone
                        pass
                    output_lines.extend(unmapped_exits)

            if not connections_exist and (not room_obj or not room_obj.exits):
                output_lines.append("  (No exits known or mapped from this room)")

        output_lines.append("\n=======================")
        output_lines.append("--- End of Map State ---")
        return "\n".join(output_lines)

    def render_mermaid(self) -> str:
        """
        Render the map as a Mermaid diagram, which is easier for LLMs to parse and understand.

        Returns:
            Mermaid diagram syntax as a string
        """
        if not self.rooms:
            return "graph LR\n    A[No rooms mapped yet]"

        lines = ["graph LR"]

        # Create node definitions with sanitized IDs
        room_to_id = {}
        node_counter = 1

        # First pass: create node IDs and definitions
        sorted_room_names = sorted(self.rooms.keys())
        for room_name in sorted_room_names:
            node_id = f"R{node_counter}"
            room_to_id[room_name] = node_id
            # Use the full room name (unique ID) for proper matching with current_room
            # Sanitize room name for Mermaid (escape special characters)
            sanitized_name = (
                room_name.replace('"', '\\"').replace("[", "\\[").replace("]", "\\]")
            )
            lines.append(f'    {node_id}["{sanitized_name}"]')
            node_counter += 1

        # Second pass: create connections
        connection_lines = []
        for room_name in sorted_room_names:
            if room_name in self.connections:
                from_id = room_to_id[room_name]
                # Sort exit actions for consistent output
                sorted_exits = sorted(self.connections[room_name].keys())
                for exit_action in sorted_exits:
                    destination_room = self.connections[room_name][exit_action]
                    if destination_room in room_to_id:
                        to_id = room_to_id[destination_room]
                        # Sanitize exit action for Mermaid
                        sanitized_action = exit_action.replace('"', '\\"')
                        connection_lines.append(
                            f'    {from_id} -->|"{sanitized_action}"| {to_id}'
                        )
                    else:
                        # Create a temporary node for unknown destinations
                        unknown_id = f"U{node_counter}"
                        sanitized_dest = (
                            destination_room.replace('"', '\\"')
                            .replace("[", "\\[")
                            .replace("]", "\\]")
                        )
                        lines.append(f'    {unknown_id}["{sanitized_dest} (Unknown)"]')
                        sanitized_action = exit_action.replace('"', '\\"')
                        connection_lines.append(
                            f'    {from_id} -->|"{sanitized_action}"| {unknown_id}'
                        )
                        node_counter += 1

        # Add all connections
        lines.extend(connection_lines)

        # Add unmapped exits as dotted connections to unknown destinations
        unmapped_counter = 1
        for room_name in sorted_room_names:
            room_obj = self.rooms.get(room_name)
            if room_obj and room_obj.exits:
                from_id = room_to_id[room_name]
                for room_exit in sorted(list(room_obj.exits)):
                    # Check if this exit is already mapped
                    is_mapped = (
                        room_name in self.connections
                        and room_exit in self.connections[room_name]
                    )
                    if not is_mapped:
                        unknown_id = f"UNK{unmapped_counter}"
                        lines.append(f'    {unknown_id}["Unknown Destination"]')
                        sanitized_exit = room_exit.replace('"', '\\"')
                        lines.append(
                            f'    {from_id} -.->|"{sanitized_exit}"| {unknown_id}'
                        )
                        unmapped_counter += 1

        return "\n".join(lines)

    def get_high_confidence_connections(
        self, min_confidence: float = 0.7
    ) -> Dict[str, Dict[str, str]]:
        """Get only connections that meet the minimum confidence threshold."""
        high_confidence_connections = {}

        for room_name, exits in self.connections.items():
            high_confidence_exits = {}
            for exit_name, destination in exits.items():
                connection_key = (room_name, exit_name)
                confidence = self.connection_confidence.get(connection_key, 0.5)

                if confidence >= min_confidence:
                    high_confidence_exits[exit_name] = destination

            if high_confidence_exits:
                high_confidence_connections[room_name] = high_confidence_exits

        return high_confidence_connections

    def get_connection_confidence(self, from_room: str, exit_taken: str) -> float:
        """Get the confidence score for a specific connection."""
        # Use the full unique ID (no normalization for unique IDs)
        from_room_key = from_room
        # Use same normalization logic as add_connection
        normalized_action = normalize_direction(exit_taken)
        processed_exit_taken = (
            normalized_action if normalized_action else exit_taken.lower().strip()
        )
        connection_key = (from_room_key, processed_exit_taken)
        return self.connection_confidence.get(connection_key, 0.0)

    def get_map_quality_metrics(self) -> Dict[str, float]:
        """Get metrics about the overall quality of the map."""
        if not self.connection_confidence:
            return {
                "average_confidence": 0.0,
                "high_confidence_ratio": 0.0,
                "total_connections": 0,
                "verified_connections": 0,
                "conflicts_detected": 0,
            }

        confidences = list(self.connection_confidence.values())
        average_confidence = sum(confidences) / len(confidences)
        high_confidence_count = sum(1 for c in confidences if c >= 0.7)
        high_confidence_ratio = high_confidence_count / len(confidences)
        verified_connections = sum(
            1 for v in self.connection_verifications.values() if v > 1
        )

        return {
            "average_confidence": average_confidence,
            "high_confidence_ratio": high_confidence_ratio,
            "total_connections": len(confidences),
            "verified_connections": verified_connections,
            "conflicts_detected": len(self.connection_conflicts),
        }

    def render_confidence_report(self) -> str:
        """Generate a detailed confidence report for the map."""
        metrics = self.get_map_quality_metrics()

        report = [
            "🗺️  MAP CONFIDENCE REPORT",
            "=" * 40,
            f"Total Connections: {metrics['total_connections']}",
            f"Average Confidence: {metrics['average_confidence']:.2f}",
            f"High Confidence (≥0.7): {metrics['high_confidence_ratio']:.1%}",
            f"Verified Connections: {metrics['verified_connections']}",
            f"Conflicts Detected: {metrics['conflicts_detected']}",
            "",
        ]

        if self.connection_conflicts:
            report.append("⚠️  CONFLICTS DETECTED:")
            for i, conflict in enumerate(
                self.connection_conflicts[-5:], 1
            ):  # Show last 5
                report.append(f"  {i}. {conflict['from_room']} -> {conflict['exit']}")
                report.append(
                    f"     Old: {conflict['existing_destination']} ({conflict['existing_confidence']:.2f})"
                )
                report.append(
                    f"     New: {conflict['new_destination']} ({conflict['new_confidence']:.2f})"
                )
            if len(self.connection_conflicts) > 5:
                report.append(f"     ... and {len(self.connection_conflicts) - 5} more")
            report.append("")

        # Show most confident connections
        high_conf_connections = self.get_high_confidence_connections(0.8)
        if high_conf_connections:
            report.append("✅ HIGH CONFIDENCE PATHS:")
            for room, exits in list(high_conf_connections.items())[:10]:  # Show top 10
                for exit, dest in exits.items():
                    conf = self.get_connection_confidence(room, exit)
                    verifications = self.connection_verifications.get((room, exit), 0)
                    report.append(
                        f"  {room} -> {exit} -> {dest} ({conf:.2f}, {verifications}x verified)"
                    )
            report.append("")

        return "\n".join(report)

    def get_navigation_suggestions(self, current_room: str) -> List[Dict]:
        """Get navigation suggestions based on confidence scores."""
        # Use the full unique ID for lookup (no normalization)
        current_room_key = current_room
        suggestions = []

        if current_room_key in self.connections:
            for exit, destination in self.connections[current_room_key].items():
                confidence = self.get_connection_confidence(current_room, exit)
                verifications = self.connection_verifications.get(
                    (current_room_key, exit), 0
                )

                suggestions.append(
                    {
                        "exit": exit,
                        "destination": destination,
                        "confidence": confidence,
                        "verifications": verifications,
                        "recommendation": self._get_recommendation(
                            confidence, verifications
                        ),
                    }
                )

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions

    def _get_recommendation(self, confidence: float, verifications: int) -> str:
        """Get a recommendation based on confidence and verification count."""
        if confidence >= 0.9 and verifications > 2:
            return "HIGHLY_RELIABLE"
        elif confidence >= 0.7 and verifications > 1:
            return "RELIABLE"
        elif confidence >= 0.5:
            return "MODERATE"
        else:
            return "UNCERTAIN"

    def _create_unique_location_id(
        self,
        location_name: str,
        description: str = "",
        objects: List[str] = None,
        exits: List[str] = None,
    ) -> str:
        """
        Create a stable unique identifier for a location that handles cases where multiple
        locations have the same name but different characteristics.

        This version prioritizes exit patterns over descriptions since exits are more stable
        and less volatile than room descriptions which can change based on objects, lighting, etc.

        Args:
            location_name: The base location name (e.g., "Clearing")
            description: Full location description text (used sparingly)
            objects: List of visible objects in the location (mostly ignored)
            exits: List of available exits from the location (primary differentiator)

        Returns:
            Stable location identifier based primarily on exit patterns
        """
        if not location_name:
            return ""

        base_name = self._normalize_room_name(location_name)

        # PRIMARY APPROACH: Use exit patterns as the main differentiator
        # Exits are much more stable than descriptions or objects
        if exits:
            # Normalize exits to canonical directions with consistent lowercase
            normalized_exits = set()
            for exit in exits:
                if not exit or not exit.strip():
                    continue
                norm_exit = normalize_direction(exit)
                if norm_exit:
                    # Ensure all normalized directions are lowercase for consistency
                    normalized_exits.add(norm_exit.lower())
                else:
                    # Keep non-directional exits (like "window", "trapdoor") as lowercase
                    normalized_exits.add(exit.lower().strip())

            if normalized_exits:
                # Create distinctive patterns based on exit combinations
                exit_count = len(normalized_exits)
                # Sort exits in a consistent manner for deterministic IDs
                sorted_exits = sorted(list(normalized_exits))

                # Single exit rooms (dead ends) - highly distinctive
                if exit_count == 1:
                    return f"{base_name} ({sorted_exits[0]} only)"

                # Two-exit rooms (corridors) - very distinctive
                elif exit_count == 2:
                    exit_pair = tuple(sorted_exits)
                    if exit_pair == ("east", "west"):
                        return f"{base_name} (east-west corridor)"
                    elif exit_pair == ("north", "south"):
                        return f"{base_name} (north-south corridor)"
                    elif exit_pair == ("down", "up"):
                        return f"{base_name} (vertical passage)"
                    else:
                        # Other two-exit combinations
                        return f"{base_name} ({'-'.join(sorted_exits)})"

                # Three-exit rooms - moderately distinctive
                elif exit_count == 3:
                    # Check for common three-way patterns
                    normalized_exits_set = set(sorted_exits)
                    if {"north", "east", "south"}.issubset(normalized_exits_set):
                        return f"{base_name} (t-junction east)"
                    elif {"north", "west", "south"}.issubset(normalized_exits_set):
                        return f"{base_name} (t-junction west)"
                    elif {"east", "west", "north"}.issubset(normalized_exits_set):
                        return f"{base_name} (t-junction north)"
                    elif {"east", "west", "south"}.issubset(normalized_exits_set):
                        return f"{base_name} (t-junction south)"
                    else:
                        # Other three-exit combinations - ensure consistent lowercase
                        return f"{base_name} (3-way: {'-'.join(sorted_exits[:3])})"

                # Four or more exits - use count-based identifier
                elif exit_count >= 4:
                    if exit_count == 4 and {"north", "south", "east", "west"}.issubset(
                        normalized_exits
                    ):
                        return f"{base_name} (4-way intersection)"
                    else:
                        return f"{base_name} ({exit_count}-way junction)"

        # SECONDARY APPROACH: Only use descriptions for truly permanent, structural features
        # Avoid volatile content like objects, lighting, or temporary states
        # REMOVED: All hardcoded location-specific strings to maintain LLM-First Design
        # The LLM extractor should handle location identification, not hardcoded rules

        # AVOID: Volatile features that change frequently
        # - Objects that can be picked up/dropped
        # - Lighting conditions ("dimly lit", "dark")
        # - Temporary states ("open door", "closed window")
        # - Minor object detection variations
        # - Hardcoded location names that won't help with unseen areas

        # Default: return the base name without modification
        # This ensures the same room gets the same ID unless there are
        # truly distinctive permanent features discovered through exit patterns
        return base_name

    def needs_consolidation(self) -> bool:
        """Check if consolidation is needed based on new room additions or fragmentation patterns."""
        # Original condition: new rooms added
        if self.has_new_rooms_since_consolidation:
            return True

        # Enhanced condition: detect base name fragmentation
        from collections import defaultdict

        base_name_groups = defaultdict(list)

        for location_name in self.rooms.keys():
            base_name = self._extract_base_name(location_name).lower()
            base_name_groups[base_name].append(location_name)

        # Check if any base name has multiple variants
        for base_name, variants in base_name_groups.items():
            if len(variants) > 1:
                return True  # Fragmentation detected

        # Check for case variations
        normalized_groups = defaultdict(list)
        for location_name in self.rooms.keys():
            normalized_groups[location_name.lower()].append(location_name)

        for normalized_name, variants in normalized_groups.items():
            if len(variants) > 1:
                return True  # Case variations detected

        return False

    def consolidate_similar_locations(self) -> int:
        """
        Automatically detect and consolidate locations that are likely the same place
        but have different unique IDs due to extractor inconsistencies.

        This includes:
        1. Same base name with different suffixes
        2. Same base name and suffix pattern but different case

        Returns:
            Number of consolidations performed
        """
        from collections import defaultdict

        # Reset the flag since we're about to consolidate
        self.has_new_rooms_since_consolidation = False

        # Group locations by their normalized full name to catch case variations
        normalized_groups = defaultdict(list)

        for location_name in self.rooms.keys():
            # Normalize the entire location name (including suffixes) to lowercase
            normalized_full_name = location_name.lower()
            normalized_groups[normalized_full_name].append(location_name)

        consolidations_performed = 0

        # Find groups with multiple variants (case variations)
        for normalized_name, variants in normalized_groups.items():
            if len(variants) <= 1:
                continue  # No case variations for this location

            if self.logger:
                self.logger.info(
                    f"Consolidating case variations: {normalized_name}",
                    extra={
                        "event_type": "progress",
                        "stage": "map_consolidation",
                        "details": f"Variants found: {variants}",
                    },
                )

            # Choose the consolidation target - prefer the one that matches our normalization style
            # Prefer Title Case for base names and consistent patterns for suffixes
            target_location = self._choose_best_variant(variants)

            # Collect all exits from variants
            all_exits = set()
            for variant in variants:
                if variant in self.rooms:
                    all_exits.update(self.rooms[variant].exits)

            if self.logger:
                self.logger.debug(
                    f"Consolidation target: {target_location}",
                    extra={
                        "event_type": "progress",
                        "stage": "map_consolidation",
                        "details": f"Combined exits: {sorted(list(all_exits))}",
                    },
                )

            # Merge all connections from variants into the target
            for variant in variants:
                if variant == target_location:
                    continue  # Skip the target itself

                if variant in self.connections:
                    # Move outgoing connections from variant to target
                    for exit_action, destination in self.connections[variant].items():
                        if self.logger:
                            self.logger.debug(
                                f"Moving connection during consolidation: {variant} -> {exit_action} -> {destination}",
                                extra={
                                    "event_type": "progress",
                                    "stage": "map_consolidation",
                                },
                            )
                        self.add_connection(target_location, exit_action, destination)

                    # Remove the old connections
                    del self.connections[variant]

                # Update incoming connections that point to this variant
                for from_location, exits in self.connections.items():
                    for exit_action, destination in list(exits.items()):
                        if destination == variant:
                            if self.logger:
                                self.logger.debug(
                                    f"Redirecting connection during consolidation: {from_location} -> {exit_action} -> {variant} => {target_location}",
                                    extra={
                                        "event_type": "progress",
                                        "stage": "map_consolidation",
                                    },
                                )
                            exits[exit_action] = target_location

                # Remove the variant room if it's not the target
                if variant in self.rooms:
                    del self.rooms[variant]

                consolidations_performed += 1

            # Update the target location with all collected exits
            if target_location in self.rooms:
                self.rooms[target_location].exits = all_exits
                # Ensure the target has the correct base_name
                room_obj = self.rooms[target_location]
                if not hasattr(room_obj, "base_name") or not room_obj.base_name:
                    room_obj.base_name = self._extract_base_name(target_location)
            else:
                # Create the target location if it doesn't exist
                base_name = self._extract_base_name(target_location)
                self.add_room(target_location, base_name=base_name)
                self.rooms[target_location].exits = all_exits

        if consolidations_performed > 0:
            if self.logger:
                self.logger.info(
                    f"Consolidation complete: {consolidations_performed} locations merged",
                    extra={"event_type": "progress", "stage": "map_consolidation"},
                )

        return consolidations_performed

    def _choose_best_variant(self, variants: List[str]) -> str:
        """
        Choose the best variant for consolidation target.
        Prefer consistent capitalization and clean formatting.
        """
        if len(variants) == 1:
            return variants[0]

        # Scoring system for variant quality
        def score_variant(variant):
            score = 0

            # Prefer variants without parentheses (base names)
            if "(" not in variant:
                score += 1000

            # Prefer consistent Title Case in base name
            base_name = variant.split("(")[0].strip()
            words = base_name.split()
            if all(word[0].isupper() and word[1:].islower() for word in words if word):
                score += 100

            # Prefer lowercase in suffixes (our standard)
            if "(" in variant:
                suffix = variant[variant.find("(") :]
                # Count lowercase words in suffix
                suffix_words = (
                    suffix.replace("(", "")
                    .replace(")", "")
                    .replace("-", " ")
                    .replace(":", " ")
                    .split()
                )
                lowercase_count = sum(1 for word in suffix_words if word.islower())
                score += lowercase_count * 10

            # Prefer shorter variants (less verbose)
            score -= len(variant)

            return score

        # Choose the variant with the highest score
        best_variant = max(variants, key=score_variant)
        return best_variant

    def force_consolidation(self) -> int:
        """
        Force consolidation of similar locations regardless of the needs_consolidation flag.

        This is useful for fixing existing fragmented maps or when manual consolidation is needed.

        Returns:
            Number of consolidations performed
        """
        if self.logger:
            self.logger.info(
                "Forcing map consolidation (bypassing needs_consolidation flag)",
                extra={"event_type": "progress", "stage": "map_consolidation"},
            )

        # Temporarily set the flag to ensure consolidation runs
        old_flag = self.has_new_rooms_since_consolidation
        self.has_new_rooms_since_consolidation = True

        # Run consolidation
        consolidations = self.consolidate_similar_locations()

        # Don't restore the old flag since consolidation resets it

        return consolidations

    def consolidate_base_name_variants(self) -> int:
        """
        Enhanced consolidation that groups rooms by base name and merges variants.

        This addresses the main source of fragmentation: rooms with the same base location
        but different suffixes (e.g., "Forest Path" vs "Forest Path (3-way: north-south-tree)").

        Returns:
            Number of consolidations performed
        """
        from collections import defaultdict

        if self.logger:
            self.logger.info(
                "Enhanced base name consolidation starting",
                extra={"event_type": "progress", "stage": "map_consolidation"},
            )

        # Group locations by their base name
        base_name_groups = defaultdict(list)

        for location_name in self.rooms.keys():
            base_name = self._extract_base_name(location_name).lower()
            base_name_groups[base_name].append(location_name)

        consolidations_performed = 0

        # Process each base name group with multiple variants
        for base_name, variants in base_name_groups.items():
            if len(variants) <= 1:
                continue  # No variants to consolidate

            if self.logger:
                self.logger.debug(
                    f"Consolidating base name variants: {base_name}",
                    extra={
                        "event_type": "progress",
                        "stage": "map_consolidation",
                        "details": f"Variants: {variants}",
                    },
                )

            # Choose the best variant as the consolidation target
            target_location = self._choose_best_base_name_variant(variants)

            # Collect all exits from all variants
            all_exits = set()
            for variant in variants:
                if variant in self.rooms:
                    all_exits.update(self.rooms[variant].exits)

            if self.logger:
                self.logger.debug(
                    f"Consolidation target selected: {target_location}",
                    extra={
                        "event_type": "progress",
                        "stage": "map_consolidation",
                        "details": f"Combined exits: {sorted(list(all_exits))}",
                    },
                )

            # Merge all connections from variants into the target
            for variant in variants:
                if variant == target_location:
                    continue  # Skip the target itself

                # Move outgoing connections from variant to target
                if variant in self.connections:
                    for exit_action, destination in self.connections[variant].items():
                        if self.logger:
                            self.logger.debug(
                                f"Moving connection during consolidation: {variant} -> {exit_action} -> {destination}",
                                extra={
                                    "event_type": "progress",
                                    "stage": "map_consolidation",
                                },
                            )
                        self.add_connection(target_location, exit_action, destination)

                    # Remove the old connections
                    del self.connections[variant]

                # Update incoming connections that point to this variant
                for from_location, exits in self.connections.items():
                    for exit_action, destination in list(exits.items()):
                        if destination == variant:
                            if self.logger:
                                self.logger.debug(
                                    f"Redirecting connection during consolidation: {from_location} -> {exit_action} -> {variant} => {target_location}",
                                    extra={
                                        "event_type": "progress",
                                        "stage": "map_consolidation",
                                    },
                                )
                            exits[exit_action] = target_location

                # Remove the variant room if it's not the target
                if variant in self.rooms:
                    if self.logger:
                        self.logger.debug(
                            f"Removing variant during consolidation: {variant}",
                            extra={
                                "event_type": "progress",
                                "stage": "map_consolidation",
                            },
                        )
                    del self.rooms[variant]

                consolidations_performed += 1

            # Update the target location with all collected exits
            if target_location in self.rooms:
                self.rooms[target_location].exits = all_exits
                # Ensure the target has the correct base_name
                room_obj = self.rooms[target_location]
                if not hasattr(room_obj, "base_name") or not room_obj.base_name:
                    room_obj.base_name = self._extract_base_name(target_location)
            else:
                # Create the target location if it doesn't exist
                extracted_base_name = self._extract_base_name(target_location)
                self.add_room(target_location, base_name=extracted_base_name)
                self.rooms[target_location].exits = all_exits

        if consolidations_performed > 0:
            if self.logger:
                self.logger.info(
                    f"Base name consolidation complete: {consolidations_performed} locations merged",
                    extra={"event_type": "progress", "stage": "map_consolidation"},
                )
        else:
            if self.logger:
                self.logger.debug(
                    "No base name variants found to consolidate",
                    extra={"event_type": "progress", "stage": "map_consolidation"},
                )

        return consolidations_performed

    def _choose_best_base_name_variant(self, variants: List[str]) -> str:
        """
        Choose the best variant for base name consolidation.

        Prioritizes:
        1. Base names without suffixes (simplest form)
        2. Well-formed suffixes that provide navigation info
        3. Shorter, cleaner names
        """
        if len(variants) == 1:
            return variants[0]

        def score_base_name_variant(variant):
            score = 0

            # Strongly prefer variants without parentheses (pure base names)
            if "(" not in variant:
                score += 2000

            # Prefer consistent Title Case in base name
            base_name = variant.split("(")[0].strip()
            words = base_name.split()
            if all(word[0].isupper() and word[1:].islower() for word in words if word):
                score += 500

            # Evaluate suffix quality if present
            if "(" in variant:
                suffix = variant[variant.find("(") :]

                # Prefer suffixes that describe navigation topology
                navigation_keywords = [
                    "junction",
                    "corridor",
                    "intersection",
                    "way",
                    "passage",
                ]
                if any(keyword in suffix.lower() for keyword in navigation_keywords):
                    score += 200

                # Prefer suffixes with direction information
                direction_keywords = ["north", "south", "east", "west", "up", "down"]
                direction_count = sum(
                    1 for keyword in direction_keywords if keyword in suffix.lower()
                )
                score += direction_count * 50

                # Prefer lowercase in suffixes (our standard)
                suffix_words = (
                    suffix.replace("(", "")
                    .replace(")", "")
                    .replace("-", " ")
                    .replace(":", " ")
                    .split()
                )
                lowercase_count = sum(1 for word in suffix_words if word.islower())
                score += lowercase_count * 10

            # Prefer shorter variants (less verbose)
            score -= len(variant) * 2

            return score

        # Choose the variant with the highest score
        best_variant = max(variants, key=score_base_name_variant)
        if self.logger:
            self.logger.debug(
                f"Selected best variant: {best_variant}",
                extra={
                    "event_type": "progress",
                    "stage": "map_consolidation",
                    "details": f"From variants: {variants}",
                },
            )
        return best_variant

    def prune_fragmented_nodes(self) -> int:
        """
        Identify and remove fragmented nodes that serve no navigation purpose.

        Removes:
        1. Nodes with no exits and no incoming connections (isolated dead ends)
        2. Nodes that have only outgoing connections to "Unknown Destination" but no real connections

        Preserves:
        1. Nodes that have real incoming connections (they serve as destinations)
        2. Nodes that have real outgoing connections (they provide navigation options)
        3. Unknown destination placeholders (they represent future exploration potential)

        Returns:
            Number of nodes pruned
        """
        pruned_count = 0

        # Find all nodes that have incoming connections (are destinations)
        nodes_with_incoming = set()
        for from_room, exits in self.connections.items():
            for exit_action, destination in exits.items():
                if not destination.startswith("Unknown Destination"):
                    nodes_with_incoming.add(destination)

        # Identify candidates for pruning
        candidates_for_pruning = []

        for room_name in list(self.rooms.keys()):
            room = self.rooms[room_name]

            # Skip if this is an "Unknown Destination" placeholder - we want to keep these
            if room_name.startswith("Unknown Destination"):
                continue

            # Case 1: Node has no exits at all
            if not room.exits or len(room.exits) == 0:
                # Only prune if it also has no incoming connections
                if room_name not in nodes_with_incoming:
                    candidates_for_pruning.append(
                        (room_name, "no exits, no incoming connections")
                    )
                    continue

            # Case 2: Node has exits but all outgoing connections go to unknown destinations
            if room_name in self.connections:
                outgoing_connections = self.connections[room_name]
                real_connections = [
                    dest
                    for dest in outgoing_connections.values()
                    if not dest.startswith("Unknown Destination")
                ]

                if len(real_connections) == 0 and room_name not in nodes_with_incoming:
                    # All connections go to unknown destinations and no one connects TO this room
                    candidates_for_pruning.append(
                        (room_name, "only unknown destinations, no incoming")
                    )

        # Perform the pruning
        for room_name, reason in candidates_for_pruning:
            if self.logger:
                self.logger.debug(
                    f"Pruning fragmented node: {room_name} ({reason})",
                    extra={"event_type": "progress", "stage": "map_pruning"},
                )

            # Remove from rooms
            if room_name in self.rooms:
                del self.rooms[room_name]

            # Remove from connections
            if room_name in self.connections:
                del self.connections[room_name]

            # Remove any remaining incoming connections (shouldn't be any based on our logic)
            for from_room, exits in self.connections.items():
                exits_to_remove = [
                    exit_action
                    for exit_action, destination in exits.items()
                    if destination == room_name
                ]
                for exit_action in exits_to_remove:
                    if self.logger:
                        self.logger.debug(
                            f"Removing stale connection: {from_room} -> {exit_action} -> {room_name}",
                            extra={"event_type": "progress", "stage": "map_pruning"},
                        )
                    del exits[exit_action]

            pruned_count += 1

        if pruned_count > 0:
            if self.logger:
                self.logger.info(
                    f"Pruning complete: {pruned_count} fragmented nodes removed",
                    extra={"event_type": "progress", "stage": "map_pruning"},
                )
        else:
            if self.logger:
                self.logger.debug(
                    "No fragmented nodes found to prune",
                    extra={"event_type": "progress", "stage": "map_pruning"},
                )

        return pruned_count

    def get_fragmentation_report(self) -> str:
        """
        Generate a report on map fragmentation issues.

        Returns:
            Human-readable report of fragmentation status
        """
        report_lines = ["🔍 MAP FRAGMENTATION REPORT", "=" * 40]

        # Count nodes with no exits
        empty_exit_nodes = [
            name
            for name, room in self.rooms.items()
            if not room.exits or len(room.exits) == 0
        ]

        # Count nodes with incoming connections
        nodes_with_incoming = set()
        for from_room, exits in self.connections.items():
            for exit_action, destination in exits.items():
                if not destination.startswith("Unknown Destination"):
                    nodes_with_incoming.add(destination)

        # Count isolated nodes (no exits, no incoming)
        isolated_nodes = [
            name for name in empty_exit_nodes if name not in nodes_with_incoming
        ]

        # Count unknown destination placeholders
        unknown_destinations = sum(
            1
            for exits in self.connections.values()
            for dest in exits.values()
            if dest.startswith("Unknown Destination")
        )

        # Count base name variations
        from collections import defaultdict

        base_name_groups = defaultdict(list)
        for room_name in self.rooms.keys():
            base_name = self._extract_base_name(room_name)
            base_name_groups[base_name].append(room_name)

        fragmented_base_names = {
            base: variants
            for base, variants in base_name_groups.items()
            if len(variants) > 1
        }

        # Add statistics to report
        report_lines.extend(
            [
                f"Total Rooms: {len(self.rooms)}",
                f"Total Connections: {len(self.connections)}",
                f"Empty Exit Nodes: {len(empty_exit_nodes)}",
                f"Isolated Nodes: {len(isolated_nodes)}",
                f"Unknown Destinations: {unknown_destinations}",
                f"Fragmented Base Names: {len(fragmented_base_names)}",
                "",
            ]
        )

        # Detail isolated nodes
        if isolated_nodes:
            report_lines.extend(["🗑️  ISOLATED NODES (candidates for pruning):"])
            for node in isolated_nodes:
                report_lines.append(f"   - {node}")
            report_lines.append("")

        # Detail fragmented base names
        if fragmented_base_names:
            report_lines.extend(
                ["🔄 FRAGMENTED BASE NAMES (candidates for consolidation):"]
            )
            for base_name, variants in fragmented_base_names.items():
                if len(variants) > 1:
                    report_lines.append(f"   {base_name}:")
                    for variant in variants:
                        report_lines.append(f"     - {variant}")
            report_lines.append("")

        # Detail empty exit nodes that are NOT isolated
        connected_empty_nodes = [
            name for name in empty_exit_nodes if name in nodes_with_incoming
        ]
        if connected_empty_nodes:
            report_lines.extend(["⚠️  EMPTY EXIT NODES (have incoming connections):"])
            for node in connected_empty_nodes:
                report_lines.append(f"   - {node}")
            report_lines.append("")

        return "\n".join(report_lines)

    def prune_invalid_exits(self, room_name: str, min_failure_count: int = 3) -> int:
        """
        Remove exits that have been tried multiple times and consistently failed.

        Args:
            room_name: The room to prune exits from
            min_failure_count: Minimum number of failures before pruning an exit

        Returns:
            Number of exits pruned
        """
        room_key = room_name
        if room_key not in self.rooms:
            return 0

        room = self.rooms[room_key]
        exits_to_remove = []
        pruned_count = 0

        # Check each exit in the room against failure counts
        for exit_name in list(
            room.exits
        ):  # Use list() to avoid modification during iteration
            failure_key = (room_key, exit_name)
            failure_count = self.exit_failure_counts.get(failure_key, 0)

            if failure_count >= min_failure_count:
                exits_to_remove.append(exit_name)

        # Remove the failed exits
        for exit_name in exits_to_remove:
            room.exits.discard(exit_name)

            # Track that this exit has been pruned to avoid re-adding it
            if room_key not in self.pruned_exits:
                self.pruned_exits[room_key] = set()
            self.pruned_exits[room_key].add(exit_name)

            if self.logger:
                self.logger.info(
                    f"Pruned invalid exit: {room_name} -> {exit_name}",
                    extra={
                        "event_type": "progress",
                        "stage": "exit_pruning",
                        "details": f"failed {self.exit_failure_counts.get((room_key, exit_name), 0)} times",
                    },
                )
            pruned_count += 1

        if pruned_count > 0:
            if self.logger:
                self.logger.info(
                    f"Exit pruning complete for {room_name}: {pruned_count} invalid exits removed",
                    extra={"event_type": "progress", "stage": "exit_pruning"},
                )

        return pruned_count

    def get_exit_failure_stats(self, room_name: str = None) -> Dict:
        """
        Get statistics about exit failures, either for a specific room or globally.

        Args:
            room_name: If provided, return stats only for this room. If None, return global stats.

        Returns:
            Dictionary containing failure statistics
        """
        if room_name:
            # Stats for specific room
            room_key = room_name
            room_failures = {
                k: v for k, v in self.exit_failure_counts.items() if k[0] == room_key
            }
            pruned_exits = self.pruned_exits.get(room_key, set())

            return {
                "room": room_name,
                "total_failed_attempts": sum(room_failures.values()),
                "unique_failed_exits": len(room_failures),
                "pruned_exits": list(pruned_exits),
                "failure_details": {f"{k[1]}": v for k, v in room_failures.items()},
                "highest_failure_count": max(room_failures.values())
                if room_failures
                else 0,
            }
        else:
            # Global stats
            total_failures = sum(self.exit_failure_counts.values())
            total_pruned = sum(len(exits) for exits in self.pruned_exits.values())

            return {
                "total_failed_attempts": total_failures,
                "unique_failed_exits": len(self.exit_failure_counts),
                "total_pruned_exits": total_pruned,
                "rooms_with_failures": len(
                    set(k[0] for k in self.exit_failure_counts.keys())
                ),
                "rooms_with_pruned_exits": len(self.pruned_exits),
                "highest_failure_count": max(self.exit_failure_counts.values())
                if self.exit_failure_counts
                else 0,
            }

    def render_exit_failure_report(self) -> str:
        """
        Generate a detailed report on exit failures and pruning.

        Returns:
            Human-readable report of exit failure status
        """
        if not self.exit_failure_counts and not self.pruned_exits:
            return (
                "🔍 EXIT FAILURE REPORT\n" + "=" * 30 + "\nNo exit failures recorded."
            )

        report_lines = ["🔍 EXIT FAILURE REPORT", "=" * 30]

        # Overall statistics
        total_failures = sum(self.exit_failure_counts.values())
        total_pruned = sum(len(exits) for exits in self.pruned_exits.values())

        report_lines.extend(
            [
                f"Total Failed Attempts: {total_failures}",
                f"Unique Failed Exits: {len(self.exit_failure_counts)}",
                f"Total Pruned Exits: {total_pruned}",
                f"Rooms with Failures: {len(set(k[0] for k in self.exit_failure_counts.keys()))}",
                f"Rooms with Pruned Exits: {len(self.pruned_exits)}",
                "",
            ]
        )

        # Active failures (not yet pruned)
        active_failures = []
        for (room, exit), count in self.exit_failure_counts.items():
            pruned_exits_for_room = self.pruned_exits.get(room, set())
            if exit not in pruned_exits_for_room:
                active_failures.append((room, exit, count))

        if active_failures:
            report_lines.extend(["⚠️  ACTIVE FAILURES (not yet pruned):"])
            # Sort by failure count (highest first)
            active_failures.sort(key=lambda x: x[2], reverse=True)
            for room, exit, count in active_failures[:10]:  # Show top 10
                report_lines.append(f"  {room} -> {exit} ({count} failures)")
            if len(active_failures) > 10:
                report_lines.append(f"  ... and {len(active_failures) - 10} more")
            report_lines.append("")

        # Pruned exits by room
        if self.pruned_exits:
            report_lines.extend(["🗑️  PRUNED EXITS BY ROOM:"])
            for room, exits in self.pruned_exits.items():
                if exits:
                    report_lines.append(f"  {room}: {', '.join(sorted(exits))}")
            report_lines.append("")

        # Rooms with highest failure counts
        room_failure_totals = {}
        for (room, exit), count in self.exit_failure_counts.items():
            room_failure_totals[room] = room_failure_totals.get(room, 0) + count

        if room_failure_totals:
            sorted_rooms = sorted(
                room_failure_totals.items(), key=lambda x: x[1], reverse=True
            )
            report_lines.extend(["📊 ROOMS WITH MOST FAILURES:"])
            for room, total_count in sorted_rooms[:5]:  # Show top 5
                room_pruned_count = len(self.pruned_exits.get(room, set()))
                report_lines.append(
                    f"  {room}: {total_count} total failures, {room_pruned_count} exits pruned"
                )
            report_lines.append("")

        return "\n".join(report_lines)


if __name__ == "__main__":
    # Example Usage
    g = MapGraph()
    g.add_room("West of House")
    g.update_room_exits(
        "West of House", ["N", "go East", "southward"]
    )  # Test normalization

    g.add_connection("West of House", "north", "North of House")  # Already normalized
    g.add_connection(
        "Kitchen", "go east", "Living Room"
    )  # Test normalization in add_connection (caller should do it)
    # For this test, we'll assume it's pre-normalized by caller logic

    print("--- Initial Map ---")
    # Manually call normalize for the add_connection example above for clarity in test
    # In real use, main.py's logic would handle this before calling add_connection
    g.add_connection(
        "Kitchen", normalize_direction("go east") or "go east".lower(), "Living Room"
    )

    print(g.get_room_info("West of House"))
    print(g.get_room_info("North of House"))
    print(g.get_room_info("Kitchen"))
    print(g.get_room_info("Living Room"))
    print(g.get_room_info("Attic"))  # Unknown room

    print("\n--- Agent Context Examples ---")
    # Agent moves from West of House to North of House via "N" (which becomes "north")
    action_taken_by_agent = "N"
    normalized_action_for_connection = (
        normalize_direction(action_taken_by_agent) or action_taken_by_agent.lower()
    )
    # Assume this connection was added: g.add_connection("West of House", normalized_action_for_connection, "North of House")
    # For the prompt, we also use the normalized version (or the raw if not normalizable)
    print(
        g.get_context_for_prompt(
            current_room_name="North of House",
            previous_room_name="West of House",
            action_taken_to_current=normalized_action_for_connection,
        )
    )

    # Agent is just in the Kitchen, previous room unknown
    print(g.get_context_for_prompt(current_room_name="Kitchen"))

    # Agent moves from Living Room to Kitchen via "W" (which becomes "west")
    action_taken_by_agent_rev = "W"
    normalized_action_rev = (
        normalize_direction(action_taken_by_agent_rev)
        or action_taken_by_agent_rev.lower()
    )
    # (testing reverse connection, assuming it was set up correctly via normalized 'east' from Kitchen)
    print(
        g.get_context_for_prompt(
            current_room_name="Kitchen",
            previous_room_name="Living Room",
            action_taken_to_current=normalized_action_rev,
        )
    )

    # Agent explores a new exit from West of House - e.g. "jump window" (not a direction)
    g.update_room_exits(
        "West of House", ["jump window"]
    )  # This will be added as "jump window"
    print(g.get_context_for_prompt(current_room_name="West of House"))

    # Test _get_opposite_direction with a non-normalized input (it normalizes internally)
    print(f"Opposite of 'N': {g._get_opposite_direction('N')}")
    print(
        f"Opposite of 'go south': {g._get_opposite_direction('go south')}"
    )  # Should be None as normalize_direction doesn't handle "go south"
    # and _get_opposite_direction itself doesn't call normalize_direction.
    # This highlights a small inconsistency if _get_opposite_direction
    # is called with unnormalized strings.
    # However, add_connection normalizes before calling it.

    # Let's test _get_opposite_direction with already normalized inputs as used in add_connection
    print(f"Opposite of 'north': {g._get_opposite_direction('north')}")  # south
    print(
        f"Opposite of 'jump window': {g._get_opposite_direction('jump window')}"
    )  # None

    # Test a "go up" command
    g.add_connection(
        "Living Room", normalize_direction("climb up") or "climb up".lower(), "Attic"
    )
    print(g.get_room_info("Living Room"))
    print(g.get_room_info("Attic"))
    print(
        g.get_context_for_prompt(
            "Attic",
            "Living Room",
            normalize_direction("climb up") or "climb up".lower(),
        )
    )

    print(g.render_ascii())

    print("\n--- Mermaid Diagram ---")
    print(g.render_mermaid())

    print("\n--- Exit Failure Report ---")
    print(g.render_exit_failure_report())
