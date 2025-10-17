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
    def __init__(self, room_id: int, name: str):
        self.id: int = room_id  # PRIMARY KEY - Z-machine object ID
        self.name: str = name    # Display name only
        self.exits: Set[str] = set()  # Known exits from this room

    def add_exit(self, exit_name: str):
        self.exits.add(exit_name)

    def __repr__(self) -> str:
        return f"Room(id={self.id}, name='{self.name}', exits={self.exits})"


class MapGraph:
    def __init__(self, logger=None):
        self.logger = logger
        # Migrated to integer IDs (Phase 3.3)
        self.rooms: Dict[int, Room] = {}  # Integer keys = location IDs
        self.room_names: Dict[int, str] = {}  # ID -> name mapping for display
        # connections[room_id_1][exit_taken] = room_id_2
        self.connections: Dict[int, Dict[str, int]] = {}
        # Track confidence for each connection: (from_room_id, exit) -> confidence_score
        self.connection_confidence: Dict[Tuple[int, str], float] = {}
        # Track how many times each connection has been verified
        self.connection_verifications: Dict[Tuple[int, str], int] = {}
        # Track conflicts for analysis
        self.connection_conflicts: List[Dict] = []
        # Track failed exit attempts: (room_id, exit) -> failure_count
        self.exit_failure_counts: Dict[Tuple[int, str], int] = {}
        # Track exits that have been permanently pruned to avoid re-adding them
        self.pruned_exits: Dict[int, Set[str]] = {}

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




    def add_room(self, room_id: int, room_name: str) -> Room:
        """
        Add a room to the map using Jericho's integer location ID.

        Args:
            room_id: Z-machine location ID (from location.num)
            room_name: Display name of the location

        Returns:
            Room object
        """
        if room_id not in self.rooms:
            self.rooms[room_id] = Room(room_id=room_id, name=room_name)
            self.room_names[room_id] = room_name
        return self.rooms[room_id]

    def update_room_exits(self, room_id: int, new_exits: List[str]):
        """Update exits for a room using integer ID."""
        if room_id not in self.rooms:
            return  # Cannot add exits for non-existent room

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
        pruned_exits_for_room = self.pruned_exits.get(room_id, set())

        for exit_name in normalized_new_exits:
            # Don't re-add exits that have been pruned as invalid
            if exit_name not in pruned_exits_for_room:
                self.rooms[room_id].add_exit(exit_name)
            else:
                if self.logger:
                    room_name = self.room_names.get(room_id, f"Room#{room_id}")
                    self.logger.debug(
                        f"Skipping re-addition of pruned exit: {room_name} -> {exit_name}",
                        extra={
                            "event_type": "progress",
                            "stage": "map_building",
                            "details": f"Exit {exit_name} was previously pruned as invalid",
                        },
                    )

    def track_exit_failure(self, room_id: int, exit_name: str) -> int:
        """
        Track a failed exit attempt and return the current failure count.

        Args:
            room_id: The room ID where the exit was attempted
            exit_name: The exit that failed (will be normalized)

        Returns:
            The current failure count for this exit
        """
        normalized_action = normalize_direction(exit_name)
        processed_exit = (
            normalized_action if normalized_action else exit_name.lower().strip()
        )

        failure_key = (room_id, processed_exit)
        self.exit_failure_counts[failure_key] = (
            self.exit_failure_counts.get(failure_key, 0) + 1
        )

        failure_count = self.exit_failure_counts[failure_key]
        if self.logger:
            room_name = self.room_names.get(room_id, f"Room#{room_id}")
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
        from_room_id: int,
        exit_taken: str,
        to_room_id: int,
        confidence: float = 1.0,
    ):
        """Add connection between rooms using integer IDs."""
        # Ensure both rooms exist
        if from_room_id not in self.rooms or to_room_id not in self.rooms:
            if self.logger:
                self.logger.warning(
                    f"Cannot add connection: room {from_room_id} or {to_room_id} doesn't exist"
                )
            return

        # Use basic normalization for standard directions only
        # Let LLM layers handle semantic equivalence
        normalized_action = normalize_direction(exit_taken)
        processed_exit_taken = (
            normalized_action if normalized_action else exit_taken.lower().strip()
        )

        # Track confidence for this connection
        connection_key = (from_room_id, processed_exit_taken)

        # Check for existing connections and handle conflicts/verifications
        if (
            from_room_id in self.connections
            and processed_exit_taken in self.connections[from_room_id]
        ):
            existing_destination = self.connections[from_room_id][processed_exit_taken]

            if existing_destination == to_room_id:
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
                    from_name = self.room_names.get(from_room_id, f"Room#{from_room_id}")
                    to_name = self.room_names.get(to_room_id, f"Room#{to_room_id}")
                    self.logger.debug(
                        f"Map connection verified: {from_name} -> {processed_exit_taken} -> {to_name}",
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

                from_name = self.room_names.get(from_room_id, f"Room#{from_room_id}")
                existing_dest_name = self.room_names.get(existing_destination, f"Room#{existing_destination}")
                new_dest_name = self.room_names.get(to_room_id, f"Room#{to_room_id}")

                conflict = {
                    "from_room": from_name,
                    "from_room_id": from_room_id,
                    "exit": processed_exit_taken,
                    "existing_destination": existing_dest_name,
                    "existing_destination_id": existing_destination,
                    "new_destination": new_dest_name,
                    "new_destination_id": to_room_id,
                    "existing_confidence": existing_confidence,
                    "new_confidence": confidence,
                    "existing_verifications": existing_verifications,
                    "new_verifications": 1,
                }
                self.connection_conflicts.append(conflict)

                if self.logger:
                    self.logger.warning(
                        f"Map conflict detected: {from_name} -> {processed_exit_taken}",
                        extra={
                            "event_type": "progress",
                            "stage": "map_building",
                            "details": f"Existing: {existing_dest_name} ({existing_confidence:.2f}, {existing_verifications}x) vs New: {new_dest_name} ({confidence:.2f}, 1x)",
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
                        from_name = self.room_names.get(from_room_id, f"Room#{from_room_id}")
                        to_name = self.room_names.get(to_room_id, f"Room#{to_room_id}")
                        self.logger.info(
                            f"Using new connection ({reason})",
                            extra={
                                "event_type": "progress",
                                "stage": "map_building",
                                "details": f"{from_name} -> {processed_exit_taken} -> {to_name}",
                            },
                        )
                    self.connection_confidence[connection_key] = confidence
                    self.connection_verifications[connection_key] = 1
                else:
                    if self.logger:
                        from_name = self.room_names.get(from_room_id, f"Room#{from_room_id}")
                        existing_name = self.room_names.get(existing_destination, f"Room#{existing_destination}")
                        self.logger.debug(
                            f"Keeping existing connection ({reason})",
                            extra={
                                "event_type": "progress",
                                "stage": "map_building",
                                "details": f"{from_name} -> {processed_exit_taken} -> {existing_name}",
                            },
                        )
                    return  # Don't update the connection
        else:
            # New connection
            self.connection_confidence[connection_key] = confidence
            self.connection_verifications[connection_key] = 1

        # Add the forward connection
        if from_room_id not in self.connections:
            self.connections[from_room_id] = {}
        self.connections[from_room_id][processed_exit_taken] = to_room_id
        self.rooms[from_room_id].add_exit(
            processed_exit_taken
        )  # Ensure exit is recorded for the room

        # Add the reverse connection if an opposite direction exists
        opposite_exit = self._get_opposite_direction(processed_exit_taken)
        if opposite_exit:
            reverse_connection_key = (to_room_id, opposite_exit)

            if to_room_id not in self.connections:
                self.connections[to_room_id] = {}
            # Only add reverse connection if it doesn't overwrite an existing one from that direction
            # This handles cases where "north" from A leads to B, but "south" from B leads to C (unlikely but possible)
            if opposite_exit not in self.connections[to_room_id]:
                self.connections[to_room_id][opposite_exit] = from_room_id
                # Set confidence for reverse connection (slightly lower since it's inferred)
                self.connection_confidence[reverse_connection_key] = confidence * 0.9
                self.connection_verifications[reverse_connection_key] = 1
            self.rooms[to_room_id].add_exit(
                opposite_exit
            )  # Ensure reverse exit is recorded

    def get_room_info(self, room_id: int) -> str:
        """Get room information using integer ID."""
        if room_id not in self.rooms:
            return f"Room ID {room_id} is unknown."

        room = self.rooms[room_id]
        info_parts = [f"Current room: {room.name}."]

        if room.exits:
            exit_descriptions = []
            for exit_name in sorted(list(room.exits)):  # Sort for consistent output
                description = exit_name
                if (
                    room_id in self.connections
                    and exit_name in self.connections[room_id]
                ):
                    connected_room_id = self.connections[room_id][exit_name]
                    connected_room_name = self.room_names.get(connected_room_id, f"Room#{connected_room_id}")
                    description += f" (leads to {connected_room_name})"
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
        current_room_id: int,
        current_room_name: str,  # For display only
        previous_room_id: int = None,
        previous_room_name: str = None,  # For display only
        action_taken_to_current: str = None,
    ) -> str:
        """
        Get context for prompt using integer IDs for lookups and names for display.

        Args:
            current_room_id: Integer ID of current room
            current_room_name: Display name of current room
            previous_room_id: Integer ID of previous room (optional)
            previous_room_name: Display name of previous room (optional)
            action_taken_to_current: Action string used to reach current room (optional)
        """
        context_parts = []

        if current_room_id is None:  # Handle missing current_room_id
            context_parts.append(
                "Map: Current location ID is missing. Cannot provide map context."
            )
            return "--- Map Information ---\n" + "\n".join(context_parts)

        room_known = current_room_id in self.rooms

        if room_known:
            room = self.rooms[current_room_id]
            context_parts.append(
                f"Current location: {current_room_name} (according to map)."
            )

            if previous_room_name and action_taken_to_current:
                action_desc = action_taken_to_current.lower()
                context_parts.append(
                    f"You arrived from '{previous_room_name}' by going {action_desc}."
                )

            # Check for mapped exits and provide guidance about unmapped exits
            has_mapped_exits = (
                current_room_id in self.connections
                and self.connections[current_room_id]
            )

            has_detected_exits = room.exits and len(room.exits) > 0

            if has_mapped_exits:
                # Show confirmed working exits
                confirmed_exits = list(self.connections[current_room_id].keys())
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
            elif len(self.connections[current_room_id]) < 2:
                context_parts.append(
                    "🔍 TIP: Many Zork locations have additional unmapped exits. Try cardinal directions."
                )

        else:  # current_room_id is not in self.rooms
            context_parts.append(
                f"Map: Location '{current_room_name}' is new or not yet mapped. No detailed map data available for it yet."
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
        """Render ASCII map using integer IDs."""
        if not self.rooms:
            return "-- Map is Empty --"

        output_lines = ["\n--- ASCII Map State ---"]
        output_lines.append("=======================")

        # Sort room IDs for consistent output order
        sorted_room_ids = sorted(self.rooms.keys())

        for room_id in sorted_room_ids:
            room_obj = self.rooms.get(room_id)
            room_name = self.room_names.get(room_id, f"Room#{room_id}")
            output_lines.append(f"\n[ {room_name} ]")

            connections_exist = (
                room_id in self.connections and self.connections[room_id]
            )

            if connections_exist:
                # Sort exit actions for consistent output order
                sorted_exit_actions = sorted(self.connections[room_id].keys())
                for exit_action in sorted_exit_actions:
                    destination_id = self.connections[room_id][exit_action]
                    destination_name = self.room_names.get(destination_id, f"Room#{destination_id}")
                    output_lines.append(
                        f"  --({exit_action})--> [ {destination_name} ]"
                    )

            # Also list exits known to the Room object but not yet in connections (unmapped)
            if room_obj and room_obj.exits:
                unmapped_exits = []
                for room_exit in sorted(list(room_obj.exits)):
                    # Check if this room_exit is already covered by a connection display
                    is_mapped = (
                        connections_exist and room_exit in self.connections[room_id]
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
        Render the map as a Mermaid diagram using integer IDs.

        Returns:
            Mermaid diagram syntax as a string
        """
        if not self.rooms:
            return "graph LR\n    A[No rooms mapped yet]"

        lines = ["graph LR"]

        # Create node definitions with sanitized IDs
        room_id_to_node = {}
        node_counter = 1

        # First pass: create node IDs and definitions
        sorted_room_ids = sorted(self.rooms.keys())
        for room_id in sorted_room_ids:
            node_id = f"R{node_counter}"
            room_id_to_node[room_id] = node_id
            room_name = self.room_names.get(room_id, f"Room#{room_id}")
            # Sanitize room name for Mermaid (escape special characters)
            sanitized_name = (
                room_name.replace('"', '\\"').replace("[", "\\[").replace("]", "\\]")
            )
            lines.append(f'    {node_id}["{sanitized_name}"]')
            node_counter += 1

        # Second pass: create connections
        connection_lines = []
        for room_id in sorted_room_ids:
            if room_id in self.connections:
                from_id = room_id_to_node[room_id]
                # Sort exit actions for consistent output
                sorted_exits = sorted(self.connections[room_id].keys())
                for exit_action in sorted_exits:
                    destination_id = self.connections[room_id][exit_action]
                    if destination_id in room_id_to_node:
                        to_id = room_id_to_node[destination_id]
                        # Sanitize exit action for Mermaid
                        sanitized_action = exit_action.replace('"', '\\"')
                        connection_lines.append(
                            f'    {from_id} -->|"{sanitized_action}"| {to_id}'
                        )
                    else:
                        # Create a temporary node for unknown destinations
                        unknown_id = f"U{node_counter}"
                        dest_name = self.room_names.get(destination_id, f"Room#{destination_id}")
                        sanitized_dest = (
                            dest_name.replace('"', '\\"')
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
        for room_id in sorted_room_ids:
            room_obj = self.rooms.get(room_id)
            if room_obj and room_obj.exits:
                from_id = room_id_to_node[room_id]
                for room_exit in sorted(list(room_obj.exits)):
                    # Check if this exit is already mapped
                    is_mapped = (
                        room_id in self.connections
                        and room_exit in self.connections[room_id]
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
    ) -> Dict[int, Dict[str, int]]:
        """Get only connections that meet the minimum confidence threshold."""
        high_confidence_connections = {}

        for room_id, exits in self.connections.items():
            high_confidence_exits = {}
            for exit_name, destination_id in exits.items():
                connection_key = (room_id, exit_name)
                confidence = self.connection_confidence.get(connection_key, 0.5)

                if confidence >= min_confidence:
                    high_confidence_exits[exit_name] = destination_id

            if high_confidence_exits:
                high_confidence_connections[room_id] = high_confidence_exits

        return high_confidence_connections

    def get_connection_confidence(self, from_room_id: int, exit_taken: str) -> float:
        """Get the confidence score for a specific connection."""
        # Use same normalization logic as add_connection
        normalized_action = normalize_direction(exit_taken)
        processed_exit_taken = (
            normalized_action if normalized_action else exit_taken.lower().strip()
        )
        connection_key = (from_room_id, processed_exit_taken)
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
            for room_id, exits in list(high_conf_connections.items())[:10]:  # Show top 10
                room_name = self.room_names.get(room_id, f"Room#{room_id}")
                for exit, dest_id in exits.items():
                    dest_name = self.room_names.get(dest_id, f"Room#{dest_id}")
                    conf = self.get_connection_confidence(room_id, exit)
                    verifications = self.connection_verifications.get((room_id, exit), 0)
                    report.append(
                        f"  {room_name} -> {exit} -> {dest_name} ({conf:.2f}, {verifications}x verified)"
                    )
            report.append("")

        return "\n".join(report)

    def get_navigation_suggestions(self, current_room_id: int) -> List[Dict]:
        """Get navigation suggestions based on confidence scores."""
        suggestions = []

        if current_room_id in self.connections:
            for exit, destination_id in self.connections[current_room_id].items():
                confidence = self.get_connection_confidence(current_room_id, exit)
                verifications = self.connection_verifications.get(
                    (current_room_id, exit), 0
                )
                destination_name = self.room_names.get(destination_id, f"Room#{destination_id}")

                suggestions.append(
                    {
                        "exit": exit,
                        "destination_id": destination_id,
                        "destination_name": destination_name,
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










    def prune_invalid_exits(self, room_id: int, min_failure_count: int = 3) -> int:
        """
        Remove exits that have been tried multiple times and consistently failed.

        Args:
            room_id: The room ID to prune exits from
            min_failure_count: Minimum number of failures before pruning an exit

        Returns:
            Number of exits pruned
        """
        if room_id not in self.rooms:
            return 0

        room = self.rooms[room_id]
        exits_to_remove = []
        pruned_count = 0

        # Check each exit in the room against failure counts
        for exit_name in list(
            room.exits
        ):  # Use list() to avoid modification during iteration
            failure_key = (room_id, exit_name)
            failure_count = self.exit_failure_counts.get(failure_key, 0)

            if failure_count >= min_failure_count:
                exits_to_remove.append(exit_name)

        # Remove the failed exits
        for exit_name in exits_to_remove:
            room.exits.discard(exit_name)

            # Track that this exit has been pruned to avoid re-adding it
            if room_id not in self.pruned_exits:
                self.pruned_exits[room_id] = set()
            self.pruned_exits[room_id].add(exit_name)

            if self.logger:
                room_name = self.room_names.get(room_id, f"Room#{room_id}")
                self.logger.info(
                    f"Pruned invalid exit: {room_name} -> {exit_name}",
                    extra={
                        "event_type": "progress",
                        "stage": "exit_pruning",
                        "details": f"failed {self.exit_failure_counts.get((room_id, exit_name), 0)} times",
                    },
                )
            pruned_count += 1

        if pruned_count > 0:
            if self.logger:
                room_name = self.room_names.get(room_id, f"Room#{room_id}")
                self.logger.info(
                    f"Exit pruning complete for {room_name}: {pruned_count} invalid exits removed",
                    extra={"event_type": "progress", "stage": "exit_pruning"},
                )

        return pruned_count

    def get_exit_failure_stats(self, room_id: int = None) -> Dict:
        """
        Get statistics about exit failures, either for a specific room or globally.

        Args:
            room_id: If provided, return stats only for this room. If None, return global stats.

        Returns:
            Dictionary containing failure statistics
        """
        if room_id is not None:
            # Stats for specific room
            room_failures = {
                k: v for k, v in self.exit_failure_counts.items() if k[0] == room_id
            }
            pruned_exits = self.pruned_exits.get(room_id, set())
            room_name = self.room_names.get(room_id, f"Room#{room_id}")

            return {
                "room_id": room_id,
                "room_name": room_name,
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
