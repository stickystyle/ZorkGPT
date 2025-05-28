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
    def __init__(self, name: str):
        self.name: str = name
        self.exits: Set[str] = set()  # Known exits from this room

    def add_exit(self, exit_name: str):
        self.exits.add(exit_name)

    def __repr__(self) -> str:
        return f"Room(name='{self.name}', exits={self.exits})"


class MapGraph:
    def __init__(self):
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

    def add_room(self, room_name: str) -> Room:
        # Use the room name as-is (it should already be a unique ID if needed)
        room_key = room_name
        if room_key not in self.rooms:
            self.rooms[room_key] = Room(name=room_key)
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
                # Use canonical direction (e.g., "north", "up")
                normalized_new_exits.add(norm_exit)
            else:
                # For non-directional exits, preserve original case but strip whitespace
                # This handles things like "window", "trapdoor", "ladder", etc.
                clean_exit = exit_name.strip()
                if clean_exit:
                    normalized_new_exits.add(clean_exit.lower())

        for exit_name in normalized_new_exits:
            self.rooms[room_key].add_exit(exit_name)

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
        processed_exit_taken = normalized_action if normalized_action else exit_taken.lower().strip()
        
        # Track confidence for this connection
        connection_key = (from_room_key, processed_exit_taken)

        # Check for existing connections and handle conflicts/verifications
        if (
            from_room_key in self.connections
            and processed_exit_taken in self.connections[from_room_key]
        ):
            existing_destination = self.connections[from_room_key][
                processed_exit_taken
            ]

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
                print(
                    f"✅ Map connection verified: {from_room_key} -> {processed_exit_taken} -> {to_room_key} (verifications: {self.connection_verifications[connection_key]})"
                )
            else:
                # Conflicting connection - this is important to track
                existing_confidence = self.connection_confidence.get(connection_key, 0.5)
                existing_verifications = self.connection_verifications.get(connection_key, 0)
                
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

                print(
                    f"⚠️  Map conflict detected: {from_room_key} -> {processed_exit_taken}"
                )
                print(
                    f"   Existing: {existing_destination} (confidence: {existing_confidence:.2f}, verifications: {existing_verifications})"
                )
                print(f"   New: {to_room_key} (confidence: {confidence:.2f}, verifications: 1)")

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
                        print(f"   🚨 CRITICAL CONFLICT: Both connections have equal confidence and low verifications!")
                        print(f"   🚨 This suggests inconsistent movement behavior or extraction errors.")
                        print(f"   🚨 Keeping existing connection but flagging for investigation.")
                        should_update = False
                        reason = "keeping existing due to critical conflict (needs investigation)"
                else:
                    should_update = False
                    reason = "existing has higher confidence"

                if should_update:
                    print(f"   → Using new connection ({reason})")
                    self.connection_confidence[connection_key] = confidence
                    self.connection_verifications[connection_key] = 1
                else:
                    print(f"   → Keeping existing connection ({reason})")
                    return  # Don't update the connection
        else:
            # New connection
            self.connection_confidence[connection_key] = confidence
            self.connection_verifications[connection_key] = 1

        # Add the forward connection
        if from_room_key not in self.connections:
            self.connections[from_room_key] = {}
        self.connections[from_room_key][processed_exit_taken] = (
            to_room_key
        )
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
                self.connections[to_room_key][opposite_exit] = (
                    from_room_key
                )
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

            # Keep the exit processing logic for internal map tracking, but don't include
            # the "mapped exits are" text in the prompt to avoid confusion with consensus map
            if room.exits:
                exit_details = []
                # Sort exits for consistent output order
                for exit_dir in sorted(list(room.exits)):
                    if (
                        current_room_normalized in self.connections
                        and exit_dir in self.connections[current_room_normalized]
                    ):
                        leads_to_room = self.connections[current_room_normalized][
                            exit_dir
                        ]
                        exit_details.append(f"{exit_dir} (leads to {leads_to_room})")
                    else:
                        exit_details.append(f"{exit_dir} (destination unknown)")

                # Note: We build exit_details for internal consistency but don't add them to context_parts
                # This keeps the mapping logic intact while cleaning up the agent prompt

            # No exit information added to context_parts - the consensus map will provide navigation info
        else:  # current_room_name is not in self.rooms
            context_parts.append(
                f"Map: Location '{display_name}' is new or not yet mapped. No detailed map data available for it yet."
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
            return "graph TD\n    A[No rooms mapped yet]"

        lines = ["graph TD"]

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
        processed_exit_taken = normalized_action if normalized_action else exit_taken.lower().strip()
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

    def _create_unique_location_id(self, location_name: str, description: str = "", objects: List[str] = None, exits: List[str] = None) -> str:
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
            # Normalize exits to canonical directions
            normalized_exits = set()
            for exit in exits:
                if not exit or not exit.strip():
                    continue
                norm_exit = normalize_direction(exit)
                if norm_exit:
                    normalized_exits.add(norm_exit)
                else:
                    # Keep non-directional exits (like "window", "trapdoor") as-is
                    normalized_exits.add(exit.lower().strip())
            
            if normalized_exits:
                # Create distinctive patterns based on exit combinations
                exit_count = len(normalized_exits)
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
                    if "north" in normalized_exits and "east" in normalized_exits and "south" in normalized_exits:
                        return f"{base_name} (T-junction east)"
                    elif "north" in normalized_exits and "west" in normalized_exits and "south" in normalized_exits:
                        return f"{base_name} (T-junction west)"
                    elif "east" in normalized_exits and "west" in normalized_exits and "north" in normalized_exits:
                        return f"{base_name} (T-junction north)"
                    elif "east" in normalized_exits and "west" in normalized_exits and "south" in normalized_exits:
                        return f"{base_name} (T-junction south)"
                    else:
                        # Other three-exit combinations
                        return f"{base_name} (3-way: {'-'.join(sorted_exits[:3])})"
                
                # Four or more exits - use count-based identifier
                elif exit_count >= 4:
                    if exit_count == 4 and {"north", "south", "east", "west"}.issubset(normalized_exits):
                        return f"{base_name} (4-way intersection)"
                    else:
                        return f"{base_name} ({exit_count}-way junction)"
        
        # SECONDARY APPROACH: Only use descriptions for truly permanent, structural features
        # Avoid volatile content like objects, lighting, or temporary states
        if description:
            desc_lower = description.lower()
            
            # Only use highly distinctive, permanent structural features
            if "well house" in desc_lower:
                return f"{base_name} (well house)"
            elif "white house" in desc_lower and "front" in desc_lower:
                return f"{base_name} (front of house)"
            elif "white house" in desc_lower and ("back" in desc_lower or "behind" in desc_lower):
                return f"{base_name} (back of house)"
            elif "attic" in desc_lower:
                return f"{base_name} (attic)"
            elif "basement" in desc_lower or "cellar" in desc_lower:
                return f"{base_name} (basement)"
            elif "kitchen" in desc_lower:
                return f"{base_name} (kitchen)"
            elif "living room" in desc_lower:
                return f"{base_name} (living room)"
        
        # AVOID: Volatile features that change frequently
        # - Objects that can be picked up/dropped
        # - Lighting conditions ("dimly lit", "dark")
        # - Temporary states ("open door", "closed window")
        # - Minor object detection variations
        
        # Default: return the base name without modification
        # This ensures the same room gets the same ID unless there are
        # truly distinctive permanent features
        return base_name

    def needs_consolidation(self) -> bool:
        """Check if consolidation is needed based on new room additions."""
        return self.has_new_rooms_since_consolidation

    def consolidate_similar_locations(self) -> int:
        """
        Automatically detect and consolidate locations that are likely the same place
        but have different unique IDs due to extractor inconsistencies.
        
        Returns:
            Number of consolidations performed
        """
        from collections import defaultdict
        
        # Reset the flag since we're about to consolidate
        self.has_new_rooms_since_consolidation = False
        
        # Group locations by their base name (removing parenthetical suffixes)
        location_groups = defaultdict(list)
        
        for location_name in self.rooms.keys():
            # Extract base name by removing parenthetical suffixes like "(north only)"
            base_name = location_name.split('(')[0].strip()
            # Normalize case and spacing
            base_name = ' '.join(word.capitalize() for word in base_name.split())
            location_groups[base_name].append(location_name)
        
        consolidations_performed = 0
        
        # Find groups with multiple variants (fragmentation)
        for base_name, variants in location_groups.items():
            if len(variants) <= 1:
                continue  # No fragmentation for this location
                
            print(f"🔄 Consolidating fragmented location: {base_name}")
            print(f"   Variants found: {variants}")
            
            # Choose the consolidation target (prefer the base name if it exists)
            if base_name in variants:
                target_location = base_name
            else:
                # Choose the shortest variant as the target
                target_location = min(variants, key=len)
            
            # Collect all exits from variants
            all_exits = set()
            for variant in variants:
                if variant in self.rooms:
                    all_exits.update(self.rooms[variant].exits)
            
            print(f"   Target location: {target_location}")
            print(f"   Combined exits: {sorted(list(all_exits))}")
            
            # Merge all connections from variants into the target
            for variant in variants:
                if variant == target_location:
                    continue  # Skip the target itself
                    
                if variant in self.connections:
                    # Move outgoing connections from variant to target
                    for exit_action, destination in self.connections[variant].items():
                        print(f"   Moving connection: {variant} --[{exit_action}]--> {destination}")
                        self.add_connection(target_location, exit_action, destination)
                    
                    # Remove the old connections
                    del self.connections[variant]
                
                # Update incoming connections that point to this variant
                for from_location, exits in self.connections.items():
                    for exit_action, destination in list(exits.items()):
                        if destination == variant:
                            print(f"   Redirecting connection: {from_location} --[{exit_action}]--> {variant} => {target_location}")
                            exits[exit_action] = target_location
                
                # Remove the variant room if it's not the target
                if variant in self.rooms:
                    del self.rooms[variant]
                    
                consolidations_performed += 1
            
            # Update the target location with all collected exits
            if target_location in self.rooms:
                self.rooms[target_location].exits = all_exits
            else:
                # Create the target location if it doesn't exist
                self.add_room(target_location)
                self.rooms[target_location].exits = all_exits
        
        if consolidations_performed > 0:
            print(f"✅ Consolidation complete: {consolidations_performed} locations merged")
        
        return consolidations_performed


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
