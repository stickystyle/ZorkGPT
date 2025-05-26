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
        normalized_name = self._normalize_room_name(room_name)
        if normalized_name not in self.rooms:
            self.rooms[normalized_name] = Room(name=normalized_name)
        return self.rooms[normalized_name]

    def update_room_exits(self, room_name: str, new_exits: List[str]):
        normalized_name = self._normalize_room_name(room_name)
        if normalized_name not in self.rooms:
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
            self.rooms[normalized_name].add_exit(exit_name)

    def add_connection(
        self,
        from_room_name: str,
        exit_taken: str,
        to_room_name: str,
        confidence: float = 1.0,
    ):
        # Ensure rooms exist and normalize names
        from_room_normalized = self._normalize_room_name(from_room_name)
        to_room_normalized = self._normalize_room_name(to_room_name)
        self.add_room(from_room_name)
        self.add_room(to_room_name)

        # exit_taken should already be normalized (or the raw action) by the caller in main.py
        # Here, we just ensure it's lowercase if it was a raw action.
        # If it was a normalized direction, it's already lowercase.
        processed_exit_taken = exit_taken.lower()

        # Track confidence for this connection
        connection_key = (from_room_normalized, processed_exit_taken)

        # Check for existing connections and handle conflicts/verifications
        if (
            from_room_normalized in self.connections
            and processed_exit_taken in self.connections[from_room_normalized]
        ):
            existing_destination = self.connections[from_room_normalized][
                processed_exit_taken
            ]

            if existing_destination == to_room_normalized:
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
                    f"✅ Map connection verified: {from_room_normalized} -> {processed_exit_taken} -> {to_room_normalized}"
                )
            else:
                # Conflicting connection - this is important to track
                conflict = {
                    "from_room": from_room_normalized,
                    "exit": processed_exit_taken,
                    "existing_destination": existing_destination,
                    "new_destination": to_room_normalized,
                    "existing_confidence": self.connection_confidence.get(
                        connection_key, 0.5
                    ),
                    "new_confidence": confidence,
                }
                self.connection_conflicts.append(conflict)

                print(
                    f"⚠️  Map conflict detected: {from_room_normalized} -> {processed_exit_taken}"
                )
                print(
                    f"   Existing: {existing_destination} (confidence: {conflict['existing_confidence']:.2f})"
                )
                print(f"   New: {to_room_normalized} (confidence: {confidence:.2f})")

                # Use higher confidence connection
                if confidence > conflict["existing_confidence"]:
                    print(f"   → Using new connection (higher confidence)")
                    self.connection_confidence[connection_key] = confidence
                    self.connection_verifications[connection_key] = 1
                else:
                    print(f"   → Keeping existing connection (higher confidence)")
                    return  # Don't update the connection
        else:
            # New connection
            self.connection_confidence[connection_key] = confidence
            self.connection_verifications[connection_key] = 1

        # Add the forward connection
        if from_room_normalized not in self.connections:
            self.connections[from_room_normalized] = {}
        self.connections[from_room_normalized][processed_exit_taken] = (
            to_room_normalized
        )
        self.rooms[from_room_normalized].add_exit(
            processed_exit_taken
        )  # Ensure exit is recorded for the room

        # Add the reverse connection if an opposite direction exists
        opposite_exit = self._get_opposite_direction(processed_exit_taken)
        if opposite_exit:
            reverse_connection_key = (to_room_normalized, opposite_exit)

            if to_room_normalized not in self.connections:
                self.connections[to_room_normalized] = {}
            # Only add reverse connection if it doesn't overwrite an existing one from that direction
            # This handles cases where "north" from A leads to B, but "south" from B leads to C (unlikely but possible)
            if opposite_exit not in self.connections[to_room_normalized]:
                self.connections[to_room_normalized][opposite_exit] = (
                    from_room_normalized
                )
                # Set confidence for reverse connection (slightly lower since it's inferred)
                self.connection_confidence[reverse_connection_key] = confidence * 0.9
                self.connection_verifications[reverse_connection_key] = 1
            self.rooms[to_room_normalized].add_exit(
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

        current_room_normalized = self._normalize_room_name(current_room_name)
        room_known = current_room_normalized in self.rooms

        if room_known:
            room = self.rooms[current_room_normalized]
            context_parts.append(
                f"Current location: {room.name} (according to map)."
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
                f"Map: Location '{current_room_name}' is new or not yet mapped. No detailed map data available for it yet."
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
        from_room_normalized = self._normalize_room_name(from_room)
        connection_key = (from_room_normalized, exit_taken.lower())
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
        current_room_normalized = self._normalize_room_name(current_room)
        suggestions = []

        if current_room_normalized in self.connections:
            for exit, destination in self.connections[current_room_normalized].items():
                confidence = self.get_connection_confidence(current_room, exit)
                verifications = self.connection_verifications.get(
                    (current_room_normalized, exit), 0
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
