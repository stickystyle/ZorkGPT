"""
Enhanced MapGraph with confidence tracking for single-episode map building.

This extends the basic MapGraph functionality with confidence scoring,
verification tracking, and quality metrics for long single episodes.
"""

from typing import Dict, Tuple, List
from map_graph import MapGraph, Room


class EnhancedMapGraph(MapGraph):
    """
    Enhanced MapGraph with confidence tracking for single-episode optimization.
    
    Features:
    - Connection confidence scoring
    - Verification tracking for repeated paths
    - Conflict detection and resolution
    - Quality metrics for map assessment
    """
    
    def __init__(self):
        super().__init__()
        # Track confidence for each connection: (from_room, exit) -> confidence_score
        self.connection_confidence: Dict[Tuple[str, str], float] = {}
        # Track how many times each connection has been verified
        self.connection_verifications: Dict[Tuple[str, str], int] = {}
        # Track conflicts for analysis
        self.connection_conflicts: List[Dict] = []
    
    def add_connection(self, from_room_name: str, exit_taken: str, to_room_name: str, confidence: float = 1.0):
        """Add a connection with confidence tracking."""
        # Ensure rooms exist and normalize names
        from_room_normalized = self._normalize_room_name(from_room_name)
        to_room_normalized = self._normalize_room_name(to_room_name)
        self.add_room(from_room_name)
        self.add_room(to_room_name)

        processed_exit_taken = exit_taken.lower()
        connection_key = (from_room_normalized, processed_exit_taken)
        
        # Check for existing connections and handle conflicts/verifications
        if (from_room_normalized in self.connections and 
            processed_exit_taken in self.connections[from_room_normalized]):
            existing_destination = self.connections[from_room_normalized][processed_exit_taken]
            
            if existing_destination == to_room_normalized:
                # Same connection verified again - increase confidence
                current_verifications = self.connection_verifications.get(connection_key, 0)
                self.connection_verifications[connection_key] = current_verifications + 1
                # Confidence increases with verifications but caps at 1.0
                self.connection_confidence[connection_key] = min(1.0, 
                    self.connection_confidence.get(connection_key, 0.5) + 0.1)
                print(f"✅ Map connection verified: {from_room_normalized} -> {processed_exit_taken} -> {to_room_normalized}")
            else:
                # Conflicting connection - this is important to track
                conflict = {
                    "from_room": from_room_normalized,
                    "exit": processed_exit_taken,
                    "existing_destination": existing_destination,
                    "new_destination": to_room_normalized,
                    "existing_confidence": self.connection_confidence.get(connection_key, 0.5),
                    "new_confidence": confidence
                }
                self.connection_conflicts.append(conflict)
                
                print(f"⚠️  Map conflict detected: {from_room_normalized} -> {processed_exit_taken}")
                print(f"   Existing: {existing_destination} (confidence: {conflict['existing_confidence']:.2f})")
                print(f"   New: {to_room_normalized} (confidence: {confidence:.2f})")
                
                # Use higher confidence connection
                if confidence > conflict['existing_confidence']:
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

        # Call parent method to actually add the connection
        super().add_connection(from_room_name, exit_taken, to_room_name)
        
        # Update confidence for reverse connection if it was added
        opposite_exit = self._get_opposite_direction(processed_exit_taken)
        if opposite_exit:
            reverse_connection_key = (to_room_normalized, opposite_exit)
            if reverse_connection_key not in self.connection_confidence:
                # Set confidence for reverse connection (slightly lower since it's inferred)
                self.connection_confidence[reverse_connection_key] = confidence * 0.9
                self.connection_verifications[reverse_connection_key] = 1

    def get_high_confidence_connections(self, min_confidence: float = 0.7) -> Dict[str, Dict[str, str]]:
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
                "conflicts_detected": 0
            }
        
        confidences = list(self.connection_confidence.values())
        average_confidence = sum(confidences) / len(confidences)
        high_confidence_count = sum(1 for c in confidences if c >= 0.7)
        high_confidence_ratio = high_confidence_count / len(confidences)
        verified_connections = sum(1 for v in self.connection_verifications.values() if v > 1)
        
        return {
            "average_confidence": average_confidence,
            "high_confidence_ratio": high_confidence_ratio,
            "total_connections": len(confidences),
            "verified_connections": verified_connections,
            "conflicts_detected": len(self.connection_conflicts)
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
            ""
        ]
        
        if self.connection_conflicts:
            report.append("⚠️  CONFLICTS DETECTED:")
            for i, conflict in enumerate(self.connection_conflicts[-5:], 1):  # Show last 5
                report.append(f"  {i}. {conflict['from_room']} -> {conflict['exit']}")
                report.append(f"     Old: {conflict['existing_destination']} ({conflict['existing_confidence']:.2f})")
                report.append(f"     New: {conflict['new_destination']} ({conflict['new_confidence']:.2f})")
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
                    report.append(f"  {room} -> {exit} -> {dest} ({conf:.2f}, {verifications}x verified)")
            report.append("")
        
        return "\n".join(report)

    def get_navigation_suggestions(self, current_room: str) -> List[Dict]:
        """Get navigation suggestions based on confidence scores."""
        current_room_normalized = self._normalize_room_name(current_room)
        suggestions = []
        
        if current_room_normalized in self.connections:
            for exit, destination in self.connections[current_room_normalized].items():
                confidence = self.get_connection_confidence(current_room, exit)
                verifications = self.connection_verifications.get((current_room_normalized, exit), 0)
                
                suggestions.append({
                    "exit": exit,
                    "destination": destination,
                    "confidence": confidence,
                    "verifications": verifications,
                    "recommendation": self._get_recommendation(confidence, verifications)
                })
        
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