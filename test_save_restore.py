#!/usr/bin/env python3
"""
Test script for game server save/restore functionality.

This script tests the core save/restore mechanism by:
1. Creating a session and executing a sequence of commands
2. Waiting for auto-save to trigger
3. Disconnecting and reconnecting to the same session
4. Verifying the game state was properly restored
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any


class GameServerTester:
    """Test harness for the game server save/restore functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session_id = None
        
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new game session."""
        print(f"📝 Creating session: {session_id}")
        
        response = requests.post(f"{self.base_url}/sessions/{session_id}")
        response.raise_for_status()
        
        data = response.json()
        self.session_id = session_id
        
        print(f"✅ Session created successfully")
        print(f"📍 Starting location from intro: {self._extract_location_from_intro(data['intro_text'])}")
        return data
        
    def send_command(self, command: str) -> Dict[str, Any]:
        """Send a command to the current session."""
        if not self.session_id:
            raise RuntimeError("No active session")
            
        print(f"🎮 Sending command: '{command}'")
        
        response = requests.post(
            f"{self.base_url}/sessions/{self.session_id}/command",
            json={"command": command}
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract room info from response
        parsed = data.get("parsed", {})
        room_name = parsed.get("room_name")
        game_text = parsed.get("game_text", data.get("raw_response", ""))
        
        if room_name:
            print(f"📍 Room: {room_name}")
        else:
            # Try to extract room from game text
            room_from_text = self._extract_location_from_text(game_text)
            if room_from_text:
                print(f"📍 Room (from text): {room_from_text}")
        
        print(f"🔄 Turn: {data.get('turn_number')}")
        print(f"📝 Response: {game_text[:100]}...")
        
        return data
        
    def get_inventory(self) -> Dict[str, Any]:
        """Get current inventory."""
        return self.send_command("inventory")
        
    def get_session_state(self) -> Dict[str, Any]:
        """Get current session state."""
        if not self.session_id:
            raise RuntimeError("No active session")
            
        response = requests.get(f"{self.base_url}/sessions/{self.session_id}/state")
        response.raise_for_status()
        return response.json()
        
    def get_session_history(self) -> Dict[str, Any]:
        """Get full session history."""
        if not self.session_id:
            raise RuntimeError("No active session")
            
        response = requests.get(f"{self.base_url}/sessions/{self.session_id}/history")
        response.raise_for_status()
        return response.json()
        
    def close_session(self):
        """Close the current session."""
        if not self.session_id:
            return
            
        print(f"🔚 Closing session: {self.session_id}")
        
        response = requests.delete(f"{self.base_url}/sessions/{self.session_id}")
        response.raise_for_status()
        
        self.session_id = None
        print("✅ Session closed")
        
    def reconnect_session(self, session_id: str) -> Dict[str, Any]:
        """Reconnect to an existing session."""
        print(f"🔄 Reconnecting to session: {session_id}")
        
        response = requests.post(f"{self.base_url}/sessions/{session_id}")
        response.raise_for_status()
        
        data = response.json()
        self.session_id = session_id
        
        print(f"✅ Reconnected successfully")
        return data
        
    def _extract_location_from_intro(self, intro_text: str) -> str:
        """Extract location from game intro text."""
        lines = intro_text.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('ZORK') and not line.startswith('Copyright') and not line.startswith('Revision') and not line.startswith('Loading'):
                if i + 1 < len(lines) and lines[i + 1].strip():
                    return line.strip()
        return "Unknown"
        
    def _extract_location_from_text(self, game_text: str) -> str:
        """Extract location from game response text."""
        lines = game_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('>') and not line.startswith('You') and not line.startswith('There'):
                # This might be a room name
                if len(line) < 50 and not line.endswith('.'):
                    return line
        return None
        
    def _extract_inventory_from_text(self, inv_text: str) -> list:
        """Extract inventory items from inventory command response."""
        if "empty-handed" in inv_text.lower():
            return []
            
        items = []
        lines = inv_text.split('\n')
        collecting = False
        
        for line in lines:
            line = line.strip()
            if line == "You are carrying:":
                collecting = True
                continue
                
            if collecting and line and not line.startswith('>'):
                if line.endswith('.'):
                    line = line[:-1]
                if line:
                    items.append(line)
                    
        return items


def run_save_restore_test():
    """Main test function."""
    print("🧪 Starting Save/Restore Test")
    print("=" * 50)
    
    # Generate unique session ID
    session_id = f"test-save-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    tester = GameServerTester()
    
    try:
        # Phase 1: Create session and execute commands
        print("\n📋 PHASE 1: Initial Game Session")
        print("-" * 30)
        
        tester.create_session(session_id)
        
        # Execute the test sequence
        commands = ["south", "east", "open window", "enter window", "take sack"]
        
        for command in commands:
            result = tester.send_command(command)
            time.sleep(0.5)  # Small delay between commands
            
        # Get final inventory
        print("\n📦 Checking inventory...")
        inv_result = tester.get_inventory()
        inventory_items = tester._extract_inventory_from_text(inv_result.get("raw_response", ""))
        print(f"📦 Inventory: {inventory_items}")
        
        # Get current location with 'look' command
        print("\n📍 Checking current location...")
        look_result = tester.send_command("look")
        
        # Wait for auto-save to trigger (should happen after turn 5+ or score change)
        print("\n⏳ Waiting for auto-save to trigger...")
        state = tester.get_session_state()
        print(f"💾 Last save at turn: {state.get('last_save_turn', 0)}")
        print(f"🎯 Current turn: {state.get('turn_number', 0)}")
        
        # If no auto-save yet, wait a moment for it to process
        if state.get('last_save_turn', 0) == 0:
            print("⏳ Waiting for auto-save...")
            time.sleep(2)
            
        # Get session history before disconnecting
        history = tester.get_session_history()
        total_turns = len(history.get("turns", []))
        print(f"📊 Total turns executed: {total_turns}")
        
        print(f"\n✅ Phase 1 complete. Session state saved.")
        
        # Phase 2: Disconnect and reconnect
        print("\n📋 PHASE 2: Disconnect and Reconnect")
        print("-" * 30)
        
        tester.close_session()
        
        print("⏳ Waiting 2 seconds before reconnect...")
        time.sleep(2)
        
        # Reconnect to the same session
        reconnect_result = tester.reconnect_session(session_id)
        
        # Phase 3: Verify restored state
        print("\n📋 PHASE 3: Verify Restored State")
        print("-" * 30)
        
        # Check that we're in the Kitchen
        print("📍 Checking current location...")
        look_result = tester.send_command("look")
        current_location = tester._extract_location_from_text(look_result.get("raw_response", ""))
        
        # Check inventory
        print("📦 Checking inventory...")
        inv_result = tester.get_inventory()
        restored_inventory = tester._extract_inventory_from_text(inv_result.get("raw_response", ""))
        
        # Verify session state
        restored_state = tester.get_session_state()
        restored_history = tester.get_session_history()
        restored_turns = len(restored_history.get("turns", []))
        
        print(f"📍 Current location: {current_location}")
        print(f"📦 Restored inventory: {restored_inventory}")
        print(f"📊 Restored turns: {restored_turns}")
        print(f"🎯 Active session: {restored_state.get('active', False)}")
        
        # Test movement from restored location
        print("\n🚶 Testing movement from restored location...")
        move_result = tester.send_command("west")
        new_location = tester._extract_location_from_text(move_result.get("raw_response", ""))
        print(f"🚪 After moving west: {new_location}")
        
        # Phase 4: Validate results
        print("\n📋 PHASE 4: Test Results")
        print("-" * 30)
        
        success = True
        
        # Check location
        if "kitchen" not in (current_location or "").lower():
            print("❌ FAIL: Expected to be in Kitchen")
            success = False
        else:
            print("✅ PASS: Location correctly restored (Kitchen)")
            
        # Check inventory
        has_sack = any("sack" in item.lower() for item in restored_inventory)
        if not has_sack:
            print("❌ FAIL: Expected brown sack in inventory")
            print(f"   Found inventory: {restored_inventory}")
            success = False
        else:
            print("✅ PASS: Inventory correctly restored (has sack)")
            
        # Check turn count
        if restored_turns < total_turns:
            print(f"❌ FAIL: Turn count not preserved ({restored_turns} < {total_turns})")
            success = False
        else:
            print(f"✅ PASS: Turn history preserved ({restored_turns} turns)")
            
        # Check movement from restored location
        if "living room" not in (new_location or "").lower():
            print(f"❌ FAIL: Expected to be in Living Room after moving west")
            print(f"   Found location: {new_location}")
            success = False
        else:
            print("✅ PASS: Movement from restored location works correctly (Living Room)")
            
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Save/restore functionality is working correctly")
        else:
            print("\n💥 SOME TESTS FAILED!")
            print("❌ Save/restore functionality needs investigation")
            
        return success
        
    except Exception as e:
        print(f"\n💥 TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            tester.close_session()
        except:
            pass


if __name__ == "__main__":
    success = run_save_restore_test()
    exit(0 if success else 1)