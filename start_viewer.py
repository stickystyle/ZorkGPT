#!/usr/bin/env python3
"""
Simple launcher for ZorkGPT Live Viewer.

This script starts a local HTTP server and opens the viewer in your default browser.
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import os
import sys


def start_server(port=8000):
    """Start HTTP server in a separate thread."""
    handler = http.server.SimpleHTTPRequestHandler

    # Suppress server logs for cleaner output
    class QuietHandler(handler):
        def log_message(self, format, *args):
            pass

    with socketserver.TCPServer(("", port), QuietHandler) as httpd:
        print(f"🌐 HTTP server started at http://localhost:{port}")
        print(f"📁 Serving files from: {os.getcwd()}")
        httpd.serve_forever()


def main():
    port = 8000

    print("🎮 ZorkGPT Live Viewer Launcher")
    print("=" * 40)

    # Check if viewer file exists
    if not os.path.exists("zork_viewer.html"):
        print("❌ Error: zork_viewer.html not found in current directory")
        print("   Make sure you're running this from the ZorkGPT project root")
        sys.exit(1)

    # Check if state file exists
    state_files = ["current_state.json", "test_current_state.json"]
    state_file_exists = any(os.path.exists(f) for f in state_files)

    if not state_file_exists:
        print("⚠️  Warning: No state file found")
        print("   Run ZorkGPT first to generate current_state.json")
        print("   Or run test_state_export.py for a quick test")

    # Start server in background thread
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    # Give server time to start
    time.sleep(1)

    # Open browser
    viewer_url = f"http://localhost:{port}/zork_viewer.html"
    print(f"🚀 Opening viewer at: {viewer_url}")

    try:
        webbrowser.open(viewer_url)
        print("✅ Browser opened successfully")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"   Please manually open: {viewer_url}")

    print("\n📋 Instructions:")
    print("1. The viewer will automatically refresh every 3 seconds")
    print("2. Run ZorkGPT in another terminal to see live updates")
    print("3. Press Ctrl+C to stop the server")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
        print("   Browser tab will stop updating but can remain open")


if __name__ == "__main__":
    main()
