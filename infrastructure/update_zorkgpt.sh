#!/bin/bash

# ZorkGPT Update Script
# This script handles the complete update process locally on the EC2 instance
# to avoid issues with multiple SSH calls

# Don't exit on error - we want to handle errors explicitly
# set -e  # Commented out for better error visibility

ZORKGPT_DIR="/home/zorkgpt/ZorkGPT"
ZORKGPT_USER="zorkgpt"
SAVE_SIGNAL_FILE="$ZORKGPT_DIR/.SAVE_REQUESTED_BY_SYSTEM"
MAX_WAIT_TIME=60
WAIT_INTERVAL=3

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
    
    if ! id "$ZORKGPT_USER" &>/dev/null; then
        log_error "User $ZORKGPT_USER does not exist"
        exit 1
    fi
    
    if [[ ! -d "$ZORKGPT_DIR" ]]; then
        log_error "ZorkGPT directory $ZORKGPT_DIR does not exist"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

trigger_save() {
    log "Triggering game save before update..."
    
    # Create save signal file as zorkgpt user
    if sudo -u "$ZORKGPT_USER" touch "$SAVE_SIGNAL_FILE"; then
        log "Save signal created successfully"
    else
        log_error "Failed to create save signal"
        return 1
    fi
}

wait_for_save_completion() {
    log "Waiting for ZorkGPT to process save signal..."
    
    local total_waited=0
    
    while [[ $total_waited -lt $MAX_WAIT_TIME ]]; do
        sleep $WAIT_INTERVAL
        total_waited=$((total_waited + WAIT_INTERVAL))
        
        # Check if signal file still exists
        if sudo -u "$ZORKGPT_USER" test ! -f "$SAVE_SIGNAL_FILE"; then
            log "Save signal was processed by ZorkGPT"
            return 0
        else
            log "Still waiting for save... (${total_waited}s elapsed)"
        fi
    done
    
    log "Timeout waiting for save signal processing - proceeding anyway"
    return 1
}

check_save_files() {
    log "Checking save file status..."
    
    # Check if game_files directory exists
    if sudo -u "$ZORKGPT_USER" test -d "$ZORKGPT_DIR/game_files"; then
        log "Game files directory exists:"
        sudo -u "$ZORKGPT_USER" ls -la "$ZORKGPT_DIR/game_files/" 2>/dev/null || log "Could not list game_files directory"
    else
        log "Game files directory not found"
    fi
    
    # Check for current_state.json
    if sudo -u "$ZORKGPT_USER" test -f "$ZORKGPT_DIR/current_state.json"; then
        log "current_state.json exists"
    else
        log "current_state.json missing"
    fi
    
    # Check for any .qzl files
    local qzl_files
    qzl_files=$(sudo -u "$ZORKGPT_USER" find "$ZORKGPT_DIR" -name "*.qzl" 2>/dev/null | wc -l)
    if [[ $qzl_files -gt 0 ]]; then
        log "Found $qzl_files .qzl files:"
        sudo -u "$ZORKGPT_USER" find "$ZORKGPT_DIR" -name "*.qzl" -ls 2>/dev/null
    else
        log "No .qzl files found"
    fi
}

stop_service() {
    log "Stopping ZorkGPT service..."
    
    if systemctl stop zorkgpt; then
        log "ZorkGPT service stopped successfully"
        return 0
    else
        log_error "Failed to stop ZorkGPT service"
        return 1
    fi
}

update_code() {
    log "Updating ZorkGPT code..."
    
    # Change to ZorkGPT directory and pull latest code as zorkgpt user
    if sudo -u "$ZORKGPT_USER" bash -c "cd '$ZORKGPT_DIR' && git pull"; then
        log "ZorkGPT code updated successfully"
        return 0
    else
        log_error "Failed to update ZorkGPT code"
        return 1
    fi
}

start_service() {
    log "Starting ZorkGPT service..."
    
    if systemctl start zorkgpt; then
        log "ZorkGPT service started successfully"
        return 0
    else
        log_error "Failed to start ZorkGPT service"
        return 1
    fi
}

check_service_status() {
    log "Checking ZorkGPT service status..."
    
    # Wait a moment for service to start
    sleep 5
    
    if systemctl is-active zorkgpt --quiet; then
        log "ZorkGPT service is active"
        systemctl status zorkgpt --no-pager -l
        return 0
    else
        log_error "ZorkGPT service is not active"
        systemctl status zorkgpt --no-pager -l
        return 1
    fi
}

main() {
    log "Starting ZorkGPT update process..."
    
    # Step 1: Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed, aborting update"
        exit 1
    fi
    
    # Step 2: Trigger save
    if ! trigger_save; then
        log_error "Failed to trigger save, aborting update"
        exit 1
    fi
    
    # Step 3: Wait for save completion
    wait_for_save_completion
    
    # Step 4: Check save files
    check_save_files
    
    # Step 5: Stop service
    log "About to stop service..."
    if ! stop_service; then
        log_error "Failed to stop service, aborting update"
        exit 1
    fi
    
    # Step 6: Update code
    log "About to update code..."
    if ! update_code; then
        log_error "Code update failed, attempting to restart service anyway"
        start_service
        exit 1
    fi
    
    # Step 7: Start service
    log "About to start service..."
    if ! start_service; then
        log_error "Failed to start service after update"
        exit 1
    fi
    
    # Step 8: Check final status
    log "About to check service status..."
    if check_service_status; then
        log "ZorkGPT update completed successfully!"
        log "Game state will be automatically restored from save file"
        log "Monitor logs with: sudo journalctl -u zorkgpt -f"
    else
        log_error "Update completed but service status check failed"
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    log "Script interrupted, attempting to start service..."
    start_service || true
    exit 1
}

trap cleanup INT TERM

# Run main function
main 