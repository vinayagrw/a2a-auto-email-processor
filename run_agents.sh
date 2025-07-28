#!/bin/bash

# Set the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist
LOG_DIR="$PROJECT_ROOT/logs"

rm -rf "$LOG_DIR"

mkdir -p "$LOG_DIR"

# Function to start an agent
start_agent() {
    local agent_name=$1
    local agent_module=$2
    local log_file="$LOG_DIR/${agent_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=== Starting $agent_name ===" | tee -a "$log_file"
    # Run with debug logging enabled
    PYTHONPATH="$PROJECT_ROOT" \
    python -m "agents.$agent_module" --log_level debug 2>&1 | tee -a "$log_file" &
    echo "$!" > "$LOG_DIR/${agent_name}.pid"
    echo "Started $agent_name (PID: $(cat "$LOG_DIR/${agent_name}.pid"))"
    sleep 2
}

# Clean up function
cleanup() {
    echo -e "\nStopping all agents..."
    for agent in "EmailProcessor" "Response" "Summary"; do
        if [ -f "$LOG_DIR/${agent}.pid" ]; then
            pid=$(cat "$LOG_DIR/${agent}.pid")
            kill "$pid" 2>/dev/null && echo "Stopped $agent (PID: $pid)" || echo "Could not stop $agent"
            rm -f "$LOG_DIR/${agent}.pid"
        fi
    done
    exit 0
}

# Set up trap
trap cleanup SIGINT

# Start all agents
start_agent "EmailProcessor" "email_processor_agent"
start_agent "Response" "response_agent"
start_agent "Summary" "summary_agent"

echo -e "\nAll agents started. Press Ctrl+C to stop all agents."
echo "Logs are being written to: $LOG_DIR"

# Keep script running
wait