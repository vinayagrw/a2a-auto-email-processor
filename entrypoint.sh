#!/bin/bash
set -e

# Create necessary directories if they don't exist
mkdir -p /app/data/chroma /app/logs /app/output /app/agents /app/utils /app/config

# Set proper permissions
chown -R 1000:1000 /app/data /app/logs /app/output

# Initialize application
if [ "$1" = "python" ] && [ "$2" = "-m" ] && [ "$3" = "a2a.server" ]; then
    echo "Starting A2A Server..."
    exec python -m uvicorn a2a.server.main:app --host 0.0.0.0 --port 8000
elif [ "$1" = "python" ] && [ "$2" = "-m" ] && [ "$3" = "a2a_agents.email_processor.__main__" ]; then
    echo "Starting Email Processor..."
    exec python -m a2a_agents.email_processor.__main__
elif [ "$1" = "python" ] && [ "$2" = "-m" ] && [ "$3" = "a2a_agents.response_agent.__main__" ]; then
    echo "Starting Response Agent..."
    exec python -m a2a_agents.response_agent.__main__
elif [ "$1" = "python" ] && [ "$2" = "-m" ] && [ "$3" = "a2a_agents.summary_agent.__main__" ]; then
    echo "Starting Summary Agent..."
    exec python -m a2a_agents.summary_agent.__main__
else
    # Execute the passed command
    exec "$@"
fi
