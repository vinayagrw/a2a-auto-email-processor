#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p data/chroma output/templates output/summaries config

# Set permissions for the directories
chmod -R 777 data output config

# Start each agent in the background
echo "Starting agents..."

# Start Email Processor
python -m  agents.email_processor_agent:app --host 0.0.0.0 --port 8001 &
EMAIL_PID=$!
echo "Email Processor started with PID $EMAIL_PID"

# Start Response Agent
python -m  agents.response_agent:app --host 0.0.0.0 --port 8002 &
RESPONSE_PID=$!
echo "Response Agent started with PID $RESPONSE_PID"

# Start Summary Agent
python -m  agents.summary_agent:app --host 0.0.0.0 --port 8003 &
SUMMARY_PID=$!
echo "Summary Agent started with PID $SUMMARY_PID"

# Save PIDs to a file
echo $EMAIL_PID > logs/EmailProcessor.pid
echo $RESPONSE_PID > logs/Response.pid
echo $SUMMARY_PID > logs/Summary.pid

echo "\nAll agents are running:"
echo "- Email Processor (PID: $EMAIL_PID): http://localhost:8001/health"
echo "- Response Agent  (PID: $RESPONSE_PID): http://localhost:8002/health"
echo "- Summary Agent   (PID: $SUMMARY_PID): http://localhost:8003/health"

echo "\nTo stop all agents, run: ./stop_agents.sh"
