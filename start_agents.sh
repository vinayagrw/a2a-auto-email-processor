#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p data/chroma output/templates output/summaries config

# Set permissions for the directories
chmod -R 777 data output config

# Build and start the services
docker-compose up --build -d

echo "Waiting for services to start..."
sleep 5

echo "Services are running:"
echo "- Email Processor: http://localhost:8001"
echo "- Response Agent:  http://localhost:8002"
echo "- Summary Agent:  http://localhost:8003"

echo "\nTo view logs, run: docker-compose logs -f"
echo "To stop the services, run: docker-compose down"
