#!/bin/bash

echo "Stopping all services..."
docker-compose down

echo "Removing unused containers, networks, and volumes..."
docker system prune -f

echo "Services have been stopped and cleaned up."
