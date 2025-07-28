# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copy only the requirements files first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/chroma /app/logs

# Expose the ports the app runs on
EXPOSE 8000 8001 8002 8003

# Command to run the application (will be overridden by docker-compose)
CMD ["python", "-m", "a2a_agents.email_processor.__main__"]
