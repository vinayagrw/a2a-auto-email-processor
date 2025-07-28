FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app
ENV A2A_SERVER_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .



# Create necessary directories
RUN mkdir -p /app/data/chroma /app/output/templates /app/output/summaries /app/config

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "a2a_agents.email_processor.__main__"]
