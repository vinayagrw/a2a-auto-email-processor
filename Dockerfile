FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/output/templates /app/output/summaries

# Set environment variables
ENV PYTHONPATH=/app
ENV A2A_SERVER_HOST=0.0.0.0

# Default command
CMD ["python", "-m", "uvicorn", "agents.email_processor_agent:app", "--host", "0.0.0.0", "--port", "8001"]
