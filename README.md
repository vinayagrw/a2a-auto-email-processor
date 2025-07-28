# A2A Contractor Automation - Agent Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker)](https://www.docker.com/)

![Concept](.gitbook/concept.png)

## Overview

This document provides a comprehensive guide to the Agent-to-Agent (A2A) architecture used in the Contractor Automation system. The system is built using Google's A2A framework, which enables seamless communication between specialized agents to automate email processing, response generation, and summarization for contractor management.

## Architecture Overview

The system follows a microservices architecture where each agent is an independent service with specific responsibilities. Agents communicate using the A2A protocol over HTTP, enabling loose coupling and scalability.

![Architecture](.gitbook/architecture.png)



```
                                                     ┌──────────────────┐  
                                                     │   Response Agent │
                                                     └──────────────────┘
 ┌─────────────────┐        ┌─────────────────┐        
 │   Gmail API     │─────▶ │  Email Processor│────▶
 └─────────────────┘        └─────────────────┘     
                                                    ┌─────────────────┐
                                                    │   Summary Agent │
                                                    └─────────────────┘
```

## Agent Communication Flow

1. **Email Processor**
   - Listens for new emails via Gmail API
   - Processes and classifies incoming emails
   - Forwards emails to the appropriate agent based on classification
   - Maintains the state of email processing tasks

2. **Response Agent**
   - Generates contextually appropriate responses to emails
   - Handles direct queries and common questions
   - Implements response templates and dynamic content generation

3. **Summary Agent**
   - Processes and summarizes email threads
   - Maintains conversation history
   - Provides executive summaries of ongoing communications

## A2A Framework Implementation

The system leverages Google's A2A (Agent-to-Agent) framework, which provides:

### Core A2A Features

1. **Service Discovery**: Automatic discovery of available agents through Agent Cards
2. **Standardized Communication**: Unified API for inter-agent communication
3. **Task Management**: Built-in support for long-running tasks with progress tracking
4. **Error Handling**: Consistent error reporting and recovery mechanisms
5. **Streaming Support**: Real-time, incremental updates for tasks and artifacts

### A2A Streaming Protocol

The A2A framework supports real-time, incremental updates through Server-Sent Events (SSE), enabling efficient streaming of task status changes and artifacts. This is particularly useful for long-running tasks where immediate feedback is valuable.

#### Key Streaming Features

- **Real-time Updates**: Receive immediate notifications about task status changes
- **Efficient Data Transfer**: Only send incremental updates
- **Bi-directional Communication**: While primarily server-to-client, the protocol supports ongoing interaction
- **Support for Large Payloads**: Stream large artifacts in chunks



#### Response Format (SSE Stream)

The server responds with a stream of Server-Sent Events (SSE), where each event contains a JSON-RPC 2.0 response object.

##### Task Status Update Event

```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "result": {
    "taskId": "task-123",
    "contextId": "context-456",
    "kind": "status-update",
    "status": "in-progress",
    "final": false,
    "metadata": {
      "progress": 50,
      "message": "Processing your request..."
    }
  }
}
```

##### Artifact Update Event

```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "result": {
    "taskId": "task-123",
    "contextId": "context-456",
    "kind": "artifact-update",
    "artifact": {
      "id": "artifact-789",
      "type": "text/plain",
      "content": "This is a chunk of the artifact..."
    },
    "append": true,
    "lastChunk": false
  }
}
```

### Error Handling

- **Connection Errors**: Automatically retry with exponential backoff
- **Invalid Messages**: Log and skip malformed events
- **Rate Limiting**: Respect Retry-After headers
- **Timeouts**: Implement appropriate timeouts for both connection and response streaming

- **Service Discovery**: Automatic discovery of available agents
- **Standardized Communication**: Unified API for inter-agent communication
- **Task Management**: Built-in support for long-running tasks with progress tracking
- **Error Handling**: Consistent error reporting and recovery mechanisms

### Key A2A Components

1. **AgentCard**
   - Defines the agent's capabilities and metadata
   - Includes supported input/output formats
   - Specifies the agent's API endpoints

2. **AgentExecutor**
   - Implements the core business logic of each agent
   - Handles incoming requests and delegates to appropriate handlers
   - Manages task state and progress updates

3. **Request Handler**
   - Processes incoming HTTP requests
   - Validates input data
   - Routes requests to the appropriate executor methods

## Agent Implementation Details

### Email Processor Agent

**Purpose**: Processes incoming emails and routes them to the appropriate handler.

**Key Features**:
- Email parsing and validation
- Content classification
- Task initiation and management
- Error handling and retry logic

**Dependencies**:
- A2A Server
- Gmail API credentials

### Response Agent

**Purpose**: Generates appropriate responses to incoming emails.

**Key Features**:
- Response template management
- Dynamic content generation
- Context-aware response selection
- Multi-language support

**Dependencies**:
- A2A Server
- LLM integration (Ollama/LM Studio)

### Summary Agent

**Purpose**: Generates summaries of email threads and conversations.

**Key Features**:
- Thread analysis
- Key point extraction
- Summary generation
- Context preservation

**Dependencies**:
- A2A Server
- Vector database (ChromaDB)
- LLM integration

## Communication Protocol

Agents communicate using a standardized JSON-based protocol over HTTP. Each message includes:

```json
{
  "task_id": "unique-task-identifier",
  "action": "action-name",
  "parameters": {
    "key": "value"
  },
  "metadata": {
    "source": "source-agent",
    "timestamp": "ISO-8601-timestamp"
  }
}
```

## Error Handling

The system implements a comprehensive error handling strategy:

1. **Input Validation**: All inputs are validated against schemas
2. **Retry Logic**: Transient failures are automatically retried
3. **Circuit Breaker**: Prevents cascading failures
4. **Dead Letter Queue**: Failed messages are stored for later analysis

## Monitoring and Logging

- Structured logging with correlation IDs
- Performance metrics collection
- Health check endpoints
- Distributed tracing

## Security Considerations

- All inter-service communication is encrypted (HTTPS)
- API key authentication between services
- Role-based access control
- Input sanitization and validation

## 🚀 Quick Start with Docker

The easiest way to get started is using Docker Compose, which will set up all the necessary services with a single command.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- [Ollama](https://ollama.ai/) or another LLM service running locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/a2a-mcp-contractor-automation.git
cd a2a-mcp-contractor-automation/a2a_agents
```

### 2. Set Up Environment

Copy the example environment file and update it with your configuration:

```bash
cp .env.example .env
# Edit the .env file with your settings
```

### 3. Start the Services

```bash
docker-compose up --build -d
```

### 4. Access the Services

Once all services are up and running, you can access them at:

- **A2A Server**: http://localhost:8000
- **Email Processor**: http://localhost:8001
- **Response Agent**: http://localhost:8002
- **Summary Agent**: http://localhost:8003
- **ChromaDB UI**: http://localhost:8004

### 5. View Logs

To monitor the services:

```bash
# View all logs
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f email_processor
docker-compose logs -f response_agent
docker-compose logs -f summary_agent
```

## 🛠 Manual Setup (Development)

If you prefer to run the services manually outside of Docker:

### Prerequisites

- Python 3.13+
- [Poetry](https://python-poetry.org/) for dependency management
- Ollama or another LLM service
- ChromaDB or another vector database

### 1. Install Dependencies

```bash
# Install Python dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### 2. Configure Environment

Create and configure your `.env` file as shown in the Docker section.

### 3. Start the Services

In separate terminal windows:

```bash
# Start A2A Server
python -m a2a.server.main

# Start Email Processor
python -m a2a_agents.email_processor.__main__

# Start Response Agent
python -m a2a_agents.response_agent.__main__

# Start Summary Agent
python -m a2a_agents.summary_agent.__main__
```

## 🧪 Development

### Project Structure

```
a2a_agents/
├── a2a_agents/
│   ├── email_processor/  # Email processing agent
│   ├── response_agent/   # Response generation agent
│   └── summary_agent/    # Summary generation agent
├── tests/               # Test files
├── .env.example         # Example environment variables
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile           # Docker configuration
└── pyproject.toml       # Project dependencies
```

### Setting Up the Development Environment

1. **Install Dependencies**
   ```bash
   poetry install
   ```

2. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=a2a_agents tests/

# Run a specific test file
pytest tests/test_email_processor.py
```

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting


## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find and kill the process using a specific port
   sudo lsof -i :8000
   kill -9 <PID>
   ```

2. **Docker Container Issues**
   ```bash
   # Rebuild and restart all services
   docker-compose down
   docker-compose up --build -d
   ```

3. **LLM Connection Issues**
   - Ensure Ollama (or your LLM service) is running
   - Check the OLLAMA_API_BASE in your .env file
   - Verify network connectivity between containers

4. **ChromaDB Issues**
   - Ensure the data directory has proper permissions
   - Check if ChromaDB is accessible
   - Reset the database if needed (data will be lost):
     ```bash
     docker-compose down -v
     ```

### Viewing Logs

```bash
# View logs for all services
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f email_processor

# View the last 100 lines of logs
docker-compose logs --tail=100

# Follow logs in real-time
docker-compose logs -f --tail=50
```




## Future Enhancements

- Support for additional communication channels (Slack, Teams)
- Advanced analytics dashboard
- Automated testing framework
- Enhanced security features
- Multi-tenant support



MIT License - See [LICENSE](../LICENSE) for details
