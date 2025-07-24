# A2A Contractor Email Automation

A lightweight demo application showcasing Agent-to-Agent (A2A) communication using A2A SDK for automating email management in a small business contractor setting.

## Features

- Automated email processing and classification
- Intelligent response generation for quote requests
- Daily email interaction summaries
- Clean A2A architecture with three collaborating agents
- Local AI infrastructure integration
- Containerized deployment

## Architecture

The system consists of three main agents:

1. **EmailProcessorAgent**
   - Monitors and processes incoming emails
   - Classifies email intent using local LLM
   - Delegates tasks to appropriate agents
   - Manages artifact exchange

2. **ResponseAgent**
   - Generates customized responses for quote requests
   - Uses local templates and LLM for response customization
   - Maintains response consistency

3. **SummaryAgent**
   - Creates concise summaries of important email interactions
   - Maintains daily logs of email summaries
   - Uses local LLM for summarization

## Prerequisites

1. Docker and Docker Compose
2. Python 3.13
3. Local LLM (Ollama or LM Studio)
4. ChromaDB (latest version)
5. Gmail API credentials

## Recent Updates

### ChromaDB Migration (2024-07-23)
- Updated to use the new `PersistentClient` API
- Removed deprecated `Settings` configuration
- Simplified ChromaDB initialization

To upgrade an existing installation:
```bash
pip install --upgrade chromadb
# If you have existing data, migrate it using:
# pip install chroma-migrate
# chroma-migrate
```

## Testing the System

### 1. Start the Summary Agent
```bash
python -m agents.summary_agent
```

### 2. In a new terminal, run the email processor test
```bash
python test_email_processor.py
```

### Expected Output
- The test script will:
  1. Authenticate with Gmail API (first time will open browser)
  2. Fetch the latest 3 emails
  3. Send them to the summary agent
  4. Display the summary results

### Verifying the Results
1. Check the terminal output for summary results
2. Review `summary_agent.log` for detailed logs
3. The agent's API is available at `http://localhost:8003`

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Ollama**
   ```bash
   # Install Ollama
   curl https://ollama.ai/install.sh | sh
   
   # Run Ollama
   ollama serve
   
   # Pull a model (e.g., llama2)
   ollama pull llama2
   ```

3. **Set up Gmail API**
   - Go to Google Cloud Console
   - Create a new project
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download credentials.json and place it in the project root

4. **Environment Variables**
   Create a `.env` file with:
   ```
   GMAIL_API_CREDENTIALS=/path/to/credentials.json
   LLM_URL=http://localhost:11434/api/generate
   ```

5. **Run with Docker**
   ```bash
   # Build and run containers
   docker-compose up --build
   ```

6. **Run Directly**
   ```bash
   # Run EmailProcessorAgent
   python agents/email_processor_agent.py
   
   # Run ResponseAgent
   python agents/response_agent.py
   
   # Run SummaryAgent
   python agents/summary_agent.py
   ```

## Usage

1. **Process Emails**
   ```bash
   curl -X POST http://localhost:8001/process_email \
   -H "Content-Type: application/json" \
   -d '{
     "sender": "customer@example.com",
     "subject": "Request for Quote",
     "body": "I'm interested in your services..."
   }'
   ```

2. **Check Outputs**
   - Responses: `output/templates/`
   - Summaries: `output/summaries/daily_summary_YYYY-MM-DD.txt`

## Code Structure

```
a2a-mcp-contractor-automation/
├── agents/
│   ├── email_processor_agent.py
│   ├── response_agent.py
│   └── summary_agent.py
├── config.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── output/
    ├── templates/
    └── summaries/
```

## AI Integration

The system uses:
- Local LLM (Ollama/LM Studio) for:
  - Email intent classification
  - Response generation
  - Summary creation
- ChromaDB for:
  - Context storage
  - Template management
  - Summary history



## License

MIT License - see LICENSE file for details