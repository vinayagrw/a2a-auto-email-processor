"""Configuration settings for A2A MCP Contractor Automation."""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
DRAFTS_DIR = BASE_DIR / "drafts"
SUMMARIES_DIR = BASE_DIR / "summaries"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DRAFTS_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# A2A Server Configuration
A2A_SERVER_HOST = os.getenv("A2A_SERVER_HOST", "0.0.0.0")
A2A_SERVER_PORT = int(os.getenv("A2A_SERVER_PORT", "8000"))  # Default A2A server port
A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}")

# Agent Ports
EMAIL_PROCESSOR_AGENT_PORT = int(os.getenv("EMAIL_PROCESSOR_AGENT_PORT", "8001"))
RESPONSE_AGENT_PORT = int(os.getenv("RESPONSE_AGENT_PORT", "8002"))
SUMMARY_AGENT_PORT = int(os.getenv("SUMMARY_AGENT_PORT", "8003"))

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))

# LLM Configuration
OLLAMA_API_BASE = os.getenv("LLM_URL", "http://localhost:11434/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

# Gmail API Configuration (if needed)
GMAIL_CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
GMAIL_TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE", "token.json")
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "a2a_mcp.log"))
