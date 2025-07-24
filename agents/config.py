import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Email configuration
GMAIL_API_CREDENTIALS = os.getenv("GMAIL_API_CREDENTIALS")
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# AI Configuration
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434/api/generate")
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma")

# A2A Configuration
A2A_SERVER_HOST = os.getenv("A2A_SERVER_HOST", "localhost")
A2A_SERVER_PORT = int(os.getenv("A2A_SERVER_PORT", 8000))

# Agent configurations
EMAIL_PROCESSOR_AGENT_PORT = int(os.getenv("EMAIL_PROCESSOR_AGENT_PORT", 8001))
RESPONSE_AGENT_PORT = int(os.getenv("RESPONSE_AGENT_PORT", 8002))
SUMMARY_AGENT_PORT = int(os.getenv("SUMMARY_AGENT_PORT", 8003))
