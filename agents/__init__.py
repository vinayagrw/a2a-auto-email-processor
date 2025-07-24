"""
A2A MCP Contractor Automation - Agents Package

This package contains the agent implementations for the A2A MCP Contractor Automation system.
"""

__version__ = "0.1.0"

# Import configuration
from .config import (
    BASE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    GMAIL_API_CREDENTIALS,
    GMAIL_SCOPES,
    LLM_URL,
    CHROMA_PERSIST_DIR,
    A2A_SERVER_HOST,
    A2A_SERVER_PORT,
    EMAIL_PROCESSOR_AGENT_PORT,
    RESPONSE_AGENT_PORT,
    SUMMARY_AGENT_PORT,
)

# Import agent classes
from .email_processor_agent import EmailProcessorAgent
from .response_agent import ResponseAgent, app as response_app
from .summary_agent import SummaryAgent, app as summary_app

# Import models
from .models import (
    ArtifactModel,
    ArtifactType,
    Email,
    ErrorResponse,
    SuccessResponse,
    TaskModel,
    TaskStatus,
)

# Import utilities
from .utils import create_fastapi_app, get_environment_variable

# Export all public symbols
__all__ = [
    # Configuration
    "BASE_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "GMAIL_API_CREDENTIALS",
    "GMAIL_SCOPES",
    "LLM_URL",
    "CHROMA_PERSIST_DIR",
    "A2A_SERVER_HOST",
    "A2A_SERVER_PORT",
    "EMAIL_PROCESSOR_AGENT_PORT",
    "RESPONSE_AGENT_PORT",
    "SUMMARY_AGENT_PORT",
    
    # Agent classes
    "EmailProcessorAgent",
    "ResponseAgent",
    "SummaryAgent",
    
    # FastAPI apps
    "response_app",
    "summary_app",
    
    # Models
    "ArtifactModel",
    "ArtifactType",
    "Email",
    "ErrorResponse",
    "SuccessResponse",
    "TaskModel",
    "TaskStatus",
    
    # Utilities
    "create_fastapi_app",
    "get_environment_variable",
]
