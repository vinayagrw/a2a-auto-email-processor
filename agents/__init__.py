"""
A2A MCP Contractor Automation - Agents Package

This package contains the agent implementations for the A2A MCP Contractor Automation system.
"""

__version__ = "0.1.0"

# Import configuration
from config import (
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
    LOG_LEVEL
)

from .email_processor_agent import create_app as create_email_processor_app
from .response_agent import create_app as create_response_app
from .summary_agent import create_app as create_summary_app



# Create FastAPI app instances lazily to avoid circular imports
def _get_email_processor_app():
    return create_email_processor_app()

def _get_response_app():
    return create_response_app()

def _get_summary_app():
    return create_summary_app()

# Lazy-loaded app instances
email_processor_app = None
response_app = None
summary_app = None

def _ensure_apps_initialized():
    global email_processor_app, response_app, summary_app
    if email_processor_app is None:
        email_processor_app = _get_email_processor_app()
    if response_app is None:
        response_app = _get_response_app()
    if summary_app is None:
        summary_app = _get_summary_app()

# Initialize apps on first access
_ensure_apps_initialized()


# Export all public symbols
__all__ = [
    # Agent modules
    "email_processor_agent",
    "response_agent",
    "summary_agent",
    
    # App creation functions
    "create_email_processor_app",
    "create_response_app",
    "create_summary_app",
    
    # App instances
    "email_processor_app",
    "response_app",
    "summary_app",
    
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
