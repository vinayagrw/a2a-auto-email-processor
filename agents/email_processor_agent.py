"""
Email Processor Agent for A2A MCP Contractor Automation.

This agent processes incoming emails, classifies them, and delegates to appropriate agents.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
import httpx
import uvicorn
from a2a.client import A2AClient
from a2a.types import Artifact as A2AArtifact
from a2a.types import Task as A2ATask
from chromadb.config import Settings
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

# Import configuration
from config import (
    A2A_SERVER_HOST,
    A2A_SERVER_PORT,
    CHROMA_PERSIST_DIR,
    EMAIL_PROCESSOR_AGENT_PORT,
    RESPONSE_AGENT_PORT,
    SUMMARY_AGENT_PORT,
    LOG_LEVEL
)

# Import LLM classifier
from utils.llm_client import LLMClassifier

# Configure logging
import os
import sys
import argparse
from pathlib import Path

# Parse command line arguments first
parser = argparse.ArgumentParser()
parser.add_argument('--log-level', type=str.upper, default='DEBUG',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                   help='Set the logging level (DEBUG, INFO, WARNING, ERROR)')

# Parse known args and ignore the rest to avoid conflicts with other argument parsers
args, _ = parser.parse_known_args()

# Convert log level from string to logging level
try:
    log_level = getattr(logging, args.log_level.upper())
except (AttributeError, TypeError) as e:
    print(f"Invalid log level: {args.log_level}. Defaulting to INFO")
    log_level = logging.INFO

# Set up logging directory
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Clear any existing handlers to avoid duplicate logs
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()

# Configure root logger
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "email_processor_agent.log", mode='a'),  # Append to log file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info(f"Email Processor Agent logging initialized at level {args.log_level}")
logger.debug("Debug logging is enabled")

# Models
class SuccessResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[dict] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[dict] = None

class EmailModel(BaseModel):
    id: str
    subject: str
    sender: EmailStr
    recipients: List[EmailStr]
    body: str
    received_at: datetime
    metadata: Optional[dict] = {}
    attachments: Optional[List[Dict[str, Any]]] = []

class EmailProcessorAgent:
    def __init__(self):
        self.httpx_client = None
        self.client = None
        self.chroma_client = None
        self.collection = None
        self.llm_classifier = None
        self.initialized = False
        self.response_agent_url = f"http://localhost:{RESPONSE_AGENT_PORT}"
        self.summary_agent_url = f"http://localhost:{SUMMARY_AGENT_PORT}"
        
    async def initialize(self) -> bool:
        """Initialize the agent and its dependencies."""
        try:
            # Initialize HTTP client
            self.httpx_client = httpx.AsyncClient()
            
            # Initialize A2A client
            self.client = A2AClient(
                httpx_client=self.httpx_client,
                url=f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}"
            )
            
            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            
            # Create or get collection for email storage
            self.collection = self.chroma_client.get_or_create_collection(
                name="emails",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize LLM classifier
            self.llm_classifier = LLMClassifier()
            
            self.initialized = True
            logger.info("EmailProcessorAgent initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize EmailProcessorAgent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.httpx_client:
                await self.httpx_client.aclose()
            raise Exception(error_msg)
    
    async def _classify_email(self, email: EmailModel) -> str:
        """Classify the email intent using the LLM classifier.
        
        Args:
            email: The email to classify
            
        Returns:
            str: The classified intent ("quote_request", "common_question", or "summary_needed")
        """
        try:
            if not self.initialized or not self.llm_classifier:
                await self.initialize()
                
            # Use LLM for classification
            intent = await self.llm_classifier.classify_email(
                email_subject=email.subject,
                email_body=email.body
            )
            
            logger.info(f"Classified email '{email.subject}' as: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Error in LLM email classification: {str(e)}")
            # Fallback to keyword-based classification if LLM fails
            try:
                subject = email.subject.lower()
                body = email.body.lower()
                
                # Check for summary-related keywords
                summary_keywords = ["daily report", "summary", "update", "status"]
                if any(keyword in subject or keyword in body for keyword in summary_keywords):
                    return "summary_needed"
                    
                # Check for quote request keywords
                quote_keywords = ["quote", "pricing", "cost", "how much", "price"]
                if any(keyword in subject or keyword in body for keyword in quote_keywords):
                    return "quote_request"
                    
            except Exception as fallback_error:
                logger.error(f"Fallback classification failed: {str(fallback_error)}")
                
            return "common_question"  # Default fallback
            
    async def _delegate_task(self, task_data: dict, intent: str) -> dict:
        """Delegate task to the appropriate agent.
        
        Args:
            task_data: The task data to delegate
            intent: The classified intent
            
        Returns:
            dict: The response from the delegated agent
        """
        try:
            target_url = (
                f"{self.response_agent_url}/process"
                if intent in ["quote_request", "common_question"]
                else f"{self.summary_agent_url}/summarize"
            )
            
            logger.info(f"Delegating task to {target_url} with intent: {intent}")
            
            response = await self.httpx_client.post(
                target_url,
                json=task_data,
                timeout=300.0  # 5 minute timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} while delegating task: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except Exception as e:
            error_msg = f"Error delegating task: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    async def process_email(self, email: EmailModel) -> dict:
        """Process an incoming email through the complete workflow.
        
        1. Store the email in ChromaDB
        2. Classify the email intent using LLM
        3. Create an A2A task with the email content and metadata
        4. Delegate to the appropriate agent based on intent
        5. Return the processing results
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # 1. Store email in ChromaDB for future reference
            await self._store_email(email)
            logger.info(f"Stored email {email.id} in ChromaDB")
            
            # 2. Classify the email using LLM
            intent = await self._classify_email(email)
            logger.info(f"Classified email '{email.subject}' as intent: {intent}")
            
            # 3. Create A2A task with email content and metadata
            task_data = {
                "id": f"email_task_{email.id}",
                "artifacts": [
                    {
                        "id": f"email_{email.id}",
                        "type": "text/plain",
                        "content": email.body,
                        "metadata": {
                            "sender": email.sender,
                            "recipients": email.recipients,
                            "subject": email.subject,
                            "date": email.received_at.isoformat(),
                            "intent": intent,
                            "has_attachments": bool(email.attachments),
                            **email.metadata
                        }
                    }
                ],
                "metadata": {
                    "source": "email_processor",
                    "received_at": email.received_at.isoformat(),
                    "processed_at": datetime.now().isoformat(),
                    "intent": intent,
                    "priority": "high" if intent == "quote_request" else "normal"
                }
            }
            
            # 4. Delegate to appropriate agent based on intent
            logger.info(f"Delegating email '{email.subject}' to {'response' if intent in ['quote_request', 'common_question'] else 'summary'} agent")
            result = await self._delegate_task(task_data, intent)
            
            # 5. Prepare and return response
            response = {
                "email_id": email.id,
                "status": "processed",
                "intent": intent,
                "delegated_to": "response_agent" if intent in ["quote_request", "common_question"] else "summary_agent",
                "delegated_at": datetime.now().isoformat(),
                "result": result,
                "metadata": {
                    "sender": email.sender,
                    "subject": email.subject,
                    "received_at": email.received_at.isoformat(),
                    **email.metadata
                }
            }
            
            logger.info(f"Successfully processed email '{email.subject}' (ID: {email.id})")
            return response
            
        except Exception as e:
            error_msg = f"Failed to process email {getattr(email, 'id', 'unknown')}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return error response
            return {
                "email_id": getattr(email, 'id', 'unknown'),
                "status": "error",
                "error": str(e),
                "metadata": getattr(email, 'metadata', {})
            }
    
    async def _store_email(self, email: EmailModel) -> str:
        """Store an email in ChromaDB with proper metadata.
        
        Args:
            email: The email to store
            
        Returns:
            str: The document ID in ChromaDB
            
        Raises:
            Exception: If storage fails
        """
        try:
            if not self.initialized:
                await self.initialize()
                
            # Prepare document content (combine subject and body for better search)
            document = f"Subject: {email.subject}\n\n{email.body}"
            
            # Prepare metadata for ChromaDB
            metadata = {
                "source": "email",
                "sender": email.sender,
                "recipients": ", ".join(email.recipients) if email.recipients else "",
                "subject": email.subject,
                "received_at": email.received_at.isoformat(),
                "has_attachments": bool(email.attachments),
                "attachment_count": len(email.attachments) if email.attachments else 0,
                "intent": "pending",  # Will be updated after classification
                **email.metadata  # Include any additional metadata
            }
            
            # Generate a unique document ID
            doc_id = email.id if hasattr(email, 'id') and email.id else f"email_{int(datetime.now().timestamp())}"
            
            # Store in ChromaDB
            self.collection.upsert(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
            
            logger.debug(f"Stored email in ChromaDB - ID: {doc_id}, Subject: {email.subject}")
            return doc_id
            
        except Exception as e:
            error_msg = f"Failed to store email in ChromaDB: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Create a global instance of the agent
agent = EmailProcessorAgent()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup logic
    try:
        await agent.initialize()
        logger.info("Email Processor Agent started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize EmailProcessorAgent: {str(e)}")
        raise
    
    yield  # This is where the application runs
    
    # Shutdown logic
    if agent.httpx_client:
        await agent.httpx_client.aclose()
    logger.info("Email Processor Agent shutdown complete")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Email Processor Agent API",
        description="API for the Email Processor Agent that processes incoming emails",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routes
    @app.post(
        "/process_email",
        response_model=SuccessResponse,
        responses={
            200: {"model": SuccessResponse},
            400: {"model": ErrorResponse},
            500: {"model": ErrorResponse}
        }
    )
    async def process_email_route(email: EmailModel):
        """
        Process an incoming email through the complete workflow.
        
        - **email**: The email to process
        - **returns**: The processing results
        """
        try:
            logger.info(f"Processing email from {email.sender} with subject: {email.subject}")
            result = await agent.process_email(email)
            logger.info(f"Email processed successfully: {result}")
            return {
                "success": True,
                "message": "Email processed successfully",
                "data": result
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            error_msg = f"Failed to process email: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": error_msg}
            )
    
    @app.get(
        "/health",
        response_model=SuccessResponse,
        summary="Health check endpoint",
        description="Check if the agent is running"
    )
    async def health_check():
        """Health check endpoint."""
        return {
            "success": True,
            "message": "Email Processor Agent is running",
            "data": {
                "status": "healthy",
                "version": "1.0.0"
            }
        }
    
    return app

def main():
    """Run the FastAPI application with uvicorn."""
    import argparse
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Email Processor Agent')
    parser.add_argument('--log-level', type=str, default='info',
                      choices=['debug', 'info', 'warning', 'error'],
                      help='Set the logging level')
    args = parser.parse_args()
    
    # Convert log level string to logging level
    log_level = getattr(logging, args.log_level.upper())
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("email_processor_agent.log")
        ]
    )
    
    # Set uvicorn log level to match
    uvicorn_log_level = args.log_level if args.log_level != 'debug' else 'debug'
    
    # Create the FastAPI app
    app = create_app()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=EMAIL_PROCESSOR_AGENT_PORT,
        log_level=uvicorn_log_level
    )

if __name__ == "__main__":
    main()
