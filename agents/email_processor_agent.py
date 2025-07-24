"""
Email Processor Agent for A2A MCP Contractor Automation.

This agent processes incoming emails and extracts relevant information.
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import chromadb
import httpx
import uvicorn
from a2a.client import A2AClient, A2ACardResolver
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
    EMAIL_PROCESSOR_AGENT_PORT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("email_processor_agent.log")
    ]
)
logger = logging.getLogger(__name__)

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

class EmailProcessorAgent:
    def __init__(self):
        self.httpx_client = None
        self.client = None
        self.chroma_client = None
        self.collection = None
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the agent and its dependencies."""
        try:
            # Initialize HTTP client
            self.httpx_client = httpx.AsyncClient()
            
            # Initialize A2A client
            self.client = A2AClient(
                server_url=f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}",
                client=self.httpx_client
            )
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=CHROMA_PERSIST_DIR
                )
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="emails",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.initialized = True
            logger.info("EmailProcessorAgent initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize EmailProcessorAgent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
    
    async def process_email(self, email: EmailModel) -> dict:
        """Process an incoming email."""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Store email in ChromaDB
            await self._store_email(email)
            
            # Process the email
            result = {
                "email_id": email.id,
                "status": "processed",
                "extracted_info": {
                    "subject": email.subject,
                    "sender": email.sender,
                    "recipients": email.recipients,
                    "received_at": email.received_at.isoformat(),
                },
                "metadata": email.metadata or {}
            }
            
            logger.info(f"Processed email {email.id}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process email {email.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": error_msg}
            )
    
    async def _store_email(self, email: EmailModel) -> str:
        """Store an email in ChromaDB."""
        try:
            doc_id = f"email_{email.id}"
            
            # Prepare metadata
            metadata = {
                "subject": email.subject,
                "sender": email.sender,
                "recipients": ", ".join(email.recipients),
                "received_at": email.received_at.isoformat(),
                "source": "email_processor_agent"
            }
            
            if email.metadata:
                metadata.update(email.metadata)
            
            # Prepare email data
            email_data = {
                "id": email.id,
                "subject": email.subject,
                "sender": email.sender,
                "recipients": email.recipients,
                "body": email.body,
                "received_at": email.received_at.isoformat(),
                "metadata": email.metadata or {}
            }
            
            # Add to ChromaDB
            self.collection.add(
                documents=[json.dumps(email_data)],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Stored email in ChromaDB with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to store email: {str(e)}", exc_info=True)
            pass

def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Email Processor Agent API",
        description="API for processing incoming emails",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize agent
    agent = EmailProcessorAgent()
    
    # Add routes
    @app.post(
        "/process_email",
        response_model=SuccessResponse,
        responses={
            200: {"model": SuccessResponse, "description": "Email processed successfully"},
            400: {"model": ErrorResponse, "description": "Invalid input"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        },
        summary="Process an incoming email",
        description="Processes an incoming email, extracts information, and stores it in the database"
    )
    async def process_email_route(email: EmailModel = Body(...)):
        """Process an incoming email."""
        try:
            result = await agent.process_email(email)
            return SuccessResponse(
                message="Email processed successfully",
                data=result
            )
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error processing email: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Failed to process email: {str(e)}"}
            )
    
    @app.get(
        "/health",
        response_model=SuccessResponse,
        summary="Health check endpoint",
        description="Check if the agent is running"
    )
    async def health_check():
        """Health check endpoint."""
        return SuccessResponse(
            message="Email Processor Agent is running",
            data={"status": "healthy"}
        )
    
    return app

def main():
    """Run the FastAPI application with uvicorn."""
    import uvicorn
    
    # Create the FastAPI app
    app = create_app()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=EMAIL_PROCESSOR_AGENT_PORT,
        log_level="info"
    )

if __name__ == "__main__":
    main()
