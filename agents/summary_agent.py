"""
Summary Agent for A2A MCP Contractor Automation.

This module handles generating summaries of incoming requests.
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import httpx
import uvicorn
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import Artifact as A2AArtifact
from a2a.types import Task as A2ATask
from chromadb.config import Settings
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import configuration
try:
    from .config import (
        A2A_SERVER_HOST,
        A2A_SERVER_PORT,
        SUMMARY_AGENT_PORT,
        CHROMA_PERSIST_DIR
    )
except ImportError:
    # Fall back to absolute import if running directly
    from config import (
        A2A_SERVER_HOST,
        A2A_SERVER_PORT,
        SUMMARY_AGENT_PORT,
        CHROMA_PERSIST_DIR
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("summary_agent.log")
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

class TaskModel(BaseModel):
    id: str
    artifacts: List[Dict[str, Any]]
    metadata: Optional[dict] = {}

class ArtifactModel(BaseModel):
    id: str
    type: str
    content: str
    metadata: Optional[dict] = {}

class SummaryAgent:
    def __init__(self):
        self.httpx_client = None
        self.client = None
        self.chroma_client = None
        self.collection = None
        self.initialized = False
        self.resolver = None
        self.agent_card = None
        
    async def initialize(self) -> bool:
        """
        Initialize the agent and its dependencies.
        
        Returns:
            bool: True if initialization was successful
            
        Raises:
            Exception: If initialization fails
        """
        try:
            # Initialize HTTP client
            self.httpx_client = httpx.AsyncClient()
            
            # Initialize A2A client with correct signature
            self.client = A2AClient(
                httpx_client=self.httpx_client,
                url=f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}"
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
                name="summaries",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize card resolver
            self.resolver = A2ACardResolver()
            
            # Register the agent card
            self.agent_card = {
                "type": "agent",
                "name": "summary_agent",
                "description": "Generates summaries of incoming content.",
                "version": "1.0.0"
            }
            
            self.initialized = True
            logger.info("SummaryAgent initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize SummaryAgent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

    async def process_task_internal(self, task: TaskModel):
        """
        Process a task that requires summarization.
        
        - **task**: The task containing artifacts to be processed
        - **returns**: A summary of the provided content
        """
        return await self._process_task(task)

    async def _process_task(self, task: TaskModel):
        try:
            # Extract email content and metadata
            email_content = None
            sender = None
            subject = None
            intent = None
            
            for artifact in task.artifacts:
                if artifact.type == "text/plain":
                    email_content = artifact.content
                    sender = artifact.metadata.get("sender")
                    subject = artifact.metadata.get("subject")
                elif artifact.type == "application/json" and artifact.metadata.get("type") == "classification":
                    intent = artifact.content

            if not email_content or intent != "summary_needed":
                raise ValueError("Invalid task format or intent")

            # Generate summary
            summary = await self.generate_summary(email_content)
            
            # Create response
            response = SuccessResponse(
                status="success",
                message="Summary generated successfully",
                summary=summary,
                sender=sender,
                subject=subject,
                timestamp=datetime.now().isoformat()
            )
            
            return response
        except Exception as e:
            print(f"Error processing task: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def generate_summary(self, content: str) -> str:
        """Generate a summary of the given content"""
        # This is a placeholder - implement your actual summarization logic here
        return f"Summary of: {content[:100]}..."

    async def initialize(self) -> bool:
        """
        Initialize the agent and its dependencies.
        
        Returns:
            bool: True if initialization was successful
            
        Raises:
            Exception: If initialization fails
        """
        try:
            # Initialize HTTP client
            self.httpx_client = httpx.AsyncClient()
            
            # Initialize A2A client with correct signature
            self.client = A2AClient(
                httpx_client=self.httpx_client,
                url=f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}"
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
                name="summaries",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize card resolver
            self.resolver = A2ACardResolver()
            
            # Register the agent card
            self.agent_card = {
                "type": "agent",
                "name": "summary_agent",
                "description": "Generates summaries of incoming content.",
                "version": "1.0.0"
            }
            
            self.initialized = True
            logger.info("SummaryAgent initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize SummaryAgent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Summary Agent API",
        description="API for processing summary tasks",
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
    agent = SummaryAgent()
    
    # Add routes
    @app.post(
        "/process_task",
        response_model=SuccessResponse,
        responses={
            200: {"model": SuccessResponse, "description": "Task processed successfully"},
            400: {"model": ErrorResponse, "description": "Invalid input"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        },
        summary="Process a task that requires summarization",
        description="Processes a task that requires summarization of the provided content"
    )
    async def process_task_route(task: TaskModel = Body(...)):
        """Process a task that requires summarization"""
        try:
            if not hasattr(agent, 'initialized') or not agent.initialized:
                await agent.initialize()
                
            result = await agent.process_task_internal(task)
            return SuccessResponse(
                message="Task processed successfully",
                data=result
            )
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Failed to process task: {str(e)}"}
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
            message="Summary Agent is running",
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
        port=SUMMARY_AGENT_PORT,
        log_level="info"
    )

# This allows the agent to be run directly with: python -m agents.summary_agent
if __name__ == "__main__":
    main()
