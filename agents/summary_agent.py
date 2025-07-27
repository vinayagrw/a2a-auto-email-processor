"""
Summary Agent for A2A MCP Contractor Automation.

This module handles generating summaries of incoming requests.
"""
import sys
import socket
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
import uvicorn
from a2a.client import A2AClient,A2ACardResolver
from a2a.types import Artifact as A2AArtifact
from a2a.types import Task as A2ATask
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Import configuration
from config import (
    SUMMARY_AGENT_PORT,
    LOG_LEVEL,
    LOG_FILE,
    OLLAMA_API_BASE,
    OLLAMA_MODEL
)


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
    level=LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "summary_agent.log", mode='a'),  # Append to log file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info(f"Summary Agent logging initialized at level {LOG_LEVEL}")

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
    def __init__(self, base_url: str = None, model: str = None):
        self.httpx_client = None
        self.client = None
        self.initialized = False
        self.resolver = None
        self.agent_card = None
        self.a2a_server_url = os.getenv("A2A_SERVER_URL", "http://a2a_server:8000")
        self.service_name = "summary_agent"
        self.service_port = int(os.getenv("SUMMARY_AGENT_PORT", "8003"))
        self.base_url = base_url or OLLAMA_API_BASE
        self.model = model or OLLAMA_MODEL

    async def initialize(self) -> bool:
        """
        Initialize the agent, register with A2A server, and set up dependencies.
        
        Returns:
            bool: True if initialization was successful
            
        Raises:
            Exception: If initialization fails
        """
        try:
            # Initialize HTTP client
            self.httpx_client = httpx.AsyncClient()
            
            # Initialize A2A client with the configured server URL
            self.client = A2AClient(
                httpx_client=self.httpx_client,
                url=self.a2a_server_url
            )
            
            # Set up agent card with service information
            service_url = f"http://{socket.gethostname()}:{self.service_port}"
            # Register with A2A server using direct HTTP call
            registration_data = {
                "name": self.service_name,
                "url": service_url,
                "type": "agent",
                "capabilities": ["summarization"]
            }

            self.agent_card = {
                "type": "agent",
                "name": self.service_name,
                "description": "Generates summaries of incoming requests",
                "version": "1.0.0",
                "url": service_url,
                "capabilities": ["summarization"],
                "status": "ready"
            }
            
            try:
                response = await self.httpx_client.post(
                    f"{self.a2a_server_url}/services",
                    json=registration_data
                )
                response.raise_for_status()
                logger.info(f"Successfully registered with A2A server at {self.a2a_server_url}")
                
                self.initialized = True
                logger.info(f"Summary Agent initialized successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Could not register with A2A server: {str(e)}")
    
            
        except Exception as e:
            error_msg = f"Failed to initialize SummaryAgent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.httpx_client:
                await self.httpx_client.aclose()
            self.initialized = False
            raise Exception(error_msg) from e
    

    async def process_task_internal(self, task: TaskModel) -> Dict[str, Any]:
        """
        Process a task that requires summarization.
        
        - **task**: The task containing artifacts to be processed
        - **returns**: A summary of the provided content with metadata
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            # Process each artifact in the task
            summaries = []
            for artifact in task.artifacts:
                if artifact.get("type") == "text/plain":
                    content = artifact.get("content", "")
                    metadata = {
                        "sender": artifact.get("metadata", {}).get("sender", "Unknown"),
                        "subject": artifact.get("metadata", {}).get("subject", "No Subject"),
                        "date": artifact.get("metadata", {}).get("date", datetime.now().isoformat())
                    }
                    
                    # Generate summary with metadata for logging
                    summary = await self.generate_summary(content, metadata)
                    
                    summaries.append({
                        "artifact_id": artifact.get("id"),
                        "summary": summary,
                        "metadata": metadata
                    })
                    
                    logger.info(f"Generated summary for artifact {artifact.get('id')}")
            
            # Prepare the response
            response = {
                "task_id": task.id,
                "status": "completed",
                "summaries": summaries,
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    **task.metadata
                } if task.metadata else {"processed_at": datetime.now().isoformat()}
            }
            
            logger.info(f"Successfully processed task {task.id}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Failed to process task: {str(e)}"}
            )

    async def _process_task(self, task: TaskModel) -> Dict[str, Any]:
        """Wrapper method for process_task_internal with error handling"""
        try:
            return await self.process_task_internal(task)
        except Exception as e:
            logger.error(f"Error in _process_task: {str(e)}")
            raise
            
    async def _store_summary(self, content: str, metadata: dict) -> dict:
        """
        Prepare summary data for storage.
        
        Args:
            content: The summary content
            metadata: Additional metadata for the summary
            
        Returns:
            dict: The summary data to be stored
        """
        return {
            "content": content,
            "metadata": {
                "type": "summary",
                "created_at": datetime.now().isoformat(),
                **metadata
            }
        }

    async def generate_summary(self, content: str, metadata: Optional[dict] = None) -> str:
        """
        Generate a summary of the given content using a local LLM.
        
        Args:
            content: The text content to summarize
            metadata: Optional dictionary containing metadata (sender, subject, etc.)
            
        Returns:
            str: The generated summary
        """
        if not content.strip():
            return "[No content to summarize]"
            
        metadata = metadata or {}
        
        try:
            # Generate the summary using LLM
            summary_text = await self._generate_with_llm(content)
            
            # Prepare summary data
            summary_data = await self._store_summary(
                content=summary_text,
                metadata={
                    "task_id": metadata.get("task_id", "Unknown"),
                    "model": self.model,
                    **metadata
                }
            )
            
            logger.info(f"Generated summary for task: {metadata.get('task_id', 'Unknown')}")
            
            return summary_text
            
        except Exception as e:
            logger.error(f"Error in generate_summary: {str(e)}")
            # Return a basic fallback summary if anything goes wrong
            return f"Summary: {content}..."

    async def _generate_with_llm(self, content: str) -> str:
        """Generate a summary using the local LLM (Ollama)"""
        # Verify Ollama is available
        await self._check_ollama_available()
        
        try:
            # Using Ollama's API - adjust the model name as needed
            model_name = self.model # or "phi", "neural-chat"
            
            # Prepare the prompt for summarization
            prompt = f"""
            Please provide a concise summary of the following email content.
            Focus on key points, action items, and important details.
            Keep the summary under 100 words.
            
            Email Content:
            {content}
            
            Summary:
            """
            logger.info(f"Generating summary with LLM for prompt: {prompt}")
            # Call the local LLM (Ollama)
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt.strip(),
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Lower temperature for more focused summaries
                            "top_p": 0.9,
                            "max_tokens": 300
                        }
                    },
                    timeout=500
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Generated summary with LLM for prompt: {result}")
                    return result.get("response", "").strip()
                else:
                    error_msg = f"LLM API error: {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except Exception as e:
            error_msg = f"Error generating summary with LLM: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

# Create a global instance of the agent
agent = SummaryAgent()

# For backward compatibility
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup logic
    try:
        await agent.initialize()
        logger.info("Summary Agent started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SummaryAgent: {str(e)}")
        raise
    
    yield  # This is where the application runs
    
    # Shutdown logic
    if agent.httpx_client:
        await agent.httpx_client.aclose()
    logger.info("Summary Agent shutdown complete")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Create the FastAPI app with lifespan handler
    app = FastAPI(
        title="Summary Agent API",
        description="API for the Summary Agent that generates summaries of incoming requests",
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
    
    @app.post(
        "/summarize",
        response_model=SuccessResponse,
        responses={
            200: {"model": SuccessResponse},
            500: {"model": ErrorResponse}
        }
    )
    async def summarize_task(task: TaskModel = Body(...)):
        """
        Process a task and generate a summary of the provided content.

        - **task**: The task containing artifacts to be processed
        - **returns**: The summary of the provided content
        """
        try:
            if not agent.initialized:
                await agent.initialize()

            result = await agent._process_task(task)
            return {
                "success": True,
                "message": "Task processed successfully",
                "data": result
            }
            logger.info(f"Task processed successfully: {result}")
        
        except Exception as e:
            error_msg = f"Failed to process task: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": error_msg}
            )
    
    @app.get("/health", response_model=SuccessResponse)
    async def health_check():
        """Health check endpoint."""
        return {
            "success": True,
            "message": "Summary Agent is running",
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
    parser = argparse.ArgumentParser(description='Run the Summary Agent')
    parser.add_argument('--log_level', type=str, default=LOG_LEVEL,
                      choices=['debug', 'info', 'warning', 'error'],
                      help='Set the logging level')
    args = parser.parse_args()

        # Convert log level string to logging level
    log_level = getattr(logging, args.log_level.upper())
    
    
    # Set uvicorn log level to match
    uvicorn_log_level = args.log_level if args.log_level != 'debug' else LOG_LEVEL
    

    # Create the FastAPI app
    app = create_app()
    
    # Initialize the agent
    if not agent.initialize():
        logger.error("Failed to initialize SummaryAgent")
        return
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SUMMARY_AGENT_PORT,
        log_level=uvicorn_log_level
    )



# This allows the agent to be run directly with: python -m agents.summary_agent
if __name__ == "__main__":
    main()
