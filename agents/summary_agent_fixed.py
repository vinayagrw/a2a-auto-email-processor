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

# Ensure the persist directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

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
            
            # Initialize ChromaDB client with new persistent client
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="summaries",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize card resolver
            self.resolver = A2ACardResolver(
                httpx_client=self.httpx_client,
                base_url=f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}"
            )
            
            # Register the agent card
            self.agent_card = {
                "type": "agent",
                "name": "Summary Agent",
                "description": "Generates summaries of incoming requests",
                "version": "1.0.0",
                "url": f"http://localhost:{SUMMARY_AGENT_PORT}",
                "capabilities": ["summarization"],
                "status": "ready"
            }
            
            self.initialized = True
            logger.info("Summary Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SummaryAgent: {str(e)}")
            self.initialized = False
            raise

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
            
    def _get_daily_log_path(self) -> Path:
        """Get the path for today's summary log file"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_dir = Path("logs/summaries")
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"daily_summary_{today}.txt"

    def _log_summary_to_file(self, summary: str, metadata: dict):
        """Log the summary to today's log file with metadata"""
        try:
            log_path = self._get_daily_log_path()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = f"\n\n=== Summary - {timestamp} ===\n"
            log_entry += f"From: {metadata.get('sender', 'Unknown')}\n"
            log_entry += f"Subject: {metadata.get('subject', 'No Subject')}\n"
            log_entry += f"Summary: {summary}\n"
            log_entry += "=" * 50
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
                
            logger.info(f"Summary logged to {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to write summary to log file: {str(e)}")
            # Don't fail the operation if logging fails

    async def _check_ollama_available(self) -> None:
        """Verify Ollama service is available"""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama API returned status {response.status_code}")
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            raise RuntimeError("Failed to connect to Ollama service. Please ensure it's running.") from e

    async def _generate_with_llm(self, content: str) -> str:
        """Generate a summary using the local LLM (Ollama)"""
        # Verify Ollama is available
        await self._check_ollama_available()
        
        try:
            # Using Ollama's API - adjust the model name as needed
            model_name = "mistral"  # or "phi", "neural-chat"
            
            # Prepare the prompt for summarization
            prompt = f"""
            Please provide a concise summary of the following email content.
            Focus on key points, action items, and important details.
            Keep the summary under 100 words.
            
            Email Content:
            {content}
            
            Summary:
            """
            
            # Call the local LLM (Ollama)
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
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
                    return result.get("response", "").strip()
                else:
                    error_msg = f"LLM API error: {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except Exception as e:
            error_msg = f"Error generating summary with LLM: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

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
            summary = await self._generate_with_llm(content)
            
            # Log the summary to the daily log file
            self._log_summary_to_file(summary, metadata)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in generate_summary: {str(e)}")
            # Return a basic fallback summary if anything goes wrong
            return f"Summary: {content[:150]}..."

# Create a global instance of the agent
agent = SummaryAgent()

# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Summary Agent API",
        description="API for the Summary Agent that generates summaries of incoming requests",
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
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the agent when the application starts."""
        try:
            await agent.initialize()
            logger.info("Summary Agent started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SummaryAgent: {str(e)}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources when the application shuts down."""
        if agent.httpx_client:
            await agent.httpx_client.aclose()
        logger.info("Summary Agent shutdown complete")
    
    @app.post(
        "/process_task",
        response_model=SuccessResponse,
        responses={
            200: {"model": SuccessResponse},
            500: {"model": ErrorResponse}
        }
    )
    async def process_task(task: TaskModel = Body(...)):
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
    app = create_app()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=SUMMARY_AGENT_PORT,
        log_level="info",
        reload=False
    )
    
    server = uvicorn.Server(config)
    
    try:
        logger.info(f"Starting Summary Agent on http://0.0.0.0:{SUMMARY_AGENT_PORT}")
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Shutting down Summary Agent...")
    except Exception as e:
        logger.error(f"Error running Summary Agent: {str(e)}")
        sys.exit(1)

# This allows the agent to be run directly with: python -m agents.summary_agent
if __name__ == "__main__":
    main()
