"""
Response Agent for A2A MCP Contractor Automation.

This module handles generating responses to incoming requests.
"""
import os
import sys
import json
import logging
import asyncio
import uvicorn
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import Task, Artifact
from fastapi import FastAPI, HTTPException, Body, status, Request, Response
import httpx
from pydantic import BaseModel, Field

# Import configuration
try:
    from .config import (
        A2A_SERVER_HOST,
        A2A_SERVER_PORT,
        RESPONSE_AGENT_PORT
    )
except ImportError:
    # Fall back to absolute import if running directly
    from config import (
        A2A_SERVER_HOST,
        A2A_SERVER_PORT,
        RESPONSE_AGENT_PORT
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("response_agent.log")
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

# Create FastAPI app
app = FastAPI(
    title="Response Agent API",
    description="API for generating responses to emails",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup routes
@app.post(
    "/process_task",
    response_model=SuccessResponse,
    responses={
        200: {"model": SuccessResponse, "description": "Response generated successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate a response for a task",
    description="Processes a task that requires generating a response to the provided content"
)
async def process_task_route(task: TaskModel = Body(...)):
    """Process a task that requires a response"""
    return await agent.process_task_internal(task)

class ResponseAgent:
    """Agent responsible for generating responses to emails based on their content and intent."""
    
    def __init__(self):
        """Initialize the ResponseAgent with required components."""
        self.httpx_client = None
        self.resolver = None
        self.agent_card = None
        self.client = None
    
    async def process_task_internal(self, task: TaskModel) -> dict:
        """
        Process a task that requires a response.
        
        Args:
            task: The task containing artifacts to be processed
            
        Returns:
            dict: A generated response for the provided content
            
        Raises:
            HTTPException: If there's an error processing the task
        """
        try:
            logger.info(f"Processing task: {task.description}")
            result = await self._process_task(task)
            return {
                "status": "success",
                "message": "Response generated successfully",
                "data": result
            }
            
        except HTTPException:
            raise
            
        except Exception as e:
            error_msg = f"Error processing task: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
    
    async def _process_task(self, task: TaskModel) -> dict:
        """
        Process the task and generate a response.
        
        Args:
            task: The task to process
            
        Returns:
            dict: The generated response
        """
        try:
            # Extract email content from artifacts
            email_content = ""
            intent = "general"
            
            for artifact in task.artifacts:
                if artifact.type == "email":
                    try:
                        email_data = json.loads(artifact.content)
                        email_content = f"From: {email_data.get('sender', 'Unknown')}\n"
                        email_content += f"Subject: {email_data.get('subject', 'No Subject')}\n\n"
                        email_content += email_data.get('body', '')
                        
                        # Get intent from metadata if available
                        intent = artifact.metadata.get('intent', 'general')
                        break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse email artifact: {str(e)}")
            
            if not email_content:
                raise ValueError("No valid email content found in task artifacts")
            
            # Generate response based on content and intent
            response = await self.generate_response(email_content, intent)
            
            return {
                "response": response,
                "intent": intent,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error in _process_task: {str(e)}", exc_info=True)
            raise

    async def generate_response(self, email_content: str, intent: str) -> str:
        """
        Generate a response for the given email content and intent.
        
        Args:
            email_content: The content of the email to respond to
            intent: The determined intent of the email
            
        Returns:
            str: The generated response
            
        Note:
            This is a simplified implementation. In a real application, you would typically use
            a language model or template-based system to generate more sophisticated responses.
        """
        try:
            # Simple template-based response generation
            response_templates = {
                "greeting": "Thank you for your email. How can I assist you today?",
                "question": "Thank you for your inquiry. Here's the information you requested:",
                "complaint": "We're sorry to hear about your experience. Let us help resolve this issue for you.",
                "general": "Thank you for your email. We've received your message and will get back to you soon."
            }
            
            # Select template based on intent
            if intent in response_templates:
                return response_templates[intent]
                
            # Default response
            return response_templates["general"]
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
            return "Thank you for your email. We've received your message and will get back to you soon."

    async def generate_response(self, email_content: str, intent: str) -> str:
        # Load quote template
        template_path = OUTPUT_DIR / "templates" / "quote_template.txt"
        template_path.parent.mkdir(exist_ok=True, parents=True)
        
        if not template_path.exists():
            # Create default template if it doesn't exist
            default_template = """
Dear {sender},

Thank you for your interest in our services. Based on your request regarding {subject},
we would be happy to provide you with a detailed quote.

[Customized response based on email content]

Best regards,
Your Contractor Team
"""
            with open(template_path, "w") as f:
                f.write(default_template)

        # Read template
        with open(template_path, "r") as f:
            template = f.read()

        # Generate response using local LLM
        prompt = f"""
        Customize the following template based on the email content:
        
        Template: {template}
        
        Email Content: {email_content}
        """

        response = requests.post(
            LLM_URL,
            json={
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            return response.json()["response"]
        raise Exception("Failed to generate response")

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
            
            # Initialize A2A client
            self.client = A2AClient(
                httpx_client=self.httpx_client,
                url=f"http://{A2A_SERVER_HOST}:{A2A_SERVER_PORT}"
            )
            
            # Initialize card resolver
            self.resolver = A2ACardResolver()
            
            # Register the agent card
            self.agent_card = {
                "type": "agent",
                "name": "response_agent",
                "description": "Generates responses to incoming emails based on their content and intent.",
                "version": "1.0.0"
            }
            
            self.initialized = True
            logger.info("ResponseAgent initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize ResponseAgent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)

def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Response Agent API",
        description="API for generating responses to emails",
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
    agent = ResponseAgent()
    
    # Add routes
    @app.post(
        "/process_task",
        response_model=SuccessResponse,
        responses={
            200: {"model": SuccessResponse, "description": "Task processed successfully"},
            400: {"model": ErrorResponse, "description": "Invalid input"},
            500: {"model": ErrorResponse, "description": "Internal server error"}
        },
        summary="Process a task",
        description="Processes a task that requires generating a response"
    )
    async def process_task_route(task: TaskModel = Body(...)):
        """Process a task that requires a response."""
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
            message="Response Agent is running",
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
        port=RESPONSE_AGENT_PORT,
        log_level="info"
    )

if __name__ == "__main__":
    main()
