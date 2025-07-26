"""
Response Agent for A2A MCP Contractor Automation.

This module handles generating responses to incoming requests.
"""
import sys,os
import socket
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import Task, Artifact
from fastapi import FastAPI, HTTPException, Body, status, Request, Response
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Import from consolidated config
from config import (
    LOG_LEVEL, 
    A2A_SERVER_HOST, A2A_SERVER_PORT, LOG_FILE,
    OLLAMA_API_BASE, OLLAMA_MODEL,
)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Parse command line arguments first
import argparse
from pathlib import Path

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
        logging.FileHandler(LOG_DIR / "response_agent.log", mode='a'),  # Append to log file
        logging.StreamHandler(sys.stdout)  # Log to console
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info(f"Response Agent logging initialized at level {args.log_level}")
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

class TaskModel(BaseModel):
    id: str
    artifacts: List[Dict[str, Any]]
    metadata: Optional[dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup logic
    try:
        await agent.initialize()
        logger.info("Response Agent started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ResponseAgent: {str(e)}")
        raise
    
    yield  # This is where the application runs
    
    # Shutdown logic
    if agent.httpx_client:
        await agent.httpx_client.aclose()
    logger.info("Response Agent shutdown complete")

# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Create the FastAPI app with lifespan handler
    app = FastAPI(
        title="Response Agent API",
        description="API for the Response Agent that generates responses to incoming requests",
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
            "message": "Response Agent is running",
            "data": {
                "status": "healthy",
                "version": "1.0.0"
            }
        }
    
    return app

class ResponseAgent:
    """Agent responsible for generating responses to emails based on their content and intent."""
    
    def __init__(self, base_url: str = None, model: str = None):
        """Initialize the ResponseAgent with required components."""
        self.httpx_client = None
        self.client = None
        self.initialized = False
        self.llm = None
        self.template_engine = None
        self.a2a_server_url = os.getenv("A2A_SERVER_URL", "http://a2a_server:8000")
        self.service_name = "response_agent"
        self.service_port = int(os.getenv("RESPONSE_AGENT_PORT", "8002"))
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
            
            # Register with A2A server using direct HTTP call
            service_url = f"http://{socket.gethostname()}:{self.service_port}"
            registration_data = {
                "name": self.service_name,
                "url": service_url,
                "type": "agent",
                "capabilities": ["response_generation"]  # Only response generation for this agent
            }
            
            # Set up agent card with service information
            self.agent_card = {
                "type": "agent",
                "name": "Response Agent",
                "description": "Generates responses to incoming requests",
                "version": "1.0.0",
                "url": service_url,
                "capabilities": ["response_generation"],
                "status": "ready"
            }
            
            # Make HTTP request to register the service
            try:
                response = await self.httpx_client.post(
                    f"{self.a2a_server_url}/services",
                    json=registration_data
                )
                response.raise_for_status()
                logger.info(f"Successfully registered with A2A server at {self.a2a_server_url}")
                
                self.initialized = True
                logger.info(f"Response Agent initialized successfully")
                return True
                
            except Exception as e:
                error_msg = f"Failed to register with A2A server: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to initialize ResponseAgent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.initialized = False
            raise Exception(error_msg) from e

    async def process_task_internal(self, task: TaskModel) -> dict:
        """
        Process a task that requires a response or summary.
        
        Args:
            task: The task containing artifacts to be processed
            
        Returns:
            dict: A generated response or summary for the provided content
            
        Raises:
            HTTPException: If there's an error processing the task
        """
        try:
            logger.info(f"Processing task: {task.id}")
            

            result = await self._process_task(task)
            return {
                "status": "success",
                "message": "Response generated successfully",
                "data": result,
                "is_summary": False
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
            dict: The generated response with metadata
        """
        try:
            # Extract email content and metadata from artifacts
            email_content = ""
            sender = "Unknown"
            subject = "No Subject"
            intent = "general"
            
            for artifact in task.artifacts:
                if isinstance(artifact, dict):  # Handle dict artifacts
                    if artifact.get("type") == "text/plain":
                        email_content = artifact.get("content", "")
                        sender = artifact.get("metadata", {}).get("sender", "Unknown")
                        subject = artifact.get("metadata", {}).get("subject", "No Subject")
                        intent = artifact.get("metadata", {}).get("intent", "general")
                        break
                elif hasattr(artifact, 'type') and artifact.type == "text/plain":  # Handle Pydantic models
                    email_content = getattr(artifact, 'content', '')
                    metadata = getattr(artifact, 'metadata', {})
                    sender = metadata.get("sender", "Unknown")
                    subject = metadata.get("subject", "No Subject")
                    intent = metadata.get("intent", "general")
                    break
            
            if not email_content:
                raise ValueError("No valid email content found in task artifacts")
            
            # Format email content for display
            formatted_content = f"From: {sender}\n"
            formatted_content += f"Subject: {subject}\n\n"
            formatted_content += email_content
            
            # Generate response based on intent
            response = await self.generate_response(formatted_content, intent)
            
            # Prepare response with metadata
            result = {
                "response": response,
                "intent": intent,
                "status": "completed",
                "metadata": {
                    "sender": sender,
                    "subject": subject,
                    "processed_at": datetime.now().isoformat(),
                    **task.metadata
                } if hasattr(task, 'metadata') else {
                    "sender": sender,
                    "subject": subject,
                    "processed_at": datetime.now().isoformat()
                }
            }
            
            logger.info(f"Generated response for intent: {intent}")
            logger.info(f"Generated response: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in _process_task: {str(e)}", exc_info=True)
            raise

    async def generate_response(self, email_content: str, intent: str) -> str:
        """
        Generate a response for the given email content and intent using templates and LLM.
        
        Args:
            email_content: The content of the email to respond to
            intent: The determined intent of the email
            
        Returns:
            str: The generated response
            
        Raises:
            Exception: If response generation fails
        """
        try:
            logger.info(f"Generating response for intent: {intent}")
            
            # Load quote template
            template_path = OUTPUT_DIR / "templates" / f"{intent}_template.txt"
            template_path.parent.mkdir(exist_ok=True, parents=True)
            
            if not template_path.exists():
                logger.warning(f"No template found for intent: {intent}, using default template")
                # Create default template if it doesn't exist
                default_template = """
    Dear {sender},

    Thank you for your email regarding {subject}.
    {custom_response}

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
            Customize the following template based on the email content and intent.
            
            Intent: {intent}
            Template: {template}
            Email Content: {email_content}
            
            Please generate a professional and helpful response.
            """

            logger.debug(f"Sending prompt to LLM for intent: {intent}")
            logger.info(f"Generating response with LLM for prompt: {prompt}")
            # Use httpx.AsyncClient for async HTTP requests
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/generate", # Ollama API endpoint
                    json={
                        "model": self.model,
                        "prompt": prompt.strip(),
                        "stream": False
                    },
                    timeout=500
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "Thank you for your email. We've received your message and will get back to you soon.")
                else:
                    error_msg = f"LLM API error: {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Fallback response if LLM fails
            fallback_responses = {
                "quote_request": "Thank you for your quote request. We'll get back to you with a detailed proposal soon.",
                "common_question": "Thank you for your inquiry. We'll provide you with the information you need shortly.",
                "general": "Thank you for your email. We've received your message and will get back to you soon."
            }
            return fallback_responses.get(intent, fallback_responses["general"])

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
            
            # Initialize A2A client with the configured server URL
            try:
                logger.info(f"Initializing A2A client with server at {self.a2a_server_url}")
                self.client = A2AClient(
                    httpx_client=self.httpx_client,
                    url=self.a2a_server_url
                )
                
                # Initialize card resolver for service discovery
                self.resolver = A2ACardResolver(
                    httpx_client=self.httpx_client,
                    base_url=self.a2a_server_url
                )
                
                # Register the agent card with A2A server
                self.agent_card = {
                    "type": "agent",
                    "name": "response_agent",
                    "description": "Generates responses to incoming emails based on their content and intent.",
                    "version": "1.0.0",
                    "capabilities": ["response_generation"],
                    "status": "ready"
                }
                
                # Register with A2A server using direct HTTP call
                service_url = f"http://{socket.gethostname()}:{self.service_port}"
                registration_data = {
                    "name": "response_agent",
                    "url": service_url,
                    "type": "agent",
                    "capabilities": ["response_generation"]
                }
                
                # Make HTTP request to register the service
                response = await self.httpx_client.post(
                    f"{self.a2a_server_url}/services",
                    json=registration_data
                )
                response.raise_for_status()
                
                logger.info(f"Successfully registered with A2A server at {self.a2a_server_url}")
                
            except Exception as e:
                error_msg = f"Failed to initialize A2A client: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
            
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
        "/process",
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Response Agent')
    parser.add_argument('--log-level', type=str, default='info',
                      choices=['debug', 'info', 'warning', 'error'],
                      help='Set the logging level')
    args = parser.parse_args()
    
    # Convert log level string to logging level
    log_level = getattr(logging, args.log_level.upper())
    
    
    # Set uvicorn log level to match
    uvicorn_log_level = args.log_level if args.log_level != 'debug' else 'debug'
    
    import uvicorn
    
    # Create the FastAPI app
    app = create_app()
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=RESPONSE_AGENT_PORT,
        log_level=uvicorn_log_level
    )


if __name__ == "__main__":
    main()
