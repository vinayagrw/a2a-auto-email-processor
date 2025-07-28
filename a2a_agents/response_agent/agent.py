import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException, Body, status, Request, Response
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import from consolidated config
from config import (
    LOG_LEVEL, 
    A2A_SERVER_HOST, A2A_SERVER_PORT, LOG_FILE,
    OLLAMA_API_BASE, OLLAMA_MODEL,
    RESPONSE_AGENT_PORT,
    A2A_SERVER_URL,
    OUTPUT_DIR,
    BASE_DIR,
    DATA_DIR,
    LOGS_DIR
)


logger = logging.getLogger(__name__)

class ResponseRequest(BaseModel):
    """Request model for generating an email response."""
    email_id: str
    sender: str
    recipients: List[str]
    subject: str
    body: str
    classification: Dict[str, str]
    tone: str = "professional"

class ResponseAgent:
    """Agent responsible for generating email responses."""

    def __init__(self):
        """Initialize the ResponseAgent."""
        self.initialized = False
        self.a2a_server_url = os.getenv("A2A_SERVER_URL", "http://a2a_server:8000")
        self.service_port = int(os.getenv("RESPONSE_AGENT_PORT", "8002"))
        self.service_name = "response_agent"
        logger.info(f"Initialized {self.service_name} agent")
        self.base_url = OLLAMA_API_BASE
        self.model = OLLAMA_MODEL

    async def initialize(self):
        """Initialize the agent and its dependencies."""
        try:
            self.initialized = True
            logger.info(f"{self.service_name} agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.service_name} agent: {e}")
            self.initialized = False
            raise


    async def generate_response(self, email_content: str, task_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a response for the given email content using templates and LLM.
        
        Args:
            email_content: The content of the email to respond to
            task_id: The A2A task ID for tracking
            
        Returns:
            str: The generated response 
            
        Raises:
            Exception: If response generation fails
        """
        try:
            logger.info(f"Generating response for email content: {email_content}")
            logger.info(f"Task ID: {task_id}")
            # Load quote template
            template_path = OUTPUT_DIR / "templates" / "response_template.txt"
            template_path.parent.mkdir(exist_ok=True, parents=True)
            
            if not template_path.exists():
                logger.warning(f"No template found for response, using default template")
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
            
            Template: {template}
            Email Content: {email_content}
            
            Please generate a professional and helpful response.
            """

            logger.debug(f"Sending prompt to LLM for response generation: {prompt}")

            # Update progress
            yield {
                'is_task_complete': False,
                'task_state': 'working',
                'content': f"Generating response with LLM for prompt: {prompt}",
                'progress': 50,
                'metadata': {}
            }
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
                    logger.info(f"Response generated with LLM for prompt: {response.json()}")
                    # Update progress
                    yield {
                        'is_task_complete': True,
                        'task_state': 'completed',
                        'content': f"Response generated with LLM for prompt: {response.json()}",
                        'progress': 100,
                        'metadata': {}
                    }

                else:
                    error_msg = f"LLM API error: {response.text}"
                    logger.error(error_msg)
                    yield {
                        'is_task_complete': True,
                        'task_state': 'completed',
                        'content': f"Response generated with LLM for prompt: {error_msg}",
                        'progress': 100,
                        'metadata': {}
                    }

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Fallback response if LLM fails
            fallback_response = "Thank you for your email. We've received your message and will get back to you soon."
            yield {
                        'is_task_complete': True,
                        'task_state': 'completed',
                        'content': f"Fallback response generated with LLM for prompt: {fallback_response}",
                        'progress': 100,
                        'metadata': {}
                    }
   