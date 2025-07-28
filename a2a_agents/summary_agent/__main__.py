"""
Summary Agent Service.

This module serves as the entry point for the Summary Agent service.
"""

import asyncio
import logging
import sys
from pathlib import Path
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import click
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

try:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill
    from a2a_agents.summary_agent.agent import SummaryAgent
    from a2a_agents.summary_agent.agent_executor import SummaryAgentExecutor
except ImportError as e:
    logging.error(f"Import error: {e}")
    raise

# Load environment variables from .env file
load_dotenv()

# Ensure logs directory exists
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / 'summary_agent.log', mode='a')],
        format='%(asctime)s - %(name)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s',
        force=True)
logger = logging.getLogger(__name__)



@click.command()
@click.option('--host', default=os.getenv("SUMMARY_AGENT_HOST", "localhost"), help='Host to bind the server to', show_default=True)
@click.option('--port', default=int(os.getenv("SUMMARY_AGENT_PORT", "8003")), help='Port to bind the server to', show_default=True)
@click.option('--debug', default=True, is_flag=True, help='Enable debug mode')
def main(host: str, port: int, debug: bool):
    """Run the Summary Agent.
    
    This agent generates contextually appropriate email summaries.
    It supports Server-Sent Events (SSE) for real-time progress updates.
    """
    try:
        # Define agent capabilities
        AGENT_CAPABILITIES = AgentCapabilities(streaming=True)

        # Define agent skills
        AGENT_SKILLS = [
            AgentSkill(
                id="summary_generation_skill",
                name="summary_generation",
                description="Generate contextually appropriate email summaries",
                examples=[
                    "Can you generate a summary for this email?",
                    "Please analyze this email and determine the next steps.",
                    "Classify this email and suggest appropriate actions."
                ],
                input_modes=["application/json"],
                output_modes=["application/json"],
                tags=["email", "classification", "automation", "processing"]
            )
        ]

        logger.info(f"Starting Summary Agent on {host}:{port}")
        
        request_handler = DefaultRequestHandler(
            task_store=InMemoryTaskStore(),
            agent_executor=SummaryAgentExecutor(),
        )
        
        # Create and configure the agent card
        agent_card = AgentCard(
            name="summary_agent",
            description="An agent that generates contextually appropriate email summaries.",
            url=f"http://{host}:{port}",
            version="1.0.0",
            default_input_modes=["application/json"],
            default_output_modes=["application/json"],
            capabilities=AGENT_CAPABILITIES,
            skills=AGENT_SKILLS,
            requires_authorization=False
        )
        
        # Create the FastAPI application with the agent card and request handler
        app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        # Get the actual underlying FastAPI/Starlette app
        fastapi_app = app.build()

        # Add CORSMiddleware to the actual FastAPI/Starlette app
        logger.info("Adding CORSMiddleware to the FastAPI application...")
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Adjust in production for security
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORSMiddleware added successfully.")        

        # Start the server
        logger.info(f"Summary Agent is running on http://{host}:{port}")
        uvicorn.run(
            fastapi_app,  # Access the underlying FastAPI app
            host=host,
            port=port,
            log_level="debug" if debug else "debug"
        )

    except ImportError as e:
        logger.error(
            f"Missing dependency: {e}. Please install all required packages."
        )
        exit(1)
    except Exception as e:
        logger.error(
            f"Failed to start server: {e}",
            exc_info=True,
        )
        exit(1)


if __name__ == '__main__':
    main()
