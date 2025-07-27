import asyncio
import logging
import sys
from pathlib import Path

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
    from a2a_agents.email_processor.agent import EmailProcessorAgent
    from a2a_agents.email_processor.agent_executor import EmailProcessorAgentExecutor
except ImportError as e:
    logging.error(f"Import error: {e}")
    raise

# Load environment variables from .env file
load_dotenv()

# Ensure logs directory exists
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'email_processor_agent.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)



@click.command()
@click.option('--host', default='localhost', help='Host to bind the server to', show_default=True)
@click.option('--port', default=8001, help='Port to bind the server to', show_default=True)
@click.option('--debug', default=True, is_flag=True, help='Enable debug mode')
def main(host: str, port: int, debug: bool):
    """Run the Email Processor Agent.
    
    This agent processes incoming emails, classifies them, and takes appropriate actions.
    It supports Server-Sent Events (SSE) for real-time progress updates.
    """
    try:
        # Define agent capabilities
        AGENT_CAPABILITIES = AgentCapabilities(streaming=True)

        # Define agent skills
        AGENT_SKILLS = [
            AgentSkill(
                id="email_processing_skill",
                name="email_processing",
                description="Process and classify incoming emails for appropriate handling",
                examples=[
                    "Can you process this email and classify its intent?",
                    "Please analyze this email and determine the next steps.",
                    "Classify this email and suggest appropriate actions."
                ],
                input_modes=["application/json"],
                output_modes=["application/json"],
                tags=["email", "classification", "automation", "processing"]
            )
        ]
        # Set log level based on debug flag
        logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
        logger.info(f"Starting Email Processor Agent on {host}:{port}")
        
        request_handler = DefaultRequestHandler(
            task_store=InMemoryTaskStore(),
            agent_executor=EmailProcessorAgentExecutor(),
        )
        
        # Create and configure the agent card
        agent_card = AgentCard(
            name="email_processor_agent",
            description="An agent that processes incoming emails, classifies them, and takes appropriate actions.",
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
        logger.info(f"Email Processor Agent is running on http://{host}:{port}")
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
