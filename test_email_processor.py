"""
Test script for the Email Processor Agent.
"""
import sys
import os
import logging
from pathlib import Path

# Print current working directory and Python path
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
print(f"Project root: {project_root}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# List the contents of the agents directory
agents_dir = os.path.join(project_root, 'agents')
print(f"Agents directory contents: {os.listdir(agents_dir) if os.path.exists(agents_dir) else 'Directory not found'}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import the module
try:
    from agents.email_processor_agent import EmailProcessorAgent
    logger.info("Successfully imported EmailProcessorAgent")
except Exception as e:
    logger.error(f"Failed to import EmailProcessorAgent: {e}", exc_info=True)
    sys.exit(1)

def main():
    """Main function to run the test."""
    logger.info("Test script started")
    
    # Print the module's location
    try:
        import agents.email_processor_agent
        logger.info(f"EmailProcessorAgent module location: {agents.email_processor_agent.__file__}")
    except Exception as e:
        logger.error(f"Error getting module location: {e}")
    
    logger.info("Test script completed")

if __name__ == "__main__":
    main()
