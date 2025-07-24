#!/usr/bin/env python3
"""
Test script to verify agent imports and basic functionality.
"""
import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path to allow absolute imports
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_agent_initialization():
    """Test that all agents can be imported and initialized."""
    # Import each agent class directly from its module
    from agents.email_processor_agent import EmailProcessorAgent
    from agents.response_agent import ResponseAgent
    from agents.summary_agent import SummaryAgent
    
    logger.info("Testing agent imports and initialization...")
    
    # Test EmailProcessorAgent
    try:
        email_agent = EmailProcessorAgent()
        if hasattr(email_agent, 'initialize'):
            await email_agent.initialize()
        logger.info("✅ EmailProcessorAgent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize EmailProcessorAgent: {e}", exc_info=True)
    
    # Test ResponseAgent
    try:
        response_agent = ResponseAgent()
        if hasattr(response_agent, 'initialize'):
            await response_agent.initialize()
        logger.info("✅ ResponseAgent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ResponseAgent: {e}", exc_info=True)
    
    # Test SummaryAgent
    try:
        summary_agent = SummaryAgent()
        if hasattr(summary_agent, 'initialize'):
            await summary_agent.initialize()
        logger.info("✅ SummaryAgent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SummaryAgent: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_agent_initialization())
