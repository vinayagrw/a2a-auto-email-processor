"""
Main entry point for the A2A MCP Contractor Automation system.

This script allows you to start different agents from the command line.
"""
import argparse
import asyncio
import importlib
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Available agents
AGENTS = {
    "email": "agents.email_processor_agent.main",
    "response": "agents.response_agent.main",
    "summary": "agents.summary_agent.main"
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A2A MCP Contractor Automation")
    parser.add_argument(
        "agent",
        choices=AGENTS.keys(),
        help="The agent to start"
    )
    return parser.parse_args()

async def run_agent(agent_module_path):
    """Dynamically import and run the specified agent."""
    try:
        module_path, func_name = agent_module_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        
        if asyncio.iscoroutinefunction(func):
            await func()
        else:
            func()
    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        raise

def main():
    """Main entry point."""
    args = parse_arguments()
    agent_module = AGENTS[args.agent]
    
    try:
        print(f"Starting {args.agent} agent...")
        asyncio.run(run_agent(agent_module))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
