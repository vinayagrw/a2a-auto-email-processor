"""
Email Processor Agent for A2A MCP Contractor Automation.

This agent processes incoming emails, classifies them, and takes appropriate actions.
It supports Server-Sent Events (SSE) for real-time progress updates.
"""

from .agent import EmailProcessorAgent
from .agent_executor import EmailProcessorAgentExecutor

__all__ = ["EmailProcessorAgent", "EmailProcessorAgentExecutor"]