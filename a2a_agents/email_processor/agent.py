import asyncio
import logging
import json
import os
import uuid
from datetime import datetime
from email.utils import parseaddr
from typing import Any, AsyncGenerator, Dict, List

import email
from email.policy import default

from pydantic import BaseModel, Field, field_validator

# Add parent directory to path to allow absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.llm_client import LLMClassifier

logger = logging.getLogger(__name__)

class EmailModel(BaseModel):
    """Model representing an email message."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    sender: str
    recipients: List[str]
    body: str
    received_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator('sender', mode='before')
    def validate_sender(cls, v):
        _, email = parseaddr(v)
        if not email or '@' not in email:
            raise ValueError('Invalid sender email format')
        return email

    @field_validator('recipients', mode='before')
    def validate_recipients(cls, v):
        if not isinstance(v, list):
            raise ValueError('Recipients must be a list')
        for recipient in v:
            _, email = parseaddr(recipient)
            if not email or '@' not in email:
                raise ValueError(f'Invalid recipient email format: {recipient}')
        return v

class EmailProcessorAgent:
    """An agent that processes incoming emails and classifies them."""

    def __init__(self):
        """Initialize the EmailProcessorAgent with configuration from environment."""
        self.initialized = False
        self.a2a_server_url = os.getenv('A2A_SERVER_URL', 'http://localhost:8000')
        self.service_port = int(os.getenv('EMAIL_PROCESSOR_PORT', '8001'))
        self.service_name = "email_processor"
        self.llm_classifier = LLMClassifier()
        logger.info(f"Initialized {self.service_name} agent")

    async def initialize(self):
        """Initialize the agent and its dependencies."""
        try:
            # Initialize LLM classifier
            await self.llm_classifier.client.__aenter__()
            self.initialized = True
            logger.info(f"{self.service_name} agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.service_name} agent: {e}")
            self.initialized = False
            raise

    async def _classify_email(self, email) -> Dict[str, Any]:
        """Classify the email using LLM.
        
        Args:
            email: The email to classify
            
        Returns:
            Dict containing classification results with category, priority, 
            requires_response flag, and summary
        """
        try:
            # Get the intent from LLM
            intent = await self.llm_classifier.classify_email(
                email_subject=email   ,
                email_body=email
            )
            
            # Map intent to priority and response requirements
            priority_map = {
                'quote_request': 'high',
                'common_question': 'normal',
                'summary_needed': 'low'
            }
            
            requires_response_map = {
                'quote_request': True,
                'common_question': True,
                'summary_needed': False
            }
            
            # Generate a summary (first 150 chars or full body if shorter)
            summary = email[:150] + '...' if len(email) > 150 else email
            
            return {
                'category': intent,
                'priority': priority_map.get(intent, 'normal'),
                'requires_response': requires_response_map.get(intent, True),
                'summary': summary,
                'full_body': email  # Include full body for reference
            }
            
        except Exception as e:
            logger.error(f"Error in email classification: {str(e)}")
            # Fallback to default classification
            return {
                'category': 'common_question',
                'priority': 'normal',
                'requires_response': True,
                'summary': email[:150] + '...' if len(email) > 150 else email,
                'full_body': email,
                'error': str(e)
            }

    async def process_email_stream(
        self, email_data: Dict[str, Any], task_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process an email and stream the results.
        
        Args:
            email_data: Dictionary containing email data
            task_id: The A2A task ID for tracking
            
        Yields:
            Dict containing processing updates and final results with the following structure:
            {
                'is_task_complete': bool,
                'task_state': str (TaskState enum value),
                'content': str (message content),
                'progress': int (0-100),
                'metadata': dict (optional additional data)
            }
        """
        try:

            logger.info(f"Processing email data: {task_id}")

            yield {
                'is_task_complete': False,
                'task_state': 'working',
                'content': f"Processing email from {email_data}",
                'progress': 10,
                'metadata': {}
            }
            
            # Classify the email
            classification = await self._classify_email(email_data)
            
            # Update progress
            yield {
                'is_task_complete': False,
                'task_state': 'working',
                'content': f"Email classified as '{classification['category']}' with priority '{classification['priority']}'",
                'progress': 50,
                'metadata': {'classification': classification}
            }
            
            # Here you would add more processing steps as needed
            # For example, extract entities, check against rules, etc.
            
            # Simulate additional processing
            await asyncio.sleep(0.5)  # Simulate processing time
            
            # Final result
            result = {
                'task_id': task_id,
                'classification': classification,
                'processed_at': datetime.utcnow().isoformat(),
                'actions': [{
                    'type': 'respond',
                    'priority': classification['priority'],
                    'summary': classification['summary']
                }]
            }
            
            yield {
                'is_task_complete': True,
                'task_state': 'completed',
                'final_message_text': 'Email processing completed',
                'result': result,
                'progress_percent': 100
            }

        except Exception as e:
            error_message = f'Error processing email: {str(e)}'
            logger.error(error_message, exc_info=True)
            yield {
                'is_task_complete': True,
                'error': error_message,
                'task_state': 'failed',
                'final_message_text': 'Failed to process email',
                'progress_percent': 100
            }

            # Email processing completed
    async def _parse_email_string(self, raw_email_string: str) -> Dict[str, Any]:
        """Helper to parse raw email string into a structured dictionary."""
        msg = email.message_from_string(raw_email_string, policy=default)

        sender = msg['sender']
        recipient = msg['recipients']
        subject = msg['subject']

        body_parts = []
        if msg.is_multipart():
            for part in msg.iter_parts():
                # Skip attachments and non-text parts for simplicity
                if part.get_content_maintype() == 'text':
                    body_parts.append(part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore'))
        else:
            if msg.get_content_maintype() == 'text':
                body_parts.append(msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore'))
        
        body = "\n\n".join(body_parts).strip()

        return {
            "sender": sender,
            "recipient": recipient,
            "subject": subject,
            "body": body
        }