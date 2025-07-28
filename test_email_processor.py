import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from uuid import uuid4

import asyncclick as click
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    AgentCard,
    GetTaskRequest,
    Message,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskQueryParams,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    Part,
    Role,
)

# Load environment variables if .env file exists
if os.path.exists('.env'):
    load_dotenv()


# Email model to match the agent's expected input
class EmailModel(BaseModel):
    """Model representing an email message."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    subject: str
    sender: str
    recipients: List[str]
    body: str
    received_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)


logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
        format='%(asctime)s - %(name)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s',
        force=True)
logger = logging.getLogger(__name__)

# def create_test_email() -> EmailModel:
#     return EmailModel(
#         subject="Request for Quote - Kitchen Renovation",
#         sender="john.doe@example.com",
#         recipients=["sales@contractor.com"],
#         body="""Dear Sales Team,

# I'm interested in getting a quote for a kitchen renovation project. 
# The area is approximately 200 sq ft and we're looking to:
# - Install new cabinets
# - Replace countertops
# - Update the backsplash
# - Install new flooring

# Please let me know your availability for a consultation.

# Best regards,
# John Doe
# """,
#         metadata={
#             "priority": "normal",
#             "labels": ["quote-request"]
#         },
#         received_at=datetime.now(timezone.utc),
#         attachments=[]
#     )

def create_test_email() -> EmailModel:
    return EmailModel(
        subject="Project Update: Q3 Marketing Campaign",
        sender="vinay-123@example.com",
        recipients=["sales@contractor.com"],
        body="""Dear Team,

I hope this message finds you well. I wanted to provide an update on our Q3 Marketing Campaign progress.

1. Social Media: We've achieved 75% of our Q3 engagement goals across all platforms, with particularly strong performance on LinkedIn (up 23% from Q2).

2. Email Campaigns: Our open rates have improved to 28.5%, exceeding our target of 25%. Click-through rates are at 4.2%, slightly below our 5% goal.

3. Web Traffic: We've seen a 15% increase in organic traffic month-over-month, with the new blog content performing exceptionally well.

4. Upcoming Initiatives:
   - Launch of the new product demo video (scheduled for next week)
   - Webinar series on industry trends (registration now open)
   - Case study featuring our work with Client X (in final review)

5. Challenges:
   - Some delays in content approval process
   - Lower-than-expected conversion from social media traffic

Please review the attached detailed report and let me know if you have any questions or need additional information.

Best regards,
Sarah Johnson
Marketing Director""",
        metadata={
            "priority": "high",
            "category": "marketing"
        }
    )

def create_email_message(email: EmailModel) -> Message:
    """Create a properly formatted A2A message from an EmailModel.
    
    Args:
        email: The email model to convert to a Message
        
    Returns:
        A properly formatted Message object
    """

    
    # Create message parts
    parts = [
        Part(
            type="text/plain",
            text=f"""{email}""",
            metadata={
                "email_metadata": {
                    "subject": email.subject,
                    "from": email.sender,
                    "to": email.recipients,
                    "date": email.received_at.isoformat(),
                    "message_id": email.id
                }
            }
        )
    ]
    
    # Add attachments if any
    for idx, attachment in enumerate(email.attachments, 1):
        parts.append(
            Part(
                type=attachment.get("content_type", "application/octet-stream"),
                text=attachment.get("content", ""),
                metadata={
                    "filename": attachment.get("filename", f"attachment_{idx}"),
                    "size": len(attachment.get("content", "")),
                    "disposition": "attachment"
                }
            )
        )
    
    # Create and return the A2A message
    return Message(
        role="user",
        message_id=str(uuid4()),
        parts=parts,
        metadata={
            "email": {
                "id": email.id,
                "subject": email.subject,
                "received_at": email.received_at.isoformat(),
                **email.metadata
            }
        }
    )


async def process_email(client: A2AClient, email: EmailModel, streaming: bool = True, debug: bool = False) -> bool:
    """Process an email using the email processor agent."""
    try:
        # Create the A2A message
        message = create_email_message(email)
        logger.info(f"\nEmail message created: {message}")
        
        # Create send parameters
        send_params = MessageSendParams(
            message=message,
            configuration={
                'acceptedOutputModes': ['text'],
            },
        )
        logger.info(f"\nSend parameters created: {send_params}")
        
        task_id = str(uuid4())
        logger.info(f"\nTask ID: {task_id}")

 
        
        if streaming:
            # Process with streaming
            logger.info("\nProcessing with streaming...")
            request = SendStreamingMessageRequest(
                id=task_id,
                params=send_params
            )
            logger.info(f"\nSend request created: {request}")
            stream_response = client.send_message_streaming(request)
            async for chunk in stream_response:
                logger.info("response: " + chunk.model_dump_json(exclude_none=True, indent=2))
            return True
        else:
            logger.info("\nProcessing without streaming...")
            # Process without streaming
            request = SendMessageRequest(
                id=task_id,
                params=send_params
            )
            
            response = await client.send_message(request)
            task = response.root.result
            logger.info(f"\nTask completed: {task.status.state}")
            logger.info(f"Result: {task.model_dump_json(exclude_none=True, indent=2)}")
            
    except Exception as e:
        logger.error(f"\n Error processing email: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False
    

def create_send_params(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> MessageSendParams:
    """Helper function to create the payload for sending a task."""
    send_params: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [{'type': 'text', 'text': text}],
            'messageId': str(uuid4()),
        },
        'configuration': {
            'acceptedOutputModes': ['text'],
        },
    }

    if task_id:
        send_params['message']['taskId'] = task_id

    if context_id:
        send_params['message']['contextId'] = context_id

    return MessageSendParams(**send_params)


async def test_streaming_connection(client: A2AClient, message: Message) -> bool:
    """Test if the streaming connection works with the given message."""
    logger.info("Testing streaming connection...")
    
    try:
        # Create a streaming request
        request = SendStreamingMessageRequest(
            params=MessageSendParams(message=message)
        )
        
        # Try to open a streaming connection
        async with client.send_message_streaming(request) as stream:
            logger.info("Streaming connection established")
            
            # Read the first chunk to verify it's working
            try:
                chunk = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
                logger.info(f"Received first chunk: {chunk}")
                return True
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for first chunk")
                return False
                
    except A2AClientError as e:
        logger.error(f"A2A client error during streaming test: {e}")
        return False
    except Exception as e:
        logger.exception("Unexpected error during streaming test")
        return False


async def check_agent_endpoints(agent_url: str) -> bool:
    """Check if agent endpoints are accessible."""
    endpoints = [
        f"{agent_url}/.well-known/agent.json",
        f"{agent_url}/openapi.json",
        f"{agent_url}/docs"
    ]
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for endpoint in endpoints:
            try:
                logger.debug(f"Testing endpoint: {endpoint}")
                response = await client.get(endpoint)
                if response.status_code == 200:
                    logger.info(f" Successfully connected to {endpoint}")
                    return True
            except Exception as e:
                logger.warning(f" Could not connect to {endpoint}: {str(e)}")
    return False


@click.command()
@click.option('--agent', default='http://localhost:8001', help='Agent URL')
@click.option('--streaming/--no-streaming', default=True, help='Use streaming API')
@click.option('--debug', is_flag=True, help='Enable debug output')
async def main(agent: str, streaming: bool, debug: bool):
    """Test the Email Processor agent."""
    # Set up logging

    
    try:
        # Initialize HTTP client
        async with httpx.AsyncClient() as httpx_client:
            # Initialize A2A client directly
            client = A2AClient(httpx_client=httpx_client, url=agent)
            
            logger.info('\n' + '='*80)
            logger.info(f' Testing Email Processor Agent at {agent}')
            logger.info('='*80)
            
            # # Create test email
            email = create_test_email()
            logger.info(f"\nTest email created with ID: {email}")
            
            
            # Process the email
            success = await process_email(
                client=client,
                email=email,
                streaming=streaming,
                debug=debug
            )
            
            if success:

                logger.info('Test completed successfully!')

                return 0
            else:
                logger.error("failed")
                
                return 1
                
    except Exception as e:
        logger.error(f'Error: {e}', exc_info=debug)
        return 1


# Initialize logger at module level
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import asyncio
    import sys
    

    # Log startup
    logger.info("Starting test script...")
    logger.debug("Debug mode enabled")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)