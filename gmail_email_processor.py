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

# Gmail API configuration
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]
TOKEN_FILE = Path('token.json')
CREDENTIALS_FILE = Path('credentials.json')


class GmailServiceError(Exception):
    """Custom exception for Gmail service errors."""
    pass


def get_gmail_service():
    """Initialize and return an authorized Gmail API service instance.
    
    Returns:
        A Gmail API service instance.
        
    Raises:
        GmailServiceError: If there's an error initializing the Gmail service.
    """
    creds = None
    
    # The file token.json stores the user's access and refresh tokens
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            logger.info("Loaded credentials from token file")
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            creds = None
    
    # If there are no (valid) credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Refreshing expired credentials")
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Error refreshing token: {e}")
                creds = None
        
        if not creds:
            try:
                if not CREDENTIALS_FILE.exists():
                    raise FileNotFoundError(
                        f"Credentials file not found at {CREDENTIALS_FILE}. "
                        "Please download it from Google Cloud Console."
                    )
                
                logger.info("Initiating OAuth flow for Gmail API")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_FILE), SCOPES)
                creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(TOKEN_FILE, 'w') as token:
                    token.write(creds.to_json())
                logger.info("Successfully obtained and saved new credentials")
                
            except Exception as e:
                error_msg = f"Failed to obtain Gmail API credentials: {e}"
                logger.error(error_msg)
                raise GmailServiceError(error_msg) from e
    
    try:
        service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
        logger.info("Successfully initialized Gmail API service")
        return service
    except Exception as e:
        error_msg = f"Failed to create Gmail service: {e}"
        logger.error(error_msg)
        raise GmailServiceError(error_msg) from e

def decode_text(text: str) -> str:
    """Decode email headers that might be encoded in base64 or quoted-printable.
    
    Args:
        text: The text to decode (could be subject, sender, or other header)
        
    Returns:
        Decoded text as a string
    """
    if not text:
        return ""
    
    try:
        # Handle None or non-string input
        if not isinstance(text, str):
            text = str(text)
            
        # Try to decode the text if it's encoded
        decoded_parts = []
        for part, encoding in decode_header(text):
            if part is None:
                continue
                
            if isinstance(part, bytes):
                # Try the specified encoding first, then fallback to common encodings
                encodings_to_try = []
                if encoding:
                    encodings_to_try.append(encoding.lower())
                
                # Add common email encodings to try
                encodings_to_try.extend(['utf-8', 'iso-8859-1', 'us-ascii'])
                
                decoded = None
                for enc in encodings_to_try:
                    try:
                        decoded = part.decode(enc, errors='strict')
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                
                # If all decodings failed, use replace to handle errors
                if decoded is None:
                    decoded = part.decode('utf-8', errors='replace')
                
                decoded_parts.append(decoded)
            else:
                decoded_parts.append(str(part))
        
        # Join all parts and clean up any extra whitespace
        result = ' '.join(part.strip() for part in decoded_parts if part)
        return re.sub(r'\s+', ' ', result).strip()
        
    except Exception as e:
        logger.warning(f"Error decoding text '{text}': {e}")
        # Return the original text if we can't decode it
        return str(text) if text else ""

def get_email_body(message: Dict[str, Any]) -> str:
    """Extract and decode the email body from a Gmail message.
    
    Args:
        message: The Gmail API message object
        
    Returns:
        The decoded email body as a string
    """
    try:
        # Get the raw email data
        if 'raw' in message:
            msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
            mime_msg = email.message_from_bytes(msg_str)
        else:
            mime_msg = message
        
        # Check if this is a multipart message
        if not mime_msg.is_multipart():
            # Not multipart, just get the payload
            payload = mime_msg.get_payload(decode=True)
            if payload:
                charset = mime_msg.get_content_charset() or 'utf-8'
                return payload.decode(charset, errors='replace')
            return ""
        
        # For multipart messages, collect all text/plain parts
        text_parts = []
        
        for part in mime_msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip any attached files
            if "attachment" in content_disposition:
                continue
                
            # Get text/plain parts
            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        decoded = payload.decode(charset, errors='replace')
                        text_parts.append(decoded)
                except Exception as e:
                    logger.warning(f"Error decoding part: {e}")
            # Also check for HTML parts if no plain text was found
            elif content_type == "text/html" and not text_parts:
                try:
                    import html2text
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        html_content = payload.decode(charset, errors='replace')
                        # Convert HTML to plain text
                        text_content = html2text.html2text(html_content)
                        text_parts.append(text_content)
                except ImportError:
                    logger.warning("html2text not installed, falling back to raw HTML")
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            text_parts.append(payload.decode(charset, errors='replace'))
                    except Exception as e:
                        logger.warning(f"Error decoding HTML part: {e}")
        
        # Join all text parts with double newlines
        return "\n\n".join(part.strip() for part in text_parts if part.strip())
        
    except Exception as e:
        logger.error(f"Error getting email body: {e}", exc_info=True)
        return "Error extracting email content"

def get_attachments(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract attachments from Gmail message."""
    attachments = []
    if 'parts' not in message['payload']:
        return attachments
        
    for part in message['payload']['parts']:
        if part.get('filename') and part.get('body', {}).get('attachmentId'):
            attachment = {
                'filename': part['filename'],
                'mimeType': part.get('mimeType', 'application/octet-stream'),
                'size': part.get('body', {}).get('size', 0),
                'attachment_id': part['body']['attachmentId']
            }
            attachments.append(attachment)
    return attachments


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