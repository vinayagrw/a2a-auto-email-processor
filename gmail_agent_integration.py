import os
import base64
import json
import email
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.message import MIMEMessage
from email.header import Header
from email.utils import formataddr, parseaddr, formatdate
import re
import uuid
from datetime import datetime, timezone, timedelta
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Literal, TypedDict, List as ListType
from pathlib import Path
import asyncio
import logging
import httpx
from pprint import pprint
from uuid import UUID, uuid4

# Google API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
        format='%(asctime)s - %(name)s - %(pathname)s - %(lineno)d - %(levelname)s - %(message)s',
        force=True)
logger = logging.getLogger(__name__)


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


# Load environment variables
from dotenv import load_dotenv
load_dotenv()



# Gmail API configuration
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]
TOKEN_FILE = Path('token.json')
CREDENTIALS_FILE = Path('credentials.json')

# Email processor configuration
EMAIL_PROCESSOR_URL = os.getenv('EMAIL_PROCESSOR_URL', 'http://localhost:8001')
PROCESSING_INTERVAL = int(os.getenv('PROCESSING_INTERVAL', '300'))  # 5 minutes
A2A_AGENT_URL = os.getenv('A2A_AGENT_URL', 'http://localhost:8001')

# Models
class EmailModel(TypedDict):
    """Model representing an email message."""
    subject: str
    sender: str
    recipients: List[str]
    body: str
    received_at: datetime
    metadata: Dict[str, Any]
    attachments: List[Dict[str, Any]]
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

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
        logger.debug(f"Processing message with keys: {list(message.keys())}")
        
        # Try to get the snippet first as it's usually available
        snippet = message.get('snippet', '').strip()
        
        # Check if we have a payload
        payload = message.get('payload', {})
        if not isinstance(payload, dict):
            logger.warning(f"Payload is not a dictionary: {type(payload)}")
            return snippet or "No content available"
            
        # Try to get the body data directly
        if 'body' in payload and isinstance(payload['body'], dict):
            body = payload['body']
            if 'data' in body and body['data']:
                try:
                    decoded = base64.urlsafe_b64decode(body['data'] + '=' * (-len(body['data']) % 4)).decode('utf-8')
                    if decoded.strip():
                        return decoded
                except Exception as e:
                    logger.warning(f"Error decoding body data: {e}")
        
        # Check for parts in the payload
        if 'parts' in payload and isinstance(payload['parts'], list):
            # First pass: look for text/plain parts
            for part in payload['parts']:
                if not isinstance(part, dict):
                    continue
                    
                mime_type = str(part.get('mimeType', '')).lower()
                body = part.get('body', {})
                
                if mime_type == 'text/plain' and isinstance(body, dict) and 'data' in body:
                    try:
                        decoded = base64.urlsafe_b64decode(body['data'] + '=' * (-len(body['data']) % 4)).decode('utf-8')
                        if decoded.strip():
                            return decoded
                    except Exception as e:
                        logger.warning(f"Error decoding text/plain part: {e}")
            
            # Second pass: look for any text/* part
            for part in payload['parts']:
                if not isinstance(part, dict):
                    continue
                    
                mime_type = str(part.get('mimeType', '')).lower()
                body = part.get('body', {})
                
                if mime_type.startswith('text/') and isinstance(body, dict) and 'data' in body:
                    try:
                        decoded = base64.urlsafe_b64decode(body['data'] + '=' * (-len(body['data']) % 4)).decode('utf-8')
                        if decoded.strip():
                            return decoded
                    except Exception as e:
                        logger.warning(f"Error decoding {mime_type} part: {e}")
        
        # If we have a raw message, try to parse it
        if 'raw' in message and message['raw']:
            try:
                msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII') + b'=' * (-len(message['raw']) % 4))
                mime_msg = email.message_from_bytes(msg_str)
                
                if hasattr(mime_msg, 'is_multipart') and callable(mime_msg.is_multipart):
                    if mime_msg.is_multipart():
                        for part in mime_msg.walk():
                            content_type = part.get_content_type()
                            if content_type == 'text/plain':
                                payload = part.get_payload(decode=True)
                                if payload:
                                    charset = part.get_content_charset() or 'utf-8'
                                    try:
                                        decoded = payload.decode(charset, errors='replace')
                                        if decoded.strip():
                                            return decoded
                                    except Exception as e:
                                        logger.warning(f"Error decoding part: {e}")
                    else:
                        payload = mime_msg.get_payload(decode=True)
                        if payload:
                            charset = mime_msg.get_content_charset() or 'utf-8'
                            try:
                                return payload.decode(charset, errors='replace')
                            except Exception as e:
                                logger.warning(f"Error decoding payload: {e}")
            except Exception as e:
                logger.warning(f"Error processing raw message: {e}")
        
        return snippet or "No content available"
        
    except Exception as e:
        logger.error(f"Error in get_email_body: {e}", exc_info=True)
        return f"Error extracting email content: {str(e)}"

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

def process_gmail_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Gmail API message to our EmailModel format."""
    try:
        # Extract headers
        headers = {}
        for header in ['From', 'To', 'Subject', 'Date', 'Message-ID']:
            headers[header] = next((h['value'] for h in message['payload']['headers'] 
                                 if h['name'] == header), '')
        
        # Process sender and recipients
        sender = headers.get('From', '')
        to_recipients = [r.strip() for r in headers.get('To', '').split(',') if r.strip()]
        
        # Process date
        date_str = headers.get('Date', '')
        received_at = parsedate_to_datetime(date_str) if date_str else datetime.datetime.now(datetime.timezone.utc)
        
        # Get email body and attachments
        body = get_email_body(message)
        attachments = get_attachments(message)
        
        # Create email data
        email_data = {
            'id': message['id'],
            'subject': decode_text(headers.get('Subject', '(No subject)')),
            'sender': decode_text(sender),
            'recipients': [decode_text(r) for r in to_recipients],
            'body': body,
            'received_at': received_at.isoformat(),
            'metadata': {
                'message_id': headers.get('Message-ID', ''),
                'labels': message.get('labelIds', []),
            },
            'attachments': attachments
        }
        
        return email_data
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise

def create_email_message(email: EmailModel) -> Dict[str, Any]:
    """Create an A2A message from an email model.
    
    Args:
        email: The email model to convert
        
    Returns:
        A formatted A2A message
    """
    # Format the email content
    email_content = f"""Subject: {email['subject']}
From: {email['sender']}
To: {', '.join(email['recipients'])}
Date: {email['received_at']}

{email['body']}"""
    
    # Create the message parts
    parts = [
        {
            'type': 'text',
            'text': email_content,
            'metadata': {
                'email': {
                    'id': email.get('id', str(uuid.uuid4())),
                    'subject': email['subject'],
                    'received_at': email['received_at'],
                    **email.get('metadata', {})
                }
            }
        }
    ]
    
    # Add attachments if any
    for attachment in email.get('attachments', []):
        parts.append({
            'type': 'attachment',
            'data': attachment.get('data', ''),
            'filename': attachment.get('filename', 'attachment'),
            'mimeType': attachment.get('mimeType', 'application/octet-stream')
        })
    
    # Create the message
    message = {
        'role': 'user',
        'parts': parts,
        'messageId': str(uuid.uuid4())
    }
    
    return message

async def get_latest_emails(service, max_results: int = 5) -> List[Dict[str, Any]]:
    """Get the latest emails from Gmail and process them."""
    try:
        # Call the Gmail API to get only unread emails
        results = service.users().messages().list(
            userId='me',
            labelIds=['INBOX', 'UNREAD'],
            maxResults=max_results,
            q='is:unread'  # Only get unread emails
        ).execute()
        
        messages = results.get('messages', [])
        processed_emails = []
        
        for msg in messages:
            try:
                # Get full message
                message = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='full'
                ).execute()
                
                # Process the message into our format
                email_data = process_gmail_message(message)
                processed_emails.append(email_data)
                
            except Exception as e:
                logger.error(f"Error processing message {msg.get('id')}: {str(e)}")
                continue
                
        return processed_emails
        
    except Exception as e:
        logger.error(f"Error fetching emails: {str(e)}")
        return []

async def send_reply(original_email: EmailModel, response_text: str) -> None:
    """Send a reply to the original email.
    
    Args:
        original_email: The original email to reply to
        response_text: The response text to send
    """
    try:
        service = get_gmail_service()
        
        # Create the email message
        msg = MIMEMultipart()
        msg['To'] = original_email['sender']
        msg['From'] = os.getenv('GMAIL_SENDER_EMAIL')
        msg['Subject'] = f"Re: {original_email['subject']}"
        msg['In-Reply-To'] = original_email.get('message_id', '')
        msg['References'] = original_email.get('message_id', '')
        msg['Date'] = formatdate(localtime=True)
        
        # Add the response text
        msg.attach(MIMEText(response_text, 'plain', 'utf-8'))
        
        # Encode the message
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
        
        # Send the message
        message = service.users().messages().send(
            userId='me',
            body={
                'raw': raw_message,
                'threadId': original_email.get('thread_id')
            }
        ).execute()
        
        logger.info(f"Reply sent successfully. Message ID: {message['id']}")
        
    except Exception as e:
        logger.error(f"Error sending reply: {str(e)}")
        raise

async def process_email(client: httpx.AsyncClient, email: EmailModel, streaming: bool = True, debug: bool = False) -> bool:
    """Process an email using the email processor agent.
    
    Args:
        client: The A2A client instance
        email: The email to process
        streaming: Whether to use streaming mode
        debug: Enable debug logging
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Create the A2A message
        message = create_email_message(email)
        if debug:
            logger.info(f"\nEmail message created: {json.dumps(message, indent=2, default=str)}")
        
        # Create send parameters
        send_params = {
            'message': message,
            'configuration': {
                'acceptedOutputModes': ['text'],
            },
        }
        
        if debug:
            logger.info(f"\nSend parameters created: {send_params}")
        
        task_id = str(uuid.uuid4())
        logger.info(f"\nProcessing email with task ID: {task_id}")
        if streaming:
            # Process with streaming
            logger.info("\nProcessing with streaming...")
            request = SendStreamingMessageRequest(
                id=task_id,
                params=send_params
            )
            logger.info(f"\nSend request created: {request}")
            stream_response = client.send_message_streaming(request)
            response_text = ""
            
            try:
                async for chunk in stream_response:
                    logger.info("response: " + chunk.model_dump_json(exclude_none=True, indent=2))
                    try:
                        result = chunk.model_dump()
                        if result.get('result') and result['result'].get('artifact'):
                            for part in result['result']['artifact'].get('parts', []):
                                if 'text' in part:
                                    response_text += part['text']
                                    if response_text.strip():
                                        logger.info(f"Sending reply with text: {response_text[:200]}...")  # Log first 200 chars
                                        await send_reply(email, response_text)
                                    else:
                                        logger.warning("No response text to send as reply")
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        continue
                
 
                    
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                raise

                
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
        if streaming:
            # Process with streaming
            logger.info("\nProcessing with streaming...")
            request = {
                'id': task_id,
                'params': send_params
            }
            
            if debug:
                logger.info(f"\nSending streaming request: {request}")
                
            async with client.stream('POST', A2A_AGENT_URL, json=request) as response:
                # Process streaming response
                async for chunk in response.aiter_bytes():
                    if debug:
                        logger.info("Received chunk: " + json.dumps(chunk.decode('utf-8'), indent=2))
                    
                    # Here you can process each chunk as it arrives
                    # For example, you might want to update a progress indicator
                    # or process partial results
                    
            logger.info("\nStreaming completed successfully")
            return True
        else:
            # Process without streaming
            logger.info("\nProcessing without streaming...")
            request = {
                'id': task_id,
                'params': send_params
            }
            
            if debug:
                logger.info(f"\nSending request: {request}")
                
            response = await client.post(A2A_AGENT_URL, json=request)
            task = response.json()
            
            if debug:
                logger.info(f"\nTask completed: {task['status']['state']}")
                logger.info(f"Result: {json.dumps(task, indent=2)}")
            else:
                logger.info(f"\nTask completed with status: {task['status']['state']}")
                
            return task['status']['state'].lower() in ['completed', 'succeeded']
            
    except Exception as e:
        logger.error(f"\nError processing email: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


async def send_to_email_processor(email_data: Dict[str, Any], streaming: bool = True, debug: bool = False) -> bool:
    """Send email data to the email processor asynchronously.
    
    Args:
        email_data: The email data to process
        streaming: Whether to use streaming mode
        debug: Enable debug logging
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    logger.info(f"\nSending email to email processor: {email_data}")
    try:
        # Create A2A client
        async with httpx.AsyncClient() as httpx_client:
            client = A2AClient(httpx_client=httpx_client, url=EMAIL_PROCESSOR_URL)
                
            # Ensure required fields are present
            email_model = {
                'subject': email_data.get('subject', '(No subject)'),
                'sender': email_data.get('sender', ''),
                'recipients': email_data.get('recipients', []),
                'body': email_data.get('body', ''),
                'received_at': email_data.get('received_at', datetime.datetime.now(datetime.timezone.utc).isoformat()),
                'metadata': email_data.get('metadata', {}),
                'attachments': email_data.get('attachments', []),
                'id': email_data.get('id', str(uuid.uuid4()))
            }
            logger.info(f"\nEmail model created: {email_model}")
            # Process the email
            return await process_email(client, email_model, streaming=streaming, debug=debug)
            
    except Exception as e:
        logger.error(f"Unexpected error in send_to_email_processor: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    return False

async def process_emails():
    """Process all unread emails asynchronously."""
    logger.info("Starting email processing...")
    
    # Get Gmail service
    service = get_gmail_service()
    if not service:
        logger.error("Failed to initialize Gmail service. Exiting.")
        return
    
    try:
        # Get and process emails
        emails = await get_latest_emails(service)
        if not emails:
            logger.info("No new emails to process.")
            return
        
        logger.info(f"Found {len(emails)} email(s) to process")
        
        # Process each email asynchronously
        tasks = []
        for email in emails:
            email_id = email.get('id')
            logger.info(f"Queueing email - Subject: {email.get('subject', 'No subject')}")
            # Process the email
            async with httpx.AsyncClient() as httpx_client:
                client = A2AClient(httpx_client=httpx_client, url=EMAIL_PROCESSOR_URL)
                success = await process_email(
                    client=client,
                    email=email,
                    streaming=True,
                    debug=True
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


        # Wait for all tasks to complete (but don't raise exceptions)
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except Exception as e:
        logger.error(f"Error in email processing: {str(e)}")
    finally:
        logger.info("Email processing completed")

async def main():
    """Main async function to run the email processor."""
    while True:
        try:
            await process_emails()
            # Wait for 60 seconds before checking for new emails again
            await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down email processor...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}")
            await asyncio.sleep(60)  # Wait before retrying

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())