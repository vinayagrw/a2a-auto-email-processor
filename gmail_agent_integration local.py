import os
import base64
import json
import email
from datetime import datetime, timezone
from email.header import decode_header
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import httpx
from pprint import pprint
from typing import List, Dict, Any, Optional
import asyncio
import logging
from pathlib import Path
import httpx
import asyncio

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'gmail_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the token.json file.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service

def decode_text(text: str) -> str:
    """Decode email subject or sender information."""
    if not text:
        return ""
    try:
        decoded = []
        for part, encoding in decode_header(text):
            if isinstance(part, bytes):
                decoded.append(part.decode(encoding or 'utf-8', errors='replace'))
            else:
                decoded.append(part)
        return ' '.join(decoded)
    except Exception as e:
        logger.error(f"Error decoding text: {e}")
        return str(text)

def get_email_body(message: Dict[str, Any]) -> str:
    """Extract email body from Gmail message."""
    if 'parts' in message['payload']:
        parts = message['payload']['parts']
        for part in parts:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data', '')
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
            elif part['mimeType'] == 'multipart/alternative':
                for subpart in part['parts']:
                    if subpart['mimeType'] == 'text/plain':
                        data = subpart['body'].get('data', '')
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
    else:
        data = message['payload']['body'].get('data', '')
        if data:
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
    return ""

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
    headers = {h['name'].lower(): h['value'] for h in message['payload'].get('headers', [])}
    
    # Parse email addresses
    def parse_email_addresses(header: str) -> List[str]:
        if not header:
            return []
        # Simple email extraction - for more robust parsing, consider using email.utils.parseaddr
        return [addr.strip() for addr in header.split(',') if '@' in addr]
    
    # Convert labels to a comma-separated string for ChromaDB compatibility
    labels = message.get('labelIds', [])
    labels_str = ','.join(labels) if isinstance(labels, list) else str(labels)
    
    # Get email data
    email_data = {
        "id": message['id'],
        "subject": decode_text(headers.get('subject', 'No Subject')),
        "sender": headers.get('from', ''),
        "recipients": parse_email_addresses(headers.get('to', '')),
        "body": get_email_body(message),
        "received_at": datetime.fromtimestamp(int(message['internalDate'])/1000, tz=timezone.utc),
        "metadata": {
            "gmail_id": message['id'],
            "labels": labels_str,  # Store as string instead of list
            "thread_id": message.get('threadId', ''),
            # Only include string values in headers
            "headers": json.dumps({k: str(v) for k, v in headers.items() 
                                 if k not in ['from', 'to', 'subject']})
        },
        "attachments": get_attachments(message)
    }
    return email_data

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

async def send_to_email_processor(email_data: Dict[str, Any], processor_url: str = "http://localhost:8001/process_email") -> None:
    """Send email data to the email processor asynchronously."""
    try:
        # Convert datetime to ISO format for JSON serialization
        payload = email_data.copy()
        if 'received_at' in payload and payload['received_at']:
            payload['received_at'] = payload['received_at'].isoformat()
        
        logger.info(f"Sending email to processor - Subject: {payload.get('subject', 'No subject')}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Send the request and wait for response
            response = await client.post(
                processor_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Log the response
            logger.info(f"Response from email processor - Status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Error response: {response.text}")
            else:
                logger.info(f"Successfully processed email: {response.text}")
            
    except Exception as e:
        logger.error(f"Error sending email to processor: {str(e)}")

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
            
            # Create a task for each email
            task = asyncio.create_task(send_to_email_processor(email))
            tasks.append(task)
        
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

async def check_health():
    """Check if the email processor is healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/health", timeout=10.0)
            return {
                "status": response.status_code,
                "response": response.text,
                "error": None
            }
    except Exception as e:
        return {
            "status": None,
            "response": None,
            "error": str(e)
        }

if __name__ == '__main__':
    # First check if the email processor is running
    health_check = asyncio.run(check_health())
    print("\n=== Health Check ===")
    if health_check["error"]:
        print("Please make sure the email processor is running on http://localhost:8001")
    else:
        print(f"Status: {health_check['status']}")
        print(f"Response: {health_check['response']}")
        print("\nStarting email processing...")
        asyncio.run(main())