import os
import base64
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import httpx
from pprint import pprint

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

def get_latest_emails(service, max_results=5):
    """Get the latest emails from Gmail."""
    results = service.users().messages().list(
        userId='me', 
        maxResults=max_results
    ).execute()
    
    messages = results.get('messages', [])
    emails = []
    
    for msg in messages:
        txt = service.users().messages().get(
            userId='me',
            id=msg['id']
        ).execute()
        
        # Get headers
        headers = txt.get('payload', {}).get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
        
        # Get message body
        try:
            if 'parts' in txt['payload']:
                data = txt['payload']['parts'][0]['body']['data']
            else:
                data = txt['payload']['body']['data']
            body = base64.urlsafe_b64decode(data).decode('utf-8')
        except:
            body = "Could not decode email body"
        
        emails.append({
            'id': msg['id'],
            'subject': subject,
            'sender': sender,
            'date': date,
            'snippet': txt.get('snippet', ''),
            'body': body
        })
        
    return emails

async def send_to_summary_agent(email_data):
    """Send email data to the summary agent."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8003/process_task",
            json={
                "id": email_data['id'],
                "artifacts": [{
                    "type": "email",
                    "content": json.dumps({
                        "subject": email_data['subject'],
                        "sender": email_data['sender'],
                        "body": email_data['body']
                    }),
                    "metadata": {
                        "date": email_data['date'],
                        "snippet": email_data['snippet']
                    }
                }]
            }
        )
        return response.json()

async def main():
    # Get Gmail service
    service = get_gmail_service()
    
    # Get latest emails
    print("Fetching latest emails...")
    emails = get_latest_emails(service, max_results=3)
    
    # Process each email with the summary agent
    for email in emails:
        print(f"\nProcessing email: {email['subject']}")
        result = await send_to_summary_agent(email)
        print("Summary Agent Response:")
        pprint(result)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())