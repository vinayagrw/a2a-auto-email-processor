"""
Test script for the Summary Agent using sample data.
This script demonstrates how to use the Summary Agent with sample email data.
"""
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pprint import pprint
from typing import Dict, Any

import httpx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_summary_agent')

# Sample email data
SAMPLE_EMAILS = [
    {
        "id": "sample1",
        "subject": "Request for Quote - Kitchen Remodel",
        "from": "john.doe@example.com",
        "date": (datetime.now() - timedelta(days=1)).isoformat(),
        "body": """
        Hello,
        
        I'm interested in getting a quote for a kitchen remodel. 
        The space is about 200 sq ft and I'd like to include:
        - New cabinets
        - Quartz countertops
        - Backsplash
        - New flooring
        
        Please let me know your availability for a consultation.
        
        Best regards,
        John Doe
        """,
        "snippet": "Request for kitchen remodel quote"
    },
    {
        "id": "sample2",
        "subject": "Follow-up on Bathroom Renovation",
        "from": "sarah.smith@example.com",
        "date": (datetime.now() - timedelta(hours=2)).isoformat(),
        "body": """
        Hi there,
        
        I'm following up on our discussion about renovating two bathrooms. 
        Here are the details we discussed:
        - Master bath: Full remodel with walk-in shower
        - Guest bath: Vanity and flooring update
        
        Could you provide a detailed quote?
        
        Thanks,
        Sarah Smith
        """,
        "snippet": "Follow-up on bathroom renovation quote"
    },
    {
        "id": "sample3",
        "subject": "Urgent: Water Damage Repair Needed",
        "from": "emergency@apartmentcomplex.com",
        "date": datetime.now().isoformat(),
        "body": """
        URGENT: Water Damage Repair Needed
        
        We have a water leak in Unit 3B that needs immediate attention. 
        The leak is coming from the ceiling in the bathroom.
        
        Please contact us as soon as possible to schedule an emergency repair.
        
        Property Manager,
        Oakwood Apartments
        """,
        "snippet": "Urgent water damage repair needed"
    }
]

SUMMARY_AGENT_PORT = 8003

async def send_to_summary_agent(email: dict) -> dict:
    """Send email data to the summary agent"""
    url = f"http://localhost:{SUMMARY_AGENT_PORT}/process_task"
    
    # Prepare the task data according to TaskModel
    task_data = {
        "id": f"test-{str(uuid.uuid4())[:8]}",
        "artifacts": [
            {
                "id": f"email-{str(uuid.uuid4())[:8]}",
                "type": "text/plain",
                "content": email['body'],
                "metadata": {
                    "sender": email['from'],  # Changed from 'from' to 'sender' to match the expected key
                    "subject": email['subject'],
                    "date": email['date'].isoformat() if hasattr(email['date'], 'isoformat') else email['date']
                }
            }
        ],
        "metadata": {
            "source": "test_summary_agent",
            "timestamp": datetime.now(datetime.timezone.utc).isoformat()
        }
    }
    
    timeout = httpx.Timeout(300.0, read=300.0)  # 5 minute timeout for LLM processing
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, json=task_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP error: {str(e)}"}
        except httpx.RequestError as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

async def main():
    logger.info("Starting Summary Agent test with sample data")
    
    # Process each sample email
    for email in SAMPLE_EMAILS:
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING EMAIL: {email['subject']}")
        logger.info("-" * 80)
        logger.info(f"From: {email['from']}")
        logger.info(f"Date: {email['date']}")
        
        logger.debug("\nORIGINAL CONTENT:" + "-"*60)
        logger.debug(email['body'].strip())
        
        try:
            # Send to summary agent
            logger.info("Sending to summary agent...")
            start_time = time.time()
            result = await send_to_summary_agent(email)
            elapsed = time.time() - start_time
            logger.info(f"Request completed in {elapsed:.1f} seconds")
            
            # Log the response
            logger.info("\n" + "="*80)
            logger.info("SUMMARY AGENT RESPONSE")
            logger.info("="*80)
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"Request failed: {error_msg}")
                
                # Try to get the summary from the log file if available
                try:
                    log_file = f"logs/summaries/daily_summary_{datetime.now().strftime('%Y-%m-%d')}.txt"
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            logger.info("Latest summaries from log file:")
                            logger.info("-" * 80)
                            logger.info(log_content[-2000:])  # Show last 2000 chars of log
                except Exception as e:
                    logger.error(f"Could not read log file: {str(e)}")
                
                continue
                
            data = result.get('data', {})
            summaries = data.get('summaries', [])
            
            if not summaries:
                logger.warning("No summaries generated in response")
                logger.debug(f"Full response: {json.dumps(data, indent=2)}")
                continue
                
            for idx, summary in enumerate(summaries, 1):
                logger.info("\n" + "="*50)
                logger.info(f"SUMMARY {idx}:")
                logger.info("="*50)
                
                # Show metadata if available
                meta = summary.get('metadata', {})
                if meta:
                    logger.info("\nMETADATA:")
                    for key, value in meta.items():
                        logger.info(f"  {key}: {value}")
                
                # Log the summary
                logger.info("\nSUMMARY:")
                logger.info("-" * 50)
                logger.info(summary.get('summary', 'No summary available').strip())
                logger.info("="*50 + "\n")
                        
            # Show any warnings or additional info
            if 'warnings' in data and data['warnings']:
                logger.warning("Warnings in response:")
                for warning in data['warnings']:
                    logger.warning(f"  - {warning}")
            
        except Exception as e:
            print(f"\nERROR PROCESSING EMAIL: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
