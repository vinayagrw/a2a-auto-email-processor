import asyncio
import logging
import sys
import httpx
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('test_workflow')

# Configuration
EMAIL_PROCESSOR_URL = "http://localhost:8001"

# Test data matching the EmailModel schema
SAMPLE_EMAILS = [
    {
        "id": "test-msg-001",
        "subject": "Request for Quote - Kitchen Renovation",
        "sender": "client@example.com",
        "recipients": ["info@contractor.com"],
        "body": "Hello, I'm looking to renovate my kitchen. The area is about 200 sq ft. Can you provide a quote?",
        "received_at": datetime.now(timezone.utc),
        "metadata": {
            "source": "test"
        },
        "attachments": []
    },
    {
        "id": "test-msg-002",
        "subject": "Daily Construction Report - Site A",
        "sender": "foreman@construction.com",
        "recipients": ["project@contractor.com"],
        "body": "Daily progress: Framing 80% complete, electrical 50% complete. Issues: Delayed lumber delivery.",
        "received_at": datetime.now(timezone.utc),
        "metadata": {
            "source": "test"
        },
        "attachments": []
    }
]

class EmailWorkflowTester:
    def __init__(self):
        self.client = httpx.AsyncClient()
        
    async def close(self):
        await self.client.aclose()
    
    async def check_health(self) -> bool:
        """Check if the email processor is healthy."""
        try:
            response = await self.client.get(f"{EMAIL_PROCESSOR_URL}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def process_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send email to EmailProcessor for processing."""
        logger.info(f"Processing email: {email_data['subject']}")
        
        try:
# Map to the expected EmailModel format
            payload = {
                "id": email_data["id"],
                "subject": email_data["subject"],
                "sender": email_data["sender"],
                "recipients": email_data["recipients"],
                "body": email_data["body"],
                "received_at": email_data["received_at"].isoformat(),
                "metadata": email_data.get("metadata", {}),
                "attachments": email_data.get("attachments", [])
            }
            
            logger.info(f"Sending payload: {payload}")
            
            response = await self.client.post(
                f"{EMAIL_PROCESSOR_URL}/process_email",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Failed to process email: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    async def test_workflow(self):
        """Test the complete workflow with sample emails."""
        if not await self.check_health():
            logger.error("Email Processor is not healthy. Exiting.")
            return
            
        logger.info("Starting workflow tests...")
        
        for i, email in enumerate(SAMPLE_EMAILS, 1):
            logger.info(f"\nTest {i}: {email['subject']}")
            logger.info("-" * 50)
            
            # Process email through EmailProcessor
            result = await self.process_email(email)
            
            if "error" in result:
                logger.error(f"Error: {result['error']}")
                continue
                
            # Log the processing result
            logger.info("Processing Result:")
            logger.info(f"Status: {result}")
            
            if 'intent' in result:
                logger.info(f"Detected Intent: {result['intent']}")
                
                # Show next steps based on intent
                if result['intent'] in ["quote_request", "common_question"]:
                    logger.info("Action: Task delegated to ResponseAgent")
                elif result['intent'] == "summary_needed":
                    logger.info("Action: Task delegated to SummaryAgent")
            
            if 'entities' in result:
                logger.info(f"Extracted Entities: {result['entities']}")
                
            logger.info("=" * 50)
        
        logger.info("\nAll tests completed")

async def main():
    tester = EmailWorkflowTester()
    try:
        await tester.test_workflow()
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())