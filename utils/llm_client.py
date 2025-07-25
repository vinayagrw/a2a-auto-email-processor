"""LLM client for email classification using local LLM (Ollama)."""
import logging
from typing import List

import httpx

from config import OLLAMA_API_BASE, OLLAMA_MODEL, LOG_LEVEL

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

class LLMClassifier:
    """LLM-based email classifier."""
    
    def __init__(self, base_url: str = None, model: str = None):
        """Initialize the LLM classifier.
        
        Args:
            base_url: Base URL for the Ollama API (default: from config)
            model: Model name to use (default: from config)
        """
        self.base_url = base_url or OLLAMA_API_BASE
        self.model = model or OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=300.0)

    async def classify_email(self, email_subject: str, email_body: str) -> str:
        """Classify email intent using LLM.
        
        Args:
            email_subject: The email subject
            email_body: The email body text
            
        Returns:
            str: One of ["quote_request", "common_question", "summary_needed"]
        """
        try:
            # Prepare the prompt
            prompt = f"""
            Analyze the following email and classify its intent. 
            Choose ONLY ONE of these intents:
            - "quote_request": For requests about pricing, quotes, or costs
            - "common_question": For general inquiries or questions
            - "summary_needed": For reports, updates, or status summaries
            
            Email Subject: {email_subject}
            Email Body: {email_body}
            
            Respond with ONLY the intent label in quotes, e.g. "quote_request".
            """
            logger.info(f"Classifying email with LLM for prompt: {prompt}")
            # Call the LLM
            response = await self.client.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1
                }
            )
            logger.info(f"LLM response: {response}")
            response.raise_for_status()
            
            # Parse and validate the response
            intent = response.json().get("response", "").strip().strip('"\'').lower()
            valid_intents = ["quote_request", "common_question", "summary_needed"]
            
            if intent not in valid_intents:
                logger.warning(f"Invalid intent from LLM: {intent}. Defaulting to 'common_question'")
                return "common_question"
                
            return intent
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {str(e)}")
            return "common_question"  # Default fallback
            
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
