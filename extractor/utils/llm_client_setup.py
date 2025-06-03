"""
Module: llm_client_setup
Functionality: Handles the initialization of the OpenAI API client.
               It loads API keys and base URLs from environment variables
               and provides a function to get a configured client instance.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def get_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    if not base_url: # Base URL can be optional if using official OpenAI
        logger.warning("OPENAI_BASE_URL not found in environment variables, using default.")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"OpenAI client initialized for base_url: {base_url if base_url else 'default'}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise