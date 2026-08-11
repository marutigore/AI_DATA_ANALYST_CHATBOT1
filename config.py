"""
Configuration settings for the AI Data Analyst Chatbot.
Loads environment variables and sets up constants.
"""
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
MAX_UPLOAD_SIZE_MB = 200

SUPPORTED_FILE_TYPES = [".csv", ".xlsx", ".xls"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_QUERY_LENGTH = 2000
SESSION_TTL_MINUTES = 30

import json
import time

class JSONLogFormatter(logging.Formatter):
    """Formats log entries as structured JSON objects for ELK / Datadog log ingestion."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def setup_structured_logging():
    """Configures structured JSON logging globally."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_keys() -> bool:
    """
    Checks if necessary API keys are present in the environment.
    
    Returns:
        bool: True if required keys are found, False otherwise.
        
    Example:
        >>> check_keys()
        True
    """
    try:
        # We check for GOOGLE_API_KEY as the primary driver for LangChain code agent
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning("GOOGLE_API_KEY is not set correctly.")
            return False
            
        logger.info("All required API keys are configured.")
        return True
    except Exception as e:
        logger.error(f"Error checking API keys: {e}")
        return False

# Settings dictionary for global access if needed
CONFIG_DICT: Dict[str, Any] = {
    "max_upload": MAX_UPLOAD_SIZE_MB,
    "chunk_size": CHUNK_SIZE,
    "overlap": CHUNK_OVERLAP
}
