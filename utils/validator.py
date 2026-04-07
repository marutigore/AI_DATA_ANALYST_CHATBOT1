"""
Validation Utility.
Validates inputs to ensure they meet constraints before processing.
"""
import logging
from config import MAX_QUERY_LENGTH

logger = logging.getLogger(__name__)

def validate_query(query: str) -> str:
    """
    Validates a user query against length and content constraints.
    
    Args:
        query (str): The natural language query from the user.
        
    Returns:
        str: The sanitized and validated query.
        
    Example:
        >>> validated = validate_query(" Show average sales ")
    """
    try:
        # Check for empty inputs directly
        if not query or not query.strip():
            logger.warning("Empty query submitted")
            raise ValueError("Query cannot be empty.")
            
        # Strip leading/trailing whitespaces to sanitize inputs systematically
        sanitized = query.strip()
        
        # Prevent overly long queries that might degrade LLM performance or increase costs limit
        if len(sanitized) > MAX_QUERY_LENGTH:
            msg = f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters."
            logger.warning(msg)
            raise ValueError(msg)
            
        logger.info("Query successfully validated.")
        return sanitized
        
    except ValueError as ve:
        # We re-raise ValueError so UI can catch and display
        logger.error(f"Validation failed: {ve}")
        raise ve
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        raise RuntimeError("Internal validation error.")
