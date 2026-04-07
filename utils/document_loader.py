"""
Document Loading Utility.
Handles loading of CSV and Excel files into a pandas DataFrame.
"""
import logging
import pandas as pd
from typing import Optional, Union, IO
from config import SUPPORTED_FILE_TYPES

logger = logging.getLogger(__name__)

def load_document(file_obj: IO, file_name: str) -> Optional[pd.DataFrame]:
    """
    Loads a CSV or Excel document into a pandas DataFrame.
    
    Args:
        file_obj (IO): The uploaded file object from Streamlit or standard IO.
        file_name (str): The name of the file to determine its extension.
        
    Returns:
        Optional[pd.DataFrame]: The loaded DataFrame, or None if loading fails.
        
    Example:
        >>> with open('data.csv', 'rb') as f:
        >>>     df = load_document(f, 'data.csv')
    """
    try:
        # Extract the file extension to determine the loader
        ext = file_name[file_name.rfind("."):].lower() if "." in file_name else ""
        
        # Check against supported file types defined in config
        if ext not in SUPPORTED_FILE_TYPES:
            logger.error(f"Unsupported file type: {ext}")
            raise ValueError(f"File type {ext} is not supported. Use {SUPPORTED_FILE_TYPES}")
            
        # Parse based on extension type using pandas
        if ext == ".csv":
            df = pd.read_csv(file_obj)
            logger.info(f"Successfully loaded CSV with shape {df.shape}")
            return df
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_obj)
            logger.info(f"Successfully loaded Excel with shape {df.shape}")
            return df
            
    except ValueError as ve:
        logger.error(f"Validation error during document load: {ve}")
        raise ve
    except pd.errors.EmptyDataError:
        logger.error("The uploaded file is empty.")
        raise ValueError("The uploaded file is empty.")
    except pd.errors.ParserError:
        logger.error("The uploaded file is corrupted or not properly formatted.")
        raise ValueError("The uploaded file is corrupted or not properly formatted.")
    except Exception as e:
        logger.error(f"Unexpected error loading document: {e}")
        raise
        
    return None
