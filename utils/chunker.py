"""
Document Chunking Utility.
Responsible for breaking down text representations into manageable chunks.
"""
import logging
from typing import List
import pandas as pd
from config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def chunk_dataframe_to_text(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Converts a DataFrame into text metadata and chunks it.
    Serves as a fallback for pure vector-based RAG if required for validation.
    
    Args:
        df (pd.DataFrame): The pandas DataFrame to chunk.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between subsequent chunks.
        
    Returns:
        List[str]: A list of text chunks representing the DataFrame content.
        
    Example:
        >>> chunks = chunk_dataframe_to_text(df)
        >>> print(len(chunks))
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided to chunker.")
            return []
            
        # Convert first 100 rows into a text representation for indexing schema/preview
        # Doing only 100 rows to keep it lightweight for vector matching
        text_data = df.head(100).to_csv(index=False)
        
        # Initialize LangChain's recursive text splitter with config bounds
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", " ", ""]
        )
        
        # Split the text so it can be indexed correctly
        chunks = text_splitter.split_text(text_data)
        logger.info(f"Successfully split data into {len(chunks)} chunks.")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error during dataframe chunking: {e}")
        raise RuntimeError(f"Chunking failed: {str(e)}")
