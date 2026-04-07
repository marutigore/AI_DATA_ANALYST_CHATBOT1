"""
Document Embedding Utility.
Converts text chunks into numerical vectors using Language Models.
"""
import logging
from typing import List, Any
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# Initialize model globally to avoid repeated loading
_MODEL = None

def get_embeddings_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Lazily loads and returns the sentence transformer embedding model.
    Uses a local lightweight model to prevent excessive API costs.
    
    Args:
        model_name (str): The name of the sentence-transformers model.
        
    Returns:
        Any: The SentenceTransformer model instance.
        
    Example:
        >>> model = get_embeddings_model()
    """
    global _MODEL
    try:
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers is not installed.")
            
        # Implementation of Singleton pattern for lazy loading
        if _MODEL is None:
            logger.info(f"Loading embedding model: {model_name}")
            _MODEL = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
            
        return _MODEL
        
    except ImportError as ie:
        logger.error(f"Missing dependency: {ie}")
        raise
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text string chunks.
    
    Args:
        texts (List[str]): List of texts to embed.
        
    Returns:
        List[List[float]]: List of corresponding embedding vectors.
        
    Example:
        >>> vecs = create_embeddings(["Hello World", "Data analysis"])
    """
    try:
        if not texts:
            logger.warning("No texts provided for embedding.")
            return []
            
        # Load the unified embedding model instance
        model = get_embeddings_model()
        
        # Convert textual chunks into dense numpy vectors and back to list
        logger.info(f"Generating embeddings for {len(texts)} chunks.")
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise RuntimeError(f"Embedding generation failed: {str(e)}")
