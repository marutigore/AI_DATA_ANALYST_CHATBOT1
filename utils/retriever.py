"""
Document Retrieval Utility.
Stores embedded chunks in vector search engine and handles similarity search.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)

# Initialize a global in-memory FAISS index
_INDEX = None
_DOCUMENTS = []

def initialize_vector_store(embeddings: List[List[float]], documents: List[str]) -> bool:
    """
    Initializes a FAISS vector index with provided embeddings and documents.
    
    Args:
        embeddings (List[List[float]]): Text vectors.
        documents (List[str]): Original text chunks matching the embeddings.
        
    Returns:
        bool: True if initialization was successful.
        
    Example:
        >>> success = initialize_vector_store(vecs, ["doc1", "doc2"])
    """
    global _INDEX, _DOCUMENTS
    try:
        if faiss is None:
            raise ImportError("faiss is not installed.")
            
        if not embeddings or not documents:
            logger.warning("Empty lists provided. Initializing empty vector store.")
            return False
            
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents lists must have the same length.")
            
        # Clear existing store for a fresh upload to maintain current context
        _DOCUMENTS = documents.copy()
        
        # Dimensions size corresponds to the embedding vector size
        d = len(embeddings[0])
        
        # We use an L2 distance based Flat index for exact similarity search
        _INDEX = faiss.IndexFlatL2(d)
        
        # Convert List of Lists to numpy array required by FAISS ingestion
        vectors = np.array(embeddings).astype('float32')
        _INDEX.add(vectors)
        
        logger.info(f"Initialized Vector Store with {len(documents)} documents.")
        return True
        
    except ImportError as ie:
        logger.error(f"Missing dependency FAISS: {ie}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize Vector Store: {e}")
        raise

def retrieve_similar(query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves the most similar documents to the provided query embedding.
    
    Args:
        query_embedding (List[float]): Vector representation of the user query.
        top_k (int): Number of top results to return.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing "content" and "score".
        
    Example:
        >>> results = retrieve_similar(q_vec, top_k=2)
    """
    global _INDEX, _DOCUMENTS
    try:
        if _INDEX is None or not _DOCUMENTS:
            logger.warning("Vector Store is empty or not initialized.")
            return []
            
        # Reshape to 2D numpy array for FAISS similarity query processing
        q = np.array([query_embedding]).astype('float32')
        
        # Perform nearest neighbor search
        distances, indices = _INDEX.search(q, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            # Validating bounds just in case
            if idx != -1 and idx < len(_DOCUMENTS):
                results.append({
                    "content": _DOCUMENTS[idx],
                    "score": float(distances[0][i])
                })
                
        logger.info(f"Retrieved {len(results)} matches for query.")
        return results
        
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise RuntimeError(f"Retrieval operation failed: {str(e)}")
