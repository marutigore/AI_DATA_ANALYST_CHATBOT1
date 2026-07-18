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

class SessionVectorStore:
    """Session-isolated Vector Store using in-memory FAISS Index Flat L2."""
    def __init__(self):
        self.index = None
        self.documents = []

    def initialize(self, embeddings: List[List[float]], documents: List[str]) -> bool:
        """Initializes a FAISS vector index with provided embeddings and documents."""
        try:
            if faiss is None:
                raise ImportError("faiss is not installed.")
                
            if not embeddings or not documents:
                logger.warning("Empty lists provided. Initializing empty vector store.")
                return False
                
            if len(embeddings) != len(documents):
                raise ValueError("Embeddings and documents lists must have the same length.")
                
            # Clear existing store for a fresh upload
            self.documents = documents.copy()
            
            # Dimensions size corresponds to the embedding vector size
            d = len(embeddings[0])
            self.index = faiss.IndexFlatL2(d)
            
            # Convert List of Lists to numpy array
            vectors = np.array(embeddings).astype('float32')
            self.index.add(vectors)
            
            logger.info(f"Initialized Vector Store with {len(documents)} documents.")
            return True
            
        except ImportError as ie:
            logger.error(f"Missing dependency FAISS: {ie}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store: {e}")
            raise

    def retrieve_similar(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves the most similar documents to the query embedding."""
        try:
            if self.index is None or not self.documents:
                logger.warning("Vector Store is empty or not initialized.")
                return []
                
            # Reshape to 2D numpy array for FAISS similarity query
            q = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(q, top_k)
            
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx != -1 and idx < len(self.documents):
                    results.append({
                        "content": self.documents[idx],
                        "score": float(distances[0][i])
                    })
                    
            logger.info(f"Retrieved {len(results)} matches for query.")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise RuntimeError(f"Retrieval operation failed: {str(e)}")

# Global instance for legacy functional calls
_GLOBAL_STORE = SessionVectorStore()

def initialize_vector_store(embeddings: List[List[float]], documents: List[str]) -> bool:
    return _GLOBAL_STORE.initialize(embeddings, documents)

def retrieve_similar(query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    return _GLOBAL_STORE.retrieve_similar(query_embedding, top_k)

