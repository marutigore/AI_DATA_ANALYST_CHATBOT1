"""
Hash-Based Vector Store Disk Cache Utility.
Computes an MD5 hash of uploaded file byte streams to cache pre-computed embeddings and vector indices
on disk, bypassing re-embedding latency for duplicate file uploads.
"""
import os
import json
import hashlib
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".vector_cache")

def _ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

def compute_file_hash(file_bytes: bytes) -> str:
    """Computes MD5 hash of byte payload."""
    return hashlib.md5(file_bytes).hexdigest()

def get_cached_embeddings(file_bytes: bytes) -> Tuple[Optional[List[List[float]]], Optional[List[str]]]:
    """
    Retrieves cached embeddings and documents if hash matches disk cache.
    """
    _ensure_cache_dir()
    file_hash = compute_file_hash(file_bytes)
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Vector Store Disk Cache HIT for MD5 hash {file_hash[:8]}")
                return data.get("embeddings"), data.get("documents")
        except Exception as e:
            logger.error(f"Error reading vector cache file {cache_file}: {e}")
            
    return None, None

def save_cached_embeddings(file_bytes: bytes, embeddings: List[List[float]], documents: List[str]) -> bool:
    """
    Persists computed embeddings and documents to disk keyed by MD5 hash.
    """
    _ensure_cache_dir()
    file_hash = compute_file_hash(file_bytes)
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}.json")
    
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"embeddings": embeddings, "documents": documents}, f)
            logger.info(f"Saved Vector Store Disk Cache for MD5 hash {file_hash[:8]}")
            return True
    except Exception as e:
        logger.error(f"Failed to save vector cache for hash {file_hash[:8]}: {e}")
        return False
