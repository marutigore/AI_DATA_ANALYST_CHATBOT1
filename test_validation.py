"""
Validation Tests for AI Data Analyst Chatbot
Runs 5 core tests to ensure the system is fully operational.
"""
import pytest
import os
import pandas as pd
import sys

def test_imports():
    """Test 1: Verify all core modules import correctly."""
    try:
        import config
        from utils import document_loader, chunker, embedder, retriever, validator
        import app
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_config():
    """Test 2: Verify config parses environment and constants correctly."""
    import config
    assert hasattr(config, "MAX_UPLOAD_SIZE_MB")
    assert hasattr(config, "CHUNK_SIZE")
    assert isinstance(config.SUPPORTED_FILE_TYPES, list)

def test_chunking():
    """Test 3: Verify document chunking logic creates non-empty text representations."""
    from utils.chunker import chunk_dataframe_to_text
    
    # Create mock dataframe
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    chunks = chunk_dataframe_to_text(df, chunk_size=50)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)

def test_embedding(monkeypatch):
    """Test 4: Verify vector store initialization works with generated embeddings."""
    from utils.embedder import create_embeddings, get_embeddings_model
    from utils.retriever import initialize_vector_store
    
    # Mocking out the actual sentence transformer to avoid downloading models during tests
    class MockModel:
        def encode(self, texts, convert_to_numpy=True):
            import numpy as np
            # Return mock vectors of dim 3
            return np.zeros((len(texts), 3))
    
    monkeypatch.setattr("utils.embedder.get_embeddings_model", lambda *args, **kwargs: MockModel())
    
    texts = ["Test chunk 1", "Test chunk 2"]
    embs = create_embeddings(texts)
    
    assert len(embs) == 2
    assert len(embs[0]) == 3
    
    success = initialize_vector_store(embs, texts)
    assert success is True

def test_end_to_end_validation():
    """Test 5: Verify query validation constraints are enforced."""
    from utils.validator import validate_query
    from config import MAX_QUERY_LENGTH
    
    # Empty query should fail
    with pytest.raises(ValueError, match="Query cannot be empty"):
        validate_query("")
        
    # Valid query should pass
    valid = "What is the average of column A?"
    assert validate_query(valid) == valid
    
    # Extremely long query should fail
    long_query = "a" * (MAX_QUERY_LENGTH + 10)
    with pytest.raises(ValueError, match="Query exceeds maximum length"):
        validate_query(long_query)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
