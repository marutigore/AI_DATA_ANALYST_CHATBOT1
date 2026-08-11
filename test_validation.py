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

def test_session_vector_store(monkeypatch):
    """Test 6: Verify SessionVectorStore class works in isolation."""
    from utils.retriever import SessionVectorStore
    from utils.embedder import create_embeddings
    
    class MockModel:
        def encode(self, texts, convert_to_numpy=True):
            import numpy as np
            return np.zeros((len(texts), 3))
            
    monkeypatch.setattr("utils.embedder.get_embeddings_model", lambda *args, **kwargs: MockModel())
    
    texts = ["Doc A", "Doc B"]
    embs = create_embeddings(texts)
    
    store = SessionVectorStore()
    assert store.initialize(embs, texts) is True
    
    # Mock a query embedding
    q_emb = [0.0, 0.0, 0.0]
    matches = store.retrieve_similar(q_emb, top_k=1)
    assert len(matches) == 1
    assert matches[0]["content"] in texts

def test_sandbox_security():
    """Test 7: Verify AST sandbox catches forbidden imports and builtins."""
    from utils.sandbox import validate_code_security
    
    # Valid pandas code
    valid_code = "import pandas as pd\ndf['total'] = df['a'] + df['b']"
    is_valid, reason = validate_code_security(valid_code)
    assert is_valid is True
    
    # Forbidden import
    invalid_import = "import os\nos.system('dir')"
    is_valid, reason = validate_code_security(invalid_import)
    assert is_valid is False
    assert "Security Violation" in reason
    
    # Forbidden builtin call
    invalid_builtin = "eval('1 + 1')"
    is_valid, reason = validate_code_security(invalid_builtin)
    assert is_valid is False
    assert "Security Violation" in reason

def test_llm_factory():
    """Test 8: Verify Multi-Provider LLM Factory returns base LLM with fallback support."""
    from utils.llm_factory import get_llm
    llm = get_llm()
    assert llm is not None

def test_rate_limiter():
    """Test 9: Verify RateLimiter blocks requests exceeding quota."""
    from utils.rate_limiter import RateLimiter
    limiter = RateLimiter(requests_per_minute=2)
    
    # First 2 requests should be allowed
    assert limiter.is_allowed("test_ip")[0] is True
    assert limiter.is_allowed("test_ip")[0] is True
    
    # 3rd request within window should be blocked
    allowed, retry_after = limiter.is_allowed("test_ip")
    assert allowed is False
    assert retry_after > 0

def test_query_engine():
    """Test 10: Verify DuckDB High-Performance Query Engine executes SQL on DataFrames."""
    import pandas as pd
    from utils.query_engine import execute_sql_query
    
    df = pd.DataFrame({"category": ["A", "A", "B"], "sales": [100, 200, 300]})
    res = execute_sql_query(df, "SELECT category, SUM(sales) as total_sales FROM df GROUP BY category ORDER BY category")
    
    assert len(res) == 2
    assert res.iloc[0]["category"] == "A"
    assert res.iloc[0]["total_sales"] == 300
    assert res.iloc[1]["category"] == "B"
    assert res.iloc[1]["total_sales"] == 300

def test_data_cleaner():
    """Test 11: Verify Automated Data Profiling & Cleaning Utility."""
    import pandas as pd
    from utils.data_cleaner import auto_clean_dataframe
    
    raw_df = pd.DataFrame({
        "name": [" Alice ", "Bob "],
        "price": ["$1,200.50 ", "$350.00"]
    })
    cleaned_df, report = auto_clean_dataframe(raw_df)
    
    assert cleaned_df["name"].iloc[0] == "Alice"
    assert cleaned_df["price"].iloc[0] == 1200.50
    assert "price" in report["parsed_currency_cols"]

def test_report_generator():
    """Test 12: Verify Executive PDF Report Generator produces valid PDF bytes."""
    from utils.report_generator import generate_pdf_report
    
    metrics = {"rows": 100, "cols": 5, "memory_mb": 1.2, "missing": 0}
    chat_history = [
        {"role": "user", "content": "Show sales summary"},
        {"role": "assistant", "content": "Total sales amount is $50,000."}
    ]
    pdf_bytes = generate_pdf_report("sales_data.csv", metrics, chat_history)
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")

def test_db_connector():
    """Test 13: Verify SQL Database Connector Interface with SQLite."""
    import sqlite3
    from utils.db_connector import get_db_table_names, query_db_to_dataframe
    
    # Create in-memory SQLite DB
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INT, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()
    
    tables = get_db_table_names(conn)
    assert "users" in tables
    
    df = query_db_to_dataframe(conn, "SELECT * FROM users")
    assert len(df) == 2
    assert df.iloc[0]["name"] == "Alice"
    conn.close()

def test_stats_engine():
    """Test 14: Verify Statistical Hypothesis Testing Engine."""
    import pandas as pd
    from utils.stats_engine import calculate_correlations, run_ttest, run_anova
    
    df = pd.DataFrame({
        "gender": ["M", "M", "F", "F"],
        "score": [90, 85, 70, 65],
        "age": [25, 26, 22, 21]
    })
    
    # Correlation
    corrs = calculate_correlations(df)
    assert "score" in corrs
    
    # T-test
    ttest_res = run_ttest(df, "gender", "score")
    assert "p_value" in ttest_res
    assert ttest_res["statistically_significant"] is True
    
    # ANOVA
    anova_res = run_anova(df, "gender", "score")
    assert "f_statistic" in anova_res

def test_hybrid_search():
    """Test 15: Verify Hybrid BM25 + FAISS Reciprocal Rank Fusion Search."""
    from utils.hybrid_search import hybrid_rrf_search
    
    docs = ["Column Name: revenue", "Column Name: customer_id", "Sales trend over years"]
    faiss_matches = [{"content": "Sales trend over years", "score": 0.1}]
    
    # Query for exact keyword 'revenue'
    fused_results = hybrid_rrf_search("revenue", [0.0, 0.0], docs, faiss_matches, top_k=2)
    assert len(fused_results) > 0
    assert fused_results[0]["content"] == "Column Name: revenue"

if __name__ == "__main__":
    pytest.main(["-v", __file__])









