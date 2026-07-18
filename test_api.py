import pytest
import io
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api import app, SESSION_STORE

client = TestClient(app)

CSV_CONTENT = """id,math score,reading score
1,80,85
2,90,95
3,70,75
4,60,65
"""

@pytest.fixture(autouse=True)
def clear_sessions():
    SESSION_STORE.clear()

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_sessions_empty():
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert response.json()["active_sessions_count"] == 0

@patch("api.ChatGoogleGenerativeAI")
@patch("api.create_pandas_dataframe_agent")
def test_upload_success(mock_agent_creator, mock_llm_creator):
    # Mock agent and LLM to avoid real API calls
    mock_agent = MagicMock()
    mock_agent_creator.return_value = mock_agent
    
    file_data = {"file": ("test.csv", io.BytesIO(CSV_CONTENT.encode("utf-8")), "text/csv")}
    response = client.post("/api/upload", files=file_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["filename"] == "test.csv"
    assert data["metrics"]["rows"] == 4
    assert data["metrics"]["cols"] == 3
    assert len(data["suggestions"]) > 0

def test_upload_file_too_large():
    # Create file content larger than 10MB (limit set in config)
    large_content = "a" * (11 * 1024 * 1024) # 11MB
    file_data = {"file": ("large.csv", io.BytesIO(large_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/upload", files=file_data)
    
    assert response.status_code == 400
    assert "exceeds the limit" in response.json()["detail"]

def test_upload_unsupported_type():
    file_data = {"file": ("test.txt", io.BytesIO(b"some text data"), "text/plain")}
    response = client.post("/api/upload", files=file_data)
    assert response.status_code == 500

@patch("api.ChatGoogleGenerativeAI")
@patch("api.create_pandas_dataframe_agent")
def test_chat_success(mock_agent_creator, mock_llm_creator):
    # Mock agent and invoke method
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"output": "Mocked analysis output."}
    mock_agent_creator.return_value = mock_agent
    
    # 1. Upload first to get session_id
    file_data = {"file": ("test.csv", io.BytesIO(CSV_CONTENT.encode("utf-8")), "text/csv")}
    response = client.post("/api/upload", files=file_data)
    session_id = response.json()["session_id"]
    
    # 2. Chat using session_id
    chat_response = client.post(
        "/api/chat",
        json={"session_id": session_id, "prompt": "Give me a summary of math scores."}
    )
    assert chat_response.status_code == 200
    chat_data = chat_response.json()
    assert chat_data["response"] == "Mocked analysis output."
    
def test_chat_session_not_found():
    response = client.post(
        "/api/chat",
        json={"session_id": "non-existent-id", "prompt": "Hi"}
    )
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]

def test_chat_empty_query():
    # Setup session
    SESSION_STORE["mock-session"] = {
        "agent": MagicMock(),
        "filename": "test.csv",
        "chart_filename": "temp_chart_mock.json",
        "created_at": 1234567890
    }
    response = client.post(
        "/api/chat",
        json={"session_id": "mock-session", "prompt": ""}
    )
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]
