import pytest
from fastapi.testclient import TestClient
from app.main import app
import httpx
from unittest.mock import patch, AsyncMock
from app.config import settings

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_missing_message_rejection():
    response = client.post("/api/v1/ai/chat", json={"context": {}})
    assert response.status_code == 422

def test_oversized_message_rejection():
    response = client.post("/api/v1/ai/chat", json={"message": "a" * 4001})
    assert response.status_code == 422

def test_rate_limiting():
    # settings.RATE_LIMIT_PER_MINUTE is 20
    # send 21 requests
    responses = []
    for _ in range(21):
        responses.append(client.get("/health")) # /health is not rate limited
    
    # But /api/v1/ai/chat is rate limited
    chat_responses = []
    # Temporarily set rate limit to 2 for testing
    old_limit = settings.RATE_LIMIT_PER_MINUTE
    settings.RATE_LIMIT_PER_MINUTE = 2
    
    client.post("/api/v1/ai/chat", json={"message": "1"})
    client.post("/api/v1/ai/chat", json={"message": "2"})
    resp3 = client.post("/api/v1/ai/chat", json={"message": "3"})
    
    settings.RATE_LIMIT_PER_MINUTE = old_limit
    assert resp3.status_code == 429
    assert "Too many requests" in resp3.json()["detail"]

@patch("app.gemini_client.httpx.AsyncClient.post")
def test_valid_ai_request(mock_post):
    class MockResponse:
        status_code = 200
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Mocked response"}]}}]}
            
    async def mock_async_post(*args, **kwargs):
        return MockResponse()
        
    mock_post.side_effect = mock_async_post
    settings.GEMINI_API_KEY = "test_key"

    response = client.post("/api/v1/ai/chat", json={"message": "What is ransomware?"})
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["response"] == "Mocked response"
    assert "X-Request-ID" in response.headers
    assert "test_key" not in response.text

@patch("app.gemini_client.httpx.AsyncClient.post")
def test_gemini_fallback_and_failure(mock_post):
    class MockErrorResponse:
        def __init__(self, status):
            self.status_code = status
            self.request = httpx.Request("POST", "http://test")
            self.response = self
    
    async def mock_async_post(*args, **kwargs):
        return MockErrorResponse(500)
        
    mock_post.side_effect = mock_async_post
    settings.GEMINI_API_KEY = "test_key"
    
    # Should trigger the 500 error mapping in the middleware
    response = client.post("/api/v1/ai/chat", json={"message": "Fail me"})
    assert response.status_code == 500
    assert response.json()["success"] == False
    assert "error" in response.json()["response"].lower()
    assert "X-Request-ID" in response.headers
    assert "test_key" not in response.text
