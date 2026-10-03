import pytest
from fastapi.testclient import TestClient
from app.main import app
import httpx
from unittest.mock import patch
from app.config import settings

client = TestClient(app)

# ─── Existing tests ──────────────────────────────────────────────────────────

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

    response = client.post("/api/v1/ai/chat", json={"message": "Fail me"})
    assert response.status_code == 500
    assert response.json()["success"] == False
    assert "error" in response.json()["response"].lower()
    assert "X-Request-ID" in response.headers
    assert "test_key" not in response.text

# ─── TEST A: Generic question without scan context ────────────────────────────
# Must call Gemini and return an actual educational answer.
# Must NOT return "No scan data is available."
@patch("app.gemini_client.httpx.AsyncClient.post")
def test_A_generic_question_no_context(mock_post):
    gemini_answer = "Ransomware is malicious software that encrypts a victim's files and demands payment for decryption."

    class MockResponse:
        status_code = 200
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": gemini_answer}]}}]}

    async def mock_async_post(*args, **kwargs):
        return MockResponse()

    mock_post.side_effect = mock_async_post
    settings.GEMINI_API_KEY = "test_key"

    response = client.post("/api/v1/ai/chat", json={"message": "What is ransomware?"})
    data = response.json()

    assert response.status_code == 200
    assert data["success"] == True
    # Must be the real Gemini answer, NOT the no-data fallback
    assert data["response"] == gemini_answer
    no_data_phrases = [
        "no scan data", "no active scan", "no scan data is available",
        "no data available", "no scan results"
    ]
    response_lower = data["response"].lower()
    for phrase in no_data_phrases:
        assert phrase not in response_lower, (
            f"Generic question returned a no-scan fallback phrase: '{phrase}'"
        )

# ─── TEST B: Scan-specific question without scan context ─────────────────────
# Must return a truthful no-scan response. Must NOT fabricate scan values.
@patch("app.gemini_client.httpx.AsyncClient.post")
def test_B_scan_question_no_context(mock_post):
    truthful_no_scan = "No active scan data is available to explain."

    class MockResponse:
        status_code = 200
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": truthful_no_scan}]}}]}

    async def mock_async_post(*args, **kwargs):
        # Check that the prompt correctly conveys no-scan context
        return MockResponse()

    mock_post.side_effect = mock_async_post
    settings.GEMINI_API_KEY = "test_key"

    response = client.post("/api/v1/ai/chat", json={"message": "Explain my latest scan"})
    data = response.json()

    assert response.status_code == 200
    assert data["success"] == True
    # The system instruction must NOT include fabricated scan values
    # Verify the prompt sent to Gemini contained the "no scan" instruction
    call_args = mock_post.call_args
    prompt_sent = call_args[1].get("json", {}).get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
    assert "No active scan data" in prompt_sent, (
        f"Prompt for scan-specific question without context did not include no-scan instruction. Got: {prompt_sent[:200]}"
    )
    assert "Scan State:" not in prompt_sent, (
        "Prompt contained fabricated scan data fields when no context was provided"
    )

# ─── TEST C: Scan-specific question WITH real scan context ───────────────────
# Gemini must receive the actual scan context. ScanState must not be mutated.
@patch("app.gemini_client.httpx.AsyncClient.post")
def test_C_scan_question_with_context(mock_post):
    gemini_explanation = "Your system scan shows a threat probability of 0.42 with state SAFE."

    class MockResponse:
        status_code = 200
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": gemini_explanation}]}}]}

    async def mock_async_post(*args, **kwargs):
        return MockResponse()

    mock_post.side_effect = mock_async_post
    settings.GEMINI_API_KEY = "test_key"

    scan_context = {
        "scan_state": "safe",
        "threat_level": "SAFE",
        "ml_executed": True,
        "ml_probability": 0.42,
        "files_inspected": 12000,
        "processes_analyzed": 150,
        "features": {"entropy": 0.12, "cpu_usage": 3.5},
        "error_message": None,
    }

    response = client.post(
        "/api/v1/ai/chat",
        json={"message": "Explain my latest scan", "context": scan_context}
    )
    data = response.json()

    assert response.status_code == 200
    assert data["success"] == True

    # Verify the prompt sent to Gemini included the real scan values
    call_args = mock_post.call_args
    prompt_sent = call_args[1].get("json", {}).get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
    assert "Scan State: safe" in prompt_sent, "Prompt did not include real scan state"
    assert "0.4200" in prompt_sent, "Prompt did not include real ML probability"
    assert "12000" in prompt_sent, "Prompt did not include real file count"
    assert "test_key" not in prompt_sent, "API key leaked into prompt"

# ─── TEST D: Generic cybersecurity question without scan context ──────────────
# Must call Gemini. Must NOT return a no-scan fallback for an educational question.
@patch("app.gemini_client.httpx.AsyncClient.post")
def test_D_generic_cybersecurity_no_context(mock_post):
    gemini_answer = "You can protect your PC with a firewall, antivirus, and regular updates."

    class MockResponse:
        status_code = 200
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": gemini_answer}]}}]}

    async def mock_async_post(*args, **kwargs):
        return MockResponse()

    mock_post.side_effect = mock_async_post
    settings.GEMINI_API_KEY = "test_key"

    response = client.post("/api/v1/ai/chat", json={"message": "How do I protect my PC from ransomware?"})
    data = response.json()

    assert response.status_code == 200
    assert data["success"] == True
    assert data["response"] == gemini_answer
    # Must be the real answer, not a no-data fallback
    assert "no scan data" not in data["response"].lower()
    assert "no active scan" not in data["response"].lower()

    # Verify prompt was constructed correctly (no-context system instruction used)
    call_args = mock_post.call_args
    prompt_sent = call_args[1].get("json", {}).get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
    assert "Scan State:" not in prompt_sent, "Fabricated scan state sent to Gemini for a generic question"
    assert "cybersecurity question" in prompt_sent or "No active scan data" in prompt_sent, (
        "No-context system instruction not present in prompt"
    )
