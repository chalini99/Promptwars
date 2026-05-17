from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
from main import app

client = TestClient(app)

def test_homepage_loads():
    assert client.get("/").status_code == 200

def test_analyze_rejects_empty():
    assert client.post("/api/analyze", data={}).status_code == 400

def test_analyze_text_works():
    r = client.post("/api/analyze", data={"text_content": "Sample contract", "doc_type": "employment"})
    assert r.status_code == 200
    assert "session_id" in r.json()

def test_invalid_pdf_session():
    assert client.get("/api/export/pdf?session_id=invalid").status_code == 404

def test_chat_endpoint():
    r = client.post("/api/chat", json={"message": "test", "history": [], "contract_text": "x", "analysis_results": {}})
    assert r.status_code == 200
