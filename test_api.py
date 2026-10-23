from fastapi.testclient import TestClient

from app.api import app


def test_health_and_ask():
    with TestClient(app) as client:  # runs the startup event (indexes ./data)
        health = client.get("/health").json()
        assert health["chunks_indexed"] > 0

        resp = client.post("/ask", json={"question": "How long is the warranty?", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body and len(body["sources"]) == 3


def test_rejects_too_short_question():
    with TestClient(app) as client:
        assert client.post("/ask", json={"question": "a"}).status_code == 422
