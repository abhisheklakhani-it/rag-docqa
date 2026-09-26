import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Fresh app with an empty index in a temporary folder for every test.
    monkeypatch.setenv("RAGQA_INDEX_DIR", str(tmp_path))
    import ragqa.api

    importlib.reload(ragqa.api)
    return TestClient(ragqa.api.app)


def upload(client, name, data):
    return client.post("/documents", files={"file": (name, data)})


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 0
    assert body["llm"] is None


def test_upload_pdf_and_ask_cites_page(client, pdf_bytes):
    res = upload(client, "guide.pdf", pdf_bytes)
    assert res.status_code == 201
    assert res.json() == {"source": "guide.pdf", "pages": 2, "chunks": 2, "replaced": False}

    body = client.post("/ask", json={"question": "How much faster is int8 quantization on CPU?"}).json()
    assert body["mode"] == "extractive"
    assert "3x faster" in body["answer"]
    assert body["sources"][0]["source"] == "guide.pdf"
    assert body["sources"][0]["page"] == 2


def test_reupload_replaces_document(client):
    upload(client, "a.md", b"old text about retrieval")
    assert upload(client, "a.md", b"new text about retrieval").json()["replaced"] is True
    assert client.get("/documents").json() == ["a.md"]
    assert client.get("/health").json()["chunks"] == 1


def test_index_survives_restart(client, tmp_path):
    upload(client, "a.md", b"persistent text about containers")
    import ragqa.api

    importlib.reload(ragqa.api)  # simulates restarting the server
    assert TestClient(ragqa.api.app).get("/documents").json() == ["a.md"]


def test_unsupported_file_is_rejected(client):
    res = upload(client, "virus.exe", b"x")
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]


def test_too_large_file_is_rejected(client, monkeypatch):
    import ragqa.api
    from ragqa.config import Settings

    monkeypatch.setattr(ragqa.api, "settings", Settings(max_upload_mb=0))
    assert upload(client, "big.txt", b"x" * 10).status_code == 413


def test_question_validation(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 0}).status_code == 422


def test_ask_with_empty_index(client):
    body = client.post("/ask", json={"question": "anything at all?"}).json()
    assert body["sources"] == []
    assert "No relevant passages" in body["answer"]


def test_delete_document(client):
    upload(client, "a.txt", b"some text about retrieval")
    assert client.delete("/documents/a.txt").json() == {"source": "a.txt", "removed_chunks": 1}
    assert client.delete("/documents/a.txt").status_code == 404
