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
    assert res.json() == {"source": "guide.pdf", "pages": 2, "chunks": 2, "replaced": False, "ocr_pages": []}

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


def read_events(response) -> list[tuple[str, object]]:
    """Parse a Server-Sent Events body into (event, data) pairs."""
    import json

    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_stream_extractive_without_llm(client, pdf_bytes):
    upload(client, "guide.pdf", pdf_bytes)
    res = client.post("/ask/stream", json={"question": "How much faster is int8 quantization on CPU?"})
    assert res.headers["content-type"].startswith("text/event-stream")
    events = read_events(res)
    assert [e for e, _ in events] == ["sources", "token", "done"]
    assert events[0][1][0]["page"] == 2
    assert "3x faster" in events[1][1]["text"]
    assert events[2][1] == {"mode": "extractive"}


def test_stream_with_llm_sends_tokens(client, monkeypatch):
    import ragqa.api

    class FakeGenerator:
        def stream(self, question, contexts):
            yield from ["Containers ", "package apps ", "[1]."]

    monkeypatch.setattr(ragqa.api, "generator", FakeGenerator())
    upload(client, "docker.md", b"Docker packages an application into a portable container image.")
    events = read_events(client.post("/ask/stream", json={"question": "What does Docker do?"}))
    assert "".join(d["text"] for e, d in events if e == "token") == "Containers package apps [1]."
    assert events[-1] == ("done", {"mode": "llm"})


def test_stream_falls_back_when_llm_is_unreachable(client, monkeypatch):
    import anthropic
    import httpx
    import ragqa.api

    class BrokenGenerator:
        def stream(self, question, contexts):
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
            yield  # makes this a generator

    monkeypatch.setattr(ragqa.api, "generator", BrokenGenerator())
    upload(client, "docker.md", b"Docker packages an application into a portable container image.")
    events = read_events(client.post("/ask/stream", json={"question": "What does Docker do?"}))
    assert events[-1] == ("done", {"mode": "extractive"})
    assert "portable container image" in events[-2][1]["text"]


def test_web_ui_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "/ask/stream" in res.text


def test_upload_scanned_pdf_reports_ocr_pages(client, scanned_pdf):
    body = upload(client, "scan.pdf", scanned_pdf).json()
    assert body["ocr_pages"] == [1]
    answer = client.post("/ask", json={"question": "How much faster was CPU inference?"}).json()
    assert answer["sources"][0]["source"] == "scan.pdf" and answer["sources"][0]["page"] == 1
