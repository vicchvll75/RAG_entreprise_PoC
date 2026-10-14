"""API de bout en bout : démarrage (construction de l'index), interface, RBAC, rate limit."""

import pytest
from fastapi.testclient import TestClient

import rag.api as api
import rag.pipeline as pipeline_mod
from rag.config import get_settings
from tests.test_pipeline import FakeChat
from tests.test_rbac import FakeEmbedder


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    monkeypatch.setattr(api, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(pipeline_mod, "get_chat_client", lambda: FakeChat("Le mardi [INT-001]."))
    monkeypatch.setattr(api, "limiter", api.RateLimiter(3, 600))
    with TestClient(api.app) as c:
        yield c
    get_settings.cache_clear()


def ask(client, question="Quel jour la présence sur site est-elle obligatoire ?", role="employee"):
    return client.post("/ask", json={"question": question}, headers={"X-User-Role": role})


def test_index_page_and_examples(client):
    assert "Altiscope" in client.get("/").text
    assert {e["type"] for e in client.get("/examples").json()} == {
        "answerable", "forbidden", "unanswerable"}


def test_ask_returns_answer_without_raw_contexts(client):
    data = ask(client).json()
    assert data["answer"] == "Le mardi [INT-001]." and "contexts" not in data


def test_unknown_role_is_forbidden(client):
    assert ask(client, role="admin").status_code == 403


def test_question_length_is_bounded(client):
    assert ask(client, question="x" * 501).status_code == 422


def test_rate_limit(client):
    assert [ask(client).status_code for _ in range(4)] == [200, 200, 200, 429]


def test_llm_provider_error_returns_readable_json(client, monkeypatch):
    import httpx
    from openai import AuthenticationError

    def boom(*_, **__):
        req = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
        raise AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)

    monkeypatch.setattr(api.state["pipeline"], "ask", boom)
    r = ask(client)
    assert r.status_code == 502 and "clé API" in r.json()["detail"]
