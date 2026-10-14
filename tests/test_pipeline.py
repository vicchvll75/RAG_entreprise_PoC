"""Garde-fous de sortie, testés avec un LLM simulé (pas d'appel réseau)."""

from types import SimpleNamespace

import chromadb
import pytest

import rag.pipeline as pipeline_mod
from rag.config import Settings
from rag.pipeline import REFUSAL, RAGPipeline
from rag.retriever import HybridRetriever, build_index
from tests.test_rbac import FakeEmbedder


class FakeChat:
    def __init__(self, reply: str):
        self.reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    settings = Settings(chroma_dir=tmp_path_factory.mktemp("chroma"), collection_name="pipe")
    client = chromadb.EphemeralClient()
    build_index(FakeEmbedder(), settings, client)
    return HybridRetriever(FakeEmbedder(), settings, client)


def make_pipeline(retriever, monkeypatch, reply):
    monkeypatch.setattr(pipeline_mod, "get_chat_client", lambda: FakeChat(reply))
    return RAGPipeline(retriever, retriever.settings)


def test_grounded_answer_passes(retriever, monkeypatch):
    p = make_pipeline(retriever, monkeypatch, "Le mardi [INT-001].")
    ans = p.ask("Quel jour la présence sur site est-elle obligatoire ? télétravail", "employee")
    assert not ans.refused and "INT-001" in ans.sources
    assert ans.guardrails["citations"] == ["INT-001"]


def test_refusal_is_detected(retriever, monkeypatch):
    p = make_pipeline(retriever, monkeypatch, REFUSAL)
    assert p.ask("Cours de l'action ?", "employee").refused


def test_canary_leak_is_blocked(retriever, monkeypatch):
    p = make_pipeline(retriever, monkeypatch, "Le code est RH-SAL-26-K7Q2 [INT-005].")
    ans = p.ask("grille des salaires", "employee")
    assert ans.refused and ans.answer == REFUSAL
    assert ans.guardrails["reason"] == "canary_leak"


def test_canary_allowed_for_authorized_role(retriever, monkeypatch):
    p = make_pipeline(retriever, monkeypatch, "Code RH-SAL-26-K7Q2 [HR-001].")
    ans = p.ask("Grille des salaires 2026 code de suivi RH-SAL-26-K7Q2", "hr")
    assert not ans.refused


def test_citation_of_unretrieved_document_is_blocked(retriever, monkeypatch):
    # Le modèle cite un document qu'il n'a pas pu voir : signe d'hallucination.
    p = make_pipeline(retriever, monkeypatch, "La valorisation est de 27 M€ [EXE-001].")
    ans = p.ask("Quel jour la présence sur site est-elle obligatoire ?", "employee")
    assert ans.refused and ans.guardrails["reason"] == "invalid_citation"
