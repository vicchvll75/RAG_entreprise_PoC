"""Aucune question, quelle que soit sa formulation, ne doit remonter un document interdit."""

import hashlib
import json
from pathlib import Path

import chromadb
import pytest

from rag.config import Settings
from rag.documents import load_corpus
from rag.rbac import ROLES, can_access
from rag.retriever import HybridRetriever, build_index

ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    """Embedding déterministe sans réseau : on teste le filtrage, pas la qualité sémantique."""
    name = "fake"

    def embed(self, texts):
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()] for t in texts]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    settings = Settings(chroma_dir=tmp_path_factory.mktemp("chroma"), collection_name="test")
    client = chromadb.EphemeralClient()
    build_index(FakeEmbedder(), settings, client)
    docs = {d.id: d for d in load_corpus(settings.corpus_dir)}
    return HybridRetriever(FakeEmbedder(), settings, client), docs


QUESTIONS = [json.loads(line)["question"] for line in
             (ROOT / "eval" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("role", ROLES)
def test_no_forbidden_document_is_ever_retrieved(setup, role):
    retriever, docs = setup
    for q in QUESTIONS:
        for chunk in retriever.retrieve(q, role, k=20):
            assert can_access(role, docs[chunk.doc_id].allowed_roles), (role, q, chunk.id)


def test_restricted_doc_only_visible_to_its_roles(setup):
    retriever, _ = setup
    q = "Projet Helix valorisation Datavolt EXE-HLX-26-R4T6"
    assert any(c.doc_id == "EXE-001" for c in retriever.retrieve(q, "executive"))
    for role in ("employee", "engineering", "hr", "finance"):
        assert all(c.doc_id != "EXE-001" for c in retriever.retrieve(q, role, k=20))


def test_unknown_role_is_rejected(setup):
    retriever, _ = setup
    with pytest.raises(ValueError):
        retriever.retrieve("salaires", "admin")


def test_every_document_declares_known_roles_and_is_chunked():
    from rag.documents import chunk_document
    for doc in load_corpus(ROOT / "corpus"):
        assert doc.allowed_roles and set(doc.allowed_roles) <= set(ROLES)
        assert chunk_document(doc)
