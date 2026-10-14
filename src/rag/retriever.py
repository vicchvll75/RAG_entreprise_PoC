"""Indexation Chroma et recherche hybride (dense + BM25) avec pré-filtrage RBAC."""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import chromadb

from rag.config import Settings, get_settings
from rag.documents import chunk_document, load_corpus
from rag.llm import Embedder
from rag.rbac import chroma_filter, role_flag, validate_role

RRF_K = 60
STOPWORDS = set(
    "le la les un une des de du d l et ou a au aux en dans par pour sur avec sans que qui quoi "
    "quel quelle quels quelles est sont ce cette ces se sa son ses il elle on nous vous ils "
    "y ne pas plus combien comment faut peut".split()
)


@dataclass
class RetrievedChunk:
    id: str
    doc_id: str
    text: str
    metadata: dict
    score: float


def get_client(settings: Settings) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def build_index(embedder: Embedder, settings: Settings | None = None,
                client: chromadb.ClientAPI | None = None) -> int:
    settings = settings or get_settings()
    client = client or get_client(settings)
    chunks = [c for doc in load_corpus(settings.corpus_dir) for c in chunk_document(doc)]
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        settings.collection_name,
        metadata={"embedder": embedder.name, "hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embedder.embed([c.text for c in chunks]),
    )
    return len(chunks)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            out.append(s)
        return out


class HybridRetriever:
    """Fusionne recherche dense et BM25 par Reciprocal Rank Fusion.

    Le dense capte les reformulations, BM25 les termes exacts (codes, montants, SEV1,
    noms d'outils) que les embeddings ratent souvent. Les deux index sont filtrés par rôle
    avant le classement.
    """

    def __init__(self, embedder: Embedder, settings: Settings | None = None,
                 client: chromadb.ClientAPI | None = None):
        self.settings = settings or get_settings()
        self.embedder = embedder
        client = client or get_client(self.settings)
        self.collection = client.get_collection(self.settings.collection_name)
        indexed_with = (self.collection.metadata or {}).get("embedder")
        if indexed_with != embedder.name:
            raise RuntimeError(
                f"Index construit avec {indexed_with}, requête avec {embedder.name} : "
                "relancer scripts/ingest.py"
            )
        data = self.collection.get(include=["documents", "metadatas"])
        self.ids, self.texts, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.bm25 = BM25([tokenize(t) for t in self.texts])

    def _dense(self, query: str, role: str, n: int) -> list[str]:
        res = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=n,
            where=chroma_filter(role),
        )
        return res["ids"][0]

    def _sparse(self, query: str, role: str, n: int) -> list[str]:
        flag = role_flag(role)
        scored = [
            (s, cid) for s, cid, m in zip(self.bm25.scores(tokenize(query)), self.ids, self.metas)
            if m.get(flag) is True and s > 0
        ]
        return [cid for _, cid in sorted(scored, reverse=True)[:n]]

    def retrieve(self, query: str, role: str, k: int | None = None) -> list[RetrievedChunk]:
        validate_role(role)
        k = k or self.settings.top_k
        n = self.settings.candidates_per_retriever
        fused: dict[str, float] = {}
        for ranking in (self._dense(query, role, n), self._sparse(query, role, n)):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1 / (RRF_K + rank + 1)
        index = {cid: i for i, cid in enumerate(self.ids)}
        results = []
        for cid, score in sorted(fused.items(), key=lambda x: -x[1])[:k]:
            i = index[cid]
            # Défense en profondeur : on revérifie le droit d'accès après la fusion.
            if self.metas[i].get(role_flag(role)) is not True:
                raise PermissionError(f"Chunk {cid} non autorisé pour {role}")
            results.append(RetrievedChunk(cid, self.metas[i]["doc_id"], self.texts[i],
                                          self.metas[i], score))
        return results
