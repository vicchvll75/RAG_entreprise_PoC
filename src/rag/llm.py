"""Clients LLM et embeddings.

Mistral et OpenAI exposent tous deux une API compatible OpenAI : un seul SDK suffit, le
fournisseur se change via LLM_PROVIDER sans toucher au code.
"""

from functools import lru_cache
from typing import Protocol

from openai import AsyncOpenAI, OpenAI

from rag.config import Settings, get_settings

# Les offres gratuites (Mistral « Experiment ») limitent à ~1 requête/s : le SDK réessaie les 429
# avec une attente exponentielle. 2 tentatives par défaut ne suffisent pas à absorber une rafale.
MAX_RETRIES = 6


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ApiEmbedder:
    def __init__(self, settings: Settings):
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                             max_retries=MAX_RETRIES)
        self.model = settings.embed_model
        self.name = f"{settings.llm_provider}:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 64):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + 64])
            out.extend(d.embedding for d in resp.data)
        return out


class LocalEmbedder:
    """MiniLM ONNX embarqué dans Chroma : sans clé API, utile hors-ligne et en CI.

    Modèle anglais, donc moins bon sur le corpus français que mistral-embed ; la
    composante BM25 de la recherche hybride compense en partie.
    """

    name = "local:all-MiniLM-L6-v2"

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedding_backend == "local":
        return LocalEmbedder()
    return ApiEmbedder(settings)


@lru_cache
def get_chat_client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.api_key, base_url=s.base_url, max_retries=MAX_RETRIES)


def get_async_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(api_key=s.api_key, base_url=s.base_url, max_retries=MAX_RETRIES)
