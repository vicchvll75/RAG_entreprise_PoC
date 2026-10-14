"""API HTTP. En production, le rôle viendrait d'un JWT validé (SSO), pas d'un header libre."""

from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from rag.llm import get_embedder
from rag.pipeline import RAGPipeline
from rag.rbac import ROLES
from rag.retriever import HybridRetriever

app = FastAPI(title="Altiscope RAG")


class AskRequest(BaseModel):
    question: str


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(HybridRetriever(get_embedder()))


@app.post("/ask")
def ask(req: AskRequest, x_user_role: str = Header(...)) -> dict:
    if x_user_role not in ROLES:
        raise HTTPException(403, "Rôle inconnu")
    result = get_pipeline().ask(req.question, x_user_role)
    # Les contextes bruts ne sont pas renvoyés au client, seulement les identifiants.
    return {k: v for k, v in result.to_dict().items() if k != "contexts"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
