"""API HTTP + interface de démo.

En production, le rôle viendrait d'un JWT validé (SSO), pas d'un header libre : ici le choix du
rôle est volontairement laissé à l'utilisateur pour montrer l'effet du RBAC.
"""

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from openai import APIError, APIStatusError
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.config import ROOT, get_settings
from rag.llm import get_embedder
from rag.pipeline import RAGPipeline
from rag.rbac import ROLES
from rag.retriever import HybridRetriever, ensure_index

STATIC = Path(__file__).parent / "static"
GOLDEN = ROOT / "eval" / "golden_set.jsonl"

log = logging.getLogger("uvicorn.error")
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    embedder = get_embedder()
    ensure_index(embedder)
    state["pipeline"] = RAGPipeline(HybridRetriever(embedder))
    yield


app = FastAPI(title="Altiscope RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class RateLimiter:
    """Fenêtre glissante par IP, en mémoire : suffisant pour une démo sur une instance.

    L'endpoint est public et chaque question coûte des appels LLM payants.
    """

    def __init__(self, max_requests: int, window_s: int):
        self.max_requests, self.window_s = max_requests, window_s
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.max_requests:
            raise HTTPException(429, "Trop de requêtes, réessayez dans quelques minutes.")
        q.append(now)


limiter = RateLimiter(get_settings().rate_limit_per_10min, 600)


def client_ip(request: Request) -> str:
    # Derrière le proxy de Hugging Face, l'IP réelle est le premier élément du header.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


@app.post("/ask")
def ask(req: AskRequest, request: Request, x_user_role: str = Header(...)) -> dict:
    if x_user_role not in ROLES:
        raise HTTPException(403, "Rôle inconnu")
    limiter.check(client_ip(request))
    try:
        result = state["pipeline"].ask(req.question, x_user_role)
    except APIStatusError as exc:
        log.error("Erreur du fournisseur LLM %s : %s", exc.status_code, exc.message)
        detail = {401: "clé API refusée par le fournisseur LLM",
                  429: "quota du fournisseur LLM atteint, réessayez dans un instant"}
        raise HTTPException(502, detail.get(exc.status_code,
                                            f"erreur du fournisseur LLM ({exc.status_code})"))
    except APIError as exc:
        log.error("Fournisseur LLM injoignable : %r", exc)
        raise HTTPException(502, "fournisseur LLM injoignable")
    except Exception:
        log.exception("Erreur sur /ask")
        raise HTTPException(500, "erreur interne, voir les logs du serveur")
    # Les contextes bruts et la réponse bloquée ne sont jamais renvoyés au client.
    out = {k: v for k, v in result.to_dict().items() if k != "contexts"}
    out["guardrails"] = {k: v for k, v in result.guardrails.items() if k != "blocked_answer"}
    return out


@app.get("/examples")
def examples() -> list[dict]:
    items = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]
    picked = ["A02", "A14", "A25", "A26", "F01", "F08", "U04"]
    return [{"id": i["id"], "type": i["type"], "role": i["role"], "question": i["question"]}
            for i in items if i["id"] in picked]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
