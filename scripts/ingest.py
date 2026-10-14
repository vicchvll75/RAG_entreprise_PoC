"""Construit l'index Chroma à partir du corpus. Usage : python scripts/ingest.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.llm import get_embedder  # noqa: E402
from rag.retriever import build_index  # noqa: E402

if __name__ == "__main__":
    embedder = get_embedder()
    n = build_index(embedder)
    print(f"{n} chunks indexés avec {embedder.name}")
