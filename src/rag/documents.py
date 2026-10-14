"""Chargement du corpus Markdown (frontmatter YAML) et découpage par section."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.rbac import ROLES, role_flag

FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
CANARY = re.compile(r"Code de suivi du document : \*\*([A-Z0-9-]+)\*\*")


@dataclass
class Document:
    id: str
    title: str
    classification: str
    allowed_roles: list[str]
    body: str
    path: str
    canary: str | None = None


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = FRONTMATTER.match(raw)
    if not match:
        raise ValueError(f"Frontmatter manquant : {path}")
    meta = yaml.safe_load(match.group(1))
    unknown = set(meta["allowed_roles"]) - set(ROLES)
    if unknown:
        raise ValueError(f"{path}: rôles inconnus {unknown}")
    canary = CANARY.search(match.group(2))
    return Document(
        id=meta["id"],
        title=meta["title"],
        classification=meta["classification"],
        allowed_roles=list(meta["allowed_roles"]),
        body=match.group(2).strip(),
        path=path.as_posix(),
        canary=canary.group(1) if canary else None,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    # Exclut les dossiers techniques (ex. graphify-out/) qui ne sont pas du contenu métier.
    paths = sorted(
        p for p in corpus_dir.rglob("*.md")
        if not any(part.startswith((".", "graphify")) for part in p.relative_to(corpus_dir).parts)
    )
    return [load_document(p) for p in paths]


def chunk_document(doc: Document) -> list[Chunk]:
    """Un chunk par section `##`, préfixé du titre et de l'identifiant du document.

    Les documents sont courts et bien structurés : la section est l'unité de sens naturelle,
    plus robuste qu'un découpage à taille fixe qui couperait les tableaux en deux.
    """
    sections = re.split(r"\n(?=## )", doc.body)
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        heading = section.splitlines()[0].lstrip("# ").strip()
        content = "\n".join(section.splitlines()[1:]).strip()
        if not content:
            continue
        metadata = {
            "doc_id": doc.id,
            "title": doc.title,
            "section": heading,
            "classification": doc.classification,
            "source": doc.path,
        }
        metadata.update({role_flag(r): r in doc.allowed_roles for r in ROLES})
        chunks.append(Chunk(
            id=f"{doc.id}#{i}",
            text=f"[{doc.id}] {doc.title} — {heading}\n{content}",
            metadata=metadata,
        ))
    return chunks
