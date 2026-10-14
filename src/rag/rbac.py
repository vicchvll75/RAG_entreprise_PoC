"""Contrôle d'accès par rôle (RBAC).

Le filtrage est appliqué AVANT la recherche (pré-filtre dans la base vectorielle et dans
l'index BM25) : un chunk non autorisé n'atteint jamais le LLM, il ne peut donc pas fuiter
dans la réponse, même paraphrasé. Refus par défaut : un chunk sans drapeau de rôle est
invisible pour tout le monde.
"""

ROLES = ("employee", "engineering", "hr", "finance", "executive")


def role_flag(role: str) -> str:
    return f"role_{role}"


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"Rôle inconnu : {role!r} (attendus : {', '.join(ROLES)})")
    return role


def can_access(role: str, allowed_roles: list[str]) -> bool:
    return role in allowed_roles


def chroma_filter(role: str) -> dict:
    return {role_flag(validate_role(role)): True}
