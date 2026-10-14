"""Métriques déterministes (sans LLM) : retrieval, refus, fuite RBAC.

Ces métriques sont gratuites, reproductibles et exactes : elles tournent à chaque commit.
Les métriques Ragas (juge LLM) complètent sur ce qu'elles ne savent pas mesurer : la
fidélité de la réponse au contexte et sa justesse sémantique.
"""

from dataclasses import dataclass


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Part des documents remontés (top-k, dédoublonnés) qui sont pertinents."""
    top = dedupe(retrieved)[:k]
    return sum(d in relevant for d in top) / len(top) if top else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Part des documents pertinents retrouvés dans le top-k."""
    if not relevant:
        raise ValueError("recall indéfini sans document pertinent")
    return len(set(dedupe(retrieved)[:k]) & relevant) / len(relevant)


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, d in enumerate(dedupe(retrieved), start=1):
        if d in relevant:
            return 1 / rank
    return 0.0


def retrieval_scores(retrieved: list[str], relevant: set[str], k: int) -> dict:
    p, r = precision_at_k(retrieved, relevant, k), recall_at_k(retrieved, relevant, k)
    return {
        "precision@k": p,
        "recall@k": r,
        "f1@k": f1(p, r),
        "hit@k": float(r > 0),
        "mrr": reciprocal_rank(retrieved[:k], relevant),
    }


@dataclass
class Confusion:
    """Classification binaire ; classe positive = « le système doit refuser »."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, should_refuse: bool, refused: bool) -> None:
        if should_refuse and refused:
            self.tp += 1
        elif refused:
            self.fp += 1
        elif should_refuse:
            self.fn += 1
        else:
            self.tn += 1

    def scores(self) -> dict:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        total = self.tp + self.fp + self.fn + self.tn
        return {
            "precision": precision,  # bas = le système refuse à tort (trop prudent)
            "recall": recall,        # bas = le système répond alors qu'il devrait refuser
            "f1": f1(precision, recall),
            "accuracy": (self.tp + self.tn) / total if total else 0.0,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def leaked_strings(answer: str, forbidden: list[str]) -> list[str]:
    norm = answer.replace(" ", " ").replace("\xa0", " ").lower()
    return [s for s in forbidden if s.lower() in norm]


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and v == v]  # écarte None et NaN
    return sum(values) / len(values) if values else None
