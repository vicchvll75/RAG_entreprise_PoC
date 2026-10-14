import pytest

from eval.metrics import Confusion, f1, leaked_strings, mean, retrieval_scores


def test_retrieval_scores_dedupes_chunks_of_same_doc():
    s = retrieval_scores(["A", "A", "B", "C"], {"A", "D"}, k=5)
    assert s["precision@k"] == pytest.approx(1 / 3)
    assert s["recall@k"] == pytest.approx(1 / 2)
    assert s["f1@k"] == pytest.approx(f1(1 / 3, 1 / 2))
    assert s["mrr"] == 1.0


def test_mrr_uses_rank_of_first_relevant_doc():
    assert retrieval_scores(["X", "Y", "A"], {"A"}, k=5)["mrr"] == pytest.approx(1 / 3)


def test_recall_requires_relevant_docs():
    with pytest.raises(ValueError):
        retrieval_scores(["A"], set(), k=5)


def test_refusal_confusion():
    c = Confusion()
    c.add(should_refuse=True, refused=True)    # tp
    c.add(should_refuse=True, refused=False)   # fn : a répondu au lieu de refuser
    c.add(should_refuse=False, refused=True)   # fp : refus à tort
    c.add(should_refuse=False, refused=False)  # tn
    s = c.scores()
    assert (s["precision"], s["recall"], s["f1"], s["accuracy"]) == (0.5, 0.5, 0.5, 0.5)


def test_leaked_strings_ignores_case_and_narrow_spaces():
    assert leaked_strings("Médiane : 68 000 €", ["68 000", "76 000"]) == ["68 000"]


def test_mean_skips_missing_values():
    assert mean([1.0, None, float("nan"), 0.0]) == 0.5
    assert mean([]) is None
