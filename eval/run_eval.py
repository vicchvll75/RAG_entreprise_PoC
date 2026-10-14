"""Évaluation complète du RAG sur le jeu golden.

    python eval/run_eval.py                    # tout : retrieval + génération + Ragas
    python eval/run_eval.py --retrieval-only   # sans LLM, gratuit (CI à chaque commit)
    python eval/run_eval.py --no-ragas         # génération + refus + fuites, sans juge LLM

Sortie : eval/results/<horodatage>.json et .md. Code retour 1 si un seuil n'est pas tenu.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.metrics import Confusion, leaked_strings, mean, retrieval_scores  # noqa: E402
from rag.config import get_settings  # noqa: E402
from rag.documents import load_corpus  # noqa: E402
from rag.llm import get_embedder  # noqa: E402
from rag.rbac import can_access  # noqa: E402
from rag.retriever import HybridRetriever, build_index  # noqa: E402

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
RESULTS = ROOT / "eval" / "results"

# Seuils du quality gate. Les fuites sont tolérance zéro : une seule suffit à échouer.
THRESHOLDS = {
    "retrieval.recall@k": 0.90,
    "retrieval.mrr": 0.80,
    "refusal.f1": 0.85,
    "ragas.faithfulness": 0.85,
    "ragas.answer_relevancy": 0.75,
    "ragas.context_recall": 0.80,
    "ragas.factual_correctness_f1": 0.60,
}
ZERO_TOLERANCE = ["rbac.retrieval_violations", "rbac.leaked_answers"]


def load_golden(path: Path, docs: dict) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for it in items:
        # Garde-fou sur le jeu lui-même : une question « répondable » doit pointer vers des
        # documents que le rôle peut lire, sinon la métrique mesurerait une erreur d'annotation.
        for d in it["expected_doc_ids"]:
            if d not in docs:
                raise ValueError(f"{it['id']}: document inconnu {d}")
            if not can_access(it["role"], docs[d].allowed_roles):
                raise ValueError(f"{it['id']}: {it['role']} n'a pas accès à {d}")
        if it["type"] == "answerable" and not it["expected_doc_ids"]:
            raise ValueError(f"{it['id']}: question répondable sans document attendu")
    return items


def build_ragas_metrics(settings):
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
        FactualCorrectness,
    )

    from rag.llm import get_async_client

    client = get_async_client()
    llm = llm_factory(settings.eval_model, client=client, temperature=0)
    embeddings = embedding_factory("openai", model=settings.embed_model, client=client)
    return {
        # Anti-hallucination : part des affirmations de la réponse soutenues par le contexte.
        "faithfulness": Faithfulness(llm=llm),
        # La réponse traite-t-elle la question posée (et pas une voisine) ?
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        # Les chunks pertinents sont-ils classés en tête ?
        "context_precision": ContextPrecisionWithReference(llm=llm),
        # Le contexte contient-il tout ce qu'il faut pour la réponse de référence ?
        "context_recall": ContextRecall(llm=llm),
        # Précision / rappel / F1 des affirmations de la réponse vs la référence.
        "factual_correctness_f1": FactualCorrectness(llm=llm, mode="f1"),
    }


async def score_ragas(rows: list[dict], metrics: dict, concurrency: int = 4) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def run(row: dict, name: str, metric, **kwargs) -> None:
        async with sem:
            try:
                row["ragas"][name] = float((await metric.ascore(**kwargs)).value)
            except Exception as exc:  # un échec du juge ne doit pas masquer les autres scores
                row["ragas"][name] = None
                row.setdefault("ragas_errors", {})[name] = repr(exc)[:300]

    tasks = []
    for row in rows:
        if row["type"] != "answerable":
            continue
        row["ragas"] = {}
        q, ctx, ref, ans = row["question"], row["contexts"], row["reference"], row["answer"]
        tasks += [
            run(row, "context_precision", metrics["context_precision"],
                user_input=q, reference=ref, retrieved_contexts=ctx),
            run(row, "context_recall", metrics["context_recall"],
                user_input=q, retrieved_contexts=ctx, reference=ref),
            # Un refus à tort compte comme réponse fausse (F1 = 0) plutôt que d'être ignoré.
            run(row, "factual_correctness_f1", metrics["factual_correctness_f1"],
                response=ans, reference=ref),
        ]
        # Fidélité et pertinence n'ont de sens que si le système a effectivement répondu ;
        # les refus sont déjà comptés par la métrique de refus.
        if not row["refused"]:
            tasks += [
                run(row, "faithfulness", metrics["faithfulness"],
                    user_input=q, response=ans, retrieved_contexts=ctx),
                run(row, "answer_relevancy", metrics["answer_relevancy"],
                    user_input=q, response=ans),
            ]
    await asyncio.gather(*tasks)


def summarize(rows: list[dict], k: int, with_generation: bool, with_ragas: bool) -> dict:
    answerable = [r for r in rows if r["type"] == "answerable"]
    summary = {"n_questions": len(rows), "k": k, "retrieval": {}, "rbac": {}}
    for m in ("precision@k", "recall@k", "f1@k", "hit@k", "mrr"):
        summary["retrieval"][m] = mean([r["retrieval"][m] for r in answerable])
    summary["rbac"]["retrieval_violations"] = sum(len(r["rbac_violations"]) for r in rows)

    if with_generation:
        conf = Confusion()
        for r in rows:
            conf.add(r["type"] != "answerable", r["refused"])
        summary["refusal"] = conf.scores()
        summary["rbac"]["leaked_answers"] = sum(bool(r["leaks"]) for r in rows)
        summary["guardrail_blocks"] = sum(bool(r["guardrails"].get("reason")) for r in rows)
        lat = sorted(r["latency_ms"] for r in rows)
        summary["latency_ms"] = {"p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95) - 1]}
        summary["tokens"] = {
            "prompt": sum(r["usage"].get("prompt_tokens", 0) for r in rows),
            "completion": sum(r["usage"].get("completion_tokens", 0) for r in rows),
        }
        by_type = {}
        for t in ("answerable", "forbidden", "unanswerable"):
            sub = [r for r in rows if r["type"] == t]
            by_type[t] = {"n": len(sub), "refused": sum(r["refused"] for r in sub)}
        summary["by_type"] = by_type

    if with_ragas:
        names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall",
                 "factual_correctness_f1"]
        summary["ragas"] = {
            n: mean([r.get("ragas", {}).get(n) for r in answerable]) for n in names
        }
        summary["ragas_errors"] = sum(len(r.get("ragas_errors", {})) for r in rows)
    return summary


def check_gate(summary: dict) -> list[str]:
    failures = []
    for path, threshold in THRESHOLDS.items():
        section, metric = path.split(".", 1)
        value = summary.get(section, {}).get(metric)
        if value is not None and value < threshold:
            failures.append(f"{path} = {value:.3f} < {threshold}")
    for path in ZERO_TOLERANCE:
        section, metric = path.split(".", 1)
        value = summary.get(section, {}).get(metric)
        if value:
            failures.append(f"{path} = {value} (tolérance zéro)")
    return failures


def fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def to_markdown(summary: dict, rows: list[dict], failures: list[str], meta: dict) -> str:
    lines = [f"# Rapport d'évaluation RAG — {meta['timestamp']}", "",
             f"Embeddings : `{meta['embedder']}` · génération : `{meta['generation_model']}` · "
             f"juge : `{meta['judge_model']}` · k = {summary['k']}", "",
             f"**Quality gate : {'ÉCHEC' if failures else 'OK'}**", ""]
    lines += [f"- {f}" for f in failures] + ([""] if failures else [])
    for section in ("retrieval", "refusal", "ragas", "rbac", "latency_ms", "by_type"):
        if section not in summary:
            continue
        lines += [f"## {section}", "", "| métrique | valeur |", "|---|---|"]
        for name, value in summary[section].items():
            lines.append(f"| {name} | {fmt(value) if not isinstance(value, dict) else value} |")
        lines.append("")

    lines += ["## Détail par question", "",
              "| id | type | rôle | docs remontés | P | R | refus | fuite | fidélité | F1 faits |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rag = r.get("ragas", {})
        lines.append(
            f"| {r['id']} | {r['type']} | {r['role']} | {', '.join(r['retrieved_docs'])} | "
            f"{fmt(r['retrieval'].get('precision@k'))} | {fmt(r['retrieval'].get('recall@k'))} | "
            f"{fmt(r.get('refused'))} | {', '.join(r.get('leaks', [])) or '—'} | "
            f"{fmt(rag.get('faithfulness'))} | {fmt(rag.get('factual_correctness_f1'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--only", nargs="*", help="identifiants de questions à évaluer")
    args = parser.parse_args()

    settings = get_settings()
    docs = {d.id: d for d in load_corpus(settings.corpus_dir)}
    items = load_golden(GOLDEN, docs)
    if args.only:
        items = [it for it in items if it["id"] in args.only]

    embedder = get_embedder(settings)
    if args.reindex:
        print(f"Indexation : {build_index(embedder, settings)} chunks")
    retriever = HybridRetriever(embedder, settings)
    with_generation = not args.retrieval_only
    with_ragas = with_generation and not args.no_ragas

    pipeline = None
    if with_generation:
        from rag.pipeline import RAGPipeline
        pipeline = RAGPipeline(retriever, settings)

    k = settings.top_k
    canaries = {d.canary: d.allowed_roles for d in docs.values() if d.canary}
    rows = []
    for it in items:
        row = {**it, "retrieval": {}, "guardrails": {}, "usage": {}}
        if pipeline:
            ans = pipeline.ask(it["question"], it["role"])
            row.update(answer=ans.answer, refused=ans.refused, contexts=ans.contexts,
                       retrieved_docs=ans.sources, guardrails=ans.guardrails,
                       latency_ms=ans.latency_ms, usage=ans.usage)
            forbidden = it.get("forbidden_strings", []) + [
                c for c, roles in canaries.items() if not can_access(it["role"], roles)]
            # On contrôle aussi la réponse bloquée : le garde-fou a pu attraper la fuite, mais
            # on veut savoir que le modèle l'a produite.
            row["leaks"] = leaked_strings(ans.answer, forbidden)
            row["blocked_leaks"] = leaked_strings(ans.guardrails.get("blocked_answer", ""),
                                                  forbidden)
            print(f"{it['id']:>4} {'REFUS' if ans.refused else 'OK   '} "
                  f"{ans.latency_ms:>5} ms  {ans.answer[:90]!r}")
        else:
            chunks = retriever.retrieve(it["question"], it["role"], k)
            row["retrieved_docs"] = list(dict.fromkeys(c.doc_id for c in chunks))

        row["rbac_violations"] = [d for d in row["retrieved_docs"]
                                  if not can_access(it["role"], docs[d].allowed_roles)]
        if it["type"] == "answerable":
            row["retrieval"] = retrieval_scores(row["retrieved_docs"],
                                                set(it["expected_doc_ids"]), k)
        rows.append(row)

    if with_ragas:
        print("Scoring Ragas…")
        t0 = time.perf_counter()
        asyncio.run(score_ragas(rows, build_ragas_metrics(settings)))
        print(f"Ragas : {time.perf_counter() - t0:.0f} s")

    summary = summarize(rows, k, with_generation, with_ragas)
    failures = check_gate(summary)
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedder": embedder.name,
        "generation_model": settings.gen_model if with_generation else None,
        "judge_model": settings.eval_model if with_ragas else None,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "gate_failures": failures, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / f"{stem}.md").write_text(to_markdown(summary, rows, failures, meta),
                                        encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRapport : eval/results/{stem}.md")
    print("QUALITY GATE :", "ÉCHEC\n  " + "\n  ".join(failures) if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
