"""Retrieval benchmark: Naive Vector Search vs Time-Filtered Hybrid + Rerank.

For each ground-truth event we build the AnomalyEvent from the detector's output for that trading
day, run all four retrieval modes with the same query, and score the top-k against the labelled
catalyst documents:
  MRR@k, Precision@k, Recall@k, Hit@k.
Note: with 2 relevant docs per event, Precision@5 is capped at 0.40.

Usage:  python -m src.eval.benchmark [--k 5] [--detected-only] [--no-prophet] [--time-filter datetime_range]
"""
from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np
import pandas as pd

from src.anomaly.detector import AnomalyDetector, DetectionResult, DetectorConfig
from src.config import CANDIDATE_K, EVAL_RESULTS_PATH, FINAL_K, TICKERS
from src.data.loaders import load_documents, load_ground_truth, load_prices
from src.rag.hybrid_retriever import HybridRetriever, RetrievalMode

MODES: dict[RetrievalMode, str] = {
    "naive": "Naive Vector Search (no filters)",
    "time_dense": "Time-Filtered Dense",
    "hybrid": "Time-Filtered Hybrid (BM25+Dense, RRF)",
    "hybrid_rerank": "Time-Filtered Hybrid + Cross-Encoder",
}


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in ranked[:k]) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(d in relevant for d in ranked[:k]) / len(relevant) if relevant else 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(d in relevant for d in ranked[:k]))


def bootstrap_ci(values: Sequence[float], n: int = 5000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _pct_gain(new: float, base: float) -> float | None:
    return None if base == 0 else (new - base) / base * 100.0


def run_benchmark(k: int = FINAL_K, candidate_k: int = CANDIDATE_K, detected_only: bool = False,
                  use_prophet: bool = True, time_filter: str = "int_range") -> dict:
    docs = load_documents()
    truth = load_ground_truth()
    retriever = HybridRetriever(docs, time_filter=time_filter)  # type: ignore[arg-type]
    detector = AnomalyDetector(DetectorConfig(use_prophet=use_prophet))
    detections: dict[str, DetectionResult] = {
        t: detector.analyze(load_prices(t), t) for t in TICKERS if any(g.ticker == t for g in truth)}

    per_event: list[dict] = []
    skipped: list[str] = []
    for g in truth:
        frame = detections[g.ticker].frame
        idx = frame.index[frame["date"] == pd.Timestamp(g.date)]
        if len(idx) == 0:
            skipped.append(g.event_id)
            continue
        row = frame.loc[idx[0]]
        detected = bool(row["is_anomaly"])
        if detected_only and not detected:
            continue
        event = detector.event_from_row(g.ticker, row)
        relevant = set(g.relevant_doc_ids)
        entry: dict = {"event_id": g.event_id, "detected": detected, "z_score": round(event.z_score, 2),
                       "pct_change": round(event.pct_change, 2), "modes": {}}
        for mode in MODES:
            res = retriever.retrieve(event, mode=mode, candidate_k=candidate_k, final_k=k)
            ranked = [c.doc.doc_id for c in res.top]
            entry["modes"][mode] = {
                "ranked": ranked, "mrr": reciprocal_rank(ranked, relevant, k),
                "precision": precision_at_k(ranked, relevant, k), "recall": recall_at_k(ranked, relevant, k),
                "hit": hit_at_k(ranked, relevant, k),
            }
        per_event.append(entry)

    if not per_event:
        raise RuntimeError("No events evaluated. Check that bootstrap data exists and dates align.")

    summary: dict[str, dict[str, float]] = {}
    for mode in MODES:
        summary[mode] = {m: float(np.mean([e["modes"][mode][m] for e in per_event]))
                         for m in ("mrr", "precision", "recall", "hit")}

    base, full = summary["naive"], summary["hybrid_rerank"]
    improvement: dict[str, float | tuple[float, float] | None] = {}
    for metric in ("mrr", "precision", "recall", "hit"):
        deltas = [
            e["modes"]["hybrid_rerank"][metric] - e["modes"]["naive"][metric]
            for e in per_event
        ]
        improvement[f"{metric}_abs_delta"] = full[metric] - base[metric]
        improvement[f"{metric}_rel_pct"] = _pct_gain(full[metric], base[metric])
        improvement[f"{metric}_delta_ci95"] = bootstrap_ci(deltas)
    out = {
        "k": k, "candidate_k": candidate_k, "time_filter": time_filter, "n_events": len(per_event),
        "n_corpus_docs": len(docs), "detected_events": int(sum(e["detected"] for e in per_event)),
        "skipped_events": skipped, "summary": summary, "improvement": improvement, "per_event": per_event,
    }
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def print_report(r: dict) -> None:
    k = r["k"]
    print(f"\nAlphaShock retrieval benchmark: {r['n_events']} events, {r['n_corpus_docs']} documents, k={k}")
    print(f"Detector flagged {r['detected_events']}/{r['n_events']} of these dates "
          f"(retrieval is scored on the ground-truth dates regardless)\n")
    print(f"{'Method':<42}{'MRR@'+str(k):>9}{'P@'+str(k):>8}{'R@'+str(k):>8}{'Hit@'+str(k):>8}")
    print("-" * 75)
    for mode, label in MODES.items():
        s = r["summary"][mode]
        print(f"{label:<42}{s['mrr']:>9.3f}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['hit']:>8.3f}")
    imp = r["improvement"]

    def fmt(rel: float | None, ab: float) -> str:
        return f"{ab:+.3f} abs" + (f" ({rel:+.1f}% relative)" if rel is not None else " (baseline is 0; relative n/a)")

    lo, hi = imp["mrr_delta_ci95"]
    plo, phi = imp["precision_delta_ci95"]
    print("\nFull pipeline vs naive baseline")
    print(f"  MRR@{k}:       {fmt(imp['mrr_rel_pct'], imp['mrr_abs_delta'])}   95% CI of delta [{lo:+.3f}, {hi:+.3f}]")
    print(f"  Precision@{k}: {fmt(imp['precision_rel_pct'], imp['precision_abs_delta'])}   95% CI of delta [{plo:+.3f}, {phi:+.3f}]")
    for metric, label in (("recall", "Recall"), ("hit", "Hit")):
        ci_lo, ci_hi = imp[f"{metric}_delta_ci95"]
        print(
            f"  {label}@{k}:      {fmt(imp[f'{metric}_rel_pct'], imp[f'{metric}_abs_delta'])}"
            f"   95% CI of delta [{ci_lo:+.3f}, {ci_hi:+.3f}]"
        )
    print(f"\nPer-event results saved to {EVAL_RESULTS_PATH}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Naive vs Time-Filtered Hybrid+Rerank retrieval benchmark")
    p.add_argument("--k", type=int, default=FINAL_K)
    p.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    p.add_argument("--detected-only", action="store_true", help="only score events the detector flagged")
    p.add_argument("--no-prophet", action="store_true", help="use the rolling Z-score detector only")
    p.add_argument("--time-filter", choices=["int_range", "datetime_range"], default="int_range")
    a = p.parse_args(argv)
    print_report(run_benchmark(a.k, a.candidate_k, a.detected_only, not a.no_prophet, a.time_filter))


if __name__ == "__main__":
    main()
