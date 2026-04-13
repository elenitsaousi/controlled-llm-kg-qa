import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from rdflib import Graph

# Ensure project root is on sys.path when running as a script/module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.infineon_eval import (
    QueryTimeout,
    _ensure_prefixes,
    _run_query,
    _strip_comments,
)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sig_to_set(sig) -> set:
    return set(sig.keys())


def _safe_run(graph: Graph, query: str, timeout_s: Optional[float]):
    try:
        sig = _run_query(graph, query, timeout_s)
        return sig, None
    except QueryTimeout as exc:
        return None, f"timeout: {exc}"
    except Exception as exc:
        return None, f"error: {exc}"


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_features(
    dataset_path: str,
    graph_path: str,
    results_path: str,
    query_timeout: Optional[float],
) -> List[Dict[str, object]]:
    dataset = {d["id"]: d for d in _load_json(dataset_path)}
    results = _load_json(results_path)
    details = results.get("details", [])

    g = Graph()
    g.parse(graph_path, format="turtle")

    features = []
    for d in details:
        qid = d["id"]
        if qid not in dataset:
            continue

        gold_query = _strip_comments(str(dataset[qid].get("query", "")).strip())
        gold_full = _ensure_prefixes(gold_query)
        gold_sig, gold_err = _safe_run(g, gold_full, query_timeout)
        if gold_sig is None:
            # Skip if gold invalid/timeout
            features.append(
                {
                    "id": qid,
                    "gold_error": gold_err,
                }
            )
            continue

        candidates = d.get("candidates", [])
        cand_sigs = []
        cand_valid = []
        cand_rows = []

        for c in candidates:
            cand_query = _strip_comments(str(c.get("query", "")).strip())
            cand_full = _ensure_prefixes(cand_query)
            sig, err = _safe_run(g, cand_full, query_timeout)
            if sig is None:
                cand_sigs.append(None)
                cand_valid.append(False)
                cand_rows.append(0)
            else:
                cand_sigs.append(sig)
                cand_valid.append(True)
                cand_rows.append(len(_sig_to_set(sig)))

        top1_sig = cand_sigs[0] if cand_sigs else None
        top2_sig = cand_sigs[1] if len(cand_sigs) > 1 else None

        top1_valid = bool(cand_valid[0]) if cand_valid else False
        top1_rows = cand_rows[0] if cand_rows else 0
        top1_correct = bool(top1_sig is not None and top1_sig == gold_sig)

        jacc = 0.0
        if top1_sig is not None and top2_sig is not None:
            jacc = _jaccard(_sig_to_set(top1_sig), _sig_to_set(top2_sig))

        features.append(
            {
                "id": qid,
                "top1_correct": top1_correct,
                "top1_valid": top1_valid,
                "top1_rows": top1_rows,
                "jaccard_top1_top2": jacc,
            }
        )

    return features


def sweep_thresholds(features: List[Dict[str, object]]) -> List[Dict[str, object]]:
    thresholds = [round(x / 10, 1) for x in range(0, 11)]
    total = len([f for f in features if "top1_correct" in f])
    results = []

    for t in thresholds:
        answered = []
        correct = 0
        for f in features:
            if "top1_correct" not in f:
                continue
            if not f["top1_valid"]:
                continue
            if f["top1_rows"] <= 0:
                continue
            if f["jaccard_top1_top2"] < t:
                continue
            answered.append(f)
            if f["top1_correct"]:
                correct += 1

        answered_n = len(answered)
        precision = correct / answered_n if answered_n else 0.0
        coverage = answered_n / total if total else 0.0
        results.append(
            {
                "threshold": t,
                "answered": answered_n,
                "correct": correct,
                "precision": precision,
                "coverage": coverage,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ambiguity thresholds using candidate agreement."
    )
    parser.add_argument(
        "--dataset",
        default="data/infineon/infineon_dataset_30.json",
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--graph",
        default="data/infineon/graph.ttl",
        help="Path to TTL graph",
    )
    parser.add_argument(
        "--results",
        default="results/infineon_eval.json",
        help="Path to evaluation results JSON",
    )
    parser.add_argument(
        "--query-timeout",
        type=float,
        default=60,
        help="Max seconds per SPARQL query",
    )
    args = parser.parse_args()

    features = compute_features(
        dataset_path=args.dataset,
        graph_path=args.graph,
        results_path=args.results,
        query_timeout=args.query_timeout,
    )
    sweep = sweep_thresholds(features)

    print("===== THRESHOLD SWEEP =====")
    print("Criteria: top1 valid, top1_rows>0, jaccard(top1, top2) >= t")
    print("t | answered | precision | coverage | correct")
    for r in sweep:
        print(
            f"{r['threshold']:.1f} | {r['answered']:>8} | {r['precision']:.2%} | {r['coverage']:.2%} | {r['correct']}"
        )


if __name__ == "__main__":
    main()
