import argparse
import json
import os
import sys
from collections import Counter
from math import log
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


def _sig_key(sig: Counter) -> Tuple:
    items = list(sig.items())
    items.sort(key=lambda x: repr(x[0]))
    return tuple(items)


def _entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * log(p)
    return h


def _safe_run(graph: Graph, query: str, timeout_s: Optional[float]):
    try:
        sig = _run_query(graph, query, timeout_s)
        return sig, None
    except QueryTimeout as exc:
        return None, f"timeout: {exc}"
    except Exception as exc:
        return None, f"error: {exc}"


def compute_entropies(
    graph_path: str,
    results_path: str,
    query_timeout: Optional[float],
) -> List[Dict[str, object]]:
    results = _load_json(results_path)
    details = results.get("details", [])

    g = Graph()
    g.parse(graph_path, format="turtle")

    rows = []
    for d in details:
        qid = d.get("id", "")
        candidates = d.get("candidates", [])
        if not candidates:
            continue

        outcome_keys = []
        for c in candidates:
            cand_query = _strip_comments(str(c.get("query", "")).strip())
            cand_full = _ensure_prefixes(cand_query)
            sig, err = _safe_run(g, cand_full, query_timeout)
            if sig is None:
                outcome_keys.append("INVALID")
            else:
                outcome_keys.append(_sig_key(sig))

        counts = Counter(outcome_keys)
        entropy = _entropy_from_counts(list(counts.values()))

        rows.append(
            {
                "id": qid,
                "entropy": entropy,
                "top1_correct": bool(d.get("top1_correct", False)),
                "any_correct": bool(d.get("any_correct", False)),
            }
        )

    return rows


def bin_by_entropy(rows: List[Dict[str, object]], bins: int = 3):
    rows_sorted = sorted(rows, key=lambda r: r["entropy"])
    n = len(rows_sorted)
    base = n // bins
    extra = n % bins

    bins_list = []
    start = 0
    for i in range(bins):
        size = base + (1 if i < extra else 0)
        end = start + size
        bins_list.append(rows_sorted[start:end])
        start = end

    return bins_list


def summarize_bins(bins_list: List[List[Dict[str, object]]]):
    summary = []
    for idx, b in enumerate(bins_list):
        total = len(b)
        if total == 0:
            summary.append(
                {
                    "bin": idx,
                    "count": 0,
                    "top1_acc": 0.0,
                    "any_acc": 0.0,
                    "entropy_min": None,
                    "entropy_max": None,
                }
            )
            continue
        top1 = sum(1 for r in b if r["top1_correct"])
        anyc = sum(1 for r in b if r["any_correct"])
        entropies = [r["entropy"] for r in b]
        summary.append(
            {
                "bin": idx,
                "count": total,
                "top1_acc": top1 / total,
                "any_acc": anyc / total,
                "entropy_min": min(entropies),
                "entropy_max": max(entropies),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entropy-based ambiguity analysis for candidate queries."
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
    parser.add_argument(
        "--bins",
        type=int,
        default=3,
        help="Number of entropy bins",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write per-question entropy JSON",
    )
    args = parser.parse_args()

    rows = compute_entropies(
        graph_path=args.graph,
        results_path=args.results,
        query_timeout=args.query_timeout,
    )
    bins_list = bin_by_entropy(rows, bins=args.bins)
    summary = summarize_bins(bins_list)

    print("===== ENTROPY BINS =====")
    print("bin | count | entropy_range | top1_acc | any_acc")
    for s in summary:
        if s["count"] == 0:
            print(f"{s['bin']} | 0 | - | 0.00% | 0.00%")
            continue
        print(
            f"{s['bin']} | {s['count']:>5} | "
            f"[{s['entropy_min']:.3f}, {s['entropy_max']:.3f}] | "
            f"{s['top1_acc']:.2%} | {s['any_acc']:.2%}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
