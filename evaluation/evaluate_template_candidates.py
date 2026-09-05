#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rdflib import Graph

from evaluation.infineon_eval import _ensure_prefixes, _run_query_cached, _strip_comments
from llm.candidate_generation import _template_candidate_queries


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_label(
    graph: Graph,
    query: str,
    gold_sig: Counter,
    timeout_s: Optional[float],
    cache: Dict[Tuple[Optional[float], str], Tuple[str, object]],
) -> str:
    try:
        sig = _run_query_cached(graph, _ensure_prefixes(_strip_comments(query)), timeout_s, cache)
    except Exception:
        return "invalid"
    return "correct" if sig == gold_sig else "valid_wrong"


def evaluate_templates(
    dataset_path: str,
    graph_path: str,
    timeout_s: Optional[float],
) -> Dict[str, object]:
    rows = _load_json(dataset_path)
    if not isinstance(rows, list):
        raise ValueError("Dataset JSON must be a list.")

    graph = Graph()
    graph.parse(graph_path, format="turtle")
    cache: Dict[Tuple[Optional[float], str], Tuple[str, object]] = {}

    details: List[Dict[str, object]] = []
    family_counts: Dict[str, Counter] = defaultdict(Counter)
    summary = Counter()

    for row in rows:
        qid = str(row.get("id", ""))
        question = str(row.get("question", ""))
        topic = str(row.get("topic") or row.get("family_id") or "unknown")
        gold_query = str(row.get("query", ""))
        summary["total"] += 1

        try:
            gold_sig = _run_query_cached(graph, _ensure_prefixes(gold_query), timeout_s, cache)
        except Exception as exc:
            details.append(
                {
                    "id": qid,
                    "question": question,
                    "topic": topic,
                    "gold_error": str(exc),
                    "top1_correct": False,
                    "any_correct": False,
                    "candidates": [],
                }
            )
            summary["gold_invalid"] += 1
            family_counts[topic]["total"] += 1
            family_counts[topic]["gold_invalid"] += 1
            continue

        candidates = []
        for idx, query in enumerate(_template_candidate_queries(question)):
            label = _candidate_label(graph, query, gold_sig, timeout_s, cache)
            candidates.append({"index": idx, "label": label, "query": query})

        top1_correct = bool(candidates and candidates[0]["label"] == "correct")
        any_correct = any(c["label"] == "correct" for c in candidates)
        summary["top1_correct"] += int(top1_correct)
        summary["any_correct"] += int(any_correct)
        summary["candidate_total"] += len(candidates)
        summary["candidate_correct"] += sum(1 for c in candidates if c["label"] == "correct")
        summary["candidate_invalid"] += sum(1 for c in candidates if c["label"] == "invalid")

        family_counts[topic]["total"] += 1
        family_counts[topic]["top1_correct"] += int(top1_correct)
        family_counts[topic]["any_correct"] += int(any_correct)
        family_counts[topic]["candidate_total"] += len(candidates)

        details.append(
            {
                "id": qid,
                "question": question,
                "topic": topic,
                "top1_correct": top1_correct,
                "any_correct": any_correct,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )

    total = int(summary["total"]) or 1
    families = {}
    for topic, counts in sorted(family_counts.items()):
        fam_total = int(counts["total"]) or 1
        families[topic] = {
            "total": int(counts["total"]),
            "top1_correct": int(counts["top1_correct"]),
            "top1_correct_rate": counts["top1_correct"] / fam_total,
            "any_correct": int(counts["any_correct"]),
            "any_correct_rate": counts["any_correct"] / fam_total,
            "candidate_total": int(counts["candidate_total"]),
            "gold_invalid": int(counts["gold_invalid"]),
        }

    return {
        "inputs": {
            "dataset": dataset_path,
            "graph": graph_path,
            "query_timeout": timeout_s,
            "mode": "template_candidates_only_no_llm",
        },
        "summary": {
            "total": int(summary["total"]),
            "gold_invalid": int(summary["gold_invalid"]),
            "top1_correct": int(summary["top1_correct"]),
            "top1_correct_rate": summary["top1_correct"] / total,
            "any_correct": int(summary["any_correct"]),
            "any_correct_rate": summary["any_correct"] / total,
            "candidate_total": int(summary["candidate_total"]),
            "candidate_correct": int(summary["candidate_correct"]),
            "candidate_invalid": int(summary["candidate_invalid"]),
        },
        "families": families,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic template candidates only, without LLM calls."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--query-timeout", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = evaluate_templates(args.dataset, args.graph, args.query_timeout)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = report["summary"]
    print("===== TEMPLATE CANDIDATE EVAL =====")
    print(f"Dataset: {args.dataset}")
    print(f"Total: {summary['total']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Candidate invalid: {summary['candidate_invalid']}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
