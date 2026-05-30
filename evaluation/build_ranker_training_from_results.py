#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.feature_extraction import extract_features, extract_query_plan


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_by_id(path: Optional[str]) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    rows = _load_json(path)
    if not isinstance(rows, list):
        return {}
    return {str(row.get("id", "")): row for row in rows}


def _candidate_valid(label: str) -> int:
    return int(label in {"correct", "valid_wrong"})


def build_training_rows(
    results_path: str,
    schema_path: str,
    dataset_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, object]]]:
    results = _load_json(results_path)
    details = list(results.get("details") or [])
    schema_dict = _load_json(schema_path)
    dataset = _dataset_by_id(dataset_path)
    out: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for detail in details:
        qid = str(detail.get("id", "")).strip()
        if not qid:
            continue
        question = str(detail.get("effective_question") or detail.get("question") or "")
        item = dataset.get(qid, {})
        family = str(item.get("family") or detail.get("family") or "").strip()
        ambiguity = str(
            item.get("ambiguity_label") or detail.get("ambiguity_label") or ""
        ).strip()
        gold_query = str(item.get("query") or "")

        for idx, cand in enumerate(detail.get("candidates") or []):
            query = str(cand.get("query", "") or "").strip()
            if not query:
                continue
            label = str(cand.get("label", "")).lower()
            try:
                features = extract_features(question, query, schema_dict)
            except Exception:
                features = {}
            try:
                plan_labels = list(extract_query_plan(query, schema_dict).get("labels", []))
            except Exception:
                plan_labels = []
            out[qid].append(
                {
                    "query_id": f"{qid}_cand{idx:02d}",
                    "question": question,
                    "ambiguity_label": ambiguity,
                    "family": family,
                    "gold_query": gold_query,
                    "query": query,
                    "is_correct": int(label == "correct"),
                    "is_valid": _candidate_valid(label),
                    "features": features,
                    "source": str(cand.get("source") or "llm"),
                    "query_plan_labels": plan_labels,
                    "original_index": cand.get("index", idx),
                    "original_label": label,
                }
            )

    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build candidate-level ML reranker training data from an evaluation results JSON."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = build_training_rows(args.results, args.schema, args.dataset or None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")

    questions = len(rows)
    candidates = sum(len(v) for v in rows.values())
    correct = sum(int(r.get("is_correct", 0)) for rs in rows.values() for r in rs)
    print("===== RANKER TRAINING DATA FROM RESULTS =====")
    print(f"Results: {args.results}")
    print(f"Questions: {questions}")
    print(f"Candidates: {candidates}")
    print(f"Correct candidates: {correct}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
