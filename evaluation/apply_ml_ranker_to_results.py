#!/usr/bin/env python3
import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.apply_selection_to_results import _recompute_summary
from ranking.np_tfidf_ranker import NPTfidfRanker, rank_candidates_with_model


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _rank_detail(
    detail: Dict[str, object],
    ranker: NPTfidfRanker,
    schema_dict: Dict[str, object],
) -> Dict[str, object]:
    updated = deepcopy(detail)
    candidates = list(updated.get("candidates") or [])
    if len(candidates) < 2:
        return updated

    question = str(updated.get("effective_question") or updated.get("question") or "")
    rank_rows = [
        {
            "query": str(cand.get("query", "") or ""),
            "source": str(cand.get("source") or "llm"),
        }
        for cand in candidates
    ]
    ranked_rows = rank_candidates_with_model(ranker, question, rank_rows, schema_dict)

    buckets: Dict[str, List[Dict[str, object]]] = {}
    for cand in candidates:
        buckets.setdefault(_query_key(str(cand.get("query", ""))), []).append(cand)

    reordered: List[Dict[str, object]] = []
    for row in ranked_rows:
        key = _query_key(str(row.get("query", "")))
        bucket = buckets.get(key) or []
        if not bucket:
            continue
        cand = deepcopy(bucket.pop(0))
        cand["ml_score"] = row.get("ml_score")
        reordered.append(cand)

    # Keep any duplicate/unmatched candidates instead of dropping them.
    for bucket in buckets.values():
        for cand in bucket:
            reordered.append(deepcopy(cand))

    if reordered:
        updated["candidates"] = reordered
    return updated


def apply_ml_ranker(results_path: str, model_path: str, schema_path: str) -> Dict[str, object]:
    payload = _load_json(results_path)
    schema_dict = _load_json(schema_path)
    ranker = NPTfidfRanker.load(model_path)

    original_details = list(payload.get("details") or [])
    details = [_rank_detail(detail, ranker, schema_dict) for detail in original_details]

    changed = []
    for before, after in zip(original_details, details):
        before_candidates = before.get("candidates") or []
        after_candidates = after.get("candidates") or []
        if not before_candidates or not after_candidates:
            continue
        before_key = _query_key(str(before_candidates[0].get("query", "")))
        after_key = _query_key(str(after_candidates[0].get("query", "")))
        if before_key == after_key:
            continue
        changed.append(
            {
                "id": after.get("id"),
                "from_label": str(before_candidates[0].get("label", "")).lower(),
                "to_label": str(after_candidates[0].get("label", "")).lower(),
                "from_index": before_candidates[0].get("index"),
                "to_index": after_candidates[0].get("index"),
                "to_ml_score": after_candidates[0].get("ml_score"),
            }
        )

    payload["summary"] = _recompute_summary(details, payload.get("summary") or {})
    payload["details"] = details
    payload["ml_rerank_rewrite"] = {
        "source_results": results_path,
        "model": model_path,
        "schema": schema_path,
        "changed_count": len(changed),
        "changed": changed,
        "note": (
            "If the model was trained on these same results, this is a diagnostic "
            "ranking upper-bound, not an unbiased held-out metric."
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a trained NP TF-IDF ML reranker to an existing evaluation JSON without new LLM calls."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    updated = apply_ml_ranker(args.results, args.model, args.schema)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = updated["summary"]
    rewrite = updated["ml_rerank_rewrite"]
    print("===== APPLY ML RANKER TO RESULTS =====")
    print(f"Input: {args.results}")
    print(f"Model: {args.model}")
    print(f"Changed selections: {rewrite['changed_count']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")
    print("Note: same-results reranking is diagnostic, not a final held-out score.")


if __name__ == "__main__":
    main()
