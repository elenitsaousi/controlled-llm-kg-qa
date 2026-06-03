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
from ranking.feature_extraction import extract_features, extract_query_plan
from ranking.pairwise_ranker import PairwiseRanker


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _candidate_payloads(candidates: List[Dict[str, object]], question: str, schema: Dict[str, object]):
    queries = []
    features = []
    labels = []
    sources = []
    for cand in candidates:
        query = str(cand.get("query", "") or "")
        queries.append(query)
        try:
            features.append(extract_features(question, query, schema))
        except Exception:
            features.append({})
        try:
            labels.append(list(extract_query_plan(query, schema).get("labels", [])))
        except Exception:
            labels.append([])
        sources.append(str(cand.get("source") or "llm"))
    return queries, features, labels, sources


def _rank_detail(
    detail: Dict[str, object],
    ranker: PairwiseRanker,
    schema: Dict[str, object],
    guarded: bool = False,
    min_margin: float = 0.15,
    max_rank: int = 4,
) -> Dict[str, object]:
    updated = deepcopy(detail)
    candidates = list(updated.get("candidates") or [])
    if len(candidates) < 2:
        return updated
    question = str(updated.get("effective_question") or updated.get("question") or "")
    queries, features, labels, sources = _candidate_payloads(candidates, question, schema)
    scores = ranker.score_question_candidates(
        question,
        queries,
        features,
        candidate_query_plan_labels=labels,
        candidate_sources=sources,
    )
    score_by_key = {_query_key(query): float(score) for query, score in zip(queries, scores)}
    rank_by_key = {_query_key(str(cand.get("query", ""))): idx for idx, cand in enumerate(candidates)}
    ranked_keys = sorted(score_by_key, key=lambda key: score_by_key[key], reverse=True)
    current_key = _query_key(str(candidates[0].get("query", "")))

    if guarded and ranked_keys:
        top_key = ranked_keys[0]
        current_score = float(score_by_key.get(current_key, 0.0))
        top_score = float(score_by_key.get(top_key, 0.0))
        top_original_rank = int(rank_by_key.get(top_key, 999))
        if (
            top_key == current_key
            or top_original_rank > int(max_rank)
            or (top_score - current_score) < float(min_margin)
        ):
            updated_candidates = []
            for cand in candidates:
                cand_copy = deepcopy(cand)
                cand_copy["pairwise_score"] = score_by_key.get(_query_key(str(cand_copy.get("query", ""))))
                updated_candidates.append(cand_copy)
            updated["candidates"] = updated_candidates
            return updated

    buckets: Dict[str, List[Dict[str, object]]] = {}
    for cand in candidates:
        buckets.setdefault(_query_key(str(cand.get("query", ""))), []).append(cand)

    reordered = []
    for key in ranked_keys:
        bucket = buckets.get(key) or []
        if not bucket:
            continue
        cand = deepcopy(bucket.pop(0))
        cand["pairwise_score"] = score_by_key.get(key)
        reordered.append(cand)
    for bucket in buckets.values():
        for cand in bucket:
            reordered.append(deepcopy(cand))
    if reordered:
        updated["candidates"] = reordered
    return updated


def apply_pairwise_ranker(
    results_path: str,
    model_path: str,
    schema_path: str,
    guarded: bool = False,
    min_margin: float = 0.15,
    max_rank: int = 4,
) -> Dict[str, object]:
    payload = _load_json(results_path)
    schema = _load_json(schema_path)
    ranker = PairwiseRanker.load(model_path)
    original_details = list(payload.get("details") or [])
    details = [
        _rank_detail(
            detail,
            ranker,
            schema,
            guarded=guarded,
            min_margin=min_margin,
            max_rank=max_rank,
        )
        for detail in original_details
    ]
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
                "to_pairwise_score": after_candidates[0].get("pairwise_score"),
            }
        )

    payload["summary"] = _recompute_summary(details, payload.get("summary") or {})
    payload["details"] = details
    payload["pairwise_rerank_rewrite"] = {
        "source_results": results_path,
        "model": model_path,
        "schema": schema_path,
        "changed_count": len(changed),
        "changed": changed,
        "guarded": bool(guarded),
        "min_margin": float(min_margin),
        "max_rank": int(max_rank),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply pairwise reranker to evaluation results.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--max-rank", type=int, default=4)
    args = parser.parse_args()

    updated = apply_pairwise_ranker(
        args.results,
        args.model,
        args.schema,
        guarded=args.guarded,
        min_margin=args.min_margin,
        max_rank=args.max_rank,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = updated["summary"]
    rewrite = updated["pairwise_rerank_rewrite"]
    print("===== APPLY PAIRWISE RANKER =====")
    print(f"Input: {args.results}")
    print(f"Model: {args.model}")
    print(f"Guarded: {'yes' if rewrite['guarded'] else 'no'}")
    if rewrite["guarded"]:
        print(f"Guard: min_margin={rewrite['min_margin']}, max_rank={rewrite['max_rank']}")
    print(f"Changed selections: {rewrite['changed_count']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
