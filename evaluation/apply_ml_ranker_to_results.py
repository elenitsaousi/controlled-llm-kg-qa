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
from ranking.query_contract import (
    compare_contracts,
    extract_query_contract,
    extract_question_contract,
)
from ranking.np_tfidf_ranker import NPTfidfRanker, rank_candidates_with_model


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _contract_counts(report: Dict[str, object]) -> Dict[str, int]:
    def count(section: str) -> int:
        payload = report.get(section)
        if not isinstance(payload, dict):
            return 0
        return sum(len(values or []) for values in payload.values())

    return {
        "matched": count("matched"),
        "missing": count("missing"),
        "conflicts": count("conflicts"),
    }


def _axis_values(report: Dict[str, object], section: str, axis: str) -> set:
    payload = report.get(section)
    if not isinstance(payload, dict):
        return set()
    values = payload.get(axis) or []
    return {str(v) for v in values}


def _structured_guard_allows(
    question: str,
    current_query: str,
    candidate_query: str,
) -> bool:
    """Reject ML switches that violate explicit question/query constraints."""

    question_contract = extract_question_contract(question)
    if not any(
        [
            question_contract.metrics,
            question_contract.aggregation,
            question_contract.scopes,
            question_contract.dimensions,
            question_contract.filters,
            question_contract.answer_shape,
        ]
    ):
        return True

    current_report = compare_contracts(
        question_contract,
        extract_query_contract(current_query),
    ).to_dict()
    candidate_report = compare_contracts(
        question_contract,
        extract_query_contract(candidate_query),
    ).to_dict()
    current_counts = _contract_counts(current_report)
    candidate_counts = _contract_counts(candidate_report)

    if candidate_counts["conflicts"] > current_counts["conflicts"]:
        return False
    if candidate_counts["missing"] > current_counts["missing"]:
        return False

    requested_axes = [
        axis
        for axis, value in [
            ("metrics", question_contract.metrics),
            ("aggregation", {question_contract.aggregation} if question_contract.aggregation else set()),
            ("scopes", question_contract.scopes),
            ("dimensions", question_contract.dimensions),
            ("filters", question_contract.filters),
            ("answer_shape", {question_contract.answer_shape} if question_contract.answer_shape else set()),
        ]
        if value
    ]
    if not requested_axes:
        return True

    current_bad = set()
    candidate_matches = set()
    for axis in requested_axes:
        if _axis_values(current_report, "missing", axis) or _axis_values(
            current_report, "conflicts", axis
        ):
            current_bad.add(axis)
        if _axis_values(candidate_report, "matched", axis):
            candidate_matches.add(axis)

    if current_bad:
        return bool(current_bad & candidate_matches)

    # If the current candidate already satisfies the explicit contract, allow a
    # switch only when the ML candidate is at least as complete.
    return candidate_counts["matched"] >= current_counts["matched"]


def _rank_detail(
    detail: Dict[str, object],
    ranker: NPTfidfRanker,
    schema_dict: Dict[str, object],
    guarded: bool = False,
    min_margin: float = 0.15,
    min_score: float = 0.50,
    max_rank: int = 4,
    structured_guard: bool = False,
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

    score_by_key = {
        _query_key(str(row.get("query", ""))): float(row.get("ml_score") or 0.0)
        for row in ranked_rows
    }
    rank_by_key = {
        _query_key(str(cand.get("query", ""))): idx
        for idx, cand in enumerate(candidates)
    }

    if guarded:
        current_key = _query_key(str(candidates[0].get("query", "")))
        current_score = float(score_by_key.get(current_key, 0.0))
        chosen_row = None
        switch_allowed = False
        for row in ranked_rows:
            row_key = _query_key(str(row.get("query", "")))
            row_score = float(score_by_key.get(row_key, 0.0))
            row_original_rank = int(rank_by_key.get(row_key, 999))
            should_switch = (
                bool(row_key)
                and row_key != current_key
                and row_original_rank <= int(max_rank)
                and row_score >= float(min_score)
                and (row_score - current_score) >= float(min_margin)
            )
            if not should_switch:
                continue
            if structured_guard and not _structured_guard_allows(
                question,
                str(candidates[0].get("query", "")),
                str(row.get("query", "")),
                ):
                continue
            chosen_row = row
            switch_allowed = True
            break
        if not switch_allowed:
            updated_candidates = []
            for cand in candidates:
                cand_copy = deepcopy(cand)
                cand_copy["ml_score"] = score_by_key.get(
                    _query_key(str(cand_copy.get("query", "")))
                )
                updated_candidates.append(cand_copy)
            updated["candidates"] = updated_candidates
            return updated
        if chosen_row is not None:
            ranked_rows = [chosen_row] + [
                row
                for row in ranked_rows
                if _query_key(str(row.get("query", "")))
                != _query_key(str(chosen_row.get("query", "")))
            ]

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


def apply_ml_ranker(
    results_path: str,
    model_path: str,
    schema_path: str,
    guarded: bool = False,
    min_margin: float = 0.15,
    min_score: float = 0.50,
    max_rank: int = 4,
    structured_guard: bool = False,
) -> Dict[str, object]:
    payload = _load_json(results_path)
    schema_dict = _load_json(schema_path)
    ranker = NPTfidfRanker.load(model_path)
    expected_features = len(ranker.feature_names)
    actual_features = len(ranker.scaler_mean)
    if expected_features != actual_features:
        raise RuntimeError(
            "The ML ranker model was trained with an older feature schema "
            f"({actual_features} scaler features for {expected_features} configured features). "
            "Retrain the ranker with the current code before applying it."
        )

    original_details = list(payload.get("details") or [])
    details = [
        _rank_detail(
            detail,
            ranker,
            schema_dict,
            guarded=guarded,
            min_margin=min_margin,
            min_score=min_score,
            max_rank=max_rank,
            structured_guard=structured_guard,
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
        "guarded": bool(guarded),
        "min_margin": float(min_margin),
        "min_score": float(min_score),
        "max_rank": int(max_rank),
        "structured_guard": bool(structured_guard),
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
    parser.add_argument(
        "--guarded",
        action="store_true",
        help="Only switch top1 when the ML winner has a clear confidence margin.",
    )
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.50)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument(
        "--structured-guard",
        action="store_true",
        help="Reject ML top1 switches that violate the question/query contract.",
    )
    args = parser.parse_args()

    updated = apply_ml_ranker(
        args.results,
        args.model,
        args.schema,
        guarded=args.guarded,
        min_margin=args.min_margin,
        min_score=args.min_score,
        max_rank=args.max_rank,
        structured_guard=args.structured_guard,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = updated["summary"]
    rewrite = updated["ml_rerank_rewrite"]
    print("===== APPLY ML RANKER TO RESULTS =====")
    print(f"Input: {args.results}")
    print(f"Model: {args.model}")
    print(f"Guarded: {'yes' if rewrite['guarded'] else 'no'}")
    if rewrite["guarded"]:
        print(
            f"Guard: min_margin={rewrite['min_margin']}, "
            f"min_score={rewrite['min_score']}, max_rank={rewrite['max_rank']}, "
            f"structured_guard={rewrite.get('structured_guard')}"
        )
    print(f"Changed selections: {rewrite['changed_count']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")
    print("Note: same-results reranking is diagnostic, not a final held-out score.")


if __name__ == "__main__":
    main()
