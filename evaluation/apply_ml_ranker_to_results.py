#!/usr/bin/env python3
import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
from ranking.xgboost_ranker import XGBoostCandidateRanker


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_ranker_model(path: str):
    if str(path).lower().endswith((".pkl", ".pickle")):
        return XGBoostCandidateRanker.load(path)
    return NPTfidfRanker.load(path)


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


def _trusted_rescue_topic(question: str, topics: Sequence[str]) -> bool:
    q = " ".join(str(question or "").lower().replace("tier 1", "tier1").split())
    allowed = {str(topic).strip().lower() for topic in topics if str(topic).strip()}
    if "inventory" in allowed and "inventory" in q:
        return True
    if "order_cancellation" in allowed and ("order cancellation" in q or "cancellation" in q):
        return True
    if "vehicle_sales" in allowed and (
        "vehicle sales" in q
        or "vehicles sold" in q
        or "units sold" in q
        or "actual sales" in q
        or "forecast" in q
    ):
        return True
    return False


def _trusted_source_rescue_row(
    *,
    question: str,
    candidates: Sequence[Dict[str, object]],
    ranked_rows: Sequence[Dict[str, object]],
    score_by_key: Dict[str, float],
    current_key: str,
    current_score: float,
    max_rank: int,
    min_score: float,
    min_margin: float,
    topics: Sequence[str],
    structured_guard: bool,
) -> Optional[Dict[str, object]]:
    if not _trusted_rescue_topic(question, topics):
        return None

    ranked_by_key = {
        _query_key(str(row.get("query", ""))): row
        for row in ranked_rows
    }
    trusted_sources = {"validated_retrieval", "template"}
    best_row = None
    best_key = (float("-inf"), float("-inf"), 999)
    for original_index, cand in enumerate(candidates[1:], start=1):
        original_rank = original_index + 1
        if original_rank > int(max_rank):
            continue
        source = str(cand.get("source") or "").strip().lower()
        if source not in trusted_sources:
            continue
        cand_key = _query_key(str(cand.get("query", "")))
        if not cand_key or cand_key == current_key:
            continue
        cand_score = float(score_by_key.get(cand_key, 0.0))
        if cand_score < float(min_score):
            continue
        if cand_score - current_score < float(min_margin):
            continue
        if structured_guard and not _structured_guard_allows(
            question,
            str(candidates[0].get("query", "")),
            str(cand.get("query", "")),
        ):
            continue
        row = ranked_by_key.get(cand_key)
        if row is None:
            continue
        key = (cand_score - current_score, cand_score, -original_rank)
        if key > best_key:
            best_key = key
            best_row = dict(row)
            best_row["trusted_source_rescue_reason"] = {
                "source": source,
                "original_rank": original_rank,
                "candidate_score": cand_score,
                "current_score": current_score,
                "score_delta": cand_score - current_score,
            }
    return best_row


def _question_requests_shortage_status_breakdown(question: str) -> bool:
    q = " ".join(str(question or "").lower().replace("tier 1", "tier1").split())
    if "shortage" not in q:
        return False
    count_requested = "how many" in q or "count" in q or "counts" in q or "number of" in q
    yes_no_requested = any(
        phrase in q
        for phrase in (
            "reported a shortage",
            "reported shortages",
            "have not",
            "did not",
            "has not",
            "shortage versus",
            "versus those",
            "whether they",
            "or not",
            "not reported",
            "not?",
        )
    )
    return bool(count_requested and yes_no_requested)


def _query_has_shortage_status_breakdown(query: str) -> bool:
    q = str(query or "").lower()
    if "reportsshortage" not in q and "shortagestatus" not in q and "shortagelabel" not in q:
        return False
    if "count(" not in q:
        return False
    return "group by" in q and (
        "shortagestatus" in q
        or "shortagelabel" in q
        or "?status" in q
        or "reportsshortage" in q
    )


def _shortage_status_rescue_row(
    *,
    question: str,
    candidates: Sequence[Dict[str, object]],
    ranked_rows: Sequence[Dict[str, object]],
    score_by_key: Dict[str, float],
    current_key: str,
    current_score: float,
    max_rank: int,
    min_score: float,
    min_margin: float,
    structured_guard: bool,
) -> Optional[Dict[str, object]]:
    if not _question_requests_shortage_status_breakdown(question):
        return None
    if _query_has_shortage_status_breakdown(str(candidates[0].get("query", ""))):
        return None

    ranked_by_key = {
        _query_key(str(row.get("query", ""))): row
        for row in ranked_rows
    }
    best_row = None
    best_key = (float("-inf"), float("-inf"), 999)
    for original_index, cand in enumerate(candidates[1:], start=1):
        original_rank = original_index + 1
        if original_rank > int(max_rank):
            continue
        cand_query = str(cand.get("query", ""))
        if not _query_has_shortage_status_breakdown(cand_query):
            continue
        cand_key = _query_key(cand_query)
        if not cand_key or cand_key == current_key:
            continue
        cand_score = float(score_by_key.get(cand_key, 0.0))
        if cand_score < float(min_score):
            continue
        if cand_score - current_score < float(min_margin):
            continue
        if structured_guard and not _structured_guard_allows(
            question,
            str(candidates[0].get("query", "")),
            cand_query,
        ):
            continue
        row = ranked_by_key.get(cand_key)
        if row is None:
            continue
        key = (cand_score - current_score, cand_score, -original_rank)
        if key > best_key:
            best_key = key
            best_row = dict(row)
            best_row["shortage_status_rescue_reason"] = {
                "original_rank": original_rank,
                "candidate_score": cand_score,
                "current_score": current_score,
                "score_delta": cand_score - current_score,
            }
    return best_row


def _question_requests_current_baseline(question: str) -> bool:
    q = " ".join(str(question or "").lower().replace("tier 1", "tier1").split())
    if "current demand" not in q:
        return False
    return any(
        term in q
        for term in (
            "baseline",
            "bl1",
            "bl2",
            "percentage change",
            "percent change",
            "pct change",
        )
    )


def _query_current_baseline_score(question: str, query: str) -> float:
    q = " ".join(str(question or "").lower().replace("tier 1", "tier1").split())
    sparql = str(query or "").lower()
    compact = "".join(sparql.split())
    if "currentdemandanalysis" not in compact:
        return 0.0
    if "baselinetype" not in compact or "percentagechange" not in compact:
        return 0.0

    score = 1.0
    if "tier1currentdemand" in compact:
        score += 1.0
    if "hasaggregatedresult" in compact:
        score += 1.0
    if "survey:tier1_survey" in compact and ("tier1" in q or "tier 1" in q):
        score += 0.5
    if "survey:automotive" in compact and "automotive" in q:
        score += 0.5

    wants_difference = any(term in q for term in ("differ", "difference", "delta", "between"))
    wants_sum = any(term in q for term in ("combined", "add up", "sum", "total"))
    wants_average = "average" in q or "avg" in q
    wants_list = any(term in q for term in ("list", "show", "what are", "percentage changes"))
    has_if_bl = "if(?baseline" in compact or "if(?baseline=" in compact
    has_subtraction = "-" in sparql and ("bl1" in compact and "bl2" in compact)
    has_sum = "sum(" in compact
    has_avg = "avg(" in compact
    has_group_baseline = "groupby?baseline" in compact

    if wants_difference:
        score += 1.0 if has_subtraction else -0.8
    if wants_sum:
        score += 0.7 if has_sum else -0.5
    if wants_average:
        score += 0.7 if has_avg else -0.5
    if wants_list and not (wants_sum or wants_average or wants_difference):
        score += 0.8 if not has_sum and not has_avg else -0.8
    if ("bl1" in q or "bl2" in q) and ("bl1" in compact and "bl2" in compact):
        score += 0.5
    if (wants_sum or wants_average) and has_group_baseline:
        score += 0.4

    return score


def _current_baseline_rescue_row(
    *,
    question: str,
    candidates: Sequence[Dict[str, object]],
    ranked_rows: Sequence[Dict[str, object]],
    score_by_key: Dict[str, float],
    current_key: str,
    current_score: float,
    max_rank: int,
    min_score: float,
    min_margin: float,
    structured_guard: bool,
) -> Optional[Dict[str, object]]:
    if not _question_requests_current_baseline(question):
        return None

    current_query = str(candidates[0].get("query", ""))
    current_shape = _query_current_baseline_score(question, current_query)
    ranked_by_key = {
        _query_key(str(row.get("query", ""))): row
        for row in ranked_rows
    }
    best_row = None
    best_key = (float("-inf"), float("-inf"), float("-inf"), 999)
    for original_index, cand in enumerate(candidates[1:], start=1):
        original_rank = original_index + 1
        if original_rank > int(max_rank):
            continue
        cand_query = str(cand.get("query", ""))
        cand_shape = _query_current_baseline_score(question, cand_query)
        if cand_shape <= max(0.0, current_shape):
            continue
        cand_key = _query_key(cand_query)
        if not cand_key or cand_key == current_key:
            continue
        cand_score = float(score_by_key.get(cand_key, 0.0))
        if cand_score < float(min_score):
            continue
        if cand_score - current_score < float(min_margin):
            continue
        if structured_guard and not _structured_guard_allows(
            question,
            current_query,
            cand_query,
        ):
            continue
        row = ranked_by_key.get(cand_key)
        if row is None:
            continue
        key = (cand_shape - current_shape, cand_score - current_score, cand_score, -original_rank)
        if key > best_key:
            best_key = key
            best_row = dict(row)
            best_row["current_baseline_rescue_reason"] = {
                "original_rank": original_rank,
                "candidate_score": cand_score,
                "current_score": current_score,
                "score_delta": cand_score - current_score,
                "candidate_shape_score": cand_shape,
                "current_shape_score": current_shape,
            }
    return best_row


def _rank_detail(
    detail: Dict[str, object],
    ranker,
    schema_dict: Dict[str, object],
    guarded: bool = False,
    min_margin: float = 0.15,
    min_score: float = 0.50,
    max_rank: int = 4,
    structured_guard: bool = False,
    enable_rank2_trusted_rescue: bool = False,
    trusted_rescue_max_rank: int = 2,
    trusted_rescue_min_score: float = 0.75,
    trusted_rescue_min_margin: float = 0.25,
    trusted_rescue_topics: Sequence[str] = ("inventory", "order_cancellation", "vehicle_sales"),
    enable_shortage_status_rescue: bool = False,
    shortage_status_rescue_max_rank: int = 3,
    shortage_status_rescue_min_score: float = 0.45,
    shortage_status_rescue_min_margin: float = -0.05,
    enable_current_baseline_rescue: bool = False,
    current_baseline_rescue_max_rank: int = 4,
    current_baseline_rescue_min_score: float = 0.35,
    current_baseline_rescue_min_margin: float = -0.10,
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
            if enable_rank2_trusted_rescue:
                rescue_row = _trusted_source_rescue_row(
                    question=question,
                    candidates=candidates,
                    ranked_rows=ranked_rows,
                    score_by_key=score_by_key,
                    current_key=current_key,
                    current_score=current_score,
                    max_rank=trusted_rescue_max_rank,
                    min_score=trusted_rescue_min_score,
                    min_margin=trusted_rescue_min_margin,
                    topics=trusted_rescue_topics,
                    structured_guard=structured_guard,
                )
                if rescue_row is not None:
                    chosen_key = _query_key(str(rescue_row.get("query", "")))
                    ranked_rows = [rescue_row] + [
                        row
                        for row in ranked_rows
                        if _query_key(str(row.get("query", ""))) != chosen_key
                    ]
                    switch_allowed = True
            if not switch_allowed and enable_shortage_status_rescue:
                rescue_row = _shortage_status_rescue_row(
                    question=question,
                    candidates=candidates,
                    ranked_rows=ranked_rows,
                    score_by_key=score_by_key,
                    current_key=current_key,
                    current_score=current_score,
                    max_rank=shortage_status_rescue_max_rank,
                    min_score=shortage_status_rescue_min_score,
                    min_margin=shortage_status_rescue_min_margin,
                    structured_guard=structured_guard,
                )
                if rescue_row is not None:
                    chosen_key = _query_key(str(rescue_row.get("query", "")))
                    ranked_rows = [rescue_row] + [
                        row
                        for row in ranked_rows
                        if _query_key(str(row.get("query", ""))) != chosen_key
                    ]
                    switch_allowed = True
            if not switch_allowed and enable_current_baseline_rescue:
                rescue_row = _current_baseline_rescue_row(
                    question=question,
                    candidates=candidates,
                    ranked_rows=ranked_rows,
                    score_by_key=score_by_key,
                    current_key=current_key,
                    current_score=current_score,
                    max_rank=current_baseline_rescue_max_rank,
                    min_score=current_baseline_rescue_min_score,
                    min_margin=current_baseline_rescue_min_margin,
                    structured_guard=structured_guard,
                )
                if rescue_row is not None:
                    chosen_key = _query_key(str(rescue_row.get("query", "")))
                    ranked_rows = [rescue_row] + [
                        row
                        for row in ranked_rows
                        if _query_key(str(row.get("query", ""))) != chosen_key
                    ]
                    switch_allowed = True
            if switch_allowed:
                pass
            else:
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
        if row.get("trusted_source_rescue_reason"):
            cand["trusted_source_rescue_reason"] = row.get("trusted_source_rescue_reason")
        if row.get("shortage_status_rescue_reason"):
            cand["shortage_status_rescue_reason"] = row.get("shortage_status_rescue_reason")
        if row.get("current_baseline_rescue_reason"):
            cand["current_baseline_rescue_reason"] = row.get("current_baseline_rescue_reason")
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
    enable_rank2_trusted_rescue: bool = False,
    trusted_rescue_max_rank: int = 2,
    trusted_rescue_min_score: float = 0.75,
    trusted_rescue_min_margin: float = 0.25,
    trusted_rescue_topics: Sequence[str] = ("inventory", "order_cancellation", "vehicle_sales"),
    enable_shortage_status_rescue: bool = False,
    shortage_status_rescue_max_rank: int = 3,
    shortage_status_rescue_min_score: float = 0.45,
    shortage_status_rescue_min_margin: float = -0.05,
    enable_current_baseline_rescue: bool = False,
    current_baseline_rescue_max_rank: int = 4,
    current_baseline_rescue_min_score: float = 0.35,
    current_baseline_rescue_min_margin: float = -0.10,
) -> Dict[str, object]:
    payload = _load_json(results_path)
    schema_dict = _load_json(schema_path)
    ranker = _load_ranker_model(model_path)
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
            enable_rank2_trusted_rescue=enable_rank2_trusted_rescue,
            trusted_rescue_max_rank=trusted_rescue_max_rank,
            trusted_rescue_min_score=trusted_rescue_min_score,
            trusted_rescue_min_margin=trusted_rescue_min_margin,
            trusted_rescue_topics=trusted_rescue_topics,
            enable_shortage_status_rescue=enable_shortage_status_rescue,
            shortage_status_rescue_max_rank=shortage_status_rescue_max_rank,
            shortage_status_rescue_min_score=shortage_status_rescue_min_score,
            shortage_status_rescue_min_margin=shortage_status_rescue_min_margin,
            enable_current_baseline_rescue=enable_current_baseline_rescue,
            current_baseline_rescue_max_rank=current_baseline_rescue_max_rank,
            current_baseline_rescue_min_score=current_baseline_rescue_min_score,
            current_baseline_rescue_min_margin=current_baseline_rescue_min_margin,
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
                "trusted_source_rescue_reason": after_candidates[0].get(
                    "trusted_source_rescue_reason"
                ),
                "shortage_status_rescue_reason": after_candidates[0].get(
                    "shortage_status_rescue_reason"
                ),
                "current_baseline_rescue_reason": after_candidates[0].get(
                    "current_baseline_rescue_reason"
                ),
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
        "enable_rank2_trusted_rescue": bool(enable_rank2_trusted_rescue),
        "trusted_rescue_max_rank": int(trusted_rescue_max_rank),
        "trusted_rescue_min_score": float(trusted_rescue_min_score),
        "trusted_rescue_min_margin": float(trusted_rescue_min_margin),
        "trusted_rescue_topics": list(trusted_rescue_topics),
        "enable_shortage_status_rescue": bool(enable_shortage_status_rescue),
        "shortage_status_rescue_max_rank": int(shortage_status_rescue_max_rank),
        "shortage_status_rescue_min_score": float(shortage_status_rescue_min_score),
        "shortage_status_rescue_min_margin": float(shortage_status_rescue_min_margin),
        "enable_current_baseline_rescue": bool(enable_current_baseline_rescue),
        "current_baseline_rescue_max_rank": int(current_baseline_rescue_max_rank),
        "current_baseline_rescue_min_score": float(current_baseline_rescue_min_score),
        "current_baseline_rescue_min_margin": float(current_baseline_rescue_min_margin),
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
    parser.add_argument(
        "--enable-rank2-trusted-rescue",
        action="store_true",
        help=(
            "After guarded ML declines to switch, allow a very narrow rescue from "
            "trusted generated sources for selected topics."
        ),
    )
    parser.add_argument("--trusted-rescue-max-rank", type=int, default=2)
    parser.add_argument("--trusted-rescue-min-score", type=float, default=0.75)
    parser.add_argument("--trusted-rescue-min-margin", type=float, default=0.25)
    parser.add_argument(
        "--trusted-rescue-topics",
        default="inventory,order_cancellation,vehicle_sales",
        help="Comma-separated topics eligible for trusted-source rescue.",
    )
    parser.add_argument(
        "--enable-shortage-status-rescue",
        action="store_true",
        help="Allow a narrow rescue for shortage yes/no count questions when a grouped shortage-status candidate is nearby.",
    )
    parser.add_argument("--shortage-status-rescue-max-rank", type=int, default=3)
    parser.add_argument("--shortage-status-rescue-min-score", type=float, default=0.45)
    parser.add_argument("--shortage-status-rescue-min-margin", type=float, default=-0.05)
    parser.add_argument(
        "--enable-current-baseline-rescue",
        action="store_true",
        help="Allow a narrow rescue for BL1/BL2 current-demand baseline questions.",
    )
    parser.add_argument("--current-baseline-rescue-max-rank", type=int, default=4)
    parser.add_argument("--current-baseline-rescue-min-score", type=float, default=0.35)
    parser.add_argument("--current-baseline-rescue-min-margin", type=float, default=-0.10)
    args = parser.parse_args()
    trusted_rescue_topics = [
        topic.strip()
        for topic in str(args.trusted_rescue_topics).split(",")
        if topic.strip()
    ]

    updated = apply_ml_ranker(
        args.results,
        args.model,
        args.schema,
        guarded=args.guarded,
        min_margin=args.min_margin,
        min_score=args.min_score,
        max_rank=args.max_rank,
        structured_guard=args.structured_guard,
        enable_rank2_trusted_rescue=args.enable_rank2_trusted_rescue,
        trusted_rescue_max_rank=args.trusted_rescue_max_rank,
        trusted_rescue_min_score=args.trusted_rescue_min_score,
        trusted_rescue_min_margin=args.trusted_rescue_min_margin,
        trusted_rescue_topics=trusted_rescue_topics,
        enable_shortage_status_rescue=args.enable_shortage_status_rescue,
        shortage_status_rescue_max_rank=args.shortage_status_rescue_max_rank,
        shortage_status_rescue_min_score=args.shortage_status_rescue_min_score,
        shortage_status_rescue_min_margin=args.shortage_status_rescue_min_margin,
        enable_current_baseline_rescue=args.enable_current_baseline_rescue,
        current_baseline_rescue_max_rank=args.current_baseline_rescue_max_rank,
        current_baseline_rescue_min_score=args.current_baseline_rescue_min_score,
        current_baseline_rescue_min_margin=args.current_baseline_rescue_min_margin,
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
    if rewrite.get("enable_rank2_trusted_rescue"):
        print(
            "Trusted rescue: "
            f"max_rank={rewrite['trusted_rescue_max_rank']}, "
            f"min_score={rewrite['trusted_rescue_min_score']}, "
            f"min_margin={rewrite['trusted_rescue_min_margin']}, "
            f"topics={','.join(rewrite['trusted_rescue_topics'])}"
        )
    if rewrite.get("enable_shortage_status_rescue"):
        print(
            "Shortage status rescue: "
            f"max_rank={rewrite['shortage_status_rescue_max_rank']}, "
            f"min_score={rewrite['shortage_status_rescue_min_score']}, "
            f"min_margin={rewrite['shortage_status_rescue_min_margin']}"
        )
    if rewrite.get("enable_current_baseline_rescue"):
        print(
            "Current baseline rescue: "
            f"max_rank={rewrite['current_baseline_rescue_max_rank']}, "
            f"min_score={rewrite['current_baseline_rescue_min_score']}, "
            f"min_margin={rewrite['current_baseline_rescue_min_margin']}"
        )
    print(f"Changed selections: {rewrite['changed_count']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")
    print("Note: same-results reranking is diagnostic, not a final held-out score.")


if __name__ == "__main__":
    main()
