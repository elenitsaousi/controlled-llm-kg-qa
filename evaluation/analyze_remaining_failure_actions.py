#!/usr/bin/env python3
"""Classify remaining selection failures into actionable improvement buckets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kg.capabilities import DEFAULT_REGISTRY
from ranking.query_contract import compare_contracts, extract_query_contract, extract_question_contract


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _candidate_query(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("query") or candidate.get("sparql") or "").strip()


def _is_correct(candidate: Dict[str, Any]) -> bool:
    return bool(
        candidate.get("is_correct")
        or candidate.get("correct")
        or candidate.get("matches_gold")
        or str(candidate.get("label") or "").lower() == "correct"
    )


def _candidate_empty(candidate: Dict[str, Any]) -> bool:
    for key in ("execution_has_rows", "has_rows", "selected_has_rows"):
        if candidate.get(key) is False:
            return True
    for key in ("row_count", "execution_row_count", "result_count"):
        if key in candidate:
            try:
                return int(candidate.get(key) or 0) == 0
            except (TypeError, ValueError):
                pass
    result = candidate.get("result") or candidate.get("execution_result")
    if isinstance(result, dict):
        if result.get("error"):
            return False
        rows = result.get("rows")
        if isinstance(rows, list):
            return len(rows) == 0
    return False


def _question_prefers_grouped(question: str) -> bool:
    q = str(question or "").lower()
    if re.search(r"\b(highest|top|largest|max|maximum|lowest|min|minimum|best)\b", q):
        return False
    return bool(re.search(r"\b(by|per|for each|grouped by|broken down by|across)\b", q))


def _query_grouped(query: str) -> bool:
    return bool(re.search(r"\bGROUP\s+BY\b", query, flags=re.I))


def _query_ranked_one(query: str) -> bool:
    return bool(re.search(r"\bLIMIT\s+1\b", query, flags=re.I)) or bool(
        re.search(r"\bORDER\s+BY\s+DESC\b", query, flags=re.I)
    )


def _query_count(query: str) -> bool:
    return bool(re.search(r"\bCOUNT\s*\(", query, flags=re.I))


def _contract_counts(report: Dict[str, Any]) -> Dict[str, int]:
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


def _classify_case(detail: Dict[str, Any]) -> Dict[str, Any] | None:
    candidates = list(detail.get("candidates") or [])
    if not candidates:
        return None
    correct_candidates = [candidate for candidate in candidates if _is_correct(candidate)]
    if not correct_candidates:
        return None
    selected = candidates[0]
    if _is_correct(selected):
        return None

    question = str(detail.get("effective_question") or detail.get("question") or "")
    selected_query = _candidate_query(selected)
    first_correct = correct_candidates[0]
    correct_query = _candidate_query(first_correct)
    categories: List[str] = []

    report = DEFAULT_REGISTRY.resolve(question)
    if DEFAULT_REGISTRY.direct_query_for(report):
        categories.append("missing_or_unused_direct_template")

    if _candidate_empty(selected):
        categories.append("empty_result_selected")

    q_contract = extract_question_contract(question)
    selected_report = compare_contracts(q_contract, extract_query_contract(selected_query)).to_dict()
    correct_report = compare_contracts(q_contract, extract_query_contract(correct_query)).to_dict()
    selected_counts = _contract_counts(selected_report)
    correct_counts = _contract_counts(correct_report)

    if selected_counts["conflicts"] > correct_counts["conflicts"]:
        categories.append("answer_shape_or_contract_conflict")
    if selected_counts["missing"] > correct_counts["missing"]:
        categories.append("missing_required_semantic_axis")

    if q_contract.aggregation:
        selected_agg = extract_query_contract(selected_query).aggregation
        correct_agg = extract_query_contract(correct_query).aggregation
        if selected_agg != correct_agg:
            categories.append("wrong_aggregation")

    if _question_prefers_grouped(question):
        if not _query_grouped(selected_query) and _query_grouped(correct_query):
            categories.append("wrong_grouping")
        if _query_ranked_one(selected_query) and not _query_ranked_one(correct_query):
            categories.append("top_vs_grouped_semantic_drift")

    q_lower = question.lower().replace("tier 1", "tier1")
    selected_lower = selected_query.lower()
    correct_lower = correct_query.lower()
    for scope in ("oem", "tier1", "semiconductor"):
        if scope in q_lower and scope not in selected_lower and scope in correct_lower:
            categories.append("wrong_survey_or_scope")
            break

    if _query_count(selected_query) and not _query_count(correct_query) and not re.search(
        r"\b(count|how many|number of)\b", q_lower
    ):
        categories.append("count_selected_for_non_count_question")

    if not categories:
        categories.append("uncategorized_needs_manual_review")

    return {
        "id": detail.get("id"),
        "question": question,
        "first_correct_rank": candidates.index(first_correct) + 1,
        "categories": sorted(set(categories)),
        "selected_source": selected.get("source"),
        "correct_source": first_correct.get("source"),
        "selected_query": selected_query,
        "first_correct_query": correct_query,
    }


def analyze(results_path: str) -> Dict[str, Any]:
    payload = _load_json(results_path)
    details = list(payload.get("details") or [])
    cases = [case for detail in details if (case := _classify_case(detail))]
    counts = Counter(category for case in cases for category in case["categories"])
    return {
        "results": results_path,
        "summary": {
            "remaining_selection_failures": len(cases),
            "category_counts": counts.most_common(),
        },
        "cases": cases,
    }


def _render_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Remaining Failure Action Analysis",
        "",
        f"Results: `{report.get('results')}`",
        "",
        "## Summary",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in report["summary"]["category_counts"]:
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases") or []:
        lines.append(
            f"- `{case.get('id')}` rank={case.get('first_correct_rank')} "
            f"categories={', '.join(case.get('categories') or [])}: {case.get('question')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify remaining selection failures into action buckets.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(args.results)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(_render_md(report), encoding="utf-8")

    print("===== REMAINING FAILURE ACTION ANALYSIS =====")
    print(f"Results: {args.results}")
    print(f"Remaining selection failures: {report['summary']['remaining_selection_failures']}")
    for category, count in report["summary"]["category_counts"][:12]:
        print(f"  {category}: {count}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
