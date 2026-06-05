#!/usr/bin/env python3
"""Diagnose why selection picks a wrong candidate when a correct one exists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.feature_extraction import extract_query_plan
from ranking.query_contract import (
    compare_contracts,
    extract_query_contract,
    extract_question_contract,
)
from validation.semantic import semantic_coverage_report, semantic_judge_report


def _load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "") or "").strip().lower()


def _one_line_query(query: str, limit: int = 320) -> str:
    text = " ".join(str(query or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _first_correct(candidates: List[Dict[str, object]]) -> Optional[Tuple[int, Dict[str, object]]]:
    for idx, candidate in enumerate(candidates):
        if _label(candidate) == "correct":
            return idx, candidate
    return None


def _dataset_by_id(path: Optional[str]) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}
    rows = _load_json(path)
    if not isinstance(rows, list):
        return {}
    return {str(row.get("id", "")): row for row in rows if isinstance(row, dict)}


def _as_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    if value is None:
        return []
    return [str(value)]


def _axis_set(report: Dict[str, object], section: str, axis: str) -> set:
    payload = report.get(section)
    if not isinstance(payload, dict):
        return set()
    return {str(v) for v in _as_list(payload.get(axis))}


def _axis_issue_summary(top_report: Dict[str, object], correct_report: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    for axis in ("metrics", "aggregation", "scopes", "dimensions", "filters", "answer_shape"):
        top_missing = _axis_set(top_report, "missing", axis)
        top_conflicts = _axis_set(top_report, "conflicts", axis)
        correct_matches = _axis_set(correct_report, "matched", axis)
        corrected = (top_missing | top_conflicts) & correct_matches
        if corrected:
            issues.append(f"{axis}:{','.join(sorted(corrected))}")
        elif top_missing:
            issues.append(f"{axis}_missing:{','.join(sorted(top_missing))}")
        elif top_conflicts:
            issues.append(f"{axis}_conflict:{','.join(sorted(top_conflicts))}")
    return issues


def _semantic_summary(question: str, candidate: Dict[str, object]) -> Dict[str, object]:
    query = str(candidate.get("query", "") or "")
    try:
        judge = semantic_judge_report(question, query)
    except Exception as exc:
        judge = {"score": 0.0, "error": str(exc), "penalties": []}
    try:
        coverage = semantic_coverage_report(question, query)
    except Exception as exc:
        coverage = {"coverage_score": 0.0, "missing": [], "error": str(exc)}
    return {
        "judge_score": float(judge.get("score", candidate.get("semantic_judge_score") or 0.0)),
        "penalties": _as_list(judge.get("penalties")),
        "coverage_score": float(coverage.get("coverage_score", 0.0)),
        "coverage_missing": _as_list(coverage.get("missing")),
    }


def _plan_summary(query: str, schema: Dict[str, object]) -> Dict[str, object]:
    try:
        plan = extract_query_plan(query, schema)
    except Exception as exc:
        return {"error": str(exc)}
    keys = [
        "labels",
        "classes",
        "predicates",
        "aggregations",
        "query_types",
        "group_by_vars",
        "group_by_predicates",
        "survey_origins",
        "select_vars",
    ]
    return {key: _as_list(plan.get(key)) for key in keys if plan.get(key)}


def _list_set(plan: Dict[str, object], key: str) -> set:
    return {str(v) for v in _as_list(plan.get(key))}


def _plan_issue_summary(
    *,
    top_plan: Dict[str, object],
    correct_plan: Dict[str, object],
    top_candidate: Dict[str, object],
    correct_candidate: Dict[str, object],
    top_semantic: Dict[str, object],
    correct_semantic: Dict[str, object],
) -> List[str]:
    issues: List[str] = []
    for key, name in [
        ("aggregations", "plan_aggregation_diff"),
        ("query_types", "plan_query_type_diff"),
        ("group_by_vars", "plan_group_by_var_diff"),
        ("group_by_predicates", "plan_group_by_predicate_diff"),
        ("select_vars", "plan_select_var_diff"),
        ("survey_origins", "plan_origin_diff"),
        ("classes", "plan_class_diff"),
        ("predicates", "plan_predicate_diff"),
    ]:
        top_values = _list_set(top_plan, key)
        correct_values = _list_set(correct_plan, key)
        if top_values != correct_values:
            missing_from_top = correct_values - top_values
            extra_in_top = top_values - correct_values
            if missing_from_top:
                issues.append(f"{name}:missing:{','.join(sorted(missing_from_top)[:4])}")
            elif extra_in_top:
                issues.append(f"{name}:extra:{','.join(sorted(extra_in_top)[:4])}")
            else:
                issues.append(name)

    top_source = str(top_candidate.get("source") or "unknown")
    correct_source = str(correct_candidate.get("source") or "unknown")
    if top_source != correct_source:
        issues.append(f"source_diff:{top_source}->{correct_source}")

    top_selection = _score(top_candidate, "selection_score")
    correct_selection = _score(correct_candidate, "selection_score")
    if top_selection > correct_selection:
        issues.append("selection_score_prefers_wrong")
    elif correct_selection > top_selection:
        issues.append("selection_score_prefers_correct")

    top_ml = _score(top_candidate, "ml_score")
    correct_ml = _score(correct_candidate, "ml_score")
    if top_ml > correct_ml:
        issues.append("ml_score_prefers_wrong")
    elif correct_ml > top_ml:
        issues.append("ml_score_prefers_correct")

    top_judge = float(top_semantic.get("judge_score", 0.0))
    correct_judge = float(correct_semantic.get("judge_score", 0.0))
    if top_judge > correct_judge:
        issues.append("semantic_score_prefers_wrong")
    elif correct_judge > top_judge:
        issues.append("semantic_score_prefers_correct")

    top_coverage = float(top_semantic.get("coverage_score", 0.0))
    correct_coverage = float(correct_semantic.get("coverage_score", 0.0))
    if top_coverage > correct_coverage:
        issues.append("coverage_prefers_wrong")
    elif correct_coverage > top_coverage:
        issues.append("coverage_prefers_correct")

    return issues or ["no_plan_difference_detected"]


def _query_type_set(plan: Dict[str, object]) -> set:
    return {str(v).lower() for v in _as_list(plan.get("query_types"))}


def _plan_has_grouped_shortage_status(plan: Dict[str, object]) -> bool:
    text = " ".join(
        str(value).lower()
        for key in ("group_by_vars", "group_by_predicates", "select_vars", "labels", "predicates")
        for value in _as_list(plan.get(key))
    )
    compact = text.replace("_", "").replace(" ", "")
    return "shortagestatus" in compact or "reportsshortage" in compact or "shortagelabel" in compact


def _semantic_category_opportunities(
    *,
    question_contract,
    top_plan: Dict[str, object],
    correct_plan: Dict[str, object],
    top_candidate: Dict[str, object],
    correct_candidate: Dict[str, object],
) -> List[str]:
    opportunities: List[str] = []
    top_types = _query_type_set(top_plan)
    correct_types = _query_type_set(correct_plan)
    top_query = str(top_candidate.get("query", "") or "").lower()
    correct_query = str(correct_candidate.get("query", "") or "").lower()

    if question_contract.answer_shape == "grouped_table":
        if "grouped" not in top_types and "grouped" in correct_types:
            opportunities.append("query_shape:grouped_missing_in_top")
        if ({"ranking", "limited"} & top_types) and not ({"ranking", "limited"} & correct_types):
            opportunities.append("query_shape:ranking_extra_in_top")

    if question_contract.answer_shape == "ranked_one":
        if not ({"ranking", "limited"} & top_types) and ({"ranking", "limited"} & correct_types):
            opportunities.append("query_shape:ranking_missing_in_top")

    if question_contract.aggregation:
        top_aggs = _list_set(top_plan, "aggregations")
        correct_aggs = _list_set(correct_plan, "aggregations")
        requested = str(question_contract.aggregation).upper()
        if requested == "RANK":
            requested = "ranking"
        elif requested not in {"SUM", "COUNT", "AVG"}:
            requested = str(question_contract.aggregation)
        if requested in {"SUM", "COUNT", "AVG"} and requested not in top_aggs and requested in correct_aggs:
            opportunities.append(f"aggregation:{requested.lower()}_missing_in_top")
        if top_aggs and correct_aggs and top_aggs != correct_aggs:
            opportunities.append("aggregation:top_correct_mismatch")

    if question_contract.answer_shape == "list_values":
        if "select distinct" not in top_query and "select distinct" in correct_query:
            opportunities.append("answer_shape:distinct_list_missing_in_top")
        if "count(" in top_query and "count(" not in correct_query:
            opportunities.append("answer_shape:count_extra_for_list")

    if "shortage_status" in getattr(question_contract, "dimensions", set()):
        if not _plan_has_grouped_shortage_status(top_plan) and _plan_has_grouped_shortage_status(correct_plan):
            opportunities.append("dimension:shortage_status_missing_in_top")

    top_group_vars = _list_set(top_plan, "group_by_vars")
    correct_group_vars = _list_set(correct_plan, "group_by_vars")
    if correct_group_vars - top_group_vars:
        opportunities.append("group_by:vars_missing_in_top")

    top_select_vars = _list_set(top_plan, "select_vars")
    correct_select_vars = _list_set(correct_plan, "select_vars")
    if correct_select_vars - top_select_vars:
        opportunities.append("select:vars_missing_in_top")

    return sorted(set(opportunities)) or ["no_semantic_opportunity_detected"]


def _score(candidate: Dict[str, object], key: str, default: float = 0.0) -> float:
    value = candidate.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def analyze_selection_failures(
    *,
    results_path: str,
    dataset_path: Optional[str],
    schema_path: str,
    max_cases: int = 0,
) -> Dict[str, object]:
    results = _load_json(results_path)
    schema = _load_json(schema_path)
    if not isinstance(results, dict):
        raise ValueError("Results JSON must be an object.")
    if not isinstance(schema, dict):
        raise ValueError("Schema JSON must be an object.")
    dataset = _dataset_by_id(dataset_path)

    axis_counts: Counter = Counter()
    plan_issue_counts: Counter = Counter()
    correct_rank_counts: Counter = Counter()
    top_source_counts: Counter = Counter()
    correct_source_counts: Counter = Counter()
    family_counts: Dict[str, Counter] = defaultdict(Counter)
    opportunity_counts: Counter = Counter()
    margin_buckets: Counter = Counter()
    cases: List[Dict[str, object]] = []

    for detail in results.get("details") or []:
        if not isinstance(detail, dict):
            continue
        candidates = list(detail.get("candidates") or [])
        if not candidates or _label(candidates[0]) == "correct":
            continue
        correct = _first_correct(candidates)
        if correct is None:
            continue

        correct_idx, correct_candidate = correct
        top_candidate = candidates[0]
        qid = str(detail.get("id", "") or "")
        gold = dataset.get(qid, {})
        question = str(
            detail.get("effective_question")
            or detail.get("question")
            or gold.get("question")
            or ""
        )
        family = str(gold.get("topic") or gold.get("family") or detail.get("family") or "unknown")

        question_contract = extract_question_contract(question)
        top_contract = extract_query_contract(str(top_candidate.get("query", "") or ""))
        correct_contract = extract_query_contract(str(correct_candidate.get("query", "") or ""))
        top_comparison = compare_contracts(question_contract, top_contract).to_dict()
        correct_comparison = compare_contracts(question_contract, correct_contract).to_dict()
        issues = _axis_issue_summary(top_comparison, correct_comparison)
        if not issues:
            issues = ["no_contract_axis_difference"]
        for issue in issues:
            axis_counts[issue] += 1

        correct_rank_counts[str(correct_idx + 1)] += 1
        top_source_counts[str(top_candidate.get("source") or "unknown")] += 1
        correct_source_counts[str(correct_candidate.get("source") or "unknown")] += 1
        family_counts[family]["selection_failures"] += 1
        family_counts[family][issues[0]] += 1

        top_semantic = _semantic_summary(question, top_candidate)
        correct_semantic = _semantic_summary(question, correct_candidate)
        top_plan = _plan_summary(str(top_candidate.get("query", "") or ""), schema)
        correct_plan = _plan_summary(str(correct_candidate.get("query", "") or ""), schema)
        plan_issues = _plan_issue_summary(
            top_plan=top_plan,
            correct_plan=correct_plan,
            top_candidate=top_candidate,
            correct_candidate=correct_candidate,
            top_semantic=top_semantic,
            correct_semantic=correct_semantic,
        )
        for issue in plan_issues:
            plan_issue_counts[issue] += 1
        opportunities = _semantic_category_opportunities(
            question_contract=question_contract,
            top_plan=top_plan,
            correct_plan=correct_plan,
            top_candidate=top_candidate,
            correct_candidate=correct_candidate,
        )
        for opportunity in opportunities:
            opportunity_counts[opportunity] += 1
        ml_margin = _score(top_candidate, "ml_score") - _score(correct_candidate, "ml_score")
        if ml_margin < 0:
            margin_buckets["ml_correct_scores_higher"] += 1
        elif ml_margin < 0.03:
            margin_buckets["ml_margin_0_to_0.03"] += 1
        elif ml_margin < 0.10:
            margin_buckets["ml_margin_0.03_to_0.10"] += 1
        elif ml_margin < 0.25:
            margin_buckets["ml_margin_0.10_to_0.25"] += 1
        else:
            margin_buckets["ml_margin_over_0.25"] += 1
        case = {
            "id": qid,
            "family": family,
            "question": question,
            "first_correct_rank": correct_idx + 1,
            "axis_issues": issues,
            "plan_issues": plan_issues,
            "semantic_opportunities": opportunities,
            "question_contract": question_contract.to_dict(),
            "top1": {
                "label": _label(top_candidate),
                "source": top_candidate.get("source"),
                "selection_score": top_candidate.get("selection_score"),
                "ml_score": top_candidate.get("ml_score"),
                "semantic": top_semantic,
                "contract": top_contract.to_dict(),
                "contract_comparison": top_comparison,
                "plan": top_plan,
                "query": str(top_candidate.get("query", "") or ""),
            },
            "first_correct": {
                "rank": correct_idx + 1,
                "source": correct_candidate.get("source"),
                "selection_score": correct_candidate.get("selection_score"),
                "ml_score": correct_candidate.get("ml_score"),
                "semantic": correct_semantic,
                "contract": correct_contract.to_dict(),
                "contract_comparison": correct_comparison,
                "plan": correct_plan,
                "query": str(correct_candidate.get("query", "") or ""),
            },
            "deltas": {
                "contract_score": float(correct_comparison.get("score", 0.0))
                - float(top_comparison.get("score", 0.0)),
                "semantic_score": float(correct_semantic.get("judge_score", 0.0))
                - float(top_semantic.get("judge_score", 0.0)),
                "coverage_score": float(correct_semantic.get("coverage_score", 0.0))
                - float(top_semantic.get("coverage_score", 0.0)),
                "selection_score": _score(correct_candidate, "selection_score")
                - _score(top_candidate, "selection_score"),
                "ml_score": _score(correct_candidate, "ml_score")
                - _score(top_candidate, "ml_score"),
                "ml_margin_top_minus_correct": ml_margin,
            },
        }
        cases.append(case)

    total_cases = len(cases)
    rendered_cases = cases[:max_cases] if max_cases and max_cases > 0 else cases

    return {
        "summary": {
            "results": results_path,
            "dataset": dataset_path,
            "selection_failures_with_correct_candidate": total_cases,
            "rendered_cases": len(rendered_cases),
            "axis_issue_counts": axis_counts.most_common(),
            "plan_issue_counts": plan_issue_counts.most_common(),
            "semantic_opportunity_counts": opportunity_counts.most_common(),
            "ml_margin_buckets": margin_buckets.most_common(),
            "first_correct_rank_counts": sorted(
                ((int(k), v) for k, v in correct_rank_counts.items()), key=lambda row: row[0]
            ),
            "top_source_counts": top_source_counts.most_common(),
            "first_correct_source_counts": correct_source_counts.most_common(),
            "family_counts": {
                family: dict(counter) for family, counter in sorted(family_counts.items())
            },
        },
        "cases": rendered_cases,
    }


def _format_list(values: Iterable[object], limit: int = 12) -> str:
    items = [str(v) for v in values if str(v)]
    if not items:
        return "-"
    if len(items) > limit:
        return ", ".join(items[:limit]) + f", ... (+{len(items) - limit})"
    return ", ".join(items)


def write_markdown(report: Dict[str, object], out_path: str) -> None:
    summary = dict(report.get("summary") or {})
    lines: List[str] = []
    lines.append("# Selection Failure Diagnostics")
    lines.append("")
    lines.append(f"- Results: `{summary.get('results')}`")
    lines.append(f"- Dataset: `{summary.get('dataset')}`")
    lines.append(
        f"- Ranking failures with correct candidate: {summary.get('selection_failures_with_correct_candidate', 0)}"
    )
    lines.append("")
    lines.append("## Axis Issues")
    lines.append("")
    lines.append("| Issue | Count |")
    lines.append("|---|---:|")
    for issue, count in summary.get("axis_issue_counts") or []:
        lines.append(f"| `{issue}` | {count} |")
    lines.append("")
    lines.append("## First Correct Rank")
    lines.append("")
    lines.append("| Rank | Count |")
    lines.append("|---:|---:|")
    for rank, count in summary.get("first_correct_rank_counts") or []:
        lines.append(f"| {rank} | {count} |")
    lines.append("")
    lines.append("## Plan Issues")
    lines.append("")
    lines.append("| Issue | Count |")
    lines.append("|---|---:|")
    for issue, count in summary.get("plan_issue_counts") or []:
        lines.append(f"| `{issue}` | {count} |")
    lines.append("")
    lines.append("## Semantic Opportunities")
    lines.append("")
    lines.append("| Opportunity | Count |")
    lines.append("|---|---:|")
    for opportunity, count in summary.get("semantic_opportunity_counts") or []:
        lines.append(f"| `{opportunity}` | {count} |")
    lines.append("")
    lines.append("## ML Margin Buckets")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|---|---:|")
    for bucket, count in summary.get("ml_margin_buckets") or []:
        lines.append(f"| `{bucket}` | {count} |")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    for case in report.get("cases") or []:
        top = dict(case.get("top1") or {})
        correct = dict(case.get("first_correct") or {})
        deltas = dict(case.get("deltas") or {})
        lines.append(f"### {case.get('id')} - {case.get('family')}")
        lines.append("")
        lines.append(f"Question: {case.get('question')}")
        lines.append("")
        lines.append(f"- First correct rank: {case.get('first_correct_rank')}")
        lines.append(f"- Axis issues: {_format_list(case.get('axis_issues') or [])}")
        lines.append(f"- Plan issues: {_format_list(case.get('plan_issues') or [])}")
        lines.append(f"- Semantic opportunities: {_format_list(case.get('semantic_opportunities') or [])}")
        lines.append(
            "- Deltas correct-minus-top1: "
            f"contract={deltas.get('contract_score', 0):.3f}, "
            f"semantic={deltas.get('semantic_score', 0):.3f}, "
            f"coverage={deltas.get('coverage_score', 0):.3f}, "
            f"selection={deltas.get('selection_score', 0):.3f}, "
            f"ml={deltas.get('ml_score', 0):.3f}"
        )
        lines.append("")
        top_sem = dict(top.get("semantic") or {})
        correct_sem = dict(correct.get("semantic") or {})
        lines.append(
            f"Top1 wrong: source={top.get('source')}, "
            f"semantic={top_sem.get('judge_score')}, coverage={top_sem.get('coverage_score')}, "
            f"contract={dict(top.get('contract_comparison') or {}).get('score')}"
        )
        lines.append("")
        lines.append(f"`{_one_line_query(str(top.get('query') or ''))}`")
        lines.append("")
        lines.append(
            f"First correct: rank={correct.get('rank')}, source={correct.get('source')}, "
            f"semantic={correct_sem.get('judge_score')}, coverage={correct_sem.get('coverage_score')}, "
            f"contract={dict(correct.get('contract_comparison') or {}).get('score')}"
        )
        lines.append("")
        lines.append(f"`{_one_line_query(str(correct.get('query') or ''))}`")
        lines.append("")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ranking failures where a correct candidate exists but was not selected."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    report = analyze_selection_failures(
        results_path=args.results,
        dataset_path=args.dataset or None,
        schema_path=args.schema,
        max_cases=args.max_cases,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_markdown(report, args.out_md)

    summary = report["summary"]
    print("===== SELECTION FAILURE DIAGNOSTICS =====")
    print(f"Results: {args.results}")
    print(f"Failures with correct candidate: {summary['selection_failures_with_correct_candidate']}")
    print("Top axis issues:")
    for issue, count in summary["axis_issue_counts"][:15]:
        print(f"  {issue}: {count}")
    print("Top plan issues:")
    for issue, count in summary["plan_issue_counts"][:15]:
        print(f"  {issue}: {count}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
