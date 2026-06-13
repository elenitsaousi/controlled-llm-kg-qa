#!/usr/bin/env python3
"""Explain what entropy regimes mean in candidate query selection.

This diagnostic is intentionally separate from the core metric scripts.  It
does not claim that entropy is a ground-truth ambiguity label.  It inspects the
cases behind each entropy bucket so the thesis discussion can distinguish:

* harmful ambiguity, where candidates represent competing meanings;
* harmless uncertainty, where candidates are near-duplicates;
* confident wrong selections, where low entropy still fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.compare_entropy_regime_selection import analyze as compare_entropy_regimes
from ranking.feature_extraction import extract_query_plan
from ranking.query_contract import (
    QueryContract,
    compare_contracts,
    extract_query_contract,
    extract_question_contract,
)


def _load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_details(path: Optional[str]) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {}
    details = payload.get("details")
    if not isinstance(details, list):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for idx, detail in enumerate(details):
        if isinstance(detail, dict):
            out[str(detail.get("id") or f"row_{idx}")] = detail
    return out


def _label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "") or "").strip().lower()


def _score(candidate: Dict[str, object], preferred: str = "ml_score") -> Optional[float]:
    for key in (preferred, "ml_score", "selection_score", "score", "semantic_judge_score"):
        value = candidate.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _one_line(text: object, limit: int = 260) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def _candidates(detail: Dict[str, object]) -> List[Dict[str, object]]:
    return [c for c in list(detail.get("candidates") or []) if isinstance(c, dict)]


def _first_correct(candidates: Sequence[Dict[str, object]]) -> Optional[Tuple[int, Dict[str, object]]]:
    for idx, candidate in enumerate(candidates):
        if _label(candidate) == "correct":
            return idx, candidate
    return None


def _listify(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted(str(v) for v in value if str(v))
    return [str(value)]


def _contract_signature(contract: QueryContract) -> Dict[str, object]:
    return {
        "metrics": sorted(contract.metrics),
        "aggregation": contract.aggregation,
        "scopes": sorted(contract.scopes),
        "dimensions": sorted(contract.dimensions),
        "filters": sorted(contract.filters),
        "answer_shape": contract.answer_shape,
    }


def _plan_signature(query: str) -> Dict[str, List[str]]:
    try:
        plan = extract_query_plan(query)
    except Exception:
        plan = {}
    keys = [
        "aggregations",
        "query_types",
        "group_by_vars",
        "group_by_predicates",
        "select_vars",
        "survey_origins",
        "classes",
        "predicates",
    ]
    return {key: _listify(plan.get(key)) for key in keys if plan.get(key)}


def _set_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = {str(v) for v in a if str(v)}
    right = {str(v) for v in b if str(v)}
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _contract_similarity(a: QueryContract, b: QueryContract) -> float:
    parts = [
        _set_jaccard(a.metrics, b.metrics),
        1.0 if a.aggregation == b.aggregation else 0.0,
        _set_jaccard(a.scopes, b.scopes),
        _set_jaccard(a.dimensions, b.dimensions),
        _set_jaccard(a.filters, b.filters),
        1.0 if a.answer_shape == b.answer_shape else 0.0,
    ]
    return sum(parts) / len(parts)


def _axis_differences(a: QueryContract, b: QueryContract) -> List[str]:
    diffs: List[str] = []
    if a.metrics != b.metrics:
        diffs.append("metric_difference")
    if a.aggregation != b.aggregation:
        diffs.append("aggregation_difference")
    if a.scopes != b.scopes:
        diffs.append("scope_difference")
    if a.dimensions != b.dimensions:
        diffs.append("dimension_difference")
    if a.filters != b.filters:
        diffs.append("filter_difference")
    if a.answer_shape != b.answer_shape:
        diffs.append("answer_shape_difference")
    return diffs


def _question_alignment_issues(question: str, top_query: str, correct_query: str) -> List[str]:
    q_contract = extract_question_contract(question)
    top_contract = extract_query_contract(top_query)
    correct_contract = extract_query_contract(correct_query)
    top_cmp = compare_contracts(q_contract, top_contract).to_dict()
    correct_cmp = compare_contracts(q_contract, correct_contract).to_dict()
    issues: List[str] = []

    for axis in ("metrics", "aggregation", "scopes", "dimensions", "filters", "answer_shape"):
        top_missing = set(_listify(dict(top_cmp.get("missing") or {}).get(axis)))
        top_conflicts = set(_listify(dict(top_cmp.get("conflicts") or {}).get(axis)))
        correct_matched = set(_listify(dict(correct_cmp.get("matched") or {}).get(axis)))
        if (top_missing | top_conflicts) & correct_matched:
            issues.append(f"top_misses_requested_{axis}")

    if not issues:
        issues.extend(_axis_differences(top_contract, correct_contract))
    return issues or ["no_contract_explanation_detected"]


def _top_entropy_candidates(row: Dict[str, object]) -> List[Dict[str, object]]:
    return [c for c in list(row.get("entropy_top_candidates") or []) if isinstance(c, dict)]


def _candidate_from_entropy(row: Dict[str, object], rank: int) -> Dict[str, object]:
    for candidate in _top_entropy_candidates(row):
        if int(candidate.get("rank") or 0) == rank:
            return candidate
    return {}


def _case_candidates(
    row: Dict[str, object],
    ml_details: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, object], Optional[Tuple[int, Dict[str, object]]], List[Dict[str, object]]]:
    qid = str(row.get("id") or "")
    detail = ml_details.get(qid, {})
    candidates = _candidates(detail)
    if candidates:
        return candidates[0], _first_correct(candidates), candidates
    top = {
        "query": row.get("ml_top_query") or _candidate_from_entropy(row, 1).get("query"),
        "label": "correct" if row.get("ml_top1_correct") else "valid_wrong",
        "score": row.get("score1"),
    }
    correct_entropy = next((c for c in _top_entropy_candidates(row) if _label(c) == "correct"), None)
    correct = (int(correct_entropy.get("rank") or 0) - 1, correct_entropy) if correct_entropy else None
    return top, correct, _top_entropy_candidates(row)


def _diagnose_wrong_case(
    row: Dict[str, object],
    *,
    top: Dict[str, object],
    correct: Optional[Tuple[int, Dict[str, object]]],
) -> Tuple[List[str], Dict[str, object]]:
    categories: List[str] = []
    evidence: Dict[str, object] = {}
    margin = float(row.get("margin") or 0.0)
    h_norm = float(row.get("normalized_entropy") or 0.0)

    if not row.get("any_correct") or correct is None:
        return ["no_correct_candidate"], {"reason": "No candidate labeled correct was available."}

    correct_rank, correct_candidate = correct
    top_query = str(top.get("query") or "")
    correct_query = str(correct_candidate.get("query") or "")
    question = str(row.get("question") or "")

    top_contract = extract_query_contract(top_query)
    correct_contract = extract_query_contract(correct_query)
    similarity = _contract_similarity(top_contract, correct_contract)
    categories.extend(_question_alignment_issues(question, top_query, correct_query))

    if similarity >= 0.82:
        categories.append("near_duplicate_or_minor_surface_difference")
    if margin >= 0.15:
        categories.append("confident_wrong_selection")
    elif margin <= 0.05:
        categories.append("low_margin_competition")
    if correct_rank >= 1:
        categories.append(f"correct_at_rank_{correct_rank + 1}")

    evidence.update(
        {
            "contract_similarity_top_vs_correct": similarity,
            "normalized_entropy": h_norm,
            "margin": margin,
            "correct_rank": correct_rank + 1,
            "top_contract": _contract_signature(top_contract),
            "correct_contract": _contract_signature(correct_contract),
            "top_plan": _plan_signature(top_query),
            "correct_plan": _plan_signature(correct_query),
        }
    )
    return sorted(set(categories)), evidence


def _diagnose_correct_case(row: Dict[str, object], *, top: Dict[str, object], candidates: List[Dict[str, object]]) -> Tuple[List[str], Dict[str, object]]:
    categories: List[str] = []
    evidence: Dict[str, object] = {}
    top_query = str(top.get("query") or row.get("ml_top_query") or "")
    top_contract = extract_query_contract(top_query)
    margin = float(row.get("margin") or 0.0)

    competitor = None
    if len(candidates) > 1:
        competitor = candidates[1]
    elif len(_top_entropy_candidates(row)) > 1:
        competitor = _candidate_from_entropy(row, 2)

    if competitor:
        competitor_query = str(competitor.get("query") or "")
        competitor_contract = extract_query_contract(competitor_query)
        similarity = _contract_similarity(top_contract, competitor_contract)
        if similarity >= 0.82:
            categories.append("harmless_high_entropy_near_duplicate")
        else:
            categories.append("ranker_correct_despite_semantic_competition")
        evidence["contract_similarity_top_vs_second"] = similarity
        evidence["second_contract"] = _contract_signature(competitor_contract)
        evidence["second_query"] = _one_line(competitor_query)

    if margin <= 0.05:
        categories.append("low_margin_but_correct")
    elif margin >= 0.15:
        categories.append("clear_margin_correct")
    evidence["top_contract"] = _contract_signature(top_contract)
    evidence["margin"] = margin
    evidence["normalized_entropy"] = float(row.get("normalized_entropy") or 0.0)
    return sorted(set(categories)) or ["correct_no_extra_explanation"], evidence


def _summarize_subset(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "count": 0,
            "ml_accuracy": 0.0,
            "baseline_accuracy": 0.0,
            "any_correct_rate": 0.0,
            "avg_entropy": 0.0,
            "avg_margin": 0.0,
        }
    return {
        "count": len(rows),
        "ml_accuracy": sum(1 for r in rows if r.get("ml_top1_correct")) / len(rows),
        "baseline_accuracy": sum(1 for r in rows if r.get("baseline_top1_correct")) / len(rows),
        "any_correct_rate": sum(1 for r in rows if r.get("any_correct")) / len(rows),
        "avg_entropy": mean(float(r.get("normalized_entropy") or 0.0) for r in rows),
        "avg_margin": mean(float(r.get("margin") or 0.0) for r in rows),
    }


def analyze_diagnostics(
    *,
    comparison_report: Dict[str, object],
    ml_results_path: Optional[str],
    example_limit: int,
) -> Dict[str, object]:
    ml_details = _load_details(ml_results_path)
    rows = [r for r in list(comparison_report.get("cases") or []) if isinstance(r, dict)]
    by_subset: Dict[str, List[Dict[str, object]]] = {
        "low_entropy_wrong": [],
        "medium_entropy_wrong": [],
        "high_entropy_correct": [],
        "high_entropy_wrong": [],
    }
    for row in rows:
        regime = str(row.get("entropy_regime") or "unknown")
        correct = bool(row.get("ml_top1_correct"))
        if regime == "low" and not correct:
            by_subset["low_entropy_wrong"].append(row)
        elif regime == "medium" and not correct:
            by_subset["medium_entropy_wrong"].append(row)
        elif regime == "high" and correct:
            by_subset["high_entropy_correct"].append(row)
        elif regime == "high" and not correct:
            by_subset["high_entropy_wrong"].append(row)

    subset_reports: Dict[str, Dict[str, object]] = {}
    global_category_counts: Counter = Counter()
    for subset, subset_rows in by_subset.items():
        category_counts: Counter = Counter()
        cases: List[Dict[str, object]] = []
        for row in subset_rows:
            top, first_correct, candidates = _case_candidates(row, ml_details)
            if row.get("ml_top1_correct"):
                categories, evidence = _diagnose_correct_case(row, top=top, candidates=candidates)
            else:
                categories, evidence = _diagnose_wrong_case(row, top=top, correct=first_correct)
            for category in categories:
                category_counts[category] += 1
                global_category_counts[f"{subset}:{category}"] += 1
            if len(cases) < example_limit:
                cases.append(
                    {
                        "id": row.get("id"),
                        "question": row.get("question"),
                        "family": row.get("family"),
                        "entropy_regime": row.get("entropy_regime"),
                        "normalized_entropy": row.get("normalized_entropy"),
                        "margin": row.get("margin"),
                        "baseline_top1_correct": row.get("baseline_top1_correct"),
                        "ml_top1_correct": row.get("ml_top1_correct"),
                        "any_correct": row.get("any_correct"),
                        "baseline_first_correct_rank": row.get("baseline_first_correct_rank"),
                        "ml_first_correct_rank": row.get("ml_first_correct_rank"),
                        "categories": categories,
                        "evidence": evidence,
                        "top_query": _one_line(top.get("query") or row.get("ml_top_query")),
                        "correct_query": _one_line(first_correct[1].get("query")) if first_correct else "",
                        "entropy_top_candidates": row.get("entropy_top_candidates"),
                    }
                )
        subset_reports[subset] = {
            "summary": _summarize_subset(subset_rows),
            "category_counts": category_counts.most_common(),
            "examples": cases,
        }

    return {
        "inputs": comparison_report.get("inputs") or {},
        "metric_summary": comparison_report.get("summary") or {},
        "by_entropy_regime": comparison_report.get("by_entropy_regime") or [],
        "diagnostic_interpretation": _interpret(subset_reports),
        "diagnostic_category_counts": global_category_counts.most_common(),
        "subsets": subset_reports,
    }


def _interpret(subset_reports: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    high_correct_categories = Counter(dict(subset_reports.get("high_entropy_correct", {}).get("category_counts") or []))
    low_wrong_categories = Counter(dict(subset_reports.get("low_entropy_wrong", {}).get("category_counts") or []))
    medium_wrong_categories = Counter(dict(subset_reports.get("medium_entropy_wrong", {}).get("category_counts") or []))

    supported: List[str] = []
    caveats: List[str] = []

    high_correct_total = int(subset_reports.get("high_entropy_correct", {}).get("summary", {}).get("count") or 0)
    high_near_dup = high_correct_categories.get("harmless_high_entropy_near_duplicate", 0)
    if high_correct_total and high_near_dup / high_correct_total >= 0.4:
        supported.append(
            "Many high-entropy correct cases are near-duplicate competitions, so high score entropy is often harmless candidate uncertainty rather than true semantic ambiguity."
        )
    elif high_correct_total:
        caveats.append(
            "High-entropy correct cases are not dominated by near-duplicates; inspect them before claiming high entropy is mostly harmless."
        )

    low_wrong_total = int(subset_reports.get("low_entropy_wrong", {}).get("summary", {}).get("count") or 0)
    low_confident = low_wrong_categories.get("confident_wrong_selection", 0)
    if low_wrong_total and low_confident / low_wrong_total >= 0.3:
        supported.append(
            "Low entropy can still fail through confident wrong selections; entropy is therefore not a ground-truth difficulty measure."
        )
    elif low_wrong_total:
        caveats.append(
            "Low-entropy wrong cases are not mostly high-margin errors; check whether the low regime is being defined too broadly."
        )

    medium_wrong_total = int(subset_reports.get("medium_entropy_wrong", {}).get("summary", {}).get("count") or 0)
    medium_semantic = sum(
        medium_wrong_categories.get(key, 0)
        for key in (
            "aggregation_difference",
            "dimension_difference",
            "scope_difference",
            "answer_shape_difference",
            "top_misses_requested_aggregation",
            "top_misses_requested_dimensions",
            "top_misses_requested_scopes",
            "top_misses_requested_answer_shape",
        )
    )
    if medium_wrong_total and medium_semantic / medium_wrong_total >= 0.4:
        supported.append(
            "Medium-regime failures are frequently semantic competitions over aggregation, dimension, scope, or answer shape."
        )
    elif medium_wrong_total:
        caveats.append(
            "Medium-regime failures are not clearly explained by semantic-axis differences; the current entropy bucket may not isolate harmful ambiguity."
        )

    return {
        "supported_claims": supported,
        "caveats": caveats,
        "recommended_thesis_wording": (
            "Candidate-score entropy should be described as an operational proxy for selection uncertainty. "
            "The diagnostics show where this proxy aligns with harmful semantic ambiguity and where it does not."
        ),
    }


def _pct(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def write_markdown(report: Dict[str, object], out_path: str) -> None:
    lines: List[str] = []
    lines.append("# Entropy Regime Diagnostic Evidence")
    lines.append("")
    lines.append("This report tests whether candidate-score entropy explains selection behavior. It treats entropy as an operational proxy for selector uncertainty, not as a ground-truth semantic ambiguity label.")
    lines.append("")

    interpretation = dict(report.get("diagnostic_interpretation") or {})
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"Recommended thesis wording: {interpretation.get('recommended_thesis_wording')}")
    lines.append("")
    lines.append("### Supported By This Diagnostic")
    lines.append("")
    supported = interpretation.get("supported_claims") or []
    if supported:
        for item in supported:
            lines.append(f"- {item}")
    else:
        lines.append("- No strong diagnostic claim was automatically supported. Use the case tables as qualitative evidence.")
    lines.append("")
    lines.append("### Caveats")
    lines.append("")
    caveats = interpretation.get("caveats") or []
    if caveats:
        for item in caveats:
            lines.append(f"- {item}")
    else:
        lines.append("- No major caveat was triggered by the automatic thresholds.")
    lines.append("")

    lines.append("## Regime Metrics")
    lines.append("")
    lines.append("| Regime | Count | Baseline Top-1 | ML Top-1 | Delta | Any correct | Avg H_norm | Avg margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report.get("by_entropy_regime") or []:
        lines.append(
            f"| `{row.get('regime')}` | {row.get('count')} | "
            f"{row.get('baseline_correct')} ({_pct(row.get('baseline_accuracy'))}) | "
            f"{row.get('ml_correct')} ({_pct(row.get('ml_accuracy'))}) | "
            f"{row.get('delta_correct')} ({_pct(row.get('delta_accuracy'))}) | "
            f"{row.get('any_correct')} ({_pct(row.get('any_correct_rate'))}) | "
            f"{_pct(row.get('avg_normalized_entropy'))} | {_pct(row.get('avg_margin'))} |"
        )
    lines.append("")

    lines.append("## Diagnostic Subsets")
    lines.append("")
    for subset, payload in (report.get("subsets") or {}).items():
        summary = dict(payload.get("summary") or {})
        lines.append(f"### {subset.replace('_', ' ').title()}")
        lines.append("")
        lines.append(
            f"Count: {summary.get('count', 0)}, ML accuracy: {_pct(summary.get('ml_accuracy'))}, "
            f"Any-correct: {_pct(summary.get('any_correct_rate'))}, "
            f"Avg H_norm: {_pct(summary.get('avg_entropy'))}, Avg margin: {_pct(summary.get('avg_margin'))}"
        )
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|---|---:|")
        for category, count in payload.get("category_counts") or []:
            lines.append(f"| `{category}` | {count} |")
        lines.append("")
        for case in payload.get("examples") or []:
            lines.append(f"#### {case.get('id')} - {case.get('family')}")
            lines.append("")
            lines.append(f"Question: {case.get('question')}")
            lines.append("")
            lines.append(
                f"- H_norm={float(case.get('normalized_entropy') or 0.0):.4f}, "
                f"margin={float(case.get('margin') or 0.0):.4f}, "
                f"baseline_correct={case.get('baseline_top1_correct')}, "
                f"ml_correct={case.get('ml_top1_correct')}, "
                f"any_correct={case.get('any_correct')}"
            )
            lines.append(f"- Categories: {', '.join(case.get('categories') or [])}")
            lines.append(f"- Top query: `{case.get('top_query')}`")
            if case.get("correct_query"):
                lines.append(f"- First correct query: `{case.get('correct_query')}`")
            lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create evidence for interpreting entropy ambiguity regimes.")
    parser.add_argument("--comparison-json", default="", help="Existing output from compare_entropy_regime_selection.py.")
    parser.add_argument("--baseline-results", default="", help="Baseline results if --comparison-json is not supplied.")
    parser.add_argument("--ml-results", default="", help="ML results if --comparison-json is not supplied.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--entropy-source", default="ml", choices=["baseline", "ml"])
    parser.add_argument("--score-key", default="ml_score")
    parser.add_argument("--sort-by-score", action="store_true")
    parser.add_argument("--normalization", default="softmax", choices=["auto", "positive", "softmax", "minmax"])
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--bucket-mode", default="quantiles", choices=["thresholds", "quantiles"])
    parser.add_argument("--low-threshold", type=float, default=0.33)
    parser.add_argument("--high-threshold", type=float, default=0.66)
    parser.add_argument("--example-limit", type=int, default=10)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    if args.comparison_json:
        comparison = _load_json(args.comparison_json)
        if not isinstance(comparison, dict):
            raise ValueError("--comparison-json must contain a JSON object.")
        ml_results_path = str(dict(comparison.get("inputs") or {}).get("ml_results") or args.ml_results or "")
    else:
        if not args.baseline_results or not args.ml_results:
            raise ValueError("Provide either --comparison-json or both --baseline-results and --ml-results.")
        comparison = compare_entropy_regimes(
            baseline_results_path=args.baseline_results,
            ml_results_path=args.ml_results,
            dataset_path=args.dataset,
            entropy_source=args.entropy_source,
            score_key=args.score_key,
            sort_by_score=args.sort_by_score,
            normalization=args.normalization,
            temperature=args.temperature,
            bucket_mode=args.bucket_mode,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            example_limit=args.example_limit,
        )
        ml_results_path = args.ml_results

    report = analyze_diagnostics(
        comparison_report=comparison,
        ml_results_path=ml_results_path,
        example_limit=args.example_limit,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_markdown(report, args.out_md)

    print("===== ENTROPY REGIME DIAGNOSTIC EVIDENCE =====")
    for subset, payload in report["subsets"].items():
        summary = payload["summary"]
        print(
            f"{subset}: count={summary['count']}, ml_acc={summary['ml_accuracy']:.3f}, "
            f"avg_H={summary['avg_entropy']:.3f}, avg_margin={summary['avg_margin']:.3f}"
        )
        for category, count in payload["category_counts"][:8]:
            print(f"  {category}: {count}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
