#!/usr/bin/env python3
"""Inspect high-confidence wrong KGQA selections."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.analyze_selection_failures import (
    _axis_issue_summary,
    _dataset_by_id,
    _label,
    _latent_issue_summary,
    _one_line_query,
    _plan_issue_summary,
    _plan_summary,
    _semantic_category_opportunities,
    _semantic_summary,
)
from ranking.query_contract import compare_contracts, extract_query_contract, extract_question_contract


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Results JSON must be an object.")
    return payload


def _score(candidate: Dict[str, object], preferred: str) -> float:
    keys = [preferred]
    if preferred != "ml_score":
        keys.append("ml_score")
    if preferred != "selection_score":
        keys.append("selection_score")
    keys.extend(["score", "semantic_judge_score"])
    for key in keys:
        value = candidate.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _rank_candidates(
    detail: Dict[str, object],
    *,
    score_key: str,
    sort_by_score: bool,
) -> List[Dict[str, object]]:
    candidates = [c for c in list(detail.get("candidates") or []) if isinstance(c, dict)]
    if sort_by_score:
        return sorted(candidates, key=lambda c: _score(c, score_key), reverse=True)
    return candidates


def _first_correct(candidates: List[Dict[str, object]]) -> Optional[Tuple[int, Dict[str, object]]]:
    for idx, candidate in enumerate(candidates):
        if _label(candidate) == "correct":
            return idx, candidate
    return None


def _question(detail: Dict[str, object], gold: Dict[str, object]) -> str:
    return str(
        detail.get("effective_question")
        or detail.get("question")
        or gold.get("question")
        or ""
    )


def analyze(
    *,
    results_path: str,
    dataset_path: str,
    schema_path: str,
    score_key: str,
    min_score: float,
    min_margin: float,
    sort_by_score: bool,
) -> Dict[str, object]:
    results = _load_json(results_path)
    schema = _load_json(schema_path)
    dataset = _dataset_by_id(dataset_path)

    cases: List[Dict[str, object]] = []
    axis_counts: Counter = Counter()
    plan_counts: Counter = Counter()
    opportunity_counts: Counter = Counter()
    latent_counts: Counter = Counter()
    family_counts: Counter = Counter()
    aggregation_counts: Counter = Counter()
    dimension_counts: Counter = Counter()

    for detail in results.get("details") or []:
        if not isinstance(detail, dict):
            continue
        qid = str(detail.get("id", "") or "")
        candidates = _rank_candidates(detail, score_key=score_key, sort_by_score=sort_by_score)
        if len(candidates) < 2:
            continue
        top = candidates[0]
        second = candidates[1]
        score1 = _score(top, score_key)
        score2 = _score(second, score_key)
        margin = score1 - score2
        if score1 < min_score or margin < min_margin or _label(top) == "correct":
            continue

        correct = _first_correct(candidates)
        correct_rank = None
        correct_candidate: Dict[str, object] = {}
        if correct:
            correct_idx, correct_candidate = correct
            correct_rank = correct_idx + 1

        gold = dataset.get(qid, {})
        question = _question(detail, gold)
        family = str(gold.get("topic") or gold.get("family") or detail.get("family") or "unknown")
        question_contract = extract_question_contract(question)
        top_contract = extract_query_contract(str(top.get("query", "") or ""))
        correct_contract = extract_query_contract(str(correct_candidate.get("query", "") or "")) if correct else None
        top_comparison = compare_contracts(question_contract, top_contract).to_dict()
        correct_comparison = (
            compare_contracts(question_contract, correct_contract).to_dict()
            if correct_contract is not None
            else {"score": 0.0, "matched": {}, "missing": {}, "conflicts": {}}
        )
        axis_issues = _axis_issue_summary(top_comparison, correct_comparison) if correct else ["no_correct_candidate"]
        if not axis_issues:
            axis_issues = ["no_contract_axis_difference"]

        top_semantic = _semantic_summary(question, top)
        correct_semantic = _semantic_summary(question, correct_candidate) if correct else {"judge_score": 0.0, "coverage_score": 0.0}
        top_plan = _plan_summary(str(top.get("query", "") or ""), schema)
        correct_plan = _plan_summary(str(correct_candidate.get("query", "") or ""), schema) if correct else {}
        plan_issues = (
            _plan_issue_summary(
                top_plan=top_plan,
                correct_plan=correct_plan,
                top_candidate=top,
                correct_candidate=correct_candidate,
                top_semantic=top_semantic,
                correct_semantic=correct_semantic,
            )
            if correct
            else ["no_correct_candidate"]
        )
        opportunities = (
            _semantic_category_opportunities(
                question_contract=question_contract,
                top_plan=top_plan,
                correct_plan=correct_plan,
                top_candidate=top,
                correct_candidate=correct_candidate,
            )
            if correct
            else ["no_correct_candidate"]
        )
        latent_issues = (
            _latent_issue_summary(
                axis_issues=axis_issues,
                top_plan=top_plan,
                correct_plan=correct_plan,
                top_candidate=top,
                correct_candidate=correct_candidate,
            )
            if correct
            else []
        )

        for issue in axis_issues:
            axis_counts[issue] += 1
        for issue in plan_issues:
            plan_counts[issue] += 1
        for issue in opportunities:
            opportunity_counts[issue] += 1
        for issue in latent_issues:
            latent_counts[issue] += 1
        family_counts[family] += 1
        contract = question_contract.to_dict()
        aggregation_counts[str(contract.get("aggregation") or "none")] += 1
        dimensions = list(contract.get("dimensions") or [])
        if not dimensions:
            dimension_counts["none"] += 1
        for dimension in dimensions:
            dimension_counts[str(dimension)] += 1

        cases.append(
            {
                "id": qid,
                "family": family,
                "question": question,
                "score1": score1,
                "score2": score2,
                "margin": margin,
                "correct_rank": correct_rank,
                "question_contract": contract,
                "axis_issues": axis_issues,
                "plan_issues": plan_issues,
                "semantic_opportunities": opportunities,
                "latent_issues": latent_issues,
                "top1": {
                    "source": top.get("source"),
                    "score": score1,
                    "semantic": top_semantic,
                    "contract": top_contract.to_dict(),
                    "contract_comparison": top_comparison,
                    "plan": top_plan,
                    "query": str(top.get("query", "") or ""),
                },
                "first_correct": {
                    "rank": correct_rank,
                    "source": correct_candidate.get("source") if correct else None,
                    "score": _score(correct_candidate, score_key) if correct else None,
                    "semantic": correct_semantic,
                    "contract": correct_contract.to_dict() if correct_contract is not None else {},
                    "contract_comparison": correct_comparison,
                    "plan": correct_plan,
                    "query": str(correct_candidate.get("query", "") or "") if correct else "",
                },
            }
        )

    cases.sort(key=lambda row: (float(row["score1"]), float(row["margin"])), reverse=True)
    return {
        "inputs": {
            "results": results_path,
            "dataset": dataset_path,
            "schema": schema_path,
            "score_key": score_key,
            "min_score": min_score,
            "min_margin": min_margin,
            "sort_by_score": sort_by_score,
        },
        "summary": {
            "high_confidence_wrong": len(cases),
            "axis_issue_counts": axis_counts.most_common(),
            "plan_issue_counts": plan_counts.most_common(),
            "semantic_opportunity_counts": opportunity_counts.most_common(),
            "latent_issue_counts": latent_counts.most_common(),
            "family_counts": family_counts.most_common(),
            "aggregation_counts": aggregation_counts.most_common(),
            "dimension_counts": dimension_counts.most_common(),
        },
        "cases": cases,
    }


def _fmt_list(values: List[str], limit: int = 10) -> str:
    if not values:
        return "-"
    if len(values) > limit:
        return ", ".join(values[:limit]) + f", ... (+{len(values) - limit})"
    return ", ".join(values)


def write_markdown(report: Dict[str, object], out_path: str) -> None:
    summary = dict(report.get("summary") or {})
    inputs = dict(report.get("inputs") or {})
    lines: List[str] = []
    lines.append("# High-Confidence Mistake Analysis")
    lines.append("")
    lines.append(f"- Results: `{inputs.get('results')}`")
    lines.append(f"- Dataset: `{inputs.get('dataset')}`")
    lines.append(f"- Score key: `{inputs.get('score_key')}`")
    lines.append(f"- Policy: score >= {float(inputs.get('min_score') or 0.0):.2f}, margin >= {float(inputs.get('min_margin') or 0.0):.2f}")
    lines.append(f"- Sort by score: `{inputs.get('sort_by_score')}`")
    lines.append(f"- High-confidence wrong cases: {summary.get('high_confidence_wrong', 0)}")
    lines.append("")
    for title, key in [
        ("Axis Issues", "axis_issue_counts"),
        ("Plan Issues", "plan_issue_counts"),
        ("Semantic Opportunities", "semantic_opportunity_counts"),
        ("Latent Issues", "latent_issue_counts"),
        ("Families", "family_counts"),
        ("Aggregations", "aggregation_counts"),
        ("Dimensions", "dimension_counts"),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Value | Count |")
        lines.append("|---|---:|")
        for value, count in summary.get(key) or []:
            lines.append(f"| `{value}` | {count} |")
        lines.append("")

    lines.append("## Cases")
    lines.append("")
    for case in report.get("cases") or []:
        top = dict(case.get("top1") or {})
        correct = dict(case.get("first_correct") or {})
        lines.append(f"### {case.get('id')} - {case.get('family')}")
        lines.append("")
        lines.append(f"Question: {case.get('question')}")
        lines.append("")
        lines.append(
            f"- score1={float(case.get('score1', 0.0)):.4f}, "
            f"score2={float(case.get('score2', 0.0)):.4f}, "
            f"margin={float(case.get('margin', 0.0)):.4f}, "
            f"correct_rank={case.get('correct_rank')}"
        )
        lines.append(f"- Contract: `{case.get('question_contract')}`")
        lines.append(f"- Axis issues: {_fmt_list(list(case.get('axis_issues') or []))}")
        lines.append(f"- Plan issues: {_fmt_list(list(case.get('plan_issues') or []))}")
        lines.append(f"- Semantic opportunities: {_fmt_list(list(case.get('semantic_opportunities') or []))}")
        lines.append(f"- Latent issues: {_fmt_list(list(case.get('latent_issues') or []))}")
        lines.append("")
        lines.append(
            f"Top wrong: source={top.get('source')}, "
            f"contract={dict(top.get('contract_comparison') or {}).get('score')}, "
            f"semantic={dict(top.get('semantic') or {}).get('judge_score')}"
        )
        lines.append("")
        lines.append(f"`{_one_line_query(str(top.get('query') or ''))}`")
        lines.append("")
        lines.append(
            f"First correct: rank={correct.get('rank')}, source={correct.get('source')}, "
            f"score={correct.get('score')}, "
            f"contract={dict(correct.get('contract_comparison') or {}).get('score')}, "
            f"semantic={dict(correct.get('semantic') or {}).get('judge_score')}"
        )
        lines.append("")
        lines.append(f"`{_one_line_query(str(correct.get('query') or ''))}`")
        lines.append("")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze high-confidence wrong KGQA selections.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--score-key", default="ml_score", choices=["ml_score", "selection_score"])
    parser.add_argument("--min-score", type=float, default=0.90)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--sort-by-score", action="store_true")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(
        results_path=args.results,
        dataset_path=args.dataset,
        schema_path=args.schema,
        score_key=args.score_key,
        min_score=args.min_score,
        min_margin=args.min_margin,
        sort_by_score=args.sort_by_score,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_markdown(report, args.out_md)

    summary = report["summary"]
    print("===== HIGH-CONFIDENCE MISTAKE ANALYSIS =====")
    print(f"Results: {args.results}")
    print(f"High-confidence wrong: {summary['high_confidence_wrong']}")
    print("Top axis issues:")
    for value, count in summary["axis_issue_counts"][:12]:
        print(f"  {value}: {count}")
    print("Top plan issues:")
    for value, count in summary["plan_issue_counts"][:12]:
        print(f"  {value}: {count}")
    print("Top semantic opportunities:")
    for value, count in summary["semantic_opportunity_counts"][:12]:
        print(f"  {value}: {count}")
    print("Top families:")
    for value, count in summary["family_counts"][:12]:
        print(f"  {value}: {count}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
