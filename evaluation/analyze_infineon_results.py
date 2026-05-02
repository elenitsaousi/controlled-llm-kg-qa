#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.feature_extraction import extract_query_plan
from validation.semantic import semantic_coverage_report


def _load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _candidate_label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "")).strip().lower()


def _first_correct_rank(candidates: List[Dict[str, object]]) -> Optional[int]:
    for idx, candidate in enumerate(candidates):
        if _candidate_label(candidate) == "correct":
            return idx
    return None


def _failure_category(detail: Dict[str, object]) -> str:
    top1_correct = bool(detail.get("top1_correct"))
    any_correct = bool(detail.get("any_correct"))
    candidates = list(detail.get("candidates") or [])
    if top1_correct:
        return "correct"
    if not candidates:
        return "generation_failure_no_candidates"
    if not any_correct:
        return "generation_failure_no_correct_candidate"

    first_label = _candidate_label(candidates[0])
    if first_label in {"invalid", "timeout", "error"}:
        return "selection_failure_invalid_top1"
    return "ranking_failure_correct_candidate_not_top1"


def _plan_labels(query: str, schema: Dict[str, object]) -> List[str]:
    if not query:
        return []
    try:
        plan = extract_query_plan(query, schema)
    except Exception:
        return []
    return sorted(str(label) for label in plan.get("labels", []) if str(label).strip())


def _selected_query(detail: Dict[str, object]) -> str:
    candidates = list(detail.get("candidates") or [])
    if not candidates:
        return ""
    return str(candidates[0].get("query", "") or "")


def _gold_for_detail(
    detail: Dict[str, object],
    dataset_by_id: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    row_id = str(detail.get("id", ""))
    return dataset_by_id.get(row_id, {})


def analyze_results(
    *,
    results_path: str,
    dataset_path: str,
    schema_path: str,
) -> Dict[str, object]:
    results = _load_json(results_path)
    dataset = _load_json(dataset_path)
    schema = _load_json(schema_path)
    if not isinstance(results, dict):
        raise ValueError("Results JSON must be an object.")
    if not isinstance(dataset, list):
        raise ValueError("Dataset JSON must be a list.")

    details = list(results.get("details") or [])
    dataset_by_id = {str(row.get("id", "")): row for row in dataset if isinstance(row, dict)}

    cases: List[Dict[str, object]] = []
    family_counts: Dict[str, Counter] = defaultdict(Counter)
    category_counts: Counter = Counter()
    correct_rank_counts: Counter = Counter()
    missing_concept_counts: Counter = Counter()

    for detail in details:
        if not isinstance(detail, dict):
            continue
        gold = _gold_for_detail(detail, dataset_by_id)
        candidates = list(detail.get("candidates") or [])
        selected = _selected_query(detail)
        gold_query = str(gold.get("query", "") or "")
        question = str(detail.get("question") or gold.get("question") or "")
        family = str(gold.get("topic") or gold.get("family_id") or "unknown")
        failure_category = _failure_category(detail)
        first_correct_rank = _first_correct_rank(candidates)
        if first_correct_rank is None:
            correct_rank_text = "missing"
        else:
            correct_rank_text = str(first_correct_rank)

        selected_semantic = semantic_coverage_report(question, selected) if selected else {
            "required": [],
            "covered": [],
            "missing": [],
            "coverage_score": 0.0,
        }
        gold_semantic = semantic_coverage_report(question, gold_query) if gold_query else {
            "required": [],
            "covered": [],
            "missing": [],
            "coverage_score": 0.0,
        }
        for concept in selected_semantic.get("missing", []):
            missing_concept_counts[str(concept)] += 1

        candidate_label_counts = Counter(_candidate_label(c) or "unknown" for c in candidates)
        category_counts[failure_category] += 1
        correct_rank_counts[correct_rank_text] += 1
        family_counts[family]["total"] += 1
        family_counts[family]["top1_correct"] += int(bool(detail.get("top1_correct")))
        family_counts[family]["any_correct"] += int(bool(detail.get("any_correct")))
        family_counts[family][failure_category] += 1

        cases.append(
            {
                "id": detail.get("id"),
                "family": family,
                "ambiguity_label": detail.get("ambiguity_label") or gold.get("ambiguity_label"),
                "question": question,
                "top1_correct": bool(detail.get("top1_correct")),
                "any_correct": bool(detail.get("any_correct")),
                "failure_category": failure_category,
                "first_correct_candidate_rank": first_correct_rank,
                "candidate_count": len(candidates),
                "candidate_label_counts": dict(candidate_label_counts),
                "selected_semantic_coverage": selected_semantic,
                "gold_semantic_coverage": gold_semantic,
                "gold_query_plan_labels": _plan_labels(gold_query, schema),
                "selected_query_plan_labels": _plan_labels(selected, schema),
                "selected_query": selected,
                "gold_query": gold_query,
            }
        )

    total = len(cases)
    top1_correct = sum(1 for case in cases if case["top1_correct"])
    any_correct = sum(1 for case in cases if case["any_correct"])
    generation_failures = sum(
        1 for case in cases if str(case["failure_category"]).startswith("generation_failure")
    )
    ranking_failures = sum(
        1 for case in cases if case["failure_category"] == "ranking_failure_correct_candidate_not_top1"
    )

    families = []
    for family, counts in sorted(family_counts.items()):
        fam_total = int(counts["total"])
        families.append(
            {
                "family": family,
                "total": fam_total,
                "top1_correct": int(counts["top1_correct"]),
                "any_correct": int(counts["any_correct"]),
                "top1_correct_rate": _safe_rate(int(counts["top1_correct"]), fam_total),
                "any_correct_rate": _safe_rate(int(counts["any_correct"]), fam_total),
                "failure_counts": {
                    key: int(value)
                    for key, value in sorted(counts.items())
                    if key not in {"total", "top1_correct", "any_correct"}
                },
            }
        )

    return {
        "inputs": {
            "results_path": results_path,
            "dataset_path": dataset_path,
            "schema_path": schema_path,
        },
        "summary": {
            "total": total,
            "top1_correct": top1_correct,
            "top1_correct_rate": _safe_rate(top1_correct, total),
            "any_correct": any_correct,
            "any_correct_rate": _safe_rate(any_correct, total),
            "ranking_failures_with_correct_candidate": ranking_failures,
            "generation_failures_without_correct_candidate": generation_failures,
            "failure_category_counts": dict(category_counts),
            "first_correct_candidate_rank_counts": dict(correct_rank_counts),
            "selected_missing_concept_counts": dict(missing_concept_counts),
        },
        "families": sorted(
            families,
            key=lambda row: (row["top1_correct_rate"], row["any_correct_rate"], row["family"]),
        ),
        "cases": cases,
    }


def _format_pct(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(report: Dict[str, object]) -> str:
    summary = dict(report.get("summary") or {})
    lines: List[str] = []
    lines.append("# Infineon KGQA Error Analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total questions: {summary.get('total', 0)}")
    lines.append(
        f"- Top1 correct: {summary.get('top1_correct', 0)} "
        f"({_format_pct(float(summary.get('top1_correct_rate', 0.0)))})"
    )
    lines.append(
        f"- Any correct candidate: {summary.get('any_correct', 0)} "
        f"({_format_pct(float(summary.get('any_correct_rate', 0.0)))})"
    )
    lines.append(
        f"- Ranking failures with a correct candidate present: "
        f"{summary.get('ranking_failures_with_correct_candidate', 0)}"
    )
    lines.append(
        f"- Generation failures without a correct candidate: "
        f"{summary.get('generation_failures_without_correct_candidate', 0)}"
    )
    lines.append("")

    lines.append("## Failure Categories")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for key, value in sorted((summary.get("failure_category_counts") or {}).items()):
        lines.append(f"| `{key}` | {value} |")
    lines.append("")

    lines.append("## Family Performance")
    lines.append("")
    lines.append("| Family | Total | Top1 | Any | Main failures |")
    lines.append("|---|---:|---:|---:|---|")
    for fam in report.get("families") or []:
        failures = ", ".join(
            f"{key}={value}" for key, value in (fam.get("failure_counts") or {}).items()
        )
        lines.append(
            f"| `{fam.get('family')}` | {fam.get('total')} | "
            f"{_format_pct(float(fam.get('top1_correct_rate', 0.0)))} | "
            f"{_format_pct(float(fam.get('any_correct_rate', 0.0)))} | "
            f"{failures or '-'} |"
        )
    lines.append("")

    lines.append("## Ranking Failures")
    lines.append("")
    lines.append("| ID | Family | Correct rank | Heuristic missing concepts | Question |")
    lines.append("|---|---|---:|---|---|")
    for case in report.get("cases") or []:
        if case.get("failure_category") != "ranking_failure_correct_candidate_not_top1":
            continue
        missing = ", ".join(case.get("selected_semantic_coverage", {}).get("missing", []))
        rank = case.get("first_correct_candidate_rank")
        lines.append(
            f"| `{case.get('id')}` | `{case.get('family')}` | {rank} | "
            f"{missing or '-'} | {case.get('question')} |"
        )
    lines.append("")

    lines.append("## Generation Failures")
    lines.append("")
    lines.append("| ID | Family | Heuristic missing concepts | Question |")
    lines.append("|---|---|---|---|")
    for case in report.get("cases") or []:
        if not str(case.get("failure_category", "")).startswith("generation_failure"):
            continue
        missing = ", ".join(case.get("selected_semantic_coverage", {}).get("missing", []))
        lines.append(
            f"| `{case.get('id')}` | `{case.get('family')}` | "
            f"{missing or '-'} | {case.get('question')} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Infineon held-out KGQA results without rerunning LLM generation."
    )
    parser.add_argument("--results", default="results/infineon_test_final_results.json")
    parser.add_argument("--dataset", default="data/infineon/infineon_test_final.json")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out-json", default="results/infineon_test_final_error_analysis.json")
    parser.add_argument("--out-md", default="results/infineon_test_final_error_analysis.md")
    args = parser.parse_args()

    report = analyze_results(
        results_path=args.results,
        dataset_path=args.dataset,
        schema_path=args.schema,
    )
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    summary = report["summary"]
    print("===== INFINEON ERROR ANALYSIS =====")
    print(f"Results: {args.results}")
    print(f"Dataset: {args.dataset}")
    print(f"Total: {summary['total']}")
    print(
        f"Top1 correct: {summary['top1_correct']} "
        f"({summary['top1_correct_rate']:.3f})"
    )
    print(
        f"Any correct: {summary['any_correct']} "
        f"({summary['any_correct_rate']:.3f})"
    )
    print(
        "Ranking failures with correct candidate: "
        f"{summary['ranking_failures_with_correct_candidate']}"
    )
    print(
        "Generation failures without correct candidate: "
        f"{summary['generation_failures_without_correct_candidate']}"
    )
    print(f"JSON report: {args.out_json}")
    print(f"Markdown report: {args.out_md}")


if __name__ == "__main__":
    main()
