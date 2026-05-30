#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.feature_extraction import extract_query_plan
from validation.semantic import semantic_coverage_report


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "")).strip().lower()


def _is_generation_failure(detail: Dict[str, object]) -> bool:
    candidates = list(detail.get("candidates") or [])
    if not candidates:
        return True
    return not any(_label(c) == "correct" for c in candidates)


def _plan_labels(query: str, schema: Dict[str, object]) -> List[str]:
    if not query:
        return []
    try:
        return sorted(str(x) for x in extract_query_plan(query, schema).get("labels", []))
    except Exception:
        return []


def _dataset_by_id(path: str) -> Dict[str, Dict[str, object]]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        return {}
    return {str(row.get("id", "")): row for row in rows if isinstance(row, dict)}


def _suggest_pattern(question: str, gold_labels: List[str], selected_labels: List[str]) -> List[str]:
    q = question.lower()
    suggestions = []
    missing_labels = set(gold_labels) - set(selected_labels)
    if any("DemandForRegion" in lab for lab in missing_labels):
        suggestions.append("regional_demand_template_or_prompt_gap")
    if any("FutureDemandAnalysis" in lab for lab in missing_labels):
        suggestions.append("future_demand_template_or_prompt_gap")
    if any("VehicleSalesObservation" in lab for lab in missing_labels):
        suggestions.append("vehicle_sales_template_or_prompt_gap")
    if any("AutonomousDrivingDevelopment" in lab for lab in missing_labels):
        suggestions.append("autonomous_driving_template_or_prompt_gap")
    if any("OrderCancellation" in lab for lab in missing_labels):
        suggestions.append("order_cancellation_template_or_prompt_gap")
    if any("InventoryDevelopment" in lab for lab in missing_labels):
        suggestions.append("inventory_template_or_prompt_gap")
    if "level 5" in q or "fully autonomous" in q or "full autonomy" in q:
        suggestions.append("needs_level_5_autonomy_filter")
    if "time period" in q and "forecast" in q:
        suggestions.append("needs_forecast_time_period_aggregation")
    if "survey-origin" in q or "survey origin" in q or "survey group" in q:
        suggestions.append("needs_survey_origin_grouping")
    return sorted(set(suggestions)) or ["uncategorized_generation_gap"]


def analyze_generation_failures(
    results_path: str,
    dataset_path: str,
    schema_path: str,
) -> Dict[str, object]:
    results = _load_json(results_path)
    schema = _load_json(schema_path)
    dataset = _dataset_by_id(dataset_path)
    details = list(results.get("details") or [])

    cases = []
    by_topic: Dict[str, Counter] = defaultdict(Counter)
    missing_label_counts = Counter()
    suggestion_counts = Counter()

    for detail in details:
        if not isinstance(detail, dict) or not _is_generation_failure(detail):
            continue
        qid = str(detail.get("id", ""))
        gold = dataset.get(qid, {})
        question = str(detail.get("question") or gold.get("question") or "")
        topic = str(gold.get("topic") or gold.get("family_id") or detail.get("family") or "unknown")
        candidates = list(detail.get("candidates") or [])
        selected_query = str(candidates[0].get("query", "") if candidates else "")
        gold_query = str(gold.get("query", "") or "")
        gold_labels = _plan_labels(gold_query, schema)
        selected_labels = _plan_labels(selected_query, schema)
        missing_labels = sorted(set(gold_labels) - set(selected_labels))
        for label in missing_labels:
            missing_label_counts[label] += 1
        suggestions = _suggest_pattern(question, gold_labels, selected_labels)
        for suggestion in suggestions:
            suggestion_counts[suggestion] += 1

        by_topic[topic]["total"] += 1
        by_topic[topic]["no_candidates"] += int(not candidates)
        by_topic[topic]["with_candidates_no_correct"] += int(bool(candidates))

        cases.append(
            {
                "id": qid,
                "topic": topic,
                "question": question,
                "candidate_count": len(candidates),
                "candidate_label_counts": dict(Counter(_label(c) or "unknown" for c in candidates)),
                "gold_required_concepts": semantic_coverage_report(question, gold_query).get("required", []),
                "gold_query_plan_labels": gold_labels,
                "selected_query_plan_labels": selected_labels,
                "missing_query_plan_labels": missing_labels,
                "suggested_pattern_fixes": suggestions,
                "gold_query": gold_query,
                "selected_query": selected_query,
            }
        )

    topics = {
        topic: {
            "total": int(counts["total"]),
            "no_candidates": int(counts["no_candidates"]),
            "with_candidates_no_correct": int(counts["with_candidates_no_correct"]),
        }
        for topic, counts in sorted(by_topic.items(), key=lambda kv: (-kv[1]["total"], kv[0]))
    }

    return {
        "inputs": {
            "results": results_path,
            "dataset": dataset_path,
            "schema": schema_path,
        },
        "summary": {
            "generation_failure_count": len(cases),
            "topic_count": len(topics),
            "top_missing_query_plan_labels": missing_label_counts.most_common(30),
            "suggested_pattern_fix_counts": suggestion_counts.most_common(),
        },
        "topics": topics,
        "cases": cases,
    }


def render_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Generation Failure Analysis",
        "",
        "This report groups cases where no correct candidate was generated. It is intended for pattern-level fixes, not hardcoded ID-level fixes.",
        "",
        "## Summary",
        "",
        f"- Generation failures: {report['summary']['generation_failure_count']}",
        f"- Topics affected: {report['summary']['topic_count']}",
        "",
        "## Suggested Pattern Fixes",
        "",
        "| Pattern | Count |",
        "|---|---:|",
    ]
    for pattern, count in report["summary"]["suggested_pattern_fix_counts"]:
        lines.append(f"| `{pattern}` | {count} |")

    lines.extend(["", "## Topics", "", "| Topic | Failures | No candidates | With candidates but no correct |", "|---|---:|---:|---:|"])
    for topic, row in report["topics"].items():
        lines.append(
            f"| `{topic}` | {row['total']} | {row['no_candidates']} | {row['with_candidates_no_correct']} |"
        )

    lines.extend(["", "## Cases", "", "| ID | Topic | Candidate Count | Suggested Fixes | Question |", "|---|---|---:|---|---|"])
    for case in report["cases"]:
        fixes = ", ".join(f"`{x}`" for x in case["suggested_pattern_fixes"])
        question = str(case["question"]).replace("|", "\\|")
        lines.append(
            f"| `{case['id']}` | `{case['topic']}` | {case['candidate_count']} | {fixes} | {question} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generation failures without adding ID-specific fixes.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze_generation_failures(args.results, args.dataset, args.schema)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")

    print("===== GENERATION FAILURE ANALYSIS =====")
    print(f"Results: {args.results}")
    print(f"Generation failures: {report['summary']['generation_failure_count']}")
    print(f"Topics affected: {report['summary']['topic_count']}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
