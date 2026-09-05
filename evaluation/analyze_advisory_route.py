#!/usr/bin/env python3
"""Audit deterministic advisory-route behavior from a full-system audit CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


PSEUDOCODE = """\
if request_type == "advisory":
    plan = resolve_advisory_plan(question)
    if plan is None:
        route_to_llm_or_clarification()
    rows = execute_checked_sparql(plan.query)
    if rows are empty:
        return controlled_no_answer()
    ranked_rows = sort_rows_by_plan_metric(rows, plan.value_key)
    return conservative_advice(
        leading_group=ranked_rows[0][plan.group_key],
        evidence_rows=ranked_rows[:3],
        caveat="data-grounded signal, not an autonomous business decision",
    )
"""


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _is_advisory(row: Dict[str, str]) -> bool:
    return (
        (row.get("expected_route") or "").strip().lower() == "advisory"
        or (row.get("selected_source") or "").strip().lower() == "advisory"
        or "advisory" in (row.get("execution_selection_reason") or "").lower()
    )


def analyze(audit_csv: str, max_examples: int) -> Dict[str, object]:
    rows = _load_rows(audit_csv)
    advisory = [r for r in rows if _is_advisory(r)]
    correct = [r for r in advisory if (r.get("correctness") or "").strip().lower() == "correct"]
    by_topic = Counter((r.get("topic") or "unknown").strip() or "unknown" for r in advisory)
    by_source = Counter((r.get("selected_source") or "unknown").strip() or "unknown" for r in advisory)
    llm_calls = sum(int(float(r.get("estimated_llm_calls") or 0)) for r in advisory)
    return {
        "audit_csv": audit_csv,
        "summary": {
            "advisory_rows": len(advisory),
            "correct": len(correct),
            "accuracy": len(correct) / len(advisory) if advisory else 0.0,
            "estimated_llm_calls": llm_calls,
        },
        "by_topic": by_topic.most_common(),
        "by_selected_source": by_source.most_common(),
        "examples": advisory[:max_examples],
        "deterministic_boundary_pseudocode": PSEUDOCODE,
    }


def _write_md(report: Dict[str, object], out_md: str) -> None:
    s = report["summary"]
    lines = [
        "# Advisory Route Audit",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Advisory rows | {s['advisory_rows']} |",
        f"| Correct | {s['correct']} |",
        f"| Accuracy | {100 * s['accuracy']:.1f}% |",
        f"| Estimated LLM calls | {s['estimated_llm_calls']} |",
        "",
        "## Deterministic Boundary",
        "",
        "```python",
        report["deterministic_boundary_pseudocode"].rstrip(),
        "```",
        "",
        "## Topics",
        "",
    ]
    for topic, count in report["by_topic"]:
        lines.append(f"- {topic}: {count}")
    lines.extend(["", "## Examples", ""])
    for row in report["examples"]:
        lines.append(f"- `{row.get('request_id')}` {row.get('question')} -> {row.get('correctness')}")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize deterministic advisory-route accuracy and boundary.")
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(args.audit_csv, args.max_examples)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(report, args.out_md)
    print("===== ADVISORY ROUTE AUDIT =====")
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
