"""Compare audited deterministic answers with the current direct resolver.

This script does not re-run the LLM and does not use answer labels to make
routing decisions. It replays the current capability registry over an existing
audited system CSV to estimate whether stricter direct-routing guards make the
deterministic route more conservative.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from kg.capabilities import DEFAULT_REGISTRY


def _label(row: Dict[str, str]) -> str:
    return (row.get("correctness") or row.get("manual_label") or row.get("label") or "").strip().lower()


def analyze(audit_csv: str) -> Dict[str, object]:
    with open(audit_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    direct_rows = [row for row in rows if (row.get("selected_source") or "").strip() == "capability_inventory"]
    counters: Counter[str] = Counter()
    by_failure_type: Counter[str] = Counter()
    abstained_examples: List[Dict[str, str]] = []
    still_direct_wrong_examples: List[Dict[str, str]] = []

    for row in direct_rows:
        label = _label(row)
        report = DEFAULT_REGISTRY.resolve(row.get("question") or "")
        current_query = DEFAULT_REGISTRY.direct_query_for(report)
        current_direct = bool(current_query)
        counters[f"{label or 'unlabeled'}__current_direct_{str(current_direct).lower()}"] += 1
        if label == "incorrect" and not current_direct:
            by_failure_type[row.get("failure_type") or "unknown"] += 1
            if len(abstained_examples) < 20:
                abstained_examples.append(
                    {
                        "request_id": row.get("request_id", ""),
                        "question": row.get("question", ""),
                        "failure_type": row.get("failure_type", ""),
                        "human_sparql_difficulty": row.get("human_sparql_difficulty", ""),
                    }
                )
        elif label == "incorrect" and current_direct and len(still_direct_wrong_examples) < 20:
            still_direct_wrong_examples.append(
                {
                    "request_id": row.get("request_id", ""),
                    "question": row.get("question", ""),
                    "failure_type": row.get("failure_type", ""),
                    "human_sparql_difficulty": row.get("human_sparql_difficulty", ""),
                }
            )

    old_correct = counters["correct__current_direct_true"] + counters["correct__current_direct_false"]
    old_incorrect = counters["incorrect__current_direct_true"] + counters["incorrect__current_direct_false"]
    current_direct_correct = counters["correct__current_direct_true"]
    current_direct_incorrect = counters["incorrect__current_direct_true"]
    current_direct_total = current_direct_correct + current_direct_incorrect
    current_precision = current_direct_correct / current_direct_total if current_direct_total else 0.0

    return {
        "audit_csv": audit_csv,
        "previous_capability_inventory_rows": len(direct_rows),
        "previous_capability_inventory_correct": old_correct,
        "previous_capability_inventory_incorrect": old_incorrect,
        "previous_capability_inventory_accuracy": old_correct / max(1, old_correct + old_incorrect),
        "current_direct_kept": current_direct_total,
        "current_direct_correct_from_previous_audit": current_direct_correct,
        "current_direct_incorrect_from_previous_audit": current_direct_incorrect,
        "current_direct_precision_from_previous_audit": current_precision,
        "old_incorrect_now_abstained": counters["incorrect__current_direct_false"],
        "old_correct_now_abstained": counters["correct__current_direct_false"],
        "abstained_incorrect_by_failure_type": dict(by_failure_type.most_common()),
        "abstained_incorrect_examples": abstained_examples,
        "still_direct_wrong_examples": still_direct_wrong_examples,
        "note": (
            "This is a resolver replay over a previous audit. Rows that remain direct may still "
            "be fixed by changed query shape, so this is a conservative lower-bound estimate."
        ),
    }


def _write_md(report: Dict[str, object], out_md: str) -> None:
    lines = [
        "# Deterministic Strictness Replay",
        "",
        f"Audit CSV: `{report['audit_csv']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Previous capability-inventory rows | {report['previous_capability_inventory_rows']} |",
        f"| Previous capability-inventory accuracy | {100 * float(report['previous_capability_inventory_accuracy']):.1f}% |",
        f"| Current direct rows kept from previous audit | {report['current_direct_kept']} |",
        f"| Current direct precision from previous audit | {100 * float(report['current_direct_precision_from_previous_audit']):.1f}% |",
        f"| Old incorrect rows now abstained | {report['old_incorrect_now_abstained']} |",
        f"| Old correct rows now abstained | {report['old_correct_now_abstained']} |",
        "",
        "## Abstained Incorrect Rows By Failure Type",
        "",
        "| Failure type | Rows |",
        "|---|---:|",
    ]
    for failure_type, count in dict(report["abstained_incorrect_by_failure_type"]).items():
        lines.append(f"| {failure_type} | {count} |")
    lines.extend(["", "## Note", "", str(report["note"]), ""])
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay current strict direct routing over an audited CSV.")
    parser.add_argument("--audit-csv", default="results/kgqa_system_accuracy_audit_1000_after_direct_labeled.csv")
    parser.add_argument("--out-json", default="results/deterministic_strictness_replay.json")
    parser.add_argument("--out-md", default="results/deterministic_strictness_replay.md")
    args = parser.parse_args()

    report = analyze(args.audit_csv)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(report, args.out_md)
    print("===== DETERMINISTIC STRICTNESS REPLAY =====")
    print(f"Previous direct rows: {report['previous_capability_inventory_rows']}")
    print(
        "Current precision from previous audit: "
        f"{report['current_direct_correct_from_previous_audit']}/"
        f"{report['current_direct_kept']} "
        f"({float(report['current_direct_precision_from_previous_audit']):.3f})"
    )
    print(f"Old incorrect now abstained: {report['old_incorrect_now_abstained']}")
    print(f"Old correct now abstained: {report['old_correct_now_abstained']}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
