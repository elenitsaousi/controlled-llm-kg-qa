#!/usr/bin/env python3
"""Estimate manual SPARQL effort from audited difficulty labels.

The estimate is intentionally transparent and conservative. If a
`human_difficulty` column is available, it is used directly. Otherwise the
script looks for difficulty labels in `notes`; rows without a label are grouped
as `unlabeled`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _difficulty(row: Dict[str, str]) -> str:
    direct = (row.get("human_difficulty") or row.get("difficulty") or "").strip().lower()
    if direct in {"easy", "medium", "hard"}:
        return direct
    notes = (row.get("notes") or "").lower()
    for label in ("easy", "medium", "hard"):
        if re.search(rf"\b{label}\b", notes):
            return label
    return "unlabeled"


def _minutes(label: str, args: argparse.Namespace) -> float:
    return {
        "easy": args.easy_minutes,
        "medium": args.medium_minutes,
        "hard": args.hard_minutes,
        "unlabeled": args.unlabeled_minutes,
    }.get(label, args.unlabeled_minutes)


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    manual_counts = _parse_counts(args.difficulty_counts)
    if manual_counts:
        rows = []
        counts = Counter(manual_counts)
    else:
        rows = _load_rows(args.audit_csv)
        if args.only_incorrect:
            rows = [r for r in rows if (r.get("correctness") or "").strip().lower() == "incorrect"]
        if args.only_llm:
            rows = [r for r in rows if (r.get("system_mode") or "").strip().lower() == "llm_ranking" or int(float(r.get("estimated_llm_calls") or 0)) > 0]
        counts = Counter(_difficulty(row) for row in rows)
    tiers = []
    total_minutes = 0.0
    for label in ["easy", "medium", "hard", "unlabeled"]:
        count = counts[label]
        minutes = _minutes(label, args)
        subtotal = count * minutes
        total_minutes += subtotal
        tiers.append({"difficulty": label, "questions": count, "minutes_per_query": minutes, "total_minutes": subtotal})

    return {
        "audit_csv": args.audit_csv,
        "filters": {"only_incorrect": args.only_incorrect, "only_llm": args.only_llm},
        "manual_difficulty_counts": dict(manual_counts),
        "assumptions": {
            "easy_minutes": args.easy_minutes,
            "medium_minutes": args.medium_minutes,
            "hard_minutes": args.hard_minutes,
            "unlabeled_minutes": args.unlabeled_minutes,
        },
        "summary": {
            "questions": sum(counts.values()) if manual_counts else len(rows),
            "estimated_manual_minutes": total_minutes,
            "estimated_manual_hours": total_minutes / 60.0,
        },
        "tiers": tiers,
    }


def _parse_counts(items: List[str] | None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--difficulty-counts must use LABEL=COUNT, e.g. easy=53")
        label, value = item.split("=", 1)
        label = label.strip().lower()
        if label not in {"easy", "medium", "hard", "unlabeled"}:
            raise ValueError(f"unsupported difficulty label: {label}")
        counts[label] = int(value)
    return counts


def _write_md(report: Dict[str, object], out_md: str) -> None:
    s = report["summary"]
    lines = [
        "# Manual SPARQL Effort Estimate",
        "",
        "| Difficulty | Questions | Minutes / query | Total minutes |",
        "|---|---:|---:|---:|",
    ]
    for row in report["tiers"]:
        lines.append(
            f"| {row['difficulty']} | {row['questions']} | {row['minutes_per_query']:.1f} | {row['total_minutes']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Estimated manual effort: **{s['estimated_manual_minutes']:.1f} minutes** "
            f"(**{s['estimated_manual_hours']:.1f} hours**).",
            "",
            "This is an explicit assumption-based estimate, not a user study.",
        ]
    )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate manual SPARQL effort from audited difficulty labels.")
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--only-incorrect", action="store_true")
    parser.add_argument("--only-llm", action="store_true")
    parser.add_argument("--easy-minutes", type=float, default=5.0)
    parser.add_argument("--medium-minutes", type=float, default=15.0)
    parser.add_argument("--hard-minutes", type=float, default=30.0)
    parser.add_argument("--unlabeled-minutes", type=float, default=10.0)
    parser.add_argument(
        "--difficulty-counts",
        nargs="*",
        help="Optional audited counts, e.g. easy=53 medium=37 hard=23. When provided, CSV difficulty labels are not required.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(args)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(report, args.out_md)
    print("===== MANUAL SPARQL EFFORT ESTIMATE =====")
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
