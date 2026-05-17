#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _contains_any(text: str, terms: List[str]) -> bool:
    low = text.lower()
    return any(term in low for term in terms)


TIME_TERMS = ["time", "year", "month", "quarter"]
RANKING_TERMS = [
    "highest",
    "largest",
    "top",
    "maximum",
    "greatest",
    "peak",
    "most",
    "leads",
    "strongest",
    "most common",
    "most frequently",
]
YEAR_TERMS = ["year", "annual"]


def row_warnings(row: Dict[str, object]) -> List[str]:
    source = str(row.get("source_question") or "")
    question = str(row.get("question") or "")
    warnings = []
    shape = str(row.get("answer_shape") or "")
    if shape == "ranking_top" and not _contains_any(question, RANKING_TERMS):
        warnings.append("missing_ranking_language")
    if shape == "count" and not _contains_any(question, ["how many", "number of", "count"]):
        warnings.append("missing_count_language")
    if shape == "average" and not _contains_any(question, ["average", "mean"]):
        warnings.append("missing_average_language")
    if shape == "sum" and not _contains_any(question, ["total", "sum", "overall", "how much"]):
        warnings.append("missing_sum_language")
    if source:
        if _contains_any(source, ["percentage", "percent"]) and not _contains_any(question, ["percentage", "percent"]):
            warnings.append("lost_percentage_measure")
        if _contains_any(source, ["percentage change", "percent change"]) and not _contains_any(
            question,
            ["percentage change", "percent change"],
        ):
            warnings.append("lost_percentage_change_measure")
        if _contains_any(source, ["participant"]) and not _contains_any(question, ["participant"]):
            warnings.append("lost_participant_measure")
        if _contains_any(question, ["over time"]) and not _contains_any(source, TIME_TERMS):
            warnings.append("added_time_dimension")
        for term in TIME_TERMS[1:]:
            aliases = YEAR_TERMS if term == "year" else [term]
            if _contains_any(question, aliases) and not _contains_any(source, aliases):
                warnings.append(f"added_{term}_dimension")
        for term in TIME_TERMS[1:]:
            aliases = YEAR_TERMS if term == "year" else [term]
            if _contains_any(source, aliases) and not _contains_any(question, aliases):
                warnings.append(f"lost_{term}_dimension")
        if _contains_any(source, ["trend"]) and not _contains_any(question, ["trend"]):
            warnings.append("lost_trend_dimension")
    return warnings


def audit_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    cases = []
    for row in rows:
        source = str(row.get("source_question") or "")
        question = str(row.get("question") or "")
        shape = str(row.get("answer_shape") or "")
        warnings = row_warnings(row)
        if warnings:
            cases.append(
                {
                    "id": row.get("id"),
                    "family": row.get("family"),
                    "answer_shape": shape,
                    "source_question": source,
                    "question": question,
                    "warnings": warnings,
                }
            )
    return {
        "summary": {
            "total": len(rows),
            "flagged": len(cases),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated benchmark wording for likely intent drift.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    report = audit_rows(rows)
    print("===== GENERATED BENCHMARK WORDING AUDIT =====")
    print(f"Total: {report['summary']['total']}")
    print(f"Flagged: {report['summary']['flagged']}")
    for case in report["cases"]:
        print(f"  {case['id']}: {', '.join(case['warnings'])}")
        print(f"    source: {case['source_question']}")
        print(f"    draft:  {case['question']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
