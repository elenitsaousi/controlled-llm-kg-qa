#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _normalized_question(question: str) -> str:
    return " ".join(str(question or "").strip().lower().split())


def find_duplicates(rows: List[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_normalized_question(str(row.get("question") or ""))].append(row)
    cases = []
    for normalized, matches in grouped.items():
        if len(matches) < 2:
            continue
        cases.append(
            {
                "normalized_question": normalized,
                "count": len(matches),
                "ids": [row.get("id") for row in matches],
                "template_ids": [row.get("template_id") for row in matches],
                "question": matches[0].get("question"),
            }
        )
    cases.sort(key=lambda case: (-int(case["count"]), str(case["normalized_question"])))
    return {
        "summary": {
            "total": len(rows),
            "unique_questions": len(grouped),
            "duplicate_groups": len(cases),
            "duplicate_rows": sum(int(case["count"]) - 1 for case in cases),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report duplicate natural-language questions in a generated benchmark.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    report = find_duplicates(rows)
    print("===== GENERATED BENCHMARK DUPLICATES =====")
    print(f"Total: {report['summary']['total']}")
    print(f"Unique questions: {report['summary']['unique_questions']}")
    print(f"Duplicate groups: {report['summary']['duplicate_groups']}")
    print(f"Duplicate rows beyond first occurrence: {report['summary']['duplicate_rows']}")
    for case in report["cases"]:
        print(f"  {', '.join(str(item) for item in case['ids'])}")
        print(f"    question: {case['question']}")
        print(f"    templates: {', '.join(str(item) for item in case['template_ids'])}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
