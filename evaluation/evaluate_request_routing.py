#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kg.schema import load_schema
from pipeline.request_routing import route_request


def evaluate(dataset_path: str, schema_path: str) -> Dict[str, object]:
    rows = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    schema = load_schema(schema_path)
    cases: List[Dict[str, object]] = []
    counts = Counter()
    for row in rows:
        actual = route_request(str(row["question"]), schema=schema)
        expected_route = str(row["expected_route"])
        actual_route = str(actual.get("route"))
        correct = actual_route == expected_route
        counts["total"] += 1
        counts["correct"] += int(correct)
        cases.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "expected_route": expected_route,
                "actual_route": actual_route,
                "correct": correct,
            }
        )
    total = counts["total"]
    return {
        "summary": {
            "total": total,
            "correct": counts["correct"],
            "accuracy": (counts["correct"] / total) if total else 0.0,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic request routing behavior.")
    parser.add_argument("--dataset", default="data/infineon/request_routing_eval.json")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = evaluate(args.dataset, args.schema)
    summary = report["summary"]
    print("===== REQUEST ROUTING EVAL =====")
    print(f"Dataset: {args.dataset}")
    print(f"Total: {summary['total']}")
    print(f"Correct: {summary['correct']} ({summary['accuracy']:.3f})")
    failures = [row for row in report["cases"] if not row["correct"]]
    if failures:
        print("Failures:")
        for row in failures:
            print(f"  {row['id']}: expected={row['expected_route']} actual={row['actual_route']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
