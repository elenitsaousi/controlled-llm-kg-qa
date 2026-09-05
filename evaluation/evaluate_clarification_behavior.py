#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def evaluate(expectations_path: str, results_path: str) -> Dict[str, object]:
    expectations = json.loads(Path(expectations_path).read_text(encoding="utf-8"))
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    by_id = {str(row.get("id")): row for row in list(results.get("details") or [])}
    cases: List[Dict[str, object]] = []
    correct = 0
    for expected in expectations:
        detail = by_id.get(str(expected["id"]), {})
        actual = bool(
            isinstance(detail.get("clarification"), dict)
            and detail["clarification"].get("needs_clarification")
        )
        expected_value = bool(expected["expected_needs_clarification"])
        row_correct = actual == expected_value
        correct += int(row_correct)
        cases.append(
            {
                "id": expected["id"],
                "question": expected["question"],
                "topic": expected.get("topic"),
                "expected_needs_clarification": expected_value,
                "actual_needs_clarification": actual,
                "correct": row_correct,
                "result_present": bool(detail),
            }
        )
    total = len(cases)
    return {
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clarification trigger behavior from QA results.")
    parser.add_argument("--expectations", default="data/infineon/clarification_behavior_eval.json")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = evaluate(args.expectations, args.results)
    summary = report["summary"]
    print("===== CLARIFICATION BEHAVIOR EVAL =====")
    print(f"Expectations: {args.expectations}")
    print(f"Results: {args.results}")
    print(f"Total: {summary['total']}")
    print(f"Correct: {summary['correct']} ({summary['accuracy']:.3f})")
    failures = [row for row in report["cases"] if not row["correct"]]
    if failures:
        print("Failures:")
        for row in failures:
            print(
                f"  {row['id']}: expected={row['expected_needs_clarification']} "
                f"actual={row['actual_needs_clarification']} present={row['result_present']}"
            )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
