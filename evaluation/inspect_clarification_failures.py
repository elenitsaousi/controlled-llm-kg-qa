#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def inspect(expectations_path: str, results_path: str) -> List[Dict[str, object]]:
    expectations = json.loads(Path(expectations_path).read_text(encoding="utf-8"))
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    result_by_id = {str(row.get("id")): row for row in list(results.get("details") or [])}
    rows: List[Dict[str, object]] = []
    for expected in expectations:
        detail = result_by_id.get(str(expected["id"]), {})
        clarification = detail.get("clarification") or {}
        actual = bool(
            isinstance(clarification, dict)
            and clarification.get("needs_clarification")
        )
        wanted = bool(expected["expected_needs_clarification"])
        if actual == wanted:
            continue
        rows.append(
            {
                "id": expected["id"],
                "question": expected["question"],
                "expected": wanted,
                "actual": actual,
                "conflicts": list(clarification.get("conflicts") or []),
                "resolved_intent": dict(clarification.get("resolved_intent") or {}),
                "options": [
                    {
                        "label": option.get("label"),
                        "signature": option.get("signature"),
                    }
                    for option in list(clarification.get("options") or [])
                ],
                "candidates": [
                    {
                        "index": candidate.get("index"),
                        "query": candidate.get("query"),
                    }
                    for candidate in list(detail.get("candidates") or [])
                ],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Print detailed clarification behavior failures.")
    parser.add_argument("--expectations", default="data/infineon/clarification_behavior_eval.json")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = inspect(args.expectations, args.results)
    print("===== CLARIFICATION FAILURE DETAILS =====")
    print(f"Results: {args.results}")
    print(f"Failures: {len(rows)}")
    for row in rows:
        print()
        print(f"{row['id']} | expected={row['expected']} actual={row['actual']}")
        print(f"Q: {row['question']}")
        print(f"resolved_intent={row['resolved_intent']}")
        print(f"conflicts={row['conflicts']}")
        print("options=")
        for option in row["options"]:
            print(f"  - {option['label']} | {option['signature']}")
    if args.out:
        Path(args.out).write_text(json.dumps({"failures": rows}, indent=2), encoding="utf-8")
        print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
