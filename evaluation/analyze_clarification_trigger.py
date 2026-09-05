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

from ranking.clarification import build_clarification_payload


def analyze_results(results_path: str) -> Dict[str, object]:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for detail in payload.get("details", []):
        clarification = build_clarification_payload(
            str(detail.get("question", "")),
            list(detail.get("candidates") or []),
        )
        rows.append(
            {
                "id": detail.get("id"),
                "question": detail.get("question"),
                "needs_clarification": bool(clarification),
                "conflicts": list((clarification or {}).get("conflicts") or []),
                "resolved_intent": dict((clarification or {}).get("resolved_intent") or {}),
                "top1_correct": detail.get("top1_correct"),
                "any_correct": detail.get("any_correct"),
            }
        )
    triggered = [row for row in rows if row["needs_clarification"]]
    conflict_counts = Counter()
    for row in triggered:
        for conflict in row["conflicts"]:
            conflict_counts[str(conflict.get("axis"))] += 1
    return {
        "summary": {
            "total": len(rows),
            "triggered": len(triggered),
            "trigger_rate": (len(triggered) / len(rows)) if rows else 0.0,
            "conflict_axis_counts": dict(conflict_counts),
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze structural clarification trigger coverage.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = analyze_results(args.results)
    summary = report["summary"]
    print("===== CLARIFICATION TRIGGER ANALYSIS =====")
    print(f"Results: {args.results}")
    print(f"Total: {summary['total']}")
    print(f"Triggered: {summary['triggered']} ({summary['trigger_rate']:.3f})")
    print("Conflict axes:")
    for axis, count in sorted(summary["conflict_axis_counts"].items()):
        print(f"  {axis}: {count}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
