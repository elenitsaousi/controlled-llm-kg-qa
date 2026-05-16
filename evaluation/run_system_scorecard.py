#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_json(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clarification_metrics(report: Dict[str, object]) -> Dict[str, object]:
    cases = list(report.get("cases") or [])
    false_pos = sum(
        1
        for row in cases
        if not row.get("expected_needs_clarification") and row.get("actual_needs_clarification")
    )
    false_neg = sum(
        1
        for row in cases
        if row.get("expected_needs_clarification") and not row.get("actual_needs_clarification")
    )
    summary = dict(report.get("summary") or {})
    return {
        "total": summary.get("total", 0),
        "correct": summary.get("correct", 0),
        "accuracy": summary.get("accuracy", 0.0),
        "false_positives": false_pos,
        "false_negatives": false_neg,
    }


def build_scorecard(
    *,
    kgqa_results_path: str,
    routing_report_path: str,
    clarification_report_path: str,
) -> Dict[str, object]:
    kgqa = _load_json(kgqa_results_path)
    routing = _load_json(routing_report_path)
    clarification = _load_json(clarification_report_path)

    kgqa_summary = dict(kgqa.get("summary") or {})
    any_correct = float(kgqa_summary.get("any_correct_rate", 0.0))
    top1 = float(kgqa_summary.get("top1_correct_rate", 0.0))

    return {
        "inputs": {
            "kgqa_results": kgqa_results_path,
            "routing_report": routing_report_path,
            "clarification_report": clarification_report_path,
        },
        "kgqa": {
            "total": kgqa_summary.get("total", 0),
            "top1_correct": kgqa_summary.get("top1_correct", 0),
            "top1_accuracy": top1,
            "any_correct": kgqa_summary.get("any_correct", 0),
            "candidate_recall": any_correct,
            "selection_gap": any_correct - top1,
            "gold_invalid": kgqa_summary.get("gold_invalid", 0),
            "gold_timeout": kgqa_summary.get("gold_timeout", 0),
            "llm_generation_failures": kgqa_summary.get("llm_generation_failures", 0),
        },
        "routing": dict(routing.get("summary") or {}),
        "clarification": _clarification_metrics(clarification),
    }


def _print_scorecard(scorecard: Dict[str, object]) -> None:
    kgqa = dict(scorecard["kgqa"])
    routing = dict(scorecard["routing"])
    clarification = dict(scorecard["clarification"])

    print("===== SYSTEM SCORECARD =====")
    print(
        "KGQA: "
        f"top1={kgqa['top1_correct']}/{kgqa['total']} ({kgqa['top1_accuracy']:.3f}) | "
        f"any_correct={kgqa['any_correct']}/{kgqa['total']} ({kgqa['candidate_recall']:.3f}) | "
        f"selection_gap={kgqa['selection_gap']:.3f}"
    )
    print(
        "Routing: "
        f"correct={routing.get('correct', 0)}/{routing.get('total', 0)} "
        f"({float(routing.get('accuracy', 0.0)):.3f})"
    )
    print(
        "Clarification: "
        f"correct={clarification['correct']}/{clarification['total']} "
        f"({clarification['accuracy']:.3f}) | "
        f"fp={clarification['false_positives']} | fn={clarification['false_negatives']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine KGQA, routing, and clarification metrics.")
    parser.add_argument("--kgqa-results", required=True)
    parser.add_argument("--routing-report", required=True)
    parser.add_argument("--clarification-report", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    scorecard = build_scorecard(
        kgqa_results_path=args.kgqa_results,
        routing_report_path=args.routing_report,
        clarification_report_path=args.clarification_report,
    )
    _print_scorecard(scorecard)
    if args.out:
        Path(args.out).write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
