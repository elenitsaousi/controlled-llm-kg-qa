"""Build a compact final evaluation summary from canonical report files.

The summary intentionally keeps three metrics separate:

* system-level accuracy on the 500-question engineering audit,
* selection accuracy on the LLM-needed held-out benchmark,
* deterministic ontology-definition routing accuracy.

This prevents the deterministic direct-template layer from being confused with
the harder LLM query-selection benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _pct_points(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if 0.0 <= number <= 1.0:
        number *= 100
    return f"{number:.1f}%"


def _count_pair(correct: object, total: object) -> str:
    if correct is None or total is None:
        return "n/a"
    return f"{correct}/{total}"


def _system_row(report: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not report:
        return {
            "layer": "System-level KGQA",
            "scope": "500 mixed questions",
            "accuracy": "missing",
            "coverage": "n/a",
            "cost": "n/a",
            "note": "Run build_system_accuracy_audit.py on a labeled audit CSV.",
        }
    overall = report.get("overall", {})
    return {
        "layer": "System-level KGQA",
        "scope": f"{overall.get('labeled_rows', 'n/a')} manually labeled questions",
        "accuracy": _pct(overall.get("accuracy")),
        "coverage": "100% of audited rows",
        "cost": "see efficiency row",
        "note": (
            f"{overall.get('correct', 0)} correct, {overall.get('incorrect', 0)} incorrect, "
            f"{overall.get('unclear', 0)} unclear"
        ),
    }


def _efficiency_row(report: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not report:
        return {
            "layer": "Cost / LLM usage",
            "scope": "500 mixed questions",
            "accuracy": "n/a",
            "coverage": "missing",
            "cost": "missing",
            "note": "Run analyze_system_efficiency.py on the 500-question JSONL log.",
        }
    summary = report.get("summary", report)
    queries = summary.get("queries") or summary.get("total_queries") or "n/a"
    llm_calls = summary.get("llm_calls", "n/a")
    cost = summary.get("estimated_cost")
    savings = summary.get("cost_reduction")
    if savings is None:
        savings = summary.get("cost_reduction_pct")
    return {
        "layer": "Cost / LLM usage",
        "scope": f"{queries} questions",
        "accuracy": "n/a",
        "coverage": f"{llm_calls} LLM calls",
        "cost": f"€{float(cost):.2f}" if isinstance(cost, (int, float)) else str(cost or "n/a"),
        "note": f"cost reduction {_pct_points(savings)}" if isinstance(savings, (int, float)) else "cost reduction n/a",
    }


def _selection_row(report: Optional[Dict[str, Any]], label: str) -> Dict[str, str]:
    if not report:
        return {
            "layer": label,
            "scope": "held-out LLM-needed selection",
            "accuracy": "missing",
            "coverage": "n/a",
            "cost": "n/a",
            "note": "Run analyze_infineon_results.py or compare_entropy_regime_selection.py.",
        }
    summary = report.get("summary", report)
    total = summary.get("total") or summary.get("questions") or summary.get("total_with_scores")
    top1_correct = summary.get("top1_correct") or summary.get("ml_top1_correct")
    top1 = summary.get("top1_accuracy") or summary.get("ml_top1_accuracy")
    any_correct = summary.get("any_correct")
    any_acc = summary.get("any_accuracy")
    if top1 is None and top1_correct is not None and total:
        top1 = float(top1_correct) / float(total)
    note_bits: List[str] = []
    if top1_correct is not None:
        note_bits.append(f"Top-1 {_count_pair(top1_correct, total)}")
    if any_correct is not None:
        note_bits.append(f"Any-correct {_count_pair(any_correct, total)} ({_pct(any_acc)})")
    delta = summary.get("delta_accuracy")
    if delta is None:
        delta = summary.get("delta")
    if isinstance(delta, (int, float)):
        note_bits.append(f"delta {_pct(delta)}")
    return {
        "layer": label,
        "scope": f"{total or 'n/a'} held-out questions",
        "accuracy": _pct(top1),
        "coverage": _pct(any_acc) if any_acc is not None else "n/a",
        "cost": "offline evaluation",
        "note": "; ".join(note_bits) or "selection benchmark",
    }


def _ontology_row(report: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not report:
        return {
            "layer": "DR ontology QA",
            "scope": "ontology definition questions",
            "accuracy": "missing",
            "coverage": "deterministic",
            "cost": "€0.00",
            "note": "Run build_dr_ontology_benchmark.py and run_dr_ontology_benchmark.py.",
        }
    summary = report.get("summary", {})
    total = summary.get("total")
    correct = summary.get("correct")
    return {
        "layer": "DR ontology QA",
        "scope": f"{total or 'n/a'} ontology questions",
        "accuracy": _pct(summary.get("accuracy")),
        "coverage": "deterministic",
        "cost": "€0.00",
        "note": f"{_count_pair(correct, total)} correct; {summary.get('llm_calls', 0)} LLM calls",
    }


def build_summary(args: argparse.Namespace) -> Dict[str, Any]:
    rows = [
        _system_row(_load_json(args.system_accuracy)),
        _efficiency_row(_load_json(args.efficiency)),
        _selection_row(_load_json(args.selection), "LLM-needed selection"),
        _selection_row(_load_json(args.baseline_vs_ml), "Baseline vs ML selection"),
        _ontology_row(_load_json(args.dr_ontology)),
    ]
    return {
        "inputs": {
            "system_accuracy": args.system_accuracy,
            "efficiency": args.efficiency,
            "selection": args.selection,
            "baseline_vs_ml": args.baseline_vs_ml,
            "dr_ontology": args.dr_ontology,
        },
        "rows": rows,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Final Evaluation Summary",
        "",
        "Report these rows separately. They answer different evaluation questions and should not be merged into one accuracy number without explanation.",
        "",
        "| Layer | Scope | Accuracy | Coverage / Recall | Cost | Note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['layer']} | {row['scope']} | {row['accuracy']} | "
            f"{row['coverage']} | {row['cost']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use **system-level KGQA** for the engineering claim: how often the complete routed system answers correctly.",
            "- Use **LLM-needed selection** for the scientific claim: how hard query selection remains after candidate generation.",
            "- Use **cost / LLM usage** for the industrial claim: how many paid LLM calls are avoided by deterministic graph-supported routing.",
            "- Use **DR ontology QA** for the ontology-accessibility claim: whether users can ask definition-style questions without SPARQL.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final KGQA/ontology evaluation summary table.")
    parser.add_argument("--system-accuracy", default="results/kgqa_system_accuracy_audit_500_v2_labeled.json")
    parser.add_argument("--efficiency", default="results/kgqa_efficiency_500_after_direct_report.json")
    parser.add_argument("--selection", default="results/infineon_test_final_error_analysis.json")
    parser.add_argument("--baseline-vs-ml", default="results/final1000_wf_test_entropy_regime_schema_vs_ml.json")
    parser.add_argument("--dr-ontology", default="results/dr_ontology_benchmark_report.json")
    parser.add_argument("--out-json", default="results/final_evaluation_summary.json")
    parser.add_argument("--out-md", default="results/final_evaluation_summary.md")
    args = parser.parse_args()

    report = build_summary(args)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")

    print("===== FINAL EVALUATION SUMMARY =====")
    print(f"Rows: {len(report['rows'])}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
