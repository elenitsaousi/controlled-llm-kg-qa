"""Evaluate deterministic Digital Reference ontology question answering."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg.schema import load_schema
from pipeline.request_routing import route_request


def _load_rows(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark file must contain a JSON list.")
    return [row for row in payload if isinstance(row, dict) and row.get("question")]


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _evaluate_row(row: Dict[str, Any], schema: Any) -> Dict[str, Any]:
    started = time.perf_counter()
    question = str(row.get("question") or "")
    result = route_request(question, schema=schema, alias_index=None)
    expected_route = str(row.get("expected_route") or "definition")
    expected_source = str(row.get("expected_source") or "digital_reference_ontology")
    expected_term = str(row.get("expected_term") or "")
    expected_kind = str(row.get("expected_kind") or "")

    route_ok = _norm(result.get("route")) == _norm(expected_route)
    source_ok = _norm(result.get("source")) == _norm(expected_source)
    term_ok = _norm(result.get("matched_term")) == _norm(expected_term)
    kind_ok = not expected_kind or _norm(result.get("term_kind")) == _norm(expected_kind)
    answer_ok = bool(str(result.get("answer") or "").strip())
    correct = route_ok and source_ok and term_ok and kind_ok and answer_ok

    return {
        "id": row.get("id"),
        "question": question,
        "expected_route": expected_route,
        "expected_source": expected_source,
        "expected_term": expected_term,
        "expected_kind": expected_kind,
        "actual_route": result.get("route"),
        "actual_source": result.get("source"),
        "actual_term": result.get("matched_term"),
        "actual_kind": result.get("term_kind"),
        "actual_uri": result.get("term_uri"),
        "answer_preview": str(result.get("answer") or "")[:500],
        "route_ok": route_ok,
        "source_ok": source_ok,
        "term_ok": term_ok,
        "kind_ok": kind_ok,
        "answer_ok": answer_ok,
        "correct": correct,
        "latency_s": round(time.perf_counter() - started, 4),
    }


def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    by_kind: Dict[str, Counter] = {}
    for row in rows:
        kind = str(row.get("expected_kind") or "unknown")
        by_kind.setdefault(kind, Counter())["total"] += 1
        by_kind[kind]["correct"] += int(bool(row.get("correct")))
    return {
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "llm_calls": 0,
            "deterministic": True,
        },
        "by_kind": {
            kind: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for kind, counts in sorted(by_kind.items())
        },
        "rows": rows,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Digital Reference Ontology Benchmark",
        "",
        "This benchmark evaluates deterministic ontology definition routing. It does not use LLM calls.",
        "",
        "## Summary",
        "",
        "| Questions | Correct | Accuracy | LLM Calls |",
        "|---:|---:|---:|---:|",
        f"| {summary['total']} | {summary['correct']} | {summary['accuracy']:.3f} | {summary['llm_calls']} |",
        "",
        "## By Ontology Term Type",
        "",
        "| Term type | Questions | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for kind, row in report.get("by_kind", {}).items():
        lines.append(f"| {kind} | {row['total']} | {row['correct']} | {row['accuracy']:.3f} |")
    wrong = [row for row in report.get("rows", []) if not row.get("correct")]
    if wrong:
        lines.extend(["", "## Failures", "", "| ID | Question | Expected | Actual |", "|---|---|---|---|"])
        for row in wrong[:30]:
            lines.append(
                f"| {row.get('id')} | {row.get('question')} | {row.get('expected_term')} | {row.get('actual_term')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DR ontology definition routing.")
    parser.add_argument("--benchmark", default="evaluation/question_sets/dr_ontology_benchmark.json")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--dr-ontology", default="")
    parser.add_argument("--out-json", default="results/dr_ontology_benchmark_report.json")
    parser.add_argument("--out-md", default="results/dr_ontology_benchmark_report.md")
    args = parser.parse_args()

    if args.dr_ontology:
        os.environ["TRUE_DEMAND_DR_ONTOLOGY_PATH"] = args.dr_ontology

    schema = load_schema(args.schema)
    evaluated = [_evaluate_row(row, schema) for row in _load_rows(args.benchmark)]
    report = build_report(evaluated)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")

    summary = report["summary"]
    print("===== DR ONTOLOGY BENCHMARK =====")
    print(f"Benchmark: {args.benchmark}")
    print(f"Questions: {summary['total']}")
    print(f"Correct: {summary['correct']} ({summary['accuracy']:.3f})")
    print(f"LLM calls: {summary['llm_calls']}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
