"""Summarize KGQA runtime efficiency from Streamlit session logs.

The Streamlit app writes one JSONL row per question to logs/kgqa_sessions.jsonl.
This report estimates LLM calls and cost for thesis/demo tables. Cost is counted
per LLM request, not per token, because the Infineon deployment cost discussed in
the thesis work is charged per call.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _llm_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("llm")
    return value if isinstance(value, dict) else {}


def _estimated_calls(row: Dict[str, Any]) -> int:
    llm = _llm_payload(row)
    if "estimated_calls" in llm:
        return _as_int(llm.get("estimated_calls"), 0)

    # Backward-compatible inference for older logs.
    route = str(row.get("route") or "").lower()
    selected_source = str(row.get("selected_source") or "").lower()
    if route == "auto_answer" and selected_source in {"guided", "capability_inventory"}:
        return 0
    return 1


def _mode(row: Dict[str, Any]) -> str:
    llm = _llm_payload(row)
    route = str(row.get("route") or "").lower()
    selected_source = str(row.get("selected_source") or "").lower()
    if llm.get("skipped") or selected_source in {"guided", "capability_inventory"}:
        return "Direct graph-supported"
    if llm.get("cache_hit"):
        return "Cached LLM + ranking"
    if route == "clarification":
        return "LLM + ranking + clarification"
    return "LLM + ranking"


def _row_count(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(1 for _ in rows)


def _summarize_group(
    rows: List[Dict[str, Any]],
    *,
    cost_per_call: float,
    baseline_calls_per_query: float,
) -> Dict[str, Any]:
    count = len(rows)
    calls = sum(_estimated_calls(row) for row in rows)
    baseline_calls = baseline_calls_per_query * count
    latencies = [_as_float(row.get("latency_s")) for row in rows if row.get("latency_s") is not None]
    candidate_counts = [
        _as_float(row.get("candidate_count"))
        for row in rows
        if row.get("candidate_count") is not None
    ]
    graph_rows = [
        _as_float(row.get("graph_row_count"))
        for row in rows
        if row.get("graph_row_count") is not None
    ]
    return {
        "queries": count,
        "avg_latency_ms": round(1000 * mean(latencies), 1) if latencies else None,
        "avg_candidate_count": round(mean(candidate_counts), 2) if candidate_counts else None,
        "llm_calls": calls,
        "avg_llm_calls": round(calls / count, 3) if count else 0.0,
        "estimated_cost": round(calls * cost_per_call, 4),
        "baseline_cost": round(baseline_calls * cost_per_call, 4),
        "estimated_savings": round(max(0.0, baseline_calls - calls) * cost_per_call, 4),
        "cost_reduction_pct": round(
            100 * (1 - (calls / baseline_calls)), 1
        )
        if baseline_calls
        else 0.0,
        "avg_graph_rows": round(mean(graph_rows), 2) if graph_rows else None,
    }


def build_report(
    rows: List[Dict[str, Any]],
    *,
    cost_per_call: float,
    baseline_calls_per_query: float,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_mode(row)].append(row)

    ordered_modes = [
        "Direct graph-supported",
        "Cached LLM + ranking",
        "LLM + ranking",
        "LLM + ranking + clarification",
    ]
    mode_rows = []
    for mode in ordered_modes:
        if mode in groups:
            mode_rows.append(
                {
                    "mode": mode,
                    **_summarize_group(
                        groups[mode],
                        cost_per_call=cost_per_call,
                        baseline_calls_per_query=baseline_calls_per_query,
                    ),
                }
            )
    for mode in sorted(set(groups) - set(ordered_modes)):
        mode_rows.append(
            {
                "mode": mode,
                **_summarize_group(
                    groups[mode],
                    cost_per_call=cost_per_call,
                    baseline_calls_per_query=baseline_calls_per_query,
                ),
            }
        )

    overall = _summarize_group(
        rows,
        cost_per_call=cost_per_call,
        baseline_calls_per_query=baseline_calls_per_query,
    )
    return {
        "summary": {
            "total_queries": len(rows),
            "cost_per_llm_call": cost_per_call,
            "baseline_calls_per_query": baseline_calls_per_query,
            **overall,
        },
        "by_mode": mode_rows,
    }


def _fmt_money(value: object, currency: str) -> str:
    return f"{currency}{_as_float(value):.2f}"


def _fmt_number(value: object, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{_as_float(value):.{digits}f}"


def _render_markdown(report: Dict[str, Any], currency: str) -> str:
    lines = [
        "# System Efficiency Metrics",
        "",
        "Cost is estimated per LLM request. Direct graph-supported answers and exact cache hits are counted as zero new LLM calls.",
        "",
        "## By Mode",
        "",
        "| Mode | #Queries | Avg Latency (ms) | Avg K | LLM Calls | Avg LLM Calls | Estimated Cost | Savings vs all-LLM | Cost Reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("by_mode") or []:
        lines.append(
            "| {mode} | {queries} | {latency} | {avg_k} | {calls} | {avg_calls} | {cost} | {savings} | {reduction}% |".format(
                mode=row["mode"],
                queries=row["queries"],
                latency=_fmt_number(row.get("avg_latency_ms"), 1),
                avg_k=_fmt_number(row.get("avg_candidate_count"), 2),
                calls=row["llm_calls"],
                avg_calls=_fmt_number(row.get("avg_llm_calls"), 3),
                cost=_fmt_money(row.get("estimated_cost"), currency),
                savings=_fmt_money(row.get("estimated_savings"), currency),
                reduction=_fmt_number(row.get("cost_reduction_pct"), 1),
            )
        )

    summary = dict(report.get("summary") or {})
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| #Queries | Avg Latency (ms) | Avg K | LLM Calls | Estimated Cost | Baseline Cost | Savings | Cost Reduction |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| {queries} | {latency} | {avg_k} | {calls} | {cost} | {baseline} | {savings} | {reduction}% |".format(
                queries=summary.get("queries", summary.get("total_queries", 0)),
                latency=_fmt_number(summary.get("avg_latency_ms"), 1),
                avg_k=_fmt_number(summary.get("avg_candidate_count"), 2),
                calls=summary.get("llm_calls", 0),
                cost=_fmt_money(summary.get("estimated_cost"), currency),
                baseline=_fmt_money(summary.get("baseline_cost"), currency),
                savings=_fmt_money(summary.get("estimated_savings"), currency),
                reduction=_fmt_number(summary.get("cost_reduction_pct"), 1),
            ),
            "",
            "## Thesis Wording",
            "",
            (
                f"Using a cost-aware routing layer, the system answered "
                f"{summary.get('queries', summary.get('total_queries', 0))} logged question(s) "
                f"with {summary.get('llm_calls', 0)} estimated LLM request(s). "
                f"At {currency}{_as_float(summary.get('cost_per_llm_call')):.2f} per LLM call, "
                f"this corresponds to an estimated cost of {_fmt_money(summary.get('estimated_cost'), currency)} "
                f"compared with {_fmt_money(summary.get('baseline_cost'), currency)} under a one-LLM-call-per-question baseline."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build system efficiency tables from KGQA Streamlit logs.")
    parser.add_argument("--log", default="logs/kgqa_sessions.jsonl")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--cost-per-call", type=float, default=0.20)
    parser.add_argument("--currency", default="€")
    parser.add_argument(
        "--baseline-calls-per-query",
        type=float,
        default=1.0,
        help="Baseline used for savings. Default assumes one LLM call per question.",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.log)
    report = build_report(
        rows,
        cost_per_call=args.cost_per_call,
        baseline_calls_per_query=args.baseline_calls_per_query,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(_render_markdown(report, args.currency), encoding="utf-8")

    summary = report["summary"]
    print("===== SYSTEM EFFICIENCY METRICS =====")
    print(f"Log: {args.log}")
    print(f"Queries: {summary['queries']}")
    print(f"LLM calls: {summary['llm_calls']}")
    print(f"Estimated cost: {args.currency}{summary['estimated_cost']:.2f}")
    print(f"Baseline cost: {args.currency}{summary['baseline_cost']:.2f}")
    print(f"Estimated savings: {args.currency}{summary['estimated_savings']:.2f}")
    print(f"Cost reduction: {summary['cost_reduction_pct']:.1f}%")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
