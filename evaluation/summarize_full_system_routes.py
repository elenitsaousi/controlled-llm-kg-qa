#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _accuracy(rows: Iterable[Dict[str, str]]) -> Dict[str, object]:
    rows = list(rows)
    labeled = [r for r in rows if (r.get("correctness") or "").strip()]
    correct = [r for r in labeled if (r.get("correctness") or "").strip().lower() == "correct"]
    incorrect = [r for r in labeled if (r.get("correctness") or "").strip().lower() == "incorrect"]
    unclear = [r for r in labeled if (r.get("correctness") or "").strip().lower() == "unclear"]
    return {
        "total": len(rows),
        "labeled": len(labeled),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "unclear": len(unclear),
        "accuracy": len(correct) / len(labeled) if labeled else 0.0,
    }


def _group_accuracy(rows: List[Dict[str, str]], key: str) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = (row.get(key) or "").strip() or "unknown"
        buckets[value].append(row)
    out = []
    for value, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        summary = _accuracy(bucket)
        summary[key] = value
        out.append(summary)
    return out


def _llm_reason(row: Dict[str, str]) -> str:
    expected = (row.get("expected_route") or "").strip()
    source = (row.get("selected_source") or "").strip()
    topic = (row.get("topic") or "").strip()
    reason = (row.get("execution_selection_reason") or "").strip()
    if expected == "definition":
        return "ontology definition route"
    if expected == "advisory":
        return "advisory/recommendation route"
    if source == "validated_retrieval":
        return "validated retrieval candidate selected"
    if source == "infineon":
        return "LLM-generated KG query selected"
    if source == "capability_inventory":
        return "capability inventory fallback"
    if reason:
        return reason[:120]
    if topic:
        return f"no direct template for topic: {topic}"
    return "no direct template matched"


def _write_md(path: str, payload: Dict[str, object]) -> None:
    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    lines = ["# Full-System Route and Accuracy Breakdown", ""]
    overall = payload["overall"]
    lines.extend(
        [
            "## Overall",
            "",
            "| Total | Correct | Incorrect | Unclear | Accuracy |",
            "|---:|---:|---:|---:|---:|",
            f"| {overall['total']} | {overall['correct']} | {overall['incorrect']} | {overall['unclear']} | {pct(overall['accuracy'])} |",
            "",
        ]
    )

    for title, key in [
        ("By System Mode", "system_mode"),
        ("By Selected Source", "selected_source"),
        ("By Expected Route", "expected_route"),
        ("By Topic", "topic"),
    ]:
        lines.extend(
            [
                f"## {title}",
                "",
                f"| {key} | Total | Correct | Incorrect | Unclear | Accuracy |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in payload[f"by_{key}"]:
            lines.append(
                f"| `{row[key]}` | {row['total']} | {row['correct']} | {row['incorrect']} | "
                f"{row['unclear']} | {pct(row['accuracy'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## LLM Fallback Reasons",
            "",
            "| Reason | Rows |",
            "|---|---:|",
        ]
    )
    for reason, count in payload["llm_fallback_reasons"]:
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full-system route usage and labeled accuracy.")
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    rows = _load_rows(args.audit_csv)
    llm_rows = [r for r in rows if (r.get("system_mode") or "").strip() == "llm_ranking"]
    reason_counts = Counter(_llm_reason(row) for row in llm_rows)
    payload = {
        "audit_csv": args.audit_csv,
        "overall": _accuracy(rows),
        "by_system_mode": _group_accuracy(rows, "system_mode"),
        "by_selected_source": _group_accuracy(rows, "selected_source"),
        "by_expected_route": _group_accuracy(rows, "expected_route"),
        "by_topic": _group_accuracy(rows, "topic"),
        "llm_fallback_reasons": reason_counts.most_common(),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(args.out_md, payload)

    print("===== FULL-SYSTEM ROUTE SUMMARY =====")
    print(f"Rows: {payload['overall']['total']}")
    print(f"Accuracy: {payload['overall']['accuracy']:.3f}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
