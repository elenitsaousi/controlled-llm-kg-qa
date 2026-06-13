"""Build and summarize a manual system-accuracy audit sheet.

This script intentionally separates engineering system accuracy from the
scientific selection benchmark. It consumes a KGQA efficiency/session JSONL log
and produces a CSV sheet where a human can label each answer as correct,
incorrect, or unclear. Running the script again on the filled CSV produces an
accuracy summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CORRECT_VALUES = {"correct", "yes", "y", "1", "true", "ok"}
INCORRECT_VALUES = {"incorrect", "wrong", "no", "n", "0", "false"}
UNCLEAR_VALUES = {"unclear", "ambiguous", "partial", "unknown", "?"}


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


def _load_questions(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(payload, start=1):
        if isinstance(item, dict):
            request_id = str(item.get("id") or f"Q{idx:04d}")
            by_id[request_id] = item
    return by_id


def _mode(row: Dict[str, Any]) -> str:
    llm = row.get("llm") if isinstance(row.get("llm"), dict) else {}
    selected_source = str(row.get("selected_source") or "").lower()
    route = str(row.get("route") or "").lower()
    if llm.get("skipped") or selected_source in {"guided", "capability_inventory"}:
        return "direct_graph_supported"
    if route == "llm_required_estimate":
        return "llm_needed_not_executed"
    if route == "clarification":
        return "clarification"
    return "llm_ranking"


def _estimated_calls(row: Dict[str, Any]) -> int:
    llm = row.get("llm") if isinstance(row.get("llm"), dict) else {}
    try:
        return int(llm.get("estimated_calls", 0))
    except (TypeError, ValueError):
        return 0


def _compact_json(value: object, max_chars: int = 700) -> str:
    if value in (None, "", []):
        return ""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _one_line_query(query: object, max_chars: int = 900) -> str:
    text = " ".join(str(query or "").split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _sample_rows(rows: List[Dict[str, Any]], sample_size: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if not sample_size or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_mode(row)].append(row)
    sampled: List[Dict[str, Any]] = []
    total = len(rows)
    for mode_rows in groups.values():
        take = max(1, round(sample_size * len(mode_rows) / total))
        sampled.extend(rng.sample(mode_rows, min(take, len(mode_rows))))
    if len(sampled) > sample_size:
        sampled = rng.sample(sampled, sample_size)
    sampled.sort(key=lambda row: str(row.get("request_id") or ""))
    return sampled


def build_audit_rows(
    log_rows: List[Dict[str, Any]],
    *,
    questions_by_id: Dict[str, Dict[str, Any]],
    sample_size: Optional[int],
    seed: int,
) -> List[Dict[str, str]]:
    selected = _sample_rows(log_rows, sample_size, seed)
    audit_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(selected, start=1):
        request_id = str(row.get("request_id") or f"ROW{idx:04d}")
        question_meta = questions_by_id.get(request_id, {})
        preview = row.get("graph_rows_preview") or row.get("answer_rows_preview") or []
        audit_rows.append(
            {
                "request_id": request_id,
                "question": str(row.get("question") or question_meta.get("question") or ""),
                "topic": str(question_meta.get("topic") or ""),
                "expected_route": str(question_meta.get("expected_route") or ""),
                "system_mode": _mode(row),
                "route": str(row.get("route") or ""),
                "selected_source": str(row.get("selected_source") or ""),
                "estimated_llm_calls": str(_estimated_calls(row)),
                "candidate_count": str(row.get("candidate_count") if row.get("candidate_count") is not None else ""),
                "graph_row_count": str(row.get("graph_row_count") if row.get("graph_row_count") is not None else ""),
                "graph_error": str(row.get("graph_error") or ""),
                "row_preview": _compact_json(preview),
                "answer_text": str(row.get("answer_text") or ""),
                "execution_selection_reason": str(row.get("execution_selection_reason") or ""),
                "selected_query": _one_line_query(row.get("selected_query")),
                "correctness": "",
                "notes": "",
            }
        )
    return audit_rows


def write_csv(rows: List[Dict[str, str]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id",
        "question",
        "topic",
        "expected_route",
        "system_mode",
        "route",
        "selected_source",
        "estimated_llm_calls",
        "candidate_count",
        "graph_row_count",
        "graph_error",
        "row_preview",
        "answer_text",
        "execution_selection_reason",
        "selected_query",
        "correctness",
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in CORRECT_VALUES:
        return "correct"
    if text in INCORRECT_VALUES:
        return "incorrect"
    if text in UNCLEAR_VALUES:
        return "unclear"
    return "unlabeled"


def _pct(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def summarize_labeled_csv(path: str, *, unclear_as_incorrect: bool) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_mode: Dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for row in rows:
        label = _label(row.get("correctness"))
        mode = str(row.get("system_mode") or "unknown")
        by_mode[mode][label] += 1
        overall[label] += 1

    def metrics(counter: Counter) -> Dict[str, Any]:
        labeled = counter["correct"] + counter["incorrect"] + counter["unclear"]
        denom = labeled if unclear_as_incorrect else counter["correct"] + counter["incorrect"]
        return {
            "total_rows": sum(counter.values()),
            "labeled_rows": labeled,
            "unlabeled_rows": counter["unlabeled"],
            "correct": counter["correct"],
            "incorrect": counter["incorrect"],
            "unclear": counter["unclear"],
            "accuracy": _pct(counter["correct"], denom),
            "denominator": denom,
        }

    return {
        "settings": {"unclear_as_incorrect": unclear_as_incorrect},
        "overall": metrics(overall),
        "by_mode": {mode: metrics(counter) for mode, counter in sorted(by_mode.items())},
    }


def render_markdown(report: Dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# System Accuracy Audit",
        "",
        "This report is for the engineering system view. It should be reported separately from selection accuracy on LLM-needed questions.",
        "",
        "## Overall",
        "",
        "| Labeled | Correct | Incorrect | Unclear | Accuracy | Denominator | Unlabeled |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['labeled_rows']} | {overall['correct']} | {overall['incorrect']} | "
            f"{overall['unclear']} | {overall['accuracy']:.3f} | {overall['denominator']} | "
            f"{overall['unlabeled_rows']} |"
        ),
        "",
        "## By Mode",
        "",
        "| Mode | Labeled | Correct | Incorrect | Unclear | Accuracy | Denominator | Unlabeled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in report.get("by_mode", {}).items():
        lines.append(
            f"| `{mode}` | {row['labeled_rows']} | {row['correct']} | {row['incorrect']} | "
            f"{row['unclear']} | {row['accuracy']:.3f} | {row['denominator']} | {row['unlabeled_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Reporting Note",
            "",
            "Use this as system-level accuracy only after manual labels are completed. Keep it separate from Top-1 selection accuracy on the LLM-needed subset.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or summarize a manual KGQA system accuracy audit.")
    parser.add_argument("--log", help="KGQA JSONL log to convert into an audit CSV.")
    parser.add_argument("--questions", default="evaluation/question_sets/true_demand_efficiency_500.json")
    parser.add_argument("--out-csv", help="Audit CSV to create.")
    parser.add_argument("--sample-size", type=int, help="Optional stratified sample size for manual auditing.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--labeled-csv", help="Filled audit CSV to summarize.")
    parser.add_argument("--out-json", help="Summary JSON path.")
    parser.add_argument("--out-md", help="Summary Markdown path.")
    parser.add_argument("--unclear-as-incorrect", action="store_true")
    args = parser.parse_args()

    if args.log:
        if not args.out_csv:
            raise SystemExit("--out-csv is required when --log is provided.")
        log_rows = _load_jsonl(args.log)
        audit_rows = build_audit_rows(
            log_rows,
            questions_by_id=_load_questions(args.questions),
            sample_size=args.sample_size,
            seed=args.seed,
        )
        write_csv(audit_rows, args.out_csv)
        print("===== SYSTEM ACCURACY AUDIT SHEET =====")
        print(f"Log rows: {len(log_rows)}")
        print(f"Audit rows: {len(audit_rows)}")
        print(f"CSV: {args.out_csv}")

    if args.labeled_csv:
        report = summarize_labeled_csv(args.labeled_csv, unclear_as_incorrect=args.unclear_as_incorrect)
        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.out_md:
            Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
        overall = report["overall"]
        print("===== SYSTEM ACCURACY AUDIT SUMMARY =====")
        print(f"Labeled: {overall['labeled_rows']}")
        print(f"Correct: {overall['correct']}")
        print(f"Incorrect: {overall['incorrect']}")
        print(f"Unclear: {overall['unclear']}")
        print(f"Accuracy: {overall['accuracy']:.3f}")
        if args.out_json:
            print(f"JSON: {args.out_json}")
        if args.out_md:
            print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
