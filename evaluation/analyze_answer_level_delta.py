#!/usr/bin/env python3
"""Find cases where strict selection and audited answer correctness diverge."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _load_json(path: str | None) -> Dict[str, object]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_audit(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _selection_rows(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = payload.get("details") or payload.get("rows") or []
    return rows if isinstance(rows, list) else []


def _selection_correct(row: Dict[str, object]) -> bool:
    if isinstance(row.get("top1_correct"), bool):
        return bool(row["top1_correct"])
    candidates = row.get("candidates") or []
    if isinstance(candidates, list) and candidates:
        return str(candidates[0].get("label") or "").lower() == "correct"
    return False


def _row_id(row: Dict[str, object]) -> str:
    return str(row.get("id") or row.get("question_id") or row.get("request_id") or "")


def _match_selection(audit_row: Dict[str, str], by_id: Dict[str, Dict[str, object]], by_question: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    rid = str(audit_row.get("request_id") or audit_row.get("id") or "")
    if rid in by_id:
        return by_id[rid]
    return by_question.get(_norm(audit_row.get("question") or ""), {})


def analyze(selection_results: str, audit_csv: str, max_examples: int) -> Dict[str, object]:
    selection = _selection_rows(_load_json(selection_results))
    by_id = {_row_id(row): row for row in selection if _row_id(row)}
    by_question = {_norm(str(row.get("question") or row.get("effective_question") or "")): row for row in selection}
    audit_rows = _load_audit(audit_csv)

    matched = 0
    strict_wrong_answer_correct = []
    strict_correct_answer_wrong = []
    both_correct = 0
    both_wrong = 0

    for audit in audit_rows:
        sel = _match_selection(audit, by_id, by_question)
        if not sel:
            continue
        matched += 1
        strict_ok = _selection_correct(sel)
        answer_ok = str(audit.get("correctness") or "").strip().lower() == "correct"
        if strict_ok and answer_ok:
            both_correct += 1
        elif (not strict_ok) and (not answer_ok):
            both_wrong += 1
        elif (not strict_ok) and answer_ok:
            strict_wrong_answer_correct.append(
                {
                    "request_id": audit.get("request_id"),
                    "question": audit.get("question"),
                    "selected_source": audit.get("selected_source"),
                    "route": audit.get("route"),
                    "answer_text": audit.get("answer_text"),
                    "selected_query": audit.get("selected_query"),
                }
            )
        elif strict_ok and (not answer_ok):
            strict_correct_answer_wrong.append(
                {
                    "request_id": audit.get("request_id"),
                    "question": audit.get("question"),
                    "selected_source": audit.get("selected_source"),
                    "route": audit.get("route"),
                    "answer_text": audit.get("answer_text"),
                    "selected_query": audit.get("selected_query"),
                }
            )

    return {
        "selection_results": selection_results,
        "audit_csv": audit_csv,
        "summary": {
            "audit_rows": len(audit_rows),
            "matched_rows": matched,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "strict_wrong_but_answer_correct": len(strict_wrong_answer_correct),
            "strict_correct_but_answer_wrong": len(strict_correct_answer_wrong),
        },
        "strict_wrong_but_answer_correct_examples": strict_wrong_answer_correct[:max_examples],
        "strict_correct_but_answer_wrong_examples": strict_correct_answer_wrong[:max_examples],
    }


def _write_md(report: Dict[str, object], out_md: str) -> None:
    s = report["summary"]
    lines = [
        "# Selection vs Answer-Level Correctness Delta",
        "",
        "| Metric | Count |",
        "|---|---:|",
    ]
    for key, value in s.items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines.extend(["", "## Strict Selection Wrong, Audited Answer Correct", ""])
    for row in report["strict_wrong_but_answer_correct_examples"]:
        lines.append(f"- `{row.get('request_id')}` {row.get('question')}")
        lines.append(f"  - Source: {row.get('selected_source')}; route: {row.get('route')}")
        lines.append(f"  - Answer: {str(row.get('answer_text') or '')[:300]}")
    lines.extend(["", "## Strict Selection Correct, Audited Answer Wrong", ""])
    for row in report["strict_correct_but_answer_wrong_examples"]:
        lines.append(f"- `{row.get('request_id')}` {row.get('question')}")
        lines.append(f"  - Source: {row.get('selected_source')}; route: {row.get('route')}")
        lines.append(f"  - Answer: {str(row.get('answer_text') or '')[:300]}")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze strict query-selection correctness vs answer-level audit correctness.")
    parser.add_argument("--selection-results", required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(args.selection_results, args.audit_csv, args.max_examples)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(report, args.out_md)
    print("===== SELECTION VS ANSWER DELTA =====")
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
