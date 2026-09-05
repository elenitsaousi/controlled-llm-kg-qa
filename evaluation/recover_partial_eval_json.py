"""Recover completed details from a partially written evaluation JSON.

Use this when a long JSON evaluation file was interrupted mid-write and normal
json.load fails. The script scans the top-level "details" array and keeps only
complete JSON objects. It writes a valid resume file with a minimal summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _find_details_array(text: str) -> tuple[int, int]:
    key = '"details"'
    key_pos = text.find(key)
    if key_pos < 0:
        raise ValueError('Could not find top-level "details" key.')
    start = text.find("[", key_pos)
    if start < 0:
        raise ValueError('Could not find start of "details" array.')
    return start + 1, len(text)


def _extract_complete_objects(text: str) -> List[Dict[str, Any]]:
    start, end = _find_details_array(text)
    rows: List[Dict[str, Any]] = []
    i = start
    decoder = json.JSONDecoder()
    while i < end:
        while i < end and text[i] in " \r\n\t,":
            i += 1
        if i >= end or text[i] == "]":
            break
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, next_i = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            rows.append(obj)
        i = next_i
    return rows


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    top1 = sum(1 for row in rows if row.get("top1_correct"))
    any_correct = sum(1 for row in rows if row.get("any_correct"))
    gen_failures = sum(1 for row in rows if row.get("generation_error"))
    return {
        "total": total,
        "top1_correct": top1,
        "top1_correct_rate": top1 / total if total else 0.0,
        "any_correct": any_correct,
        "any_correct_rate": any_correct / total if total else 0.0,
        "llm_generation_failures": gen_failures,
        "recovered_partial_json": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover a valid evaluation JSON from a partial/corrupt JSON file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    rows = _extract_complete_objects(text)
    payload = {"summary": _summary(rows), "details": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("===== RECOVER PARTIAL EVAL JSON =====")
    print(f"Input: {args.input}")
    print(f"Recovered details: {len(rows)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
