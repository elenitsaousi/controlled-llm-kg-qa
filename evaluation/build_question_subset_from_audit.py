#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_questions(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("questions", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise RuntimeError(f"Unsupported question set format: {path}")


def _load_audit_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _matches(row: Dict[str, str], filters: List[str]) -> bool:
    for item in filters:
        if "=" not in item:
            raise RuntimeError(f"Filter must use key=value syntax: {item}")
        key, value = item.split("=", 1)
        if (row.get(key.strip()) or "").strip() != value.strip():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a question subset from full-system audit rows.")
    parser.add_argument("--questions", required=True, help="Original question-set JSON.")
    parser.add_argument("--audit-csv", required=True, help="Audit CSV with request_id/system_mode/etc.")
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        help="Filter in key=value form. Repeat for AND filters, e.g. --where system_mode=llm_ranking.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-ids", default="")
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    by_id = {str(q.get("id") or q.get("request_id") or ""): q for q in questions}
    audit_rows = _load_audit_rows(args.audit_csv)
    selected_ids = [r["request_id"] for r in audit_rows if _matches(r, args.where)]
    missing = [qid for qid in selected_ids if qid not in by_id]
    selected = [by_id[qid] for qid in selected_ids if qid in by_id]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if args.out_ids:
        Path(args.out_ids).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_ids, "w", encoding="utf-8") as f:
            json.dump({"count": len(selected), "missing": missing, "ids": [str(q.get("id")) for q in selected]}, f, indent=2)
            f.write("\n")

    print("===== QUESTION SUBSET FROM AUDIT =====")
    print(f"Questions: {args.questions}")
    print(f"Audit: {args.audit_csv}")
    print(f"Filters: {args.where}")
    print(f"Selected: {len(selected)}")
    print(f"Missing IDs: {len(missing)}")
    print(f"Output: {args.out}")
    if args.out_ids:
        print(f"IDs: {args.out_ids}")


if __name__ == "__main__":
    main()
