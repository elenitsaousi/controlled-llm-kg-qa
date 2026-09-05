#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


def _load_json(path: str) -> Dict[str, List[Dict[str, object]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Training data must be a JSON object: {path}")
    return payload


def _prefixed_rows(
    payload: Dict[str, List[Dict[str, object]]],
    prefix: str,
    source_name: str,
) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for qid, rows in payload.items():
        new_qid = f"{prefix}{qid}"
        new_rows: List[Dict[str, object]] = []
        for row in list(rows or []):
            item = deepcopy(row)
            item["training_source"] = source_name
            item["original_group_id"] = qid
            new_rows.append(item)
        if new_rows:
            out[new_qid] = new_rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge historical ranker training data with calibration rows while "
            "prefixing group IDs to avoid accidental collisions."
        )
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-prefix", default="base__")
    parser.add_argument("--extra-prefix", default="cal__")
    args = parser.parse_args()

    base = _prefixed_rows(_load_json(args.base), args.base_prefix, "base")
    extra = _prefixed_rows(_load_json(args.extra), args.extra_prefix, "calibration")
    merged = dict(base)
    overlap = set(merged) & set(extra)
    if overlap:
        raise RuntimeError(f"Unexpected prefixed ID overlap: {sorted(overlap)[:5]}")
    merged.update(extra)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")

    base_candidates = sum(len(v or []) for v in base.values())
    extra_candidates = sum(len(v or []) for v in extra.values())
    print("===== MERGED RANKER TRAINING DATA =====")
    print(f"Base groups: {len(base)} candidates: {base_candidates}")
    print(f"Calibration groups: {len(extra)} candidates: {extra_candidates}")
    print(f"Merged groups: {len(merged)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
