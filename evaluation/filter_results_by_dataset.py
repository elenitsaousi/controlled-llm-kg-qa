#!/usr/bin/env python3
import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.apply_selection_to_results import _recompute_summary


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_ids(path: str) -> Set[str]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Dataset must be a JSON list: {path}")
    ids = {str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()}
    if not ids:
        raise RuntimeError(f"No question IDs found in dataset: {path}")
    return ids


def filter_results(results_path: str, dataset_path: str) -> Dict[str, object]:
    payload = _load_json(results_path)
    keep_ids = _dataset_ids(dataset_path)
    details = [
        deepcopy(detail)
        for detail in list(payload.get("details") or [])
        if str(detail.get("id", "")).strip() in keep_ids
    ]
    missing_ids = sorted(keep_ids - {str(detail.get("id", "")).strip() for detail in details})

    out = deepcopy(payload)
    out["details"] = details
    out["summary"] = _recompute_summary(details, payload.get("summary") or {})
    out["filtered_results"] = {
        "source_results": results_path,
        "dataset": dataset_path,
        "requested_ids": len(keep_ids),
        "kept_details": len(details),
        "missing_ids": missing_ids,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter an evaluation results JSON to the question IDs from a dataset split."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = filter_results(args.results, args.dataset)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = out["summary"]
    meta = out["filtered_results"]
    print("===== FILTER RESULTS BY DATASET =====")
    print(f"Input results: {args.results}")
    print(f"Dataset split: {args.dataset}")
    print(f"Requested IDs: {meta['requested_ids']}")
    print(f"Kept details: {meta['kept_details']}")
    print(f"Missing IDs: {len(meta['missing_ids'])}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
