#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.apply_ml_ranker_to_results import apply_ml_ranker


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the best guarded ML thresholds selected by a calibration sweep."
    )
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sweep = _load_json(args.sweep)
    best = dict(sweep.get("best") or {})
    grid = dict(sweep.get("grid") or {})
    if not best:
        raise RuntimeError(f"No best sweep row found in {args.sweep}")

    updated = apply_ml_ranker(
        results_path=args.results,
        model_path=args.model,
        schema_path=args.schema,
        guarded=True,
        min_margin=float(best["min_margin"]),
        min_score=float(best["min_score"]),
        max_rank=int(best["max_rank"]),
        structured_guard=bool(grid.get("structured_guard", False)),
        enable_rank2_trusted_rescue=bool(grid.get("enable_rank2_trusted_rescue", False)),
        trusted_rescue_max_rank=int(grid.get("trusted_rescue_max_rank", 2)),
        trusted_rescue_min_score=float(grid.get("trusted_rescue_min_score", 0.75)),
        trusted_rescue_min_margin=float(grid.get("trusted_rescue_min_margin", 0.25)),
        trusted_rescue_topics=list(grid.get("trusted_rescue_topics") or []),
        enable_shortage_status_rescue=bool(grid.get("enable_shortage_status_rescue", False)),
        shortage_status_rescue_max_rank=int(grid.get("shortage_status_rescue_max_rank", 3)),
        shortage_status_rescue_min_score=float(grid.get("shortage_status_rescue_min_score", 0.45)),
        shortage_status_rescue_min_margin=float(grid.get("shortage_status_rescue_min_margin", -0.05)),
        enable_current_baseline_rescue=bool(grid.get("enable_current_baseline_rescue", False)),
        current_baseline_rescue_max_rank=int(grid.get("current_baseline_rescue_max_rank", 4)),
        current_baseline_rescue_min_score=float(grid.get("current_baseline_rescue_min_score", 0.35)),
        current_baseline_rescue_min_margin=float(grid.get("current_baseline_rescue_min_margin", -0.10)),
    )

    updated["applied_calibration_sweep"] = {
        "sweep": args.sweep,
        "best": best,
        "grid": grid,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = updated["summary"]
    print("===== APPLY BEST GUARDED ML FROM SWEEP =====")
    print(f"Sweep: {args.sweep}")
    print(
        "Thresholds: "
        f"min_margin={best['min_margin']}, "
        f"min_score={best['min_score']}, "
        f"max_rank={best['max_rank']}"
    )
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
