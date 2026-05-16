#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_kgqa_seed_bank import summarize_seed_bank


TARGET_SHAPES = ["raw_or_lookup", "count", "sum", "average", "ranking_top"]
TARGET_AMBIGUITY_SHARES = {"low": 0.30, "mid": 0.35, "high": 0.35}
MAX_VARIANTS_PER_TEMPLATE = 4
FINAL_BENCHMARK_QUOTAS = {
    "regional_demand": 40,
    "current_demand_baselines": 40,
    "future_demand": 50,
    "vehicle_sales": 45,
    "autonomous_driving": 40,
    "order_cancellation": 35,
    "shortage": 35,
    "inventory": 35,
    "catalog_lookup": 40,
}
MIN_SHAPE_DIVERSITY = 3


def _target_seed_templates(final_question_quota: int) -> int:
    # One validated template can usually support a small controlled paraphrase set.
    return int(math.ceil(final_question_quota / 4.0))


def build_report(seed_bank_path: str) -> Dict[str, object]:
    rows: List[Dict[str, object]] = json.loads(Path(seed_bank_path).read_text(encoding="utf-8"))
    summary = summarize_seed_bank(rows)
    families = dict(summary["families"])
    matrix = dict(summary["family_shape_matrix"])

    gaps = []
    family_rows = []
    for family, final_quota in sorted(FINAL_BENCHMARK_QUOTAS.items()):
        count = int(families.get(family, 0))
        target_min = _target_seed_templates(final_quota)
        shape_counts = {shape: int(matrix.get(family, {}).get(shape, 0)) for shape in TARGET_SHAPES}
        present_shapes = [shape for shape, value in shape_counts.items() if value > 0]
        ambiguity_counts = {
            label: sum(
                1
                for row in rows
                if row["family"] == family and str(row.get("ambiguity_label") or "unknown") == label
            )
            for label in TARGET_AMBIGUITY_SHARES
        }
        ambiguity_targets = {
            label: int(math.ceil((final_quota * share) / MAX_VARIANTS_PER_TEMPLATE))
            for label, share in TARGET_AMBIGUITY_SHARES.items()
        }
        family_rows.append(
            {
                "family": family,
                "final_question_quota": final_quota,
                "templates": count,
                "target_min_templates": target_min,
                "template_deficit": max(0, target_min - count),
                "shape_counts": shape_counts,
                "present_shapes": present_shapes,
                "ambiguity_counts": ambiguity_counts,
                "ambiguity_target_min_templates": ambiguity_targets,
            }
        )
        if count < target_min:
            gaps.append(
                {
                    "family": family,
                    "gap_type": "low_template_count",
                    "current": count,
                    "target_min": target_min,
                }
            )
        if count > 0 and len(present_shapes) < MIN_SHAPE_DIVERSITY:
            gaps.append(
                {
                    "family": family,
                    "gap_type": "low_shape_diversity",
                    "current": len(present_shapes),
                    "target_min": MIN_SHAPE_DIVERSITY,
                    "present_shapes": present_shapes,
                }
            )
        for label, target_min in ambiguity_targets.items():
            current = ambiguity_counts[label]
            if current < target_min:
                gaps.append(
                    {
                        "family": family,
                        "gap_type": "low_ambiguity_seed_count",
                        "label": label,
                        "current": current,
                        "target_min": target_min,
                    }
                )
    return {
        "summary": summary,
        "family_rows": family_rows,
        "coverage_gaps": gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report KGQA seed coverage gaps by family and answer shape.")
    parser.add_argument("--seed-bank", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = build_report(args.seed_bank)
    summary = dict(report["summary"])
    print("===== KGQA SEED COVERAGE REPORT =====")
    print(f"Unique templates: {summary['unique_templates']}")
    print("Family coverage:")
    for row in report["family_rows"]:
        cells = ", ".join(f"{shape}={row['shape_counts'][shape]}" for shape in TARGET_SHAPES)
        print(
            f"  {row['family']}: templates={row['templates']}/{row['target_min_templates']} "
            f"(final quota={row['final_question_quota']}), {cells}"
        )
        ambiguity = ", ".join(
            f"{label}={row['ambiguity_counts'][label]}/{row['ambiguity_target_min_templates'][label]}"
            for label in TARGET_AMBIGUITY_SHARES
        )
        print(f"    ambiguity seeds: {ambiguity}")
    print("Coverage gaps:")
    for gap in report["coverage_gaps"]:
        if gap["gap_type"] == "low_template_count":
            print(f"  {gap['family']}: only {gap['current']} templates (target >= {gap['target_min']})")
        else:
            if gap["gap_type"] == "low_shape_diversity":
                present = ", ".join(gap["present_shapes"]) or "none"
                print(
                    f"  {gap['family']}: only {gap['current']} distinct answer shapes "
                    f"(target >= {gap['target_min']}; present: {present})"
                )
            else:
                print(
                    f"  {gap['family']}: only {gap['current']} {gap['label']} seeds "
                    f"(target >= {gap['target_min']})"
                )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
