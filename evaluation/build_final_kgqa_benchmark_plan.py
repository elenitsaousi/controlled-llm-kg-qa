#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.feature_extraction import extract_query_plan


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
TARGET_AMBIGUITY_COUNTS = {
    "low": 108,
    "mid": 126,
    "high": 126,
}
DEFAULT_TOTAL = sum(FINAL_BENCHMARK_QUOTAS.values())
DEFAULT_SCHEMA = Path(__file__).resolve().parents[1] / "data" / "infineon" / "schema.json"
COMPLEXITY_BY_CLASS_COUNT = {
    1: "low",
    2: "mid",
}


def _class_count_complexity(query: str, schema: Dict[str, object]) -> Dict[str, object]:
    plan = extract_query_plan(query, schema=schema)
    classes = sorted(set(str(cls) for cls in plan.get("classes", []) if str(cls)))
    explicit_class_count = len(classes)
    # Some validated queries are anchored by a known instance rather than an
    # explicit rdf:type triple. Treat these as one-class queries for the
    # complexity label instead of creating a fourth "zero-class" bucket.
    class_count = max(1, explicit_class_count)
    label = COMPLEXITY_BY_CLASS_COUNT.get(class_count, "high" if class_count >= 3 else "low")
    return {
        "class_count": class_count,
        "explicit_class_count": explicit_class_count,
        "classes": classes,
        "complexity_label": label,
    }


def _next_seed(rows: List[Dict[str, object]], template_counts: Counter) -> Dict[str, object]:
    return min(rows, key=lambda row: (template_counts[str(row["template_id"])], str(row["template_id"])))


def _round_robin_assign(
    rows: List[Dict[str, object]],
    quota: int,
    template_counts: Counter,
) -> List[Dict[str, object]]:
    if not rows:
        raise ValueError("Cannot assign benchmark rows from an empty family.")
    assignments = []
    for slot in range(quota):
        seed = _next_seed(rows, template_counts)
        template_counts[str(seed["template_id"])] += 1
        assignments.append(
            {
                "slot_index": slot + 1,
                "template_id": seed["template_id"],
                "family": seed["family"],
                "answer_shape": seed["answer_shape"],
                "seed_ambiguity_label": seed.get("ambiguity_label"),
                "source_id": seed.get("source_id"),
                "example_question": seed.get("example_question"),
                "query": seed.get("query"),
                "variant_index": template_counts[str(seed["template_id"])],
            }
        )
    return assignments


def _scaled_counts(base_counts: Dict[str, int], target_total: int) -> Dict[str, int]:
    if target_total <= 0:
        raise ValueError("target_total must be positive")
    base_total = sum(base_counts.values())
    if base_total <= 0:
        raise ValueError("base_counts must not be empty")

    scaled: Dict[str, int] = {}
    remainders = []
    for key, value in base_counts.items():
        exact = target_total * (value / base_total)
        rounded_down = int(exact)
        scaled[key] = rounded_down
        remainders.append((exact - rounded_down, key))

    missing = target_total - sum(scaled.values())
    for _fraction, key in sorted(remainders, reverse=True)[:missing]:
        scaled[key] += 1
    return scaled


def _interleave_family_rows(by_family_rows: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    rows = []
    max_len = max((len(items) for items in by_family_rows.values()), default=0)
    for idx in range(max_len):
        for family in FINAL_BENCHMARK_QUOTAS:
            family_rows = by_family_rows[family]
            if idx < len(family_rows):
                rows.append(family_rows[idx])
    return rows


def build_plan(
    seed_rows: Iterable[Dict[str, object]],
    target_total: int = DEFAULT_TOTAL,
    schema: Dict[str, object] | None = None,
) -> Dict[str, object]:
    schema = schema or {}
    rows = list(seed_rows)
    by_family: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    for family in by_family:
        by_family[family].sort(key=lambda row: (str(row["answer_shape"]), str(row["template_id"])))

    family_quotas = _scaled_counts(FINAL_BENCHMARK_QUOTAS, target_total)

    family_assignments: Dict[str, List[Dict[str, object]]] = {}
    template_counts: Counter = Counter()
    for family, quota in family_quotas.items():
        family_assignments[family] = _round_robin_assign(
            by_family.get(family, []),
            quota,
            template_counts=template_counts,
        )
    planned_rows = _interleave_family_rows(family_assignments)
    for row in planned_rows:
        complexity = _class_count_complexity(str(row.get("query") or ""), schema=schema)
        row["target_complexity_label"] = complexity["complexity_label"]
        row["query_class_count"] = complexity["class_count"]
        row["explicit_query_class_count"] = complexity["explicit_class_count"]
        row["query_classes"] = complexity["classes"]
        # Runtime ambiguity is determined later from candidate disagreement,
        # not from the gold query alone.
        row["target_ambiguity_label"] = "runtime_candidate_disagreement"

    per_family = Counter(str(row["family"]) for row in planned_rows)
    per_shape = Counter(str(row["answer_shape"]) for row in planned_rows)
    per_complexity = Counter(str(row["target_complexity_label"]) for row in planned_rows)
    per_template = Counter(str(row["template_id"]) for row in planned_rows)
    return {
        "summary": {
            "total_questions": len(planned_rows),
            "families": dict(per_family),
            "answer_shapes": dict(per_shape),
            "structural_complexity": dict(per_complexity),
            "unique_templates": len(per_template),
            "max_reuse_per_template": max(per_template.values(), default=0),
            "target_complexity": {
                "low": "1 distinct query class",
                "mid": "2 distinct query classes",
                "high": "3 or more distinct query classes",
            },
            "ambiguity_definition": (
                "Runtime ambiguity is candidate-set ambiguity: multiple valid "
                "candidate answers/queries are plausible and the selector cannot "
                "confidently distinguish the intended one."
            ),
            "family_quotas": dict(family_quotas),
            "class_count_distribution": dict(
                Counter(str(row["query_class_count"]) for row in planned_rows)
            ),
        },
        "rows": planned_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the quota-controlled final KGQA benchmark plan.")
    parser.add_argument("--seed-bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--target-total",
        type=int,
        default=DEFAULT_TOTAL,
        help="Total planned questions. Defaults to the original 360-question benchmark.",
    )
    parser.add_argument(
        "--schema",
        default=str(DEFAULT_SCHEMA),
        help="Schema JSON used to count distinct query classes for ambiguity labels.",
    )
    args = parser.parse_args()
    seed_rows = json.loads(Path(args.seed_bank).read_text(encoding="utf-8"))
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    plan = build_plan(seed_rows, target_total=args.target_total, schema=schema)
    Path(args.out).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    summary = plan["summary"]
    print("===== FINAL KGQA BENCHMARK PLAN =====")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Unique templates used: {summary['unique_templates']}")
    print(f"Max reuse per template: {summary['max_reuse_per_template']}")
    print("Families:")
    for key, value in summary["families"].items():
        print(f"  {key}: {value}")
    print("Answer shapes:")
    for key, value in sorted(summary["answer_shapes"].items()):
        print(f"  {key}: {value}")
    print("Structural complexity labels:")
    for key, value in sorted(summary["structural_complexity"].items()):
        print(f"  {key}: {value}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
