#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


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


def _next_seed(rows: List[Dict[str, object]], template_counts: Counter) -> Dict[str, object]:
    return min(rows, key=lambda row: (template_counts[str(row["template_id"])], str(row["template_id"])))


def _balanced_assign(
    rows: List[Dict[str, object]],
    quota: int,
    ambiguity_remaining: Dict[str, int],
    template_counts: Counter,
) -> List[Dict[str, object]]:
    if not rows:
        raise ValueError("Cannot assign benchmark rows from an empty family.")
    by_label: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("ambiguity_label") or "unknown"), []).append(row)
    assignments = []
    for slot in range(quota):
        available_labels = sorted(by_label)
        label = max(
            available_labels,
            key=lambda item: (
                ambiguity_remaining.get(item, 0),
                -sum(template_counts[str(row["template_id"])] for row in by_label[item]),
                item,
            ),
        )
        seed = _next_seed(by_label[label], template_counts)
        template_counts[str(seed["template_id"])] += 1
        if ambiguity_remaining.get(label, 0) > 0:
            ambiguity_remaining[label] -= 1
        assignments.append(
            {
                "slot_index": slot + 1,
                "template_id": seed["template_id"],
                "family": seed["family"],
                "answer_shape": seed["answer_shape"],
                "ambiguity_label": seed.get("ambiguity_label"),
                "source_id": seed.get("source_id"),
                "example_question": seed.get("example_question"),
                "query": seed.get("query"),
                "variant_index": template_counts[str(seed["template_id"])],
            }
        )
    return assignments


def build_plan(seed_rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(seed_rows)
    by_family: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    for family in by_family:
        by_family[family].sort(key=lambda row: (str(row["answer_shape"]), str(row["template_id"])))

    planned_rows: List[Dict[str, object]] = []
    ambiguity_remaining = dict(TARGET_AMBIGUITY_COUNTS)
    template_counts: Counter = Counter()
    family_order = sorted(
        FINAL_BENCHMARK_QUOTAS,
        key=lambda family: (
            len({str(row.get("ambiguity_label") or "unknown") for row in by_family.get(family, [])}),
            family,
        ),
    )
    for family in family_order:
        quota = FINAL_BENCHMARK_QUOTAS[family]
        planned_rows.extend(
            _balanced_assign(
                by_family.get(family, []),
                quota,
                ambiguity_remaining=ambiguity_remaining,
                template_counts=template_counts,
            )
        )

    per_family = Counter(str(row["family"]) for row in planned_rows)
    per_shape = Counter(str(row["answer_shape"]) for row in planned_rows)
    per_ambiguity = Counter(str(row.get("ambiguity_label") or "unknown") for row in planned_rows)
    per_template = Counter(str(row["template_id"]) for row in planned_rows)
    return {
        "summary": {
            "total_questions": len(planned_rows),
            "families": dict(per_family),
            "answer_shapes": dict(per_shape),
            "ambiguity": dict(per_ambiguity),
            "unique_templates": len(per_template),
            "max_reuse_per_template": max(per_template.values(), default=0),
            "target_ambiguity": dict(TARGET_AMBIGUITY_COUNTS),
        },
        "rows": planned_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the quota-controlled final KGQA benchmark plan.")
    parser.add_argument("--seed-bank", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seed_rows = json.loads(Path(args.seed_bank).read_text(encoding="utf-8"))
    plan = build_plan(seed_rows)
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
    print("Ambiguity labels:")
    for key, value in sorted(summary["ambiguity"].items()):
        print(f"  {key}: {value}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
