"""Build a mixed full-system KGQA benchmark.

This question set is for engineering evaluation of the deployed system, not for
the LLM-only selection benchmark. It intentionally mixes:

* True Demand KG/data questions,
* Digital Reference ontology definition questions,
* deterministic advisory/business interpretation questions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ADVISORY_QUESTIONS = [
    ("Based on the current demand data, which region should be monitored more closely?", "advisory_current_demand"),
    ("Where should planning attention focus based on current demand by region?", "advisory_current_demand"),
    ("Which region has the strongest current demand signal?", "advisory_current_demand"),
    ("Based on the survey data, which region should I inspect first for demand?", "advisory_current_demand"),
    ("What should I look at first to understand current demand risk by region?", "advisory_current_demand"),
    ("Which region shows the strongest future demand signal?", "advisory_future_demand"),
    ("Based on future demand, where should planning attention focus?", "advisory_future_demand"),
    ("Which region should be monitored for future demand risk?", "advisory_future_demand"),
    ("What should I review first to understand future demand risk by region?", "advisory_future_demand"),
    ("Based on future demand data, which region looks most important?", "advisory_future_demand"),
    ("Which vehicle type shows the strongest future demand signal?", "advisory_vehicle_signal"),
    ("What vehicle type should I review first for future demand?", "advisory_vehicle_signal"),
    ("Based on the graph, which vehicle type has the strongest demand signal?", "advisory_vehicle_signal"),
    ("Which vehicle segment appears most relevant for future demand planning?", "advisory_vehicle_signal"),
    ("Where should I focus if I want to understand vehicle-related future demand?", "advisory_vehicle_signal"),
    ("Which technology category shows the strongest future demand signal?", "advisory_technology_signal"),
    ("Which technology category should be reviewed first for future demand risk?", "advisory_technology_signal"),
    ("Based on the data, which technology category appears most relevant for future demand?", "advisory_technology_signal"),
    ("What technology category should I inspect first for future demand?", "advisory_technology_signal"),
    ("Which technology area looks most important for future demand planning?", "advisory_technology_signal"),
    ("Which survey group appears most exposed to shortage?", "advisory_shortage"),
    ("Based on shortage data, where should planning attention focus?", "advisory_shortage"),
    ("Which survey group should I review first for shortage risk?", "advisory_shortage"),
    ("Where do shortage signals appear strongest in the survey data?", "advisory_shortage"),
    ("What should I inspect first to understand shortage exposure?", "advisory_shortage"),
]


def _load_rows(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [row for row in payload if isinstance(row, dict) and row.get("question")]


def _topic(row: Dict[str, Any]) -> str:
    return str(row.get("topic") or row.get("family") or row.get("expected_route") or "unknown")


def _round_robin_sample(rows: Iterable[Dict[str, Any]], count: int, seed: int) -> List[Dict[str, Any]]:
    rows = list(rows)
    if count <= 0:
        return []
    if count >= len(rows):
        return rows

    rng = random.Random(seed)
    by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_topic[_topic(row)].append(row)
    for topic_rows in by_topic.values():
        rng.shuffle(topic_rows)

    selected: List[Dict[str, Any]] = []
    topics = sorted(by_topic)
    while len(selected) < count and any(by_topic.values()):
        for topic in topics:
            if by_topic[topic]:
                selected.append(by_topic[topic].pop())
                if len(selected) >= count:
                    break
    return selected


def _advisory_rows(count: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if count <= 0:
        return rows
    idx = 0
    while len(rows) < count:
        question, topic = ADVISORY_QUESTIONS[idx % len(ADVISORY_QUESTIONS)]
        suffix = ""
        if idx >= len(ADVISORY_QUESTIONS):
            suffix = " using the graph"
        rows.append(
            {
                "question": question.rstrip("?") + suffix + "?",
                "topic": topic,
                "expected_route": "advisory",
                "expected_layer": "direct_graph_supported",
                "expected_behavior": "deterministic graph-grounded advisory answer",
            }
        )
        idx += 1
    return rows


def _normalize_rows(rows: Iterable[Dict[str, Any]], *, source: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["source_question_id"] = str(row.get("id") or "")
        item["source_benchmark"] = source
        item.pop("id", None)
        normalized.append(item)
    return normalized


def build_full_system_set(
    *,
    kg_questions: str,
    dr_questions: str,
    target_total: int,
    kg_count: int,
    ontology_count: int,
    advisory_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if kg_count + ontology_count + advisory_count != target_total:
        raise ValueError("kg-count + ontology-count + advisory-count must equal target-total")

    kg_rows = _normalize_rows(
        _round_robin_sample(_load_rows(kg_questions), kg_count, seed),
        source="true_demand_kg_gold",
    )
    dr_rows = _normalize_rows(
        _round_robin_sample(_load_rows(dr_questions), ontology_count, seed + 1),
        source="digital_reference_ontology",
    )
    advisory_rows = _normalize_rows(_advisory_rows(advisory_count), source="deterministic_advisory")

    mixed: List[Dict[str, Any]] = []
    pools = [kg_rows, dr_rows, advisory_rows]
    while any(pools):
        for pool in pools:
            if pool:
                mixed.append(pool.pop(0))

    for idx, row in enumerate(mixed, start=1):
        row["id"] = f"FULLKGQA{idx:04d}"
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed full-system KGQA evaluation set.")
    parser.add_argument("--kg-questions", required=True)
    parser.add_argument("--dr-questions", required=True)
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--kg-count", type=int, default=800)
    parser.add_argument("--ontology-count", type=int, default=150)
    parser.add_argument("--advisory-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="evaluation/question_sets/true_demand_full_system_1000.json")
    args = parser.parse_args()

    rows = build_full_system_set(
        kg_questions=args.kg_questions,
        dr_questions=args.dr_questions,
        target_total=args.target_total,
        kg_count=args.kg_count,
        ontology_count=args.ontology_count,
        advisory_count=args.advisory_count,
        seed=args.seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("source_benchmark") or "unknown")] += 1
    print("===== FULL SYSTEM QUESTION SET =====")
    print(f"Rows: {len(rows)}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()

