"""Build a controlled question set for KGQA cost/efficiency experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set


def _add(rows: List[Dict[str, str]], seen: Set[str], question: str, topic: str, expected_route: str) -> None:
    cleaned = " ".join(question.strip().split())
    key = cleaned.lower()
    if not cleaned or key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "id": f"EFFKGQA{len(rows) + 1:04d}",
            "question": cleaned,
            "topic": topic,
            "expected_route": expected_route,
        }
    )


def build_questions(target: int = 500, seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    seen: Set[str] = set()

    future_dims = [
        "region",
        "quarter",
        "vehicle type",
        "technology category",
    ]
    future_templates = [
        "How does future demand change by {dim}?",
        "Show future demand by {dim}.",
        "Give me future demand change grouped by {dim}.",
        "What is the average future demand percentage change by {dim}?",
        "Break down future demand by {dim}.",
        "Summarize future demand across {dim}.",
        "Return future demand values for each {dim}.",
        "How is future demand distributed by {dim}?",
        "Compare future demand change across {dim}.",
        "Can you show the future demand trend by {dim}?",
        "Future demand by {dim}, please.",
        "What does future demand look like for each {dim}?",
        "Analyze future demand change per {dim}.",
        "Give a graph-backed breakdown of future demand by {dim}.",
        "Show the average future demand change for every {dim}.",
        "How does demand forecast vary by {dim}?",
        "Display future-demand percentage by {dim}.",
        "I need future demand grouped by {dim}.",
        "For future demand, show the result by {dim}.",
        "Use the True Demand KG to show future demand by {dim}.",
        "How does future demand evolve across {dim}?",
        "Show future demand percentage changes by {dim}.",
        "Return the future demand analysis by {dim}.",
        "Give me the future demand overview by {dim}.",
        "What are the future demand changes for each {dim}?",
    ]
    for dim in future_dims:
        for template in future_templates:
            _add(
                rows,
                seen,
                template.format(dim=dim),
                topic="future_demand",
                expected_route="direct_capability",
            )

    domain_blocks = {
        "regional_demand": {
            "metrics": [
                "current demand",
                "regional demand",
                "OEM current demand",
                "Tier1 current demand",
                "semiconductor current demand",
                "total demand",
            ],
            "dimensions": ["region", "survey group", "vehicle type", "quarter"],
            "templates": [
                "Show {metric} by {dim}.",
                "Give me {metric} grouped by {dim}.",
                "What is the total {metric} for each {dim}?",
                "Compare {metric} across {dim}.",
                "Return a breakdown of {metric} by {dim}.",
                "How does {metric} vary by {dim}?",
                "Summarize {metric} per {dim}.",
                "Use the graph to show {metric} by {dim}.",
            ],
        },
        "vehicle_sales": {
            "metrics": [
                "actual vehicle sales",
                "forecast vehicle sales",
                "vehicle sales units",
                "vehicles sold",
                "sales volume",
            ],
            "dimensions": ["month", "year", "vehicle type", "time period"],
            "templates": [
                "Show {metric} by {dim}.",
                "What is the total number of {metric} for each {dim}?",
                "Give monthly results for {metric}.",
                "Compare {metric} across {dim}.",
                "Return {metric} grouped by {dim}.",
                "Which {dim} has the highest {metric}?",
                "Summarize {metric} over {dim}.",
            ],
        },
        "shortage": {
            "metrics": [
                "shortage information",
                "companies reporting shortage",
                "shortage status",
                "reported shortages",
                "semiconductor shortage responses",
            ],
            "dimensions": ["survey group", "shortage status", "technology category"],
            "templates": [
                "Show {metric} by {dim}.",
                "Count {metric} for each {dim}.",
                "Summarize {metric} across {dim}.",
                "Compare {metric} by {dim}.",
                "Return {metric} grouped by {dim}.",
                "How many companies are in each {dim} for {metric}?",
            ],
        },
        "autonomous_driving": {
            "metrics": [
                "autonomous driving development",
                "autonomous driving percentage",
                "SAE level development",
                "self-driving development",
            ],
            "dimensions": ["vehicle type", "SAE level", "year", "survey group"],
            "templates": [
                "Show average {metric} by {dim}.",
                "Compare {metric} across {dim}.",
                "Return {metric} grouped by {dim}.",
                "What is the average {metric} for each {dim}?",
                "Which {dim} has the highest {metric}?",
                "Summarize {metric} by {dim}.",
            ],
        },
        "inventory": {
            "metrics": [
                "inventory trend",
                "inventory development",
                "inventory responses",
                "component inventory",
            ],
            "dimensions": ["component", "technology category", "trend"],
            "templates": [
                "Show {metric} by {dim}.",
                "Summarize {metric} across {dim}.",
                "Return {metric} grouped by {dim}.",
                "Count {metric} for each {dim}.",
                "Compare {metric} by {dim}.",
                "Which {dim} appears most often in {metric}?",
            ],
        },
        "order_cancellation": {
            "metrics": [
                "order cancellation",
                "order cancellation responses",
                "cancellation participant counts",
                "cancellation behavior",
            ],
            "dimensions": ["technology category", "response type"],
            "templates": [
                "Show {metric} by {dim}.",
                "Summarize {metric} across {dim}.",
                "Return {metric} grouped by {dim}.",
                "Count {metric} for each {dim}.",
                "Compare {metric} by {dim}.",
                "Which {dim} has the highest {metric}?",
            ],
        },
    }

    for topic, spec in domain_blocks.items():
        for metric in spec["metrics"]:
            for dim in spec["dimensions"]:
                templates = list(spec["templates"])
                rng.shuffle(templates)
                for template in templates:
                    _add(
                        rows,
                        seen,
                        template.format(metric=metric, dim=dim),
                        topic=topic,
                        expected_route="llm_or_ranking",
                    )

    catalog_questions = [
        "List all region names in the True Demand KG.",
        "How many distinct regions exist in the True Demand KG?",
        "Which technology categories are available in the graph?",
        "List quarter labels in the data.",
        "How many company instances are present?",
        "How many FutureDemandAnalysis entries exist overall?",
        "How many OrderCancellation entries exist overall?",
        "How many AutonomousDrivingDevelopment entries exist?",
        "List available survey groups.",
        "Show the catalog of technology categories.",
    ]
    for question in catalog_questions:
        _add(rows, seen, question, topic="catalog_lookup", expected_route="llm_or_ranking")

    prefixes = [
        "Please",
        "Can you",
        "Using the True Demand KG,",
        "For the survey data,",
        "From the graph,",
    ]
    base_snapshot = list(rows)
    for row in base_snapshot:
        if len(rows) >= target * 2:
            break
        prefix = prefixes[len(rows) % len(prefixes)]
        q = row["question"].rstrip("?").rstrip(".")
        if prefix == "Can you":
            variant = f"Can you {q[:1].lower() + q[1:]}?"
        else:
            variant = f"{prefix} {q[:1].lower() + q[1:]}?"
        _add(rows, seen, variant, topic=row["topic"], expected_route=row["expected_route"])

    rng.shuffle(rows)
    selected = rows[:target]
    for idx, row in enumerate(selected, start=1):
        row["id"] = f"EFFKGQA{idx:04d}"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 500-question KGQA efficiency set.")
    parser.add_argument("--out", default="evaluation/question_sets/true_demand_efficiency_500.json")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_questions(target=args.target, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
