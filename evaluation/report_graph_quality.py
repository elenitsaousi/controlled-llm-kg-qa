#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rdflib import Graph

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kg.entity_profiles import build_entity_profiles, summarize_graph_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Report graph/entity data quality.")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--out", default="results/graph_quality_report.json")
    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.graph, format="turtle")
    report = summarize_graph_quality(graph, build_entity_profiles(graph))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("===== GRAPH QUALITY REPORT =====")
    print(f"Graph: {args.graph}")
    print(f"Triples: {report['triple_count']}")
    print(f"Entities profiled: {report['entity_count']}")
    print("Quality flags:")
    for flag, count in report["quality_flag_counts"].items():
        print(f"  {flag}: {count}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
