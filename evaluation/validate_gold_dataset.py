#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from rdflib import Graph
except ImportError:  # pragma: no cover - production environment has rdflib installed.
    Graph = None

DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query


def _strip_comments(query: str) -> str:
    cleaned_lines = []
    for line in query.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


def validate_dataset(dataset_path: str, graph_path: str) -> Dict[str, object]:
    if Graph is None:
        raise RuntimeError("rdflib is required to validate gold datasets.")
    graph = Graph()
    graph.parse(graph_path, format="turtle")
    rows: List[Dict[str, object]] = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    cases = []
    valid = 0
    non_empty = 0
    for row in rows:
        query = _ensure_prefixes(_strip_comments(str(row.get("query") or "")))
        case = {"id": row.get("id"), "question": row.get("question")}
        try:
            results = list(graph.query(query))
            case["valid"] = True
            case["row_count"] = len(results)
            valid += 1
            non_empty += int(bool(results))
        except Exception as exc:
            case["valid"] = False
            case["row_count"] = None
            case["error"] = str(exc)
        cases.append(case)
    total = len(cases)
    return {
        "summary": {
            "total": total,
            "valid": valid,
            "non_empty": non_empty,
            "valid_rate": (valid / total) if total else 0.0,
            "non_empty_rate": (non_empty / total) if total else 0.0,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate gold KGQA queries against the RDF graph.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = validate_dataset(args.dataset, args.graph)
    summary = report["summary"]
    print("===== GOLD DATASET VALIDATION =====")
    print(f"Dataset: {args.dataset}")
    print(f"Total: {summary['total']}")
    print(f"Valid queries: {summary['valid']} ({summary['valid_rate']:.3f})")
    print(f"Non-empty queries: {summary['non_empty']} ({summary['non_empty_rate']:.3f})")
    for case in report["cases"]:
        print(f"  {case['id']}: valid={case['valid']} rows={case['row_count']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
