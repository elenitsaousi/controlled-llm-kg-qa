#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
except ImportError:  # pragma: no cover - production environment has rdflib installed.
    Graph = None
    SPARQLStore = None

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


def _make_graph(graph_path: str, fuseki_query_url: str, *, progress: bool = False) -> Graph:
    if Graph is None:
        raise RuntimeError("rdflib is required to validate gold datasets.")
    if fuseki_query_url:
        if SPARQLStore is None:
            raise RuntimeError("rdflib SPARQLStore is required for Fuseki validation.")
        if progress:
            print(f"Using Fuseki endpoint: {fuseki_query_url}", flush=True)
        return Graph(store=SPARQLStore(fuseki_query_url))
    graph = Graph()
    if progress:
        print(f"Loading local graph: {graph_path}", flush=True)
    graph.parse(graph_path, format="turtle")
    return graph


def validate_dataset(
    dataset_path: str,
    graph_path: str,
    *,
    fuseki_query_url: str = "",
    progress: bool = False,
) -> Dict[str, object]:
    graph = _make_graph(graph_path, fuseki_query_url, progress=progress)
    rows: List[Dict[str, object]] = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    cases = []
    valid = 0
    non_empty = 0
    empty = 0
    invalid = 0
    for index, row in enumerate(rows, start=1):
        query = _ensure_prefixes(_strip_comments(str(row.get("query") or "")))
        case = {
            "id": row.get("id"),
            "question": row.get("question"),
            "expected_family": row.get("family") or row.get("capability") or row.get("category"),
            "expected_answer_shape": row.get("answer_shape") or row.get("aggregation"),
        }
        try:
            results = list(graph.query(query))
            case["valid"] = True
            case["row_count"] = len(results)
            case["needs_manual_review"] = len(results) == 0
            case["review_reason"] = "empty_result" if len(results) == 0 else ""
            valid += 1
            if results:
                non_empty += 1
            else:
                empty += 1
        except Exception as exc:
            case["valid"] = False
            case["row_count"] = None
            case["error"] = str(exc)
            case["needs_manual_review"] = True
            case["review_reason"] = "query_error"
            invalid += 1
        cases.append(case)
        if progress:
            print(f"[{index}/{len(rows)}] {case['id']} valid={case['valid']} rows={case['row_count']}", flush=True)
    total = len(cases)
    return {
        "summary": {
            "total": total,
            "valid": valid,
            "non_empty": non_empty,
            "empty": empty,
            "invalid": invalid,
            "valid_rate": (valid / total) if total else 0.0,
            "non_empty_rate": (non_empty / total) if total else 0.0,
            "manual_review_needed": empty + invalid,
        },
        "cases": cases,
    }


def write_review_csv(report: Dict[str, object], out_csv: str) -> None:
    cases = report.get("cases") if isinstance(report.get("cases"), list) else []
    fieldnames = [
        "id",
        "question",
        "valid",
        "row_count",
        "needs_manual_review",
        "review_reason",
        "expected_family",
        "expected_answer_shape",
        "error",
        "human_gold_valid",
        "human_notes",
    ]
    with Path(out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            if not isinstance(case, dict):
                continue
            writer.writerow({key: case.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate gold KGQA queries against the RDF graph.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--fuseki-query-url", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    report = validate_dataset(
        args.dataset,
        args.graph,
        fuseki_query_url=args.fuseki_query_url,
        progress=args.progress,
    )
    summary = report["summary"]
    print("===== GOLD DATASET VALIDATION =====")
    print(f"Dataset: {args.dataset}")
    if args.fuseki_query_url:
        print(f"Fuseki: {args.fuseki_query_url}")
    else:
        print(f"Graph: {args.graph}")
    print(f"Total: {summary['total']}")
    print(f"Valid queries: {summary['valid']} ({summary['valid_rate']:.3f})")
    print(f"Non-empty queries: {summary['non_empty']} ({summary['non_empty_rate']:.3f})")
    print(f"Empty queries: {summary['empty']}")
    print(f"Invalid queries: {summary['invalid']}")
    print(f"Manual review needed: {summary['manual_review_needed']}")
    for case in report["cases"]:
        print(f"  {case['id']}: valid={case['valid']} rows={case['row_count']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Output: {args.out}")
    if args.out_csv:
        write_review_csv(report, args.out_csv)
        print(f"CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
