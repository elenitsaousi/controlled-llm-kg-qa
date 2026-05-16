#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rdflib import BNode, Graph, URIRef


def build_stats(graph_path: str) -> dict:
    path = Path(graph_path)
    graph = Graph()
    graph.parse(path, format="turtle")
    resources = set()
    subjects = set()
    for subject, _predicate, obj in graph:
        if isinstance(subject, (URIRef, BNode)):
            subjects.add(subject)
            resources.add(subject)
        if isinstance(obj, (URIRef, BNode)):
            resources.add(obj)
    stat = path.stat()
    return {
        "graph_path": str(path.resolve()),
        "graph_size_bytes": int(stat.st_size),
        "graph_mtime_ns": int(stat.st_mtime_ns),
        "triples": len(graph),
        "resource_nodes": len(resources),
        "subject_entities": len(subjects),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached graph statistics for the UI overview page.")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--out", default="data/infineon/graph_stats.json")
    args = parser.parse_args()
    stats = build_stats(args.graph)
    Path(args.out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("===== GRAPH STATS =====")
    print(f"Graph: {args.graph}")
    print(f"Triples: {stats['triples']:,}")
    print(f"Resource nodes / entities: {stats['resource_nodes']:,}")
    print(f"Subject entities: {stats['subject_entities']:,}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
