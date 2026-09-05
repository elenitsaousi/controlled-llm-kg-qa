from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef


SURVEY_NS = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"


def _local(term: object) -> str:
    value = str(term)
    return value[len(SURVEY_NS) :] if value.startswith(SURVEY_NS) else value


def _survey_term(term: object) -> bool:
    return isinstance(term, URIRef) and str(term).startswith(SURVEY_NS)


def analyze_graph(graph: Graph) -> dict[str, Any]:
    usage: dict[URIRef, Counter[str]] = defaultdict(Counter)
    for _subject, predicate, obj in graph:
        if _survey_term(predicate):
            usage[predicate]["literal" if isinstance(obj, Literal) else "resource"] += 1

    object_properties = set(graph.subjects(RDF.type, OWL.ObjectProperty))
    datatype_properties = set(graph.subjects(RDF.type, OWL.DatatypeProperty))
    errors: list[dict[str, Any]] = []

    for predicate, counts in sorted(usage.items(), key=lambda item: str(item[0])):
        is_object = predicate in object_properties
        is_datatype = predicate in datatype_properties
        detail = {
            "property": _local(predicate),
            "literal_objects": counts["literal"],
            "resource_objects": counts["resource"],
        }
        if counts["literal"] and counts["resource"]:
            errors.append({"kind": "mixed_object_kinds", **detail})
        if is_datatype and counts["resource"]:
            errors.append({"kind": "datatype_property_with_resource", **detail})
        if is_object and counts["literal"]:
            errors.append({"kind": "object_property_with_literal", **detail})
        if is_object and is_datatype:
            errors.append({"kind": "conflicting_property_declarations", **detail})
        if not is_object and not is_datatype:
            errors.append({"kind": "unclassified_property", **detail})

    schema_classes = {
        term
        for term in (
            set(graph.subjects(RDF.type, OWL.Class))
            | set(graph.subjects(RDF.type, RDFS.Class))
            | set(graph.subjects(RDFS.subClassOf, None))
            | set(graph.objects(None, RDFS.subClassOf))
        )
        if _survey_term(term)
    }
    individuals = {
        subject
        for subject, _predicate, class_ in graph.triples((None, RDF.type, None))
        if _survey_term(subject)
        and _survey_term(class_)
        and class_ not in {OWL.Class, RDFS.Class, OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property}
    }
    punned = sorted(schema_classes & individuals, key=str)
    warnings = [
        {
            "kind": "class_individual_punning",
            "term": _local(term),
            "types": sorted(_local(value) for value in graph.objects(term, RDF.type)),
        }
        for term in punned
    ]

    return {
        "summary": {
            "triples": len(graph),
            "classes": len(schema_classes),
            "typed_individuals": len(individuals),
            "properties": len(usage),
            "object_properties": sum(predicate in object_properties for predicate in usage),
            "datatype_properties": sum(predicate in datatype_properties for predicate in usage),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "errors": errors,
        "warnings": warnings,
        "properties": [
            {
                "name": _local(predicate),
                "type": (
                    "object"
                    if predicate in object_properties and predicate not in datatype_properties
                    else "datatype"
                    if predicate in datatype_properties and predicate not in object_properties
                    else "conflicting"
                    if predicate in object_properties and predicate in datatype_properties
                    else "unclassified"
                ),
                "literal_objects": counts["literal"],
                "resource_objects": counts["resource"],
            }
            for predicate, counts in sorted(usage.items(), key=lambda item: str(item[0]))
        ],
        "classes": sorted(_local(term) for term in schema_classes),
    }


def _markdown(report: dict[str, Any], graph_path: Path) -> str:
    summary = report["summary"]
    lines = [
        "# Ontology Semantic Validation",
        "",
        f"Graph: `{graph_path}`",
        "",
        "| Metric | Count |",
        "|---|---:|",
        *[f"| {key.replace('_', ' ').title()} | {value} |" for key, value in summary.items()],
        "",
        "## Errors",
        "",
    ]
    if report["errors"]:
        lines.extend(f"- `{item['kind']}`: `{item['property']}`" for item in report["errors"])
    else:
        lines.append("No property-semantics errors detected.")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(
            f"- `{item['kind']}`: `{item['term']}` ({', '.join(item['types'])})"
            for item in report["warnings"]
        )
    else:
        lines.append("No warnings detected.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate True Demand ontology classes and property object kinds.")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--out-json", default="results/ontology_semantic_validation.json")
    parser.add_argument("--out-md", default="results/ontology_semantic_validation.md")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    graph = Graph()
    graph.parse(graph_path, format="turtle")
    report = analyze_graph(graph)

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_markdown(report, graph_path), encoding="utf-8")

    summary = report["summary"]
    print("===== ONTOLOGY SEMANTIC VALIDATION =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
