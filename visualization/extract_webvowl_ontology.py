from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote

from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef, XSD

SURVEY_NS = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"


def _short(term: URIRef) -> str:
    value = str(term)
    if value.startswith(SURVEY_NS):
        return value[len(SURVEY_NS) :]
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    if "/" in value:
        return value.rstrip("/").rsplit("/", 1)[-1]
    return value


def _human_label(term: URIRef) -> str:
    text = unquote(_short(term)).replace("_", " ").replace("-", " ")
    out = []
    previous_lower = False
    for ch in text:
        if ch.isupper() and previous_lower:
            out.append(" ")
        out.append(ch)
        previous_lower = ch.islower() or ch.isdigit()
    return " ".join("".join(out).split())


def _is_survey_uri(term) -> bool:
    return isinstance(term, URIRef) and str(term).startswith(SURVEY_NS)


def _is_datatype(term) -> bool:
    return isinstance(term, URIRef) and str(term).startswith(XSD_NS)


def _class_candidates(graph: Graph) -> set[URIRef]:
    classes: set[URIRef] = set()
    classes.update(x for x in graph.subjects(RDF.type, RDFS.Class) if _is_survey_uri(x))
    classes.update(x for x in graph.subjects(RDF.type, OWL.Class) if _is_survey_uri(x))
    classes.update(x for x in graph.objects(None, RDFS.subClassOf) if _is_survey_uri(x))
    classes.update(x for x in graph.subjects(RDFS.subClassOf, None) if _is_survey_uri(x))

    # The source graph often models concepts as both class and instance.
    # Add frequently used rdf:type objects as class nodes, but keep this at the
    # type level rather than importing every individual instance.
    type_counts = Counter(o for _s, _p, o in graph.triples((None, RDF.type, None)) if _is_survey_uri(o))
    for term, count in type_counts.items():
        if count >= 1:
            classes.add(term)
    return classes


def _predicate_candidates(graph: Graph) -> set[URIRef]:
    predicates = {p for _s, p, _o in graph if _is_survey_uri(p)}
    predicates.update(x for x in graph.subjects(RDF.type, OWL.ObjectProperty) if _is_survey_uri(x))
    predicates.update(x for x in graph.subjects(RDF.type, OWL.DatatypeProperty) if _is_survey_uri(x))
    predicates.discard(RDF.type)
    return predicates


def _node_types(graph: Graph) -> dict[URIRef, set[URIRef]]:
    out: dict[URIRef, set[URIRef]] = defaultdict(set)
    for subject, _p, klass in graph.triples((None, RDF.type, None)):
        if isinstance(subject, URIRef) and _is_survey_uri(klass):
            out[subject].add(klass)
    return out


def _best_type(types: Iterable[URIRef], classes: set[URIRef]) -> URIRef | None:
    for klass in sorted(set(types), key=lambda x: _short(x)):
        if klass in classes:
            return klass
    return None


def extract_ontology(graph_path: Path, out_path: Path, rdfxml_out_path: Path | None = None) -> dict[str, int]:
    graph = Graph()
    graph.parse(graph_path)

    classes = _class_candidates(graph)
    predicates = _predicate_candidates(graph)
    node_types = _node_types(graph)

    explicit_domains: dict[URIRef, set[URIRef]] = defaultdict(set)
    explicit_ranges: dict[URIRef, set[URIRef]] = defaultdict(set)
    inferred_domains: dict[URIRef, Counter[URIRef]] = defaultdict(Counter)
    inferred_ranges: dict[URIRef, Counter[URIRef]] = defaultdict(Counter)
    literal_ranges: dict[URIRef, Counter[URIRef]] = defaultdict(Counter)

    for predicate, _p, domain in graph.triples((None, RDFS.domain, None)):
        if _is_survey_uri(predicate) and isinstance(domain, URIRef):
            explicit_domains[predicate].add(domain)
            if _is_survey_uri(domain):
                classes.add(domain)
    for predicate, _p, range_ in graph.triples((None, RDFS.range, None)):
        if _is_survey_uri(predicate) and isinstance(range_, URIRef):
            explicit_ranges[predicate].add(range_)
            if _is_survey_uri(range_):
                classes.add(range_)

    for subject, predicate, obj in graph:
        if predicate not in predicates:
            continue
        domain = _best_type(node_types.get(subject, set()), classes)
        if domain is not None:
            inferred_domains[predicate][domain] += 1
        if isinstance(obj, Literal):
            datatype = obj.datatype or (RDF.langString if obj.language else XSD.string)
            literal_ranges[predicate][URIRef(str(datatype))] += 1
        elif isinstance(obj, URIRef):
            range_ = _best_type(node_types.get(obj, set()), classes)
            if range_ is not None:
                inferred_ranges[predicate][range_] += 1

    out = Graph()
    out.bind("survey", SURVEY_NS)
    out.bind("owl", OWL)
    out.bind("rdfs", RDFS)
    out.bind("xsd", XSD_NS)
    ontology = URIRef(SURVEY_NS + "TrueDemandExtractedOntology")
    out.add((ontology, RDF.type, OWL.Ontology))
    out.add((ontology, RDFS.label, Literal("True Demand extracted ontology")))
    out.add(
        (
            ontology,
            RDFS.comment,
            Literal("Schema-level ontology extracted from the full True Demand RDF graph for WebVOWL visualization."),
        )
    )

    for klass in sorted(classes, key=_short):
        out.add((klass, RDF.type, OWL.Class))
        out.add((klass, RDFS.label, Literal(_human_label(klass))))

    for subject, _p, parent in graph.triples((None, RDFS.subClassOf, None)):
        if _is_survey_uri(subject) and isinstance(parent, URIRef) and parent in classes:
            out.add((subject, RDFS.subClassOf, parent))

    for predicate in sorted(predicates, key=_short):
        explicit_literal_ranges = {value for value in explicit_ranges.get(predicate, set()) if _is_datatype(value)}
        explicit_resource_ranges = {
            value for value in explicit_ranges.get(predicate, set()) if not _is_datatype(value)
        }
        has_literal_range = bool(literal_ranges.get(predicate) or explicit_literal_ranges)
        has_uri_range = bool(inferred_ranges.get(predicate) or explicit_resource_ranges)
        out.add((predicate, RDF.type, OWL.DatatypeProperty if has_literal_range and not has_uri_range else OWL.ObjectProperty))
        out.add((predicate, RDFS.label, Literal(_human_label(predicate))))

        domains = set(explicit_domains.get(predicate, set()))
        ranges = set(explicit_ranges.get(predicate, set()))
        if not domains and inferred_domains.get(predicate):
            domains.add(inferred_domains[predicate].most_common(1)[0][0])
        if not ranges and inferred_ranges.get(predicate):
            ranges.add(inferred_ranges[predicate].most_common(1)[0][0])
        if not ranges and literal_ranges.get(predicate):
            ranges.add(literal_ranges[predicate].most_common(1)[0][0])

        for domain in sorted(domains, key=_short):
            if domain in classes:
                out.add((predicate, RDFS.domain, domain))
        for range_ in sorted(ranges, key=lambda x: str(x)):
            if range_ in classes or _is_datatype(range_):
                out.add((predicate, RDFS.range, range_))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=str(out_path), format="turtle")
    if rdfxml_out_path is not None:
        rdfxml_out_path.parent.mkdir(parents=True, exist_ok=True)
        out.serialize(destination=str(rdfxml_out_path), format="xml")
    return {
        "source_triples": len(graph),
        "classes": len(classes),
        "predicates": len(predicates),
        "ontology_triples": len(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WebVOWL-ready ontology from the full True Demand RDF graph.")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--out", default="data/infineon/true_demand_ontology_extracted.ttl")
    parser.add_argument(
        "--rdfxml-out",
        default="data/infineon/true_demand_ontology_extracted.owl",
        help="Optional RDF/XML output for tools such as OWL2VOWL. Use an empty string to disable.",
    )
    args = parser.parse_args()
    rdfxml_out = Path(args.rdfxml_out) if args.rdfxml_out else None
    stats = extract_ontology(Path(args.graph), Path(args.out), rdfxml_out)
    print("===== TRUE DEMAND ONTOLOGY EXTRACTION =====")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"Output: {args.out}")
    if rdfxml_out is not None:
        print(f"RDF/XML output: {rdfxml_out}")


if __name__ == "__main__":
    main()
