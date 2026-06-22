from rdflib import Graph, Literal, OWL, RDF, URIRef

from validation.validate_ontology_semantics import SURVEY_NS, analyze_graph


def uri(local: str) -> URIRef:
    return URIRef(SURVEY_NS + local)


def test_valid_object_and_datatype_properties_pass():
    graph = Graph()
    graph.add((uri("relatedTo"), RDF.type, OWL.ObjectProperty))
    graph.add((uri("label"), RDF.type, OWL.DatatypeProperty))
    graph.add((uri("a"), uri("relatedTo"), uri("b")))
    graph.add((uri("a"), uri("label"), Literal("A")))

    assert analyze_graph(graph)["errors"] == []


def test_mixed_property_is_rejected():
    graph = Graph()
    graph.add((uri("mixed"), RDF.type, OWL.ObjectProperty))
    graph.add((uri("a"), uri("mixed"), uri("b")))
    graph.add((uri("a"), uri("mixed"), Literal("b")))

    kinds = {item["kind"] for item in analyze_graph(graph)["errors"]}
    assert "mixed_object_kinds" in kinds
    assert "object_property_with_literal" in kinds


def test_unclassified_property_is_rejected():
    graph = Graph()
    graph.add((uri("a"), uri("unknown"), Literal(1)))

    assert analyze_graph(graph)["errors"][0]["kind"] == "unclassified_property"
