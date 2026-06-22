from rdflib import Graph, Literal, OWL, RDF, URIRef, XSD

from visualization.extract_webvowl_ontology import SURVEY_NS, extract_ontology


def uri(local: str) -> URIRef:
    return URIRef(SURVEY_NS + local)


def test_extractor_distinguishes_object_and_datatype_properties(tmp_path):
    source = Graph()
    source.add((uri("ClassA"), RDF.type, OWL.Class))
    source.add((uri("ClassB"), RDF.type, OWL.Class))
    source.add((uri("a"), RDF.type, uri("ClassA")))
    source.add((uri("b"), RDF.type, uri("ClassB")))
    source.add((uri("a"), uri("relatedTo"), uri("b")))
    source.add((uri("a"), uri("active"), Literal(True)))
    source_path = tmp_path / "source.ttl"
    output_path = tmp_path / "ontology.ttl"
    source.serialize(source_path, format="turtle")

    extract_ontology(source_path, output_path)
    output = Graph()
    output.parse(output_path, format="turtle")

    assert (uri("relatedTo"), RDF.type, OWL.ObjectProperty) in output
    assert (uri("active"), RDF.type, OWL.DatatypeProperty) in output
    assert (uri("active"), URIRef("http://www.w3.org/2000/01/rdf-schema#range"), XSD.boolean) in output
