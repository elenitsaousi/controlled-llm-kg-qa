from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from api.service import KGQAService


def test_table_uses_union_of_row_columns():
    table = KGQAService._table([{"region": "EMEA"}, {"region": "APAC", "value": 2}])
    assert table is not None
    assert [column["key"] for column in table["columns"]] == ["region", "value"]


def test_autocomplete_is_limited_to_answerable_capabilities(monkeypatch):
    service = KGQAService()
    monkeypatch.setattr(
        service,
        "answerable_capabilities",
        lambda: (
            {
                "family": "Future Demand",
                "templates": 1,
                "description": "Graph-supported future demand questions.",
                "dimensions": ["region"],
                "aggregations": ["AVG"],
                "examples": [],
            },
        ),
    )
    assert service.autocomplete("fu", "")[0]["label"] == "Future Demand"
    assert service.autocomplete("re", "Show Future Demand by")[0]["label"] == "Region"


def test_graph_payload_preserves_owl_node_types():
    survey = Namespace("https://example.test/survey/")
    metadata = Graph()
    metadata.add((survey.Demand, RDF.type, OWL.Class))
    metadata.add((survey.hasRegion, RDF.type, OWL.ObjectProperty))
    metadata.add((survey.totalDemand, RDF.type, OWL.DatatypeProperty))

    service = KGQAService()
    payload = service._triples_payload(
        metadata,
        [
            (survey.Demand, survey.hasRegion, survey.Region),
            (survey.Demand, survey.totalDemand, Literal(42)),
            (survey.hasRegion, RDF.type, OWL.ObjectProperty),
            (survey.totalDemand, RDFS.range, XSD.decimal),
        ],
    )
    types = {node["id"]: node["type"] for node in payload["nodes"]}

    assert types[str(survey.Demand)] == "Class"
    assert types[str(survey.hasRegion)] == "ObjectProperty"
    assert types[str(survey.totalDemand)] == "DatatypeProperty"
    assert types[str(XSD.decimal)] == "Datatype"
    assert "Literal" in types.values()
