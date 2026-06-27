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


def test_api_returns_digital_reference_definition_without_sparql(monkeypatch, tmp_path):
    dr_path = tmp_path / "DigitalReference.ttl"
    dr_path.write_text(
        """
@prefix dr: <http://www.w3id.org/ecsel-dr#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
dr:Product a owl:Class ;
    rdfs:label "Product"@en ;
    rdfs:comment "A product is any tangible output or service intended for delivery to a customer."@en .
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRUE_DEMAND_DR_ONTOLOGY_PATH", str(dr_path))
    service = KGQAService()

    response = service.ask("What is Product?")

    assert response["decision"] == "definition"
    assert "tangible output or service" in response["answer"]
    assert response["sparql"] == ""
    assert response["diagnostics"]["source"] == "digital_reference_ontology"


def test_api_returns_deterministic_advisory_answer(monkeypatch):
    service = KGQAService()

    def fake_execute(query, max_rows=200):
        assert "DemandForRegion" in query
        return [
            {"regionName": "Europe", "avgPercentageChange": "9.5"},
            {"regionName": "Japan", "avgPercentageChange": "4.0"},
        ], False, ""

    monkeypatch.setattr(service, "execute", fake_execute)

    response = service.ask("Based on future demand data, which region should be monitored more closely?")

    assert response["decision"] == "advisory"
    assert "Europe" in response["answer"]
    assert "not an autonomous business decision" in response["answer"]
    assert response["diagnostics"]["template"] == "future_demand_region_focus"
