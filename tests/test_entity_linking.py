from rdflib import Graph, Literal, URIRef

from kg.entity_linking import (
    SURVEY_NS,
    build_entity_alias_index,
    canonicalize_question_with_index,
)


def _build_test_graph() -> Graph:
    g = Graph()
    product = URIRef(SURVEY_NS + "Product_SP123")
    region = URIRef(SURVEY_NS + "Region_Europe")
    typo_region = URIRef(SURVEY_NS + "Region_Americas")

    g.add((product, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), Literal("SP123")))
    g.add((region, URIRef(SURVEY_NS + "regionName"), Literal("Europe")))
    g.add((typo_region, URIRef(SURVEY_NS + "regionName"), Literal("Americas")))
    return g


def test_entity_linking_normalizes_sp_hash_variants() -> None:
    g = _build_test_graph()
    index = build_entity_alias_index(g)

    result = canonicalize_question_with_index(
        "Show demand for SP#123 in Europe",
        index=index,
        max_matches=5,
    )

    assert result.changed is True
    assert result.effective_question == "Show demand for SP123 in Europe"
    assert any(
        m.get("mention") == "SP#123" and m.get("canonical") == "SP123"
        for m in result.mappings
    )


def test_entity_linking_handles_small_typos_with_fuzzy_match() -> None:
    g = _build_test_graph()
    index = build_entity_alias_index(g)

    result = canonicalize_question_with_index(
        "What is total demand in Americass?",
        index=index,
        max_matches=5,
    )

    assert "Americas" in result.effective_question
    assert any(m.get("canonical") == "Americas" for m in result.mappings)
