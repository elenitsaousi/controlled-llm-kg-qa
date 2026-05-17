from rdflib import Graph, Literal, RDF, URIRef

from kg.entity_linking import SURVEY_NS, build_entity_alias_index
from kg.entity_profiles import build_entity_profiles, is_placeholder_label, summarize_graph_quality
from kg.schema import load_schema
from llm.prompts import build_candidate_prompt


def test_placeholder_labels_keep_structural_entity_profile() -> None:
    graph = Graph()
    chip = URIRef(SURVEY_NS + "Chip_123")
    graph.add((chip, RDF.type, URIRef(SURVEY_NS + "Chip")))
    graph.add((chip, URIRef(SURVEY_NS + "companyName"), Literal("N/A")))
    graph.add((chip, URIRef(SURVEY_NS + "hasComponentTypeSplit"), URIRef(SURVEY_NS + "EV")))

    profile = build_entity_profiles(graph)[str(chip)]

    assert profile.canonical_label == "Chip 123"
    assert profile.types == ["Chip"]
    assert "placeholder_display_label" in profile.quality_flags
    assert "hasComponentTypeSplit" in profile.predicates


def test_alias_index_uses_typed_uri_local_name_when_label_is_bad() -> None:
    graph = Graph()
    chip = URIRef(SURVEY_NS + "Chip_123")
    graph.add((chip, RDF.type, URIRef(SURVEY_NS + "Chip")))
    graph.add((chip, URIRef(SURVEY_NS + "companyName"), Literal("#")))

    index = build_entity_alias_index(graph)

    assert index.best_label("chip123") == "Chip_123"


def test_graph_quality_summary_counts_missing_and_placeholder_labels() -> None:
    graph = Graph()
    named = URIRef(SURVEY_NS + "Named")
    placeholder = URIRef(SURVEY_NS + "Placeholder")
    unlabelled = URIRef(SURVEY_NS + "Unlabelled")
    for entity in (named, placeholder, unlabelled):
        graph.add((entity, RDF.type, URIRef(SURVEY_NS + "Chip")))
    graph.add((named, URIRef(SURVEY_NS + "companyName"), Literal("Named chip")))
    graph.add((placeholder, URIRef(SURVEY_NS + "companyName"), Literal("N/A")))

    report = summarize_graph_quality(graph)

    assert report["quality_flag_counts"]["placeholder_display_label"] == 1
    assert report["quality_flag_counts"]["missing_display_label"] == 1
    assert is_placeholder_label("#") is True


def test_candidate_prompt_includes_structural_profile_evidence() -> None:
    graph = Graph()
    chip = URIRef(SURVEY_NS + "Chip_123")
    graph.add((chip, RDF.type, URIRef(SURVEY_NS + "Chip")))
    profile = build_entity_profiles(graph)[str(chip)]

    prompt = build_candidate_prompt(
        "Show Chip 123",
        load_schema("data/infineon/schema.json"),
        entity_profiles=[profile],
    )

    assert "ENTITY STRUCTURAL PROFILES" in prompt
    assert "types=Chip" in prompt
