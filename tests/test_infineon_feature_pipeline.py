import json

from ranking.feature_extraction import (
    extract_features,
    extract_query_labels,
    extract_query_relations,
    extract_triples,
)
from ranking.np_tfidf_ranker import load_training_data


SAMPLE_QUERY = (
    "SELECT ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE { "
    "?d a survey:DemandForRegion ; "
    "  survey:hasSurveyOrigin ?o ; "
    "  survey:inRegion ?r ; "
    "  survey:totalDemand ?unitsSold . "
    "?o a survey:OEM_Survey . "
    "?r a survey:Region ; survey:regionName ?regionName . "
    "} GROUP BY ?regionName ORDER BY DESC(?totalDemand)"
)


def test_extract_triples_handles_semicolon_chains():
    triples = extract_triples(SAMPLE_QUERY)
    assert len(triples) >= 6
    assert ("?d", "survey:hasSurveyOrigin", "?o") in triples
    assert ("?d", "survey:inRegion", "?r") in triples


def test_extract_query_labels_relations_for_survey_prefix():
    labels = extract_query_labels(SAMPLE_QUERY)
    rels = extract_query_relations(SAMPLE_QUERY)
    assert "DemandForRegion" in labels
    assert "OEM_Survey" in labels
    assert "hasSurveyOrigin" in rels
    assert "totalDemand" in rels


def test_extract_features_nonzero_infineon_signal():
    schema = json.load(open("data/infineon/schema.json", "r", encoding="utf-8"))
    question = "What is the regional demand for OEM?"
    feats = extract_features(question, SAMPLE_QUERY, schema)
    assert feats["entity_coverage"] > 0.0
    assert feats["relation_coverage"] > 0.0
    assert feats["invalid_predicate_count"] == 0.0


def test_load_training_data_excludes_gold_by_default(tmp_path):
    payload = {
        "Q1": [
            {
                "query_id": "Q1_GOLD",
                "question": "q",
                "query": "SELECT * WHERE { ?s ?p ?o }",
                "is_correct": 1,
                "is_valid": 1,
                "features": {},
                "ambiguity_label": "mid",
                "source": "gold",
            },
            {
                "query_id": "Q1_R0_C0",
                "question": "q",
                "query": "SELECT ?s WHERE { ?s a survey:Company }",
                "is_correct": 0,
                "is_valid": 1,
                "features": {},
                "ambiguity_label": "mid",
                "source": "llm",
            },
        ]
    }
    p = tmp_path / "training.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    data_default = load_training_data(str(p))
    assert "Q1" in data_default
    assert len(data_default["Q1"].candidates) == 1
    assert data_default["Q1"].candidates[0].source == "llm"

    data_with_gold = load_training_data(str(p), include_gold=True)
    assert len(data_with_gold["Q1"].candidates) == 2
