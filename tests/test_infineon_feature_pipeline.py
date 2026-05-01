import json

from rdflib import Graph, URIRef
from rdflib.namespace import RDF

import pipeline.qa as qa_pipeline
from kg.entity_linking import SURVEY_NS
from ranking.feature_extraction import (
    extract_features,
    extract_query_labels,
    extract_query_plan,
    extract_query_relations,
    extract_triples,
)
from ranking.np_tfidf_ranker import (
    QueryPlanPredictor,
    evaluate_query_plan_predictor,
    load_training_data,
    train_query_plan_predictor,
)
from llm.prompts import build_candidate_prompt
from kg.schema import load_schema
from pipeline.qa import _rerank_with_semantic_coverage
from validation.semantic import semantic_coverage_report
from llm.candidate_generation import _template_candidate_queries


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


def test_extract_query_plan_labels_for_regional_demand():
    schema = json.load(open("data/infineon/schema.json", "r", encoding="utf-8"))
    plan = extract_query_plan(SAMPLE_QUERY, schema)

    assert "DemandForRegion" in plan["classes"]
    assert "Region" in plan["classes"]
    assert "OEM_Survey" in plan["survey_origins"]
    assert "totalDemand" in plan["predicates"]
    assert "SUM" in plan["aggregations"]
    assert "regionName" in plan["group_by_predicates"]
    assert "class:DemandForRegion" in plan["labels"]
    assert "predicate:totalDemand" in plan["labels"]
    assert "survey:OEM_Survey" in plan["labels"]
    assert "aggregation:SUM" in plan["labels"]
    assert "query_type:grouped" in plan["labels"]


def test_extract_query_plan_detects_filter_survey_origins():
    schema = json.load(open("data/infineon/schema.json", "r", encoding="utf-8"))
    query = (
        "SELECT ?regionName ?originType (SUM(?unitsSold) AS ?totalDemand) WHERE { "
        "?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?origin ; "
        "survey:inRegion ?region ; survey:totalDemand ?unitsSold . "
        "?origin a ?originType . "
        "FILTER(?originType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey)) "
        "?region a survey:Region ; survey:regionName ?regionName . "
        "} GROUP BY ?regionName ?originType"
    )
    plan = extract_query_plan(query, schema)

    assert plan["survey_origins"] == [
        "OEM_Survey",
        "Semiconductor_Survey",
        "Tier1_Survey",
    ]
    assert "query_type:filtered" in plan["labels"]
    assert "survey:Semiconductor_Survey" in plan["labels"]


def test_query_plan_labels_are_training_row_friendly():
    schema = json.load(open("data/infineon/schema.json", "r", encoding="utf-8"))
    plan = extract_query_plan(SAMPLE_QUERY, schema)
    row = {
        "query": SAMPLE_QUERY,
        "query_plan": plan,
        "query_plan_labels": list(plan.get("labels", [])),
    }

    assert isinstance(row["query_plan"], dict)
    assert isinstance(row["query_plan_labels"], list)
    assert "class:DemandForRegion" in row["query_plan_labels"]
    assert "predicate:totalDemand" in row["query_plan_labels"]


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


def test_semantic_coverage_detects_missing_requested_survey_origins():
    question = "Compare total demand across Tier1, OEM and Semiconductor by region."
    oem_only = (
        "SELECT ?regionName (SUM(?unitsSold) AS ?OEMDemand) WHERE { "
        "?demandForRegion a survey:DemandForRegion ; "
        "survey:hasSurveyOrigin ?origin ; "
        "survey:inRegion ?region ; "
        "survey:totalDemand ?unitsSold . "
        "?origin a survey:OEM_Survey . "
        "?region a survey:Region ; survey:regionName ?regionName . "
        "} GROUP BY ?regionName"
    )

    report = semantic_coverage_report(question, oem_only)
    assert "Tier1_Survey" in report["missing"]
    assert "Semiconductor_Survey" in report["missing"]
    assert report["coverage_score"] < 1.0


def test_semantic_coverage_rerank_prefers_more_complete_query():
    question = "Compare total demand across Tier1, OEM and Semiconductor by region."
    oem_only = (
        "SELECT ?regionName (SUM(?unitsSold) AS ?OEMDemand) WHERE { "
        "?demandForRegion a survey:DemandForRegion ; "
        "survey:hasSurveyOrigin ?origin ; "
        "survey:inRegion ?region ; "
        "survey:totalDemand ?unitsSold . "
        "?origin a survey:OEM_Survey . "
        "?region a survey:Region ; survey:regionName ?regionName . "
        "} GROUP BY ?regionName"
    )
    complete = (
        "SELECT ?regionName ?originType (SUM(?unitsSold) AS ?totalDemand) WHERE { "
        "?demandForRegion a survey:DemandForRegion ; "
        "survey:hasSurveyOrigin ?origin ; "
        "survey:inRegion ?region ; "
        "survey:totalDemand ?unitsSold . "
        "?origin a ?originType . "
        "FILTER(?originType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey)) "
        "?region a survey:Region ; survey:regionName ?regionName . "
        "} GROUP BY ?regionName ?originType"
    )

    ranked = _rerank_with_semantic_coverage(
        question,
        [
            {"query": oem_only, "score": 0.9},
            {"query": complete, "score": 0.1},
        ],
    )
    assert ranked[0]["query"] == complete
    assert ranked[0]["coverage_score"] == 1.0


def test_candidate_prompt_includes_required_question_concepts():
    schema = load_schema("data/infineon/schema.json")
    prompt = build_candidate_prompt(
        "Compare total demand across Tier1, OEM and Semiconductor by region.",
        schema,
        k=3,
    )
    assert "REQUIRED QUESTION CONCEPTS" in prompt
    assert "Tier1_Survey" in prompt
    assert "OEM_Survey" in prompt
    assert "Semiconductor_Survey" in prompt
    assert "DemandForRegion" in prompt


def test_candidate_prompt_includes_ml_query_plan_labels():
    schema = load_schema("data/infineon/schema.json")
    prompt = build_candidate_prompt(
        "Compare total demand across Tier1, OEM and Semiconductor by region.",
        schema,
        k=3,
        predicted_query_plan_labels=[
            "class:DemandForRegion",
            "predicate:totalDemand",
            "survey:OEM_Survey",
            "aggregation:SUM",
            "query_type:grouped",
        ],
    )

    assert "ML PREDICTED QUERY PLAN LABELS" in prompt
    assert "class:DemandForRegion" in prompt
    assert "predicate:totalDemand" in prompt
    assert "query_type:grouped" in prompt


def test_query_plan_predictor_trains_and_roundtrips(tmp_path):
    rows = [
        {
            "id": "q1",
            "question": "What is total demand by region?",
            "query": SAMPLE_QUERY,
            "labels": [
                "class:DemandForRegion",
                "predicate:totalDemand",
                "aggregation:SUM",
                "query_type:grouped",
            ],
        },
        {
            "id": "q2",
            "question": "Show regional demand for OEM.",
            "query": SAMPLE_QUERY,
            "labels": [
                "class:DemandForRegion",
                "predicate:totalDemand",
                "survey:OEM_Survey",
                "aggregation:SUM",
                "query_type:grouped",
            ],
        },
        {
            "id": "q3",
            "question": "How many companies report shortage?",
            "query": "SELECT ?status (COUNT(?Company) AS ?Count) WHERE { ?Company a survey:Company ; survey:reportsShortage ?Shortage . } GROUP BY ?status",
            "labels": [
                "class:Company",
                "predicate:reportsShortage",
                "aggregation:COUNT",
                "query_type:grouped",
            ],
        },
    ]

    model = train_query_plan_predictor(
        rows,
        min_label_count=1,
        threshold=0.25,
        top_k=8,
        epochs=80,
    )
    predicted = model.predict_labels("Compare total demand by region.")
    assert "predicate:totalDemand" in predicted
    assert "aggregation:SUM" in predicted

    report = evaluate_query_plan_predictor(model, rows)
    assert report["questions"] == 3
    assert report["labels"] >= 6

    model_path = tmp_path / "query_plan_model.json"
    model.save(str(model_path), metadata={"test": True})
    loaded = QueryPlanPredictor.load(str(model_path))
    assert loaded.predict_labels("Compare total demand by region.")


def test_execution_aware_selection_skips_empty_valid_query(monkeypatch):
    g = Graph()
    company = URIRef(SURVEY_NS + "Company1")
    g.add((company, RDF.type, URIRef(SURVEY_NS + "Company")))
    monkeypatch.setattr(qa_pipeline, "_get_default_graph", lambda: g)

    empty_query = "SELECT ?x WHERE { ?x a survey:Region }"
    non_empty_query = "SELECT ?x WHERE { ?x a survey:Company }"

    selected, errors, rank = qa_pipeline._select_best_valid_query(
        [empty_query, non_empty_query]
    )

    assert selected == non_empty_query
    assert errors == []
    assert rank == 1


def test_execution_aware_selection_skips_unbound_projected_variable(monkeypatch):
    g = Graph()
    company = URIRef(SURVEY_NS + "Company1")
    g.add((company, RDF.type, URIRef(SURVEY_NS + "Company")))
    monkeypatch.setattr(qa_pipeline, "_get_default_graph", lambda: g)

    unbound_label_query = (
        "SELECT ?x ?label WHERE { "
        "?x a survey:Company . "
        "OPTIONAL { ?x survey:missingLabel ?label } "
        "}"
    )
    bound_query = (
        "SELECT ?x ?label WHERE { "
        "?x a survey:Company . "
        "BIND('company' AS ?label) "
        "}"
    )

    selected, errors, rank = qa_pipeline._select_best_valid_query(
        [unbound_label_query, bound_query]
    )

    assert selected == bound_query
    assert errors == []
    assert rank == 1


def test_template_candidates_cover_hard_infineon_intents():
    future_vehicle = _template_candidate_queries(
        "Show future demand percentage change by vehicle type and quarter."
    )
    assert future_vehicle
    assert "analyzesVehicleType" in future_vehicle[0]
    assert "forTimePeriod" in future_vehicle[0]

    order_cancel = _template_candidate_queries(
        "Show order cancellation responses by semiconductor technology category."
    )
    assert order_cancel
    assert "OrderCancellation" in order_cancel[0]
    assert "hasResponseType" in order_cancel[0]
