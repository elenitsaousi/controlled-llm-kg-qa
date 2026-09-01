import json
import hashlib
import re

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
from llm.answer_synthesis import synthesize_answer
from kg.schema import load_schema
from pipeline.qa import _intent_alignment_report, _rerank_with_semantic_coverage
from validation.semantic import (
    question_intent_report,
    rank_candidates_by_semantic_judge,
    semantic_coverage_report,
    semantic_judge_report,
)
from llm.candidate_generation import _template_candidate_queries
from evaluation.analyze_infineon_results import analyze_results, render_markdown
from ranking.clarification import build_clarification_payload, plan_signature
from pipeline.request_routing import build_domain_glossary, route_request


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

VAR_RE = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
SINGLE_QUOTE_STR_RE = re.compile(r"'[^']*'")
DOUBLE_QUOTE_STR_RE = re.compile(r'"[^"]*"')
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _query_family_signature(query: str) -> str:
    q = " ".join(query.strip().split())
    q = SINGLE_QUOTE_STR_RE.sub("'STR'", q)
    q = DOUBLE_QUOTE_STR_RE.sub('"STR"', q)
    q = NUMBER_RE.sub("NUM", q)
    q = VAR_RE.sub("?VAR", q)
    return "fam_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:16]


def test_extract_triples_handles_semicolon_chains():
    triples = extract_triples(SAMPLE_QUERY)
    assert len(triples) >= 6
    assert ("?d", "survey:hasSurveyOrigin", "?o") in triples
    assert ("?d", "survey:inRegion", "?r") in triples


def test_route_request_answers_definition_questions():
    route = route_request("What is the battery electric vehicle?")
    assert route["route"] == "definition"
    assert "battery electric vehicle" in route["answer"].lower()


def test_route_request_sends_bare_current_demand_questions_to_the_graph():
    assert route_request("What is current demand?")["route"] != "definition"
    assert route_request("Show current demand by region.")["route"] != "definition"


def test_route_request_still_defines_current_demand_when_explicitly_asked():
    assert route_request("What does current demand mean?")["route"] == "definition"
    assert route_request("Define current demand.")["route"] == "definition"


def test_route_request_clarifies_underspecified_entity_questions():
    route = route_request("What about BEV?")
    assert route["route"] == "clarification_needed"
    clarification = route["request_clarification"]
    assert clarification["needs_clarification"] is True
    assert len(clarification["options"]) >= 2


def test_route_request_abstains_out_of_domain_questions():
    route = route_request("What is the weather tomorrow?")
    assert route["route"] == "out_of_domain"


def test_route_request_keeps_kg_questions_on_existing_path():
    route = route_request("Return average future-demand change grouped by technology and quarter.")
    assert route["route"] == "kg_query"


def test_build_domain_glossary_includes_schema_terms():
    schema = load_schema("data/infineon/schema.json")
    glossary = build_domain_glossary(schema, alias_index=None)
    assert "ordercancellation" in glossary
    assert glossary["ordercancellation"]["kind"] == "class"


def test_route_request_answers_schema_term_definitions():
    schema = load_schema("data/infineon/schema.json")
    route = route_request("What is OrderCancellation?", schema=schema)
    assert route["route"] == "definition"
    assert "class used in the infineon knowledge graph" in route["answer"].lower()


def test_route_request_routes_unknown_definition_questions_to_general_definition():
    schema = load_schema("data/infineon/schema.json")
    route = route_request("What is a manufacturer?", schema=schema)
    assert route["route"] == "general_definition"
    assert route["term"] == "a manufacturer"


def test_route_request_uses_digital_reference_definition(monkeypatch, tmp_path):
    dr_path = tmp_path / "DigitalReference.ttl"
    dr_path.write_text(
        """
@prefix dr: <http://www.w3id.org/ecsel-dr#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
dr:Demand a owl:Class ;
    rdfs:label "Demand"@en ;
    rdfs:comment "Demand is a supply chain or production request for a product or service."@en .
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRUE_DEMAND_DR_ONTOLOGY_PATH", str(dr_path))

    route = route_request("What is Demand?")

    assert route["route"] == "definition"
    assert route["source"] == "digital_reference_ontology"
    assert "supply chain or production request" in route["answer"]


def test_route_request_does_not_use_dr_for_analytic_questions(monkeypatch, tmp_path):
    dr_path = tmp_path / "DigitalReference.ttl"
    dr_path.write_text(
        """
@prefix dr: <http://www.w3id.org/ecsel-dr#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
dr:Demand a owl:Class ;
    rdfs:label "Demand"@en ;
    rdfs:comment "Demand is a supply chain or production request."@en .
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRUE_DEMAND_DR_ONTOLOGY_PATH", str(dr_path))

    route = route_request("Has demand increased between these months?")

    assert route["route"] == "kg_query"


def test_answer_question_uses_general_definition_llm_path():
    class FakeClient:
        def generate_text(self, prompt):
            assert "What is a manufacturer?" in prompt
            return "A manufacturer is a company that makes goods."

    schema = load_schema("data/infineon/schema.json")
    result = qa_pipeline.answer_question(
        "What is a manufacturer?",
        schema,
        llm_client=FakeClient(),
    )
    assert result["policy"] == "general_definition"
    assert result["answer"] == "A manufacturer is a company that makes goods."


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


def test_inventory_trend_does_not_require_percentage_change():
    report = semantic_coverage_report(
        "How many inventory records are there for each component in the Tier1 inventory trend?",
        "SELECT ?component (COUNT(?entry) AS ?count) WHERE { "
        "?entry a survey:InventoryDevelopment_Tier1 ; "
        "survey:forComponent ?component ; "
        "survey:inventoryTrend ?trend . "
        "} GROUP BY ?component",
    )
    assert "percentageChange" not in report["required"]
    assert "Semiconductor_Survey" in report["missing"]
    assert report["coverage_score"] < 1.0


def test_semantic_judge_prefers_average_for_mean_question():
    question = "Return average future-demand change grouped by technology and quarter."
    sum_query = (
        "SELECT ?techLabel ?quarterLabel (SUM(?pct) AS ?totalFutureChange) WHERE { "
        "?entry a survey:FutureDemandAnalysis ; "
        "survey:analyzesTechnologyCategory ?tech ; "
        "survey:forTimePeriod ?quarter ; "
        "survey:percentageChange ?pct . "
        "} GROUP BY ?techLabel ?quarterLabel"
    )
    avg_query = sum_query.replace("SUM(?pct) AS ?totalFutureChange", "AVG(?pct) AS ?avgFutureChange")

    assert semantic_judge_report(question, avg_query)["score"] > semantic_judge_report(question, sum_query)["score"]


def test_plan_signature_distinguishes_grouped_summary_from_raw_values():
    question = "How do future-demand percentages differ across technologies over time?"
    raw_query = (
        "SELECT ?technologyCategory ?periodLabel ?percentage WHERE { "
        "?entry a survey:FutureDemandAnalysis ; "
        "survey:analyzesTechnologyCategory ?tech ; "
        "survey:forTimePeriod ?period ; "
        "survey:percentageChange ?percentage . "
        "} "
    )
    avg_query = raw_query.replace(
        "SELECT ?technologyCategory ?periodLabel ?percentage",
        "SELECT ?technologyCategory ?periodLabel (AVG(?percentage) AS ?avgPercentage)",
    ) + " GROUP BY ?technologyCategory ?periodLabel"

    raw_signature = plan_signature(question, raw_query)
    avg_signature = plan_signature(question, avg_query)

    assert raw_signature["aggregation"] == "NONE"
    assert raw_signature["answer_shape"] == "raw_values"
    assert avg_signature["aggregation"] == "AVG"
    assert avg_signature["answer_shape"] == "grouped_summary"


def test_build_clarification_payload_detects_structural_disagreement():
    question = "How do future-demand percentages differ across technologies over time?"
    raw_query = (
        "SELECT ?technologyCategory ?periodLabel ?percentage WHERE { "
        "?entry a survey:FutureDemandAnalysis ; "
        "survey:analyzesTechnologyCategory ?tech ; "
        "survey:forTimePeriod ?period ; "
        "survey:percentageChange ?percentage . "
        "} "
    )
    avg_query = raw_query.replace(
        "SELECT ?technologyCategory ?periodLabel ?percentage",
        "SELECT ?technologyCategory ?periodLabel (AVG(?percentage) AS ?avgPercentage)",
    ) + " GROUP BY ?technologyCategory ?periodLabel"
    sum_query = raw_query.replace(
        "SELECT ?technologyCategory ?periodLabel ?percentage",
        "SELECT ?technologyCategory ?periodLabel (SUM(?percentage) AS ?totalPercentage)",
    ) + " GROUP BY ?technologyCategory ?periodLabel"

    payload = build_clarification_payload(
        question,
        [{"query": avg_query}, {"query": raw_query}, {"query": sum_query}],
    )

    assert payload is not None
    assert payload["needs_clarification"] is True
    assert payload["plan_cluster_count"] == 3
    assert any(c["axis"] == "aggregation" for c in payload["conflicts"])
    assert len(payload["options"]) == 3


def test_build_clarification_payload_prefers_nonempty_runtime_plans():
    question = "Describe actual vehicle-sales observations."
    month_total_query = (
        "SELECT ?monthLabel (SUM(?unitsSold) AS ?totalUnits) WHERE { "
        "?obs a survey:VehicleSalesObservation ; "
        "survey:isActualData true ; "
        "survey:forTimePeriod ?month ; "
        "survey:unitsSold ?unitsSold . "
        "} GROUP BY ?monthLabel"
    )
    month_raw_query = (
        "SELECT ?monthLabel ?unitsSold WHERE { "
        "?obs a survey:VehicleSalesObservation ; "
        "survey:isActualData true ; "
        "survey:forTimePeriod ?month ; "
        "survey:unitsSold ?unitsSold . "
        "}"
    )
    empty_vehicle_query = (
        "SELECT ?vehicleType (SUM(?unitsSold) AS ?totalUnits) WHERE { "
        "?obs a survey:VehicleSalesObservation ; "
        "survey:isActualData true ; "
        "survey:forVehicleType ?vehicle ; "
        "survey:unitsSold ?unitsSold . "
        "} GROUP BY ?vehicleType"
    )

    payload = build_clarification_payload(
        question,
        [
            {"query": empty_vehicle_query, "execution_has_rows": False},
            {"query": month_total_query, "execution_has_rows": True},
            {"query": month_raw_query, "execution_has_rows": True},
        ],
    )

    assert payload is not None
    assert len(payload["options"]) == 2
    assert all("vehicle type" not in option["label"].lower() for option in payload["options"])


def test_build_clarification_payload_skips_explicit_average_question():
    question = "Return average future-demand change grouped by technology and quarter."
    avg_query = (
        "SELECT ?technologyCategory ?quarterLabel (AVG(?percentageChange) AS ?avgChange) WHERE { "
        "?entry a survey:FutureDemandAnalysis ; "
        "survey:analyzesTechnologyCategory ?tech ; "
        "survey:forTimePeriod ?quarter ; "
        "survey:percentageChange ?percentageChange . "
        "} GROUP BY ?technologyCategory ?quarterLabel"
    )
    sum_query = avg_query.replace(
        "AVG(?percentageChange) AS ?avgChange",
        "SUM(?percentageChange) AS ?totalChange",
    )

    payload = build_clarification_payload(
        question,
        [{"query": sum_query}, {"query": avg_query}],
    )

    assert payload is None


def test_build_clarification_payload_keeps_vague_summary_questions_ambiguous():
    question = "Describe autonomous development by vehicle category."
    avg_query = (
        "SELECT ?vehicleType (AVG(?percentage) AS ?avgPercentage) WHERE { "
        "?entry a survey:AutonomousDrivingDevelopment ; "
        "survey:hasVehicleType ?vehicle ; "
        "survey:hasPercentage ?percentage . "
        "} GROUP BY ?vehicleType"
    )
    raw_query = (
        "SELECT ?vehicleType ?percentage WHERE { "
        "?entry a survey:AutonomousDrivingDevelopment ; "
        "survey:hasVehicleType ?vehicle ; "
        "survey:hasPercentage ?percentage . "
        "}"
    )

    payload = build_clarification_payload(
        question,
        [{"query": avg_query}, {"query": raw_query}],
    )

    assert payload is not None
    assert payload["needs_clarification"] is True


def test_build_clarification_payload_skips_explicit_list_response_request():
    question = "List cancellation response types by technology category."
    raw_query = (
        "SELECT ?technologyCategory ?responseType WHERE { "
        "?entry a survey:OrderCancellation ; "
        "survey:forTechnologyCategory ?tech ; "
        "survey:hasResponseType ?responseType . "
        "}"
    )
    grouped_query = (
        "SELECT ?technologyCategory ?responseType (COUNT(?entry) AS ?count) WHERE { "
        "?entry a survey:OrderCancellation ; "
        "survey:forTechnologyCategory ?tech ; "
        "survey:hasResponseType ?responseType . "
        "} GROUP BY ?technologyCategory ?responseType"
    )

    payload = build_clarification_payload(
        question,
        [{"query": grouped_query}, {"query": raw_query}],
    )

    assert payload is None


def test_build_clarification_payload_skips_explicit_raw_value_request():
    question = "List actual sold units by time period."
    raw_query = (
        "SELECT ?timePeriod ?unitsSold WHERE { "
        "?obs a survey:VehicleSalesObservation ; "
        "survey:isActualData true ; "
        "survey:forTimePeriod ?timePeriod ; "
        "survey:unitsSold ?unitsSold . "
        "}"
    )
    total_query = (
        "SELECT ?timePeriod (SUM(?unitsSold) AS ?totalUnits) WHERE { "
        "?obs a survey:VehicleSalesObservation ; "
        "survey:isActualData true ; "
        "survey:forTimePeriod ?timePeriod ; "
        "survey:unitsSold ?unitsSold . "
        "} GROUP BY ?timePeriod"
    )

    payload = build_clarification_payload(
        question,
        [{"query": total_query}, {"query": raw_query}],
    )

    assert payload is None


def test_build_clarification_payload_skips_implicit_vs_explicit_all_origins():
    question = "Return shortage values grouped by survey origin."
    count_query = (
        "SELECT ?surveyType (COUNT(?entry) AS ?count) WHERE { "
        "?entry a survey:ShortageResponse ; "
        "survey:hasSurveyOrigin ?origin . "
        "} GROUP BY ?surveyType"
    )
    total_query = (
        "SELECT ?surveyType (SUM(?shortage) AS ?totalShortage) WHERE { "
        "VALUES (?origin ?surveyType) { "
        "(survey:OEM_Survey 'OEM') "
        "(survey:Tier1_Survey 'Tier1') "
        "(survey:Semiconductor_Survey 'Semiconductor') } "
        "?entry a survey:ShortageResponse ; "
        "survey:hasSurveyOrigin ?origin ; "
        "survey:shortageValue ?shortage . "
        "} GROUP BY ?surveyType"
    )

    payload = build_clarification_payload(
        question,
        [{"query": count_query}, {"query": total_query}],
    )

    assert payload is None


def test_question_intent_report_treats_inventory_trend_lookup_as_raw_values():
    intent = question_intent_report("Which component has decreasing inventory?")

    assert intent["answer_shape"] == "raw_values"


def test_build_clarification_payload_skips_unasked_time_for_explicit_raw_values():
    question = "List inventory trend values for each component."
    raw_query = (
        "SELECT ?component ?trend WHERE { "
        "?entry a survey:InventoryDevelopment ; "
        "survey:forComponent ?component ; "
        "survey:inventoryTrend ?trend . "
        "}"
    )
    timed_query = (
        "SELECT ?component ?period ?trend WHERE { "
        "?entry a survey:InventoryDevelopment ; "
        "survey:forComponent ?component ; "
        "survey:forTimePeriod ?period ; "
        "survey:inventoryTrend ?trend . "
        "}"
    )

    payload = build_clarification_payload(
        question,
        [{"query": timed_query}, {"query": raw_query}],
    )

    assert payload is None


def test_build_clarification_payload_skips_unasked_origin_for_explicit_raw_values():
    question = "List inventory trend values for each component."
    raw_query = (
        "SELECT ?component ?trend WHERE { "
        "?entry a survey:InventoryDevelopment ; "
        "survey:forComponent ?component ; "
        "survey:inventoryTrend ?trend . "
        "}"
    )
    origin_query = (
        "SELECT ?component ?surveyType ?trend WHERE { "
        "VALUES (?origin ?surveyType) { "
        "(survey:OEM_Survey 'OEM') "
        "(survey:Tier1_Survey 'Tier1') "
        "(survey:Semiconductor_Survey 'Semiconductor') } "
        "?entry a survey:InventoryDevelopment ; "
        "survey:forComponent ?component ; "
        "survey:hasSurveyOrigin ?origin ; "
        "survey:inventoryTrend ?trend . "
        "}"
    )

    payload = build_clarification_payload(
        question,
        [{"query": origin_query}, {"query": raw_query}],
    )

    assert payload is None


def test_question_intent_report_resolves_multiple_explicit_axes():
    intent = question_intent_report(
        "Return monthly totals for actual vehicle-sales observations."
    )

    assert intent["aggregation"] == "SUM"
    assert intent["time_dimension"] == "month"
    assert intent["answer_shape"] == "grouped_summary"
    assert "month" in intent["dimensions"]
    assert "actual" in intent["filters"]


def test_question_intent_report_resolves_bl_comparison_shape():
    intent = question_intent_report(
        "Compare BL1 and BL2 current-demand changes for Tier1 Automotive."
    )

    assert intent["baseline_comparison"] is True
    assert intent["answer_shape"] == "baseline_comparison"


def test_question_intent_report_resolves_most_common_as_top_intent():
    intent = question_intent_report(
        "Which cancellation response is most common by technology category?"
    )

    assert intent["aggregation"] == "MAX_OR_TOP"


def test_build_clarification_payload_skips_explicit_dimension_and_time_conflicts():
    question = "Return average future-demand change grouped by technology and quarter."
    quarter_query = (
        "SELECT ?technologyCategory ?quarterLabel (AVG(?percentageChange) AS ?avgChange) WHERE { "
        "?entry a survey:FutureDemandAnalysis ; "
        "survey:analyzesTechnologyCategory ?tech ; "
        "survey:forTimePeriod ?quarter ; "
        "survey:percentageChange ?percentageChange . "
        "} GROUP BY ?technologyCategory ?quarterLabel"
    )
    period_query = quarter_query.replace("?quarterLabel", "?timePeriod")

    payload = build_clarification_payload(
        question,
        [{"query": period_query}, {"query": quarter_query}],
    )

    assert payload is None


def test_build_clarification_payload_skips_explicit_bl_comparison_conflicts():
    question = "Compare BL1 and BL2 current-demand changes for Tier1 Automotive."
    total_query = (
        "SELECT ?baseline (SUM(?pct) AS ?totalChange) WHERE { "
        "?entry a survey:CurrentDemandAnalysis ; survey:baselineType ?baseline ; "
        "survey:percentageChange ?pct . FILTER(?baseline IN ('BL1', 'BL2')) "
        "} GROUP BY ?baseline"
    )
    avg_query = total_query.replace("SUM(?pct) AS ?totalChange", "AVG(?pct) AS ?avgChange")

    payload = build_clarification_payload(
        question,
        [{"query": total_query}, {"query": avg_query}],
    )

    assert payload is None


def test_semantic_judge_prefers_literal_survey_bucket_labels():
    question = "Break down total demand by region and by survey group: Tier1, OEM, and Semiconductor."
    uri_query = (
        "SELECT ?surveyType ?regionName (SUM(?units) AS ?totalDemand) WHERE { "
        "?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; "
        "survey:inRegion ?r ; survey:totalDemand ?units . "
        "?o a ?surveyType . "
        "?r survey:regionName ?regionName . "
        "FILTER(?surveyType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey)) "
        "} GROUP BY ?surveyType ?regionName"
    )
    literal_query = (
        "SELECT ?regionName ?surveyType (SUM(?units) AS ?totalDemand) WHERE { "
        "VALUES (?surveyClass ?surveyType) { "
        "(survey:OEM_Survey \"OEM\") "
        "(survey:Tier1_Survey \"Tier1\") "
        "(survey:Semiconductor_Survey \"Semiconductor\") } "
        "?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; "
        "survey:inRegion ?r ; survey:totalDemand ?units . "
        "?o a ?surveyClass . "
        "?r survey:regionName ?regionName . "
        "} GROUP BY ?regionName ?surveyType"
    )

    assert semantic_judge_report(question, literal_query)["score"] > semantic_judge_report(question, uri_query)["score"]


def test_semantic_judge_penalizes_bl_pivot_when_values_requested():
    question = "Give baseline-level current-demand percentages for Tier1 Automotive, limited to BL1 and BL2."
    pivot_query = (
        "SELECT ?marketSegment (SUM(IF(?baseline = \"BL1\", ?pct, 0)) AS ?changeBL1) "
        "(SUM(IF(?baseline = \"BL2\", ?pct, 0)) AS ?changeBL2) WHERE { "
        "?root a survey:CurrentDemandAnalysis ; survey:hasSurveyOrigin survey:Tier1_Survey ; "
        "survey:hasMarketSegment survey:Automotive ; survey:hasAggregatedResult ?entry . "
        "?entry survey:baselineType ?baseline ; survey:percentageChange ?pct . "
        "FILTER(?baseline IN (\"BL1\", \"BL2\")) } GROUP BY ?marketSegment"
    )
    row_query = (
        "SELECT ?baseline ?pct WHERE { "
        "?root a survey:CurrentDemandAnalysis ; survey:hasSurveyOrigin survey:Tier1_Survey ; "
        "survey:hasMarketSegment survey:Automotive ; survey:hasAggregatedResult ?entry . "
        "?entry survey:baselineType ?baseline ; survey:percentageChange ?pct . "
        "FILTER(?baseline IN (\"BL1\", \"BL2\")) } ORDER BY ?baseline"
    )

    assert semantic_judge_report(question, row_query)["score"] > semantic_judge_report(question, pivot_query)["score"]


def test_semantic_selector_conservative_override():
    question = "Return average future-demand change grouped by technology and quarter."
    wrong = (
        "SELECT ?techLabel ?quarterLabel (SUM(?pct) AS ?totalFutureChange) WHERE { "
        "?entry a survey:FutureDemandAnalysis ; survey:analyzesTechnologyCategory ?tech ; "
        "survey:forTimePeriod ?quarter ; survey:percentageChange ?pct . "
        "} GROUP BY ?techLabel ?quarterLabel"
    )
    right = wrong.replace("SUM(?pct) AS ?totalFutureChange", "AVG(?pct) AS ?avgFutureChange")
    ranked = rank_candidates_by_semantic_judge(
        question,
        [{"query": wrong}, {"query": right}],
        min_margin=1.0,
    )
    assert ranked[0]["query"] == right


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


def test_intent_rerank_prefers_average_for_mean_future_demand():
    question = "Return average future-demand change grouped by technology and quarter."
    sum_query = (
        "SELECT ?techLabel ?quarterLabel (SUM(?pct) AS ?totalFutureChange) WHERE { "
        "?entry a survey:FutureDemandAnalysis ; "
        "survey:analyzesTechnologyCategory ?tech ; "
        "survey:forTimePeriod ?quarter ; "
        "survey:percentageChange ?pct . "
        "} GROUP BY ?techLabel ?quarterLabel"
    )
    avg_query = sum_query.replace("SUM(?pct) AS ?totalFutureChange", "AVG(?pct) AS ?avgFutureChange")

    report = _intent_alignment_report(question, avg_query)

    assert "aggregation_avg" in report["matched"]


def test_intent_rerank_prefers_bl_values_when_question_asks_values():
    question = "For Tier1 Automotive current demand, return the BL1 and BL2 percentage-change values."
    delta_query = (
        "SELECT ((SUM(IF(?baseline = 'BL1', ?pct, 0)) - "
        "SUM(IF(?baseline = 'BL2', ?pct, 0))) AS ?deltaBL1BL2) WHERE { "
        "survey:Tier1CurrentDemand a survey:CurrentDemandAnalysis ; "
        "survey:hasAggregatedResult ?entry . "
        "?entry survey:baselineType ?baseline ; survey:percentageChange ?pct . "
        "FILTER(?baseline IN ('BL1','BL2')) }"
    )
    values_query = (
        "SELECT ?baseline ?pct WHERE { "
        "survey:Tier1CurrentDemand a survey:CurrentDemandAnalysis ; "
        "survey:hasMarketSegment survey:Automotive ; "
        "survey:hasAggregatedResult ?entry . "
        "?entry survey:baselineType ?baseline ; survey:percentageChange ?pct . "
        "FILTER(?baseline IN ('BL1','BL2')) } ORDER BY ?baseline"
    )

    report = _intent_alignment_report(question, values_query)

    assert "bl1_bl2_structure" in report["matched"]
    assert "automotive_filter" in report["matched"]


def test_intent_rerank_prefers_named_survey_origin_buckets():
    question = "Break down total demand by region and by survey group: Tier1, OEM, and Semiconductor."
    raw_uri_query = (
        "SELECT ?surveyType ?regionName (SUM(?units) AS ?totalDemand) WHERE { "
        "?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; "
        "survey:inRegion ?r ; survey:totalDemand ?units . "
        "?o a ?surveyType . ?r survey:regionName ?regionName . "
        "FILTER(?surveyType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey)) "
        "} GROUP BY ?surveyType ?regionName"
    )
    labeled_query = (
        "SELECT ?regionName ?surveyType (SUM(?units) AS ?totalDemand) WHERE { "
        "VALUES (?surveyClass ?surveyType) { "
        "(survey:OEM_Survey 'OEM') (survey:Tier1_Survey 'Tier1') "
        "(survey:Semiconductor_Survey 'Semiconductor') } "
        "?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?origin ; "
        "survey:inRegion ?r ; survey:totalDemand ?units . "
        "?origin a ?surveyClass . ?r survey:regionName ?regionName . "
        "} GROUP BY ?regionName ?surveyType"
    )

    raw_report = _intent_alignment_report(question, raw_uri_query)
    labeled_report = _intent_alignment_report(question, labeled_query)

    assert "survey_origin_labeling" in raw_report["missing"]
    assert "survey_origin_labeling" in labeled_report["matched"]


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

    yearly_sales = _template_candidate_queries(
        "Which vehicle type leads total yearly sales?"
    )
    assert yearly_sales
    assert "YearlySalesData" in yearly_sales[0]

    current_demand = _template_candidate_queries(
        "Compare Tier1 BL1 versus BL2 current-demand percentage changes."
    )
    assert current_demand
    assert "Tier1CurrentDemand" in current_demand[0]

    component_share = _template_candidate_queries(
        "For each Tier1 company, count active component-share categories."
    )
    assert component_share
    assert "ComponentShare" in component_share[0]


def test_template_candidates_use_ascending_order_for_lowest_questions():
    lowest_semiconductor = _template_candidate_queries(
        "Which region has the lowest semiconductor demand?"
    )
    assert lowest_semiconductor
    assert any("ASC(" in q for q in lowest_semiconductor)
    assert not any("DESC(" in q for q in lowest_semiconductor)

    highest_semiconductor = _template_candidate_queries(
        "Which region has the highest semiconductor demand?"
    )
    assert highest_semiconductor
    assert any("DESC(" in q for q in highest_semiconductor)
    assert not any("ASC(" in q for q in highest_semiconductor)

    lowest_tech_current_demand = _template_candidate_queries(
        "Which technology category has the lowest current demand?"
    )
    assert lowest_tech_current_demand
    assert any("ASC(" in q for q in lowest_tech_current_demand)

    lowest_yearly_sales = _template_candidate_queries(
        "Which vehicle type has the lowest yearly sales?"
    )
    assert lowest_yearly_sales
    assert any("ASC(" in q for q in lowest_yearly_sales)


def test_dev_dataset_is_separate_from_training_questions():
    train = json.load(open("data/infineon/infineon_train.json", "r", encoding="utf-8"))
    eval_rows = json.load(open("data/infineon/infineon_dev.json", "r", encoding="utf-8"))

    train_questions = {str(row.get("question", "")).strip().lower() for row in train}
    assert len(eval_rows) == 100
    assert all(str(row.get("split", "")) == "dev" for row in eval_rows)
    assert not any(
        str(row.get("question", "")).strip().lower() in train_questions
        for row in eval_rows
    )


def test_final_eval_dataset_is_separate_from_train_and_dev_questions():
    train = json.load(open("data/infineon/infineon_train.json", "r", encoding="utf-8"))
    dev = json.load(open("data/infineon/infineon_dev.json", "r", encoding="utf-8"))
    final = json.load(open("data/infineon/infineon_test_final.json", "r", encoding="utf-8"))

    train_families = {_query_family_signature(str(row.get("query", ""))) for row in train}
    dev_families = {_query_family_signature(str(row.get("query", ""))) for row in dev}
    final_families = [_query_family_signature(str(row.get("query", ""))) for row in final]
    train_questions = {str(row.get("question", "")).strip().lower() for row in train}
    dev_questions = {str(row.get("question", "")).strip().lower() for row in dev}

    assert len(final) == 50
    assert all(str(row.get("id", "")).startswith("FINAL") for row in final)
    assert all(str(row.get("split", "")) == "test_final" for row in final)
    assert len(set(final_families)) == 10
    assert not any(str(row.get("question", "")).strip().lower() in train_questions for row in final)
    assert not any(str(row.get("question", "")).strip().lower() in dev_questions for row in final)
    assert any(fam not in train_families for fam in final_families)
    assert any(fam not in dev_families for fam in final_families)


def test_final_eval_dataset_queries_execute_against_graph():
    final = json.load(open("data/infineon/infineon_test_final.json", "r", encoding="utf-8"))
    graph = Graph()
    graph.parse("data/infineon/graph.ttl", format="turtle")
    prefix = (
        "PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>\n"
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
    )

    seen_queries = set()
    for row in final:
        query = str(row["query"])
        if query in seen_queries:
            continue
        seen_queries.add(query)
        results = list(graph.query(prefix + query))
        assert results, row["id"]


def test_infineon_answer_synthesis_summarizes_regional_demand():
    answer = synthesize_answer(
        "Compare demand by region.",
        "SELECT ?regionName (SUM(?units) AS ?totalDemand) WHERE { ?d a survey:DemandForRegion }",
        {
            "rows": [
                {"regionName": "Europe", "totalDemand": "10"},
                {"regionName": "Japan", "totalDemand": "20"},
            ]
        },
    )

    assert "Regional demand returned 2 row(s)" in answer
    assert "Japan" in answer
    assert "20" in answer


def test_infineon_answer_synthesis_summarizes_future_demand():
    answer = synthesize_answer(
        "Compare future demand by technology and quarter.",
        "SELECT ?techLabel ?quarterLabel WHERE { ?entry a survey:FutureDemandAnalysis }",
        {
            "rows": [
                {"techLabel": "Tech A", "quarterLabel": "Q1", "totalFutureChange": "1.5"},
                {"techLabel": "Tech B", "quarterLabel": "Q2", "totalFutureChange": "4.25"},
            ]
        },
    )

    assert "Future-demand results returned 2 grouped row(s)" in answer
    assert "Tech B" in answer
    assert "Q2" in answer


def test_infineon_answer_synthesis_summarizes_future_demand_time_period_labels():
    answer = synthesize_answer(
        "How do future-demand percentages differ across technologies over time?",
        "SELECT ?techLabel ?timePeriod WHERE { ?entry a survey:FutureDemandAnalysis }",
        {
            "rows": [
                {"techLabel": "Tech A", "timePeriod": "Quarter_Q1_2025", "avgPercentage": "1.5"},
                {"techLabel": "Tech B", "timePeriod": "Quarter_Q2_2025", "avgPercentage": "4.25"},
            ]
        },
    )

    assert "Future-demand results returned 2 grouped row(s)" in answer
    assert "Tech B" in answer
    assert "Quarter Q2 2025" in answer


def test_infineon_answer_synthesis_summarizes_autonomous_max():
    answer = synthesize_answer(
        "Highest autonomous driving percentage.",
        "SELECT ?vehicleType ?saeLevel WHERE { ?entry a survey:AutonomousDrivingDevelopment }",
        {
            "rows": [
                {"vehicleType": "BEV", "saeLevel": "5", "maxPercentage": "102.0"},
            ]
        },
    )

    assert "Autonomous-driving development returned 1 row(s)" in answer
    assert "BEV" in answer
    assert "SAE level 5" in answer


def test_infineon_answer_synthesis_summarizes_bl_comparison():
    answer = synthesize_answer(
        "Compare BL1 and BL2.",
        "SELECT ?baseline WHERE { ?entry a survey:CurrentDemandAnalysis }",
        {
            "rows": [
                {"baseline": "BL1", "pct": "11.04"},
                {"baseline": "BL2", "pct": "-9.03"},
            ]
        },
    )

    assert "Current-demand BL comparison returned 2 row(s)" in answer
    assert "BL1" in answer
    assert "BL2" in answer
    assert "20.07" in answer


def test_infineon_answer_synthesis_summarizes_order_cancellation():
    answer = synthesize_answer(
        "Show order cancellation responses by technology category.",
        "SELECT ?technologyCategory ?responseType WHERE { ?entry a survey:OrderCancellation }",
        {
            "rows": [
                {
                    "technologyCategory": "OrderCancellationChange_Semi_10nm_to_lt28nm",
                    "responseType": "Increase",
                    "participantCount": "2",
                },
                {
                    "technologyCategory": "OrderCancellationChange_Semi_10nm_to_lt28nm",
                    "responseType": "Stable",
                    "participantCount": "1",
                },
            ]
        },
    )

    assert "Order-cancellation results returned 2 row(s)" in answer
    assert "Increase" in answer
    assert "10nm to <28nm" in answer


def test_infineon_answer_synthesis_summarizes_vehicle_sales_by_month():
    answer = synthesize_answer(
        "Aggregate vehicle sales by month.",
        "SELECT ?monthLabel WHERE { ?obs a survey:VehicleSalesObservation }",
        {
            "rows": [
                {"monthLabel": "Jan_2023", "unitsSold": "1000"},
                {"monthLabel": "Feb_2023", "unitsSold": "2500"},
            ]
        },
    )

    assert "Vehicle-sales results returned 2 monthly row(s)" in answer
    assert "Feb 2023" in answer
    assert "2,500" in answer


def test_clarification_options_dedupe_duplicate_display_labels():
    question = "What does future demand look like across technologies?"
    avg_query = (
        "SELECT ?technologyCategory (AVG(?pct) AS ?avgDemand) WHERE { "
        "?entry a survey:FutureDemandAnalysis ; "
        "survey:analyzesTechnologyCategory ?tech ; "
        "survey:percentageChange ?pct . "
        "} GROUP BY ?technologyCategory"
    )
    total_query_a = avg_query.replace("AVG(?pct) AS ?avgDemand", "SUM(?pct) AS ?totalDemand")
    total_query_b = total_query_a.replace("?technologyCategory", "?techLabel")

    payload = build_clarification_payload(
        question,
        [{"query": avg_query}, {"query": total_query_a}, {"query": total_query_b}],
    )

    labels = [option["label"] for option in payload["options"]]
    assert labels == ["Average demand by technology", "Total demand by technology"]


def test_infineon_error_analysis_classifies_failures(tmp_path):
    dataset = [
        {
            "id": "T1",
            "question": "Compare total demand by region.",
            "query": SAMPLE_QUERY,
            "topic": "regional",
            "ambiguity_label": "mid",
        },
        {
            "id": "T2",
            "question": "Show future demand by quarter.",
            "query": "SELECT ?quarterLabel (SUM(?pct) AS ?totalFutureChange) WHERE { ?x a survey:FutureDemandAnalysis ; survey:forTimePeriod ?q ; survey:percentageChange ?pct . ?q survey:periodLabel ?quarterLabel . } GROUP BY ?quarterLabel",
            "topic": "future",
            "ambiguity_label": "high",
        },
    ]
    results = {
        "summary": {},
        "details": [
            {
                "id": "T1",
                "question": dataset[0]["question"],
                "top1_correct": False,
                "any_correct": True,
                "candidates": [
                    {"index": 0, "label": "valid_wrong", "query": "SELECT ?x WHERE { ?x a survey:Region }"},
                    {"index": 1, "label": "correct", "query": SAMPLE_QUERY},
                ],
            },
            {
                "id": "T2",
                "question": dataset[1]["question"],
                "top1_correct": False,
                "any_correct": False,
                "candidates": [
                    {"index": 0, "label": "valid_wrong", "query": "SELECT ?x WHERE { ?x a survey:Region }"},
                ],
            },
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    results_path = tmp_path / "results.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    results_path.write_text(json.dumps(results), encoding="utf-8")

    report = analyze_results(
        results_path=str(results_path),
        dataset_path=str(dataset_path),
        schema_path="data/infineon/schema.json",
    )
    assert report["summary"]["ranking_failures_with_correct_candidate"] == 1
    assert report["summary"]["generation_failures_without_correct_candidate"] == 1
    assert report["cases"][0]["first_correct_candidate_rank"] == 1
    assert "ranking_failure_correct_candidate_not_top1" in render_markdown(report)
