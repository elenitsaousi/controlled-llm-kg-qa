from ranking.query_contract import (
    compare_contracts,
    extract_query_contract,
    extract_question_contract,
)
from pipeline.qa import _select_best_candidate_semantic


def test_contract_matches_total_oem_demand_by_region():
    question = "Show total OEM demand by region."
    query = """
    PREFIX survey: <http://example/>
    SELECT ?regionName (SUM(?demand) AS ?totalDemand)
    WHERE {
      ?entry a survey:DemandForRegion ;
        survey:hasSurveyOrigin survey:OEM_Survey ;
        survey:inRegion ?region ;
        survey:totalDemand ?demand .
      ?region survey:regionName ?regionName .
    }
    GROUP BY ?regionName
    """

    q_contract = extract_question_contract(question)
    query_contract = extract_query_contract(query)
    comparison = compare_contracts(q_contract, query_contract)

    assert "demand" in q_contract.metrics
    assert q_contract.aggregation == "sum"
    assert "oem" in q_contract.scopes
    assert "region" in q_contract.dimensions
    assert comparison.score > 0
    assert "contract_scope_match:oem" in comparison.reasons
    assert "contract_dimension_match:region" in comparison.reasons


def test_contract_penalizes_actual_forecast_mismatch():
    question = "How many actual vehicle sales units were sold each month?"
    query = """
    PREFIX survey: <http://example/>
    SELECT ?monthLabel (SUM(?unitsSold) AS ?units)
    WHERE {
      ?obs a survey:VehicleSalesObservation ;
        survey:isForecastData true ;
        survey:forTimePeriod ?month ;
        survey:unitsSold ?unitsSold .
      BIND(STR(?month) AS ?monthLabel)
    }
    GROUP BY ?monthLabel
    """

    comparison = compare_contracts(
        extract_question_contract(question),
        extract_query_contract(query),
    )

    assert "contract_metric_match:vehicle_sales" in comparison.reasons
    assert "contract_aggregation_match:sum" in comparison.reasons
    assert "contract_filter_conflict:forecast" in comparison.reasons


def test_contract_penalizes_wrong_metric():
    question = "What is the inventory by component?"
    query = """
    PREFIX survey: <http://example/>
    SELECT ?regionName (SUM(?demand) AS ?totalDemand)
    WHERE {
      ?entry a survey:DemandForRegion ;
        survey:inRegion ?region ;
        survey:totalDemand ?demand .
      ?region survey:regionName ?regionName .
    }
    GROUP BY ?regionName
    """

    comparison = compare_contracts(
        extract_question_contract(question),
        extract_query_contract(query),
    )

    assert "contract_metric_missing:inventory" in comparison.reasons
    assert "contract_metric_conflict:demand" in comparison.reasons
    assert comparison.score < 0


def test_question_contract_detects_min_and_max_direction():
    lowest = extract_question_contract("Which region has the lowest current demand?")
    highest = extract_question_contract("Which region has the highest current demand?")
    neutral = extract_question_contract("Show current demand by region.")

    assert lowest.direction == "min"
    assert highest.direction == "max"
    assert neutral.direction is None


def test_query_contract_detects_asc_and_desc_direction():
    asc_query = """
    PREFIX survey: <http://example/>
    SELECT ?regionName (SUM(?demand) AS ?totalDemand)
    WHERE {
      ?entry a survey:DemandForRegion ;
        survey:inRegion ?region ;
        survey:totalDemand ?demand .
      ?region survey:regionName ?regionName .
    }
    GROUP BY ?regionName
    ORDER BY ASC(?totalDemand)
    LIMIT 1
    """
    desc_query = asc_query.replace("ASC(?totalDemand)", "DESC(?totalDemand)")

    assert extract_query_contract(asc_query).direction == "min"
    assert extract_query_contract(desc_query).direction == "max"


def test_lowest_question_prefers_asc_candidate_over_desc_candidate():
    question = "Which region has the lowest current demand?"
    desc_query = """
    PREFIX survey: <http://example/>
    SELECT ?regionName (SUM(?demand) AS ?totalDemand)
    WHERE {
      ?entry a survey:DemandForRegion ;
        survey:inRegion ?region ;
        survey:totalDemand ?demand .
      ?region survey:regionName ?regionName .
    }
    GROUP BY ?regionName
    ORDER BY DESC(?totalDemand)
    LIMIT 1
    """
    asc_query = desc_query.replace("DESC(?totalDemand)", "ASC(?totalDemand)")

    q_contract = extract_question_contract(question)
    desc_comparison = compare_contracts(q_contract, extract_query_contract(desc_query))
    asc_comparison = compare_contracts(q_contract, extract_query_contract(asc_query))

    assert "contract_direction_conflict:max" in desc_comparison.reasons
    assert "contract_direction_match:min" in asc_comparison.reasons
    assert asc_comparison.score > desc_comparison.score


def test_contract_selection_override_promotes_clear_aggregation_fix(monkeypatch):
    monkeypatch.setenv("INFINEON_ENABLE_CONTRACT_SELECTION_OVERRIDE", "1")
    question = "Show total OEM demand by region."
    count_query = """
    PREFIX survey: <http://example/>
    SELECT ?regionName (COUNT(?entry) AS ?count)
    WHERE {
      ?entry a survey:DemandForRegion ;
        survey:hasSurveyOrigin survey:OEM_Survey ;
        survey:inRegion ?region ;
        survey:totalDemand ?demand .
      ?region survey:regionName ?regionName .
    }
    GROUP BY ?regionName
    """
    sum_query = """
    PREFIX survey: <http://example/>
    SELECT ?regionName (SUM(?demand) AS ?totalDemand)
    WHERE {
      ?entry a survey:DemandForRegion ;
        survey:hasSurveyOrigin survey:OEM_Survey ;
        survey:inRegion ?region ;
        survey:totalDemand ?demand .
      ?region survey:regionName ?regionName .
    }
    GROUP BY ?regionName
    """

    selected = _select_best_candidate_semantic(
        [
            {
                "query": count_query,
                "semantic_judge_score": 0.65,
                "semantic_judge_report": {"score": 0.65, "penalties": []},
            },
            {
                "query": sum_query,
                "semantic_judge_score": 0.60,
                "semantic_judge_report": {"score": 0.60, "penalties": []},
            },
        ],
        question,
    )

    assert selected is not None
    assert "SUM(?demand)" in selected["query"]


def test_validated_source_rescue_promotes_higher_trust_ml_candidate(monkeypatch):
    monkeypatch.setenv("INFINEON_ENABLE_VALIDATED_SOURCE_RESCUE", "1")
    question = "Show total OEM demand by region."
    generated_query = """
    PREFIX survey: <http://example/>
    SELECT ?regionName (COUNT(?entry) AS ?count)
    WHERE {
      ?entry a survey:DemandForRegion ;
        survey:hasSurveyOrigin survey:OEM_Survey ;
        survey:inRegion ?region ;
        survey:totalDemand ?demand .
      ?region survey:regionName ?regionName .
    }
    GROUP BY ?regionName
    """
    validated_query = """
    PREFIX survey: <http://example/>
    SELECT ?regionName (SUM(?demand) AS ?totalDemand)
    WHERE {
      ?entry a survey:DemandForRegion ;
        survey:hasSurveyOrigin survey:OEM_Survey ;
        survey:inRegion ?region ;
        survey:totalDemand ?demand .
      ?region survey:regionName ?regionName .
    }
    GROUP BY ?regionName
    """

    selected = _select_best_candidate_semantic(
        [
            {
                "query": generated_query,
                "source": "infineon",
                "ml_score": 0.10,
                "semantic_judge_score": 0.80,
                "semantic_judge_report": {"score": 0.80, "penalties": []},
            },
            {
                "query": validated_query,
                "source": "validated_retrieval",
                "ml_score": 0.40,
                "semantic_judge_score": 0.70,
                "semantic_judge_report": {"score": 0.70, "penalties": []},
            },
        ],
        question,
    )

    assert selected is not None
    assert selected["source"] == "validated_retrieval"
    assert "SUM(?demand)" in selected["query"]
