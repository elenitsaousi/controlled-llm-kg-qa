"""Regression guard for bugs found auditing the shortages, order_cancellation,
autonomous_driving, vehicle_sales, catalog_lookup, and current_demand_baselines
question categories (evaluation/question_sets/philipp_true_demand_dr_test_questions.json).

Bugs fixed in kg/capabilities.py:
1. Autonomous-driving direct queries used "?root a survey:<RootClass>" to
   scope to OEM/Tier1, but AutonomousDrivingDevelopment_OEM/_Tier1 are
   themselves the root nodes (declared rdfs:Class, used as singleton
   individuals) carrying hasSurveyOrigin/hasDetail directly -- nothing is
   ever "a" that class, so the query silently returned zero rows for any
   OEM/Tier1-scoped autonomous-driving question.
2. `_scope_from_question` didn't recognize "semiconductor shortage" as
   establishing Semiconductor scope, so "companies that reported
   semiconductor shortage" ran unscoped across all three survey groups.
3. BL1/BL2 current-demand baseline questions had no deterministic template
   and fell through to free-form LLM SPARQL generation, where a decoy
   entity (Tier1DemandAnalysis with baselineB1Percent/baselineB2Percent
   predicates) could be mistaken for the real BL1/BL2 data.
4. "total vehicles sold each year, grouped by type" had no template joining
   year and vehicle type even though the graph has both together on
   YearlySalesData.

And in app.py:
5. `_status_or_development_guided_query`'s order-cancellation check used a
   literal "order cancellation" substring match, which never matched the
   hyphenated "order-cancellation" phrasing several real questions use.
"""

import pytest
from rdflib import Graph

import app
from kg.capabilities import DEFAULT_REGISTRY

GRAPH_PATH = "data/infineon/graph.ttl"

PREFIX = "PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>\n"


@pytest.fixture(scope="module")
def graph():
    g = Graph()
    g.parse(GRAPH_PATH, format="turtle")
    return g


def test_tier1_autonomous_driving_query_returns_rows(graph):
    report = DEFAULT_REGISTRY.resolve(
        "What is the average autonomous-driving development for Tier1 suppliers, "
        "grouped by vehicle type, SAE level, and year?"
    )
    query = DEFAULT_REGISTRY.direct_query_for(report)
    assert query is not None
    rows = list(graph.query(PREFIX + query))
    assert len(rows) == 45


def test_oem_autonomous_driving_query_returns_rows(graph):
    report = DEFAULT_REGISTRY.resolve("How is the development of the autonomous driving level SAE for OEMs?")
    query = DEFAULT_REGISTRY.direct_query_for(report)
    assert query is not None
    rows = list(graph.query(PREFIX + query))
    assert len(rows) == 5  # one row per SAE level


def test_semiconductor_shortage_company_list_is_scoped(graph):
    report = DEFAULT_REGISTRY.resolve("List companies that reported semiconductor shortage.")
    query = DEFAULT_REGISTRY.direct_query_for(report)
    assert query is not None
    rows = list(graph.query(PREFIX + query))
    names = {str(row[0]) for row in rows}
    assert names == {"Company1_semi", "Company2_semi"}


def test_bl1_bl2_current_demand_returns_real_automotive_values(graph):
    for question in [
        "Which percentage changes apply to Tier1 automotive for baselines BL1 and BL2?",
        "What is the average current-demand change for BL1 and BL2 products in the Tier1 Automotive segment?",
        "What is the total Tier1 current demand percentage change difference between BL1 and BL2?",
        "Compare BL1 and BL2 current-demand changes for Tier1 Automotive.",
    ]:
        report = DEFAULT_REGISTRY.resolve(question)
        query = DEFAULT_REGISTRY.direct_query_for(report)
        assert query is not None, question
        rows = {str(row[0]): float(row[1]) for row in graph.query(PREFIX + query)}
        assert rows == {"BL1": 11.04, "BL2": -9.03}, (question, rows)
        # The decoy entity's nonsensical values must never appear.
        assert 110270 not in rows.values()


def test_vehicle_sales_by_year_and_type_returns_six_rows(graph):
    report = DEFAULT_REGISTRY.resolve("Can you show the total number of vehicles sold each year, grouped by type?")
    query = DEFAULT_REGISTRY.direct_query_for(report)
    assert query is not None
    rows = list(graph.query(PREFIX + query))
    assert len(rows) == 6
    years = {str(row[0]) for row in rows}
    types = {str(row[1]) for row in rows}
    assert years == {"2023", "2024"}
    assert types == {"BEHV", "BEV", "ICE"}


def test_hyphenated_order_cancellation_phrasing_is_recognized():
    # This question has dev-intent trigger words ("increase", "decrease",
    # "stable") and hyphenated "order-cancellation" -- the literal
    # "order cancellation" (space) substring check used to miss it.
    question = (
        "Summarize increase, decrease, and stable order-cancellation response "
        "trends by semiconductor technology category."
    )
    label, query, note = app._status_or_development_guided_query(question)
    assert query
    assert "OrderCancellation" in query


def test_non_hyphenated_order_cancellation_phrasing_still_works():
    label, query, note = app._status_or_development_guided_query(
        "What is the trend for order cancellations?"
    )
    assert query
    assert "OrderCancellation" in query
