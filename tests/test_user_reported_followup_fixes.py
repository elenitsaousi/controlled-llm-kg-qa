"""Regression guard for bugs the user found by manually testing NOVEL
questions (deliberately different phrasing from anything in
evaluation/question_sets/philipp_true_demand_dr_test_questions.json) against
questions covering capabilities fixed earlier in the same session -- exactly
the generalization check that session's fixes are meant to survive.

1. `_format_autonomous_driving` (llm/answer_synthesis.py) hardcoded that a
   grouped result always has both a SAE-level and a year column and printed
   the literal text "None" for whichever one a query didn't actually group
   by (a query grouping by vehicle type + year, with no SAE level, printed
   "... is None in 2027 ..."). Root cause of the "None" specifically:
   _clean_value(None) stringifies to the literal text "None", which is
   truthy, so a naive `if cleaned_value:` check after cleaning doesn't
   catch a genuinely-missing column -- must check the raw _row_get result.
2. The vehicle-sales-by-type formatter had the same blind spot: a query
   grouped by BOTH year and vehicle type got flattened into a single
   "outlook signal" ranking that silently dropped the year dimension and
   mixed values from different years together.
3. `resolve_advisory_plan`'s new region-decision fallback (added earlier
   this session) always reported the single highest-ranked region
   regardless of which region the question actually named -- "give me a
   recommendation for Japan" silently answered about a different region
   (whichever had the highest overall demand) instead of Japan.
4. "advice"/"advise" were missing from advisory_intent's trigger words.
5. DEFINITION_PATTERNS' "what does X mean" pattern had no anchor allowing
   trailing text after "mean" (e.g. "... mean in the Digital Reference?"),
   so it failed to match at all and fell through to a cruder keyword scan
   that matched the trailing scope qualifier ("Digital Reference") instead
   of the actual term ("has-part").
6. COMPARISON_PATTERNS ("X versus Y") has no fixed leading phrase built
   into the regex, so "What's a single lobe versus a cross lobe?" captured
   "What's a single lobe" (with the leading question phrase still attached)
   as the first side, which then failed to resolve to anything.
7. `_source_scope_answer`'s source_intent regex only recognized the exact
   phrase "used for" and "different from"/"differs from" -- not bare
   "X for?" phrasing or the base verb form "differ from".
"""

from llm.answer_synthesis import synthesize_answer
from kg.advisory import resolve_advisory_plan, synthesize_advisory_answer
from kg.capabilities import DEFAULT_REGISTRY
from kg.dr_ontology import route_dr_ontology_definition
import app

from rdflib import Graph
import pytest

PREFIX = "PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>\n"


@pytest.fixture(scope="module")
def graph():
    g = Graph()
    g.parse("data/infineon/graph.ttl", format="turtle")
    return g


def _rows_for(graph, question):
    report = DEFAULT_REGISTRY.resolve(question)
    query = DEFAULT_REGISTRY.direct_query_for(report)
    assert query, question
    rows = [{str(k): str(v) for k, v in row.asdict().items()} for row in graph.query(PREFIX + query)]
    return query, rows


def test_autonomous_driving_grouped_by_vehicle_and_year_has_no_none(graph):
    question = "Break down Tier1 autonomous driving development by vehicle type and year."
    query, rows = _rows_for(graph, question)
    answer = synthesize_answer(question, query, {"rows": rows}, None)
    assert "None" not in answer, answer


def test_autonomous_driving_grouped_by_sae_level_only_has_no_none(graph):
    question = "How is the development of the autonomous driving level SAE for OEMs?"
    label, query, note = app._status_or_development_guided_query(question)
    assert query
    rows = [{str(k): str(v) for k, v in row.asdict().items()} for row in graph.query(PREFIX + query)]
    answer = synthesize_answer(question, query, {"rows": rows}, None)
    assert "None" not in answer, answer


def test_vehicle_sales_by_year_and_type_keeps_year_dimension(graph):
    question = "Show total vehicle sales per year, broken down by type."
    query, rows = _rows_for(graph, question)
    answer = synthesize_answer(question, query, {"rows": rows}, None)
    assert "2023" in answer and "2024" in answer, answer
    assert "outlook signal" not in answer, answer


def test_advisory_recommendation_for_named_region_reports_that_region(graph):
    question = "Give me a recommendation for Japan."
    plan = resolve_advisory_plan(question)
    assert plan is not None
    rows = [{str(k): str(v) for k, v in row.asdict().items()} for row in graph.query(PREFIX + plan.query)]
    answer = synthesize_advisory_answer(question, plan, rows)
    assert "Japan" in answer
    # Must not silently substitute a different top-ranked region's name as
    # if it were answering about Japan.
    assert "Asia Pacific/China is the first area" not in answer


def test_advice_phrasing_recognized_as_advisory_intent():
    plan = resolve_advisory_plan("What's your advice regarding the European market?")
    assert plan is not None


def test_has_part_question_with_trailing_scope_resolves_correctly():
    result = route_dr_ontology_definition(
        "What does the has-part relationship mean in the Digital Reference?"
    )
    assert result is not None
    assert result["matched_term"] == "has part"


def test_whats_a_x_versus_y_comparison_resolves_both_sides():
    result = route_dr_ontology_definition("What's a single lobe versus a cross lobe?")
    assert result is not None
    assert result["route"] == "definition_comparison"
    labels = {label.strip() for label in result["matched_term"].split(",")}
    assert labels == {"Single Lobe", "Cross Lobe"}


def test_source_scope_recognizes_bare_for_and_differ_from_phrasing():
    schema_dict = app._load_schema_dict_cached("data/infineon/schema.json")
    stats = app._safe_graph_data_stats("data/infineon/graph.ttl")

    used_for = app._source_scope_answer("What's the Digital Reference for?", schema_dict, stats, "")
    assert used_for is not None and "Digital Reference" in used_for

    differ_from = app._source_scope_answer(
        "How does the Digital Reference differ from the True Demand data?", schema_dict, stats, ""
    )
    assert differ_from is not None and "True Demand" in differ_from
