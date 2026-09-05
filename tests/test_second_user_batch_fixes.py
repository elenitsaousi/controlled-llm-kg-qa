"""Regression guard for bugs found testing the user's SECOND batch of novel
questions (Digital Reference domains never touched before: BMS, CO2 burden,
ATP/AATP, order book, Incoterms; plus analytical KG questions covering
shortage comparisons and BL1/BL2 formatting).

1. DEFINITION_INTENT_PATTERNS only recognized a handful of exact phrasings
   ("what is X", "define X", "explain X", ...) so most real ontology
   questions phrased as "which X ...", "how is/are/does/do X ...", or "what
   formula/parameters ..." got NO deterministic route at all and fell to
   the (untestable without LLM credentials) generic kg_query path, which
   has zero DR-ontology content. Broadened the intent patterns; the
   existing `_looks_like_graph_query` safety check (already present, run
   right after) still rejects genuine graph-analytics questions unless they
   also carry strong "define/meaning" intent, so this is safe to broaden.
2. Once broadened, the cruder `_known_targets_from_question` fallback (a
   whole-question substring scan with no requirement that a match is
   actually the question's real subject) started firing much more often,
   and can latch onto an incidental generic word instead of the real
   subject. Rather than trying to perfectly discriminate good matches from
   bad ones, results sourced from this fallback are now marked "Medium"
   confidence (not "High") with an explicit caveat appended to the answer
   text, so a possibly-wrong guess is at least flagged as less certain.
3. "Battery Management System (BMS)" -- a trailing parenthetical
   abbreviation concatenated straight onto the alias key with no separator,
   matching neither the real term's key nor any fuzzy substring of it.
4. COMPARISON_PATTERNS didn't recognize "distinction between X and Y" (only
   "difference between").
5. `_shortage_direct_query`: "X reported experiencing a shortage COMPARED
   TO those that did not" mentions both a positive cue and a negation:
   asks_negative won unconditionally and the query silently answered with
   only the count of companies WITHOUT a shortage, dropping the very
   comparison being asked for.
6. `_format_current_bl_comparison` (llm/answer_synthesis.py) only accepted
   row keys "baseline"/"pct"/"totalChange"/"avgPct"/"avgPctChange", but the
   BL1/BL2 deterministic template (added earlier this session) emits
   "baselineType"/"avgPercentageChange" -- so its own purpose-built
   formatter never actually rendered its answer.
"""

import app
from kg.capabilities import DEFAULT_REGISTRY
from kg.dr_ontology import route_dr_ontology_definition
from llm.answer_synthesis import synthesize_answer
from rdflib import Graph
import pytest

PREFIX = "PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>\n"


@pytest.fixture(scope="module")
def graph():
    g = Graph()
    g.parse("data/infineon/graph.ttl", format="turtle")
    return g


def test_which_x_contribute_to_y_gets_a_route():
    result = route_dr_ontology_definition(
        "Which chemical, gas, and electricity factors contribute to the CO2 burden "
        "in front-end semiconductor manufacturing?"
    )
    assert result is not None
    # Sourced via the broad fallback scan -> must be honestly flagged, not
    # presented as equally certain as a precise phrase match.
    assert result["confidence"] == "Medium"
    assert "broad keyword scan" in result["answer"]


def test_how_is_x_structured_gets_a_route():
    result = route_dr_ontology_definition(
        "How is an open order book structured into open orders and order line items?"
    )
    assert result is not None
    assert "Order" in result["matched_term"]


def test_what_formula_links_x_to_y_gets_a_route():
    result = route_dr_ontology_definition(
        "What formula or parameters link wind turbine lifetime to worldwide CO2 savings?"
    )
    assert result is not None


def test_real_graph_analytics_question_still_bypasses_dr_route():
    # The intent-pattern broadening must not hijack genuine graph-analytics
    # questions -- _looks_like_graph_query's existing safety check should
    # still reject these.
    assert route_dr_ontology_definition("Which region has the highest current demand?") is None
    assert route_dr_ontology_definition("Show demand by quarter.") is None


def test_bms_parenthetical_abbreviation_resolves_correctly():
    result = route_dr_ontology_definition(
        "What is a Battery Management System (BMS), and which functional blocks does it contain?"
    )
    assert result is not None
    assert result["matched_term"] == "Battery Management System"
    assert result["confidence"] == "High"


def test_distinction_between_is_recognized_as_comparison():
    from kg.dr_ontology import _comparison_targets

    targets = _comparison_targets(
        "What is the distinction between current demand and future demand?"
    )
    assert len(targets) == 2


def test_shortage_compared_to_reports_both_sides(graph):
    question = "How many OEM companies reported experiencing a shortage compared to those that did not?"
    report = DEFAULT_REGISTRY.resolve(question)
    query = DEFAULT_REGISTRY.direct_query_for(report)
    assert query is not None
    rows = {str(row[0]): int(row[1]) for row in graph.query(PREFIX + query)}
    assert rows == {"yes": 2, "no": 1}


def test_bl1_bl2_answer_renders_through_dedicated_formatter(graph):
    question = "Which percentage changes apply to Tier1 automotive for baselines BL1 and BL2?"
    report = DEFAULT_REGISTRY.resolve(question)
    query = DEFAULT_REGISTRY.direct_query_for(report)
    rows = [{str(k): str(v) for k, v in row.asdict().items()} for row in graph.query(PREFIX + query)]
    answer = synthesize_answer(question, query, {"rows": rows}, None)
    assert "BL1" in answer and "BL2" in answer
    assert "11.04" in answer and "9.03" in answer
    assert "higher than" in answer or "lower than" in answer
