"""Regression guard for bugs found testing the user's SECOND batch of novel
questions (Digital Reference domains never touched before: BMS, CO2 burden,
ATP/AATP, order book, Incoterms; plus analytical KG questions covering
shortage comparisons and BL1/BL2 formatting).

1. DEFINITION_INTENT_PATTERNS only recognized a handful of exact phrasings
   ("what is X", "define X", "explain X", ...). Broadening it to also catch
   "which X ...", "how is/are/does/do X ...", and "what formula/parameters
   ..." was TRIED (relying on the existing `_looks_like_graph_query` check
   right after as a safety net) but had to be REVERTED: that hint list
   isn't exhaustive enough to safely gate anything this broad, and it
   confirmed hijacked real graph-analytics questions with no recognized
   hint word (see test_second_batch_broadening_reverted.py /
   test_bare_which_does_not_hijack_real_analytics_questions below and
   tests/test_second_user_batch_fixes.py's sibling checks). Only the
   narrower "represent(s)?" word was kept, since it tested safe. Unmatched
   ontology-shaped questions now rely on falling through to the LLM
   candidate-generation pipeline instead (see app.py's relative-time/
   out-of-scope guards, which no longer hard-block that fallback either).
2. The cruder `_known_targets_from_question` fallback (a whole-question
   substring scan with no requirement that a match is actually the
   question's real subject) can latch onto an incidental generic word
   instead of the real subject when it does fire. Rather than trying to
   perfectly discriminate good matches from bad ones, results sourced from
   this fallback are marked "Medium" confidence (not "High") with an
   explicit caveat appended to the answer text.
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


def test_bare_which_does_not_hijack_real_analytics_questions():
    # A bare "^which" DEFINITION_INTENT_PATTERN was tried (to catch e.g.
    # "Which chemical factors contribute to the CO2 burden?") and reverted:
    # _looks_like_graph_query's hint list isn't exhaustive enough to safely
    # gate anything this broad. "Which semiconductor companies reported a
    # shortage?" has no recognized hint word ("shortage"/"reported"/
    # "companies" aren't in it) and got wrongly intercepted by the DR
    # ontology lookup ("Semiconductor is an electronic device...") instead
    # of the real shortage-by-company data.
    result = route_dr_ontology_definition("Which semiconductor companies reported a shortage?")
    assert result is None


def test_how_is_x_structured_no_longer_gets_a_route():
    # "^how is/are/does/do" was reverted alongside "^which" -- confirmed to
    # hijack real analytics questions just as badly (e.g. "How do vehicle
    # sales compare between actual and forecast?" -> wrong DR match). This
    # DR-ontology question now correctly falls through instead (to the LLM
    # pipeline) rather than risking that same hijack class.
    result = route_dr_ontology_definition(
        "How is an open order book structured into open orders and order line items?"
    )
    assert result is None


def test_what_formula_links_x_to_y_no_longer_gets_a_route():
    # "what formula/parameters/factors" was reverted for the same reason
    # (e.g. "What factors influence the shortage status of a company?" hit
    # the same wrong DR match).
    result = route_dr_ontology_definition(
        "What formula or parameters link wind turbine lifetime to worldwide CO2 savings?"
    )
    assert result is None


def test_how_and_which_do_not_hijack_real_graph_analytics_questions():
    # The reverted broadening must stay reverted -- these are genuine
    # graph-analytics questions with no recognized _looks_like_graph_query
    # hint word, which is exactly what got hijacked before.
    for question in [
        "Which region has the highest current demand?",
        "Show demand by quarter.",
        "How do vehicle sales compare between actual and forecast?",
        "How are companies distributed by shortage status?",
        "What factors influence the shortage status of a company?",
    ]:
        assert route_dr_ontology_definition(question) is None, question


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
