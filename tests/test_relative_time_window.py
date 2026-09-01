import json
import subprocess
import sys

# app.py is a Streamlit script: importing it (st.set_page_config, st.cache_*
# decorators, etc.) has process-global side effects that both pollute
# unrelated tests' rdflib query execution when run in the same pytest process,
# and are slow (~15-20s) to pay per import. So this file makes exactly one
# subprocess call that imports app once and runs every check inside it,
# reporting per-check pass/fail as JSON.

_CHECK_SCRIPT = """
import json
import app

results = {}

def check(name, condition):
    results[name] = bool(condition)

_, q1, note1 = app._semiconductor_relative_time_query(
    "What is semiconductor demand for the last 2 months?"
)
check("two_months_limit_1", "LIMIT 1" in q1)
check("two_months_note_mentions_mapping", "2 requested month(s) maps to roughly 1 quarter(s)" in note1)

_, q2, _ = app._semiconductor_relative_time_query(
    "What is semiconductor demand for the last 6 months?"
)
check("six_months_limit_2", "LIMIT 2" in q2)

_, q2b, _ = app._semiconductor_relative_time_query(
    "What is current demand for semis for the last 2 months?"
)
check("semi_alias_two_months_limit_1", "LIMIT 1" in q2b)

_, q3, _ = app._semiconductor_relative_time_query(
    "Show semiconductor demand for the last 2 quarters."
)
check("two_quarters_limit_2", "LIMIT 2" in q3)

_, q3b, _ = app._semiconductor_relative_time_query(
    "Is semiconductor demand rising or falling over the last 2 quarters?"
)
check("trend_two_quarters_limit_2", "LIMIT 2" in q3b)
check("trend_query_is_quarter_aggregate", "?regionName" not in q3b)

_, q4, _ = app._semiconductor_relative_time_query(
    "What is semiconductor demand for the past year?"
)
check("past_year_limit_4", "LIMIT 4" in q4)

_, q5, _ = app._semiconductor_relative_time_query(
    "What is semiconductor demand for the last months?"
)
check("bare_last_months_limit_1", "LIMIT 1" in q5)

check(
    "ignores_non_semiconductor",
    app._semiconductor_relative_time_query("Show current demand by region.") == ("", "", ""),
)
check(
    "ignores_plain_quarter_breakdown",
    app._semiconductor_relative_time_query("Show semiconductor demand by quarter.") == ("", "", ""),
)

check(
    "current_demand_guided_defers_semiconductor",
    app._current_demand_guided_query(
        "What is semiconductor demand for the last 2 months?"
    ) == ("", "", "", ""),
)
_, q6, _, _ = app._current_demand_guided_query(
    "What is current demand for the last 2 months?"
)
check("current_demand_guided_still_handles_non_semiconductor", "LIMIT 2" in q6)

query_two = app._semiconductor_demand_last_n_quarters_query(2)
query_five = app._semiconductor_demand_last_n_quarters_query(5)
check("last_n_quarters_parametric_2", "LIMIT 2" in query_two)
check("last_n_quarters_parametric_5", "LIMIT 5" in query_five)
check("last_n_quarters_uses_pct_change", "survey:totalDemandPercentageChange" in query_two)
check("last_n_quarters_uses_semiconductor_survey", "survey:Semiconductor_Survey" in query_two)

# Coverage-shortfall honesty note: requested window exceeds available periods.
shortfall = app._coverage_shortfall_note(
    "What are vehicle sales for the past 3 years?",
    [{"year": "2023", "unitsSold": "1000"}, {"year": "2024", "unitsSold": "1200"}],
)
check("shortfall_note_fires_when_n_exceeds_available", "only 2 available years" in shortfall)
check("shortfall_note_lists_actual_years", "2023 and 2024" in shortfall)
check("shortfall_note_states_requested_window", "requested 3-year window" in shortfall)

no_shortfall = app._coverage_shortfall_note(
    "What are vehicle sales for the past 2 years?",
    [{"year": "2023", "unitsSold": "1000"}, {"year": "2024", "unitsSold": "1200"}],
)
check("shortfall_note_silent_when_n_satisfied", no_shortfall == "")

no_window = app._coverage_shortfall_note("Show current demand by region.", [])
check("shortfall_note_silent_without_time_window", no_window == "")

# Plural "upcoming quarters" with no explicit N, but the graph only has one
# quarter of future data past the current quarter -- reported bug where the
# answer silently gave one row with no explanation for a plural request.
upcoming_shortfall = app._coverage_shortfall_note(
    "What is the expected future demand for Tier1 for the upcoming quarters?",
    [{"surveyGroup": "Tier1", "quarterLabel": "Q4 2026", "expectedFutureDemand": "13"}],
)
check("upcoming_shortfall_fires_for_single_row", "only 1 upcoming quarter" in upcoming_shortfall)
check("upcoming_shortfall_names_the_quarter", "Q4 2026" in upcoming_shortfall)

# Singular "upcoming quarter" getting exactly one row is expected, not a shortfall.
upcoming_singular = app._coverage_shortfall_note(
    "What is the expected future demand for Tier1 for the upcoming quarter?",
    [{"surveyGroup": "Tier1", "quarterLabel": "Q4 2026", "expectedFutureDemand": "13"}],
)
check("upcoming_shortfall_silent_for_singular_phrasing", upcoming_singular == "")

# Plural phrasing with several rows actually returned needs no note.
upcoming_multi = app._coverage_shortfall_note(
    "Show semiconductor demand trend for the upcoming quarters.",
    [
        {"quarterLabel": "Q4 2025", "x": "1"},
        {"quarterLabel": "Q1 2026", "x": "2"},
        {"quarterLabel": "Q2 2026", "x": "3"},
    ],
)
check("upcoming_shortfall_silent_when_multiple_rows_returned", upcoming_multi == "")

# Unsupported-region honesty note: question mentions a world region the
# graph does not model at all.
region_note = app._unsupported_region_note("What is current demand in Oceania?")
check("region_note_fires_for_oceania", "does not include data for Oceania" in region_note)
check("region_note_lists_available_regions", "Americas, Europe, Japan, China, and Asia Pacific" in region_note)

check(
    "region_note_silent_for_known_region",
    app._unsupported_region_note("Show current demand by region.") == "",
)
check(
    "region_note_silent_for_europe_and_america",
    app._unsupported_region_note(
        "What is the combined current demand for Europe and America?"
    ) == "",
)

# Flexible Ask should redirect analytics questions to real graph execution,
# but keep source/scope/metadata questions on the existing prose path (no
# SPARQL template exists for graph-introspection questions).
check(
    "flexible_redirects_current_demand_by_region",
    app._flexible_ask_needs_graph_execution("Show current demand by region.") is True,
)
check(
    "flexible_redirects_lowest_semiconductor_demand",
    app._flexible_ask_needs_graph_execution(
        "Which region has the lowest semiconductor demand?"
    ) is True,
)
check(
    "flexible_keeps_prose_for_source_scope_question",
    app._flexible_ask_needs_graph_execution("What sources are loaded?") is False,
)
check(
    "flexible_keeps_prose_for_metadata_question",
    app._flexible_ask_needs_graph_execution("How many triples are in the graph?") is False,
)
check(
    "flexible_keeps_prose_for_definition_question",
    app._flexible_ask_needs_graph_execution("What is a Technology Node?") is False,
)
check(
    "flexible_keeps_prose_for_out_of_scope_question",
    app._flexible_ask_needs_graph_execution("What is the weather in Munich?") is False,
)

print(json.dumps(results))
"""


def test_semiconductor_relative_time_window_behavior():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    results = json.loads(result.stdout.strip().splitlines()[-1])
    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"failed checks: {failed}"
