"""Regression guard: `_semiconductor_relative_time_query` (app.py) used to
require the literal word "demand" in the question, so a perfectly natural
phrasing like "How is semiconductor developing over the past 3 months?"
(no "demand" anywhere) was silently rejected and the question fell through
to the generic "I cannot answer this exact relative-time request" refusal
instead of the deterministic trend query this exact intent already has a
supported path for (confirmed working for "Is the current demand for
semiconductors in the past 6 months rising or falling?", which does say
"demand"). Demand is the only quarter-level semiconductor signal this graph
tracks, so trend/relative-time intent about "semiconductor" alone is enough
of a signal to proceed.
"""

import app


def test_semiconductor_developing_past_n_months_without_the_word_demand():
    label, query, note = app._semiconductor_relative_time_query(
        "how is semiconductor developing the past 3 months"
    )
    assert query, "expected a real query, not the unsupported-relative-time refusal"
    assert "quarter" in note.lower()


def test_semiconductor_trend_with_demand_word_still_works():
    label, query, note = app._semiconductor_relative_time_query(
        "Is the current demand for semiconductors in the past 6 months rising or falling?"
    )
    assert query


def test_semiconductor_relative_time_without_demand_or_trend_signal_still_declines():
    # No "demand" and no trend/developing word -- genuinely too vague to
    # assume this means demand trend, should still decline gracefully.
    label, query, note = app._semiconductor_relative_time_query(
        "How many semiconductor companies were surveyed in the past 3 months?"
    )
    assert query == ""
