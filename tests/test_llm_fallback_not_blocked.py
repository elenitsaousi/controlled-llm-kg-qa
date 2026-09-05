"""Regression guard for two overly-aggressive deterministic "refusal" gates
in app.py that hard-stopped a request (metadata explicitly said
"llm_skipped": True, or Streamlit's own st.stop()) instead of letting it
fall through to the LLM candidate-generation pipeline -- which already has
its own downstream safeguards (candidate ranking, execution-evidence
checks, confidence/margin thresholds) to catch a bad guess. Found by the
user noticing that testing kept hitting deterministic-only refusals with no
chance for the LLM to try, even for questions with a reasonable answer.

1. `_is_unsupported_relative_time_question`'s hard block: when a relative
   time phrase ("the past 3 months") was detected and no deterministic
   approximation matched, the app used to short-circuit with a
   "controlled_no_answer" result and st.stop(), never reaching the LLM
   pipeline. Now it just falls through when no approximation is found.
2. `_is_out_of_scope_question`'s weak "short question, no recognized
   vocabulary" heuristic rejected any question <= 8 tokens that didn't
   happen to contain one of ~40 hardcoded in-scope words -- there's always
   another legitimate phrasing that heuristic hasn't seen. Only the
   confident, explicit off-topic patterns (weather, movies, stock price,
   sports, ...) should still hard-reject.
"""

import app


def test_out_of_scope_still_rejects_confidently_off_topic_questions():
    out_of_scope, _ = app._is_out_of_scope_question("What is the weather in Munich?")
    assert out_of_scope is True

    out_of_scope, _ = app._is_out_of_scope_question("What is the stock price of Infineon?")
    assert out_of_scope is True


def test_out_of_scope_no_longer_rejects_short_novel_in_scope_phrasing():
    # None of these contain a hardcoded in-scope word verbatim, and all are
    # <= 8 tokens -- the old heuristic rejected them outright.
    for question in [
        "What areas need attention?",
        "Give me a recommendation for Japan.",
    ]:
        out_of_scope, reason = app._is_out_of_scope_question(question)
        assert out_of_scope is False, (question, reason)


def test_semiconductor_relative_time_no_longer_hard_blocks_llm_fallback():
    # This question is now handled deterministically (a separate fix), but
    # the underlying gate function itself must no longer be a source of a
    # hard "unsupported" classification for phrasing it can't approximate --
    # confirm the raw detector still fires (so callers know this needs an
    # approximation attempt) without asserting a hard-refusal outcome.
    unsupported, reason = app._is_unsupported_relative_time_question(
        "How many semiconductor companies were surveyed in the past 3 months?"
    )
    assert unsupported is True
    assert reason
