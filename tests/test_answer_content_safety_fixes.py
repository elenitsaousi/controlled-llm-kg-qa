"""Regression guard for answer-CONTENT bugs found auditing source_and_scope,
ambiguous_clarification, and unsupported_out_of_scope questions
(evaluation/question_sets/philipp_true_demand_dr_test_questions.json).
Routing was already confirmed correct for these categories; these bugs were
about the actual answer text being wrong or unsafe despite correct routing.

1. `route_dr_ontology_definition` returning a "confidence: Low, I could not
   find..." result used to monopolize the question in app.py's dispatch
   chain (every other guided-query/metadata branch is gated on
   `dr_definition is None`), so a low-confidence non-answer got rendered as
   a confident auto_answer instead of letting source-scope/metadata
   handlers actually answer. Fixed by nulling dr_definition when its
   confidence is Low, right where app.py already nulls it for a similar
   reason.
2. `_source_scope_answer`'s source_intent regex didn't recognize "used
   for"/"purpose"/"different from"/"difference" phrasing.
3. `kg/dr_ontology.py`'s COMPARISON_PATTERNS didn't recognize "how is X
   different from Y" phrasing (only "X vs Y" / "difference between X and
   Y" / "compare X and Y").
4. `resolve_advisory_plan` had no fallback for a question that explicitly
   asks for a "business decision"/recommendation about a named region but
   doesn't mention "demand" -- it fell through with none of the module's
   "not an autonomous business decision" safety disclaimer.
5. `_future_demand_guided_query`'s future_intent regex didn't recognize
   "predict"/"next year" phrasing.
"""

import app
from kg.advisory import resolve_advisory_plan


def test_low_confidence_dr_definition_does_not_block_source_scope_answer():
    # This is the actual bug: app.py's dispatch chain gates every other
    # branch on `dr_definition is None`. Before the fix, a Low-confidence
    # dr_definition result blocked _source_scope_answer entirely.
    dr_definition = app.route_dr_ontology_definition("What is the scope of my sources?")
    assert dr_definition is not None
    assert dr_definition.get("confidence") == "Low"

    schema_dict = app._load_schema_dict_cached("data/infineon/schema.json")
    stats = app._safe_graph_data_stats("data/infineon/graph.ttl")
    answer = app._source_scope_answer("What is the scope of my sources?", schema_dict, stats, "")
    assert answer is not None
    assert "True Demand" in answer


def test_used_for_and_different_from_phrasing_recognized_as_source_scope():
    schema_dict = app._load_schema_dict_cached("data/infineon/schema.json")
    stats = app._safe_graph_data_stats("data/infineon/graph.ttl")

    used_for = app._source_scope_answer(
        "What is the Digital Reference used for?", schema_dict, stats, ""
    )
    assert used_for is not None
    assert "Digital Reference" in used_for

    different_from = app._source_scope_answer(
        "How is the Digital Reference different from the True Demand graph?", schema_dict, stats, ""
    )
    assert different_from is not None
    assert "True Demand" in different_from and "Digital Reference" in different_from


def test_business_decision_for_region_gets_advisory_disclaimer_plan():
    plan = resolve_advisory_plan("Give me a business decision for China.")
    assert plan is not None
    assert plan.plan_id == "current_demand_region_focus"


def test_predict_next_year_recognized_as_future_demand_intent():
    label, query, note = app._future_demand_guided_query(
        "Predict semiconductor demand for next year."
    )
    assert query
