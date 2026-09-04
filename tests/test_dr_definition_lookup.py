"""Regression guard for DR ontology definition/comparison lookup bugs found
by auditing Philipp's digital_reference_ontology test questions
(evaluation/question_sets/philipp_true_demand_dr_test_questions.json).

Three bugs were found and fixed in kg/dr_ontology.py:

1. `_best_term`'s property-prefix branch gave a flat 0.72-0.87 score to any
   property whose alias merely starts with the target key, regardless of how
   much longer the alias was. "lobe" (a real project-glossary concept) lost
   to the unrelated ontology property "lobe number" purely because "lobe
   number" starts with "lobe".
2. `_clean_definition_target` never stripped a bare leading "the"/"a"/"an"
   for "Explain ..."/"Define ..." phrasing (only "what is a/the X" strips it,
   via the outer regex). "Explain the has part relationship" left a literal
   "the" in the target, which then matched nothing even though "has part"
   itself resolves fine.
3. `_load_dr_terms`'s alias-collision tiebreak had no principled preference
   when two different entities shared a normalized alias (e.g. three
   distinct properties all literally labeled "has part") -- the winner was
   whichever happened first in (now-sorted, but still arbitrary) iteration
   order. Fixed to prefer whichever entity's own local name (from its URI)
   actually equals the alias, over one that merely inherited a duplicate or
   generic label.
"""

from kg.dr_ontology import route_dr_ontology_definition


def _answer_for(question: str):
    result = route_dr_ontology_definition(question)
    assert result is not None, f"expected a definition result for {question!r}"
    return result


def test_lobe_resolves_to_project_glossary_not_the_lobe_number_property():
    result = _answer_for("What is a lobe?")
    assert result["matched_term"] == "Lobe"
    assert "high-level Digital Reference domain area" in result["answer"]


def test_explain_the_has_part_relationship_resolves():
    result = _answer_for("Explain the has part relationship.")
    assert result["matched_term"] == "has part"


def test_has_part_alias_collision_prefers_the_canonical_property():
    # Planning#has_part's own local name literally is "has part"; other
    # properties (e.g. one about documentation) merely share that label.
    # The canonical one should win regardless of process/run.
    result = _answer_for("Define has part.")
    assert result["matched_term"] == "has part"
    assert "documentation" not in result["term_uri"].lower()


def test_customer_and_supply_chain_still_resolve_correctly():
    # Not part of the bug, but guards against the fixes above regressing
    # unrelated lookups.
    customer = _answer_for("What is a Customer?")
    assert customer["matched_term"] == "Customer"

    supply_chain = _answer_for("What is a Supply Chain?")
    assert supply_chain["matched_term"] == "Supply Chain"


def test_comparison_questions_resolve_both_distinct_sides():
    result = _answer_for("What is the difference between current demand and future demand?")
    assert result["route"] == "definition_comparison"
    labels = {label.strip() for label in result["matched_term"].split(",")}
    assert len(labels) == 2
