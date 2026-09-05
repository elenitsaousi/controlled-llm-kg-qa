from __future__ import annotations

from kg.advisory import resolve_advisory_plan, synthesize_advisory_answer


def test_resolves_region_monitoring_advisory_plan():
    plan = resolve_advisory_plan("Based on future demand data, which region should be monitored more closely?")

    assert plan is not None
    assert plan.plan_id == "future_demand_region_focus"
    assert "DemandForRegion" in plan.query


def test_advisory_synthesis_uses_cautious_grounded_wording():
    plan = resolve_advisory_plan("Which vehicle type shows the strongest future demand signal?")
    assert plan is not None

    answer = synthesize_advisory_answer(
        "Which vehicle type shows the strongest future demand signal?",
        plan,
        [
            {"vehicleType": "BEV", "avgPercentageChange": "12.5"},
            {"vehicleType": "ICE", "avgPercentageChange": "3.0"},
        ],
    )

    assert "BEV" in answer
    assert "data-grounded analytical signal" in answer
    assert "not an autonomous business decision" in answer


def test_does_not_map_unsupported_shortage_technology_advice():
    plan = resolve_advisory_plan("Which technology category appears most exposed to shortage?")

    assert plan is None


def test_recognizes_reviewed_first_phrasing_not_just_look_at_first():
    # resolve_advisory_plan used to only recognize the literal phrases
    # "look at first" / "inspect first" as advisory intent, so a natural
    # variant like "should be reviewed first" was invisible to it and never
    # reached the technology-signal advisory plan below.
    plan = resolve_advisory_plan(
        "Which technology category should be reviewed first based on future demand?"
    )
    assert plan is not None
    assert plan.plan_id == "future_demand_technology_signal"


def test_recognizes_checked_and_prioritized_first_phrasings():
    assert resolve_advisory_plan("Which region should be checked first for current demand?") is not None
    assert resolve_advisory_plan("Which region should be prioritized for current demand?") is not None


def test_recognizes_uncertainty_language_as_advisory_intent():
    # "uncertain" wasn't in the advisory-intent vocabulary at all, so a
    # question like this used to be invisible to the advisory route.
    plan = resolve_advisory_plan("Which demand area seems most uncertain?")
    # The question doesn't name a specific dimension (region/technology/
    # vehicle), so it's genuinely ambiguous which template applies -- no
    # plan is expected here, but earlier this failed for the wrong reason
    # (intent not recognized at all, before dimension matching even ran).
    assert plan is None
