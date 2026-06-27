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
