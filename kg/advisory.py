from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class AdvisoryPlan:
    plan_id: str
    title: str
    query: str
    group_key: str
    value_key: str
    value_label: str
    objective: str


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _has_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


FUTURE_DEMAND_BY_REGION = """
SELECT ?regionName (AVG(?pct) AS ?avgPercentageChange) WHERE {
  ?entry a survey:DemandForRegion ;
         survey:inRegion ?region ;
         survey:totalDemandPercentageChange ?pct .
  ?region survey:regionName ?regionName .
}
GROUP BY ?regionName
ORDER BY DESC(?avgPercentageChange)
""".strip()

CURRENT_DEMAND_BY_REGION = """
SELECT ?regionName (SUM(?demand) AS ?totalDemand) WHERE {
  ?entry a survey:DemandForRegion ;
         survey:inRegion ?region ;
         survey:totalDemand ?demand .
  ?region survey:regionName ?regionName .
}
GROUP BY ?regionName
ORDER BY DESC(?totalDemand)
""".strip()

FUTURE_DEMAND_BY_VEHICLE_TYPE = """
SELECT ?vehicleType (SUM(?sales) AS ?yearlySales) WHERE {
  ?entry a survey:YearlySalesData ;
         survey:analyzesVehicleType ?vehicle ;
         survey:yearlySales ?sales .
  BIND(REPLACE(STR(?vehicle), "^.*/", "") AS ?vehicleType)
}
GROUP BY ?vehicleType
ORDER BY DESC(?yearlySales)
""".strip()

FUTURE_DEMAND_BY_TECHNOLOGY = """
SELECT ?technologyCategory (AVG(?pct) AS ?avgPercentageChange) WHERE {
  ?entry a survey:FutureDemandAnalysis ;
         survey:analyzesTechnologyCategory ?technology ;
         survey:percentageChange ?pct .
  OPTIONAL { ?technology survey:technologyCategoryName ?technologyName . }
  BIND(COALESCE(?technologyName, REPLACE(STR(?technology), "^.*/", "")) AS ?technologyCategory)
}
GROUP BY ?technologyCategory
ORDER BY DESC(?avgPercentageChange)
""".strip()

SHORTAGE_BY_SURVEY_GROUP = """
SELECT ?surveyGroup ?shortageStatus (COUNT(?company) AS ?companyCount) WHERE {
  VALUES (?surveyClass ?surveyGroup) {
    (survey:OEM_Survey "OEM")
    (survey:Tier1_Survey "Tier1")
    (survey:Semiconductor_Survey "Semiconductor")
  }
  ?company a survey:Company ;
           survey:hasSurveyOrigin ?origin ;
           survey:reportsShortage ?shortage .
  ?origin a ?surveyClass .
  BIND(IF(?shortage = true, "yes", "no") AS ?shortageStatus)
}
GROUP BY ?surveyGroup ?shortageStatus
ORDER BY DESC(?companyCount)
""".strip()


def resolve_advisory_plan(question: str) -> Optional[AdvisoryPlan]:
    q = _norm(question)
    if not q:
        return None

    advisory_intent = _has_any(
        q,
        (
            "monitor",
            "look at first",
            "inspect first",
            "planning attention",
            "focus",
            "risk",
            "exposed",
            "strongest",
            "signal",
            "recommend",
            "suggest",
            "should i look",
        ),
    )
    if not advisory_intent:
        return None

    if "demand" in q and "vehicle" in q and _has_any(q, ("future", "strongest", "signal")):
        return AdvisoryPlan(
            plan_id="future_demand_vehicle_signal",
            title="Strongest future-demand signal by vehicle type",
            query=FUTURE_DEMAND_BY_VEHICLE_TYPE,
            group_key="vehicleType",
            value_key="avgPercentageChange",
            value_label="average future-demand percentage change",
            objective="identify the vehicle type with the strongest future-demand signal",
        )

    if "demand" in q and _has_any(q, ("technology", "category", "node")) and _has_any(q, ("future", "risk", "strongest", "signal")):
        return AdvisoryPlan(
            plan_id="future_demand_technology_signal",
            title="Strongest future-demand signal by technology category",
            query=FUTURE_DEMAND_BY_TECHNOLOGY,
            group_key="technologyCategory",
            value_key="avgPercentageChange",
            value_label="average future-demand percentage change",
            objective="identify the technology category with the strongest future-demand signal",
        )

    if (
        "shortage" in q
        and _has_any(q, ("survey", "group", "area", "exposed", "risk", "focus"))
        and not _has_any(q, ("technology", "category", "node"))
    ):
        return AdvisoryPlan(
            plan_id="shortage_survey_exposure",
            title="Shortage exposure by survey group",
            query=SHORTAGE_BY_SURVEY_GROUP,
            group_key="surveyGroup",
            value_key="companyCount",
            value_label="companies reporting the shortage status",
            objective="identify where shortage signals appear most visible in the survey data",
        )

    if "demand" in q and "region" in q:
        if "current" in q:
            return AdvisoryPlan(
                plan_id="current_demand_region_focus",
                title="Current-demand focus by region",
                query=CURRENT_DEMAND_BY_REGION,
                group_key="regionName",
                value_key="totalDemand",
                value_label="total current demand",
                objective="identify the region with the highest current-demand signal",
            )
        return AdvisoryPlan(
            plan_id="future_demand_region_focus",
            title="Future-demand focus by region",
            query=FUTURE_DEMAND_BY_REGION,
            group_key="regionName",
            value_key="avgPercentageChange",
            value_label="average future-demand percentage change",
            objective="identify the region with the strongest future-demand signal",
        )

    if "future" in q and "demand" in q and _has_any(q, ("planning attention", "focus", "inspect", "monitor", "risk", "important")):
        return AdvisoryPlan(
            plan_id="future_demand_region_focus",
            title="Future-demand focus by region",
            query=FUTURE_DEMAND_BY_REGION,
            group_key="regionName",
            value_key="avgPercentageChange",
            value_label="average future-demand percentage change",
            objective="identify the region with the strongest future-demand signal",
        )

    return None


def _to_float(value: object) -> Optional[float]:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def synthesize_advisory_answer(question: str, plan: AdvisoryPlan, rows: List[Dict[str, object]]) -> str:
    if not rows:
        return (
            "I could not derive a graph-grounded advisory signal for this request because "
            "the deterministic advisory query returned no rows."
        )

    ranked = sorted(
        rows,
        key=lambda row: (_to_float(row.get(plan.value_key)) is not None, _to_float(row.get(plan.value_key)) or float("-inf")),
        reverse=True,
    )
    top = ranked[0]
    group = str(top.get(plan.group_key) or "the leading group")
    value = top.get(plan.value_key)
    value_text = f" ({plan.value_label}: {value})" if value is not None else ""
    preview = []
    for row in ranked[:3]:
        label = str(row.get(plan.group_key) or "").strip()
        metric = row.get(plan.value_key)
        if label:
            preview.append(f"{label}: {metric}")
    preview_text = "; ".join(preview)

    answer = (
        f"Based on the graph results, {group} is the first area to inspect for this request"
        f"{value_text}. I selected this advice because the deterministic advisory route "
        f"computed {plan.value_label} for each returned group, sorted the graph-backed "
        "results in descending order, and used the strongest returned signal as the "
        "priority for review. This is a data-grounded analytical signal, not an autonomous "
        "business decision."
    )
    if preview_text:
        answer += f" The top returned evidence rows were: {preview_text}."
    answer += (
        f" The deterministic advisory template used here was designed to {plan.objective}; "
        "therefore the recommendation should be read as a prioritization cue for human review, "
        "not as a final operational decision."
    )
    return answer
