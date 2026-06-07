import os
import re
from typing import Dict, Iterable, List, Optional, Set

from kg.schema import KGSchema


COMMON_CLASSES = {
    "Company",
    "Region",
    "Quarter",
    "Month",
    "TechnologyCategory",
    "OEM_Survey",
    "Tier1_Survey",
    "Semiconductor_Survey",
}

COMMON_PREDICATES = {
    "forCompany",
    "forTimePeriod",
    "hasSurveyOrigin",
    "inRegion",
    "forTechnologyCategory",
    "analyzesTechnologyCategory",
}

SLICE_CLASS_KEYWORDS: Dict[str, Set[str]] = {
    "inventory": {
        "Component",
        "ComponentShare",
        "ComponentShare_Tier1",
        "ComponentType_Tier1",
        "InventoryDevelopment_Semi",
        "InventoryDevelopment_Tier1",
        "InventoryTargetIndicator_Semi",
        "EV",
        "NonEV",
        "Mixed",
    },
    "future_demand": {
        "FutureDemandAnalysis",
        "FutureRegionalDemand",
        "AggregatedDemand",
        "Demand",
        "DemandForRegion",
        "Automotive",
        "BEV",
        "BEHV",
        "ICE",
    },
    "regional_demand": {
        "CurrentRegionalDemand",
        "FutureRegionalDemand",
        "DemandForRegion",
        "Demand",
        "AggregatedDemand",
        "CurrentDemandAnalysis",
        "FutureDemandAnalysis",
    },
    "vehicle_sales": {
        "VehicleSalesObservation",
        "YearlySalesData",
        "BEV",
        "BEHV",
        "ICE",
    },
    "autonomous_driving": {
        "AutonomousDrivingDevelopment",
        "AutonomousDrivingDevelopment_OEM",
        "AutonomousDrivingDevelopment_Tier1",
        "SAELevel",
        "BEV",
        "BEHV",
        "ICE",
    },
    "order_cancellation": {
        "OrderCancellation",
        "InventoryDevelopment_Tier1",
        "TechnologyCategory",
    },
    "shortage": {
        "Company",
        "OEM_Survey",
        "Tier1_Survey",
        "Semiconductor_Survey",
    },
    "current_demand_baselines": {
        "CurrentDemandAnalysis",
        "DemandResponse",
        "AggregatedDemand",
        "Automotive",
        "BEV",
        "BEHV",
        "ICE",
    },
    "catalog_lookup": {
        "Company",
        "Region",
        "Quarter",
        "Month",
        "TechnologyCategory",
        "Component",
        "SAELevel",
        "ConversionFactor",
    },
}

SLICE_QUESTION_KEYWORDS: Dict[str, Set[str]] = {
    "inventory": {"inventory", "component", "stock", "ev", "non-ev", "nonev", "mixed"},
    "future_demand": {"future", "forecast", "forecasted", "projected", "projection", "option1", "option2", "option3"},
    "regional_demand": {"region", "regional", "americas", "europe", "asia", "demand by region", "per region"},
    "vehicle_sales": {"vehicle sales", "vehicle unit", "vehicles sold", "sold", "sales", "actual vehicle", "forecasted vehicle", "month", "yearly"},
    "autonomous_driving": {"autonomous", "sae", "driving", "level 5", "adas"},
    "order_cancellation": {"order cancellation", "order-cancellation", "cancellation", "cancel", "response type"},
    "shortage": {"shortage", "shortages", "reported shortage", "shortage status"},
    "current_demand_baselines": {"current demand", "baseline", "bl1", "bl2", "b1", "b2", "percentage change"},
    "catalog_lookup": {"list", "names", "labels", "types", "catalog", "available", "which classes"},
}


def _max_schema_slice_families() -> int:
    try:
        return max(1, min(4, int(os.environ.get("INFINEON_SCHEMA_SLICING_MAX_FAMILIES", "3") or 3)))
    except Exception:
        return 3


def _tokenize(text: str) -> Set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(text or "").lower().replace("-", " ")))
    if "tier" in tokens and "1" in tokens:
        tokens.add("tier1")
    if "non" in tokens and "ev" in tokens:
        tokens.add("nonev")
    return tokens


def infer_schema_slice_route(
    question: str,
    predicted_labels: Optional[Iterable[str]] = None,
    *,
    max_families: Optional[int] = None,
) -> Dict[str, object]:
    """Cheap family router for prompt slicing.

    It routes only the schema context sent to the LLM. Query execution still runs
    against the full graph.
    """
    text = " ".join([question or "", " ".join(str(x) for x in (predicted_labels or []))]).lower()
    tokens = _tokenize(text)
    scores: Dict[str, float] = {}
    for name, keywords in SLICE_QUESTION_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            key = keyword.lower()
            key_tokens = _tokenize(key)
            if " " in key and key in text:
                score += 2.0
            elif key in tokens:
                score += 1.0
            elif key_tokens and key_tokens.issubset(tokens):
                score += 1.4
        scores[name] = score

    # Domain combinations that intentionally need multiple families.
    if scores.get("future_demand", 0) > 0 and scores.get("regional_demand", 0) > 0:
        scores["future_demand"] += 0.5
        scores["regional_demand"] += 0.5
    if "demand" in tokens and scores.get("regional_demand", 0) > 0:
        scores["regional_demand"] += 0.5
    if "vehicle" in tokens and "sales" in tokens:
        scores["vehicle_sales"] += 1.0
    if "company" in tokens or "companies" in tokens:
        scores["catalog_lookup"] += 0.25
        if "shortage" in tokens or "shortages" in tokens:
            scores["shortage"] += 0.75

    ranked = sorted(((name, score) for name, score in scores.items() if score > 0), key=lambda item: item[1], reverse=True)
    if not ranked:
        return {"selected": [], "confidence": "low", "scores": []}

    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    max_selected = max_families or _max_schema_slice_families()
    if top_score >= 2.0 and top_score - second_score >= 1.25:
        selected = [ranked[0][0]]
        confidence = "high"
    elif top_score >= 1.5:
        selected = [name for name, score in ranked if score >= max(1.0, top_score - 1.25)][:max_selected]
        confidence = "medium"
    else:
        selected = []
        confidence = "low"

    return {
        "selected": selected,
        "confidence": confidence,
        "scores": [{"family": name, "score": round(score, 3)} for name, score in ranked],
    }


def infer_schema_slice_names(question: str, predicted_labels: Optional[Iterable[str]] = None) -> List[str]:
    route = infer_schema_slice_route(question, predicted_labels)
    return list(route.get("selected") or [])


def build_schema_slice(schema: KGSchema, slice_names: Iterable[str]) -> KGSchema:
    selected_names = [name for name in slice_names if name in SLICE_CLASS_KEYWORDS]
    if not selected_names:
        return schema

    class_keep = set(COMMON_CLASSES)
    for name in selected_names:
        class_keep.update(SLICE_CLASS_KEYWORDS[name])
    class_keep = {cls for cls in class_keep if cls in set(schema.classes)}

    relationships = []
    predicate_keep = set(COMMON_PREDICATES)
    for rel in schema.relationships:
        rel_type = str(rel.get("type", ""))
        from_classes = set(rel.get("from", []) or [])
        to_classes = set(rel.get("to", []) or [])
        if rel_type in COMMON_PREDICATES or from_classes & class_keep or to_classes & class_keep:
            relationships.append(rel)
            if rel_type:
                predicate_keep.add(rel_type)

    predicates = sorted({pred for pred in schema.predicates if pred in predicate_keep})
    property_keep = set()
    for rel in relationships:
        rel_type = str(rel.get("type", ""))
        if rel_type in schema.properties:
            property_keep.add(rel_type)
    for prop in schema.properties:
        prop_l = prop.lower()
        if any(cls.lower().replace("_", "") in prop_l.replace("_", "") for cls in class_keep):
            property_keep.add(prop)
        if prop in {
            "companyName",
            "regionName",
            "totalDemand",
            "currentDemand",
            "percentageChange",
            "participantCount",
            "inventoryTrend",
            "yearlySales",
            "hasPercentage",
            "hasYear",
            "forYear",
            "isActualData",
            "isForecastData",
            "baselineType",
            "baselineB1Percent",
            "baselineB2Percent",
            "percentChangeB1",
            "percentChangeB2",
        }:
            property_keep.add(prop)

    return KGSchema(
        {
            "description": f"{schema.description} Selected ontology slice: {', '.join(selected_names)}.",
            "classes": sorted(class_keep),
            "predicates": predicates,
            "properties": sorted(prop for prop in schema.properties if prop in property_keep),
            "relationships": relationships,
            "notes": list(schema.notes) + [
                "This is a focused ontology slice for candidate generation. The final query is still executed on the full graph."
            ],
        }
    )


def schema_slicing_enabled() -> bool:
    raw = os.environ.get("INFINEON_ENABLE_SCHEMA_SLICING", "0")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def full_schema_fallback_enabled() -> bool:
    raw = os.environ.get("INFINEON_SCHEMA_SLICING_FULL_FALLBACK", "0")
    return raw.strip().lower() not in {"0", "false", "no", "off"}
