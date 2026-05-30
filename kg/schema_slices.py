import os
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
    "inventory": {"inventory", "component", "stock", "ev", "non-ev", "mixed"},
    "future_demand": {"future", "forecast", "projected", "projection", "option1", "option2", "option3"},
    "regional_demand": {"region", "regional", "americas", "europe", "asia", "demand by region"},
    "vehicle_sales": {"vehicle sales", "sold", "sales", "forecasted vehicle", "actual vehicle", "month"},
    "autonomous_driving": {"autonomous", "sae", "driving", "level 5", "adas"},
    "order_cancellation": {"order cancellation", "cancellation", "cancel"},
    "shortage": {"shortage", "shortages"},
    "current_demand_baselines": {"current demand", "baseline", "bl1", "bl2", "percentage change"},
    "catalog_lookup": {"list", "names", "labels", "types", "how many", "number of"},
}


def infer_schema_slice_names(question: str, predicted_labels: Optional[Iterable[str]] = None) -> List[str]:
    text = " ".join([question or "", " ".join(str(x) for x in (predicted_labels or []))]).lower()
    matches: List[str] = []
    for name, keywords in SLICE_QUESTION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matches.append(name)
    return matches[:2]


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
