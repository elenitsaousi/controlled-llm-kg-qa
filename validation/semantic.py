import re
from typing import Dict, List, Sequence

_WRITE_RE = re.compile(r"\b(INSERT|DELETE|UPDATE)\b", re.IGNORECASE)


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text):
            return True
    return False


def _query_has_any(query: str, terms: Sequence[str]) -> bool:
    q = query.lower()
    return any(term.lower() in q for term in terms)


_DOMAIN_REQUIREMENTS = [
    {
        "id": "Tier1_Survey",
        "keywords": ["tier1", "tier 1", "tier-1"],
        "query_terms": ["Tier1_Survey", "Tier1_Survey_Instance", "Tier1"],
    },
    {
        "id": "OEM_Survey",
        "keywords": ["oem"],
        "query_terms": ["OEM_Survey", "OEM_Survey_Instance", "OEM"],
    },
    {
        "id": "Semiconductor_Survey",
        "keywords": ["semiconductor", "semi"],
        "query_terms": ["Semiconductor_Survey", "SemiCurrentDemand", "SemiFutureDemand"],
    },
    {
        "id": "DemandForRegion",
        "keywords": ["region", "regions", "regional", "geographic", "geographically"],
        "query_terms": ["DemandForRegion", "inRegion", "regionName", "Region"],
    },
    {
        "id": "totalDemand",
        "keywords": ["total demand", "demand per region", "demand by region", "regional demand"],
        "query_terms": ["totalDemand", "unitsSold", "SUM("],
    },
    {
        "id": "percentageChange",
        "keywords": ["percentage change", "change", "trend", "evolve", "evolution"],
        "query_terms": ["percentageChange", "totalDemandPercentageChange"],
    },
    {
        "id": "FutureDemandAnalysis",
        "keywords": ["future demand", "forecast", "projection", "projected"],
        "query_terms": ["FutureDemandAnalysis", "FutureDemand", "Option1", "Option2", "Option3"],
    },
    {
        "id": "CurrentDemandAnalysis",
        "keywords": ["current demand", "current demand change"],
        "query_terms": ["CurrentDemandAnalysis", "CurrentDemand"],
    },
    {
        "id": "TechnologyCategory",
        "keywords": ["technology", "technology category", "technology node", "nm"],
        "query_terms": [
            "TechnologyCategory",
            "analyzesTechnologyCategory",
            "forTechnologyCategory",
            "technologyCategoryName",
            "TechCategory",
        ],
    },
    {
        "id": "Quarter",
        "keywords": ["quarter", "quarters", "quarterly", "q1", "q2", "q3", "q4"],
        "query_terms": ["Quarter", "forTimePeriod", "periodLabel", "quarter"],
    },
    {
        "id": "reportsShortage",
        "keywords": ["shortage", "shortages"],
        "query_terms": ["reportsShortage"],
    },
    {
        "id": "InventoryDevelopment_Tier1",
        "keywords": ["inventory", "stock", "stocks"],
        "query_terms": ["InventoryDevelopment", "inventoryTrend", "forComponent"],
    },
    {
        "id": "AutonomousDrivingDevelopment",
        "keywords": ["autonomous", "autonomous driving", "adas", "sae"],
        "query_terms": [
            "AutonomousDrivingDevelopment",
            "hasSAELevel",
            "hasVehicleType",
            "hasPercentage",
        ],
    },
    {
        "id": "VehicleType",
        "keywords": ["vehicle", "vehicle type", "bev", "behv", "ice"],
        "query_terms": ["hasVehicleType", "forVehicleType", "BEV", "BEHV", "ICE"],
    },
    {
        "id": "OrderCancellation",
        "keywords": ["order cancellation", "order cancellations", "cancel", "cancellation"],
        "query_terms": ["OrderCancellation", "hasOrderCancellation", "hasResponseType"],
    },
    {
        "id": "baselineType",
        "keywords": ["baseline", "bl1", "bl2", "option1", "option2", "option3"],
        "query_terms": ["baselineType", "BL1", "BL2", "Option1", "Option2", "Option3"],
    },
    {
        "id": "aggregation",
        "keywords": ["how many", "count", "total", "average", "avg", "mean", "sum"],
        "query_terms": ["COUNT(", "SUM(", "AVG(", "GROUP BY"],
    },
    {
        "id": "comparison",
        "keywords": ["compare", "comparison", "difference", "vs", "versus", "between"],
        "query_terms": ["GROUP BY", "UNION", "FILTER", "VALUES", "IF("],
    },
]


def semantic_coverage_report(question: str, query: str) -> Dict[str, object]:
    """Measure whether a query covers domain concepts explicitly requested by a question."""
    q = (question or "").lower()
    required: List[str] = []
    covered: List[str] = []
    missing: List[str] = []

    for spec in _DOMAIN_REQUIREMENTS:
        if not _contains_any(q, spec["keywords"]):
            continue
        concept = str(spec["id"])
        required.append(concept)
        if _query_has_any(query or "", spec["query_terms"]):
            covered.append(concept)
        else:
            missing.append(concept)

    required_count = len(required)
    covered_count = len(covered)
    score = 1.0 if required_count == 0 else covered_count / required_count
    return {
        "required": required,
        "covered": covered,
        "missing": missing,
        "required_count": required_count,
        "covered_count": covered_count,
        "missing_count": len(missing),
        "coverage_score": float(score),
    }


def validate_query_semantic(query: str) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if _WRITE_RE.search(query):
        errors.append(
            {
                "type": "semantic",
                "message": "Write operations are not allowed in read-only QA.",
            }
        )
    return errors
