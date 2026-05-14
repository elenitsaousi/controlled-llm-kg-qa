import re
from typing import Dict, List, Optional, Sequence, Set

_WRITE_RE = re.compile(r"\b(INSERT|DELETE|UPDATE)\b", re.IGNORECASE)
_SELECT_RE = re.compile(r"\bSELECT\b(.*?)\bWHERE\b", re.IGNORECASE | re.DOTALL)
_GROUP_RE = re.compile(
    r"\bGROUP\s+BY\b(.*?)(?:\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_AGG_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
_DESC_RE = re.compile(r"\bORDER\s+BY\s+DESC\s*\(", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+1\b", re.IGNORECASE)
_VAR_RE = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text):
            return True
    return False


def _query_has_any(query: str, terms: Sequence[str]) -> bool:
    q = query.lower()
    return any(term.lower() in q for term in terms)


def _has_word(text: str, *terms: str) -> bool:
    x = (text or "").lower()
    for term in terms:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", x):
            return True
    return False


def _query_has(query: str, *terms: str) -> bool:
    q = query or ""
    return any(term.lower() in q.lower() for term in terms)


def _selected_vars(query: str) -> Set[str]:
    match = _SELECT_RE.search(query or "")
    if not match:
        return set()
    return {v.lstrip("?").lower() for v in _VAR_RE.findall(match.group(1))}


def _group_vars(query: str) -> Set[str]:
    match = _GROUP_RE.search(query or "")
    if not match:
        return set()
    return {v.lstrip("?").lower() for v in _VAR_RE.findall(match.group(1))}


def _aggregations(query: str) -> Set[str]:
    return {m.group(1).upper() for m in _AGG_RE.finditer(query or "")}


def _dimension_terms() -> Dict[str, Sequence[str]]:
    return {
        "region": ["regionname", "inregion", "survey:region"],
        "survey_origin": [
            "hassurveyorigin",
            "oem_survey",
            "tier1_survey",
            "semiconductor_survey",
            "?surveytype",
            "?origintype",
            "?surveyclass",
        ],
        "technology": [
            "technologycategory",
            "analyzestechnologycategory",
            "fortechnologycategory",
            "techlabel",
        ],
        "quarter": ["quarter", "quarterlabel"],
        "year": ["hasyear", "?year", "year "],
        "vehicle_type": ["vehicletype", "hasvehicletype", "analyzesvehicletype"],
        "baseline": ["baselinetype", "?baseline", "bl1", "bl2"],
        "response_type": ["responsetype", "hasresponsetype"],
        "month": ["monthlabel", "survey:month", "?month"],
        "component": ["forcomponent", "componenttype"],
        "market_segment": ["marketsegment", "hasmarketsegment", "automotive"],
    }


def _query_dimensions(query: str) -> Set[str]:
    q = (query or "").lower()
    dims = set()
    for dim, terms in _dimension_terms().items():
        if any(term.lower() in q for term in terms):
            dims.add(dim)
    if "quarter" in dims and "hasyear" not in q and not re.search(r"\?year\b", q):
        dims.discard("year")
    return dims


def _question_dimensions(question: str) -> Set[str]:
    q = (question or "").lower()
    dims = set()
    if _has_word(q, "region", "regions", "regional", "geographic", "geographical"):
        dims.add("region")
    if (
        "survey group" in q
        or "survey origin" in q
        or "origin type" in q
        or _has_word(q, "oem", "tier1", "semiconductor")
        and (_has_word(q, "group", "bucket", "buckets", "split", "separate") or "across" in q)
    ):
        dims.add("survey_origin")
    if _has_word(q, "technology", "tech", "node", "nm"):
        dims.add("technology")
    if _has_word(q, "quarter", "quarters", "quarterly", "q1", "q2", "q3", "q4"):
        dims.add("quarter")
    if _has_word(q, "year", "years", "yearly"):
        dims.add("year")
    if "vehicle type" in q or "vehicle category" in q or _has_word(q, "bev", "behv", "ice"):
        dims.add("vehicle_type")
    if "autonomous" in q and "vehicle" in q:
        dims.add("vehicle_type")
    if _has_word(q, "baseline", "bl1", "bl2"):
        dims.add("baseline")
    if (
        "response type" in q
        or "response label" in q
        or "response direction" in q
        or "cancellation response" in q
        or "cancellation responses" in q
        or _has_word(q, "increase", "decrease", "stable")
    ):
        if "order cancellation" in q or "cancellation" in q or "responses" in q:
            dims.add("response_type")
    if _has_word(q, "month", "months", "monthly"):
        dims.add("month")
    if _has_word(q, "component", "components"):
        dims.add("component")
    if _has_word(q, "automotive"):
        dims.add("market_segment")
    return dims


def _question_origins(question: str) -> Set[str]:
    q = (question or "").lower()
    origins = set()
    if _has_word(q, "oem"):
        origins.add("OEM_Survey")
    if _has_word(q, "tier1", "tier"):
        if "tier1" in q or "tier 1" in q or "tier-1" in q:
            origins.add("Tier1_Survey")
    if _has_word(q, "semiconductor", "semi"):
        origins.add("Semiconductor_Survey")
    return origins


def _query_origins(query: str) -> Set[str]:
    q = query or ""
    origins = set()
    if _query_has(q, "OEM_Survey"):
        origins.add("OEM_Survey")
    if _query_has(q, "Tier1_Survey"):
        origins.add("Tier1_Survey")
    if _query_has(q, "Semiconductor_Survey"):
        origins.add("Semiconductor_Survey")
    return origins


def _question_expected_aggregation(question: str) -> Optional[str]:
    q = (question or "").lower()
    if _has_word(q, "average", "avg", "mean"):
        return "AVG"
    if _has_word(q, "participant", "participants") and (
        _has_word(q, "count", "counts") or "how many" in q or "number of" in q
    ):
        return "SUM"
    if (
        ("percentage change" in q or "percentage changes" in q or _has_word(q, "percentages"))
        and (
            _has_word(q, "matrix", "table", "view", "grouped", "group", "groups")
            or " by " in q
            or " across " in q
            or " over " in q
            or _has_word(q, "quarter", "quarters", "technology", "technologies", "vehicle")
        )
    ):
        return "AVG"
    if _has_word(q, "count", "counts") or "how many" in q or "number of" in q:
        return "COUNT"
    if _has_word(q, "lowest", "smallest", "minimum", "min", "least"):
        return "MIN_OR_BOTTOM"
    if (
        _has_word(q, "highest", "largest", "maximum", "max", "top", "strongest", "leads", "leading")
        or "top-ranked" in q
    ):
        return "MAX_OR_TOP"
    if (
        _has_word(q, "total", "totals", "sum", "aggregate", "aggregated")
        or "total demand" in q
        or "units sold" in q
        or "responses" in q
    ):
        return "SUM"
    return None


def _question_wants_raw_values(question: str) -> bool:
    q = (question or "").lower()
    raw_action = _has_word(q, "list", "show", "return", "give")
    asks_value = (
        "percentage change" in q
        or _has_word(q, "percentage", "percentages", "pct", "value", "values")
    )
    asks_aggregate = (
        _has_word(q, "average", "avg", "mean", "total", "totals", "sum", "count", "highest", "lowest")
        or "how many" in q
    )
    return bool(raw_action and asks_value and not asks_aggregate)


def _question_wants_named_origin_buckets(question: str) -> bool:
    origins = _question_origins(question)
    q = (question or "").lower()
    return len(origins) >= 2 and (
        _has_word(q, "bucket", "buckets", "group", "groups", "split", "separate")
        or "survey group" in q
        or "origin type" in q
        or "survey origin" in q
        or "tier1, oem" in q
    )


def _query_has_literal_origin_labels(query: str) -> bool:
    q = (query or "").lower()
    return (
        "values" in q
        and ("'oem'" in q or '"oem"' in q)
        and ("'tier1'" in q or '"tier1"' in q)
        and ("'semiconductor'" in q or '"semiconductor"' in q)
    )


def _query_projects_origin_class_uri(query: str) -> bool:
    q = (query or "").lower()
    return bool(
        re.search(r"\?[a-z0-9_]*\s+a\s+\?[a-z0-9_]*(surveytype|origin|class)", q)
        or re.search(r"filter\s*\(\s*\?[a-z0-9_]*(surveytype|origin|class)\s+in", q)
    )


def _query_uses_bl_pivot(query: str) -> bool:
    q = (query or "").lower()
    return "if(" in q and "bl1" in q and "bl2" in q and (
        "changebl1" in q or "changebl2" in q or "avgchangebl1" in q or "avgchangebl2" in q
    )


def _question_filters(question: str) -> Set[str]:
    q = (question or "").lower()
    filters = set()
    if _has_word(q, "actual"):
        filters.add("actual")
    if _has_word(q, "forecast", "forecasted"):
        filters.add("forecast")
    if "future demand" in q:
        filters.add("future_demand")
    if "current demand" in q:
        filters.add("current_demand")
    if "order cancellation" in q or "order cancellations" in q or "cancellation response" in q:
        filters.add("order_cancellation")
    if "autonomous" in q or "sae" in q:
        filters.add("autonomous")
    if "sae level 5" in q or "level 5" in q:
        filters.add("sae_level_5")
    if "automotive" in q:
        filters.add("automotive")
    if "bl1" in q:
        filters.add("bl1")
    if "bl2" in q:
        filters.add("bl2")
    return filters


def _query_filters(query: str) -> Set[str]:
    q = (query or "").lower()
    filters = set()
    if "isactualdata" in q and "true" in q:
        filters.add("actual")
    if "isforecastdata" in q and "true" in q:
        filters.add("forecast")
    if "futuredemandanalysis" in q:
        filters.add("future_demand")
    if "currentdemandanalysis" in q:
        filters.add("current_demand")
    if "ordercancellation" in q:
        filters.add("order_cancellation")
    if "autonomousdrivingdevelopment" in q:
        filters.add("autonomous")
    if "sae_level_5" in q or "sae level 5" in q or "level_5" in q:
        filters.add("sae_level_5")
    if "automotive" in q:
        filters.add("automotive")
    if "bl1" in q:
        filters.add("bl1")
    if "bl2" in q:
        filters.add("bl2")
    return filters


def _dimension_shape_score(expected_dims: Set[str], query: str) -> float:
    if not expected_dims:
        return 1.0
    selected = _selected_vars(query)
    grouped = _group_vars(query)
    haystack = " ".join(sorted(selected | grouped))
    hits = 0
    for dim in expected_dims:
        if dim == "survey_origin" and any(x in haystack for x in ["survey", "origin", "type", "class"]):
            hits += 1
        elif dim == "vehicle_type" and "vehicle" in haystack:
            hits += 1
        elif dim == "response_type" and "response" in haystack:
            hits += 1
        elif dim == "market_segment" and ("market" in haystack or "segment" in haystack):
            hits += 1
        elif any(part in haystack for part in dim.split("_")):
            hits += 1
    return hits / float(len(expected_dims))


def semantic_judge_report(question: str, query: str) -> Dict[str, object]:
    """Score whether a candidate SPARQL matches the natural-language intent.

    This is a generic deterministic judge, not an answer-key matcher. It checks
    operation, dimensions, filters, survey-origin coverage, and returned shape.
    """
    query = query or ""
    q_aggr = _question_expected_aggregation(question)
    aggrs = _aggregations(query)
    q_dims = _question_dimensions(question)
    c_dims = _query_dimensions(query)
    q_filters = _question_filters(question)
    c_filters = _query_filters(query)
    q_origins = _question_origins(question)
    c_origins = _query_origins(query)

    score = 0.0
    penalties: List[str] = []
    rewards: List[str] = []

    aggregation_match = None
    if q_aggr:
        if q_aggr == "MAX_OR_TOP":
            aggregation_match = "MAX" in aggrs or bool(_DESC_RE.search(query) and _LIMIT_RE.search(query))
        elif q_aggr == "MIN_OR_BOTTOM":
            aggregation_match = "MIN" in aggrs or (
                "order by" in query.lower()
                and "desc" not in query.lower()
                and bool(_LIMIT_RE.search(query))
            )
        else:
            aggregation_match = q_aggr in aggrs
        if aggregation_match:
            score += 3.0
            rewards.append(f"aggregation:{q_aggr}")
        else:
            score -= 3.5
            penalties.append(f"wrong_or_missing_aggregation:{q_aggr}")
            if q_aggr == "AVG" and "SUM" in aggrs:
                score -= 1.3
                penalties.append("sum_used_for_average")
            if q_aggr == "SUM" and "AVG" in aggrs:
                score -= 1.1
                penalties.append("avg_used_for_total")
    elif aggrs and _question_wants_raw_values(question):
        score -= 1.75
        penalties.append("unexpected_aggregation_for_raw_values")

    missing_dims = sorted(q_dims - c_dims)
    extra_dims = sorted(c_dims - q_dims)
    if q_dims:
        matched_dims = q_dims & c_dims
        score += 1.4 * len(matched_dims)
        if matched_dims:
            rewards.extend(f"dimension:{d}" for d in sorted(matched_dims))
        if missing_dims:
            score -= 2.1 * len(missing_dims)
            penalties.extend(f"missing_dimension:{d}" for d in missing_dims)
        wrong_pairs = [
            ("quarter", "year"),
            ("year", "quarter"),
            ("vehicle_type", "technology"),
            ("technology", "vehicle_type"),
        ]
        for expected, wrong in wrong_pairs:
            if expected in q_dims and wrong in c_dims and expected not in c_dims:
                score -= 1.25
                penalties.append(f"wrong_dimension:{wrong}_instead_of_{expected}")
        for dim in extra_dims:
            if dim not in {"baseline"}:
                score -= 0.4
                penalties.append(f"extra_dimension:{dim}")

    grouped_vars = _group_vars(query)
    if q_aggr and q_dims:
        if grouped_vars:
            score += 0.8
            rewards.append("grouping_present")
        else:
            score -= 2.0
            penalties.append("missing_group_by_for_grouped_request")

    missing_filters = sorted(q_filters - c_filters)
    extra_filters = sorted(c_filters - q_filters)
    if q_filters:
        matched_filters = q_filters & c_filters
        score += 1.25 * len(matched_filters)
        if matched_filters:
            rewards.extend(f"filter:{f}" for f in sorted(matched_filters))
        if missing_filters:
            score -= 1.75 * len(missing_filters)
            penalties.extend(f"missing_filter:{f}" for f in missing_filters)
    if extra_filters:
        # Extra filters are often over-specific constraints that make an otherwise
        # structurally aligned candidate too narrow for the question.
        score -= 0.85 * len(extra_filters)
        penalties.extend(f"over_specific_filter:{f}" for f in extra_filters)

    missing_origins = sorted(q_origins - c_origins)
    extra_origins = sorted(c_origins - q_origins)
    if q_origins:
        score += 1.35 * len(q_origins & c_origins)
        if missing_origins:
            score -= 2.4 * len(missing_origins)
            penalties.extend(f"missing_origin:{o}" for o in missing_origins)
        # If all three origins are requested, an origin-only query is too narrow.
        if len(q_origins) >= 2 and missing_origins:
            score -= 1.25
            penalties.append("too_narrow_origin_scope")
        # If one origin is requested and the query explicitly spans other origins, it is too broad.
        if len(q_origins) == 1 and extra_origins:
            score -= 0.9 * len(extra_origins)
            penalties.append("too_broad_origin_scope")

    if _question_wants_named_origin_buckets(question):
        if _query_has_literal_origin_labels(query):
            score += 1.4
            rewards.append("literal_origin_bucket_labels")
        elif _query_projects_origin_class_uri(query):
            score -= 1.4
            penalties.append("origin_bucket_returns_class_uri")

    shape_score = _dimension_shape_score(q_dims, query)
    if q_dims:
        score += 1.5 * shape_score
        if shape_score < 1.0:
            penalties.append("answer_shape_missing_expected_columns")
        else:
            rewards.append("answer_shape")

    if "difference" in (question or "").lower() or _has_word(question or "", "delta", "subtract"):
        has_delta = _query_has(query, "delta", "bl1bl2", " - ", "- ?")
        if has_delta:
            score += 1.4
            rewards.append("delta_expression")
        else:
            score -= 1.4
            penalties.append("missing_delta_expression")
    elif {"bl1", "bl2"} <= q_filters and _query_uses_bl_pivot(query):
        score -= 1.6
        penalties.append("pivot_bl_columns_for_baseline_values")

    # Prefer queries that produce a projected metric, unless the question asks for raw rows.
    if q_aggr and aggrs:
        metric_vars = [
            v for v in _selected_vars(query)
            if any(tok in v for tok in ["total", "avg", "average", "count", "max", "pct", "percentage"])
        ]
        if metric_vars:
            score += 0.6
            rewards.append("metric_projection")
        else:
            score -= 0.6
            penalties.append("missing_metric_projection")

    return {
        "score": float(score),
        "expected_aggregation": q_aggr,
        "candidate_aggregations": sorted(aggrs),
        "aggregation_match": aggregation_match,
        "expected_dimensions": sorted(q_dims),
        "candidate_dimensions": sorted(c_dims),
        "missing_dimensions": missing_dims,
        "extra_dimensions": extra_dims,
        "expected_filters": sorted(q_filters),
        "candidate_filters": sorted(c_filters),
        "missing_filters": missing_filters,
        "extra_filters": extra_filters,
        "expected_origins": sorted(q_origins),
        "candidate_origins": sorted(c_origins),
        "missing_origins": missing_origins,
        "extra_origins": extra_origins,
        "answer_shape_score": float(shape_score),
        "rewards": rewards,
        "penalties": penalties,
    }


def rank_candidates_by_semantic_judge(
    question: str,
    candidates: Sequence[Dict[str, object]],
    min_margin: float = 1.25,
) -> List[Dict[str, object]]:
    """Conservatively promote a candidate with a clearly better semantic score.

    The original LLM order is preserved unless another candidate beats the
    current first candidate by ``min_margin``. This keeps the selector from
    overreacting to noisy heuristics.
    """
    if not candidates:
        return []

    rows: List[Dict[str, object]] = []
    for idx, cand in enumerate(candidates):
        query = str(cand.get("query", "") or "")
        report = semantic_judge_report(question, query)
        row = dict(cand)
        row["semantic_judge_score"] = float(report.get("score", 0.0))
        row["semantic_judge_report"] = report
        row["semantic_judge_original_rank"] = idx
        rows.append(row)

    first_score = float(rows[0].get("semantic_judge_score", 0.0))
    best_idx = max(
        range(len(rows)),
        key=lambda i: (
            float(rows[i].get("semantic_judge_score", 0.0)),
            -int(rows[i].get("semantic_judge_original_rank", i)),
        ),
    )
    best_score = float(rows[best_idx].get("semantic_judge_score", 0.0))
    if best_idx == 0 or best_score < first_score + float(min_margin):
        return rows

    promoted = rows[best_idx]
    return [promoted] + [row for idx, row in enumerate(rows) if idx != best_idx]


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
        "keywords": ["future demand", "future-demand", "demand forecast", "demand projection"],
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
        "keywords": ["vehicle type", "vehicle category", "bev", "behv", "ice"],
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
