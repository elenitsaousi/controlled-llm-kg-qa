"""Question/query contract extraction for Infineon KGQA selection.

The contract is intentionally coarse-grained. It captures the constraints a
user explicitly asks for, then checks whether a SPARQL candidate appears to
implement the same metric, aggregation, scope, dimensions, and filters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from ranking.feature_extraction import extract_query_plan


WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class QueryContract:
    metrics: Set[str] = field(default_factory=set)
    aggregation: Optional[str] = None
    scopes: Set[str] = field(default_factory=set)
    dimensions: Set[str] = field(default_factory=set)
    filters: Set[str] = field(default_factory=set)
    answer_shape: Optional[str] = None
    labels: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": sorted(self.metrics),
            "aggregation": self.aggregation,
            "scopes": sorted(self.scopes),
            "dimensions": sorted(self.dimensions),
            "filters": sorted(self.filters),
            "answer_shape": self.answer_shape,
            "labels": sorted(self.labels),
        }


@dataclass
class ContractComparison:
    score: float
    reasons: List[str] = field(default_factory=list)
    matched: Dict[str, List[str]] = field(default_factory=dict)
    missing: Dict[str, List[str]] = field(default_factory=dict)
    conflicts: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "score": float(self.score),
            "reasons": list(self.reasons),
            "matched": {k: sorted(v) for k, v in self.matched.items()},
            "missing": {k: sorted(v) for k, v in self.missing.items()},
            "conflicts": {k: sorted(v) for k, v in self.conflicts.items()},
        }


def _normalize(text: str) -> str:
    normalized = " ".join((text or "").lower().replace("tier 1", "tier1").split())
    return normalized.replace("order-cancellation", "order cancellation").replace(
        "future-demand", "future demand"
    )


def _tokens(text: str) -> Set[str]:
    return set(WORD_RE.findall(_normalize(text)))


def _has_any(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)


def _has_word(tokens: Set[str], *words: str) -> bool:
    return any(word in tokens for word in words)


def _has_dimension_cue(text: str, tokens: Set[str], cue: str) -> bool:
    """Match dimension cues without treating short tokens as substrings.

    This avoids false positives such as matching ``ev`` inside ``level`` while
    still allowing phrases like ``non-ev`` and ``technology category``.
    """

    normalized_cue = _normalize(cue)
    cue_tokens = _tokens(normalized_cue)
    if not cue_tokens:
        return False
    if len(cue_tokens) == 1:
        return next(iter(cue_tokens)) in tokens
    return normalized_cue in text


def _add_if(condition: bool, target: Set[str], value: str) -> None:
    if condition:
        target.add(value)


def extract_question_contract(question: str) -> QueryContract:
    q = _normalize(question)
    toks = _tokens(q)
    contract = QueryContract()

    _add_if("vehicle sales" in q or "vehicles sold" in q or "units sold" in q or "sales volume" in q, contract.metrics, "vehicle_sales")
    _add_if("inventory" in q, contract.metrics, "inventory")
    _add_if("order cancellation" in q or "cancellation" in toks, contract.metrics, "order_cancellation")
    _add_if("shortage" in q, contract.metrics, "shortage")
    _add_if("autonomous" in q or "sae" in toks, contract.metrics, "autonomous_driving")
    _add_if("catalog" in toks or "database" in toks or "included in the data" in q, contract.metrics, "catalog_lookup")
    _add_if("demand" in toks and "vehicle_sales" not in contract.metrics, contract.metrics, "demand")

    asks_unit_quantity = (
        _has_any(q, "units sold", "sales units", "vehicles sold", "vehicle sales units")
        or (_has_word(toks, "responses", "participants") and "order cancellation" in q)
    )
    asks_record_count = _has_any(
        q,
        "number of records",
        "count of records",
        "record count",
        "records are there",
        "number of entries",
        "count of entries",
        "entries are there",
        "observations are available",
    )
    asks_company_count = "how many" in q and _has_word(toks, "company", "companies", "oems", "suppliers")

    if _has_any(q, "average", " mean ", " avg "):
        contract.aggregation = "avg"
    elif _has_any(q, "highest", "largest", "maximum", "top", "most", "peak", "lowest", "smallest", "minimum"):
        contract.aggregation = "rank"
        contract.answer_shape = "ranked_one"
    elif asks_unit_quantity or _has_any(q, "total", "sum", "summed", "combined", "overall", "aggregate"):
        contract.aggregation = "sum"
    elif asks_record_count or asks_company_count or _has_any(q, "how many", "number of", "count of", "counts by"):
        contract.aggregation = "count"

    _add_if("oem" in toks, contract.scopes, "oem")
    _add_if("tier1" in toks, contract.scopes, "tier1")
    _add_if("semiconductor" in toks or "semi" in toks, contract.scopes, "semiconductor")

    dimension_checks = [
        ("region", ("region", "regions", "regional", "americas", "china", "europe")),
        ("quarter", ("quarter", "quarters", "q1", "q2", "q3", "q4")),
        ("month", ("month", "monthly", "months")),
        ("year", ("year", "yearly", "years")),
        ("technology_category", ("technology category", "technology categories", "technology", "technologies", "tech")),
        ("vehicle_type", ("vehicle type", "vehicle types", "vehicle category", "bev", "behv", "ice")),
        ("sae_level", ("sae", "sae level", "level 5", "level5")),
        ("component", ("component", "components", "ev", "non-ev", "mixed")),
        ("trend", ("trend", "trends", "increase", "decrease", "stable")),
        ("response_type", ("response type", "response types", "responses")),
        ("shortage_status", ("shortage status", "shortage statuses", "reported a shortage", "have not reported", "did not report", "shortage versus", "whether they experienced")),
        ("baseline", ("baseline", "bl1", "bl2", "option1", "option2", "option3")),
        ("survey", ("survey", "surveys", "origin")),
    ]
    for dimension, cues in dimension_checks:
        if any(_has_dimension_cue(q, toks, cue) for cue in cues):
            contract.dimensions.add(dimension)

    _add_if("actual" in toks or "actuals" in toks or "actually sold" in q, contract.filters, "actual")
    _add_if("forecast" in toks or "forecasted" in toks or "projected" in toks, contract.filters, "forecast")
    _add_if("bl1" in toks, contract.filters, "bl1")
    _add_if("bl2" in toks, contract.filters, "bl2")
    _add_if("option1" in toks, contract.filters, "option1")
    _add_if("option2" in toks, contract.filters, "option2")
    _add_if("option3" in toks, contract.filters, "option3")
    _add_if("without shortage" in q or "not showing" in q or "not indicated" in q or "no shortage" in q, contract.filters, "shortage_no")
    _add_if("with shortage" in q or "experiencing a shortage" in q or "reported a shortage" in q, contract.filters, "shortage_yes")

    if contract.answer_shape is None:
        if _has_any(q, "which ", "list ", "show me the set", "set of", "included in the catalog", "included in the data") and not contract.aggregation:
            contract.answer_shape = "list_values"
        elif _has_any(q, " by ", " across ", " broken down ", " grouped ", " per ", " for each "):
            contract.answer_shape = "grouped_table"
        elif contract.aggregation in {"sum", "count", "avg"} and not contract.dimensions:
            contract.answer_shape = "scalar"

    return contract


def _plan_haystack(plan: Dict[str, object], query: str) -> str:
    pieces: List[str] = [query or ""]
    for key in ("labels", "classes", "predicates", "survey_origins", "group_by_vars", "group_by_predicates", "select_vars", "query_types"):
        values = plan.get(key, [])
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            pieces.extend(str(v) for v in values)
    return _normalize(" ".join(pieces).replace("_", " "))


def extract_query_contract(query: str) -> QueryContract:
    try:
        plan = extract_query_plan(query)
    except Exception:
        plan = {}
    hay = _plan_haystack(plan, query)
    query_lower = (query or "").lower()
    labels = {str(label) for label in plan.get("labels", [])}
    contract = QueryContract(labels=labels)

    _add_if(
        _has_any(hay, "vehicle sales observation", "yearly sales data", "yearly sales", "units sold")
        or any(term in query_lower for term in ("vehiclesalesobservation", "yearlysalesdata", "unitssold", "yearlysales")),
        contract.metrics,
        "vehicle_sales",
    )
    _add_if("inventorydevelopment" in query_lower or "inventory development" in hay or "inventory trend" in hay, contract.metrics, "inventory")
    _add_if("order cancellation" in hay or "ordercancellation" in query_lower, contract.metrics, "order_cancellation")
    _add_if("shortage" in hay, contract.metrics, "shortage")
    _add_if("autonomous driving development" in hay or "autonomousdrivingdevelopment" in query_lower, contract.metrics, "autonomous_driving")
    _add_if(
        any(
            term in query_lower
            for term in (
                " a survey:technologycategory".lower(),
                " a survey:region".lower(),
                " a survey:quarter".lower(),
                " a survey:company".lower(),
                " a survey:vehicletype".lower(),
            )
        )
        and not any(term in query_lower for term in ("sum(", "avg(", "count(")),
        contract.metrics,
        "catalog_lookup",
    )
    _add_if(
        _has_any(hay, "demand for region", "current demand analysis", "future demand analysis", "total demand", "percentage change", "aggregated demand")
        or any(term in query_lower for term in ("demandforregion", "currentdemandanalysis", "futuredemandanalysis", "aggregateddemand")),
        contract.metrics,
        "demand",
    )

    aggregations = {str(v).lower() for v in plan.get("aggregations", [])}
    if "avg" in aggregations or "avg(" in query_lower:
        contract.aggregation = "avg"
    elif "count" in aggregations or "count(" in query_lower:
        contract.aggregation = "count"
    elif "sum" in aggregations or "sum(" in query_lower:
        contract.aggregation = "sum"
    elif "order by" in query_lower and "limit" in query_lower:
        contract.aggregation = "rank"

    _add_if("oem survey" in hay or "oem_survey" in query_lower, contract.scopes, "oem")
    _add_if("tier1 survey" in hay or "tier1_survey" in query_lower, contract.scopes, "tier1")
    _add_if("semiconductor survey" in hay or "semiconductor_survey" in query_lower, contract.scopes, "semiconductor")

    dimension_terms = [
        ("region", ("region", "inregion", "regionname")),
        ("quarter", ("quarter", "periodlabel", "fortimeperiod")),
        ("month", ("month", "monthlabel")),
        ("year", ("year", "foryear", "hasyear", "yearly")),
        ("technology_category", ("technologycategory", "technology category", "techlabel")),
        ("vehicle_type", ("vehicletype", "vehicle type", "analyzesvehicletype", "hasvehicletype")),
        ("sae_level", ("saelevel", "sae level", "hassaelevel")),
        ("component", ("component", "forcomponent")),
        ("trend", ("trend", "inventorytrend", "hasinventorytrend")),
        ("response_type", ("responsetype", "response type", "hasresponsetype")),
        ("shortage_status", ("reportsshortage", "shortagestatus", "shortage status", "shortagelabel", "isshortage")),
        ("baseline", ("baseline", "baselinetype")),
        ("survey", ("survey", "hassurveyorigin")),
    ]
    for dimension, terms in dimension_terms:
        if any(term in hay.replace(" ", "") or term in hay for term in terms):
            contract.dimensions.add(dimension)

    _add_if("isactualdata" in query_lower or "actual data" in hay, contract.filters, "actual")
    _add_if("isforecastdata" in query_lower or "forecast data" in hay, contract.filters, "forecast")
    _add_if("bl1" in hay, contract.filters, "bl1")
    _add_if("bl2" in hay, contract.filters, "bl2")
    _add_if("option1" in hay, contract.filters, "option1")
    _add_if("option2" in hay, contract.filters, "option2")
    _add_if("option3" in hay, contract.filters, "option3")

    query_types = {str(v).lower() for v in plan.get("query_types", [])}
    if "ranking" in query_types or ("order by" in query_lower and "limit" in query_lower):
        contract.answer_shape = "ranked_one"
    elif "grouped" in query_types or plan.get("group_by_vars"):
        contract.answer_shape = "grouped_table"
    elif "select distinct" in query_lower and not contract.aggregation:
        contract.answer_shape = "list_values"
    elif contract.aggregation in {"sum", "count", "avg"}:
        contract.answer_shape = "scalar"

    return contract


def _record(target: Dict[str, List[str]], key: str, value: str) -> None:
    target.setdefault(key, []).append(value)


def _metric_conflicts(requested: Set[str], actual: Set[str]) -> Set[str]:
    if not requested or not actual:
        return set()
    allowed = set(requested)
    # Demand can be represented by current/future/regional demand classes, but
    # should not be confused with sales, inventory, autonomous, etc.
    if "demand" in requested:
        allowed.add("demand")
    return {metric for metric in actual if metric not in allowed}


def compare_contracts(question_contract: QueryContract, query_contract: QueryContract) -> ContractComparison:
    score = 0.0
    reasons: List[str] = []
    matched: Dict[str, List[str]] = {}
    missing: Dict[str, List[str]] = {}
    conflicts: Dict[str, List[str]] = {}

    requested_metrics = set(question_contract.metrics)
    actual_metrics = set(query_contract.metrics)
    metric_matches = requested_metrics & actual_metrics
    for metric in sorted(metric_matches):
        score += 1.4
        reasons.append(f"contract_metric_match:{metric}")
        _record(matched, "metrics", metric)
    for metric in sorted(requested_metrics - actual_metrics):
        score -= 2.0
        reasons.append(f"contract_metric_missing:{metric}")
        _record(missing, "metrics", metric)
    for metric in sorted(_metric_conflicts(requested_metrics, actual_metrics)):
        score -= 1.8
        reasons.append(f"contract_metric_conflict:{metric}")
        _record(conflicts, "metrics", metric)

    if question_contract.aggregation:
        if question_contract.aggregation == query_contract.aggregation:
            score += 1.3
            reasons.append(f"contract_aggregation_match:{question_contract.aggregation}")
            _record(matched, "aggregation", question_contract.aggregation)
        elif question_contract.aggregation == "rank" and query_contract.answer_shape == "ranked_one":
            score += 1.0
            reasons.append("contract_aggregation_match:rank")
            _record(matched, "aggregation", "rank")
        elif query_contract.aggregation:
            score -= 1.7
            reasons.append(f"contract_aggregation_conflict:{query_contract.aggregation}")
            _record(conflicts, "aggregation", query_contract.aggregation)
        else:
            score -= 1.0
            reasons.append(f"contract_aggregation_missing:{question_contract.aggregation}")
            _record(missing, "aggregation", question_contract.aggregation)

    for scope in sorted(question_contract.scopes):
        if scope in query_contract.scopes:
            score += 1.2
            reasons.append(f"contract_scope_match:{scope}")
            _record(matched, "scopes", scope)
        else:
            score -= 1.9
            reasons.append(f"contract_scope_missing:{scope}")
            _record(missing, "scopes", scope)

    for dimension in sorted(question_contract.dimensions):
        if dimension in query_contract.dimensions:
            score += 0.85
            reasons.append(f"contract_dimension_match:{dimension}")
            _record(matched, "dimensions", dimension)
        else:
            score -= 1.25
            reasons.append(f"contract_dimension_missing:{dimension}")
            _record(missing, "dimensions", dimension)

    incompatible_filters = {
        "actual": "forecast",
        "forecast": "actual",
        "shortage_yes": "shortage_no",
        "shortage_no": "shortage_yes",
    }
    for flt in sorted(question_contract.filters):
        opposite = incompatible_filters.get(flt)
        if flt in query_contract.filters:
            score += 1.0
            reasons.append(f"contract_filter_match:{flt}")
            _record(matched, "filters", flt)
        elif opposite and opposite in query_contract.filters:
            score -= 2.0
            reasons.append(f"contract_filter_conflict:{opposite}")
            _record(conflicts, "filters", opposite)
        elif flt in {"actual", "forecast", "bl1", "bl2", "option1", "option2", "option3"}:
            score -= 0.9
            reasons.append(f"contract_filter_missing:{flt}")
            _record(missing, "filters", flt)

    if question_contract.answer_shape:
        if question_contract.answer_shape == query_contract.answer_shape:
            score += 0.7
            reasons.append(f"contract_shape_match:{question_contract.answer_shape}")
            _record(matched, "answer_shape", question_contract.answer_shape)
        elif question_contract.answer_shape == "grouped_table" and question_contract.dimensions & query_contract.dimensions:
            score += 0.35
            reasons.append("contract_shape_dimension_compatible")
        elif query_contract.answer_shape:
            score -= 0.6
            reasons.append(f"contract_shape_conflict:{query_contract.answer_shape}")
            _record(conflicts, "answer_shape", query_contract.answer_shape)

    return ContractComparison(score=float(score), reasons=reasons, matched=matched, missing=missing, conflicts=conflicts)
