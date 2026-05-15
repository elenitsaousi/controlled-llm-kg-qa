from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ranking.feature_extraction import extract_query_plan
from validation.semantic import semantic_judge_report


CRITICAL_AXES = (
    "aggregation",
    "answer_shape",
    "time_dimension",
    "dimensions",
    "filters",
)


def _answer_shape(plan: Dict[str, object]) -> str:
    query_types = set(plan.get("query_types") or [])
    aggregations = set(plan.get("aggregations") or [])
    if "ranking" in query_types:
        return "top_or_ranked"
    if aggregations and "grouped" in query_types:
        return "grouped_summary"
    if aggregations:
        return "summary"
    return "raw_values"


def _time_dimension(dimensions: Iterable[str]) -> str:
    dims = set(dimensions)
    for dim in ("quarter", "month", "year"):
        if dim in dims:
            return dim
    if "time_period" in dims:
        return "period"
    return "none"


def _aggregation(plan: Dict[str, object]) -> str:
    aggregations = list(plan.get("aggregations") or [])
    if not aggregations:
        return "NONE"
    return "+".join(sorted(str(a) for a in aggregations))


def _dimension_hints(plan: Dict[str, object], semantic_dimensions: Iterable[str]) -> Tuple[str, ...]:
    dimensions = {str(x) for x in semantic_dimensions}
    predicates = {str(x).lower() for x in plan.get("predicates") or []}
    vars_seen = {
        str(x).lower()
        for field in ("group_by_vars", "select_vars")
        for x in plan.get(field) or []
    }
    if "analyzestechnologycategory" in predicates:
        dimensions.add("technology")
    if any("vehicle" in pred for pred in predicates):
        dimensions.add("vehicle_type")
    if any("region" in pred for pred in predicates):
        dimensions.add("region")
    if any("responsetype" in pred for pred in predicates):
        dimensions.add("response_type")
    if "fortimeperiod" in predicates or any(
        token in vars_seen for token in {"period", "periodlabel", "timeperiod"}
    ):
        dimensions.add("time_period")
    for token, dim in (
        ("quarter", "quarter"),
        ("quarterlabel", "quarter"),
        ("month", "month"),
        ("monthlabel", "month"),
        ("year", "year"),
    ):
        if token in vars_seen:
            dimensions.add(dim)
    return tuple(sorted(dimensions))


def plan_signature(question: str, query: str, schema_dict: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    plan = extract_query_plan(query, schema_dict)
    semantic = semantic_judge_report(question, query)
    dimensions = _dimension_hints(plan, semantic.get("candidate_dimensions", []))
    filters = tuple(sorted(str(x) for x in semantic.get("candidate_filters", [])))
    origins = tuple(sorted(str(x) for x in semantic.get("candidate_origins", [])))
    return {
        "aggregation": _aggregation(plan),
        "answer_shape": _answer_shape(plan),
        "time_dimension": _time_dimension(dimensions),
        "dimensions": dimensions,
        "filters": filters,
        "origins": origins,
        "plan": plan,
        "semantic": semantic,
    }


def _signature_key(signature: Dict[str, object]) -> Tuple[object, ...]:
    return tuple(signature.get(axis) for axis in CRITICAL_AXES) + (signature.get("origins"),)


def _display_name(value: str) -> str:
    return value.replace("_", " ")


def _describe_signature(signature: Dict[str, object]) -> str:
    aggregation = str(signature["aggregation"])
    shape = str(signature["answer_shape"])
    dimensions = tuple(signature["dimensions"])
    filters = tuple(signature["filters"])

    if shape == "raw_values":
        prefix = "Raw values"
    elif aggregation == "NONE":
        prefix = "Grouped result"
    elif shape == "top_or_ranked":
        prefix = "Top-ranked result"
    else:
        prefix = f"{aggregation} summary"

    parts = [prefix]
    if dimensions:
        parts.append("by " + " and ".join(_display_name(x) for x in dimensions))
    if filters:
        parts.append("filtered to " + " and ".join(_display_name(x) for x in filters))
    return " ".join(parts)


def _conflicts(signatures: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    conflicts: List[Dict[str, object]] = []
    for axis in CRITICAL_AXES:
        values = sorted({signature.get(axis) for signature in signatures}, key=str)
        if len(values) > 1:
            conflicts.append({"axis": axis, "values": values})
    return conflicts


def _meaningful_conflicts(conflicts: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [c for c in conflicts if c.get("axis") in {"aggregation", "answer_shape", "time_dimension", "dimensions"}]


def build_clarification_payload(
    question: str,
    ranked_candidates: Sequence[Dict[str, object]],
    schema_dict: Optional[Dict[str, object]] = None,
    max_candidates: int = 6,
    max_options: int = 3,
) -> Optional[Dict[str, object]]:
    if len(ranked_candidates) < 2:
        return None

    rows: List[Dict[str, object]] = []
    for rank, candidate in enumerate(ranked_candidates[:max_candidates], start=1):
        query = str(candidate.get("query", "") or "").strip()
        if not query:
            continue
        signature = plan_signature(question, query, schema_dict)
        rows.append(
            {
                "rank": rank,
                "candidate": candidate,
                "query": query,
                "signature": signature,
                "key": _signature_key(signature),
            }
        )
    if len(rows) < 2:
        return None

    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["key"]].append(row)
    if len(grouped) < 2:
        return None

    representatives = [members[0] for members in grouped.values()]
    representatives.sort(key=lambda row: int(row["rank"]))
    signatures = [row["signature"] for row in representatives]
    conflicts = _conflicts(signatures)
    meaningful = _meaningful_conflicts(conflicts)
    if not meaningful:
        return None

    # Avoid asking on one-off noise: keep only cases where at least two plans are
    # present in the top three candidates or the disagreement is aggregation-level.
    top_three_keys = [row["key"] for row in rows[:3]]
    top_three_plan_count = len(set(top_three_keys))
    has_aggregation_conflict = any(c["axis"] == "aggregation" for c in meaningful)
    if top_three_plan_count < 2 and not has_aggregation_conflict:
        return None

    options: List[Dict[str, object]] = []
    for index, row in enumerate(representatives[:max_options], start=1):
        signature = row["signature"]
        options.append(
            {
                "id": f"plan_{index}",
                "label": _describe_signature(signature),
                "query": row["query"],
                "candidate_rank": row["rank"],
                "support": len(grouped[row["key"]]),
                "signature": {
                    axis: signature.get(axis)
                    for axis in ("aggregation", "answer_shape", "time_dimension", "dimensions", "filters", "origins")
                },
            }
        )

    conflict_axes = ", ".join(_display_name(str(c["axis"])) for c in meaningful)
    return {
        "needs_clarification": True,
        "reason": f"Candidates disagree on {conflict_axes}.",
        "question": "Which interpretation matches what you want?",
        "options": options,
        "conflicts": conflicts,
        "plan_cluster_count": len(grouped),
        "top_plan_support": Counter(row["key"] for row in rows).most_common(1)[0][1],
    }
