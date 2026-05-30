from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ranking.feature_extraction import extract_query_plan
from validation.semantic import question_intent_report, semantic_judge_report


CRITICAL_AXES = (
    "aggregation",
    "answer_shape",
    "time_dimension",
    "dimensions",
    "filters",
    "origins",
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
    if dimensions & {"quarter", "month", "year"}:
        dimensions.discard("time_period")
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
    return tuple(signature.get(axis) for axis in CRITICAL_AXES)


def _display_name(value: str) -> str:
    names = {
        "technology": "technology",
        "vehicle_type": "vehicle type",
        "region": "region",
        "response_type": "response type",
        "time_period": "time",
        "quarter": "quarter",
        "month": "month",
        "year": "year",
        "future_demand": "future demand",
        "actual_data": "actual data",
        "forecast_data": "forecast data",
    }
    return names.get(value, value.replace("_", " "))


def _metric_phrase(question: str, signature: Dict[str, object]) -> str:
    q = (question or "").lower()
    if "shortage" in q:
        return "shortage response"
    if "cancellation response" in q or "response type" in q:
        return "cancellation response"
    if "future" in q and "demand" in q and ("percent" in q or "change" in q):
        return "future-demand percentage"
    if "percentage" in q or "percent" in q:
        return "percentage"
    if "participant" in q:
        return "participant count"
    if "units" in q or "sales" in q:
        return "vehicle units"
    if "demand" in q:
        return "demand"
    return "values"


def _dimension_phrase(dimensions: Tuple[str, ...]) -> str:
    if not dimensions:
        return ""
    specific_time = next((d for d in ("quarter", "month", "year") if d in dimensions), None)
    non_time = [d for d in dimensions if d not in {"time_period", "quarter", "month", "year"}]
    parts = [_display_name(d) for d in non_time]
    if specific_time:
        parts.append(specific_time)
    elif "time_period" in dimensions:
        parts.append("time")
    return " and ".join(parts)


def _display_dimensions(question: str, signature: Dict[str, object]) -> Tuple[str, ...]:
    intent = question_intent_report(question)
    explicit_dims = tuple(intent.get("dimensions") or [])
    candidate_dims = tuple(signature["dimensions"])
    if explicit_dims:
        return explicit_dims
    # survey origin is often an implementation detail; do not foreground it
    # unless the user asked for it explicitly.
    return tuple(dim for dim in candidate_dims if dim != "survey_origin")


def _describe_signature(question: str, signature: Dict[str, object]) -> str:
    aggregation = str(signature["aggregation"])
    shape = str(signature["answer_shape"])
    dimensions = _display_dimensions(question, signature)
    metric = _metric_phrase(question, signature)
    dimension_text = _dimension_phrase(dimensions)

    if shape == "raw_values":
        prefix = f"Individual {metric} observations"
    elif shape == "top_or_ranked":
        prefix = f"Top {metric}"
    elif aggregation == "AVG":
        prefix = f"Average {metric}"
    elif aggregation == "SUM":
        prefix = f"Total {metric}"
    elif aggregation == "COUNT":
        prefix = f"Count of {metric}"
    elif aggregation == "MAX":
        prefix = f"Maximum {metric}"
    elif aggregation == "MIN":
        prefix = f"Minimum {metric}"
    elif aggregation == "NONE":
        prefix = f"Grouped {metric}"
    else:
        prefix = f"{aggregation} {metric}"

    parts = [prefix]
    if dimension_text:
        if "time" == dimension_text:
            parts.append("over time")
        elif dimension_text.endswith(" and time"):
            parts.append("by " + dimension_text[:-9] + " over time")
        else:
            parts.append("by " + dimension_text)
    return " ".join(parts)


def _conflicts(signatures: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    conflicts: List[Dict[str, object]] = []
    for axis in CRITICAL_AXES:
        values = sorted({signature.get(axis) for signature in signatures}, key=str)
        if len(values) > 1:
            conflicts.append({"axis": axis, "values": values})
    return conflicts


def _meaningful_conflicts(conflicts: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        c
        for c in conflicts
        if c.get("axis")
        in {"aggregation", "answer_shape", "time_dimension", "dimensions", "filters", "origins"}
    ]


def _conflict_is_resolved_by_question(intent: Dict[str, object], conflict: Dict[str, object]) -> bool:
    axis = str(conflict.get("axis"))
    values = set(conflict.get("values") or [])
    if axis == "aggregation":
        # An explicit request for individual/listed values already resolves
        # the important distinction between raw observations and summaries.
        if intent.get("answer_shape") == "raw_values":
            return True
        if not intent.get("aggregation_explicit"):
            return False
        expected = intent.get("aggregation")
        if expected is None:
            return False
        if expected == "MAX_OR_TOP":
            return "MAX" in values or "NONE" in values
        if expected == "MIN_OR_BOTTOM":
            return "MIN" in values or "NONE" in values
        return expected in values
    if axis == "answer_shape":
        expected = intent.get("answer_shape")
        return expected is not None and (
            expected == "raw_values" or bool(intent.get("aggregation_explicit"))
        ) and expected in values
    if axis == "time_dimension":
        expected = intent.get("time_dimension")
        if intent.get("answer_shape") == "raw_values" and expected is None:
            # For explicit list/raw-value requests, an unasked extra time
            # breakdown is a candidate defect rather than a user ambiguity.
            return True
        return expected is not None and expected in values
    if axis == "dimensions":
        expected_dims = set(intent.get("dimensions") or [])
        if not expected_dims:
            return False
        return any(expected_dims <= set(value or ()) for value in values)
    if axis == "filters":
        expected_filters = set(intent.get("filters") or [])
        return bool(expected_filters) and any(expected_filters <= set(value or ()) for value in values)
    if axis == "origins":
        expected_origins = set(intent.get("origins") or [])
        expected_dims = set(intent.get("dimensions") or [])
        if intent.get("answer_shape") == "raw_values" and not expected_origins:
            # For explicit raw/list requests, an unasked origin split is an
            # over-specific candidate variant, not a user-facing ambiguity.
            return True
        if "survey_origin" in expected_dims and not expected_origins:
            # When the user asks for a breakdown by survey origin, candidates
            # that enumerate all origins explicitly are equivalent to queries
            # that leave the same full-domain traversal implicit.
            return True
        return bool(expected_origins) and any(expected_origins <= set(value or ()) for value in values)
    return False


def _drop_resolved_comparison_conflicts(
    intent: Dict[str, object],
    conflicts: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not intent.get("baseline_comparison"):
        return list(conflicts)
    # For explicit BL1/BL2 comparison questions, SUM vs AVG alternatives are
    # implementation noise. The question already asks for the baseline values.
    return [
        conflict
        for conflict in conflicts
        if conflict.get("axis") not in {"aggregation", "answer_shape"}
    ]


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

    # If runtime profiling is available, do not ask users to choose between
    # empty plans when at least two non-empty interpretations are available.
    # Unknown execution state is kept for callers that have not profiled yet.
    nonempty_rows = [
        row for row in rows
        if row["candidate"].get("execution_has_rows") is True
    ]
    if len(nonempty_rows) >= 2:
        rows = nonempty_rows

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
    intent = question_intent_report(question)
    meaningful = _drop_resolved_comparison_conflicts(intent, meaningful)
    meaningful = [
        conflict
        for conflict in meaningful
        if not _conflict_is_resolved_by_question(intent, conflict)
    ]
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
    seen_labels = set()
    for index, row in enumerate(representatives[:max_options], start=1):
        signature = row["signature"]
        label = _describe_signature(question, signature)
        label_key = label.strip().lower()
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)
        options.append(
            {
                "id": f"plan_{index}",
                "label": label,
                "query": row["query"],
                "candidate_rank": row["rank"],
                "support": len(grouped[row["key"]]),
                "row_count": row["candidate"].get("execution_row_count"),
                "preview": row["candidate"].get("answer_preview"),
                "preview_rows": row["candidate"].get("preview_rows") or [],
                "signature": {
                    axis: signature.get(axis)
                    for axis in ("aggregation", "answer_shape", "time_dimension", "dimensions", "filters", "origins")
                },
            }
        )
    if len(options) < 2:
        return None

    conflict_axes = ", ".join(_display_name(str(c["axis"])) for c in meaningful)
    return {
        "needs_clarification": True,
        "reason": f"Candidates disagree on {conflict_axes}.",
        "question": "Which interpretation matches what you want?",
        "options": options,
        "conflicts": conflicts,
        "resolved_intent": intent,
        "plan_cluster_count": len(grouped),
        "top_plan_support": Counter(row["key"] for row in rows).most_common(1)[0][1],
    }
