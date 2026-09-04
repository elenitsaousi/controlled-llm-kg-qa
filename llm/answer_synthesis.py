import math
import re
from typing import Dict, Iterable, List, Optional, Tuple


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _format_row(row: Dict[str, object]) -> str:
    return ", ".join(f"{k}={v}" for k, v in row.items())


def _row_get(row: Dict[str, object], *keys: str) -> Optional[object]:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        value = lowered.get(str(key).lower())
        if value is not None and str(value) != "":
            return value
    return None


def _has_any_key(rows: List[Dict[str, object]], keys: Iterable[str]) -> bool:
    wanted = {key.lower() for key in keys}
    for row in rows:
        if any(str(key).lower() in wanted for key in row):
            return True
    return False


def _clean_value(value: object) -> str:
    text = str(value)
    if "/" in text or "#" in text:
        text = re.split(r"[/#]", text)[-1]
    text = text.replace("%3C%3D", "<=").replace("%3C", "<")
    text = text.replace("_to_lt", " to <").replace("_to_", " to ")
    text = text.replace("_or_greater", " or greater").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _format_number(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return _clean_value(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _period_sort_key(label: object) -> Tuple[int, int, str]:
    text = _clean_value(label)
    month_match = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s+(\d{4})", text, re.I)
    if month_match:
        months = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        return (int(month_match.group(2)), months[month_match.group(1).lower()], text)
    quarter_match = re.search(r"\bQ([1-4])\s*(\d{4})\b", text, re.I)
    if quarter_match:
        return (int(quarter_match.group(2)), int(quarter_match.group(1)) * 3, text)
    year_match = re.search(r"\b(\d{4})\b", text)
    if year_match:
        return (int(year_match.group(1)), 0, text)
    return (0, 0, text)


def _numeric_rows(
    rows: List[Dict[str, object]], metric_keys: Iterable[str]
) -> List[Tuple[Dict[str, object], str, float]]:
    scored = []
    for row in rows:
        for key in metric_keys:
            raw = _row_get(row, key)
            value = _to_float(raw)
            if value is not None:
                scored.append((row, key, value))
                break
    return scored


def _humanize_key(key: str) -> str:
    key = re.sub(r"[_-]+", " ", str(key)).strip()
    key = re.sub(r"(?<!^)(?=[A-Z])", " ", key)
    return re.sub(r"\s+", " ", key).lower()


def _format_generic_answer(rows: List[Dict[str, object]], max_rows: int = 8) -> str:
    if not rows:
        return "No results were found for the given query."

    if len(rows) == 1 and len(rows[0]) == 1:
        value = next(iter(rows[0].values()))
        return f"The result is {value}."

    parts = []
    for row in rows[:max_rows]:
        clean_items = [
            f"{_humanize_key(str(key))}: {value}"
            for key, value in row.items()
            if value is not None and str(value) != ""
        ]
        if clean_items:
            parts.append(", ".join(clean_items))

    if not parts:
        return f"The query returned {len(rows)} row(s), but no displayable values."

    prefix = f"The query returned {len(rows)} row(s)."
    if len(rows) > max_rows:
        return prefix + " First results: " + "; ".join(parts) + "."
    return prefix + " Results: " + "; ".join(parts) + "."


def _format_generic_ranking_answer(
    rows: List[Dict[str, object]],
    *,
    question: str,
) -> str:
    if not rows or not _is_ranking_question(question):
        return ""

    metric_name_pattern = re.compile(
        r"(avg|average|mean|sum|total|count|percentage|percent|pct|units|sold|"
        r"demand|change|participant|company|entry|value)",
        re.I,
    )
    dimension_name_pattern = re.compile(r"(year|month|quarter|label|name|type|category|region)", re.I)
    numeric_candidates: List[Tuple[Dict[str, object], str, float]] = []
    for row in rows:
        row_candidates: List[Tuple[Dict[str, object], str, float, int]] = []
        for key, value in row.items():
            key_text = str(key)
            number = _to_float(value)
            if number is not None:
                priority = 2 if metric_name_pattern.search(key_text) else 0
                if dimension_name_pattern.search(key_text):
                    priority -= 1
                row_candidates.append((row, key_text, number, priority))
        if row_candidates:
            row_candidates.sort(key=lambda item: item[3], reverse=True)
            row_, key_, number_, _ = row_candidates[0]
            numeric_candidates.append((row_, key_, number_))
    if not numeric_candidates:
        return ""

    q = str(question or "").lower()
    choose_min = bool(re.search(r"\b(lowest|bottom|smallest|min|minimum|least|worst)\b", q))
    chosen_row, metric_key, chosen_value = (
        min(numeric_candidates, key=lambda item: item[2])
        if choose_min
        else max(numeric_candidates, key=lambda item: item[2])
    )

    label_parts = []
    for key, value in chosen_row.items():
        if str(key) == metric_key:
            continue
        if value is None or str(value) == "":
            continue
        label_parts.append(f"{_humanize_key(str(key))}: {_clean_value(value)}")
    label = ", ".join(label_parts) or "the selected group"
    direction = "lowest" if choose_min else "highest"
    return (
        f"The query returned {len(rows)} grouped row(s). "
        f"The {direction} returned {_humanize_key(metric_key)} is for {label} "
        f"with {_format_number(chosen_value)}."
    )


def _is_ranking_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(highest|lowest|top|bottom|largest|smallest|max|maximum|min|minimum|best|worst|most|least)\b",
            str(question or "").lower(),
        )
    )


def _is_grouped_query(query: str) -> bool:
    return bool(re.search(r"\bGROUP\s+BY\b", str(query or ""), flags=re.I))


def _format_grouped_breakdown_answer(
    rows: List[Dict[str, object]],
    *,
    label: str,
    max_rows: int = 8,
) -> str:
    if not rows:
        return ""
    parts = []
    for row in rows[:max_rows]:
        clean_items = [
            f"{_humanize_key(str(key))}: {_clean_value(value)}"
            for key, value in row.items()
            if value is not None and str(value) != ""
        ]
        if clean_items:
            parts.append(", ".join(clean_items))
    if not parts:
        return f"{label} returned {len(rows)} grouped row(s)."
    prefix = f"{label} returned {len(rows)} grouped row(s)."
    if len(rows) > max_rows:
        return prefix + " First groups: " + "; ".join(parts) + "."
    return prefix + " Groups: " + "; ".join(parts) + "."


def _format_total_demand_by_region(
    rows: List[Dict[str, object]],
    *,
    question: str = "",
) -> str:
    if not _has_any_key(rows, ["regionName", "regions"]) or not _has_any_key(
        rows,
        [
            "totalDemand",
            "oemDemand",
            "tier1Demand",
            "semiconductorDemand",
            "avgDemand",
        ],
    ):
        return ""

    scored = _numeric_rows(
        rows,
        [
            "totalDemand",
            "oemDemand",
            "tier1Demand",
            "semiconductorDemand",
            "avgDemand",
        ],
    )
    if not scored:
        return ""

    if not _is_ranking_question(question):
        return _format_grouped_breakdown_answer(
            rows,
            label="Regional demand",
        )

    choose_min = bool(re.search(r"\b(lowest|bottom|smallest|min|minimum|least|weakest)\b", str(question or "").lower()))
    top_row, metric_key, top_value = (
        min(scored, key=lambda item: item[2]) if choose_min else max(scored, key=lambda item: item[2])
    )
    region = _clean_value(_row_get(top_row, "regionName", "regions"))
    survey_type = _row_get(top_row, "surveyType", "originType")
    survey_text = f" for {_clean_value(survey_type)}" if survey_type else ""
    metric_text = _humanize_key(metric_key)
    direction = "lowest" if choose_min else "highest"
    summary = (
        f"Regional demand returned {len(rows)} row(s). "
        f"The {direction} {metric_text}{survey_text} is in {region} "
        f"with {_format_number(top_value)}."
    )

    if len(scored) > 1:
        other_row, _, other_value = (max(scored, key=lambda item: item[2]) if choose_min else min(scored, key=lambda item: item[2]))
        other_region = _clean_value(_row_get(other_row, "regionName", "regions"))
        other_label = "highest" if choose_min else "lowest"
        summary += f" The {other_label} returned value is {other_region} with {_format_number(other_value)}."

    return summary


def _format_future_demand_by_tech_quarter(rows: List[Dict[str, object]]) -> str:
    if not _has_any_key(rows, ["techLabel", "technologyCategory", "vehicleType"]):
        return ""
    if not _has_any_key(rows, ["quarterLabel", "quarter", "timePeriod", "periodLabel"]):
        return ""
    if not _has_any_key(
        rows,
        [
            "totalFutureChange",
            "avgFutureChange",
            "avgChange",
            "avgPctChange",
            "avgPercentage",
            "pct",
            "Option1",
            "Option2",
            "Option3",
        ],
    ):
        return ""

    metric_keys = [
        "totalFutureChange",
        "avgFutureChange",
        "avgChange",
        "avgPctChange",
        "avgPercentage",
        "pct",
        "Option1",
        "Option2",
        "Option3",
    ]
    scored = _numeric_rows(rows, metric_keys)
    if not scored:
        return ""

    top_row, metric_key, top_value = max(scored, key=lambda item: item[2])
    low_row, _, low_value = min(scored, key=lambda item: item[2])
    category = _clean_value(
        _row_get(top_row, "techLabel", "technologyCategory", "vehicleType")
    )
    quarter = _clean_value(_row_get(top_row, "quarterLabel", "quarter", "timePeriod", "periodLabel"))
    low_category = _clean_value(
        _row_get(low_row, "techLabel", "technologyCategory", "vehicleType")
    )
    low_quarter = _clean_value(_row_get(low_row, "quarterLabel", "quarter", "timePeriod", "periodLabel"))
    metric_text = _humanize_key(metric_key)
    if metric_text in {"avg percentage", "avg future change", "avg change", "avg pct change"}:
        metric_text = "average future-demand percentage change"
    elif metric_text == "total future change":
        metric_text = "total future-demand change"
    return (
        f"Future-demand results returned {len(rows)} grouped row(s). "
        f"The highest {metric_text} is {category} in {quarter} "
        f"with {_format_number(top_value)}. "
        f"The lowest is {low_category} in {low_quarter} "
        f"with {_format_number(low_value)}."
    )


def _format_autonomous_driving(rows: List[Dict[str, object]], query: str = "", question: str = "") -> str:
    if not _has_any_key(rows, ["vehicleType", "vehicle", "saeLevel", "sae"]):
        return ""
    if not _has_any_key(rows, ["percentage", "maxPercentage", "avgPercentage", "avgPct"]):
        return ""

    if _is_grouped_query(query) and not _is_ranking_question(question):
        scored = _numeric_rows(rows, ["maxPercentage", "percentage", "avgPercentage", "avgPct"])
        if scored:

            def _describe_group(row: Dict[str, object]) -> str:
                # Build the description from whichever of vehicle type/SAE
                # level/year are actually present in this row -- the
                # grouping can be any subset of the three (the query is no
                # longer hardcoded to always group by SAE level and year).
                # _clean_value(None) stringifies to the literal text "None"
                # (truthy!), so check the raw _row_get result before
                # cleaning it, not the cleaned string.
                parts = []
                vehicle_raw = _row_get(row, "vehicleType", "vehicle")
                if vehicle_raw is not None:
                    parts.append(_clean_value(vehicle_raw))
                sae_raw = _row_get(row, "saeLevel", "sae", "saeLabel")
                if sae_raw is not None:
                    parts.append(_clean_value(sae_raw))
                year_raw = _row_get(row, "year")
                if year_raw is not None:
                    parts.append(f"year {_clean_value(year_raw)}")
                return ", ".join(parts) if parts else "the selected group"

            top_row, metric_key, top_value = max(scored, key=lambda item: item[2])
            low_row, _, low_value = min(scored, key=lambda item: item[2])
            return (
                f"Autonomous-driving development returned {len(rows)} grouped row(s). "
                f"The highest returned {_humanize_key(metric_key)} is {_describe_group(top_row)} "
                f"with {_format_number(top_value)}. "
                f"The lowest is {_describe_group(low_row)} with {_format_number(low_value)}."
            )
        return _format_grouped_breakdown_answer(rows, label="Autonomous-driving development")

    scored = _numeric_rows(rows, ["maxPercentage", "percentage", "avgPercentage", "avgPct"])
    if not scored:
        return ""

    top_row, metric_key, top_value = max(scored, key=lambda item: item[2])
    vehicle = _clean_value(_row_get(top_row, "vehicleType", "vehicle")) or "autonomous-driving"
    sae = _row_get(top_row, "saeLevel", "sae", "saeLabel")
    year = _row_get(top_row, "year")
    qualifiers = []
    if sae is not None:
        qualifiers.append(f"SAE level {_clean_value(sae)}")
    if year is not None:
        qualifiers.append(f"year {_clean_value(year)}")
    qualifier_text = " (" + ", ".join(qualifiers) + ")" if qualifiers else ""
    return (
        f"Autonomous-driving development returned {len(rows)} row(s). "
        f"The highest returned {_humanize_key(metric_key)} is {vehicle}{qualifier_text} "
        f"with {_format_number(top_value)}."
    )


def _format_order_cancellation(rows: List[Dict[str, object]]) -> str:
    if not _has_any_key(rows, ["responseType"]):
        return ""
    if not _has_any_key(
        rows,
        [
            "participantCount",
            "participants",
            "responses",
            "responseCount",
            "increaseCount",
            "decreaseCount",
            "stableCount",
            "totalResponses",
        ],
    ):
        return ""

    scored = _numeric_rows(
        rows,
        [
            "participantCount",
            "participants",
            "responses",
            "responseCount",
            "totalResponses",
            "increaseCount",
            "decreaseCount",
            "stableCount",
        ],
    )
    if not scored:
        return ""

    totals: Dict[str, float] = {}
    for row, metric_key_, value in scored:
        response = _clean_value(_row_get(row, "responseType"))
        if response:
            totals[response] = totals.get(response, 0.0) + value
    if totals:
        ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        leader, leader_value = ordered[0]
        evidence = "; ".join(f"{label}: {_format_number(value)}" for label, value in ordered)
        tied = [label for label, value in ordered if value == leader_value]
        if len(tied) > 1:
            return (
                f"Order-cancellation evidence returned {len(rows)} grouped row(s). "
                f"No single response type dominates: {', '.join(tied)} are tied at {_format_number(leader_value)} participants. "
                f"Totals by response type: {evidence}."
            )
        return (
            f"Order-cancellation evidence returned {len(rows)} grouped row(s). "
            f"The strongest overall response signal is {leader} with {_format_number(leader_value)} participants. "
            f"Totals by response type: {evidence}."
        )

    top_row, metric_key, top_value = max(scored, key=lambda item: item[2])
    category = _row_get(top_row, "technologyCategory", "techLabel", "tech")
    response_type = _clean_value(_row_get(top_row, "responseType"))
    category_text = f" for {_clean_value(category)}" if category is not None else ""
    return (
        f"Order-cancellation results returned {len(rows)} row(s). "
        f"The largest returned {_humanize_key(metric_key)} is {response_type}{category_text} "
        f"with {_format_number(top_value)}."
    )


def _format_inventory_status(rows: List[Dict[str, object]]) -> str:
    if _has_any_key(rows, ["coverageLimitation"]):
        return str(_row_get(rows[0], "coverageLimitation") or "").strip()
    if _has_any_key(rows, ["targetStatus"]):
        counts: Dict[str, float] = {}
        for row in rows:
            label = _clean_value(_row_get(row, "targetStatus"))
            count = _to_float(_row_get(row, "entryCount")) or 0.0
            if label:
                counts[label] = counts.get(label, 0.0) + count
        if counts:
            evidence = "; ".join(f"{label}: {_format_number(value)}" for label, value in sorted(counts.items()))
            return f"Inventory target evidence returned {len(rows)} grouped row(s). Target status distribution: {evidence}."
    if not _has_any_key(rows, ["trend"]):
        return ""
    counts: Dict[str, float] = {}
    for row in rows:
        label = _clean_value(_row_get(row, "trend"))
        count = _to_float(_row_get(row, "entryCount")) or 0.0
        if label:
            counts[label] = counts.get(label, 0.0) + count
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    leader, leader_value = ordered[0]
    evidence = "; ".join(f"{label}: {_format_number(value)}" for label, value in ordered)
    if leader.lower() == "stable" and len(ordered) == 1:
        conclusion = "The inventory signal is stable."
    elif leader.lower() == "stable":
        conclusion = "The inventory signal is partly stable, but not exclusively stable."
    else:
        conclusion = f"The inventory signal is not mainly stable; the largest returned trend is {leader}."
    return (
        f"{conclusion} The graph returned {len(rows)} grouped inventory rows. "
        f"Trend distribution: {evidence}."
    )


def _format_shortage_status(rows: List[Dict[str, object]]) -> str:
    if not _has_any_key(rows, ["shortageStatus"]) or not _has_any_key(rows, ["companyCount"]):
        return ""
    by_status: Dict[str, float] = {}
    by_group = []
    for row in rows:
        status = _clean_value(_row_get(row, "shortageStatus")).lower()
        count = _to_float(_row_get(row, "companyCount")) or 0.0
        group = _row_get(row, "surveyGroup")
        if status:
            by_status[status] = by_status.get(status, 0.0) + count
        if group:
            by_group.append(f"{_clean_value(group)} {status}: {_format_number(count)}")
    yes_count = by_status.get("yes", 0.0) + by_status.get("true", 0.0)
    no_count = by_status.get("no", 0.0) + by_status.get("false", 0.0)
    conclusion = "Yes, shortage is visible in the graph." if yes_count > 0 else "No shortage signal is visible in the returned graph rows."
    if by_group:
        return f"{conclusion} Evidence by group: {'; '.join(by_group)}."
    return f"{conclusion} Companies with shortage: {_format_number(yes_count)}; without shortage: {_format_number(no_count)}."


def _format_future_demand_split(rows: List[Dict[str, object]], *, question: str = "") -> str:
    if not _has_any_key(rows, ["expectedFutureDemand", "avgPercentageChange", "yearlySales"]):
        return ""
    if _has_any_key(rows, ["view", "dimension"]):
        scored = _numeric_rows(rows, ["expectedFutureDemand", "avgPercentageChange"])
        if not scored:
            return ""
        by_view: Dict[str, Tuple[Dict[str, object], str, float]] = {}
        for row, key, value in scored:
            view = _clean_value(_row_get(row, "view"))
            if view and (view not in by_view or value > by_view[view][2]):
                by_view[view] = (row, key, value)
        parts = []
        for view, (row, key, value) in sorted(by_view.items()):
            dim = _clean_value(_row_get(row, "dimension"))
            quarter = _clean_value(_row_get(row, "quarterLabel"))
            group = _clean_value(_row_get(row, "surveyGroup"))
            suffix = f" in {quarter}" if quarter else ""
            prefix = f"{group} " if group else ""
            parts.append(f"{view}: {prefix}{dim}{suffix} ({_humanize_key(key)} {_format_number(value)})")
        return (
            f"Future-demand split returned {len(rows)} grouped row(s). "
            f"The graph contains separate regional and technology-category views, not a single joint region-by-technology cube. "
            f"Top returned signals: {'; '.join(parts)}."
        )
    if _has_any_key(rows, ["surveyGroup"]) and _has_any_key(rows, ["quarterLabel"]):
        scored = _numeric_rows(rows, ["expectedFutureDemand", "avgPercentageChange"])
        if not scored:
            return ""
        grouped: Dict[str, List[Tuple[str, float]]] = {}
        for row, _key, value in scored:
            group = _clean_value(_row_get(row, "surveyGroup"))
            quarter = _clean_value(_row_get(row, "quarterLabel"))
            if group and quarter:
                grouped.setdefault(group, []).append((quarter, value))
        parts = []
        for group, values in sorted(grouped.items()):
            values.sort(key=lambda item: _period_sort_key(item[0]))
            if len(values) == 1:
                only_q, only_v = values[0]
                parts.append(f"{group}: {_format_number(only_v)} in {only_q}")
                continue
            first_q, first_v = values[0]
            last_q, last_v = values[-1]
            delta = last_v - first_v
            direction = "rising" if delta > 0 else "falling" if delta < 0 else "stable"
            parts.append(
                f"{group}: {direction}, from {_format_number(first_v)} in {first_q} "
                f"to {_format_number(last_v)} in {last_q} (delta {_format_number(delta)})"
            )
        return (
            f"Expected future-demand development returned {len(rows)} grouped row(s). "
            f"{'; '.join(parts)}."
        )
    if _has_any_key(rows, ["vehicleType"]) and _has_any_key(rows, ["yearlySales"]):
        scored = _numeric_rows(rows, ["yearlySales"])
        if not scored:
            return ""
        if _has_any_key(rows, ["year"]):
            # Grouped by year AND vehicle type (e.g. "total vehicles sold
            # each year, grouped by type") -- distinct from the single-
            # dimension vehicle-type "outlook" query below, which has no
            # year column. Without this branch the year was silently
            # dropped and every row got flattened into one "outlook
            # signal" ranking, mixing values from different years together.
            by_year: Dict[str, List[str]] = {}
            for row, _key, value in scored:
                year = _clean_value(_row_get(row, "year"))
                vehicle = _clean_value(_row_get(row, "vehicleType"))
                by_year.setdefault(year, []).append(f"{vehicle}: {_format_number(value)}")
            parts = [f"{year}: {', '.join(items)}" for year, items in sorted(by_year.items())]
            return (
                f"Vehicle sales by year and type returned {len(rows)} grouped row(s). "
                + "; ".join(parts)
                + "."
            )
        top_row, _, top_value = max(scored, key=lambda item: item[2])
        vehicle = _clean_value(_row_get(top_row, "vehicleType"))
        evidence = "; ".join(
            f"{_clean_value(_row_get(row, 'vehicleType'))}: {_format_number(value)}"
            for row, _, value in scored[:5]
        )
        return (
            f"The strongest available vehicle-type demand outlook signal is {vehicle} "
            f"with yearly sales of {_format_number(top_value)}. Evidence: {evidence}."
        )
    if _has_any_key(rows, ["regionName"]):
        scored = _numeric_rows(rows, ["expectedFutureDemand", "avgPercentageChange"])
        if not scored:
            return ""
        return _format_future_region_rows(rows, scored, question=question)
    return ""


def _format_future_region_rows(
    rows: List[Dict[str, object]],
    scored: List[Tuple[Dict[str, object], str, float]],
    *,
    question: str = "",
) -> str:
    choose_min = bool(re.search(r"\b(lowest|bottom|smallest|min|minimum|least|weakest)\b", str(question or "").lower()))
    top_row, metric_key, top_value = (
        min(scored, key=lambda item: item[2]) if choose_min else max(scored, key=lambda item: item[2])
    )
    region_key = "regionName" if _has_any_key(rows, ["regionName"]) else "regions"
    region = _clean_value(_row_get(top_row, region_key))
    group = _row_get(top_row, "surveyGroup")
    group_text = f" for {_clean_value(group)}" if group else ""
    direction = "lowest" if choose_min else "highest"
    evidence = "; ".join(
        f"{_clean_value(_row_get(row, region_key))}: {_format_number(value)}"
        for row, _, value in (sorted(scored, key=lambda item: item[2]) if choose_min else scored)[:5]
    )
    return (
        f"Expected future demand by region returned {len(rows)} grouped row(s). "
        f"The {direction} returned {_humanize_key(metric_key)}{group_text} is {region} with {_format_number(top_value)}. "
        f"Evidence from the returned rows: {evidence}."
    )


def _format_yearly_vehicle_sales(rows: List[Dict[str, object]]) -> str:
    if not _has_any_key(rows, ["year"]) or not _has_any_key(rows, ["unitsSold", "yearlySales"]):
        return ""
    scored = []
    for row in rows:
        year = _clean_value(_row_get(row, "year"))
        value = _to_float(_row_get(row, "unitsSold", "yearlySales"))
        if year and value is not None:
            scored.append((year, value))
    scored.sort(key=lambda item: _period_sort_key(item[0]))
    if len(scored) < 2:
        if scored:
            year, value = scored[0]
            return f"The latest available yearly vehicle-sales value is {_format_number(value)} for {year}."
        return _format_grouped_breakdown_answer(rows, label="Vehicle-sales development")
    first_year, first_value = scored[0]
    last_year, last_value = scored[-1]
    delta = last_value - first_value
    direction = "rising" if delta > 0 else "falling" if delta < 0 else "stable"
    evidence = "; ".join(f"{year}: {_format_number(value)}" for year, value in scored)
    return (
        f"Vehicle-sales development is {direction} over the available yearly data. "
        f"It changes from {_format_number(first_value)} in {first_year} to {_format_number(last_value)} in {last_year} "
        f"(delta {_format_number(delta)}). Evidence: {evidence}."
    )


def _format_current_bl_comparison(rows: List[Dict[str, object]]) -> str:
    if _has_any_key(rows, ["changeBL1", "changeBL2"]):
        row = rows[0]
        bl1 = _row_get(row, "changeBL1", "bl1")
        bl2 = _row_get(row, "changeBL2", "bl2")
        if bl1 is not None and bl2 is not None:
            delta = (_to_float(bl1) or 0.0) - (_to_float(bl2) or 0.0)
            segment = _row_get(row, "marketSegment")
            segment_text = f" for {_clean_value(segment)}" if segment else ""
            relation = "higher than" if delta > 0 else "lower than" if delta < 0 else "equal to"
            return (
                f"Current-demand BL comparison returned {len(rows)} row(s). "
                f"BL1{segment_text} is {_format_number(bl1)} and BL2 is {_format_number(bl2)}; "
                f"BL1 is {relation} BL2 by {_format_number(abs(delta))}."
            )

    if _has_any_key(rows, ["deltaBL1BL2", "delta"]):
        row = rows[0]
        delta = _row_get(row, "deltaBL1BL2", "delta")
        return (
            f"Current-demand BL comparison returned {len(rows)} row(s). "
            f"The returned BL1-BL2 delta is {_format_number(delta)}."
        )

    if not _has_any_key(rows, ["baseline"]) or not _has_any_key(
        rows, ["pct", "totalChange", "avgPct", "avgPctChange"]
    ):
        return ""

    values = {}
    metric_key = ""
    for row in rows:
        baseline = _clean_value(_row_get(row, "baseline"))
        value = _row_get(row, "totalChange", "pct", "avgPct", "avgPctChange")
        number = _to_float(value)
        if baseline and number is not None:
            values[baseline.upper()] = number
            metric_key = "totalChange" if _row_get(row, "totalChange") is not None else "percentageChange"

    if "BL1" not in values or "BL2" not in values:
        return ""

    delta = values["BL1"] - values["BL2"]
    relation = "higher than" if delta > 0 else "lower than" if delta < 0 else "equal to"
    return (
        f"Current-demand BL comparison returned {len(rows)} row(s). "
        f"BL1 {_humanize_key(metric_key)} is {_format_number(values['BL1'])}, "
        f"BL2 is {_format_number(values['BL2'])}; "
        f"BL1 is {relation} BL2 by {_format_number(abs(delta))}."
    )


def _format_vehicle_sales_by_month(
    rows: List[Dict[str, object]],
    *,
    question: str = "",
) -> str:
    if not _has_any_key(rows, ["monthLabel", "month"]):
        return ""
    if not _has_any_key(rows, ["unitsSold", "totalUnitsSold"]):
        return ""

    scored = _numeric_rows(rows, ["totalUnitsSold", "unitsSold"])
    if not scored:
        return ""

    if not _is_ranking_question(question):
        return _format_grouped_breakdown_answer(
            rows,
            label="Vehicle-sales results",
            max_rows=12,
        )

    top_row, metric_key, top_value = max(scored, key=lambda item: item[2])
    low_row, _, low_value = min(scored, key=lambda item: item[2])
    top_month = _clean_value(_row_get(top_row, "monthLabel", "month"))
    low_month = _clean_value(_row_get(low_row, "monthLabel", "month"))
    return (
        f"Vehicle-sales results returned {len(rows)} monthly row(s). "
        f"The highest monthly total is {top_month} "
        f"with {_format_number(top_value)}. "
        f"The lowest monthly total is {low_month} with {_format_number(low_value)}."
    )


def _format_current_demand_time_window(rows: List[Dict[str, object]]) -> str:
    if _has_any_key(rows, ["avgCurrentDemand"]):
        row = rows[0]
        avg_value = _row_get(row, "avgCurrentDemand")
        months_used = _row_get(row, "monthsUsed")
        evidence = _row_get(row, "evidence")
        answer = f"The average current demand over the selected latest monthly window is {_format_number(avg_value)}."
        if months_used is not None:
            answer += f" It used {_format_number(months_used)} month(s)."
        if evidence is not None:
            answer += f" Evidence: {_clean_value(evidence)}."
        return answer
    if _has_any_key(rows, ["monthLabel"]) and _has_any_key(rows, ["currentDemand"]):
        scored = []
        for row in rows:
            month = _clean_value(_row_get(row, "monthLabel"))
            value = _to_float(_row_get(row, "currentDemand"))
            if month and value is not None:
                scored.append((month, value))
        scored.sort(key=lambda item: _period_sort_key(item[0]))
        if not scored:
            return ""
        if len(scored) == 1:
            month, value = scored[0]
            return f"The latest available monthly current demand is {_format_number(value)} in {month}."
        evidence = "; ".join(f"{month}: {_format_number(value)}" for month, value in scored)
        return f"Current demand returned {len(scored)} monthly point(s): {evidence}."
    return ""


def _format_infineon_answer(rows: List[Dict[str, object]], query: str, question: str = "") -> str:
    query_lower = query.lower()

    formatters = []
    if _has_any_key(rows, ["coverageLimitation", "trend", "targetStatus"]):
        answer = _format_inventory_status(rows)
        if answer:
            return answer
    if _has_any_key(rows, ["shortageStatus", "companyCount"]):
        answer = _format_shortage_status(rows)
        if answer:
            return answer
    if _has_any_key(rows, ["expectedFutureDemand", "avgPercentageChange", "yearlySales"]):
        answer = _format_future_demand_split(rows, question=question)
        if answer:
            return answer
    if _has_any_key(rows, ["year"]) and _has_any_key(rows, ["unitsSold", "yearlySales"]):
        answer = _format_yearly_vehicle_sales(rows)
        if answer:
            return answer
    if _has_any_key(rows, ["avgCurrentDemand", "currentDemand"]):
        answer = _format_current_demand_time_window(rows)
        if answer:
            return answer
    if "currentdemandanalysis" in query_lower or "baseline" in query_lower:
        formatters.append(_format_current_bl_comparison)
    if "vehiclesalesobservation" in query_lower or _has_any_key(rows, ["monthLabel"]):
        answer = _format_vehicle_sales_by_month(rows, question=question)
        if answer:
            return answer
    if "ordercancellation" in query_lower or _has_any_key(rows, ["responseType"]):
        formatters.append(_format_order_cancellation)
    if "autonomousdrivingdevelopment" in query_lower or _has_any_key(rows, ["saeLevel", "sae"]):
        answer = _format_autonomous_driving(rows, query=query, question=question)
        if answer:
            return answer
    if "futuredemandanalysis" in query_lower or _has_any_key(rows, ["techLabel", "quarterLabel"]):
        formatters.append(_format_future_demand_by_tech_quarter)
    if "demandforregion" in query_lower or _has_any_key(rows, ["regionName", "regions"]):
        answer = _format_total_demand_by_region(rows, question=question)
        if answer:
            return answer

    for formatter in formatters:
        answer = formatter(rows)
        if answer:
            return answer
    return ""


def _extract_target_from_query(
    query: str, subject_var: str, predicate: str
) -> Optional[str]:
    """
    Very lightweight extraction of explicit FILTER literals from SPARQL.
    Used ONLY to keep answer synthesis aligned with query constraints.
    """
    if not query:
        return None

    # Look for a predicate binding like: ?y :name ?y_name .
    pred_pattern = re.compile(
        rf"\?{re.escape(subject_var)}\s+:{re.escape(predicate)}\s+\?(?P<var>[A-Za-z_][A-Za-z0-9_]*)",
        re.IGNORECASE,
    )
    match = pred_pattern.search(query)
    if not match:
        return None

    var_name = match.group("var")
    # Look for FILTER(?var = 'X')
    filter_pattern = re.compile(
        rf"FILTER\s*\(\s*\?{re.escape(var_name)}\s*=\s*'([^']+)'\s*\)",
        re.IGNORECASE,
    )
    filter_match = filter_pattern.search(query)
    if not filter_match:
        return None

    return filter_match.group(1)


# --------------------------------------------------
# Natural language answer formatting
# --------------------------------------------------

def _format_natural_answer(
    question_id: Optional[str],
    rows: List[Dict[str, object]],
    query: str,
) -> str:
    """
    IMPORTANT:
    This function must NEVER introduce information
    that is not explicitly supported by the executed query.
    """

    # -----------------------------
    # Q1: Suppliers affecting yield
    # -----------------------------
    if question_id == "Q1":
        target_yield = _extract_target_from_query(query, "y", "value")

        pairs = []
        for row in rows:
            supplier = row.get("supplier")
            yield_id = row.get("yield")

            if not supplier or not yield_id:
                continue

            # STRICT FILTER: do not leak other yields
            if target_yield and yield_id != target_yield:
                continue

            pairs.append(f"{supplier} (yield {yield_id})")

        if pairs:
            return "Suppliers affecting yield are: " + ", ".join(pairs) + "."

        return "No suppliers were found for the specified yield."

    # -----------------------------
    # Q2: Average yield by supplier
    # -----------------------------
    if question_id == "Q2":
        parts = []
        for row in rows:
            supplier = row.get("supplier")
            avg_yield = row.get("avg_yield")
            if supplier is not None and avg_yield is not None:
                parts.append(f"{supplier}: average yield {avg_yield}")

        if parts:
            return "Average yield by supplier: " + "; ".join(parts) + "."

        return "No yield statistics were found."

    # -----------------------------
    # Q3: Tools linked to defects
    # -----------------------------
    if question_id == "Q3":
        parts = []
        for row in rows:
            tool = row.get("tool")
            defect = row.get("defect")
            if tool and defect:
                parts.append(f"{tool} linked to defect {defect}")

        if parts:
            return "Tools linked to defects: " + "; ".join(parts) + "."

        return "No tool–defect relations were found."

    # -----------------------------
    # Q4: Suppliers providing materials
    # -----------------------------
    if question_id == "Q4":
        parts = []
        for row in rows:
            supplier = row.get("supplier")
            material = row.get("material")
            if supplier and material:
                parts.append(f"{supplier} supplies {material}")

        if parts:
            return (
                "Suppliers providing lithography materials: "
                + ", ".join(parts)
                + "."
            )

        return "No suppliers were found for the specified materials."

    # -----------------------------
    # Q5: Capacity constraints
    # -----------------------------
    if question_id == "Q5":
        parts = []
        for row in rows:
            fab = row.get("fab")
            constraint = row.get("constraint")
            if fab and constraint:
                parts.append(f"{fab} has {constraint}")

        if parts:
            return "Fabs with capacity constraints: " + "; ".join(parts) + "."

        return "No capacity constraints were found."

    # -----------------------------
    # Q6: Delayed shipments
    # -----------------------------
    if question_id == "Q6":
        parts = []
        for row in rows:
            order = row.get("order")
            shipment = row.get("shipment")
            if order and shipment:
                parts.append(f"order {order} depends on shipment {shipment}")

        if parts:
            return (
                "Delayed shipments impacting orders: "
                + "; ".join(parts)
                + "."
            )

        return "No delayed shipments were found."

    # -----------------------------
    # Q7: Inventory risk
    # -----------------------------
    if question_id == "Q7":
        parts = []
        for row in rows:
            inventory = row.get("inventory")
            days = row.get("days_of_supply")
            if inventory and days is not None:
                parts.append(f"{inventory} has {days} days of supply")

        if parts:
            return "Inventory risk details: " + "; ".join(parts) + "."

        return "No inventory risks were detected."

    # -----------------------------
    # Q8: Alternative suppliers
    # -----------------------------
    if question_id == "Q8":
        suppliers = {
            row.get("supplier")
            for row in rows
            if row.get("supplier")
        }
        if suppliers:
            return "Alternative suppliers: " + ", ".join(sorted(suppliers)) + "."

        return "No alternative suppliers were found."

    return ""


# --------------------------------------------------
# Public API
# --------------------------------------------------

def synthesize_answer(
    question: str,
    query: str,
    results: Dict[str, object],
    errors: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Final step of the QA pipeline.

    Guarantees:
    - No hallucination beyond executed query results
    - No leakage across query constraints
    - Deterministic mapping from rows → answer
    """

    rows = results.get("rows", [])
    matched_id = results.get("matched_question_id")

    # -----------------------------
    # Validation errors
    # -----------------------------
    if errors:
        error_text = "; ".join(err.get("message", "") for err in errors)
        return (
            "Answer (validation failed): "
            f"{error_text} The query was not executed."
        )

    # -----------------------------
    # Empty results
    # -----------------------------
    if not rows:
        return "Answer: No results returned for the given query."

    # -----------------------------
    # Natural language answer
    # -----------------------------
    natural = _format_natural_answer(matched_id, rows, query)
    if natural:
        return "Answer: " + natural

    infineon_answer = _format_infineon_answer(rows, query, question)
    if infineon_answer:
        return "Answer: " + infineon_answer

    ranking_answer = _format_generic_ranking_answer(rows, question=question)
    if ranking_answer:
        return "Answer: " + ranking_answer

    # -----------------------------
    # Fallback for Infineon graph rows
    # -----------------------------
    return "Answer: " + _format_generic_answer(rows)
