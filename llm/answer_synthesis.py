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


def _format_total_demand_by_region(rows: List[Dict[str, object]]) -> str:
    if not _has_any_key(rows, ["regionName"]) or not _has_any_key(
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

    top_row, metric_key, top_value = max(scored, key=lambda item: item[2])
    region = _clean_value(_row_get(top_row, "regionName"))
    survey_type = _row_get(top_row, "surveyType", "originType")
    survey_text = f" for {_clean_value(survey_type)}" if survey_type else ""
    metric_text = _humanize_key(metric_key)
    summary = (
        f"Regional demand returned {len(rows)} row(s). "
        f"The highest {metric_text}{survey_text} is in {region} "
        f"with {_format_number(top_value)}."
    )

    if len(scored) > 1:
        low_row, _, low_value = min(scored, key=lambda item: item[2])
        low_region = _clean_value(_row_get(low_row, "regionName"))
        summary += f" The lowest returned value is {low_region} with {_format_number(low_value)}."

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


def _format_autonomous_driving(rows: List[Dict[str, object]]) -> str:
    if not _has_any_key(rows, ["vehicleType", "vehicle"]):
        return ""
    if not _has_any_key(rows, ["percentage", "maxPercentage", "avgPercentage", "avgPct"]):
        return ""

    scored = _numeric_rows(rows, ["maxPercentage", "percentage", "avgPercentage", "avgPct"])
    if not scored:
        return ""

    top_row, metric_key, top_value = max(scored, key=lambda item: item[2])
    vehicle = _clean_value(_row_get(top_row, "vehicleType", "vehicle"))
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

    top_row, metric_key, top_value = max(scored, key=lambda item: item[2])
    category = _row_get(top_row, "technologyCategory", "techLabel", "tech")
    response_type = _clean_value(_row_get(top_row, "responseType"))
    category_text = f" for {_clean_value(category)}" if category is not None else ""
    return (
        f"Order-cancellation results returned {len(rows)} row(s). "
        f"The largest returned {_humanize_key(metric_key)} is {response_type}{category_text} "
        f"with {_format_number(top_value)}."
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


def _format_vehicle_sales_by_month(rows: List[Dict[str, object]]) -> str:
    if not _has_any_key(rows, ["monthLabel", "month"]):
        return ""
    if not _has_any_key(rows, ["unitsSold", "totalUnitsSold"]):
        return ""

    scored = _numeric_rows(rows, ["totalUnitsSold", "unitsSold"])
    if not scored:
        return ""

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


def _format_infineon_answer(rows: List[Dict[str, object]], query: str) -> str:
    query_lower = query.lower()

    formatters = []
    if "currentdemandanalysis" in query_lower or "baseline" in query_lower:
        formatters.append(_format_current_bl_comparison)
    if "vehiclesalesobservation" in query_lower or _has_any_key(rows, ["monthLabel"]):
        formatters.append(_format_vehicle_sales_by_month)
    if "ordercancellation" in query_lower or _has_any_key(rows, ["responseType"]):
        formatters.append(_format_order_cancellation)
    if "autonomousdrivingdevelopment" in query_lower or _has_any_key(rows, ["saeLevel", "sae"]):
        formatters.append(_format_autonomous_driving)
    if "futuredemandanalysis" in query_lower or _has_any_key(rows, ["techLabel", "quarterLabel"]):
        formatters.append(_format_future_demand_by_tech_quarter)
    if "demandforregion" in query_lower or _has_any_key(rows, ["regionName"]):
        formatters.append(_format_total_demand_by_region)

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

    infineon_answer = _format_infineon_answer(rows, query)
    if infineon_answer:
        return "Answer: " + infineon_answer

    # -----------------------------
    # Fallback for Infineon graph rows
    # -----------------------------
    return "Answer: " + _format_generic_answer(rows)
