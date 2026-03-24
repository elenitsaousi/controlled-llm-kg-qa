import re
from typing import Dict, List, Optional


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _format_row(row: Dict[str, object]) -> str:
    return ", ".join(f"{k}={v}" for k, v in row.items())


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

    # -----------------------------
    # Fallback (debug-safe)
    # -----------------------------
    formatted = "; ".join(_format_row(row) for row in rows)
    return "Answer: " + formatted
