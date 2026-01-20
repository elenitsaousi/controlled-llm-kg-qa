from typing import Dict, List, Optional


def _format_row(row: Dict[str, object]) -> str:
    parts = []
    for key, value in row.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_natural_answer(
    question_id: Optional[str], rows: List[Dict[str, object]]
) -> str:
    if question_id == "Q1":
        suppliers = [row.get("supplier") for row in rows]
        yields = [row.get("yield") for row in rows]
        pairs = [
            f"{supplier} (yield {yield_id})"
            for supplier, yield_id in zip(suppliers, yields)
            if supplier and yield_id
        ]
        if pairs:
            return "Suppliers affecting yield are: " + ", ".join(pairs) + "."
    if question_id == "Q2":
        parts = []
        for row in rows:
            supplier = row.get("supplier")
            avg_yield = row.get("avg_yield")
            if supplier is not None and avg_yield is not None:
                parts.append(f"{supplier}: average yield {avg_yield}")
        if parts:
            return "Average yield by supplier: " + "; ".join(parts) + "."
    if question_id == "Q3":
        parts = []
        for row in rows:
            tool = row.get("tool")
            defect = row.get("defect")
            if tool and defect:
                parts.append(f"{tool} linked to defect {defect}")
        if parts:
            return "Tools linked to defects: " + "; ".join(parts) + "."
    if question_id == "Q4":
        parts = []
        for row in rows:
            supplier = row.get("supplier")
            material = row.get("material")
            if supplier and material:
                parts.append(f"{supplier} supplies {material}")
        if parts:
            return "Suppliers providing lithography materials: " + ", ".join(
                parts
            ) + "."
    if question_id == "Q5":
        parts = []
        for row in rows:
            fab = row.get("fab")
            constraint = row.get("constraint")
            if fab and constraint:
                parts.append(f"{fab} has {constraint}")
        if parts:
            return "Fabs with capacity constraints: " + "; ".join(parts) + "."
    if question_id == "Q6":
        parts = []
        for row in rows:
            order = row.get("order")
            shipment = row.get("shipment")
            if order and shipment:
                parts.append(f"order {order} depends on shipment {shipment}")
        if parts:
            return "Delayed shipments impacting orders: " + "; ".join(parts) + "."
    if question_id == "Q7":
        parts = []
        for row in rows:
            inventory = row.get("inventory")
            days = row.get("days_of_supply")
            if inventory and days is not None:
                parts.append(f"{inventory} has {days} days of supply")
        if parts:
            return "Inventory risk details: " + "; ".join(parts) + "."
    if question_id == "Q8":
        suppliers = [row.get("supplier") for row in rows if row.get("supplier")]
        if suppliers:
            return "Alternative suppliers: " + ", ".join(suppliers) + "."
    return ""


def synthesize_answer(
    question: str,
    query: str,
    results: Dict[str, object],
    errors: Optional[List[Dict[str, str]]] = None,
) -> str:
    rows = results.get("rows", [])
    matched_id = results.get("matched_question_id")
    if errors:
        error_text = "; ".join(err["message"] for err in errors)
        return (
            "Answer (validation failed): "
            f"{error_text} The query was not executed."
        )
    if not rows:
        return (
            "Answer (placeholder): No results returned by the stub "
            "executor."
        )
    natural = _format_natural_answer(matched_id, rows)
    if natural:
        return "Answer: " + natural
    formatted = "; ".join(_format_row(row) for row in rows)
    return "Answer: " + formatted
