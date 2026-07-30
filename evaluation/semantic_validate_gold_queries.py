#!/usr/bin/env python3
"""Semantic triage for True Demand gold SPARQL queries.

This script does not replace expert review. It is a fast guardrail that checks
whether the expected question contract is reflected in the gold query shape.
It can optionally execute the queries through Fuseki or a local RDF graph, but
it also works offline with only the JSON benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rdflib import Graph
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
except ImportError:  # pragma: no cover - local thesis editing may not have rdflib.
    Graph = None
    SPARQLStore = None


DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


DIMENSION_TERMS: Dict[str, Tuple[str, ...]] = {
    "region": ("region", "regionname", "inregion"),
    "quarter": ("quarter", "timeperiod", "periodlabel", "fortimeperiod"),
    "month": ("month", "timeperiod", "periodlabel", "fortimeperiod"),
    "year": ("year", "hasyear"),
    "vehicle_type": ("vehicle", "vehicletype", "hasvehicletype", "analyzesvehicletype"),
    "sae_level": ("sae", "saelevel", "hassaelevel"),
    "technology_category": (
        "technology",
        "tech",
        "techlabel",
        "technologycategory",
        "analyzestechnologycategory",
        "fortechnologycategory",
    ),
    "component": ("component", "forcomponent", "componenttype"),
    "trend": ("trend", "inventorytrend", "hasinventorytrend"),
    "response_type": ("response", "responsetype", "hasresponsetype"),
    "survey": ("survey", "surveytype", "hassurveyorigin"),
    "company": ("company", "companyname", "forcompany"),
    "baseline": ("baseline", "baselinetype"),
}


SCOPE_TERMS: Dict[str, Tuple[str, ...]] = {
    "oem": ("oem", "oem_survey"),
    "tier1": ("tier1", "tier1_survey"),
    "semiconductor": ("semiconductor", "semiconductor_survey"),
}


FIELDNAMES = [
    "id",
    "question",
    "family",
    "answer_shape",
    "manual_status",
    "auto_decision",
    "severity",
    "flags",
    "rationale",
    "execution_checked",
    "valid",
    "row_count",
    "execution_error",
    "row_preview",
    "selected_vars",
    "group_vars",
    "has_limit",
    "has_order_by",
    "query",
]


def _load_json_rows(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in payload if isinstance(row, dict)]


def _load_manual_audit(path: str) -> Dict[str, Dict[str, str]]:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row.get("id") or ""): row for row in csv.DictReader(f)}


def _load_ids_from_csv(path: str, column: str = "id") -> List[str]:
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        ids = [str(row.get(column) or "").strip() for row in csv.DictReader(f)]
    return [request_id for request_id in ids if request_id]


def _compact_query(query: Any) -> str:
    return " ".join(str(query or "").split())


def _strip_comments(query: str) -> str:
    lines = []
    for line in query.splitlines():
        lines.append(line.split("#", 1)[0].rstrip())
    return "\n".join(lines).strip()


def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query


def _normalize(text: Any) -> str:
    return re.sub(r"[^a-z0-9_?]+", " ", str(text or "").lower()).strip()


def _query_signal(query: str) -> str:
    return _normalize(query).replace("?", " ")


def _section_between(text: str, start_pattern: str, end_patterns: Sequence[str]) -> str:
    match = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    start = match.end()
    end = len(text)
    for pattern in end_patterns:
        end_match = re.search(pattern, text[start:], flags=re.IGNORECASE | re.DOTALL)
        if end_match:
            end = min(end, start + end_match.start())
    return text[start:end]


def _selected_vars(query: str) -> List[str]:
    select = _section_between(query, r"\bselect\b", [r"\bwhere\b"])
    return sorted(set(re.findall(r"\?([A-Za-z_][A-Za-z0-9_]*)", select)))


def _group_vars(query: str) -> List[str]:
    group = _section_between(
        query,
        r"\bgroup\s+by\b",
        [r"\border\s+by\b", r"\blimit\b", r"\bhaving\b", r"\boffset\b"],
    )
    return sorted(set(re.findall(r"\?([A-Za-z_][A-Za-z0-9_]*)", group)))


def _query_features(query: str) -> Dict[str, Any]:
    q_low = query.lower()
    return {
        "selected_vars": _selected_vars(query),
        "group_vars": _group_vars(query),
        "has_group_by": bool(re.search(r"\bgroup\s+by\b", q_low)),
        "has_limit": bool(re.search(r"\blimit\s+\d+", q_low)),
        "has_limit_one": bool(re.search(r"\blimit\s+1\b", q_low)),
        "has_order_by": bool(re.search(r"\border\s+by\b", q_low)),
        "has_desc": bool(re.search(r"\bdesc\s*\(", q_low)),
        "has_asc": bool(re.search(r"\basc\s*\(", q_low)),
        "has_count": bool(re.search(r"\bcount\s*\(", q_low)),
        "has_avg": bool(re.search(r"\bavg\s*\(", q_low)),
        "has_sum": bool(re.search(r"\bsum\s*\(", q_low)),
        "has_min": bool(re.search(r"\bmin\s*\(", q_low)),
        "has_max": bool(re.search(r"\bmax\s*\(", q_low)),
        "signal": _query_signal(query),
    }


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _question_contract(question: str, row: Dict[str, Any]) -> Dict[str, Any]:
    q = _normalize(question)
    expected_shape = _normalize(row.get("answer_shape"))
    dims = []
    for name, terms in {
        "region": ("region", "regional"),
        "quarter": ("quarter",),
        "month": ("month", "monthly"),
        "year": ("year", "annual", "yearly"),
        "vehicle_type": ("vehicle type", "vehicle", "bev", "behv", "ice"),
        "sae_level": ("sae",),
        "technology_category": ("technology category", "technology bucket", "tech category"),
        "component": ("component",),
        "trend": ("trend",),
        "response_type": ("response type", "response"),
        "survey": ("survey group", "survey type", "survey origin", "survey"),
        "company": ("by company", "per company", "for each company", "which company", "which companies", "list companies", "company name"),
        "baseline": ("baseline", "bl1", "bl2"),
    }.items():
        if _has_any(q, terms):
            dims.append(name)
    metric_unit_sum = _has_any(
        q,
        (
            "sales units",
            "vehicle sales",
            "vehicles sold",
            "units sold",
            "demand units",
            "sales volume",
        ),
    )
    countable_records = _has_any(
        q,
        (
            "record",
            "records",
            "response",
            "responses",
            "observation",
            "observations",
            "participant",
            "participants",
            "report",
            "reports",
            "companies",
            "company",
            "data points",
            "entries",
        ),
    )
    count = expected_shape == "count" or (
        _has_any(q, ("how many", "number of", "count", "quantity of"))
        and not metric_unit_sum
        and countable_records
    )
    avg = expected_shape == "average" or _has_any(q, ("average", "mean", "avg"))
    total_count = _has_any(q, ("total count", "total number", "aggregate number"))
    total = expected_shape == "sum" or (
        _has_any(q, ("total", "sum", "combined", "cumulative", "aggregate"))
        and not total_count
    )
    top = expected_shape == "rank" or _has_any(
        q,
        ("highest", "greatest", "largest", "maximum", "max ", "top", "most", "leads", "peak"),
    )
    bottom = _has_any(q, ("lowest", "smallest", "minimum", "least"))
    list_like = _has_any(q, ("which", "list", "identify", "show", "provide", "return", "give me"))
    breakdown = _has_any(q, (" by ", " per ", "for each", "across", "grouped", "breakdown", "distributed"))
    all_items = _has_any(q, (" all ", " complete list", "every", "each"))
    all_years = _has_any(q, ("all years", "every year", "considering data from all years"))
    scopes = []
    if _has_any(q, ("oem survey", "oem surveys", "oem demand", "oem current", "oem future", "for oems")):
        scopes.append("oem")
    if _has_any(q, ("tier1 survey", "tier1 surveys", "tier1 demand", "tier1 current", "tier1 future", "tier1 companies", "tier1 suppliers")):
        scopes.append("tier1")
    if _has_any(
        q,
        (
            "semiconductor survey",
            "semiconductor surveys",
            "semiconductor companies",
            "semiconductor current",
            "semiconductor future",
            "semiconductor demand",
        ),
    ):
        scopes.append("semiconductor")
    return {
        "count": count,
        "avg": avg,
        "total": total,
        "top": top,
        "bottom": bottom,
        "list_like": list_like,
        "breakdown": breakdown,
        "all_items": all_items,
        "all_years": all_years,
        "dimensions": dims,
        "scopes": scopes,
        "text": q,
    }


def _dimension_present(dim: str, query_signal: str, selected: Sequence[str], grouped: Sequence[str]) -> bool:
    combined = " ".join([query_signal, *selected, *grouped]).lower()
    return any(term.lower() in combined for term in DIMENSION_TERMS.get(dim, (dim,)))


def _scope_present(scope: str, query_signal: str) -> bool:
    return any(term.lower() in query_signal for term in SCOPE_TERMS.get(scope, (scope,)))


def _only_count_selected(selected: Sequence[str]) -> bool:
    if not selected:
        return False
    non_count = [
        var
        for var in selected
        if not re.search(r"(count|total|number|quantity|num)$", var.lower())
    ]
    return len(non_count) == 0


def _semantic_flags(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    question = str(row.get("question") or "")
    query = str(row.get("query") or "")
    contract = _question_contract(question, row)
    features = _query_features(query)
    selected = features["selected_vars"]
    grouped = features["group_vars"]
    query_signal = features["signal"]
    errors: List[str] = []
    warnings: List[str] = []

    if contract["count"] and not features["has_count"]:
        errors.append("missing_COUNT_for_count_question")
    if contract["avg"] and not features["has_avg"]:
        errors.append("missing_AVG_for_average_question")
    if contract["total"] and not (features["has_sum"] or features["has_count"]):
        warnings.append("missing_SUM_or_COUNT_for_total_question")
    if contract["breakdown"] and not features["has_group_by"]:
        errors.append("missing_GROUP_BY_for_breakdown_question")
    if contract["top"] and not (
        features["has_order_by"] or features["has_max"] or features["has_limit_one"]
    ):
        errors.append("missing_top_ranking_operation")
    if contract["bottom"] and not (
        features["has_order_by"] or features["has_min"] or features["has_limit_one"]
    ):
        errors.append("missing_bottom_ranking_operation")
    if contract["breakdown"] and features["has_limit_one"] and not (contract["top"] or contract["bottom"]):
        errors.append("LIMIT_1_used_for_breakdown_question")
    if contract["all_items"] and features["has_limit_one"] and not (contract["top"] or contract["bottom"]):
        errors.append("LIMIT_1_used_for_all_items_question")

    for dim in contract["dimensions"]:
        if not _dimension_present(dim, query_signal, selected, grouped):
            errors.append(f"missing_requested_dimension:{dim}")
    for scope in contract["scopes"]:
        if not _scope_present(scope, query_signal):
            warnings.append(f"missing_requested_scope:{scope}")

    if (
        "company" in contract["dimensions"]
        and _has_any(contract["text"], ("which", "list", "identify"))
        and _only_count_selected(selected)
    ):
        errors.append("company_list_question_selects_only_count")

    if (
        "vehicle_type" in contract["dimensions"]
        and contract["all_years"]
        and _dimension_present("year", query_signal, selected, grouped)
        and "year" not in contract["dimensions"]
    ):
        errors.append("all_years_question_keeps_year_as_grouping_dimension")

    if (
        "component" in contract["dimensions"]
        and "trend" not in contract["dimensions"]
        and _dimension_present("trend", query_signal, selected, grouped)
        and contract["count"]
    ):
        warnings.append("extra_grouping_dimension:trend")

    if "actual" in contract["text"] and "forecast" in query_signal and "actual" not in query_signal:
        errors.append("actual_question_uses_forecast_signal")
    if "forecast" in contract["text"] and "actual" in query_signal and "forecast" not in query_signal:
        errors.append("forecast_question_uses_actual_signal")
    if "current demand" in contract["text"] and "future" in query_signal and "current" not in query_signal:
        errors.append("current_demand_question_uses_future_signal")
    if "future demand" in contract["text"] and "current" in query_signal and "future" not in query_signal:
        errors.append("future_demand_question_uses_current_signal")

    return errors, warnings


def _make_graph(graph_path: str, fuseki_query_url: str):
    if Graph is None:
        raise RuntimeError("rdflib is required for execution validation.")
    if fuseki_query_url:
        if SPARQLStore is None:
            raise RuntimeError("rdflib SPARQLStore is required for Fuseki validation.")
        return Graph(store=SPARQLStore(fuseki_query_url))
    graph = Graph()
    graph.parse(graph_path, format="turtle")
    return graph


def _serialize_binding_row(row: Any) -> List[str]:
    return [str(value) for value in row]


def _execute_query(graph: Any, query: str, preview_limit: int = 5) -> Tuple[bool, Optional[int], str, str]:
    try:
        rows = list(graph.query(_ensure_prefixes(_strip_comments(query))))
        preview = [_serialize_binding_row(row) for row in rows[:preview_limit]]
        return True, len(rows), "", json.dumps(preview, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - depends on external graph/Fuseki.
        return False, None, str(exc), ""


def _manual_status(row: Dict[str, str]) -> str:
    return (
        row.get("semantic_audit_status")
        or row.get("human_gold_valid")
        or ""
    ).strip().lower()


def _manual_counts(audit: Dict[str, Dict[str, str]]) -> Counter:
    counts: Counter = Counter()
    for row in audit.values():
        status = _manual_status(row)
        if status:
            counts[status] += 1
    return counts


def analyze(
    dataset_path: str,
    audit_path: str = "",
    *,
    graph_path: str = "",
    fuseki_query_url: str = "",
    execute: bool = False,
    include_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    rows = _load_json_rows(dataset_path)
    if include_ids:
        include_set = set(include_ids)
        rows = [row for row in rows if str(row.get("id") or "") in include_set]
    audit = _load_manual_audit(audit_path)
    graph = None
    execution_checked = False
    if execute:
        graph = _make_graph(graph_path, fuseki_query_url)
        execution_checked = True

    cases: List[Dict[str, Any]] = []
    flag_counts: Counter = Counter()
    severity_counts: Counter = Counter()
    decision_counts: Counter = Counter()
    for row in rows:
        request_id = str(row.get("id") or "")
        manual = _manual_status(audit.get(request_id, {}))
        errors, warnings = _semantic_flags(row)
        valid: Any = ""
        row_count: Any = ""
        exec_error = ""
        row_preview = ""
        if graph is not None:
            valid, row_count, exec_error, row_preview = _execute_query(graph, str(row.get("query") or ""))
            if not valid:
                errors.append("query_execution_error")
            elif row_count == 0:
                warnings.append("query_returns_zero_rows")

        if errors:
            severity = "error"
        elif warnings:
            severity = "warning"
        else:
            severity = "ok"

        flags = errors + warnings
        for flag in flags:
            flag_counts[flag] += 1
        severity_counts[severity] += 1

        if manual in {"correct", "probably_correct"}:
            decision = "manual_accept"
        elif manual in {"incorrect", "needs_recheck", "unclear", "probably_incorrect"}:
            decision = "manual_review"
        elif severity == "ok":
            decision = "auto_accept_candidate"
        else:
            decision = "manual_review"
        decision_counts[decision] += 1

        features = _query_features(str(row.get("query") or ""))
        rationale = "; ".join(flags)
        if exec_error:
            rationale = (rationale + "; " if rationale else "") + exec_error

        cases.append(
            {
                "id": request_id,
                "question": row.get("question", ""),
                "family": row.get("family") or row.get("topic") or "",
                "answer_shape": row.get("answer_shape") or "",
                "manual_status": manual,
                "auto_decision": decision,
                "severity": severity,
                "flags": "; ".join(flags),
                "rationale": rationale,
                "execution_checked": str(execution_checked),
                "valid": valid,
                "row_count": row_count,
                "execution_error": exec_error,
                "row_preview": row_preview,
                "selected_vars": ", ".join(features["selected_vars"]),
                "group_vars": ", ".join(features["group_vars"]),
                "has_limit": str(features["has_limit"]),
                "has_order_by": str(features["has_order_by"]),
                "query": _compact_query(row.get("query")),
            }
        )

    return {
        "summary": {
            "dataset": dataset_path,
            "audit": audit_path,
            "filtered_to_ids": len(include_ids or []),
            "questions": len(rows),
            "execution_checked": execution_checked,
            "manual_audit_counts": dict(_manual_counts(audit)),
            "severity_counts": dict(severity_counts),
            "decision_counts": dict(decision_counts),
            "flag_counts": dict(flag_counts.most_common()),
        },
        "cases": cases,
    }


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def write_markdown(report: Dict[str, Any], path: str) -> None:
    summary = report["summary"]
    lines = [
        "# Gold Query Semantic Validation",
        "",
        "This report is an automated triage layer. It checks whether the SPARQL query shape matches the natural-language question contract. It does not replace expert review.",
        "",
        "## Summary",
        "",
        f"- Questions: {summary['questions']}",
        f"- Execution checked: {summary['execution_checked']}",
        f"- Manual audit labels loaded: {sum(summary['manual_audit_counts'].values())}",
        "",
        "## Manual Audit Counts",
        "",
    ]
    if summary["manual_audit_counts"]:
        for key, value in sorted(summary["manual_audit_counts"].items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.extend(["", "## Automated Severity Counts", ""])
    for key, value in sorted(summary["severity_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Automated Decisions", ""])
    for key, value in sorted(summary["decision_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Most Common Flags", ""])
    for flag, value in list(summary["flag_counts"].items())[:25]:
        lines.append(f"- {flag}: {value}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows marked `auto_accept_candidate` are not proven correct; they simply passed the current offline semantic-shape checks. Rows marked `manual_review` should be inspected in Fuseki or compared with expert expectations.",
            "",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantically triage True Demand gold SPARQL queries.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--audit", default="")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--fuseki-query-url", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ids-from-csv", default="")
    parser.add_argument("--ids-column", default="id")
    parser.add_argument("--out-csv", default="results/gold_semantic_validation_auto.csv")
    parser.add_argument("--out-review-csv", default="results/gold_semantic_manual_review_only.csv")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="results/gold_semantic_validation_summary.md")
    args = parser.parse_args()

    report = analyze(
        args.dataset,
        args.audit,
        graph_path=args.graph,
        fuseki_query_url=args.fuseki_query_url,
        execute=args.execute,
        include_ids=_load_ids_from_csv(args.ids_from_csv, args.ids_column),
    )
    cases = report["cases"]
    review_rows = [row for row in cases if row.get("auto_decision") == "manual_review"]
    write_csv(cases, args.out_csv)
    write_csv(review_rows, args.out_review_csv)
    write_markdown(report, args.out_md)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = report["summary"]
    print("===== GOLD SEMANTIC VALIDATION =====")
    print(f"Dataset: {args.dataset}")
    print(f"Questions: {summary['questions']}")
    print(f"Execution checked: {summary['execution_checked']}")
    print(f"Manual labels loaded: {sum(summary['manual_audit_counts'].values())}")
    print(f"Severity: {summary['severity_counts']}")
    print(f"Decisions: {summary['decision_counts']}")
    print(f"Review rows: {len(review_rows)}")
    print(f"CSV: {args.out_csv}")
    print(f"Review CSV: {args.out_review_csv}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
