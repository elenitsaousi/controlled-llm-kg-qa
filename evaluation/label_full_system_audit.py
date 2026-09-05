#!/usr/bin/env python3
"""Label a full-system audit CSV with route and accuracy metrics.

The script is intentionally conservative:

* unchanged rows can inherit labels from a previous labeled audit only when the
  natural-language question, selected source, and selected query all match;
* KG analytics rows are scored by executing the selected query and the gold
  query and comparing their result signatures;
* DR ontology rows are deterministic lookup rows and are accepted only when a
  definition-style answer was returned;
* advisory rows keep previous manual labels when available, otherwise they are
  accepted only when a graph-backed answer with rows is present.

This produces the same high-level metric families used by the Streamlit UI:
overall accuracy, KG accuracy, ontology accuracy, advisory accuracy,
deterministic-route accuracy, LLM-fallback accuracy, cost/call reduction, and
remaining-failure difficulty buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

CORRECT_VALUES = {"correct", "yes", "y", "1", "true", "ok", "probably_correct"}
INCORRECT_VALUES = {"incorrect", "wrong", "no", "n", "0", "false"}
UNCLEAR_VALUES = {"unclear", "ambiguous", "partial", "unknown", "?"}


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in CORRECT_VALUES:
        return "correct"
    if text in INCORRECT_VALUES:
        return "incorrect"
    if text in UNCLEAR_VALUES:
        return "unclear"
    return ""


def _one_line(text: object) -> str:
    return " ".join(str(text or "").split())


def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query


def _norm_term(term: object) -> str:
    if term is None:
        return "NULL"
    try:
        return term.n3()  # rdflib term
    except Exception:
        return str(term)


def _signature(rows: Iterable[Tuple[object, ...]]) -> Counter:
    return Counter(tuple(_norm_term(value) for value in row) for row in rows)


def _load_graph(graph_path: str):
    from rdflib import Graph

    graph = Graph()
    graph.parse(graph_path, format="turtle")
    return graph


def _query_signature(graph, query: str) -> Counter:
    return _signature(list(graph.query(_ensure_prefixes(query))))


def _previous_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        label = _normalize_label(row.get("correctness"))
        if not label:
            continue
        key = (
            _one_line(row.get("question")),
            _one_line(row.get("selected_query")),
            str(row.get("selected_source") or "").strip(),
        )
        index[key] = row
    return index


def _gold_by_question(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        question = _one_line(row.get("question"))
        if question and question not in out:
            out[question] = row
    return out


def _expected_route(row: Dict[str, str]) -> str:
    expected = str(row.get("expected_route") or "").strip()
    if expected:
        return expected
    source = str(row.get("selected_source") or "").strip()
    if source == "digital_reference_ontology":
        return "definition"
    topic = str(row.get("topic") or "").strip()
    if topic.startswith("advisory_"):
        return "advisory"
    return "kg_analytics"


def _difficulty_for_wrong(row: Dict[str, str]) -> str:
    topic = str(row.get("topic") or "").strip()
    question = str(row.get("question") or "").lower()
    if _expected_route(row) in {"definition", "advisory"}:
        return "easy"
    if topic in {"autonomous_driving", "current_demand_baselines", "vehicle_sales"}:
        return "hard"
    if topic in {"future_demand", "regional_demand", "inventory", "shortage"}:
        return "medium" if any(t in question for t in ["by", "grouped", "for each", "across"]) else "hard"
    return "medium"


def _failure_family(row: Dict[str, str]) -> str:
    topic = str(row.get("topic") or "").strip()
    if _expected_route(row) == "definition":
        return "ontology_lookup"
    if _expected_route(row) == "advisory":
        return "advisory_not_synthesized"
    mapping = {
        "autonomous_driving": "autonomous_driving_complex_grouping",
        "current_demand_baselines": "current_demand_baseline_or_scope",
        "vehicle_sales": "vehicle_sales_metric_or_dimension",
        "future_demand": "future_demand_complex_dimension",
        "regional_demand": "regional_demand_scope_or_dimension",
        "shortage": "shortage_scope_or_shape",
        "inventory": "inventory_scope_or_dimension",
    }
    return mapping.get(topic, "other_semantic_mismatch")


def _failure_type(row: Dict[str, str], *, strict_match: bool) -> str:
    if strict_match:
        return "none"
    if str(row.get("graph_error") or "").strip():
        return "execution_error"
    if not str(row.get("selected_query") or "").strip() and _expected_route(row) == "kg_analytics":
        return "missing_selected_query"
    if str(row.get("graph_row_count") or "").strip() in {"0", "0.0"} and _expected_route(row) == "kg_analytics":
        return "empty_result"
    return "semantic_or_result_mismatch"


def _is_definition_correct(row: Dict[str, str]) -> bool:
    answer = str(row.get("answer_text") or "").strip().lower()
    if not answer:
        return False
    bad = ["not available", "out of scope", "cannot answer", "multiple plausible interpretations"]
    return not any(token in answer for token in bad)


def _is_advisory_correct(row: Dict[str, str]) -> bool:
    answer = str(row.get("answer_text") or "").strip().lower()
    if not answer:
        return False
    if "cannot answer" in answer or "out of scope" in answer:
        return False
    try:
        return int(float(row.get("graph_row_count") or 0)) > 0
    except ValueError:
        return False


def _metrics(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correctness") == "correct")
    incorrect = sum(1 for row in rows if row.get("correctness") == "incorrect")
    unclear = sum(1 for row in rows if row.get("correctness") == "unclear")
    return {
        "rows": total,
        "correct": correct,
        "incorrect": incorrect,
        "unclear": unclear,
        "accuracy": correct / total if total else 0.0,
    }


def _group(rows: List[Dict[str, str]], key_fn) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[str(key_fn(row) or "unknown")].append(row)
    return {key: _metrics(bucket) for key, bucket in sorted(buckets.items())}


def _summary(rows: List[Dict[str, str]], *, cost_per_call: float) -> Dict[str, Any]:
    overall = _metrics(rows)
    by_expected = _group(rows, _expected_route)
    by_mode = _group(rows, lambda row: row.get("system_mode") or "unknown")
    by_route = _group(rows, lambda row: row.get("route") or "unknown")
    by_source = _group(rows, lambda row: row.get("selected_source") or "unknown")
    by_topic = _group(rows, lambda row: row.get("topic") or "unknown")

    wrong = [row for row in rows if row.get("correctness") == "incorrect"]
    llm_calls = sum(int(float(row.get("estimated_llm_calls") or 0)) for row in rows)
    llm_route = [row for row in rows if row.get("system_mode") == "llm_ranking"]
    direct = [row for row in rows if row.get("system_mode") == "direct_graph_supported"]
    return {
        "rows": len(rows),
        "overall_accuracy": overall["accuracy"],
        "correct_answers": overall["correct"],
        "incorrect_answers": overall["incorrect"],
        "unclear_answers": overall["unclear"],
        "kg_accuracy": by_expected.get("kg_analytics", {}).get("accuracy", 0.0),
        "ontology_accuracy": by_expected.get("definition", {}).get("accuracy", 0.0),
        "advisory_accuracy": by_expected.get("advisory", {}).get("accuracy", 0.0),
        "deterministic_questions": len(direct),
        "deterministic_correct": sum(1 for row in direct if row.get("correctness") == "correct"),
        "deterministic_accuracy": _metrics(direct)["accuracy"],
        "llm_fallback_questions": len(llm_route),
        "llm_fallback_correct": sum(1 for row in llm_route if row.get("correctness") == "correct"),
        "llm_fallback_accuracy": _metrics(llm_route)["accuracy"],
        "cold_llm_calls": len(llm_route),
        "warm_cache_llm_calls": llm_calls,
        "cost_per_call_eur": cost_per_call,
        "cold_cost_eur": len(llm_route) * cost_per_call,
        "warm_cache_cost_eur": llm_calls * cost_per_call,
        "all_llm_baseline_cost_eur": len(rows) * cost_per_call,
        "cold_call_reduction": 1.0 - (len(llm_route) / len(rows) if rows else 0.0),
        "warm_cache_call_reduction": 1.0 - (llm_calls / len(rows) if rows else 0.0),
        "by_expected_route": by_expected,
        "by_system_mode": by_mode,
        "by_route": by_route,
        "by_source": by_source,
        "by_topic": by_topic,
        "incorrect_by_difficulty": dict(Counter(row.get("human_sparql_difficulty") or "unknown" for row in wrong)),
        "incorrect_by_failure_type": dict(Counter(row.get("failure_type") or "unknown" for row in wrong)),
        "incorrect_by_failure_family": dict(Counter(row.get("failure_family") or "unknown" for row in wrong)),
        "audit_methods": dict(Counter(row.get("audit_method") or "unknown" for row in rows)),
    }


def _write_md(path: str, payload: Dict[str, Any]) -> None:
    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    lines = [
        "# Repaired Full-System Accuracy Metrics",
        "",
        "This report evaluates the final user-facing system on the repaired 1000-question benchmark.",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Overall answer-level accuracy | {payload['correct_answers']}/{payload['rows']} ({pct(payload['overall_accuracy'])}) |",
        f"| KG analytics accuracy | {pct(payload['kg_accuracy'])} |",
        f"| DR ontology definition accuracy | {pct(payload['ontology_accuracy'])} |",
        f"| Advisory accuracy | {pct(payload['advisory_accuracy'])} |",
        f"| Deterministic route accuracy | {payload['deterministic_correct']}/{payload['deterministic_questions']} ({pct(payload['deterministic_accuracy'])}) |",
        f"| LLM fallback answer accuracy | {payload['llm_fallback_correct']}/{payload['llm_fallback_questions']} ({pct(payload['llm_fallback_accuracy'])}) |",
        f"| Cold-start LLM calls | {payload['cold_llm_calls']} ({pct(1 - payload['cold_call_reduction'])} of questions) |",
        f"| Warm-cache new LLM calls | {payload['warm_cache_llm_calls']} ({pct(1 - payload['warm_cache_call_reduction'])} of questions) |",
        f"| Cold-start estimated cost | €{payload['cold_cost_eur']:.2f} vs €{payload['all_llm_baseline_cost_eur']:.2f} all-LLM |",
        f"| Warm-cache observed cost | €{payload['warm_cache_cost_eur']:.2f} |",
        "",
        "## Accuracy by Expected Route",
        "",
        "| Route | Rows | Correct | Incorrect | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for route, row in payload["by_expected_route"].items():
        lines.append(f"| `{route}` | {row['rows']} | {row['correct']} | {row['incorrect']} | {pct(row['accuracy'])} |")
    lines.extend(["", "## Accuracy by System Mode", "", "| Mode | Rows | Correct | Incorrect | Accuracy |", "|---|---:|---:|---:|---:|"])
    for mode, row in payload["by_system_mode"].items():
        lines.append(f"| `{mode}` | {row['rows']} | {row['correct']} | {row['incorrect']} | {pct(row['accuracy'])} |")
    lines.extend(["", "## Remaining Incorrect Answers by Difficulty", "", "| Difficulty | Count |", "|---|---:|"])
    for key, value in sorted(payload["incorrect_by_difficulty"].items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Remaining Incorrect Answers by Failure Family", "", "| Failure family | Count |", "|---|---:|"])
    for key, value in sorted(payload["incorrect_by_failure_family"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def label_rows(
    audit_rows: List[Dict[str, str]],
    *,
    gold_rows: List[Dict[str, Any]],
    graph_path: str,
    previous_labeled_rows: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    previous = _previous_index(previous_labeled_rows or [])
    gold_lookup = _gold_by_question(gold_rows)
    graph = None
    query_cache: Dict[str, Counter] = {}
    labeled: List[Dict[str, str]] = []

    extra_fields = ["strict_result_match", "source_gold_id", "audit_method", "human_sparql_difficulty", "failure_type", "failure_family"]
    for row in audit_rows:
        out = dict(row)
        for field in extra_fields:
            out.setdefault(field, "")

        previous_key = (
            _one_line(row.get("question")),
            _one_line(row.get("selected_query")),
            str(row.get("selected_source") or "").strip(),
        )
        prev = previous.get(previous_key)
        if prev is not None:
            out["correctness"] = _normalize_label(prev.get("correctness"))
            out["notes"] = "Transferred from previous labeled audit because question, selected source, and selected query are unchanged."
            out["audit_method"] = "transferred_from_previous_labeled_audit"
            for field in ("human_sparql_difficulty", "failure_type", "failure_family"):
                out[field] = str(prev.get(field) or "")
            labeled.append(out)
            continue

        expected_route = _expected_route(row)
        if expected_route == "definition":
            ok = _is_definition_correct(row)
            out["correctness"] = "correct" if ok else "incorrect"
            out["notes"] = "Deterministic DR ontology lookup returned a definition-style answer." if ok else "DR ontology lookup did not return a usable definition."
            out["audit_method"] = "deterministic_definition_rule"
            out["human_sparql_difficulty"] = "" if ok else "easy_non_sparql_definition"
            out["failure_type"] = "none" if ok else "ontology_lookup_failed"
            out["failure_family"] = "none" if ok else "ontology_lookup"
            labeled.append(out)
            continue

        if expected_route == "advisory":
            ok = _is_advisory_correct(row)
            out["correctness"] = "correct" if ok else "incorrect"
            out["notes"] = "Advisory answer is graph-backed and returned evidence rows." if ok else "Advisory route did not return graph-backed evidence."
            out["audit_method"] = "deterministic_advisory_rule"
            out["human_sparql_difficulty"] = "" if ok else "easy"
            out["failure_type"] = "none" if ok else "advisory_not_synthesized"
            out["failure_family"] = "none" if ok else "advisory_not_synthesized"
            labeled.append(out)
            continue

        gold = gold_lookup.get(_one_line(row.get("question")))
        out["source_gold_id"] = str(gold.get("id") if gold else "")
        selected_query = str(row.get("selected_query") or "").strip()
        if graph is None:
            graph = _load_graph(graph_path)
        try:
            if not gold or not selected_query:
                strict_match = False
            else:
                gold_query = str(gold.get("query") or "").strip()
                for query in (gold_query, selected_query):
                    key = _one_line(query)
                    if key and key not in query_cache:
                        query_cache[key] = _query_signature(graph, query)
                strict_match = query_cache.get(_one_line(gold_query)) == query_cache.get(_one_line(selected_query))
        except Exception as exc:
            strict_match = False
            out["notes"] = f"Strict KG result comparison failed during audit: {exc}"

        out["strict_result_match"] = str(bool(strict_match))
        out["correctness"] = "correct" if strict_match else "incorrect"
        if not out.get("notes"):
            out["notes"] = (
                "Selected query result signature matches the repaired gold query."
                if strict_match
                else "Selected query result signature does not match the repaired gold query."
            )
        out["audit_method"] = "strict_gold_result_signature"
        out["human_sparql_difficulty"] = "" if strict_match else _difficulty_for_wrong(row)
        out["failure_type"] = _failure_type(row, strict_match=strict_match)
        out["failure_family"] = "none" if strict_match else _failure_family(row)
        labeled.append(out)

    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Label and summarize a full-system KGQA audit CSV.")
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--previous-labeled-csv", default="")
    parser.add_argument("--cost-per-call", type=float, default=0.20)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    previous_rows = _read_csv(args.previous_labeled_csv) if args.previous_labeled_csv else None
    rows = label_rows(
        _read_csv(args.audit_csv),
        gold_rows=_load_json_list(args.gold),
        graph_path=args.graph,
        previous_labeled_rows=previous_rows,
    )
    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(args.out_csv, rows, fieldnames)
    payload = _summary(rows, cost_per_call=args.cost_per_call)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(args.out_md, payload)

    print("===== FULL-SYSTEM LABELED ACCURACY =====")
    print(f"Rows: {payload['rows']}")
    print(f"Overall: {payload['correct_answers']}/{payload['rows']} ({payload['overall_accuracy']:.3f})")
    print(f"KG: {payload['by_expected_route'].get('kg_analytics', {}).get('correct', 0)}/800 ({payload['kg_accuracy']:.3f})")
    print(f"DR: {payload['by_expected_route'].get('definition', {}).get('correct', 0)}/150 ({payload['ontology_accuracy']:.3f})")
    print(f"Advisory: {payload['by_expected_route'].get('advisory', {}).get('correct', 0)}/50 ({payload['advisory_accuracy']:.3f})")
    print(f"Deterministic: {payload['deterministic_correct']}/{payload['deterministic_questions']} ({payload['deterministic_accuracy']:.3f})")
    print(f"LLM fallback: {payload['llm_fallback_correct']}/{payload['llm_fallback_questions']} ({payload['llm_fallback_accuracy']:.3f})")
    print(f"CSV: {args.out_csv}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
