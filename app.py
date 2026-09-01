import os
import time
import json
import re
import uuid
import math
import sqlite3
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from rdflib import Graph, BNode, URIRef

from kg.advisory import (
    AdvisoryPlan,
    CURRENT_DEMAND_BY_REGION,
    FUTURE_DEMAND_BY_REGION,
    FUTURE_DEMAND_BY_TECHNOLOGY,
    FUTURE_DEMAND_BY_VEHICLE_TYPE,
    SHORTAGE_BY_SURVEY_GROUP,
    resolve_advisory_plan,
    synthesize_advisory_answer,
)
from kg.dr_ontology import (
    DEFAULT_DR_ONTOLOGY_PATH,
    dr_ontology_counts,
    dr_ontology_terms,
    search_dr_ontology_terms,
    route_dr_ontology_definition,
)
from kg.fuseki import fuseki_authorization_header, make_sparql_store
from kg.schema import load_schema
from kg.capabilities import DEFAULT_REGISTRY as CAPABILITY_REGISTRY
from llm.answer_synthesis import synthesize_answer
from llm.candidate_generation import generate_candidate_prompt
from llm.client import InfineonGPTClient, LLMAuthError, LLMClientError
from pipeline.qa import answer_question
from ranking.feature_extraction import extract_query_plan
from ranking.query_contract import compare_contracts, extract_query_contract, extract_question_contract
from visualization.interactive_graph import (
    build_graph_html,
    collect_answer_evidence_triples,
    collect_full_graph_triples,
    collect_query_subgraph_triples,
)


PROJECT_ROOT = Path(__file__).resolve().parent
# Load .env for Streamlit runs (same behavior as training/eval scripts).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()
except Exception:
    pass

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "data" / "infineon" / "schema.json"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "infineon" / "graph.ttl"
DEFAULT_ONTOLOGY_PATH = PROJECT_ROOT / "data" / "infineon" / "true_demand_ontology_extracted.ttl"
DEFAULT_FUSEKI_QUERY_URL = "http://localhost:3030/infineon/sparql"
DEFAULT_INTERACTIVE_TIME_BUDGET_SEC = 10.0
DEFAULT_INTERACTIVE_LLM_TIMEOUT_SEC = 6.0
DEFAULT_INTERACTIVE_QUERY_TIMEOUT_SEC = 3.0
INTERACTIVE_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="kgqa-ui")
SEMICONDUCTOR_DEMAND_BY_QUARTER_QUERY = """
SELECT ?quarterLabel ?regionName (SUM(?pct) AS ?totalPctChange) WHERE {
  ?d a survey:DemandForRegion ;
     survey:hasSurveyOrigin ?o ;
     survey:inRegion ?r ;
     survey:quarter ?q ;
     survey:totalDemandPercentageChange ?pct .
  ?o a survey:Semiconductor_Survey .
  ?r survey:regionName ?regionName .
  ?q survey:periodLabel ?quarterLabel .
}
GROUP BY ?quarterLabel ?regionName
ORDER BY ?quarterLabel ?regionName
""".strip()
REGIONAL_DEMAND_BY_SURVEY_QUERY = """
SELECT ?surveyGroup ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE {
  ?demandForRegion a survey:DemandForRegion ;
    survey:hasSurveyOrigin ?origin ;
    survey:inRegion ?region ;
    survey:totalDemand ?unitsSold .
  ?origin a ?surveyGroup .
  ?region a survey:Region ;
    survey:regionName ?regionName .
}
GROUP BY ?surveyGroup ?regionName
ORDER BY ?surveyGroup DESC(?totalDemand)
""".strip()
OEM_TOTAL_DEMAND_BY_REGION_QUERY = """
SELECT ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE {
  ?demandForRegion a survey:DemandForRegion ;
    survey:hasSurveyOrigin ?origin ;
    survey:inRegion ?region ;
    survey:totalDemand ?unitsSold .
  ?origin a survey:OEM_Survey .
  ?region a survey:Region ;
    survey:regionName ?regionName .
}
GROUP BY ?regionName
ORDER BY DESC(?totalDemand)
""".strip()
ACTUAL_VEHICLE_SALES_BY_MONTH_QUERY = """
SELECT ?monthLabel (SUM(?units) AS ?unitsSold) WHERE {
  ?obs a survey:VehicleSalesObservation ;
    survey:forTimePeriod ?month ;
    survey:isActualData true ;
    survey:unitsSold ?units .
  BIND(REPLACE(STR(?month), '^.*/', '') AS ?monthLabel)
}
GROUP BY ?monthLabel
ORDER BY ?monthLabel
""".strip()
FUTURE_SEMICONDUCTOR_DEMAND_BY_TECH_QUARTER_QUERY = """
SELECT ?techLabel ?quarter
  (SUM(IF(?baseline = "Option1", ?pct, 0)) AS ?Option1)
  (SUM(IF(?baseline = "Option2", ?pct, 0)) AS ?Option2)
  (SUM(IF(?baseline = "Option3", ?pct, 0)) AS ?Option3)
WHERE {
  {
    survey:SemiFutureDemand_Option1 a survey:FutureDemandAnalysis ;
      survey:hasSurveyOrigin survey:Semiconductor_Survey ;
      survey:hasAggregatedResult ?entry .
    ?entry a survey:FutureDemandAnalysis ;
      survey:analyzesTechnologyCategory ?tech ;
      survey:forTimePeriod ?period ;
      survey:percentageChange ?pct .
    BIND("Option1" AS ?baseline)
  }
  UNION
  {
    survey:SemiFutureDemand_Option2 a survey:FutureDemandAnalysis ;
      survey:hasSurveyOrigin survey:Semiconductor_Survey ;
      survey:hasAggregatedResult ?entry .
    ?entry a survey:FutureDemandAnalysis ;
      survey:analyzesTechnologyCategory ?tech ;
      survey:forTimePeriod ?period ;
      survey:percentageChange ?pct .
    BIND("Option2" AS ?baseline)
  }
  UNION
  {
    survey:SemiFutureDemand_Option3 a survey:FutureDemandAnalysis ;
      survey:hasSurveyOrigin survey:Semiconductor_Survey ;
      survey:hasAggregatedResult ?entry .
    ?entry a survey:FutureDemandAnalysis ;
      survey:analyzesTechnologyCategory ?tech ;
      survey:forTimePeriod ?period ;
      survey:percentageChange ?pct .
    BIND("Option3" AS ?baseline)
  }
  FILTER(STRSTARTS(STR(?tech), STR(survey:TechCategory_)))
  OPTIONAL { ?period survey:periodLabel ?qLabelRaw . }
  BIND(REPLACE(COALESCE(?qLabelRaw, STRAFTER(STR(?period), "survey:")), "_", " ") AS ?quarter)
  BIND(
    REPLACE(
      REPLACE(
        REPLACE(
          REPLACE(
            REPLACE(STRAFTER(STR(?tech), "TechCategory_"), "%3C%3D", "<="),
            "_to_%3C", " to <"
          ),
          "_or_greater", " or greater"
        ),
        "_", " "
      ),
      "lte 7nm", "<= 7nm"
    ) AS ?techLabel
  )
}
GROUP BY ?techLabel ?quarter
ORDER BY ?techLabel ?quarter
""".strip()
FINAL_SYSTEM_EVALUATION = {
    "benchmark_questions": 1000,
    "kg_questions": 800,
    "ontology_definition_questions": 150,
    "advisory_questions": 50,
    "overall_accuracy": 0.894,
    "kg_accuracy": 0.87,
    "ontology_accuracy": 1.0,
    "advisory_accuracy": 0.96,
    "correct_answers": 894,
    "incorrect_answers": 106,
    "deterministic_questions": 514,
    "deterministic_correct": 503,
    "deterministic_incorrect": 11,
    "deterministic_accuracy": 0.9785992217898832,
    "llm_fallback_questions": 486,
    "llm_fallback_correct": 391,
    "llm_fallback_incorrect": 95,
    "llm_fallback_accuracy": 0.8045267489711934,
    "llm_calls": 486,
    "llm_call_reduction": 0.514,
    "estimated_cost_eur": 97.20,
    "all_llm_baseline_cost_eur": 200.00,
    "estimated_savings_eur": 102.80,
    "warm_cache_llm_calls": 35,
    "warm_cache_cost_eur": 7.00,
    "warm_cache_call_reduction": 0.965,
    "fallback_selection_accuracy": 0.683,
    "fallback_candidate_coverage": 0.913,
    "failure_easy": 7,
    "failure_medium": 25,
    "failure_hard": 74,
    "failure_advisory_interpretation": 2,
    "failure_autonomous_driving": 24,
    "failure_current_demand": 22,
    "failure_vehicle_sales": 14,
    "failure_future_demand": 16,
}
APP_LOG_DIR = PROJECT_ROOT / "logs"
SESSION_LOG_PATH = APP_LOG_DIR / "kgqa_sessions.jsonl"
FEEDBACK_LOG_PATH = APP_LOG_DIR / "kgqa_feedback.jsonl"
USER_AUDIT_LOG_PATH = APP_LOG_DIR / "kgqa_user_audit.jsonl"
USER_AUDIT_DB_PATH = APP_LOG_DIR / "kgqa_user_audit.sqlite3"
DEFAULT_ML_MODEL_PATHS = [
    PROJECT_ROOT / "ranking" / "models" / "final1000_wf_ranker_current.json",
    PROJECT_ROOT / "ranking" / "models" / "final1000_wf_ranker_scope_origin.json",
    PROJECT_ROOT / "ranking" / "models" / "final1000_wf_ranker_shortage_grouped.json",
    PROJECT_ROOT / "ranking" / "models" / "final1000_wf_ranker_shape_features.json",
    PROJECT_ROOT / "ranking" / "models" / "infineon_np_tfidf_ranker_entitylink.json",
    PROJECT_ROOT / "ranking" / "models" / "infineon_np_tfidf_ranker.json",
    PROJECT_ROOT / "ranking" / "models" / "infineon_ranker.joblib",
]
DEFAULT_AMBIGUITY_CONFIG_PATHS = [
    PROJECT_ROOT / "ranking" / "models" / "infineon_ambiguity_config_500.json",
    PROJECT_ROOT / "ranking" / "models" / "infineon_ambiguity_config.json",
]
DEFAULT_CONFIDENCE_ROUTING_REPORT_PATHS = [
    PROJECT_ROOT / "results" / "final1000_current_confidence_routing.json",
    PROJECT_ROOT / "results" / "final1000_wf_test_scope_origin_confidence_routing_safety.json",
    PROJECT_ROOT / "results" / "final1000_wf_test_scope_origin_confidence_routing_sorted_v2.json",
    PROJECT_ROOT / "results" / "final1000_wf_test_scope_origin_confidence_routing_v2.json",
]
KG_AUTOCOMPLETE_COMPONENT = components.declare_component(
    "kg_autocomplete",
    path=str(PROJECT_ROOT / "ui_components" / "kg_autocomplete"),
)
DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


def _default_ml_model_path() -> str:
    for p in DEFAULT_ML_MODEL_PATHS:
        if p.exists():
            return str(p)
    return str(DEFAULT_ML_MODEL_PATHS[0])


def _default_ambiguity_config_path() -> str:
    env_path = (os.environ.get("INFINEON_AMBIGUITY_CONFIG") or "").strip()
    if env_path:
        return env_path
    for p in DEFAULT_AMBIGUITY_CONFIG_PATHS:
        if p.exists():
            return str(p)
    return str(DEFAULT_AMBIGUITY_CONFIG_PATHS[0])


def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query


def _format_graph_value(value: object) -> str:
    text = str(value)
    ns = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"
    if text.startswith(ns):
        text = text[len(ns):]
    if text in {"OEM_Survey", "Tier1_Survey", "Semiconductor_Survey"}:
        return {
            "OEM_Survey": "OEM",
            "Tier1_Survey": "Tier1",
            "Semiconductor_Survey": "Semiconductor",
        }[text]
    return text


def _format_sparql_for_display(query: str) -> str:
    q = " ".join(str(query or "").split())
    replacements = [
        (" WHERE { ", " WHERE {\n  "),
        (" ; ", " ;\n    "),
        (" . ", " .\n  "),
        (" FILTER", "\n  FILTER"),
        (" VALUES", "\n  VALUES"),
        (" BIND", "\n  BIND"),
        (" OPTIONAL", "\n  OPTIONAL"),
        (" UNION", "\n  UNION"),
        (" } GROUP BY ", "\n} GROUP BY "),
        (" } ORDER BY ", "\n} ORDER BY "),
        (" GROUP BY ", "\nGROUP BY "),
        (" ORDER BY ", "\nORDER BY "),
        (" LIMIT ", "\nLIMIT "),
    ]
    for old, new in replacements:
        q = q.replace(old, new)
    return q


def _load_schema_from_path(schema_path: str):
    cleaned = (schema_path or "").strip()
    if not cleaned:
        cleaned = str(DEFAULT_SCHEMA_PATH)
    if not os.path.exists(cleaned):
        raise FileNotFoundError(f"Schema path not found: {cleaned}")
    return load_schema(cleaned)


def _active_fuseki_query_url() -> str:
    return os.getenv("FUSEKI_QUERY_URL", "").strip()


@st.cache_data(show_spinner=False, ttl=15)
def _fuseki_endpoint_available(fuseki_query_url: str) -> bool:
    url = (fuseki_query_url or "").strip()
    if not url:
        return False
    try:
        request = urllib.request.Request(url, method="GET")
        auth_header = fuseki_authorization_header()
        if auth_header:
            request.add_header("Authorization", auth_header)
        with urllib.request.urlopen(request, timeout=0.6):
            return True
    except urllib.error.HTTPError:
        # Fuseki may reject GET without a query but still be reachable.
        return True
    except Exception:
        return False


def _usable_fuseki_query_url(fuseki_query_url: str) -> str:
    url = (fuseki_query_url or "").strip()
    if not url:
        return ""
    return url if _fuseki_endpoint_available(url) else ""


def _graph_backend_available(graph_path: str) -> bool:
    return bool(_active_fuseki_query_url()) or bool(graph_path and os.path.exists(graph_path))


def _interactive_time_budget_sec() -> float:
    try:
        return max(1.0, float(os.getenv("KGQA_INTERACTIVE_TIME_BUDGET_SEC", str(DEFAULT_INTERACTIVE_TIME_BUDGET_SEC))))
    except ValueError:
        return DEFAULT_INTERACTIVE_TIME_BUDGET_SEC


def _interactive_query_timeout_sec() -> float:
    try:
        return max(0.5, float(os.getenv("KGQA_INTERACTIVE_QUERY_TIMEOUT_SEC", str(DEFAULT_INTERACTIVE_QUERY_TIMEOUT_SEC))))
    except ValueError:
        return DEFAULT_INTERACTIVE_QUERY_TIMEOUT_SEC


def _interactive_budget_exceeded(started_at: float, reserve_s: float = 0.0) -> bool:
    return (time.perf_counter() - started_at + float(reserve_s)) >= _interactive_time_budget_sec()


def _interactive_remaining_sec(started_at: float, reserve_s: float = 0.0) -> float:
    return max(0.1, _interactive_time_budget_sec() - (time.perf_counter() - started_at) - float(reserve_s))


def _run_with_timeout(fn, timeout_s: float, *, label: str = "operation"):
    timeout_s = max(0.1, float(timeout_s))
    future = INTERACTIVE_EXECUTOR.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"{label} exceeded {timeout_s:.1f}s") from exc


@contextmanager
def _temporary_socket_timeout(timeout_s: float):
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(float(timeout_s))
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


@st.cache_resource(show_spinner=False)
def _load_graph_cached(graph_path: str, fuseki_query_url: str = "") -> Graph:
    if fuseki_query_url.strip():
        return Graph(store=make_sparql_store(fuseki_query_url.strip()))
    g = Graph()
    g.parse(graph_path, format="turtle")
    return g


def _load_active_graph(graph_path: str) -> Graph:
    return _load_graph_cached(graph_path, _active_fuseki_query_url())


def _execute_query_preview(
    graph: Graph,
    query: str,
    max_rows: int = 200,
) -> Tuple[List[Dict[str, str]], bool]:
    def _query_to_rows() -> Tuple[List[Dict[str, str]], bool]:
        with _temporary_socket_timeout(_interactive_query_timeout_sec()):
            results = graph.query(_ensure_prefixes(query))
        rows: List[Dict[str, str]] = []
        truncated = False
        for idx, row in enumerate(results):
            if idx >= max_rows:
                truncated = True
                break
            if hasattr(row, "asdict"):
                rd = row.asdict()
                rows.append({str(k): _format_graph_value(v) for k, v in rd.items()})
            else:
                rows.append({f"col{j + 1}": _format_graph_value(v) for j, v in enumerate(row)})
        return rows, truncated

    return _run_with_timeout(
        _query_to_rows,
        _interactive_query_timeout_sec(),
        label="SPARQL query execution",
    )


@st.cache_data(show_spinner=False)
def _preview_query_rows_cached(
    graph_path: str,
    fuseki_query_url: str,
    query: str,
    max_rows: int = 3,
) -> Tuple[List[Dict[str, str]], str]:
    if not query.strip() or not (fuseki_query_url.strip() or (graph_path and os.path.exists(graph_path))):
        return [], "graph_backend_or_query_missing"
    try:
        def _load_and_preview() -> Tuple[List[Dict[str, str]], bool]:
            graph = _load_graph_cached(graph_path, fuseki_query_url)
            return _execute_query_preview(graph, query, max_rows=max_rows)

        rows, _truncated = _run_with_timeout(
            _load_and_preview,
            max(_interactive_query_timeout_sec(), 0.75),
            label="graph preview",
        )
        return rows, ""
    except Exception as exc:
        return [], str(exc)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _init_user_audit_db(path: Path = USER_AUDIT_DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_audit (
                request_id TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                question TEXT NOT NULL,
                route TEXT,
                route_family TEXT,
                confidence_index REAL,
                confidence_label TEXT,
                selected_source TEXT,
                selected_query TEXT,
                answer_text TEXT,
                graph_row_count INTEGER,
                graph_error TEXT,
                latency_s REAL,
                llm_estimated_calls INTEGER,
                advisory_plan_id TEXT,
                dr_term TEXT,
                row_preview_json TEXT,
                metadata_json TEXT
            )
            """
        )
        conn.commit()


def _write_user_audit_record(payload: Dict[str, Any]) -> None:
    _append_jsonl(USER_AUDIT_LOG_PATH, payload)
    _init_user_audit_db(USER_AUDIT_DB_PATH)
    with sqlite3.connect(USER_AUDIT_DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO qa_audit (
                request_id,
                timestamp_utc,
                question,
                route,
                route_family,
                confidence_index,
                confidence_label,
                selected_source,
                selected_query,
                answer_text,
                graph_row_count,
                graph_error,
                latency_s,
                llm_estimated_calls,
                advisory_plan_id,
                dr_term,
                row_preview_json,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("request_id"),
                payload.get("timestamp_utc"),
                payload.get("question"),
                payload.get("route"),
                payload.get("route_family"),
                payload.get("confidence_index"),
                payload.get("confidence_label"),
                payload.get("selected_source"),
                payload.get("selected_query"),
                payload.get("answer_text"),
                payload.get("graph_row_count"),
                payload.get("graph_error"),
                payload.get("latency_s"),
                payload.get("llm_estimated_calls"),
                payload.get("advisory_plan_id"),
                payload.get("dr_term"),
                json.dumps(payload.get("row_preview") or [], ensure_ascii=False, default=str),
                json.dumps(payload.get("metadata") or {}, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()


def _route_label(result: Dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return "unknown"
    route = result.get("confidence_route")
    if isinstance(route, dict):
        return str(route.get("route") or "unknown")
    if isinstance(result.get("request_clarification"), dict):
        return "request_clarification"
    if isinstance(result.get("clarification"), dict) and result["clarification"].get("needs_clarification"):
        return "legacy_clarification"
    return "auto_answer"


def _route_family(result: Dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return "unknown"
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if metadata.get("dr_ontology_route"):
        return "dr_ontology_definition"
    if metadata.get("advisory_route"):
        return "advisory"
    if metadata.get("direct_capability_route"):
        return "kg_direct_template"
    if metadata.get("guided_query"):
        return "guided_template"
    return "llm_fallback"


def _session_log_payload(
    *,
    request_id: str,
    question: str,
    result: Dict[str, Any],
    selected_query: str,
    graph_rows: List[Dict[str, str]],
    graph_exec_error: str,
    latency_s: float,
    latency_breakdown: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    confidence_route = result.get("confidence_route") if isinstance(result, dict) else None
    route = _route_label(result)
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    llm_calls = _estimated_llm_calls_from_metadata(metadata)
    return {
        "request_id": request_id,
        "timestamp_utc": _utc_now_iso(),
        "question": question,
        "route": route,
        "score1": confidence_route.get("score1") if isinstance(confidence_route, dict) else None,
        "score2": confidence_route.get("score2") if isinstance(confidence_route, dict) else None,
        "margin": confidence_route.get("margin") if isinstance(confidence_route, dict) else None,
        "selected_source": _selected_candidate_source(result, selected_query),
        "selected_query": selected_query,
        "graph_row_count": len(graph_rows or []),
        "graph_error": graph_exec_error,
        "latency_s": round(float(latency_s), 3),
        "latency_breakdown": {
            key: round(float(value), 3)
            for key, value in (latency_breakdown or {}).items()
            if isinstance(value, (int, float))
        },
        "candidate_count": len(result.get("candidates") or []),
        "llm": {
            "skipped": bool(metadata.get("llm_skipped")),
            "cache_enabled": bool(metadata.get("llm_cache_enabled")),
            "cache_hit": bool(metadata.get("llm_cache_hit")),
            "full_schema_generation_attempted": bool(metadata.get("full_schema_generation_attempted")),
            "full_schema_cache_hit": bool(metadata.get("full_schema_llm_cache_hit")),
            "estimated_calls": llm_calls,
        },
        "schema_route": {
            "applied": bool(metadata.get("schema_slicing_applied")),
            "confidence": metadata.get("schema_slice_confidence"),
            "families": metadata.get("schema_slice_names") or [],
        },
    }


def _user_audit_payload(
    *,
    request_id: str,
    question: str,
    result: Dict[str, Any],
    selected_query: str,
    graph_rows: List[Dict[str, str]],
    graph_exec_error: str,
    graph_answer: str,
    latency_s: float,
) -> Dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    confidence_value = _confidence_index_from_result(result, graph_rows=graph_rows, graph_exec_error=graph_exec_error)
    confidence_label = _confidence_label(confidence_value)
    return {
        "request_id": request_id,
        "timestamp_utc": _utc_now_iso(),
        "question": question,
        "route": _route_label(result),
        "route_family": _route_family(result),
        "confidence_index": round(confidence_value, 4),
        "confidence_label": confidence_label,
        "selected_source": _selected_candidate_source(result, selected_query),
        "selected_query": selected_query,
        "answer_text": str(graph_answer or result.get("answer") or "")[:5000],
        "graph_row_count": len(graph_rows or []),
        "graph_error": graph_exec_error,
        "latency_s": round(float(latency_s), 3),
        "llm_estimated_calls": _estimated_llm_calls_from_metadata(metadata),
        "advisory_plan_id": metadata.get("advisory_plan_id"),
        "dr_term": metadata.get("matched_term"),
        "row_preview": list(graph_rows or [])[:10],
        "metadata": {
            "policy": result.get("policy"),
            "selection_reason": result.get("selection_reason"),
            "answerability": result.get("answerability"),
            "confidence_route": result.get("confidence_route"),
            "candidate_count": len(result.get("candidates") or []),
            "llm_skipped": bool(metadata.get("llm_skipped")),
            "llm_cache_hit": bool(metadata.get("llm_cache_hit")),
        },
    }


def _estimated_llm_calls_from_metadata(metadata: Dict[str, Any]) -> int:
    if metadata.get("llm_skipped") or metadata.get("guided_query"):
        return 0

    calls = 0
    if metadata.get("llm_cache_enabled") and metadata.get("llm_cache_hit"):
        calls += 0
    else:
        calls += 1

    if metadata.get("full_schema_generation_attempted"):
        if metadata.get("llm_cache_enabled") and metadata.get("full_schema_llm_cache_hit"):
            calls += 0
        else:
            calls += 1
    return calls


def _selected_candidate_source(result: Dict[str, Any], selected_query: str) -> str:
    selected_key = " ".join(str(selected_query or "").split()).lower()
    if not selected_key:
        return ""
    for candidate in result.get("candidates") or []:
        query_key = " ".join(str(candidate.get("query") or "").split()).lower()
        if query_key == selected_key:
            return str(candidate.get("source") or "")
    return ""


@st.cache_data(show_spinner=False, ttl=30)
def _system_health_snapshot(
    *,
    schema_path: str,
    graph_path: str,
    fuseki_query_url: str,
    model_path: str,
    api_url: str,
    api_key_present: bool,
    user_credentials_present: bool,
) -> Dict[str, Dict[str, str]]:
    health: Dict[str, Dict[str, str]] = {}
    health["Schema"] = {
        "status": "ok" if schema_path and os.path.exists(schema_path) else "missing",
        "detail": schema_path or "not configured",
    }
    health["Model"] = {
        "status": "ok" if model_path and os.path.exists(model_path) else "missing",
        "detail": model_path or "not configured",
    }
    if fuseki_query_url:
        try:
            graph = Graph(store=make_sparql_store(fuseki_query_url))
            _execute_query_preview(graph, "SELECT * WHERE { ?s ?p ?o } LIMIT 1", max_rows=1)
            health["Graph"] = {"status": "ok", "detail": "Fuseki endpoint reachable"}
        except Exception as exc:
            health["Graph"] = {"status": "error", "detail": f"Fuseki error: {exc}"}
    else:
        health["Graph"] = {
            "status": "ok" if graph_path and os.path.exists(graph_path) else "missing",
            "detail": graph_path or "not configured",
        }
    health["LLM config"] = {
        "status": "ok" if api_url and (api_key_present or user_credentials_present) else "missing",
        "detail": "token or refresh credentials present" if (api_key_present or user_credentials_present) else "missing token/credentials",
    }
    return health


def _render_system_status(
    *,
    schema_path: str,
    graph_path: str,
    fuseki_query_url: str,
    model_path: str,
    api_url: str,
    api_key: str,
) -> None:
    user_credentials_present = bool(
        os.environ.get("USER_LLM")
        or os.environ.get("INFINEON_API_USER")
    )
    try:
        health = _system_health_snapshot(
            schema_path=schema_path,
            graph_path=graph_path,
            fuseki_query_url=fuseki_query_url,
            model_path=model_path,
            api_url=api_url,
            api_key_present=bool(api_key.strip() or os.environ.get("INFINEON_API_KEY")),
            user_credentials_present=user_credentials_present,
        )
    except Exception as exc:
        st.caption(f"System status unavailable: {exc}")
        return
    icon = {"ok": "OK", "missing": "Missing", "error": "Error"}
    for name, row in health.items():
        status = str(row.get("status") or "unknown")
        st.caption(f"{name}: {icon.get(status, status)} - {row.get('detail', '')}")


def _render_recommended_settings() -> None:
    st.markdown(
        """
Recommended demo configuration:

- Fuseki endpoint enabled
- Use ML ranking: on
- ML policy: `all`
- Fast interactive mode: on
- Candidate diagnostics: off
- Confidence routing: on
- Auto-answer score: `0.90`
- Family-aware schema routing: optional, retry full schema: off
"""
    )


def _render_feedback_panel() -> None:
    result = st.session_state.get("last_qa_result")
    question = str(st.session_state.get("last_question") or "").strip()
    selected_query = str(st.session_state.get("last_selected_query") or "").strip()
    if not isinstance(result, dict) or not question:
        return

    st.subheader("Feedback")
    st.caption("Optional: record whether the answer matched your intended interpretation.")
    col_ok, col_bad, col_intent = st.columns(3)
    request_id = str(st.session_state.get("last_request_id") or "")
    base_payload = {
        "request_id": request_id,
        "timestamp_utc": _utc_now_iso(),
        "question": question,
        "route": _route_label(result),
        "selected_query": selected_query,
        "selected_source": _selected_candidate_source(result, selected_query),
        "score": (result.get("confidence_route") or {}).get("score1") if isinstance(result.get("confidence_route"), dict) else None,
        "margin": (result.get("confidence_route") or {}).get("margin") if isinstance(result.get("confidence_route"), dict) else None,
    }
    if col_ok.button("Answer was correct", use_container_width=True, key="feedback_correct"):
        _append_jsonl(FEEDBACK_LOG_PATH, {**base_payload, "feedback": "correct"})
        st.success("Feedback saved.")
    if col_bad.button("Answer was wrong", use_container_width=True, key="feedback_wrong"):
        _append_jsonl(FEEDBACK_LOG_PATH, {**base_payload, "feedback": "wrong"})
        st.warning("Feedback saved for review.")
    if col_intent.button("Not what I meant", use_container_width=True, key="feedback_intent"):
        _append_jsonl(FEEDBACK_LOG_PATH, {**base_payload, "feedback": "wrong_intent"})
        st.warning("Intent feedback saved for review.")


def _render_selection_explainability(result: Dict[str, Any]) -> None:
    expl = result.get("selection_explanation")
    if not isinstance(expl, dict):
        st.info("No explainability payload available.")
        return

    st.subheader("Why This Query Was Selected")
    left, mid, right = st.columns(3)
    left.metric("Policy Mode", str(expl.get("policy_mode", "unknown")))
    mid.metric("Selected Policy", str(expl.get("selected_policy", "unknown")))
    right.metric("Valid Candidates", f"{expl.get('valid_candidate_count', 0)}/{expl.get('candidate_count', 0)}")

    regime = expl.get("predicted_regime")
    entropy = expl.get("predicted_entropy")
    if regime is not None:
        st.caption(
            f"Predicted ambiguity regime: {regime}"
            + (f" (entropy={float(entropy):.3f})" if entropy is not None else "")
        )
    metadata = result.get("metadata") or {}
    predicted_plan = metadata.get("predicted_query_plan_labels") or []
    if predicted_plan:
        st.caption("Predicted query-plan labels: " + ", ".join(map(str, predicted_plan[:16])))

    st.write(f"Selection reason: `{expl.get('selection_reason', 'n/a')}`")
    st.write(
        "Selected query source: "
        f"`{expl.get('selected_from', 'n/a')}` | "
        "Rank in preference order: "
        f"`{expl.get('selected_rank_in_preference_order', 'n/a')}`"
    )
    selected_coverage = expl.get("selected_coverage")
    if isinstance(selected_coverage, dict):
        cov = float(selected_coverage.get("coverage_score", 1.0))
        missing = selected_coverage.get("missing") or []
        st.write(f"Semantic coverage: `{cov:.2f}`")
        if missing:
            st.warning(
                "Selected query is missing requested concepts: "
                + ", ".join(map(str, missing))
            )
    if "selected_execution_has_rows" in expl:
        st.write(f"Execution returned rows during selection: `{expl.get('selected_execution_has_rows')}`")
        unbound = expl.get("selected_execution_unbound_vars") or []
        if unbound:
            st.warning(
                "Selected query has unbound projected variables: "
                + ", ".join(map(str, unbound))
            )
        if expl.get("selected_execution_error"):
            st.warning(f"Selection execution check failed: {expl.get('selected_execution_error')}")

    selected_errors = expl.get("selected_query_errors") or []
    if selected_errors:
        st.warning("Selected query had validation issues before fallback:")
        st.json(selected_errors)

    rows = expl.get("candidates") or []
    if rows:
        st.caption("Candidate diagnostics")
        st.dataframe(rows, width="stretch")
    elif not expl.get("candidate_diagnostics_included", True):
        st.caption("Candidate diagnostics skipped. Enable 'Run candidate diagnostics' for per-candidate execution checks.")


def _confidence_summary(result: Dict[str, Any]) -> Tuple[str, str]:
    expl = result.get("selection_explanation")
    clarification = result.get("clarification")
    answerability = result.get("answerability")
    if isinstance(answerability, dict):
        status = str(answerability.get("status", ""))
        if status in {
            "selected_query_empty_but_alternatives_exist",
            "selected_query_semantic_mismatch_but_alternatives_exist",
        }:
            return (
                "Low",
                "The selected query was not answerable for the requested meaning while another valid interpretation returned data.",
            )
        if status in {
            "query_invalid",
            "query_execution_error",
            "generation_failure",
            "selected_query_semantic_mismatch",
        }:
            return ("Low", str(answerability.get("reason", "The system could not produce an answerable graph query.")))
        if status == "no_rows_for_generated_queries":
            return (
                "Low",
                "No checked valid interpretation returned graph rows for this request.",
            )
    if not isinstance(expl, dict):
        return "Unknown", "No selection diagnostics are available."

    coverage = 0.0
    selected_coverage = expl.get("selected_coverage")
    if isinstance(selected_coverage, dict):
        coverage = float(selected_coverage.get("coverage_score", 0.0))
    valid = bool(expl.get("selected_query_valid"))
    has_rows = expl.get("selected_execution_has_rows")
    entropy = expl.get("predicted_entropy")
    clarification_needed = bool(
        isinstance(clarification, dict) and clarification.get("needs_clarification")
    )
    clarified = bool(st.session_state.get("clarification_choice_id"))

    if clarification_needed and not clarified:
        return (
            "Low",
            "The candidates express materially different interpretations, so the system needs clarification before it can answer confidently.",
        )
    if valid and coverage >= 0.99 and has_rows is not False and clarified:
        return (
            "High",
            "You selected the intended interpretation, and the chosen query is valid, covers the requested concepts, and returned results.",
        )
    if valid and coverage >= 0.99 and has_rows is not False:
        if entropy is not None and float(entropy) >= 0.66:
            return (
                "Medium",
                "The selected query is valid and covers the request, but candidate uncertainty is still elevated.",
            )
        return (
            "High",
            "The selected query is valid, covers the requested concepts, and returned results.",
        )
    if valid and coverage >= 0.75:
        return (
            "Medium",
            "The selected query is valid, but some requested concepts are only partially covered.",
        )
    return (
        "Low",
        "The selected query has limited support from the available selection signals.",
    )


def _bounded_score(value: object) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= raw <= 1.0:
        return raw
    # Some ranker scores are unbounded logits. Map them to 0..1 for display.
    try:
        return 1.0 / (1.0 + math.exp(-raw))
    except OverflowError:
        return 1.0 if raw > 0 else 0.0


def _confidence_index_from_route(confidence_route: Dict[str, Any]) -> float:
    score = _bounded_score(confidence_route.get("score1"))
    try:
        margin = float(confidence_route.get("margin") or 0.0)
    except (TypeError, ValueError):
        margin = 0.0
    margin_signal = max(0.0, min(1.0, margin / (margin + 0.15))) if margin > 0 else 0.0
    blocking_flags = confidence_route.get("blocking_safety_flags") or []
    safety_penalty = 0.35 if blocking_flags else 0.0
    if confidence_route.get("route") != "auto_answer":
        safety_penalty = max(safety_penalty, 0.45)
    return max(0.0, min(1.0, 0.85 * score + 0.15 * margin_signal - safety_penalty))


def _confidence_index_from_result(
    result: Dict[str, Any],
    *,
    graph_rows: List[Dict[str, str]] | None = None,
    graph_exec_error: str = "",
) -> float:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    has_rows = bool(graph_rows)
    if graph_exec_error:
        return 0.12
    if metadata.get("dr_ontology_route"):
        return 0.98
    if metadata.get("direct_capability_route"):
        return 0.96 if has_rows else 0.72
    if metadata.get("guided_query"):
        return 0.94 if has_rows else 0.68
    if metadata.get("advisory_route"):
        return 0.91 if has_rows else 0.62
    confidence_route = result.get("confidence_route")
    if isinstance(confidence_route, dict):
        value = _confidence_index_from_route(confidence_route)
        if has_rows:
            value = min(0.97, value + 0.04)
        return value
    confidence, _reason = _confidence_summary(result)
    return {"High": 0.86, "Medium": 0.64, "Low": 0.31}.get(confidence, 0.5)


def _confidence_label(value: float) -> str:
    if value >= 0.85:
        return "High"
    if value >= 0.60:
        return "Medium"
    return "Low"


def _confidence_percent(value: float) -> str:
    return f"{100.0 * max(0.0, min(1.0, value)):.0f}%"


def _confidence_formula_note(result: Dict[str, Any]) -> str:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if metadata.get("dr_ontology_route"):
        return "Definition confidence is high because the question matched one declared Digital Reference ontology term and no LLM generation was used."
    if metadata.get("advisory_route"):
        return "Advisory confidence is based on a deterministic graph query, non-empty evidence rows, and a conservative template that reports a planning signal rather than making a decision."
    if metadata.get("direct_capability_route"):
        return "Direct-template confidence is high because the capability resolver found one supported metric, dimension, and aggregation path and the graph returned evidence."
    if metadata.get("guided_query"):
        return "Guided-builder confidence is high because the query comes from a pre-validated graph-supported path."
    return (
        "Fallback confidence combines the ranker score, the separation from competing candidates, "
        "execution evidence, and safety flags. Lower values mean that alternatives remain plausible."
    )


def _render_compact_explainability(result: Dict[str, Any]) -> None:
    request_route = result.get("request_route")
    if isinstance(request_route, dict) and request_route.get("route") != "kg_query":
        st.subheader("Why This Answer")
        left, right = st.columns([1, 3])
        left.metric("Confidence", str(request_route.get("confidence", "Unknown")))
        right.write(str(request_route.get("reason", "The request was handled before graph querying.")))
        return

    confidence_route = result.get("confidence_route")
    if isinstance(confidence_route, dict):
        st.subheader("Why This Answer")
        route = str(confidence_route.get("route", "unknown")).replace("_", " ")
        graph_rows = list(st.session_state.get("last_graph_rows") or [])
        graph_exec_error = ""
        confidence_value = _confidence_index_from_result(
            result,
            graph_rows=graph_rows,
            graph_exec_error=graph_exec_error,
        )
        left, right = st.columns([1.2, 3])
        left.metric("Decision", route)
        left.metric("Confidence", _confidence_percent(confidence_value))
        left.caption(_confidence_label(confidence_value))
        reason = str(confidence_route.get("reason") or "").strip()
        score = confidence_route.get("score1")
        margin = confidence_route.get("margin")
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if metadata.get("advisory_route"):
            plan_title = str(metadata.get("advisory_title") or "advisory template")
            right.write(
                f"The system selected the deterministic **{plan_title}** route. "
                "It executed a fixed graph query, sorted the returned evidence by the relevant metric, "
                "and converted the leading signal into conservative planning guidance. "
                "The recommendation is therefore tied to returned graph rows, not free-form generation."
            )
        elif metadata.get("dr_ontology_route"):
            term = str(metadata.get("matched_term") or "the requested ontology term")
            right.write(
                f"The system treated this as an ontology definition request and matched **{term}** "
                "in the Digital Reference ontology. The answer was retrieved deterministically from "
                "ontology labels, definitions, type, domain/range, and hierarchy metadata."
            )
        elif metadata.get("direct_capability_route"):
            right.write(
                "The capability resolver found a single graph-supported interpretation for the requested "
                "metric, dimension, and aggregation. Because that template returned graph evidence, the LLM was skipped."
            )
        elif reason:
            right.write(reason[0].upper() + reason[1:] if reason else reason)
        else:
            right.write("The selected query passed the confidence policy and safety checks.")
        right.caption(_confidence_formula_note(result))
        score_bits = []
        try:
            score_bits.append(f"score={float(score):.3f}")
        except (TypeError, ValueError):
            pass
        try:
            score_bits.append(f"margin={float(margin):.3f}")
        except (TypeError, ValueError):
            pass
        if score_bits:
            st.caption("Selection confidence: " + ", ".join(score_bits))
        if graph_rows:
            st.caption(f"Evidence check: the selected path returned {len(graph_rows)} row(s).")
        flags = confidence_route.get("safety_flags") or []
        if flags:
            st.caption("Technical safety notes are logged for review by the deployment owner.")
        return

    expl = result.get("selection_explanation")
    if not isinstance(expl, dict):
        return
    confidence, confidence_reason = _confidence_summary(result)
    selected_coverage = expl.get("selected_coverage") or {}
    coverage = float(selected_coverage.get("coverage_score", 0.0))
    reason_parts = []
    if bool(expl.get("selected_query_valid")):
        reason_parts.append("it is structurally valid")
    if coverage >= 0.99:
        reason_parts.append("it covers the requested concepts")
    elif coverage > 0:
        reason_parts.append(f"semantic coverage is {coverage:.2f}")
    if expl.get("selected_execution_has_rows") is True:
        reason_parts.append("it returned graph results")

    st.subheader("Why This Answer")
    left, right = st.columns([1, 3])
    confidence_value = _confidence_index_from_result(
        result,
        graph_rows=list(st.session_state.get("last_graph_rows") or []),
    )
    left.metric("Confidence", _confidence_percent(confidence_value))
    left.caption(confidence)
    right.write(
        "Selected because "
        + (", ".join(reason_parts) if reason_parts else "it ranked highest among the available candidates")
        + "."
    )
    st.caption(confidence_reason)


def _render_answer_block(
    *,
    answer_text: str,
    selected_query: str,
    graph_rows: List[Dict[str, str]],
    graph_exec_error: str,
    execute_selected: bool,
    answerability: Dict[str, Any] | None = None,
) -> None:
    st.subheader("Answer")
    if isinstance(answerability, dict):
        status = str(answerability.get("status", "unknown"))
        reason = str(answerability.get("reason", "")).strip()
        if status == "answer_available":
            st.caption("Answerability: answer available from graph execution.")
        elif status in {
            "selected_query_empty_but_alternatives_exist",
            "selected_query_semantic_mismatch_but_alternatives_exist",
        }:
            st.warning(reason)
            alternatives = answerability.get("nonempty_alternatives") or []
            if alternatives:
                st.write("I found answerable alternative interpretations:")
                chosen_alternative: Dict[str, Any] | None = None
                for idx, alternative in enumerate(alternatives, start=1):
                    label = str(alternative.get("label") or f"Alternative {idx}")
                    preview = str(alternative.get("preview") or "This interpretation returned graph rows.")
                    row_count = alternative.get("row_count")
                    with st.container(border=True):
                        st.markdown(f"**{label}**")
                        st.write(preview)
                        if row_count is not None:
                            st.caption(f"Rows found in graph: {row_count}")
                        if st.button(
                            "Use this interpretation",
                            key=f"answerability_alt_{idx}_{abs(hash(str(alternative.get('query', ''))))}",
                            use_container_width=True,
                        ):
                            chosen_alternative = alternative
                if chosen_alternative is not None:
                    selected_query = str(chosen_alternative.get("query", "") or "").strip()
                    graph_rows = list(chosen_alternative.get("preview_rows") or [])
                    preview = str(chosen_alternative.get("preview") or "").strip()
                    answer_text = f"Answer: {preview}" if preview and not preview.lower().startswith("answer:") else preview
                    graph_exec_error = ""
                    execute_selected = True
                    st.session_state["last_selected_query"] = selected_query
                    st.session_state["last_graph_rows"] = graph_rows
                    st.session_state["last_graph_answer"] = answer_text
                    st.success(f"Using alternative interpretation: {chosen_alternative.get('label')}")
                with st.expander("Technical alternative details", expanded=False):
                    st.json(alternatives)
        elif status in {
            "no_rows_for_generated_queries",
            "query_invalid",
            "query_execution_error",
            "generation_failure",
            "selected_query_semantic_mismatch",
        }:
            if status == "no_rows_for_generated_queries":
                st.warning(
                    "I could not find a graph-backed answer for this exact request. "
                    "No checked valid interpretation returned rows."
                )
            else:
                st.warning(reason or f"Answerability status: {status}")

    if graph_exec_error:
        st.error(f"Query execution error: {graph_exec_error}")
        st.write(answer_text or "No answer.")
    elif execute_selected and selected_query:
        if graph_rows:
            st.success(f"Returned {len(graph_rows)} rows from the True Demand KG.")
            st.write(answer_text)
        else:
            st.warning("No graph rows were found for the selected interpretation.")
            st.write(answer_text or "No results were found for this question.")
    else:
        st.write(answer_text or "No answer.")


def _clarified_answerability(graph_rows: List[Dict[str, str]], graph_exec_error: str = "") -> Dict[str, Any]:
    if graph_exec_error:
        return {
            "status": "query_execution_error",
            "can_answer": False,
            "reason": "The clarified query could not be executed against the graph.",
        }
    if graph_rows:
        return {
            "status": "answer_available",
            "can_answer": True,
            "reason": "The clarified query returned graph rows.",
        }
    return {
        "status": "no_rows_for_generated_queries",
        "can_answer": False,
        "reason": (
            "The clarified query executed but returned 0 rows. This means no matching "
            "data was found for that exact interpretation, or the selected graph backend "
            "does not contain the expected data."
        ),
    }


def _render_answer_subgraph(
    *,
    selected_query: str,
    graph_rows: List[Dict[str, str]],
    graph_path: str,
) -> None:
    if not selected_query or not graph_path or not _graph_backend_available(graph_path):
        return
    if not graph_rows:
        st.subheader("Inspected Query Path")
        st.warning(
            "The selected query returned no graph rows, so no answer evidence graph is shown."
        )
        return
    try:
        def _collect_evidence() -> Tuple[List[Tuple[Any, Any, Any]], Dict[str, Any]]:
            graph = _load_active_graph(graph_path)
            return collect_answer_evidence_triples(
                graph=graph,
                query=selected_query,
                limit=18,
            )

        triples, meta = _run_with_timeout(
            _collect_evidence,
            _interactive_query_timeout_sec(),
            label="answer evidence graph",
        )
    except Exception:
        return
    if not triples:
        return

    st.subheader("Answer Evidence Graph")
    st.caption(
        "The key graph relationships used by the selected query. "
        f"Predicates: {meta.get('predicate_count', 0)} | Edges shown: {meta.get('edge_count', 0)}"
    )
    graph_nodes = {node for s, _p, o in triples for node in (s, o)}
    graph_col, legend_col = st.columns([4.2, 1.2], gap="large")
    with graph_col:
        html = build_graph_html(
            triples,
            height_px=520,
            heading="Answer Evidence Graph",
            max_nodes=24,
            max_edges=36,
        )
        components.html(
            html,
            height=560,
            scrolling=True,
        )
    with legend_col:
        _render_graph_side_panel(
            node_count=len(graph_nodes),
            edge_count=len(triples),
            relationship_label="Business relationship",
            has_entity_nodes=any(isinstance(node, URIRef) for node in graph_nodes),
            has_literal_nodes=any(not isinstance(node, URIRef) for node in graph_nodes),
            graph_rows=graph_rows,
        )


def _render_clarification(
    clarification: Dict[str, Any],
    *,
    execute_selected: bool,
    graph_path: str,
    max_preview_rows: int,
) -> None:
    st.subheader("Clarify Interpretation")
    st.write(str(clarification.get("reason", "Candidates disagree on the intended query plan.")))
    st.write(str(clarification.get("question", "Which interpretation matches what you want?")))
    options = list(clarification.get("options") or [])
    if not options:
        st.warning(
            "No answerable clarification option was found. The generated interpretations "
            "returned no rows or did not match the requested meaning."
        )
        return
    for option in options:
        with st.container(border=True):
            st.markdown(f"**{str(option.get('label', 'Interpretation'))}**")
            preview = str(option.get("preview") or "").strip()
            if preview:
                st.write(preview)
            row_count = option.get("row_count")
            if row_count is not None:
                try:
                    row_count_value = int(row_count)
                except (TypeError, ValueError):
                    row_count_value = None
                if row_count_value == 0:
                    st.markdown(
                        "<div style='color:#b42318;font-weight:700;margin:0.25rem 0;'>"
                        "No rows returned for this interpretation.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(f"Rows found in graph: {row_count}")
            if st.button(
                "Use this interpretation",
                key=f"clarify_{option.get('id')}",
                use_container_width=True,
            ):
                chosen_query = str(option.get("query", "") or "").strip()
                st.session_state["last_selected_query"] = chosen_query
                st.session_state["clarification_choice_id"] = option.get("id")
                if option.get("preview_rows"):
                    rows = list(option.get("preview_rows") or [])
                    st.session_state["last_graph_rows"] = rows
                    st.session_state["last_graph_answer"] = synthesize_answer(
                        str(st.session_state.get("last_question", "")),
                        chosen_query,
                        {
                            "rows": rows,
                            "matched_question_id": None,
                            "error": None,
                        },
                        None,
                    )
                elif execute_selected and chosen_query and _graph_backend_available(graph_path):
                    try:
                        graph = _run_with_timeout(
                            lambda: _load_active_graph(graph_path),
                            _interactive_query_timeout_sec(),
                            label="graph backend load",
                        )
                        rows, _truncated = _execute_query_preview(
                            graph,
                            chosen_query,
                            max_rows=int(max_preview_rows),
                        )
                        st.session_state["last_graph_rows"] = rows
                        st.session_state["last_graph_answer"] = synthesize_answer(
                            str(st.session_state.get("last_question", "")),
                            chosen_query,
                            {
                                "rows": rows,
                                "matched_question_id": None,
                                "error": None,
                            },
                            None,
                        )
                    except Exception:
                        st.session_state["last_graph_rows"] = []
                        st.session_state["last_graph_answer"] = ""

    chosen_id = st.session_state.get("clarification_choice_id")
    if chosen_id:
        chosen = next((opt for opt in options if opt.get("id") == chosen_id), None)
        if chosen is not None:
            st.success(f"Using clarified interpretation: {chosen.get('label')}")


def _render_graph_side_panel(
    *,
    node_count: int,
    edge_count: int,
    relationship_label: str,
    has_entity_nodes: bool,
    has_literal_nodes: bool,
    graph_rows: List[Dict[str, str]] | None = None,
) -> None:
    node_rows = []
    if has_entity_nodes:
        node_rows.append('<div class="kg-side-row"><span class="kg-side-dot"></span> Classes / entities</div>')
    if has_literal_nodes:
        node_rows.append('<div class="kg-side-row"><span class="kg-side-dot muted"></span> Literal / value nodes</div>')
    evidence_rows = _graph_evidence_summary(graph_rows or [])
    st.markdown(
        f"""
        <aside class="kg-side-legend">
          <div class="kg-side-kicker">Legend</div>
          <div class="kg-side-section">
            <div class="kg-side-title">Nodes</div>
            {''.join(node_rows)}
          </div>
          <div class="kg-side-section">
            <div class="kg-side-title">Connections</div>
            <div class="kg-side-row"><span class="kg-side-line"></span> {escape(relationship_label)}</div>
          </div>
          <div class="kg-side-section">
            <div class="kg-side-title">Quick facts</div>
            <div class="kg-side-copy">Nodes shown: {node_count}<br>Edges shown: {edge_count}<br>Scroll to zoom.</div>
          </div>
          {evidence_rows}
        </aside>
        """,
        unsafe_allow_html=True,
    )


def _graph_evidence_summary(graph_rows: List[Dict[str, str]]) -> str:
    if not graph_rows:
        return ""
    first = graph_rows[0]
    items = []
    for key, value in list(first.items())[:5]:
        cleaned_key = escape(str(key).replace("_", " "))
        cleaned_value = escape(str(value))
        if len(cleaned_value) > 80:
            cleaned_value = cleaned_value[:77] + "..."
        items.append(
            f"<div class='kg-evidence-row'><span>{cleaned_key}</span><strong>{cleaned_value}</strong></div>"
        )
    if not items:
        return ""
    return (
        "<div class='kg-side-section'>"
        "<div class='kg-side-title'>Answer row preview</div>"
        f"<div class='kg-side-copy'>Rows returned: {len(graph_rows)}</div>"
        + "".join(items)
        + "</div>"
    )


def _request_clarification_guided_query(
    rewritten_question: str,
    *,
    graph_path: str,
    fuseki_query_url: str,
) -> str:
    question_text = str(rewritten_question or "").strip()
    if not question_text:
        return ""
    normalized = _normalize_question_key(question_text)
    if normalized in {
        "show semiconductor demand by quarter",
        "show summed percentage change in semiconductor demand for each region per quarter",
        "show semiconductor demand percentage change by region and quarter",
    }:
        return SEMICONDUCTOR_DEMAND_BY_QUARTER_QUERY
    if normalized in {
        "show current demand by region",
        "show total demand by region",
        "break down total regional demand by survey origin and region",
    }:
        return REGIONAL_DEMAND_BY_SURVEY_QUERY
    if normalized == "list oem total demand by region":
        return OEM_TOTAL_DEMAND_BY_REGION_QUERY
    if normalized in {
        "show vehicle sales by month",
        "show actual vehicle sales by month",
        "what are the total vehicle sales units reported each month in the actual data",
    }:
        return ACTUAL_VEHICLE_SALES_BY_MONTH_QUERY
    if normalized in {
        "show future semiconductor demand by technology category and quarter",
        "show the aggregate future semiconductor demand broken down by technology category and quarter",
        "which technology categories account for the highest future semiconductor demand in each quarter",
    }:
        return FUTURE_SEMICONDUCTOR_DEMAND_BY_TECH_QUARTER_QUERY
    if normalized in {
        "review current demand by region",
        "review regional current demand evidence",
        "which region should be monitored more closely based on current demand",
        "based on current demand which region should be monitored more closely",
    }:
        return CURRENT_DEMAND_BY_REGION
    if normalized in {
        "review future demand by region",
        "review future demand regional evidence",
        "where should planning attention focus based on future demand",
        "based on the survey data where should planning attention focus",
    }:
        return FUTURE_DEMAND_BY_REGION
    if normalized in {
        "review future demand by vehicle type",
        "review vehicle type future demand evidence",
        "which vehicle type shows the strongest future demand signal",
    }:
        return FUTURE_DEMAND_BY_VEHICLE_TYPE
    if normalized in {
        "review future demand by technology category",
        "review technology category future demand evidence",
        "which demand area seems most uncertain",
        "what should i look at if i want to understand future demand risk",
    }:
        return FUTURE_DEMAND_BY_TECHNOLOGY
    if normalized in {
        "review shortage exposure by survey group",
        "review shortage risk evidence",
        "which survey group appears most exposed to shortage",
    }:
        return SHORTAGE_BY_SURVEY_GROUP
    lookup_query = _load_guided_query_lookup().get(_normalize_question_key(question_text), "")
    if lookup_query:
        return lookup_query
    try:
        option = _single_direct_capability_option(
            question=question_text,
            graph_path=graph_path,
            fuseki_query_url=fuseki_query_url,
        )
    except Exception:
        option = None
    if option:
        return str(option.get("query", "") or "").strip()
    return ""


def _clear_last_response_state() -> None:
    for key, value in {
        "last_qa_result": None,
        "last_graph_rows": [],
        "last_selected_query": "",
        "last_graph_answer": "",
        "last_request_id": "",
        "last_latency_s": 0.0,
        "last_latency_breakdown": {},
        "clarification_choice_id": None,
        "confidence_clarification_choice_id": None,
    }.items():
        st.session_state[key] = value


def _render_request_clarification(
    clarification: Dict[str, Any],
    *,
    graph_path: str,
    fuseki_query_url: str,
) -> None:
    st.subheader("Clarify Request")
    st.write(str(clarification.get("reason", "The requested task is not specific enough yet.")))
    st.write(str(clarification.get("question", "What do you want to know?")))
    rendered = 0
    for option in list(clarification.get("options") or []):
        rewritten = str(option.get("rewritten_question", "") or "").strip()
        query = _request_clarification_guided_query(
            rewritten,
            graph_path=graph_path,
            fuseki_query_url=fuseki_query_url,
        )
        if not query:
            continue
        rendered += 1
        cols = st.columns([3, 1])
        cols[0].write(str(option.get("label", "Option")))
        cols[1].button(
            "Use",
            key=f"request_clarify_{option.get('id')}",
            use_container_width=True,
            on_click=_execute_guided_question_now,
            args=(rewritten, query, graph_path, 200, bool(option.get("advisory_context"))),
        )
    if rendered == 0:
        st.warning(
            "I could not find a validated executable option for this clarification. "
            "Please use the guided builder or ask with an explicit month, quarter, region, "
            "technology category, or vehicle type."
        )


def _set_question_input(value: str) -> None:
    st.session_state["question_input"] = value
    st.session_state["guided_query_override_question"] = ""
    st.session_state["guided_query_override"] = ""
    _clear_last_response_state()
    st.session_state["question_input_revision"] = int(st.session_state.get("question_input_revision", 0) or 0) + 1


def _set_guided_question_input(value: str, query: str) -> None:
    st.session_state["question_input"] = value
    st.session_state["guided_query_override_question"] = value
    st.session_state["guided_query_override"] = query
    _clear_last_response_state()
    st.session_state["question_input_revision"] = int(st.session_state.get("question_input_revision", 0) or 0) + 1


def _advisory_plan_from_evidence_selection(question: str, query: str) -> AdvisoryPlan | None:
    query_text = str(query or "").strip()
    if not query_text:
        return None
    if query_text == CURRENT_DEMAND_BY_REGION:
        return AdvisoryPlan(
            plan_id="selected_current_demand_region_focus",
            title="Current-demand focus by region",
            query=CURRENT_DEMAND_BY_REGION,
            group_key="regionName",
            value_key="totalDemand",
            value_label="total current demand",
            objective="identify the region with the highest current-demand signal",
        )
    if query_text == FUTURE_DEMAND_BY_REGION:
        return AdvisoryPlan(
            plan_id="selected_future_demand_region_focus",
            title="Future-demand focus by region",
            query=FUTURE_DEMAND_BY_REGION,
            group_key="regionName",
            value_key="avgPercentageChange",
            value_label="average future-demand percentage change",
            objective="identify the region with the strongest future-demand signal",
        )
    if query_text == FUTURE_DEMAND_BY_VEHICLE_TYPE:
        return AdvisoryPlan(
            plan_id="selected_future_demand_vehicle_signal",
            title="Strongest future-demand signal by vehicle type",
            query=FUTURE_DEMAND_BY_VEHICLE_TYPE,
            group_key="vehicleType",
            value_key="avgPercentageChange",
            value_label="average future-demand percentage change",
            objective="identify the vehicle type with the strongest future-demand signal",
        )
    if query_text == FUTURE_DEMAND_BY_TECHNOLOGY:
        return AdvisoryPlan(
            plan_id="selected_future_demand_technology_signal",
            title="Strongest future-demand signal by technology category",
            query=FUTURE_DEMAND_BY_TECHNOLOGY,
            group_key="technologyCategory",
            value_key="avgPercentageChange",
            value_label="average future-demand percentage change",
            objective="identify the technology category with the strongest future-demand signal",
        )
    if query_text == SHORTAGE_BY_SURVEY_GROUP:
        return AdvisoryPlan(
            plan_id="selected_shortage_survey_exposure",
            title="Shortage exposure by survey group",
            query=SHORTAGE_BY_SURVEY_GROUP,
            group_key="surveyGroup",
            value_key="companyCount",
            value_label="companies reporting the shortage status",
            objective="identify where shortage signals appear most visible in the survey data",
        )
    return None


def _execute_guided_question_now(
    value: str,
    query: str,
    graph_path: str,
    max_preview_rows: int,
    advisory_context: bool = False,
) -> None:
    question_text = str(value or "").strip()
    query_text = str(query or "").strip()
    advisory_plan = _advisory_plan_from_evidence_selection(question_text, query_text) if advisory_context else None
    st.session_state["question_input"] = question_text
    st.session_state["guided_query_override_question"] = question_text
    st.session_state["guided_query_override"] = query_text
    st.session_state["question_input_revision"] = int(st.session_state.get("question_input_revision", 0) or 0) + 1

    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "answer": (
            "Graph-grounded advisory evidence selected."
            if advisory_plan is not None
            else "Validated graph-supported option selected."
        ),
        "selected_query": query_text,
        "candidates": [{"query": query_text, "source": "clarification_guided"}] if query_text else [],
        "schema_ranked": [],
        "learning_ranked": [],
        "metadata": {
            "guided_query": True,
            "clarification_option_executed": True,
            "llm_skipped": True,
            "advisory_route": advisory_plan is not None,
            "advisory_plan_id": advisory_plan.plan_id if advisory_plan is not None else None,
            "advisory_title": advisory_plan.title if advisory_plan is not None else None,
        },
        "errors": [],
        "prompt": "",
        "policy": "advisory_guided_clarification" if advisory_plan is not None else "guided_clarification",
        "entropy": 0.0,
        "selection_reason": (
            "User selected a graph-backed evidence view for advisory synthesis."
            if advisory_plan is not None
            else "User selected a validated graph-backed clarification option."
        ),
        "used_ml": False,
        "effective_question": question_text,
        "selection_explanation": {
            "selected_policy": "advisory_guided_clarification" if advisory_plan is not None else "guided_clarification",
            "selection_reason": (
                "User selected a graph-backed evidence view for advisory synthesis."
                if advisory_plan is not None
                else "User selected a validated graph-backed clarification option."
            ),
            "selected_query_valid": bool(query_text),
            "selected_query_errors": [],
            "selected_execution_has_rows": None,
        },
        "answerability": {
            "status": "guided_pending_execution",
            "can_answer": None,
            "reason": "A validated clarification option was selected and executed directly.",
        },
        "confidence_route": {
            "enabled": True,
            "route": "auto_answer",
            "score1": 0.94,
            "score2": 0.12,
            "margin": 0.82,
            "selected_query": query_text,
            "reason": (
                "user-selected graph-backed advisory evidence view"
                if advisory_plan is not None
                else "user-selected graph-backed clarification option"
            ),
            "options": [],
            "safety_flags": ["graph_grounded_advisory_not_business_decision"] if advisory_plan is not None else [],
            "blocking_safety_flags": [],
        },
        "clarification": None,
        "request_clarification": None,
    }
    rows: List[Dict[str, str]] = []
    graph_answer = ""
    graph_exec_error = ""
    graph_load_elapsed = 0.0
    graph_query_elapsed = 0.0
    answer_synthesis_elapsed = 0.0
    if query_text and _graph_backend_available(graph_path):
        try:
            graph_load_started = time.perf_counter()
            graph = _run_with_timeout(
                lambda: _load_active_graph(graph_path),
                _interactive_query_timeout_sec(),
                label="graph backend load",
            )
            graph_load_elapsed = time.perf_counter() - graph_load_started
            graph_query_started = time.perf_counter()
            rows, _truncated = _execute_query_preview(
                graph,
                query_text,
                max_rows=int(max_preview_rows),
            )
            graph_query_elapsed = time.perf_counter() - graph_query_started
            answer_synthesis_started = time.perf_counter()
            if advisory_plan is not None:
                graph_answer = synthesize_advisory_answer(question_text, advisory_plan, rows)
            else:
                graph_answer = synthesize_answer(
                    question_text,
                    query_text,
                    {"rows": rows, "matched_question_id": None, "error": None},
                    None if rows else None,
                )
            answer_synthesis_elapsed = time.perf_counter() - answer_synthesis_started
            result["answerability"] = _guided_answerability(rows)
            result["selection_explanation"]["selected_execution_has_rows"] = bool(rows)
        except Exception as exc:
            graph_exec_error = str(exc)
            graph_answer = ""
            result["errors"] = [graph_exec_error]
            result["answerability"] = _guided_answerability([], graph_exec_error)
    elif query_text:
        graph_exec_error = "Graph backend unavailable."
        result["errors"] = [graph_exec_error]
        result["answerability"] = _guided_answerability([], graph_exec_error)

    total_elapsed = time.perf_counter() - started
    latency_breakdown = {
        "pipeline_s": 0.0,
        "graph_load_s": graph_load_elapsed,
        "graph_query_s": graph_query_elapsed,
        "answer_format_s": answer_synthesis_elapsed,
        "total_s": total_elapsed,
    }
    try:
        _append_jsonl(
            SESSION_LOG_PATH,
            _session_log_payload(
                request_id=request_id,
                question=question_text,
                result=result,
                selected_query=query_text,
                graph_rows=rows,
                graph_exec_error=graph_exec_error,
                latency_s=total_elapsed,
                latency_breakdown=latency_breakdown,
            ),
        )
        _write_user_audit_record(
            _user_audit_payload(
                request_id=request_id,
                question=question_text,
                result=result,
                selected_query=query_text,
                graph_rows=rows,
                graph_exec_error=graph_exec_error,
                graph_answer=graph_answer,
                latency_s=total_elapsed,
            )
        )
    except Exception:
        pass

    st.session_state["last_qa_result"] = result
    st.session_state["last_graph_rows"] = rows
    st.session_state["last_selected_query"] = query_text
    st.session_state["last_graph_answer"] = graph_answer
    st.session_state["last_question"] = question_text
    st.session_state["last_request_id"] = request_id
    st.session_state["last_latency_s"] = total_elapsed
    st.session_state["last_latency_breakdown"] = latency_breakdown
    st.session_state["clarification_choice_id"] = None
    st.session_state["confidence_clarification_choice_id"] = None


def _append_question_input(current: str, phrase: str) -> None:
    base = str(current or "").strip()
    addition = str(phrase or "").strip()
    if not addition:
        return
    if not base:
        st.session_state["question_input"] = addition
        st.session_state["question_input_revision"] = int(st.session_state.get("question_input_revision", 0) or 0) + 1
        return
    if addition.lower() in base.lower():
        return
    separator = " " if base.endswith((" ", "-", "/", ",")) else " "
    st.session_state["question_input"] = f"{base}{separator}{addition}".strip()
    st.session_state["question_input_revision"] = int(st.session_state.get("question_input_revision", 0) or 0) + 1


def _complete_question_fragment(current: str, phrase: str) -> None:
    text = str(current or "")
    completion = str(phrase or "").strip()
    if not completion:
        return
    match = re.search(r"([A-Za-z0-9_+-]*)$", text)
    if match:
        prefix = text[: match.start(1)]
        fragment = match.group(1)
        if fragment and completion.lower().startswith(fragment.lower()):
            st.session_state["question_input"] = f"{prefix}{completion} "
            st.session_state["question_input_revision"] = int(st.session_state.get("question_input_revision", 0) or 0) + 1
            return
    separator = "" if text.endswith((" ", "\n", "\t", "-", "/", ",")) or not text else " "
    st.session_state["question_input"] = f"{text}{separator}{completion} "
    st.session_state["question_input_revision"] = int(st.session_state.get("question_input_revision", 0) or 0) + 1


def _active_guided_query(question: str) -> str:
    if str(st.session_state.get("guided_query_override_question", "")).strip() == (question or "").strip():
        return str(st.session_state.get("guided_query_override", "") or "").strip()
    return ""


@st.cache_data(show_spinner=False)
def _load_schema_dict_cached(schema_path: str) -> Dict[str, object]:
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _candidate_score(candidate: Dict[str, object]) -> float:
    for key in ("ml_score", "score", "selection_score", "semantic_judge_score"):
        try:
            return float(candidate.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def _ranked_confidence_candidates(
    result: Dict[str, Any],
    *,
    sort_by_score: bool,
) -> List[Dict[str, object]]:
    candidates = [
        dict(row)
        for row in list(result.get("learning_ranked") or [])
        if isinstance(row, dict) and str(row.get("query", "") or "").strip()
    ]
    if not candidates:
        candidates = [
            dict(row)
            for row in list(result.get("schema_ranked") or [])
            if isinstance(row, dict) and str(row.get("query", "") or "").strip()
        ]
    if not candidates:
        candidates = [
            dict(row)
            for row in list(result.get("candidates") or [])
            if isinstance(row, dict) and str(row.get("query", "") or "").strip()
        ]
    if sort_by_score:
        candidates.sort(key=_candidate_score, reverse=True)
    return candidates


def _expected_classes_from_question(question: str, question_contract: Dict[str, object]) -> set[str]:
    q = " ".join(str(question or "").lower().replace("tier 1", "tier1").split())
    metrics = set(str(v) for v in question_contract.get("metrics") or [])
    dimensions = set(str(v) for v in question_contract.get("dimensions") or [])
    expected: set[str] = set()
    if any(term in q for term in ["company", "companies", "firms", "oems", "suppliers"]):
        expected.add("Company")
    if "technology_category" in dimensions or "technology category" in q or "technology categories" in q:
        expected.add("TechnologyCategory")
    if "region" in dimensions:
        expected.add("Region")
    if "quarter" in dimensions:
        expected.add("Quarter")
    if "autonomous_driving" in metrics:
        expected.add("AutonomousDrivingDevelopment")
    if "vehicle_type" in dimensions:
        expected.add("VehicleType")
    if "shortage" in metrics and any(term in q for term in ["company", "companies", "survey type", "survey types"]):
        expected.add("Company")
    return expected


def _required_query_terms_from_question(question: str) -> List[Tuple[str, Tuple[str, ...]]]:
    q = " ".join(str(question or "").lower().replace("tier 1", "tier1").split())
    required: List[Tuple[str, Tuple[str, ...]]] = []
    try:
        capability_report = CAPABILITY_REGISTRY.resolve(question)
        required.extend(capability_report.required_terms)
    except Exception:
        pass
    if "oem" in q:
        required.append(("oem_scope", ("oem", "oem_survey")))
    if "tier1" in q:
        required.append(("tier1_scope", ("tier1", "tier1_survey")))
    if "semiconductor" in q:
        required.append(("semiconductor_scope", ("semiconductor", "semiconductor_survey")))
    if "region" in q or "regional" in q:
        required.append(("region", ("region", "inregion", "regionname", "demandforregion")))
    if "vehicle" in q:
        required.append(("vehicle", ("vehicle", "hasvehicletype", "forvehicletype")))
    if "sae" in q:
        required.append(("sae", ("sae", "hassaelevel")))
    if "year" in q or "yearly" in q:
        required.append(("year", ("year", "hasyear", "baselineyear")))
    if "quarter" in q or "quarterly" in q:
        required.append(("quarter", ("quarter", "fortimeperiod", "periodlabel")))
    if "percentage change" in q or "percent change" in q or re.search(r"\bchange\b", q):
        required.append(
            (
                "percentage_change",
                (
                    "percentagechange",
                    "percentchange",
                    "totaldemandpercentagechange",
                    "baselineb1percent",
                    "baselineb2percent",
                ),
            )
        )
    elif "percentage" in q or "percent" in q:
        required.append(
            (
                "percentage",
                (
                    "percentage",
                    "percent",
                    "pct",
                    "haspercentage",
                    "splitpercentage",
                ),
            )
        )
    if re.search(r"\bb1\b|\bbl1\b", q):
        required.append(
            (
                "b1_or_bl1",
                (
                    "b1",
                    "bl1",
                    "baselineb1",
                    "percentchangeb1",
                    "percentagechangeb1",
                    "percentagechangebl1",
                ),
            )
        )
    if re.search(r"\bb2\b|\bbl2\b", q):
        required.append(
            (
                "b2_or_bl2",
                (
                    "b2",
                    "bl2",
                    "baselineb2",
                    "percentchangeb2",
                    "percentagechangeb2",
                    "percentagechangebl2",
                ),
            )
        )
    seen = set()
    deduped: List[Tuple[str, Tuple[str, ...]]] = []
    for name, terms in required:
        if name in seen:
            continue
        seen.add(name)
        deduped.append((name, terms))
    return deduped


def _confidence_safety_flags(
    *,
    question: str,
    query: str,
    schema_dict: Dict[str, object],
) -> List[str]:
    try:
        question_contract_obj = extract_question_contract(question)
        question_contract = question_contract_obj.to_dict()
    except Exception:
        question_contract = {}
    try:
        query_contract = extract_query_contract(query)
        comparison = compare_contracts(extract_question_contract(question), query_contract).to_dict()
    except Exception:
        query_contract = None
        comparison = {"missing": {}, "conflicts": {}}
    try:
        plan = extract_query_plan(query, schema_dict or None)
    except Exception:
        plan = {}

    flags: List[str] = []
    query_lower = str(query or "").lower()
    answer_shape = str(question_contract.get("answer_shape") or "")
    metrics = set(str(v) for v in question_contract.get("metrics") or [])
    requested_scopes = set(str(v) for v in question_contract.get("scopes") or [])
    actual_scopes = set(getattr(query_contract, "scopes", set()) or set()) if query_contract else set()
    query_aggregation = getattr(query_contract, "aggregation", None) if query_contract else None
    query_shape = getattr(query_contract, "answer_shape", None) if query_contract else None

    if (answer_shape == "list_values" or "catalog_lookup" in metrics) and (
        query_aggregation == "count" or "count(" in query_lower
    ):
        flags.append("list_count_conflict")
    if answer_shape == "scalar" and query_shape == "grouped_table":
        flags.append("scalar_grouping_conflict")
    for scope in sorted(requested_scopes - actual_scopes):
        flags.append(f"scope_missing:{scope}")

    missing_payload = comparison.get("missing") if isinstance(comparison, dict) else {}
    conflict_payload = comparison.get("conflicts") if isinstance(comparison, dict) else {}
    if isinstance(missing_payload, dict):
        for axis in ("metrics", "aggregation", "answer_shape"):
            for value in missing_payload.get(axis) or []:
                flags.append(f"contract_missing:{axis}:{value}")
    if isinstance(conflict_payload, dict):
        for axis in ("metrics", "aggregation", "answer_shape"):
            for value in conflict_payload.get(axis) or []:
                flags.append(f"contract_conflict:{axis}:{value}")

    plan_classes = {str(v) for v in plan.get("classes") or []}
    for expected_class in sorted(_expected_classes_from_question(question, question_contract)):
        if expected_class not in plan_classes:
            flags.append(f"class_missing:{expected_class}")
    normalized_query = re.sub(r"[^a-z0-9]+", "", query_lower)
    loose_query = query_lower.replace("_", "").replace("-", "")
    for name, terms in _required_query_terms_from_question(question):
        normalized_terms = [re.sub(r"[^a-z0-9]+", "", str(term).lower()) for term in terms]
        if not any(term and (term in normalized_query or term in loose_query) for term in normalized_terms):
            flags.append(f"required_missing:{name}")
    q_lower = str(question or "").lower()
    ranking_requested = bool(
        re.search(r"\b(highest|lowest|top|bottom|largest|smallest|max|maximum|min|minimum|best|worst|most|least)\b", q_lower)
    )
    grouped_requested = bool(re.search(r"\b(by|per|grouped|breakdown|across)\b", q_lower))
    query_types = set(str(v) for v in plan.get("query_types") or [])
    if grouped_requested and not ranking_requested and ("ranking" in query_types or re.search(r"\bLIMIT\s+1\b", query_lower)):
        flags.append("ranking_without_ranking_intent")
    try:
        capability_report = CAPABILITY_REGISTRY.evaluate_query(question, query)
        for warning in capability_report.typo_warnings:
            flags.append("near_match:" + warning)
        for missing in capability_report.missing_required_terms:
            flags.append(f"required_missing:{missing}")
    except Exception:
        pass

    return sorted(set(flags))


def _blocking_confidence_safety_flags(flags: List[str]) -> List[str]:
    blocking_prefixes = (
        "list_count_conflict",
        "scalar_grouping_conflict",
        "scope_missing:",
        "contract_conflict:",
        "class_missing:",
        "required_missing:",
        "ranking_without_ranking_intent",
    )
    return [
        flag
        for flag in flags
        if any(str(flag).startswith(prefix) for prefix in blocking_prefixes)
    ]


def _humanize_axis_value(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    replacements = {
        "avg": "average",
        "sum": "total",
        "count": "count",
        "cnt": "count",
        "tier1": "Tier1",
        "oem": "OEM",
        "sae": "SAE",
        "ev": "EV",
        "nonev": "non-EV",
        "list": "list",
        "values": "values",
    }
    text = text.replace("_", " ").replace("-", " ")
    words = [replacements.get(word.lower(), word) for word in text.split()]
    return " ".join(words)


def _join_human(items: List[str], limit: int = 3) -> str:
    cleaned = [item for item in items if item]
    if not cleaned:
        return ""
    shown = cleaned[:limit]
    if len(shown) == 1:
        return shown[0]
    if len(shown) == 2:
        return f"{shown[0]} and {shown[1]}"
    return ", ".join(shown[:-1]) + f", and {shown[-1]}"


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _query_scope_hint(query: str, plan: Dict[str, object], scopes: List[str], surveys: List[str]) -> str:
    text = str(query or "").lower()
    class_names = " ".join(str(v).lower() for v in list(plan.get("classes") or []))
    origin_names = " ".join(str(v).lower() for v in list(plan.get("origins") or []) + list(plan.get("survey_origins") or []))
    combined = " ".join([text, class_names, origin_names, " ".join(scopes).lower(), " ".join(surveys).lower()])
    hints = []
    if "oem" in combined:
        hints.append("OEM")
    if "tier1" in combined or "tier 1" in combined:
        hints.append("Tier1")
    if "semiconductor" in combined or "_semi" in combined or " semi" in combined:
        hints.append("Semiconductor")
    return _join_human(_dedupe_keep_order(hints), limit=3)


def _query_data_path_hint(query: str, plan: Dict[str, object]) -> str:
    text = str(query or "")
    lowered = text.lower()
    classes = {str(v) for v in list(plan.get("classes") or [])}
    if "hasdetail" in lowered:
        return "detail records"
    if "hassurveyorigin" in lowered:
        return "survey-origin filtered records"
    if any(name.endswith(("_OEM", "_Tier1", "_Semiconductor")) for name in classes):
        return "survey-specific records"
    if "autonomousdrivingdevelopment" in lowered:
        return "all autonomous-driving records"
    if "futuredemandanalysis" in lowered:
        return "all future-demand records"
    if "currentdemandanalysis" in lowered:
        return "all current-demand records"
    return ""


def _query_difference_hint(details: Dict[str, object]) -> str:
    scope_hint = str(details.get("scope_hint") or "").strip()
    path_hint = str(details.get("path_hint") or "").strip()
    if scope_hint and path_hint:
        return f"{scope_hint} scope, {path_hint}"
    return scope_hint or path_hint


def _candidate_interpretation_details(question: str, query: str, candidate: Dict[str, object]) -> Dict[str, object]:
    try:
        contract = extract_query_contract(query).to_dict()
    except Exception:
        contract = {}
    try:
        plan = extract_query_plan(query)
    except Exception:
        plan = {}

    aggregation = str(contract.get("aggregation") or "").lower()
    metrics = [_humanize_axis_value(v) for v in list(contract.get("metrics") or [])]
    dimensions = [_humanize_axis_value(v) for v in list(contract.get("dimensions") or [])]
    scopes = [_humanize_axis_value(v) for v in list(contract.get("scopes") or [])]
    answer_shape = str(contract.get("answer_shape") or "")
    group_by = [_humanize_axis_value(v) for v in list(plan.get("group_by_predicates") or plan.get("group_by_vars") or [])]
    raw_select_vars = [str(v) for v in list(plan.get("select_vars") or [])]
    helper_select_names = {"count", "cnt", "total", "totaldemand", "avg", "average", "mean", "value"}
    select_vars = [
        _humanize_axis_value(v)
        for v in raw_select_vars
        if v.lower().replace("_", "") not in helper_select_names
    ]
    classes = [
        _humanize_axis_value(v)
        for v in list(plan.get("classes") or [])
        if str(v) not in {"OEM_Survey", "Tier1_Survey", "Semiconductor_Survey"}
    ]
    surveys = [_humanize_axis_value(v) for v in list(plan.get("survey_origins") or [])]
    query_types = {str(v) for v in plan.get("query_types") or []}
    scope_hint = _query_scope_hint(query, plan, scopes, surveys)
    path_hint = _query_data_path_hint(query, plan)

    if aggregation == "avg":
        action = "calculates the average of"
        result_shape = "an average value"
        choose_prefix = "you want an average"
    elif aggregation == "sum":
        action = "calculates the total of"
        result_shape = "a total value"
        choose_prefix = "you want totals"
    elif aggregation == "count":
        action = "counts"
        result_shape = "a count"
        choose_prefix = "you want to count records or entities"
    elif answer_shape == "ranked_one" or "ranking" in query_types or "limited" in query_types:
        action = "finds the top matching value for"
        result_shape = "the top matching result"
        choose_prefix = "you want the highest, lowest, or top result"
    elif answer_shape == "list_values":
        action = "lists"
        result_shape = "a list of values"
        choose_prefix = "you want to see the values themselves"
    else:
        action = "returns"
        result_shape = "matching graph rows"
        choose_prefix = "you want the matching records"

    target = _join_human(metrics) or _join_human(classes) or _join_human(select_vars) or "graph values"
    description = f"This option {action} {target}"
    grouping = _join_human(dimensions or group_by)
    if grouping:
        description += f", grouped by {grouping}"
    if scopes:
        description += f", restricted to {_join_human(scopes)}"
    elif scope_hint:
        description += f", for {scope_hint}"
    if surveys:
        description += f", using {_join_human(surveys)} data"
    description += "."

    what_you_will_see = f"{result_shape.capitalize()} for {target}"
    if grouping:
        what_you_will_see += f", broken down by {grouping}"
    if scopes:
        what_you_will_see += f", filtered to {_join_human(scopes)}"
    elif scope_hint:
        what_you_will_see += f", for {scope_hint}"
    if surveys:
        what_you_will_see += f", using {_join_human(surveys)} survey data"
    what_you_will_see += "."

    choose_if = choose_prefix
    if grouping:
        choose_if += f" by {grouping}"
    if scopes:
        choose_if += f" for {_join_human(scopes)}"
    elif scope_hint:
        choose_if += f" for {scope_hint}"
    if surveys:
        choose_if += f" from {_join_human(surveys)} survey data"
    choose_if += "."

    bullets: List[str] = []
    if aggregation:
        bullets.append(f"Calculation: {_humanize_axis_value(aggregation)}")
    elif answer_shape:
        bullets.append(f"Answer type: {_humanize_axis_value(answer_shape)}")
    if grouping:
        bullets.append(f"Breakdown: {grouping}")
    if scopes:
        bullets.append(f"Scope: {_join_human(scopes)}")
    elif scope_hint:
        bullets.append(f"Scope: {scope_hint}")
    if path_hint:
        bullets.append(f"Data path: {path_hint}")
    if surveys:
        bullets.append(f"Survey source: {_join_human(surveys)}")
    if select_vars:
        bullets.append(f"Returned fields: {_join_human(select_vars, limit=4)}")
    source = str(candidate.get("source") or "").strip()
    if source:
        bullets.append(f"Candidate source: {_humanize_axis_value(source)}")

    return {
        "description": description,
        "what_you_will_see": what_you_will_see,
        "choose_if": choose_if,
        "bullets": bullets,
        "action": action,
        "target": target,
        "grouping": grouping,
        "scopes": scopes,
        "surveys": surveys,
        "scope_hint": scope_hint,
        "path_hint": path_hint,
        "difference_hint": _query_difference_hint({"scope_hint": scope_hint, "path_hint": path_hint}),
    }


def _candidate_interpretation_summary(question: str, query: str, candidate: Dict[str, object]) -> str:
    details = _candidate_interpretation_details(question, query, candidate)
    target = str(details.get("target") or "graph data")
    grouping = str(details.get("grouping") or "")
    scopes = list(details.get("scopes") or [])
    surveys = list(details.get("surveys") or [])
    scope_hint = str(details.get("scope_hint") or "").strip()
    path_hint = str(details.get("path_hint") or "").strip()
    try:
        contract = extract_query_contract(query).to_dict()
    except Exception:
        contract = {}
    try:
        plan = extract_query_plan(query)
    except Exception:
        plan = {}
    aggregation = str(contract.get("aggregation") or "").lower()
    answer_shape = str(contract.get("answer_shape") or "")
    query_types = {str(v) for v in plan.get("query_types") or []}
    if aggregation == "avg":
        prefix = "Average"
    elif aggregation == "sum":
        prefix = "Total"
    elif aggregation == "count":
        prefix = "Count of"
    elif answer_shape == "ranked_one" or "ranking" in query_types or "limited" in query_types:
        prefix = "Top"
    elif answer_shape == "list_values":
        prefix = "List of"
    else:
        prefix = "Individual"

    label = f"{prefix} {target}".strip()
    if scopes:
        label += " for " + _join_human([str(v) for v in scopes], limit=2)
    elif scope_hint:
        label += f" for {scope_hint}"
    if grouping:
        label += " by " + grouping
    elif surveys:
        label += " from " + _join_human([str(v) for v in surveys], limit=2)
    elif path_hint and not scope_hint:
        label += f" from {path_hint}"
    return label


def _dedupe_confidence_option_labels(options: List[Dict[str, object]]) -> None:
    label_counts = Counter(str(option.get("label") or "").strip().lower() for option in options)
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for option in options:
        grouped[str(option.get("label") or "").strip().lower()].append(option)
    for label_key, duplicates in grouped.items():
        if not label_key or label_counts[label_key] <= 1:
            continue
        used = set()
        for option in duplicates:
            base_label = str(option.get("label") or "Interpretation").strip()
            hint = str(option.get("difference_hint") or "").strip()
            if not hint:
                hint = _humanize_axis_value(str(option.get("source") or "alternative"))
            new_label = f"{base_label} ({hint})" if hint and hint.lower() not in base_label.lower() else base_label
            if new_label.lower() in used:
                new_label = f"{base_label} (rank {option.get('rank')})"
            used.add(new_label.lower())
            option["label"] = new_label


def _build_confidence_route(
    result: Dict[str, Any],
    *,
    question: str,
    schema_path: str,
    min_score: float,
    min_margin: float,
    enable_safety_guard: bool,
    sort_by_score: bool,
) -> Dict[str, object] | None:
    if not isinstance(result, dict) or result.get("request_route", {}).get("route") not in {None, "kg_query"}:
        return None
    candidates = _ranked_confidence_candidates(result, sort_by_score=sort_by_score)
    if not candidates:
        return None
    selected_query = " ".join(str(result.get("selected_query") or "").split()).lower()
    if selected_query:
        selected_matches = [
            candidate
            for candidate in candidates
            if " ".join(str(candidate.get("query") or "").split()).lower() == selected_query
        ]
        if selected_matches:
            selected_match = selected_matches[0]
            candidates = [selected_match] + [
                candidate
                for candidate in candidates
                if " ".join(str(candidate.get("query") or "").split()).lower() != selected_query
            ]
        else:
            candidates = [{"query": result.get("selected_query"), "source": "execution_aware"}] + candidates
    schema_dict = _load_schema_dict_cached(schema_path)
    top1 = candidates[0]
    top2 = candidates[1] if len(candidates) > 1 else {}
    score1 = _candidate_score(top1)
    score2 = _candidate_score(top2) if top2 else 0.0
    margin = score1 - score2
    flags = _confidence_safety_flags(
        question=question,
        query=str(top1.get("query", "") or ""),
        schema_dict=schema_dict,
    )
    reason_parts = []
    if score1 < min_score:
        reason_parts.append(f"score {score1:.3f} is below {min_score:.2f}")
    if margin < min_margin:
        reason_parts.append(f"margin {margin:.3f} is below {min_margin:.2f}")
    blocking_flags = _blocking_confidence_safety_flags(flags)
    if enable_safety_guard and blocking_flags:
        reason_parts.append("safety guard found " + ", ".join(blocking_flags[:4]))
    route = "auto_answer" if not reason_parts else "clarification"
    options = []
    seen = set()
    for idx, candidate in enumerate(candidates, start=1):
        query = str(candidate.get("query", "") or "").strip()
        if not query:
            continue
        key = " ".join(query.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        interpretation = _candidate_interpretation_details(question, query, candidate)
        options.append(
            {
                "id": f"confidence_{len(options) + 1}",
                "rank": idx,
                "score": _candidate_score(candidate),
                "source": candidate.get("source"),
                "label": _candidate_interpretation_summary(question, query, candidate),
                "description": interpretation.get("description"),
                "what_you_will_see": interpretation.get("what_you_will_see"),
                "choose_if": interpretation.get("choose_if"),
                "details": interpretation.get("bullets"),
                "scope_hint": interpretation.get("scope_hint"),
                "path_hint": interpretation.get("path_hint"),
                "difference_hint": interpretation.get("difference_hint"),
                "query": query,
                "safety_flags": _confidence_safety_flags(
                    question=question,
                    query=query,
                    schema_dict=schema_dict,
                ),
            }
        )
        if len(options) >= 3:
            break
    _dedupe_confidence_option_labels(options)

    return {
        "enabled": True,
        "route": route,
        "score1": score1,
        "score2": score2,
        "margin": margin,
        "min_score": min_score,
        "min_margin": min_margin,
        "safety_guard": bool(enable_safety_guard),
        "safety_flags": flags,
        "blocking_safety_flags": blocking_flags,
        "reason": "; ".join(reason_parts) if reason_parts else "high confidence and no safety flags",
        "selected_query": str(top1.get("query", "") or "").strip(),
        "options": options,
    }


@st.cache_data(show_spinner=False, ttl=120)
def _capability_backed_clarification_options(
    *,
    question: str,
    graph_path: str,
    fuseki_query_url: str,
    max_options: int = 3,
) -> List[Dict[str, object]]:
    started_at = time.perf_counter()
    option_budget_s = min(2.5, _interactive_time_budget_sec() / 3.0)
    try:
        report = CAPABILITY_REGISTRY.resolve(question)
    except Exception:
        return []
    capability = report.primary_capability
    if not capability:
        return []
    requested_dims = {str(item.name).lower() for item in report.detected_dimensions}
    direct_query = CAPABILITY_REGISTRY.direct_query_for(report)
    if direct_query:
        rows, error = _preview_query_rows_cached(
            graph_path,
            fuseki_query_url,
            direct_query,
            max_rows=1,
        )
        if rows and not error:
            dimension = next(iter(requested_dims), "requested dimension")
            profile_hint = _capability_dimension_profile_hint(capability, dimension)
            details = [
                f"Capability: {capability}",
                f"Breakdown: by {dimension}",
                "Source: capability registry direct query",
            ]
            if profile_hint:
                details.append(profile_hint)
            return [
                {
                    "id": "capability_direct_1",
                    "rank": 1,
                    "score": 0.0,
                    "source": "capability_inventory",
                    "label": f"{capability.title()} by {dimension}",
                    "description": f"Direct graph-supported {capability} query.",
                    "what_you_will_see": f"{capability} values grouped by {dimension}.",
                    "choose_if": f"Choose this if you meant {capability} by {dimension}.",
                    "details": details,
                    "scope_hint": "all graph data",
                    "path_hint": f"by {dimension}",
                    "difference_hint": f"by {dimension}",
                    "query": direct_query,
                    "safety_flags": [],
                }
            ]
    options: List[Dict[str, object]] = []
    seen_queries = set()
    prefers_non_count = _question_prefers_non_count(question)
    for row in list(_validated_guided_patterns()) + _capability_answerable_patterns(
        graph_path,
        fuseki_query_url,
    ):
        if time.perf_counter() - started_at >= option_budget_s:
            break
        pattern_question = str(row.get("question", "") or "")
        try:
            pattern_report = CAPABILITY_REGISTRY.resolve(pattern_question)
        except Exception:
            continue
        if pattern_report.primary_capability != capability:
            continue
        pattern_dims = {str(item.name).lower() for item in pattern_report.detected_dimensions}
        if requested_dims and pattern_dims != requested_dims:
            continue
        if requested_dims and not pattern_dims:
            continue
        if prefers_non_count and str(row.get("metric", "")).lower() in {"record count", "available names"}:
            continue
        if prefers_non_count and re.search(r"\b(how many|count|number of)\b", pattern_question, flags=re.I):
            continue
        query = str(row.get("query", "") or "").strip()
        if not query:
            continue
        key = " ".join(query.split()).lower()
        if key in seen_queries:
            continue
        rows, error = _preview_query_rows_cached(
            graph_path,
            fuseki_query_url,
            query,
            max_rows=1,
        )
        if error or not rows:
            continue
        seen_queries.add(key)
        option_id = f"capability_{len(options) + 1}"
        label = str(row.get("question") or row.get("metric") or "Graph-supported interpretation").strip()
        options.append(
            {
                "id": option_id,
                "rank": len(options) + 1,
                "score": 0.0,
                "source": "capability_inventory",
                "label": _humanize_question_label(label),
                "description": f"Graph-supported {capability} interpretation.",
                "what_you_will_see": _capability_result_hint(row),
                "choose_if": f"Choose this if you meant {str(row.get('metric') or capability)} {str(row.get('breakdown') or '').strip()}.".strip(),
                "details": [
                    f"Capability: {capability}",
                    f"Metric: {row.get('metric') or 'graph-supported metric'}",
                    f"Breakdown: {row.get('breakdown') or 'overall'}",
                    f"Scope: {row.get('scope') or 'all graph data'}",
                ],
                "scope_hint": str(row.get("scope") or ""),
                "path_hint": str(row.get("breakdown") or ""),
                "difference_hint": str(row.get("breakdown") or ""),
                "query": query,
                "safety_flags": [],
            }
        )
        if len(options) >= max_options:
            break
    ranked_options = _rank_relevant_option_dicts(
        question,
        options,
        max_options=max_options,
        min_score=0.12 if requested_dims else 0.2,
    )
    return ranked_options or options[:max_options]


def _capability_dimension_profile_hint(capability_name: str, dimension_name: str) -> str:
    capability = CAPABILITY_REGISTRY.find_capability(capability_name)
    if not capability:
        return ""
    dimension = capability.dimension_by_name(dimension_name)
    if not dimension:
        return ""
    parts = []
    if dimension.distinct_values is not None:
        parts.append(f"{dimension.distinct_values} distinct value(s)")
    if dimension.estimated_rows is not None:
        parts.append(f"about {dimension.estimated_rows} row(s)")
    if not parts:
        return ""
    return f"Graph profile for {dimension.name}: " + ", ".join(parts)


def _single_direct_capability_option(
    *,
    question: str,
    graph_path: str,
    fuseki_query_url: str,
) -> Dict[str, object] | None:
    options = _capability_backed_clarification_options(
        question=question,
        graph_path=graph_path,
        fuseki_query_url=fuseki_query_url,
        max_options=2,
    )
    if len(options) != 1:
        return None
    option = options[0]
    if not str(option.get("id", "")).startswith("capability_direct"):
        return None
    return option


def _direct_capability_result(question: str, option: Dict[str, object]) -> Dict[str, object]:
    query = str(option.get("query", "") or "").strip()
    reason = (
        "single graph-supported capability interpretation resolved from "
        "capability, dimension, and intent"
    )
    return {
        "answer": "Validated graph-supported capability query selected.",
        "selected_query": query,
        "candidates": [
            {
                "query": query,
                "source": option.get("source", "capability_inventory"),
                "score": option.get("score", 0.0),
            }
        ],
        "schema_ranked": [],
        "learning_ranked": [],
        "metadata": {
            "guided_query": True,
            "direct_capability_route": True,
            "llm_skipped": True,
        },
        "errors": [],
        "prompt": "",
        "policy": "direct_capability",
        "entropy": 0.0,
        "selection_reason": reason,
        "used_ml": False,
        "effective_question": question,
        "ml_policy": "skipped",
        "ml_model_path": None,
        "predicted_regime": "low",
        "predicted_entropy": 0.0,
        "query_plan_ml_used": False,
        "ml_ranker_applied": False,
        "candidate_duplicates_removed": 0,
        "selection_explanation": {
            "selected_policy": "direct_capability",
            "selection_reason": reason,
            "selected_query_valid": True,
            "selected_query_errors": [],
            "selected_execution_has_rows": None,
            "candidate_count": 1,
            "valid_candidate_count": 1,
        },
        "answerability": {
            "status": "direct_capability_pending_execution",
            "can_answer": None,
            "reason": "A single graph-supported capability query was selected before using the LLM.",
            "selected_has_rows": None,
            "selected_error": None,
            "alternative_nonempty_count": 0,
            "valid_candidate_count": 1,
        },
        "clarification": None,
        "request_clarification": None,
        "confidence_route": {
            "enabled": True,
            "route": "auto_answer",
            "selected_query": query,
            "reason": reason,
            "score1": 0.96,
            "score2": 0.18,
            "margin": 0.78,
            "options": [option],
            "safety_flags": [],
            "blocking_safety_flags": [],
        },
    }


def _question_prefers_non_count(question: str) -> bool:
    q = str(question or "").lower()
    if re.search(r"\b(how many|count|number of)\b", q):
        return False
    return bool(
        re.search(r"\b(change|changes|evolve|vary|trend|average|avg|mean|total|sum|share|percentage|percent)\b", q)
        or " by " in q
        or " across " in q
    )


def _question_prefers_grouped_breakdown(question: str) -> bool:
    q = str(question or "").lower()
    if re.search(r"\b(highest|top|largest|max|maximum|lowest|min|minimum|best)\b", q):
        return False
    return bool(re.search(r"\b(by|per|for each|grouped by|broken down by|across)\b", q))


def _query_has_group_by(query: str) -> bool:
    return bool(re.search(r"\bGROUP\s+BY\b", str(query or ""), flags=re.I))


def _query_has_limit_one(query: str) -> bool:
    return bool(re.search(r"\bLIMIT\s+1\b", str(query or ""), flags=re.I))


def _query_has_count_aggregation(query: str) -> bool:
    return bool(re.search(r"\bCOUNT\s*\(", str(query or ""), flags=re.I))


def _execution_aware_selected_query_override(
    result: Dict[str, Any],
    *,
    question: str,
    graph_path: str,
    fuseki_query_url: str,
    sort_by_score: bool,
    max_candidates: int = 5,
) -> Dict[str, object] | None:
    selected_query = str(result.get("selected_query") or "").strip()
    if not selected_query or not _graph_backend_available(graph_path):
        return None

    candidates = _ranked_confidence_candidates(result, sort_by_score=sort_by_score)
    ordered_queries = [selected_query]
    for candidate in candidates:
        query = str(candidate.get("query") or "").strip()
        if query and " ".join(query.split()).lower() not in {
            " ".join(q.split()).lower() for q in ordered_queries
        }:
            ordered_queries.append(query)
        if len(ordered_queries) >= max_candidates:
            break

    prefers_grouped = _question_prefers_grouped_breakdown(question)
    prefers_non_count = _question_prefers_non_count(question)

    inspected: List[Dict[str, object]] = []
    for idx, query in enumerate(ordered_queries, start=1):
        rows, error = _preview_query_rows_cached(
            graph_path,
            fuseki_query_url,
            query,
            max_rows=1,
        )
        shape_flags = []
        if prefers_grouped and not _query_has_group_by(query):
            shape_flags.append("missing_group_by_for_breakdown_question")
        if prefers_grouped and _query_has_limit_one(query):
            shape_flags.append("ranked_one_for_breakdown_question")
        if prefers_non_count and _query_has_count_aggregation(query):
            shape_flags.append("count_for_non_count_question")
        inspected.append(
            {
                "rank": idx,
                "query": query,
                "has_rows": bool(rows) and not bool(error),
                "error": error,
                "shape_flags": shape_flags,
            }
        )

    selected_info = inspected[0]
    selected_bad = bool(selected_info.get("error")) or not selected_info.get("has_rows") or bool(
        selected_info.get("shape_flags")
    )
    if not selected_bad:
        return None

    for alternative in inspected[1:]:
        if not alternative.get("has_rows") or alternative.get("error"):
            continue
        alt_flags = list(alternative.get("shape_flags") or [])
        if alt_flags and not selected_info.get("shape_flags"):
            continue
        if len(alt_flags) > len(list(selected_info.get("shape_flags") or [])):
            continue
        return {
            "from_query": selected_query,
            "to_query": alternative["query"],
            "reason": "execution-aware switch to non-empty/shape-compatible candidate",
            "selected_profile": selected_info,
            "alternative_profile": alternative,
            "inspected_count": len(inspected),
        }
    return {
        "from_query": selected_query,
        "to_query": None,
        "reason": "selected query failed execution/shape check but no better alternative was found",
        "selected_profile": selected_info,
        "inspected_count": len(inspected),
    }


def _humanize_question_label(text: str) -> str:
    text = str(text or "").strip().rstrip("?")
    if not text:
        return "Graph-supported interpretation"
    text = re.sub(r"^(can you|please|show me|show|list|provide|return)\s+", "", text, flags=re.I)
    return text[:1].upper() + text[1:]


def _capability_result_hint(row: Dict[str, str]) -> str:
    metric = str(row.get("metric") or "values").strip()
    breakdown = str(row.get("breakdown") or "overall").strip()
    scope = str(row.get("scope") or "all graph data").strip()
    if breakdown and breakdown.lower() != "overall":
        return f"{metric} for {scope}, broken down {breakdown}."
    return f"{metric} for {scope}."


def _render_confidence_route_badge(route: Dict[str, object]) -> None:
    if route.get("route") == "auto_answer":
        reason = str(route.get("reason") or "").strip().lower()
        if "ontology" in reason or "digital reference" in reason:
            st.success("Deterministic ontology answer. LLM skipped.")
            return
        score = route.get("score1")
        margin = route.get("margin")
        suffix = ""
        try:
            suffix = f" (score={float(score):.3f}, margin={float(margin):.3f})"
        except (TypeError, ValueError):
            pass
        st.success("High-confidence graph answer" + suffix + ".")
    elif route.get("route") == "controlled_no_answer":
        st.warning("No graph-backed answer was found for this exact request.")
        reason = str(route.get("reason") or "").strip()
        if reason:
            st.caption(reason[0].upper() + reason[1:] if reason else reason)
    else:
        st.warning(
            "The system found multiple plausible interpretations. "
            "Please choose one before it answers."
        )
        reason = str(route.get("reason") or "").strip()
        if reason:
            st.caption(reason[0].upper() + reason[1:] if reason else reason)


def _accept_confidence_option(
    option: Dict[str, object],
    *,
    graph_path: str,
    max_preview_rows: int,
) -> None:
    chosen_query = str(option.get("query", "") or "").strip()
    st.session_state["last_selected_query"] = chosen_query
    st.session_state["confidence_clarification_choice_id"] = option.get("id")
    if not chosen_query or not _graph_backend_available(graph_path):
        return
    try:
        rows, error = _preview_query_rows_cached(
            graph_path,
            _active_fuseki_query_url(),
            chosen_query,
            max_rows=int(max_preview_rows),
        )
        if error:
            raise RuntimeError(error)
        st.session_state["last_graph_rows"] = rows
        st.session_state["last_graph_answer"] = synthesize_answer(
            str(st.session_state.get("last_question", "")),
            chosen_query,
            {
                "rows": rows,
                "matched_question_id": None,
                "error": None,
            },
            None,
        )
    except Exception:
        st.session_state["last_graph_rows"] = []
        st.session_state["last_graph_answer"] = ""


def _render_confidence_clarification(
    route: Dict[str, object],
    *,
    execute_selected: bool,
    graph_path: str,
    max_preview_rows: int,
) -> None:
    options = list(route.get("options") or [])
    active_question = str(st.session_state.get("last_question") or "")
    fuseki_url = _active_fuseki_query_url()
    if not options:
        options = _capability_backed_clarification_options(
            question=active_question,
            graph_path=graph_path,
            fuseki_query_url=fuseki_url,
            max_options=3,
        )
        if not options:
            st.warning(
                "I could not find an answerable graph interpretation for this request. "
                "Please revise the question or use one of the validated examples."
            )
            return
    display_options: List[Dict[str, object]] = []
    answerable_count = 0
    skipped_empty = 0
    if execute_selected and _graph_backend_available(graph_path):
        schema_dict = _load_schema_dict_cached(str(DEFAULT_SCHEMA_PATH))
        for option in options:
            enriched = dict(option)
            query = str(option.get("query", "") or "").strip()
            flags = _confidence_safety_flags(
                question=active_question,
                query=query,
                schema_dict=schema_dict,
            )
            blocking = _blocking_confidence_safety_flags(flags)
            if blocking:
                skipped_empty += 1
                continue
            rows, error = _preview_query_rows_cached(
                graph_path,
                fuseki_url,
                query,
                max_rows=max(1, min(int(max_preview_rows), 3)),
            )
            if rows:
                enriched["preview_rows"] = rows
                enriched["row_count_preview"] = len(rows)
                enriched["answerable"] = True
                enriched["selectable"] = True
                answerable_count += 1
                display_options.append(enriched)
            else:
                skipped_empty += 1
                enriched["preview_error"] = error
                enriched["answerable"] = False
                enriched["selectable"] = True
                enriched["disabled_reason"] = "No rows returned for this interpretation."
                display_options.append(enriched)
        options = display_options
    else:
        for option in options:
            enriched = dict(option)
            enriched["answerable"] = True
            enriched["selectable"] = True
            display_options.append(enriched)
        options = display_options
        answerable_count = len(options)
    selectable_count = sum(1 for option in options if bool(option.get("selectable", option.get("answerable"))))
    if not options:
        options = _capability_backed_clarification_options(
            question=active_question,
            graph_path=graph_path,
            fuseki_query_url=fuseki_url,
            max_options=3,
        )
        if options:
            for option in options:
                option["answerable"] = True
                option["selectable"] = True
            answerable_count = len(options)
            st.caption("Using graph-supported interpretations from the capability inventory.")
        else:
            st.warning(
                "I found candidate interpretations, but none returned graph rows. "
                "Please revise the question or choose graph terms that are connected in the data."
            )
            return
    if skipped_empty:
        st.caption(
            f"{skipped_empty} generated interpretation(s) were removed or marked because they failed "
            "semantic or graph-evidence checks."
        )
    if len(options) == 1 and answerable_count == 1 and execute_selected:
        _accept_confidence_option(
            options[0],
            graph_path=graph_path,
            max_preview_rows=max_preview_rows,
        )
        st.info("Using the only graph-supported interpretation.")
        return
    if len(options) < 2 or selectable_count == 0:
        st.warning(
            "I could not find enough selectable graph-backed interpretations. "
            "Please revise the question or use the guided builder."
        )
        return

    st.subheader("Clarify Interpretation")
    st.write("I found multiple possible interpretations.")
    st.markdown("**Did you mean...**")
    for option in options[:3]:
        with st.container(border=True):
            st.markdown(f"**{str(option.get('label') or 'Interpretation')}**")
            if option.get("answerable"):
                row_preview = option.get("row_count_preview")
                if row_preview is not None:
                    st.success(f"Graph evidence available. Preview rows: {row_preview}.")
                else:
                    st.success("Graph-supported interpretation.")
            else:
                reason = str(option.get("disabled_reason") or "This interpretation is not currently answerable.")
                st.markdown(
                    f"<div style='color:#b42318;font-weight:700;margin:0.25rem 0;'>{escape(reason)}</div>",
                    unsafe_allow_html=True,
                )
            what_you_will_see = str(option.get("what_you_will_see") or option.get("description") or "").strip()
            choose_if = str(option.get("choose_if") or "").strip()
            details = [str(item) for item in (option.get("details") or []) if str(item).strip()]
            with st.expander("More details", expanded=False):
                if what_you_will_see:
                    st.markdown(f"**What you will see:** {what_you_will_see}")
                if choose_if:
                    st.markdown(f"**Choose this if:** {choose_if}")
                if details:
                    st.markdown("\n".join(f"- {item}" for item in details))
                st.caption(f"Candidate source: {option.get('source') or 'unknown'}")
            with st.expander("Technical SPARQL query", expanded=False):
                st.code(_format_sparql_for_display(str(option.get("query", ""))), language="sparql")
            if st.button(
                "Use this interpretation",
                key=f"confidence_clarify_{option.get('id')}",
                use_container_width=True,
                disabled=not bool(option.get("selectable", option.get("answerable"))),
            ):
                _accept_confidence_option(
                    option,
                    graph_path=graph_path,
                    max_preview_rows=max_preview_rows,
                )

    chosen_id = st.session_state.get("confidence_clarification_choice_id")
    if chosen_id:
        chosen = next((opt for opt in options if opt.get("id") == chosen_id), None)
        if chosen is not None:
            st.success(f"Using clarified interpretation: {chosen.get('label')}")


def _normalize_question_key(question: str) -> str:
    return " ".join(str(question or "").strip().lower().rstrip("?.!").split())


OFF_TOPIC_EXAMPLES = [
    "What is a Technology Node?",
    "Show future demand by region.",
    "Which region should be monitored based on current demand?",
    "Show vehicle sales by month.",
    "How many companies reported shortages by survey group?",
]


def _is_out_of_scope_question(question: str) -> Tuple[bool, str]:
    q = _normalize_question_key(question)
    if not q:
        return False, ""

    off_topic_patterns = (
        r"\bweather\b",
        r"\btemperature outside\b",
        r"\brain\b",
        r"\bforecast for (today|tomorrow|athens|zurich|munich|new york|london)\b",
        r"\brecipe\b",
        r"\bcook\b",
        r"\bmovie\b",
        r"\bsports?\b",
        r"\bfootball\b",
        r"\bstock price\b",
        r"\bexchange rate\b",
        r"\bbitcoin\b",
        r"\biphone\b",
        r"\bmacbook\b",
        r"\btesla\b",
        r"\bcompetitor products?\b",
        r"\bproducts? (do you|does infineon|are they) sell\b",
        r"\bwho is\b.*\b(president|prime minister|ceo)\b",
    )
    for pattern in off_topic_patterns:
        if re.search(pattern, q):
            return True, "The question appears to be outside the True Demand KG and Digital Reference ontology scope."

    in_scope_terms = (
        "true demand",
        "demand",
        "future demand",
        "current demand",
        "regional demand",
        "oem",
        "tier1",
        "semiconductor",
        "survey",
        "region",
        "quarter",
        "month",
        "year",
        "vehicle",
        "vehicle type",
        "sales",
        "inventory",
        "shortage",
        "order cancellation",
        "autonomous driving",
        "sae",
        "technology",
        "technology node",
        "technology category",
        "component",
        "company",
        "market segment",
        "baseline",
        "percentage",
        "trend",
        "capacity",
        "process",
        "product",
        "material",
        "resource",
        "planning",
        "supply chain",
        "digital reference",
        "ontology",
        "class",
        "property",
        "domain",
        "range",
    )
    if any(term in q for term in in_scope_terms):
        return False, ""

    # Very short generic questions without graph/ontology vocabulary are safer to reject.
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", q)
    if len(tokens) <= 8:
        return True, "I could not match the question to the True Demand graph or Digital Reference ontology."
    return False, ""


def _is_unsupported_relative_time_question(question: str) -> Tuple[bool, str]:
    q = _normalize_question_key(question)
    if not q:
        return False, ""
    relative_time_patterns = (
        r"\b(?:past|last|recent|previous)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*(?:days?|weeks?|months?)\b",
        r"\b(?:over|during|in)\s+the\s+(?:past|last|recent|previous)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*(?:days?|weeks?|months?)\b",
    )
    if not any(re.search(pattern, q) for pattern in relative_time_patterns):
        return False, ""
    in_scope_terms = (
        "demand",
        "semiconductor",
        "survey",
        "region",
        "quarter",
        "month",
        "year",
        "vehicle",
        "sales",
        "inventory",
        "shortage",
        "order cancellation",
        "technology",
        "component",
    )
    if not any(term in q for term in in_scope_terms):
        return False, ""
    return True, (
        "The graph contains explicit month, quarter, year, and time-period values, "
        "but it does not define a live rolling window such as 'the past 3 months'."
    )


def _relative_time_approximation_query(question: str) -> Tuple[str, str, str]:
    q = _normalize_question_key(question)
    if not q:
        return "", "", ""
    if "semiconductor" in q and "demand" in q and re.search(r"\b(?:past|last|recent|previous)\s+(?:3|three)?\s*months?\b", q):
        return (
            "Show semiconductor demand by quarter.",
            SEMICONDUCTOR_DEMAND_BY_QUARTER_QUERY,
            (
                "The question asks for the past three months. The KG does not provide a live rolling "
                "three-month window, but the closest available time granularity is quarter-level "
                "semiconductor demand, so the system uses the validated quarter breakdown."
            ),
        )
    return "", "", ""


def _semiconductor_demand_quarter_guided_query(question: str) -> Tuple[str, str, str]:
    q = _normalize_question_key(question)
    if not q:
        return "", "", ""
    if "semiconductor" in q and "demand" in q and "quarter" in q:
        return (
            "Show semiconductor demand by quarter.",
            SEMICONDUCTOR_DEMAND_BY_QUARTER_QUERY,
            "The system found a supported deterministic graph path for semiconductor demand by quarter.",
        )
    if (
        "semiconductor" in q
        and "demand" in q
        and "current demand" not in q
        and re.search(r"\b(?:latest|live|right now|real time|real-time|currently|now)\b", q)
    ):
        return (
            "Show semiconductor demand by quarter.",
            SEMICONDUCTOR_DEMAND_BY_QUARTER_QUERY,
            (
                "The KG is not a live feed, so the system shows the available quarter-level "
                "semiconductor demand breakdown instead of inventing a real-time latest value."
            ),
        )
    return "", "", ""


def _render_out_of_scope_message(reason: str) -> None:
    st.warning(
        "I cannot answer this question because it is outside the available True Demand knowledge graph "
        "and Digital Reference ontology."
    )
    if reason:
        st.caption(reason)
    st.markdown("You can ask graph- or ontology-grounded questions such as:")
    st.markdown("\n".join(f"- {example}" for example in OFF_TOPIC_EXAMPLES))


def _render_unsupported_time_message(reason: str) -> None:
    st.warning("I cannot answer this exact relative-time request from the current graph.")
    if reason:
        st.caption(reason)
    st.markdown("Use an explicit graph time dimension instead, for example:")
    examples = [
        "Show semiconductor demand by quarter.",
        "Show summed percentage change in semiconductor demand for each region per quarter.",
        "Show future semiconductor demand by technology category and quarter.",
        "Show vehicle sales by month.",
    ]
    st.markdown("\n".join(f"- {example}" for example in examples))


def _is_advisory_like_question(question: str) -> bool:
    q = _normalize_question_key(question)
    if not q:
        return False
    advisory_terms = (
        "advise",
        "advice",
        "recommend",
        "suggest",
        "should",
        "monitor",
        "focus",
        "planning attention",
        "look at",
        "inspect",
        "review first",
        "risk",
        "uncertain",
        "uncertainty",
        "exposed",
        "strongest signal",
        "priority",
        "prioritize",
    )
    graph_terms = (
        "demand",
        "shortage",
        "inventory",
        "vehicle",
        "sales",
        "region",
        "technology",
        "semiconductor",
        "survey",
    )
    return any(term in q for term in advisory_terms) and any(term in q for term in graph_terms)


CLARIFICATION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "show",
    "the",
    "this",
    "to",
    "want",
    "what",
    "which",
    "with",
}


def _clarification_tokens(text: str) -> set[str]:
    normalized = _normalize_question_key(text)
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return {token for token in tokens if token not in CLARIFICATION_STOPWORDS and len(token) > 1}


def _clarification_relevance_score(question: str, *texts: str) -> float:
    q_norm = _normalize_question_key(question)
    option_norm = _normalize_question_key(" ".join(str(text or "") for text in texts))
    q_tokens = _clarification_tokens(q_norm)
    option_tokens = _clarification_tokens(option_norm)
    if not q_tokens or not option_tokens:
        return 0.0

    overlap = q_tokens & option_tokens
    score = len(overlap) / max(1, len(q_tokens))
    score += 0.25 * (len(overlap) / max(1, len(option_tokens)))
    score += 0.15 * SequenceMatcher(None, q_norm, option_norm).ratio()

    domain_terms = {
        "demand",
        "current",
        "future",
        "region",
        "regional",
        "quarter",
        "month",
        "year",
        "vehicle",
        "sales",
        "shortage",
        "inventory",
        "technology",
        "semiconductor",
        "survey",
        "oem",
        "tier1",
        "autonomous",
        "driving",
        "risk",
        "uncertain",
        "monitor",
        "planning",
        "focus",
    }
    shared_domain = (q_tokens & option_tokens) & domain_terms
    score += 0.2 * len(shared_domain)
    if "demand" in q_tokens and "demand" not in option_tokens:
        score -= 0.35
    if "shortage" in q_tokens and "shortage" not in option_tokens:
        score -= 0.35
    if "vehicle" in q_tokens and not {"vehicle", "sales"} & option_tokens:
        score -= 0.25
    if "current" in q_tokens and "future" in option_tokens and "future" not in q_tokens:
        score -= 0.6
    if "future" in q_tokens and "current" in option_tokens and "current" not in q_tokens:
        score -= 0.6
    return max(0.0, score)


def _rank_relevant_option_pairs(
    question: str,
    options: List[Tuple[str, str]],
    *,
    max_options: int,
    min_score: float = 0.18,
) -> List[Tuple[str, str]]:
    ranked = []
    for pos, (label, rewritten) in enumerate(options):
        score = _clarification_relevance_score(question, label, rewritten)
        if score >= min_score:
            ranked.append((score, -pos, label, rewritten))
    ranked.sort(reverse=True)
    return [(label, rewritten) for _score, _pos, label, rewritten in ranked[:max_options]]


def _rank_relevant_option_dicts(
    question: str,
    options: List[Dict[str, object]],
    *,
    max_options: int,
    min_score: float = 0.18,
) -> List[Dict[str, object]]:
    ranked = []
    for pos, option in enumerate(options):
        texts = [
            str(option.get("label") or ""),
            str(option.get("rewritten_question") or ""),
            str(option.get("description") or ""),
            str(option.get("what_you_will_see") or ""),
            str(option.get("choose_if") or ""),
            " ".join(str(item) for item in option.get("details") or []),
        ]
        score = _clarification_relevance_score(question, *texts)
        if score >= min_score:
            enriched = dict(option)
            enriched["relevance_score"] = round(score, 3)
            ranked.append((score, -pos, enriched))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item for _score, _pos, item in ranked[:max_options]]


def _advisory_timeout_clarification_options(question: str) -> List[Tuple[str, str]]:
    q = _normalize_question_key(question)
    options: List[Tuple[str, str]] = []

    def add(label: str, rewritten: str) -> None:
        if rewritten and rewritten not in {item[1] for item in options}:
            options.append((label, rewritten))

    if "current" in q or "region" in q or "monitor" in q:
        add(
            "Review current demand by region",
            "Review current demand by region.",
        )
    if "future" in q or "risk" in q or "planning" in q or "focus" in q:
        add(
            "Review future demand by region",
            "Review future demand by region.",
        )
    if "vehicle" in q or "strongest" in q or "signal" in q:
        add(
            "Review future demand by vehicle type",
            "Review future demand by vehicle type.",
        )
    if "technology" in q or "semiconductor" in q or "demand area" in q:
        add(
            "Review future demand by technology category",
            "Review future demand by technology category.",
        )
    if "shortage" in q or "exposed" in q:
        add(
            "Review shortage exposure by survey group",
            "Review shortage exposure by survey group.",
        )

    return _rank_relevant_option_pairs(question, options, max_options=5, min_score=0.12)


def _advisory_request_clarification_result(question: str) -> Dict[str, Any]:
    options = [
        {
            "id": f"advisory_{idx}",
            "label": label,
            "rewritten_question": rewritten,
            "advisory_context": True,
        }
        for idx, (label, rewritten) in enumerate(_advisory_timeout_clarification_options(question), start=1)
    ]
    message = (
        "This is an advisory request, so the system needs a graph-backed evidence view before it can "
        "give conservative guidance. Please choose the evidence path that best matches the decision "
        "you want to support."
    )
    return {
        "answer": message,
        "selected_query": "",
        "candidates": [],
        "errors": [],
        "metadata": {
            "llm_skipped": True,
            "advisory_clarification": True,
        },
        "policy": "advisory_clarification",
        "selection_reason": "Advisory request was too broad for a single deterministic template.",
        "request_clarification": {
            "needs_clarification": True,
            "reason": message,
            "question": "Which graph-backed evidence view should the advice use?",
            "options": options,
        },
        "confidence_route": {
            "enabled": True,
            "route": "clarification",
            "score1": 0.0,
            "score2": 0.0,
            "margin": 0.0,
            "selected_query": "",
            "reason": "advisory request requires a specific evidence view",
            "options": options,
            "safety_flags": ["advisory_requires_evidence_view"],
            "blocking_safety_flags": ["advisory_requires_evidence_view"],
        },
        "answerability": {
            "status": "needs_advisory_clarification",
            "can_answer": False,
            "reason": "Advisory guidance is only generated after choosing a graph-backed evidence view.",
        },
        "clarification": None,
    }


def _timeout_clarification_options(question: str) -> List[Dict[str, str]]:
    q = _normalize_question_key(question)
    options: List[Tuple[str, str]] = []

    def add(label: str, rewritten: str) -> None:
        if rewritten and rewritten not in {item[1] for item in options}:
            options.append((label, rewritten))

    if _is_advisory_like_question(question):
        options = _advisory_timeout_clarification_options(question)
        return [
            {
                "id": f"timeout_{idx}",
                "label": label,
                "rewritten_question": rewritten,
                "advisory_context": True,
            }
            for idx, (label, rewritten) in enumerate(options[:5], start=1)
        ]

    if "semiconductor" in q and "demand" in q:
        add("Available quarter-level semiconductor demand", "Show semiconductor demand by quarter.")
        add(
            "Future semiconductor demand by technology category and quarter",
            "Show future semiconductor demand by technology category and quarter.",
        )
        add("Regional demand by survey group and region", "Break down total regional demand by survey origin and region.")
    if "current" in q and "demand" in q:
        add("Current/total demand by region", "Show current demand by region.")
        add("OEM total demand by region", "List OEM total demand by region.")
    if "region" in q or "regional" in q:
        add("Regional demand by survey group and region", "Break down total regional demand by survey origin and region.")
        add("OEM total demand by region", "List OEM total demand by region.")
    if "vehicle" in q or "sales" in q:
        add("Actual vehicle sales by month", "Show actual vehicle sales by month.")
    if "autonomous" in q or "driving" in q:
        add(
            "Average autonomous-driving development by year",
            "For each year, what is the average autonomous driving development percentage?",
        )
    if "future" in q and "demand" in q:
        add(
            "Future semiconductor demand by technology category and quarter",
            "Show future semiconductor demand by technology category and quarter.",
        )

    ranked_options = _rank_relevant_option_pairs(question, options, max_options=5, min_score=0.18)

    return [
        {
            "id": f"timeout_{idx}",
            "label": label,
            "rewritten_question": rewritten,
        }
        for idx, (label, rewritten) in enumerate(ranked_options, start=1)
    ]


def _interactive_timeout_result(question: str, elapsed_s: float, reason: str = "") -> Dict[str, Any]:
    timeout_s = _interactive_time_budget_sec()
    message = (
        f"The request exceeded the interactive time budget of {timeout_s:.0f} seconds. "
        "The free-text interpretation may still be answerable, but it is not suitable for immediate "
        "automatic answering. Please choose a supported interpretation below or ask with a more "
        "specific metric, time period, scope, and breakdown."
    )
    if reason:
        message = f"{message} {reason}"
    options = _timeout_clarification_options(question)
    return {
        "answer": message,
        "selected_query": "",
        "candidates": [],
        "errors": [message],
        "metadata": {
            "interactive_timeout": True,
            "llm_skipped": False,
            "elapsed_s": elapsed_s,
        },
        "policy": "interactive_timeout_guard",
        "selection_reason": "The interactive request exceeded the configured time budget.",
        "request_clarification": {
            "needs_clarification": True,
            "reason": message,
            "question": "Which supported interpretation is closest to what you meant?",
            "options": options,
        },
        "confidence_route": {
            "enabled": True,
            "route": "clarification",
            "score1": 0.0,
            "score2": 0.0,
            "margin": 0.0,
            "selected_query": "",
            "reason": "interactive timeout; user clarification required",
            "options": options,
            "safety_flags": ["interactive_timeout"],
            "blocking_safety_flags": ["interactive_timeout"],
        },
        "answerability": {
            "status": "interactive_timeout",
            "can_answer": False,
            "reason": message,
        },
    }


@st.cache_data(show_spinner=False)
def _load_guided_query_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for relative in (
        "data/infineon/kgqa_seed_expansion_round1.json",
        "data/infineon/infineon_dev.json",
        "data/infineon/infineon_train.json",
    ):
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in rows if isinstance(rows, list) else []:
            question = str(row.get("question", "") or "").strip()
            query = str(row.get("query", "") or "").strip()
            if question and query:
                lookup[_normalize_question_key(question)] = query
    return lookup


def _guided_pattern_query(pattern: Dict[str, str]) -> str:
    direct = str(pattern.get("query", "") or "").strip()
    if direct:
        return direct
    return _load_guided_query_lookup().get(_normalize_question_key(str(pattern.get("question", ""))), "")


@st.cache_data(show_spinner=False)
def _guided_query_row_count(graph_path: str, fuseki_query_url: str, query: str) -> Tuple[int, str]:
    if not query.strip() or not _graph_backend_available(graph_path):
        return 0, "graph_backend_or_query_missing"
    try:
        rows, error = _preview_query_rows_cached(
            graph_path,
            fuseki_query_url,
            query,
            max_rows=1,
        )
        if error:
            return 0, error
        return len(rows), ""
    except Exception as exc:
        return 0, str(exc)


def _answerable_guided_rows(
    rows: List[Dict[str, str]],
    *,
    graph_path: str,
    fuseki_query_url: str,
) -> List[Dict[str, str]]:
    answerable: List[Dict[str, str]] = []
    for row in rows:
        row_count, query_error = _guided_query_row_count(
            graph_path,
            fuseki_query_url,
            str(row.get("query", "")),
        )
        if row_count <= 0:
            continue
        enriched = dict(row)
        enriched["row_count"] = str(row_count)
        enriched["query_error"] = query_error
        answerable.append(enriched)
    return answerable


@st.cache_data(show_spinner=False)
def _capability_answerable_patterns(graph_path: str, fuseki_query_url: str) -> List[Dict[str, str]]:
    """Build guided rows directly from executable capability templates.

    This is a runtime safety net for the UI. The main guided library may be
    missing or may not match the active backend, but capability templates are
    still allowed if they execute and return graph rows.
    """
    rows: List[Dict[str, str]] = []
    seen = set()
    if not _graph_backend_available(graph_path):
        return rows
    for capability in CAPABILITY_REGISTRY.capabilities:
        for dimension in capability.dimensions:
            question = f"Show {capability.name} by {dimension.name}."
            report = CAPABILITY_REGISTRY.resolve(question)
            query = CAPABILITY_REGISTRY.direct_query_for(report)
            if not query:
                continue
            preview_rows, error = _preview_query_rows_cached(
                graph_path,
                fuseki_query_url,
                query,
                max_rows=1,
            )
            if error or not preview_rows:
                continue
            key = (capability.name.lower(), dimension.name.lower(), query.strip())
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "topic": capability.name.title(),
                    "metric": "values" if capability.name != "shortage" else "count",
                    "breakdown": f"by {dimension.name}",
                    "scope": "all graph data",
                    "question": question,
                    "query": query.strip(),
                    "row_count": str(len(preview_rows)),
                    "query_error": "",
                    "source": "capability_inventory",
                }
            )
    rows.sort(key=lambda row: (row["topic"], row["breakdown"], row["question"]))
    return rows


def _guided_topic_for_question(question: str) -> str:
    q = (question or "").lower()
    if "inventory" in q:
        return "Inventory"
    if "future" in q and "demand" in q:
        return "Future demand"
    if "current" in q and "demand" in q:
        return "Current demand baselines" if ("bl1" in q or "bl2" in q or "baseline" in q) else "Regional demand"
    if "demand" in q and "region" in q:
        return "Regional demand"
    if "vehicle sales" in q or "vehicles sold" in q or "units sold" in q:
        return "Vehicle sales"
    if "shortage" in q:
        return "Shortage"
    if "order cancellation" in q or "cancellation" in q:
        return "Order cancellation"
    if "autonomous" in q or "sae" in q:
        return "Autonomous driving"
    if any(token in q for token in ("names of all", "labels", "how many companies", "how many quarter", "how many technology")):
        return "Catalog lookup"
    return "Other graph-backed questions"


def _guided_metric_for_question(question: str) -> str:
    q = (question or "").lower()
    if "average" in q or "avg" in q or "mean" in q:
        return "average"
    if "highest" in q or "largest" in q or "most" in q or "strongest" in q:
        return "highest / ranked"
    if "count" in q or "how many" in q or "number of" in q:
        return "count"
    if "compare" in q or "versus" in q or " vs " in q or "difference" in q:
        return "comparison"
    if "actual" in q and "forecast" in q:
        return "actual versus forecast"
    if "forecast" in q:
        return "forecasted values"
    if "actual" in q:
        return "actual values"
    if "future demand" in q:
        return "future-demand percentage"
    if "participant" in q:
        return "participant counts"
    if "inventory" in q:
        return "inventory trend / participants"
    if "shortage" in q:
        return "shortage status"
    if "response" in q:
        return "responses"
    if "total" in q or "sum" in q:
        return "total"
    return "values"


def _guided_breakdown_for_question(question: str) -> str:
    q = (question or "").lower()
    dims = []
    for token, label in (
        ("technology", "technology category"),
        ("response type", "response type"),
        ("vehicle type", "vehicle type"),
        ("sae", "SAE level"),
        ("component", "component"),
        ("survey", "survey group"),
        ("region", "region"),
        ("quarter", "quarter"),
        ("month", "month"),
        ("year", "year"),
        ("baseline", "baseline"),
    ):
        if token in q:
            dims.append(label)
    if "bl1" in q or "bl2" in q:
        dims.append("baseline")
    dims = _unique_preserving_order(dims)
    return "by " + " and ".join(dims) if dims else "overall"


def _guided_scope_for_question(question: str) -> str:
    q = (question or "").lower()
    scopes = []
    if "oem" in q:
        scopes.append("OEM")
    if "tier1" in q or "tier 1" in q:
        scopes.append("Tier1")
    if "semiconductor" in q:
        scopes.append("Semiconductor")
    if "automotive" in q:
        scopes.append("Automotive")
    return " + ".join(scopes) if scopes else "all graph data"


@st.cache_data(show_spinner=False)
def _validated_guided_patterns() -> List[Dict[str, str]]:
    patterns: List[Dict[str, str]] = []
    seen = set()
    for relative in (
        "data/infineon/kgqa_seed_expansion_round1.json",
        "data/infineon/infineon_dev.json",
        "data/infineon/infineon_train.json",
    ):
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in rows if isinstance(rows, list) else []:
            question = str(row.get("question", "") or "").strip()
            query = str(row.get("query", "") or "").strip()
            if not question or not query:
                continue
            key = _normalize_question_key(question)
            if key in seen:
                continue
            seen.add(key)
            topic = _guided_topic_for_question(question)
            if topic == "Other graph-backed questions":
                continue
            patterns.append(
                {
                    "topic": topic,
                    "metric": _guided_metric_for_question(question),
                    "breakdown": _guided_breakdown_for_question(question),
                    "scope": _guided_scope_for_question(question),
                    "question": question,
                    "query": query,
                }
            )
    manual = [
        dict(row, query=_guided_pattern_query(row))
        for row in GUIDED_PATTERNS
        if _guided_pattern_query(row)
    ]
    for row in manual:
        key = _normalize_question_key(str(row.get("question", "")))
        if key not in seen:
            seen.add(key)
            patterns.append(row)
    patterns.sort(key=lambda row: (row["topic"], row["metric"], row["breakdown"], row["scope"], row["question"]))
    return patterns


def _guided_answerability(rows: List[Dict[str, str]], error: str = "") -> Dict[str, Any]:
    if error:
        return {
            "status": "query_execution_error",
            "can_answer": False,
            "reason": "The validated guided query could not be executed against the graph.",
            "selected_error": error,
        }
    if rows:
        return {
            "status": "answer_available",
            "can_answer": True,
            "reason": "The validated guided query executed and returned graph rows.",
            "selected_has_rows": True,
            "selected_row_count": len(rows),
        }
    return {
        "status": "no_rows_for_generated_queries",
        "can_answer": False,
        "reason": (
            "The validated guided query returned 0 rows. This pattern should be reviewed "
            "or removed from the guided builder."
        ),
        "selected_has_rows": False,
        "selected_row_count": 0,
    }


EXAMPLE_QUESTIONS = [
    "List OEM total demand by region.",
    "Summarize Tier1 inventory participant totals by component.",
    "Compare actual and forecast vehicle-sales totals by month.",
    "List raw future-demand percentages by technology category and quarter.",
    "What is the average BL1 and BL2 current-demand change for Tier1 Automotive?",
    "List order-cancellation participant observations by technology category and response type.",
    "Which vehicle type has the highest average future-demand change?",
    "How many companies report shortage by survey type?",
    "List companies that reported semiconductor shortage.",
]

GUIDED_PATTERNS = [
    {
        "topic": "Regional demand",
        "metric": "total demand",
        "breakdown": "by region",
        "scope": "OEM",
        "question": "Show total current demand from OEM customers by region.",
    },
    {
        "topic": "Regional demand",
        "metric": "total demand",
        "breakdown": "by region",
        "scope": "Tier1",
        "question": "Show total current demand from Tier1 customers by region.",
    },
    {
        "topic": "Regional demand",
        "metric": "total demand",
        "breakdown": "by region",
        "scope": "Semiconductor",
        "question": "Show total current demand from Semiconductor customers by region.",
    },
    {
        "topic": "Regional demand",
        "metric": "average demand",
        "breakdown": "by quarter",
        "scope": "OEM",
        "question": "Can you show me the average quarterly demand percentage trend based on the OEM survey results?",
    },
    {
        "topic": "Regional demand",
        "metric": "average demand",
        "breakdown": "by quarter",
        "scope": "Tier1",
        "question": "Can you show me the average demand percentage trend for the Tier1 survey by quarter?",
    },
    {
        "topic": "Regional demand",
        "metric": "average demand",
        "breakdown": "by quarter",
        "scope": "Semiconductor",
        "question": "Can you show me the average quarterly percentage trend in demand for the Semiconductor survey?",
    },
    {
        "topic": "Future demand",
        "metric": "future-demand percentage",
        "breakdown": "by region and quarter",
        "scope": "OEM",
        "question": "Show the total percentage of future demand for OEM, detailed by quarter and region.",
    },
    {
        "topic": "Future demand",
        "metric": "future-demand percentage",
        "breakdown": "by region and quarter",
        "scope": "Tier1",
        "question": "Show the overall future demand for Tier1, grouped by region and quarter.",
    },
    {
        "topic": "Future demand",
        "metric": "future-demand percentage",
        "breakdown": "by region and quarter",
        "scope": "Semiconductor",
        "question": "Show the total percentage of future demand for Semiconductor, detailed by quarter and region.",
    },
    {
        "topic": "Future demand",
        "metric": "future-demand percentage",
        "breakdown": "by technology category and quarter",
        "scope": "Semiconductor",
        "question": "Can you provide the total future demand for semiconductors segmented by technology category and quarter?",
    },
    {
        "topic": "Future demand",
        "metric": "average future-demand change",
        "breakdown": "by vehicle type and quarter",
        "scope": "Automotive",
        "question": "What is the average percentage change in future demand broken down by vehicle type and quarter?",
    },
    {
        "topic": "Future demand",
        "metric": "future demand options",
        "breakdown": "by quarter",
        "scope": "Automotive",
        "question": "What is the combined future demand for Option1, Option2, and Option3 in Automotive, broken down by quarter?",
    },
    {
        "topic": "Current demand baselines",
        "metric": "percentage change",
        "breakdown": "for BL1 and BL2",
        "scope": "Tier1",
        "question": "Which percentage changes apply to Tier1 automotive for baselines BL1 and BL2?",
    },
    {
        "topic": "Current demand baselines",
        "metric": "average current-demand change",
        "breakdown": "for BL1 and BL2",
        "scope": "Tier1",
        "question": "What is the average current-demand change for BL1 and BL2 products in the Tier1 Automotive segment?",
    },
    {
        "topic": "Current demand baselines",
        "metric": "difference between BL1 and BL2",
        "breakdown": "overall",
        "scope": "Tier1",
        "question": "What is the total Tier1 current demand percentage change difference between BL1 and BL2?",
    },
    {
        "topic": "Vehicle sales",
        "metric": "actual vehicle sales",
        "breakdown": "by month",
        "scope": "all vehicle sales",
        "question": "What are the monthly vehicle sales totals from actual transactions?",
    },
    {
        "topic": "Vehicle sales",
        "metric": "forecasted vehicle sales",
        "breakdown": "by month",
        "scope": "all vehicle sales",
        "question": "How do the forecasted vehicle unit totals break down by month?",
    },
    {
        "topic": "Vehicle sales",
        "metric": "actual versus forecast",
        "breakdown": "by month",
        "scope": "all vehicle sales",
        "question": "Show me the difference between actual and forecasted vehicle sales totals broken down by month.",
    },
    {
        "topic": "Vehicle sales",
        "metric": "total vehicle sales",
        "breakdown": "by year and vehicle type",
        "scope": "all vehicle sales",
        "question": "Can you show the total number of vehicles sold each year, grouped by type?",
    },
    {
        "topic": "Inventory",
        "metric": "inventory entries",
        "breakdown": "by component and trend",
        "scope": "Tier1",
        "question": "What is the overall Tier1 inventory amount for each component and trend?",
    },
    {
        "topic": "Inventory",
        "metric": "inventory entries",
        "breakdown": "by technology category and trend",
        "scope": "Semiconductor",
        "question": "For each semiconductor technology category and inventory trend, how many inventory entries are recorded?",
    },
    {
        "topic": "Inventory",
        "metric": "total inventory participants",
        "breakdown": "by component",
        "scope": "Tier1",
        "question": "Summarize Tier1 inventory participant totals by component.",
    },
    {
        "topic": "Shortage",
        "metric": "companies reporting shortages",
        "breakdown": "by shortage status",
        "scope": "OEM",
        "question": "What is the number of OEM companies with and without a shortage?",
    },
    {
        "topic": "Shortage",
        "metric": "companies reporting shortages",
        "breakdown": "by shortage status",
        "scope": "Tier1",
        "question": "How many Tier1 companies are experiencing a shortage compared to those that are not?",
    },
    {
        "topic": "Shortage",
        "metric": "companies reporting shortages",
        "breakdown": "by shortage status",
        "scope": "Semiconductor",
        "question": "How many semiconductor companies report a shortage versus no shortage?",
    },
    {
        "topic": "Shortage",
        "metric": "companies reporting shortages",
        "breakdown": "by survey group",
        "scope": "all surveys",
        "question": "How many companies have indicated shortages, grouped by the type of survey?",
    },
    {
        "topic": "Order cancellation",
        "metric": "order-cancellation responses",
        "breakdown": "by technology category",
        "scope": "Semiconductor",
        "question": "What is the total count of order cancellation responses per semiconductor technology category?",
    },
    {
        "topic": "Order cancellation",
        "metric": "order-cancellation responses",
        "breakdown": "by technology category and response type",
        "scope": "Semiconductor",
        "question": "Can you provide the total count of semiconductor order-cancellation responses grouped by technology category and response type?",
    },
    {
        "topic": "Order cancellation",
        "metric": "response trends",
        "breakdown": "by technology category",
        "scope": "Semiconductor",
        "question": "Summarize increase, decrease, and stable order-cancellation response trends by semiconductor technology category.",
    },
    {
        "topic": "Autonomous driving",
        "metric": "average autonomous-driving development",
        "breakdown": "by vehicle type and SAE level",
        "scope": "all autonomous data",
        "question": "What is the average autonomous driving development broken down by vehicle type and SAE level?",
    },
    {
        "topic": "Autonomous driving",
        "metric": "average autonomous-driving development",
        "breakdown": "by vehicle type, SAE level, and year",
        "scope": "OEM",
        "question": "What is the average autonomous driving development for OEMs by vehicle type, SAE level, and year?",
    },
    {
        "topic": "Autonomous driving",
        "metric": "average autonomous-driving development",
        "breakdown": "by vehicle type, SAE level, and year",
        "scope": "Tier1",
        "question": "What is the average autonomous-driving development for Tier1 suppliers, grouped by vehicle type, SAE level, and year?",
    },
    {
        "topic": "Autonomous driving",
        "metric": "Level 5 autonomy percentage",
        "breakdown": "by vehicle type",
        "scope": "all autonomous data",
        "question": "Which vehicle type makes up the largest percentage at Level 5 autonomy?",
    },
    {
        "topic": "Catalog lookup",
        "metric": "available names",
        "breakdown": "regions",
        "scope": "catalog",
        "question": "What are the names of all regions recorded in our database?",
    },
    {
        "topic": "Catalog lookup",
        "metric": "available names",
        "breakdown": "technology categories",
        "scope": "catalog",
        "question": "What are the names of all technology categories?",
    },
    {
        "topic": "Catalog lookup",
        "metric": "available names",
        "breakdown": "quarter labels",
        "scope": "catalog",
        "question": "What are the quarter labels present in our dataset?",
    },
    {
        "topic": "Catalog lookup",
        "metric": "record count",
        "breakdown": "companies",
        "scope": "catalog",
        "question": "Can you tell me how many companies are currently listed?",
    },
]


def _unique_preserving_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _selectbox_index(key: str, options: List[str]) -> int:
    previous = st.session_state.get(key)
    if previous in options:
        return options.index(previous)
    return 0


SMART_QUERY_DOMAIN_TERMS = [
    {
        "label": "future demand",
        "type": "concept",
        "aliases": ["future", "forecast", "projected", "demand"],
        "insert": "future demand",
    },
    {
        "label": "current demand",
        "type": "concept",
        "aliases": ["current", "baseline", "bl1", "bl2", "demand"],
        "insert": "current demand",
    },
    {
        "label": "vehicle sales",
        "type": "concept",
        "aliases": ["vehicle", "sales", "sold", "units"],
        "insert": "vehicle sales",
    },
    {
        "label": "inventory",
        "type": "concept",
        "aliases": ["inventory", "stock", "component"],
        "insert": "inventory",
    },
    {
        "label": "shortage status",
        "type": "concept",
        "aliases": ["shortage", "shortages", "reported"],
        "insert": "shortage status",
    },
    {
        "label": "order cancellation",
        "type": "concept",
        "aliases": ["order", "cancellation", "cancel"],
        "insert": "order cancellation",
    },
    {
        "label": "autonomous driving",
        "type": "concept",
        "aliases": ["autonomous", "driving", "sae"],
        "insert": "autonomous driving",
    },
    {
        "label": "technology category",
        "type": "dimension",
        "aliases": ["technology", "category", "semiconductor"],
        "insert": "technology category",
    },
    {
        "label": "OEM",
        "type": "scope",
        "aliases": ["oem"],
        "insert": "OEM",
    },
    {
        "label": "Tier1",
        "type": "scope",
        "aliases": ["tier", "tier1", "supplier"],
        "insert": "Tier1",
    },
    {
        "label": "Semiconductor",
        "type": "scope",
        "aliases": ["semiconductor", "chip"],
        "insert": "Semiconductor",
    },
]

SMART_QUERY_DIMENSIONS = [
    "region",
    "quarter",
    "year",
    "month",
    "technology category",
    "vehicle type",
    "SAE level",
    "survey",
    "response type",
    "shortage status",
    "component",
    "baseline",
]

def _smart_query_tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", str(text or "").lower()) if len(token) >= 2]


def _smart_query_relevance(text: str, values: List[str]) -> int:
    tokens = set(_smart_query_tokens(text))
    if not tokens:
        return 0
    score = 0
    for value in values:
        value_tokens = set(_smart_query_tokens(value))
        if tokens & value_tokens:
            score += len(tokens & value_tokens)
    return score


def _smart_query_current_fragment(question: str) -> str:
    match = re.search(r"([A-Za-z0-9_+-]*)$", str(question or ""))
    return match.group(1).lower() if match else ""


def _smart_query_has_selected_concept(question: str) -> bool:
    q_lower = str(question or "").lower()
    labels = [str(term.get("label", "")).lower() for term in SMART_QUERY_DOMAIN_TERMS]
    labels.extend(["demand", "sales", "inventory", "shortage", "autonomous driving"])
    return any(re.search(rf"\b{re.escape(label)}\b", q_lower) for label in labels if label)


def _smart_query_domain_suggestions(question: str, schema_path: str, limit: int = 10) -> List[Dict[str, str]]:
    tokens = _smart_query_tokens(question)
    last = _smart_query_current_fragment(question) or (tokens[-1] if tokens else "")
    suggestions: List[Dict[str, str]] = []
    q_lower = str(question or "").lower()
    by_context = bool(re.search(r"\b(by|per|grouped|breakdown)\s+[a-zA-Z0-9_+-]*$", q_lower))
    if by_context:
        for dimension in SMART_QUERY_DIMENSIONS:
            dim_tokens = _smart_query_tokens(dimension)
            if not last or any(token.startswith(last) or last.startswith(token[: min(3, len(token))]) for token in dim_tokens):
                suggestions.append({
                    "label": dimension,
                    "insert": dimension,
                    "type": "dimension",
                    "raw": dimension.upper().replace(" ", "_").replace("-", "_"),
                })
    for term in SMART_QUERY_DOMAIN_TERMS:
        aliases = [str(v).lower() for v in term.get("aliases", [])]
        label = str(term.get("label", ""))
        if not tokens:
            continue
        exact = any(alias in tokens for alias in aliases)
        prefix = bool(last) and any(alias.startswith(last) or last.startswith(alias[: min(3, len(alias))]) for alias in aliases)
        if exact or prefix:
            suggestions.append({
                "label": label,
                "insert": str(term.get("insert", label)),
                "type": str(term.get("type", "concept")),
                "raw": label.upper().replace(" ", "_").replace("-", "_"),
            })

    schema_dict = _load_schema_dict_cached(schema_path)
    schema_values: List[Tuple[str, str]] = []
    schema_values.extend((str(v), "class") for v in list(schema_dict.get("classes") or []))
    schema_values.extend((str(v), "relationship") for v in list(schema_dict.get("predicates") or []))
    schema_values.extend((str(v), "property") for v in list(schema_dict.get("properties") or []))
    schema_values.extend((str(v.get("type", "")), "relationship") for v in list(schema_dict.get("relationships") or []) if isinstance(v, dict))
    for value, value_type in schema_values:
        human = _humanize_axis_value(str(value))
        if not human or len(human) > 42:
            continue
        human_tokens = _smart_query_tokens(human)
        contains_context = any(token in tokens for token in human_tokens)
        prefix_match = bool(last) and any(token.startswith(last) or last.startswith(token[: min(3, len(token))]) for token in human_tokens)
        if prefix_match or (contains_context and len(tokens) <= 5):
            suggestions.append({"label": human, "insert": human, "type": value_type, "raw": str(value)})

    deduped = []
    seen = set()
    for suggestion in suggestions:
        key = suggestion["label"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(suggestion)
    deduped.sort(
        key=lambda item: (
            0 if str(item.get("label", "")).lower().startswith(last) else 1,
            0 if str(item.get("type")) == "dimension" and by_context else 1,
            str(item.get("label", "")).lower(),
        )
    )
    return deduped[:limit]


def _breakdown_dimensions(breakdown: str) -> List[str]:
    cleaned = str(breakdown or "").strip()
    if not cleaned or cleaned.lower() == "overall":
        return []
    cleaned = re.sub(r"^by\s+", "", cleaned, flags=re.I).strip()
    if not cleaned:
        return []
    return [
        part.strip()
        for part in re.split(r"\s+and\s+|,\s*", cleaned)
        if part.strip()
    ]


def _capability_context_terms(row: Dict[str, str]) -> List[str]:
    values = [
        str(row.get("topic", "") or ""),
        str(row.get("metric", "") or ""),
        str(row.get("scope", "") or ""),
    ]
    values.extend(_breakdown_dimensions(str(row.get("breakdown", "") or "")))
    contexts: List[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        contexts.append(value)
        contexts.extend(_smart_query_tokens(value))
    return _unique_preserving_order([c for c in contexts if len(c) >= 2])


def _autocomplete_raw_token(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(label).upper()).strip("_")


def _add_autocomplete_entry(
    entries: Dict[Tuple[str, str], Dict[str, object]],
    *,
    label: str,
    entry_type: str,
    raw: str = "",
    aliases: Optional[List[str]] = None,
    contexts: Optional[List[str]] = None,
    profile: str = "",
    support: int = 1,
) -> None:
    label = str(label or "").strip()
    if not label or len(label) > 58:
        return
    entry_type = str(entry_type or "term").strip().lower()
    key = (label.lower(), entry_type)
    next_aliases = list(aliases or []) + _smart_query_tokens(label) + _smart_query_tokens(raw)
    next_contexts = list(contexts or [])
    if key not in entries:
        entries[key] = {
            "label": label,
            "insert": label,
            "type": entry_type,
            "raw": raw or _autocomplete_raw_token(label),
            "aliases": _unique_preserving_order([a for a in next_aliases if len(str(a)) >= 2]),
            "contexts": _unique_preserving_order([c for c in next_contexts if len(str(c)) >= 2]),
            "profile": profile,
            "support": int(support),
        }
        return
    current = entries[key]
    current["aliases"] = _unique_preserving_order(list(current.get("aliases") or []) + next_aliases)
    current["contexts"] = _unique_preserving_order(list(current.get("contexts") or []) + next_contexts)
    if profile and not current.get("profile"):
        current["profile"] = profile
    current["support"] = int(current.get("support") or 0) + int(support)


def _autocomplete_profile_hint(contexts: List[str], dimension: str) -> str:
    context_set = {str(value).lower() for value in contexts}
    for capability in CAPABILITY_REGISTRY.capabilities:
        if capability.name.lower() not in context_set:
            continue
        dim = capability.dimension_by_name(dimension)
        if not dim:
            continue
        if dim.distinct_values is not None:
            return f"{dim.distinct_values} values"
        if dim.estimated_rows is not None:
            return f"~{dim.estimated_rows} rows"
    return ""


@st.cache_data(show_spinner=False)
def _kg_autocomplete_entries(
    schema_path: str,
    graph_path: str,
    fuseki_query_url: str,
) -> List[Dict[str, object]]:
    """Build answerable autocomplete terms from executable KGQA capabilities.

    This intentionally avoids raw schema-wide completion. A term is shown only when it
    belongs to at least one validated query pattern that returned graph rows.
    """
    schema_dict = _load_schema_dict_cached(schema_path)

    entries: Dict[Tuple[str, str], Dict[str, object]] = {}
    capability_inventory = {}

    answerable_patterns = 0

    autocomplete_patterns = list(_validated_guided_patterns()) + _capability_answerable_patterns(
        graph_path,
        fuseki_query_url,
    )

    for row in autocomplete_patterns:

        query = str(row.get("query", "") or "").strip()

        if not query:
            continue

        rows, _error = _preview_query_rows_cached(
            graph_path,
            fuseki_query_url,
            query,
            max_rows=1,
        )

        if not rows:
            continue

        answerable_patterns += 1

        contexts = _capability_context_terms(row)

        topic = str(row.get("topic", "") or "").strip()
        metric = str(row.get("metric", "") or "").strip()
        scope = str(row.get("scope", "") or "").strip()

        dimensions = _breakdown_dimensions(
            str(row.get("breakdown", "") or "")
        )

        if topic:
            capability_inventory.setdefault(topic, set())

            for dim in dimensions:
                capability_inventory[topic].add(dim)

        if topic:
            _add_autocomplete_entry(
                entries,
                label=topic,
                entry_type="concept",
                raw=_autocomplete_raw_token(topic),
                aliases=_smart_query_tokens(topic),
                contexts=contexts,
            )
            if "demand" in topic.lower():
                _add_autocomplete_entry(
                    entries,
                    label="Demand",
                    entry_type="concept",
                    raw="DEMAND",
                    aliases=["demand", "demands"],
                    contexts=contexts,
                )

        if metric and metric.lower() not in {"available names", "record count"}:
            _add_autocomplete_entry(
                entries,
                label=metric,
                entry_type="metric",
                raw=_autocomplete_raw_token(metric),
                aliases=_smart_query_tokens(metric),
                contexts=contexts,
            )

        if scope and scope.lower() not in {"all graph data", "all surveys", "catalog"}:
            _add_autocomplete_entry(
                entries,
                label=scope,
                entry_type="scope",
                raw=_autocomplete_raw_token(scope),
                aliases=_smart_query_tokens(scope),
                contexts=contexts,
            )

        for dimension in dimensions:
            _add_autocomplete_entry(
                entries,
                label=dimension,
                entry_type="dimension",
                raw=_autocomplete_raw_token(dimension),
                aliases=_smart_query_tokens(dimension),
                contexts=contexts,
                profile=_autocomplete_profile_hint(contexts, dimension),
            )

    out = list(entries.values())

    out.sort(
        key=lambda entry: (
            0 if str(entry.get("type")) in {"concept", "metric"} else 1,
            -int(entry.get("support") or 0),
            str(entry.get("label", "")).lower(),
        )
    )

    for entry in out:
        entry["source"] = "answerable_capability"
        entry["inventory_patterns"] = answerable_patterns

    return out


def _smart_query_pattern_suggestions(question: str, limit: int = 5) -> List[Dict[str, str]]:
    tokens = _smart_query_tokens(question)
    if not tokens:
        return []
    patterns = _validated_guided_patterns()
    scored = []
    for row in patterns:
        haystack = " ".join(
            str(row.get(key, ""))
            for key in ("topic", "metric", "breakdown", "scope", "question")
        )
        score = _smart_query_relevance(" ".join(tokens), [haystack])
        if score <= 0:
            continue
        scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1].get("topic", ""), item[1].get("question", "")))
    return [dict(row) for _score, row in scored[:limit]]


def _smart_query_dimension_suggestions(patterns: List[Dict[str, str]], limit: int = 6) -> List[str]:
    values = []
    for row in patterns:
        breakdown = str(row.get("breakdown", "")).strip()
        if not breakdown or breakdown == "overall":
            continue
        cleaned = breakdown.removeprefix("by ").strip()
        for part in re.split(r"\s+and\s+|,\s*", cleaned):
            part = part.strip()
            if part:
                values.append(part)
    return _unique_preserving_order(values)[:limit]


def _render_smart_query_assistant(question: str, schema_path: str) -> None:
    q = str(question or "")
    if len(q.strip()) < 3:
        return
    term_suggestions = _smart_query_domain_suggestions(q, schema_path=schema_path)
    if not term_suggestions:
        return

    st.markdown(
        """
        <style>
          .kg-autocomplete-title {
            color: #008b84;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin: -0.45rem 0 0.2rem;
            text-transform: uppercase;
          }
          .kg-autocomplete-footer {
            border-top: 1px solid #d9e5e8;
            color: #64748b;
            font-size: 0.78rem;
            margin-top: 0.15rem;
            padding-top: 0.35rem;
          }
          .kg-type-badge {
            background: #eef8f7;
            border: 1px solid #c8e9e5;
            border-radius: 999px;
            color: #0f766e;
            display: inline-block;
            font-size: 0.76rem;
            font-weight: 700;
            padding: 0.18rem 0.55rem;
          }
          .kg-raw-token {
            color: #475569;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.78rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("<div class='kg-autocomplete-title'>Graph-aware autocomplete</div>", unsafe_allow_html=True)
        for idx, suggestion in enumerate(term_suggestions[:7]):
            label = str(suggestion.get("label", ""))
            suggestion_type = str(suggestion.get("type", "term"))
            raw = str(suggestion.get("raw", "") or "")
            cols = st.columns([4.3, 1.6, 3.1])
            cols[0].button(
                label,
                key=f"smart_term_{idx}_{label}_{suggestion_type}",
                use_container_width=True,
                on_click=_complete_question_fragment,
                args=(q, str(suggestion.get("insert", label))),
            )
            cols[1].markdown(f"<span class='kg-type-badge'>{escape(suggestion_type)}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"<span class='kg-raw-token'>{escape(raw)}</span>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='kg-autocomplete-footer'>Showing {min(7, len(term_suggestions))} of {len(term_suggestions)} suggestions from the schema/query context.</div>",
            unsafe_allow_html=True,
        )


def _render_kg_autocomplete_input(schema_path: str, graph_path: str, fuseki_query_url: str) -> str:
    current = str(st.session_state.get("question_input", "") or "")
    revision = int(st.session_state.get("question_input_revision", 0) or 0)
    result = KG_AUTOCOMPLETE_COMPONENT(
        label="Your question",
        value=current,
        entries=_kg_autocomplete_entries(schema_path, graph_path, fuseki_query_url),
        key=f"kg_question_autocomplete_{revision}",
        default={"text": current},
    )
    if isinstance(result, dict):
        text = str(result.get("text", "") or "")
    else:
        text = current
    st.session_state["question_input"] = text
    return text


def _dr_question_for_term(term: Dict[str, object]) -> str:
    label = str(term.get("label") or "Unknown term")
    kind = str(term.get("kind") or "resource")
    return f"What does {label} mean?" if "property" in kind else f"What is {label}?"


def _render_dr_term_card(term: Dict[str, object], key_prefix: str, expanded: bool = False) -> None:
    label = str(term.get("label") or "Unknown term")
    kind = str(term.get("kind") or "resource")
    definition = str(term.get("definition") or "").strip()
    parents = [str(value) for value in term.get("parents") or []]
    domains = [str(value) for value in term.get("domains") or []]
    ranges = [str(value) for value in term.get("ranges") or []]
    question = _dr_question_for_term(term)

    with st.expander(f"{label} [{kind}]", expanded=expanded):
        if definition:
            st.write(definition)
        else:
            st.caption("No explicit definition text is available; the ontology still declares this term.")
        meta_rows = []
        if parents:
            meta_rows.append({"Field": "Parents / superclasses", "Value": ", ".join(parents[:8])})
        if domains:
            meta_rows.append({"Field": "Domain", "Value": ", ".join(domains[:8])})
        if ranges:
            meta_rows.append({"Field": "Range", "Value": ", ".join(ranges[:8])})
        if meta_rows:
            st.dataframe(meta_rows, width="stretch", hide_index=True)
        st.button(
            f"Ask: {question}",
            key=f"{key_prefix}_{_normalize_question_key(label)}",
            type="secondary",
            on_click=_set_guided_question_input,
            args=(question, ""),
        )


def _dr_class_roots_and_descendants(terms: List[Dict[str, object]]) -> Dict[str, List[Tuple[int, Dict[str, object]]]]:
    class_terms = [term for term in terms if str(term.get("kind") or "") == "class"]
    by_label = {str(term.get("label") or ""): term for term in class_terms if str(term.get("label") or "")}
    children: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    roots: List[Dict[str, object]] = []
    for term in class_terms:
        parents = [parent for parent in term.get("parents") or [] if str(parent) in by_label]
        if not parents:
            roots.append(term)
        for parent in parents:
            children[str(parent)].append(term)

    grouped: Dict[str, List[Tuple[int, Dict[str, object]]]] = {}
    for root in sorted(roots, key=lambda item: str(item.get("label") or "").lower()):
        root_label = str(root.get("label") or "")
        rows: List[Tuple[int, Dict[str, object]]] = []
        seen = set()

        def visit(node: Dict[str, object], depth: int) -> None:
            label = str(node.get("label") or "")
            if label in seen:
                return
            seen.add(label)
            rows.append((depth, node))
            for child in sorted(children.get(label, []), key=lambda item: str(item.get("label") or "").lower()):
                visit(child, depth + 1)

        visit(root, 0)
        grouped[root_label] = rows
    return grouped


def _dr_property_groups(terms: List[Dict[str, object]], kind: str) -> Dict[str, List[Dict[str, object]]]:
    rows = [term for term in terms if str(term.get("kind") or "") == kind]
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for term in rows:
        domains = [str(value) for value in term.get("domains") or [] if str(value).strip()]
        ranges = [str(value) for value in term.get("ranges") or [] if str(value).strip()]
        if domains:
            group = f"Domain: {domains[0]}"
        elif ranges:
            group = f"Range: {ranges[0]}"
        else:
            group = "No declared domain/range"
        grouped[group].append(term)
    return {
        group: sorted(items, key=lambda item: str(item.get("label") or "").lower())
        for group, items in sorted(grouped.items(), key=lambda item: item[0].lower())
    }


def _render_dr_search_tab(total_terms: int) -> None:
    search = st.text_input(
        "Search Digital Reference concepts",
        value="",
        placeholder="capacity, resource, product group, processed by...",
        key="dr_ontology_search",
    )
    kind_options = ["class", "object property", "datatype property", "annotation property"]
    selected_kinds = st.multiselect(
        "Term types",
        kind_options,
        default=["class", "object property", "datatype property"],
        key="dr_ontology_kind_filter",
    )
    limit = st.slider("Results to show", 10, 100, 25, 5, key="dr_ontology_limit")
    if not search.strip():
        st.info(
            "Type a term to search the ontology, or use Browse hierarchy to inspect all DR terms by lobe/domain."
        )
        return
    rows = search_dr_ontology_terms(search=search, kinds=selected_kinds, limit=int(limit))
    st.caption(
        f"Showing {len(rows)} search result(s) from {total_terms:,} DR ontology terms. "
        "Use Browse hierarchy to page through all terms."
    )

    if not rows:
        st.info("No matching Digital Reference terms found.")
        return

    for idx, term in enumerate(rows):
        _render_dr_term_card(
            term,
            key_prefix=f"dr_search_ask_{idx}",
            expanded=(idx < 3 and bool(search.strip())),
        )


def _render_dr_browse_tab() -> None:
    terms = dr_ontology_terms()
    if not terms:
        st.warning("No Digital Reference terms are available for browsing.")
        return

    browse_kind = st.selectbox(
        "Browse type",
        ["class", "object property", "datatype property", "annotation property"],
        key="dr_browse_kind",
    )
    page_size = st.selectbox(
        "Terms per page",
        [25, 50, 100, 200],
        index=1,
        key="dr_browse_page_size",
    )

    if browse_kind == "class":
        grouped = _dr_class_roots_and_descendants(terms)
        group_options = [
            f"{label} ({len(rows)})"
            for label, rows in grouped.items()
        ]
        group_lookup = {f"{label} ({len(rows)})": label for label, rows in grouped.items()}
        st.caption("Classes are grouped by the top-level DR lobe / root class, similar to Protege.")
        selected_group = st.selectbox("Class hierarchy root", group_options, key="dr_class_root")
        selected_label = group_lookup[selected_group]
        hierarchy_rows = grouped[selected_label]
        total = len(hierarchy_rows)
        max_page = max(1, math.ceil(total / int(page_size)))
        page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="dr_class_page")
        start = (int(page) - 1) * int(page_size)
        visible_rows = hierarchy_rows[start:start + int(page_size)]
        st.caption(f"Showing {start + 1}-{min(start + len(visible_rows), total)} of {total} terms in {selected_label}.")
        for idx, (depth, term) in enumerate(visible_rows):
            indent = "&nbsp;" * min(depth * 5, 40)
            st.markdown(
                f"{indent}<span style='color:#d4a400;font-weight:700;'>●</span> "
                f"<strong>{escape(str(term.get('label') or 'Unknown term'))}</strong>",
                unsafe_allow_html=True,
            )
            _render_dr_term_card(
                term,
                key_prefix=f"dr_browse_class_{start + idx}",
                expanded=False,
            )
        return

    grouped_properties = _dr_property_groups(terms, browse_kind)
    group_options = [f"{label} ({len(rows)})" for label, rows in grouped_properties.items()]
    group_lookup = {f"{label} ({len(rows)})": label for label, rows in grouped_properties.items()}
    st.caption("Properties are grouped by declared domain first, then by range when no domain is available.")
    selected_group = st.selectbox("Property group", group_options, key=f"dr_{browse_kind}_group")
    selected_label = group_lookup[selected_group]
    property_rows = grouped_properties[selected_label]
    total = len(property_rows)
    max_page = max(1, math.ceil(total / int(page_size)))
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key=f"dr_{browse_kind}_page")
    start = (int(page) - 1) * int(page_size)
    visible_rows = property_rows[start:start + int(page_size)]
    st.caption(f"Showing {start + 1}-{min(start + len(visible_rows), total)} of {total} terms in {selected_label}.")
    for idx, term in enumerate(visible_rows):
        _render_dr_term_card(
            term,
            key_prefix=f"dr_browse_property_{browse_kind}_{start + idx}",
            expanded=False,
        )


def _render_dr_ontology_browser() -> None:
    counts = dr_ontology_counts()
    if not counts.get("total"):
        st.warning(
            "Digital Reference ontology terms are not available. "
            "Set TRUE_DEMAND_DR_ONTOLOGY_PATH or DR_ONTOLOGY_PATH to the DigitalReference.ttl file."
        )
        return

    count_cols = st.columns(4)
    count_cols[0].metric("Unique DR terms", f"{counts.get('total', 0):,}")
    count_cols[1].metric("Classes", f"{counts.get('class', 0):,}")
    count_cols[2].metric("Object properties", f"{counts.get('object property', 0):,}")
    count_cols[3].metric("Datatype properties", f"{counts.get('datatype property', 0):,}")

    st.caption(
        "Browse or search the Digital Reference vocabulary. "
        f"The index contains {counts.get('searchable_entries', counts.get('total', 0)):,} searchable labels/aliases. "
        "Selecting a term creates a deterministic definition question; it does not call the LLM."
    )
    browse_tab, search_tab = st.tabs(["Browse hierarchy", "Search"])
    with browse_tab:
        _render_dr_browse_tab()
    with search_tab:
        _render_dr_search_tab(int(counts.get("total") or 0))


def _render_question_guidance(graph_path: str, fuseki_query_url: str) -> None:
    with st.expander("Question guide", expanded=False):
        st.caption(
            "Use the question box above for free text, or pick an example/builder option here."
        )
        tabs = st.tabs(["Examples", "Guided builder", "Available topics", "Digital Reference"])
        with tabs[0]:
            answerable_patterns = _answerable_guided_rows(
                _validated_guided_patterns(),
                graph_path=graph_path,
                fuseki_query_url=fuseki_query_url,
            )
            if not answerable_patterns:
                answerable_patterns = _capability_answerable_patterns(
                    graph_path,
                    fuseki_query_url,
                )
            query_lookup = {
                _normalize_question_key(str(row.get("question", ""))): str(row.get("query", ""))
                for row in answerable_patterns
            }
            example_options = [
                example
                for example in EXAMPLE_QUESTIONS
                if query_lookup.get(_normalize_question_key(example), "")
            ]
            if not example_options:
                example_options = [
                    str(row.get("question", ""))
                    for row in answerable_patterns[:20]
                    if str(row.get("question", "")).strip()
                ]
            if not example_options:
                st.warning("No validated examples are available.")
            else:
                selected_example = st.selectbox(
                    "Example question",
                    example_options,
                    key="selected_example_question",
                )
                st.button(
                    "Use selected example",
                    key="use_selected_example",
                    type="secondary",
                    on_click=_set_guided_question_input,
                    args=(
                        selected_example,
                        query_lookup.get(_normalize_question_key(selected_example), ""),
                    ),
                )
                st.caption(
                    "Examples use validated graph queries that returned rows on the active graph backend. "
                    "Press Ask after selecting one."
                )
        with tabs[1]:
            validated_patterns = _answerable_guided_rows(
                _validated_guided_patterns(),
                graph_path=graph_path,
                fuseki_query_url=fuseki_query_url,
            )
            if not validated_patterns:
                validated_patterns = _capability_answerable_patterns(
                    graph_path,
                    fuseki_query_url,
                )
            topic_options = _unique_preserving_order([row["topic"] for row in validated_patterns])
            if not topic_options:
                st.warning("No answerable guided patterns are available for the active graph backend.")
                return
            topic = st.selectbox(
                "Topic",
                topic_options,
                index=_selectbox_index("guided_topic", topic_options),
                key="guided_topic",
            )
            topic_rows = [row for row in validated_patterns if row["topic"] == topic]

            metric_options = _unique_preserving_order([row["metric"] for row in topic_rows])
            metric = st.selectbox(
                "Metric",
                metric_options,
                index=_selectbox_index("guided_metric", metric_options),
                key="guided_metric",
            )
            metric_rows = [row for row in topic_rows if row["metric"] == metric]

            breakdown_options = _unique_preserving_order([row["breakdown"] for row in metric_rows])
            breakdown = st.selectbox(
                "Breakdown",
                breakdown_options,
                index=_selectbox_index("guided_breakdown", breakdown_options),
                key="guided_breakdown",
            )
            breakdown_rows = [row for row in metric_rows if row["breakdown"] == breakdown]

            scope_options = _unique_preserving_order([row["scope"] for row in breakdown_rows])
            scope = st.selectbox(
                "Survey / scope",
                scope_options,
                index=_selectbox_index("guided_scope", scope_options),
                key="guided_scope",
            )
            scope_rows = [row for row in breakdown_rows if row["scope"] == scope]
            if not scope_rows:
                st.warning(
                    "No validated question exists for this exact combination. "
                    "Choose a different metric, breakdown, or scope."
                )
            else:
                question_options = [row["question"] for row in scope_rows]
                selected_question = st.selectbox(
                    "Validated question",
                    question_options,
                    index=_selectbox_index("guided_question", question_options),
                    key="guided_question",
                )
                selected_pattern = next(
                    row for row in scope_rows
                    if row["question"] == selected_question
                )
                built_question = str(selected_pattern["question"])
                st.text_input("Generated question", value=built_question, disabled=True)
                st.caption("This question comes from the validated graph-query library and returned rows.")
                st.button(
                    "Use generated question",
                    key="use_guided_question",
                    type="secondary",
                    on_click=_set_guided_question_input,
                    args=(built_question, str(selected_pattern["query"])),
                )
                st.caption("Press Ask after using the generated question.")
        with tabs[2]:
            topic_rows = []
            validated_patterns = _answerable_guided_rows(
                _validated_guided_patterns(),
                graph_path=graph_path,
                fuseki_query_url=fuseki_query_url,
            )
            if not validated_patterns:
                validated_patterns = _capability_answerable_patterns(
                    graph_path,
                    fuseki_query_url,
                )
            for topic in _unique_preserving_order([row["topic"] for row in validated_patterns]):
                rows = [row for row in validated_patterns if row["topic"] == topic]
                topic_rows.append(
                    {
                        "Topic": topic,
                        "Typical metrics": ", ".join(_unique_preserving_order([row["metric"] for row in rows])[:4]),
                        "Useful breakdowns": ", ".join(_unique_preserving_order([row["breakdown"] for row in rows])[:4]),
                        "Scopes": ", ".join(_unique_preserving_order([row["scope"] for row in rows])),
                        "Answerable patterns": len(rows),
                    }
                )
            st.dataframe(topic_rows, width="stretch", hide_index=True)
            st.caption(
                "The builder is generated from validated question/query pairs. "
                "Free text remains available for other questions."
            )
        with tabs[3]:
            _render_dr_ontology_browser()


def _overview_topic_groups(schema_dict: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    classes = set(schema_dict.get("classes") or [])
    groups = [
        ("Demand analysis", ["CurrentDemandAnalysis", "FutureDemandAnalysis", "DemandForRegion"]),
        ("Vehicle sales", ["VehicleSalesObservation", "YearlySalesData"]),
        ("Autonomous driving", ["AutonomousDrivingDevelopment"]),
        ("Order cancellation", ["OrderCancellation"]),
        ("Inventory", ["InventoryDevelopment", "Inventory"]),
        ("Shortages and companies", ["Company", "Shortage"]),
    ]
    return [(label, [name for name in names if name in classes]) for label, names in groups]


def _overview_relationship_triples(schema_dict: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    seen = set()
    for rel in list(schema_dict.get("relationships") or []):
        predicate = str(rel.get("type") or "").strip()
        from_nodes = list(rel.get("from") or [])
        to_nodes = list(rel.get("to") or [])
        if not predicate or not from_nodes or not to_nodes:
            continue
        for source in from_nodes:
            for target in to_nodes:
                triple = (str(source), predicate, str(target))
                if triple in seen:
                    continue
                seen.add(triple)
                triples.append(triple)
    return triples


def _graph_data_stats(graph_path: str) -> Dict[str, int]:
    if not graph_path or not _graph_backend_available(graph_path):
        return {}
    graph = _run_with_timeout(
        lambda: _load_active_graph(graph_path),
        _interactive_query_timeout_sec(),
        label="graph backend load",
    )
    if _active_fuseki_query_url():
        try:
            triples, _ = _execute_query_preview(
                graph,
                "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }",
                max_rows=1,
            )
            subjects, _ = _execute_query_preview(
                graph,
                "SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE { ?s ?p ?o }",
                max_rows=1,
            )
            resources, _ = _execute_query_preview(
                graph,
                """
                SELECT (COUNT(DISTINCT ?x) AS ?count)
                WHERE {
                  { ?x ?p ?o }
                  UNION
                  { ?s ?p ?x FILTER(isIRI(?x) || isBlank(?x)) }
                }
                """,
                max_rows=1,
            )
            return {
                "triples": int(next(iter(triples[0].values()))) if triples else 0,
                "resource_nodes": int(next(iter(resources[0].values()))) if resources else 0,
                "subject_entities": int(next(iter(subjects[0].values()))) if subjects else 0,
            }
        except Exception:
            return {}
    resources = set()
    subjects = set()
    for s, _p, o in graph:
        if isinstance(s, (URIRef, BNode)):
            subjects.add(s)
            resources.add(s)
        if isinstance(o, (URIRef, BNode)):
            resources.add(o)
    return {
        "triples": len(graph),
        "resource_nodes": len(resources),
        "subject_entities": len(subjects),
    }


def _graph_overview_report(
    schema_dict: Dict[str, Any],
    graph_stats: Dict[str, int],
    *,
    include_inventory: bool,
) -> str:
    groups = _overview_topic_groups(schema_dict)
    predicates = list(schema_dict.get("predicates") or [])
    properties = list(schema_dict.get("properties") or [])
    relationships = list(schema_dict.get("relationships") or [])
    topic_lines = "\n".join(
        f"- **{label}:** {', '.join(items)}"
        for label, items in groups
        if items
    )
    examples = [
        "Return monthly totals for actual vehicle-sales observations.",
        "Compare BL1 and BL2 current-demand changes for Tier1 Automotive.",
        "Group order-cancellation participant counts by technology category and response type.",
        "Which month has the highest actual vehicle sales?",
        "Break down total regional demand by survey origin.",
        "Show inventory trends by component.",
        "What is a Technology Node?",
        "Define Future Demand.",
        "Which region should be monitored more closely based on current demand?",
    ]
    example_lines = "\n".join(f"- {item}" for item in examples)
    triples_text = f"{graph_stats['triples']:,}" if "triples" in graph_stats else "Unavailable"
    resource_nodes_text = (
        f"{graph_stats['resource_nodes']:,}" if "resource_nodes" in graph_stats else "Unavailable"
    )
    subject_entities_text = (
        f"{graph_stats['subject_entities']:,}" if "subject_entities" in graph_stats else "Unavailable"
    )
    class_lines = "\n".join(f"- {item}" for item in sorted(schema_dict.get("classes") or []))
    predicate_lines = "\n".join(f"- {item}" for item in sorted(predicates))
    property_lines = "\n".join(f"- {item}" for item in sorted(properties))
    relationship_lines = "\n".join(
        f"- {rel.get('type')}: {', '.join(rel.get('from') or []) or '(unspecified)'} -> "
        f"{', '.join(rel.get('to') or []) or '(literal / unspecified)'}"
        for rel in relationships
    )
    summary = f"""# True Demand KG Overview

## One-page summary
This application provides natural-language access to the True Demand knowledge graph and selected ontology definitions. The main graph describes survey and analytical data around semiconductor and automotive demand. It connects regional demand, current- and future-demand analyses, vehicle sales, autonomous-driving development, order-cancellation responses, shortages, inventory trends, technology categories, vehicle types, companies, components, survey origins, and time periods. The Digital Reference ontology is used as a deterministic definition layer for concept and property questions.

## What the graph contains
{topic_lines}

## What users can ask
- **KG analytics questions:** totals, averages, counts, rankings, and grouped breakdowns over True Demand data.
- **Ontology definition questions:** concept and property explanations such as "What is a Technology Node?" or "What does is processed by mean?"
- **Graph-grounded advisory questions:** conservative planning signals based on available graph results, such as which region or technology category should be reviewed first.

## Main dimensions users can ask about
- **Time:** month, quarter, year, time period
- **Technology:** technology category / technology node
- **Vehicle:** BEV, BEHV, ICE, SAE level
- **Business grouping:** survey origin, region, company, component, response type, baseline, market segment

## Main measures users can ask about
- Demand and total demand
- Percentage change and current/future demand changes
- Units sold and yearly sales
- Participant counts and shortage values
- Inventory trend / target status
- Autonomous-driving percentages

## Data graph scale
- RDF triples: {triples_text}
- Resource nodes / entities: {resource_nodes_text}
- Subject entities: {subject_entities_text}

## Schema scale
- Classes: {len(schema_dict.get("classes") or [])}
- Predicates: {len(predicates)}
- Properties: {len(properties)}
- Declared relationships: {len(relationships)}

## Example questions
{example_lines}

## How to use it
Use precise wording when you know the intended calculation, such as **average**, **total**, **count**, **highest**, **by month**, or **by technology category**. Use definition-style wording for ontology questions, for example **define**, **what is**, or **what does ... mean**. If a question leaves the intended interpretation open, the QA system may ask for clarification before answering.
"""
    if not include_inventory:
        return summary

    return summary + f"""
## Complete schema inventory

### All classes
{class_lines}

### All predicates
{predicate_lines}

### All properties
{property_lines}

### All declared relationships
{relationship_lines}
"""


def _schema_inventory_summary(schema_dict: Dict[str, Any]) -> Dict[str, int]:
    return {
        "classes": len(schema_dict.get("classes") or []),
        "predicates": len(schema_dict.get("predicates") or []),
        "properties": len(schema_dict.get("properties") or []),
        "relationships": len(schema_dict.get("relationships") or []),
    }


def _safe_graph_data_stats(graph_path: str) -> Dict[str, int]:
    try:
        stats = _graph_data_stats(graph_path)
        if stats:
            return stats
    except Exception:
        pass
    quality_path = PROJECT_ROOT / "results" / "graph_quality_report.json"
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, int] = {}
    if quality.get("triple_count") is not None:
        out["triples"] = int(quality.get("triple_count") or 0)
    if quality.get("entity_count") is not None:
        entity_count = int(quality.get("entity_count") or 0)
        out["resource_nodes"] = entity_count
        out["subject_entities"] = entity_count
    return out


def _supported_topics_lines(schema_dict: Dict[str, Any], *, compact: bool = False) -> List[str]:
    groups = _overview_topic_groups(schema_dict)
    lines = []
    for label, items in groups:
        if not items:
            continue
        joined = ", ".join(items[:4] if compact else items)
        if compact and len(items) > 4:
            joined += ", ..."
        lines.append(f"- **{label}:** {joined}")
    return lines


def _capability_examples_for_response() -> List[str]:
    return [
        "Show current demand by region.",
        "Show semiconductor demand by quarter.",
        "Show future demand by technology category and quarter.",
        "Show vehicle sales by month.",
        "Review shortage exposure by survey group.",
        "What is a Technology Node?",
    ]


def _source_scope_answer(
    question: str,
    schema_dict: Dict[str, Any],
    graph_stats: Dict[str, int],
    dr_ontology_path: str,
) -> Optional[str]:
    q_norm = _normalize_question_key(question)
    source_intent = bool(
        re.search(r"\b(sources?|data sources?|scope|summar(y|ize)|overview|brief|loaded|contains?|cover|covers|covered|coverage)\b", q_norm)
    )
    if not source_intent:
        return None

    mentions_true_demand = bool(re.search(r"\b(true demand|demand graph|kg|knowledge graph|graph data)\b", q_norm))
    mentions_dr = bool(re.search(r"\b(digital reference|dr ontology|ontology|definitions?|concepts?)\b", q_norm))
    asks_broad_sources = bool(re.search(r"\b(sources?|data sources?|my sources?|loaded)\b", q_norm))
    if not (mentions_true_demand or mentions_dr or asks_broad_sources):
        return None

    inventory = _schema_inventory_summary(schema_dict)
    triples = graph_stats.get("triples")
    entities = graph_stats.get("resource_nodes") or graph_stats.get("subject_entities")
    dr_counts: Dict[str, int] = {}
    if mentions_dr or asks_broad_sources:
        try:
            dr_counts = dr_ontology_counts(dr_ontology_path)
        except Exception:
            dr_counts = {}

    true_demand_bits = [
        "The **True Demand knowledge graph** is the analytical source.",
        "It contains survey-grounded semiconductor and automotive supply-chain data: demand, current/future demand, regional demand, shortages, inventory, order cancellation, vehicle sales, autonomous-driving indicators, companies, regions, technologies, vehicles, and time periods.",
        f"Schema scale: {inventory['classes']} classes, {inventory['predicates']} object-style predicates, {inventory['properties']} datatype/data properties, and {inventory['relationships']} declared relationships.",
    ]
    if triples is not None:
        true_demand_bits.append(f"Data scale: {triples:,} RDF triples.")
    if entities is not None:
        true_demand_bits.append(f"Entity scale: about {entities:,} graph entities/resources.")

    dr_bits = [
        "The **Digital Reference ontology** is the terminology and definition source.",
        "It is used for concept-level questions, relationship meanings, labels, aliases, class/property descriptions, and ontology browsing.",
        "It does not replace the True Demand analytical graph; it explains the vocabulary and model concepts used around the graph.",
    ]
    if dr_counts:
        dr_bits.append(
            "DR searchable scale: "
            f"{int(dr_counts.get('searchable_entries') or dr_counts.get('total') or 0):,} indexed terms "
            f"({int(dr_counts.get('class') or 0):,} classes, "
            f"{int(dr_counts.get('object_property') or 0):,} object properties, "
            f"{int(dr_counts.get('datatype_property') or 0):,} datatype properties)."
        )

    if mentions_true_demand and not mentions_dr:
        return " ".join(true_demand_bits)
    if mentions_dr and not mentions_true_demand:
        return " ".join(dr_bits)
    return "The app currently uses two main sources:\n\n- " + "\n- ".join(
        [" ".join(true_demand_bits), " ".join(dr_bits)]
    )


def _capability_support_answer(question: str, schema_dict: Dict[str, Any]) -> Optional[str]:
    q_norm = _normalize_question_key(question)
    support_intent = bool(
        re.search(r"\b(can|could|do|does|support|have|available|covered|contain|contains|include|includes)\b", q_norm)
    )
    if not support_intent:
        return None
    report = CAPABILITY_REGISTRY.resolve(question)
    capability_name = report.primary_capability
    dimension_names = [item.name for item in report.detected_dimensions]
    if not capability_name and not dimension_names:
        return None

    if capability_name:
        capability = CAPABILITY_REGISTRY.find_capability(capability_name)
        if not capability:
            return None
        supported_dimensions = [dimension.name for dimension in capability.dimensions]
        if dimension_names:
            matched = [name for name in dimension_names if name in supported_dimensions]
            if matched:
                return (
                    f"Yes. The system has graph-supported **{capability_name}** questions "
                    f"with breakdowns such as **{', '.join(matched)}**. "
                    "For best results, ask with an explicit metric and breakdown, for example: "
                    f"\"Show {capability_name} by {matched[0]}.\""
                )
            return (
                f"The system supports **{capability_name}**, but I do not see that exact breakdown "
                f"as a deterministic capability. Supported breakdowns are: {', '.join(supported_dimensions)}."
            )
        return (
            f"Yes. The system supports **{capability_name}** questions. "
            f"Useful breakdowns include: {', '.join(supported_dimensions)}."
        )

    if dimension_names:
        matches = []
        for capability in CAPABILITY_REGISTRY.capabilities:
            if "demand" in q_norm and "demand" not in capability.name:
                continue
            capability_dims = {dimension.name for dimension in capability.dimensions}
            if any(name in capability_dims for name in dimension_names):
                matches.append(capability.name)
        if matches:
            return (
                f"Yes. The graph has supported questions using **{', '.join(dimension_names)}** "
                f"for these topics: {', '.join(matches)}."
            )
    return None


def _metadata_help_result(
    question: str,
    schema_path: str,
    graph_path: str,
    dr_ontology_path: str = "",
) -> Optional[Dict[str, Any]]:
    q_norm = _normalize_question_key(question)
    if not q_norm:
        return None
    schema_dict = _load_schema_dict_cached(str(schema_path or DEFAULT_SCHEMA_PATH))
    inventory = _schema_inventory_summary(schema_dict)

    asks_count = bool(re.search(r"\b(how many|number of|count|counts|size|scale)\b", q_norm))
    graph_terms = bool(re.search(r"\b(graph|kg|knowledge graph|rdf|data)\b", q_norm))
    asks_nodes = bool(re.search(r"\b(nodes?|entities|resources?)\b", q_norm))
    asks_triples = bool(re.search(r"\b(triples?|rdf triples?)\b", q_norm))
    asks_classes = bool(re.search(r"\b(classes?|concepts?)\b", q_norm))
    asks_predicates = bool(re.search(r"\b(predicates?|object properties|relationships?|relations?)\b", q_norm))
    asks_properties = bool(re.search(r"\b(properties|datatype properties|data properties|attributes?)\b", q_norm))
    asks_topics = bool(
        re.search(r"\b(what can i ask|what can we ask|available topics|topics covered|which topics|what topics|coverage|covered|capabilities|supported questions|question types|what questions)\b", q_norm)
    )
    asks_examples = bool(re.search(r"\b(examples?|sample questions?)\b", q_norm))

    source_answer = _source_scope_answer(
        question,
        schema_dict,
        _safe_graph_data_stats(graph_path),
        dr_ontology_path,
    )
    if source_answer:
        answer = source_answer
    elif asks_topics or asks_examples:
        topic_lines = "\n".join(_supported_topics_lines(schema_dict, compact=True))
        example_lines = "\n".join(f"- {item}" for item in _capability_examples_for_response())
        answer = (
            "The system is a domain-bounded True Demand KGQA assistant. It supports graph analytics, "
            "Digital Reference ontology definitions, graph metadata questions, and conservative graph-grounded advisory questions.\n\n"
            f"Supported topic areas include:\n{topic_lines}\n\n"
            f"Example questions:\n{example_lines}"
        )
    elif asks_count and (asks_nodes or asks_triples or asks_classes or asks_predicates or asks_properties or graph_terms):
        stats = _safe_graph_data_stats(graph_path) if (asks_nodes or asks_triples or graph_terms) else {}
        parts = []
        if asks_triples or graph_terms:
            value = stats.get("triples")
            parts.append(f"- RDF triples: {value:,}" if value is not None else "- RDF triples: unavailable from the active backend")
        if asks_nodes or graph_terms:
            resource_nodes = stats.get("resource_nodes")
            subject_entities = stats.get("subject_entities")
            parts.append(
                f"- Resource nodes/entities: {resource_nodes:,}"
                if resource_nodes is not None
                else "- Resource nodes/entities: unavailable from the active backend"
            )
            parts.append(
                f"- Subject entities: {subject_entities:,}"
                if subject_entities is not None
                else "- Subject entities: unavailable from the active backend"
            )
        if asks_classes or graph_terms:
            parts.append(f"- Schema classes: {inventory['classes']}")
        if asks_predicates or graph_terms:
            parts.append(f"- Object-style predicates: {inventory['predicates']}")
            parts.append(f"- Declared relationships: {inventory['relationships']}")
        if asks_properties or graph_terms:
            parts.append(f"- Datatype/data properties: {inventory['properties']}")
        answer = "Here is the current True Demand graph/schema scale:\n\n" + "\n".join(parts)
    elif asks_classes or asks_predicates or asks_properties:
        parts = []
        if asks_classes:
            classes = sorted(str(x) for x in schema_dict.get("classes") or [])
            parts.append(f"Classes ({len(classes)}): " + ", ".join(classes[:20]) + (" ..." if len(classes) > 20 else ""))
        if asks_predicates:
            predicates = sorted(str(x) for x in schema_dict.get("predicates") or [])
            parts.append(f"Predicates ({len(predicates)}): " + ", ".join(predicates[:20]) + (" ..." if len(predicates) > 20 else ""))
        if asks_properties:
            properties = sorted(str(x) for x in schema_dict.get("properties") or [])
            parts.append(f"Properties ({len(properties)}): " + ", ".join(properties[:20]) + (" ..." if len(properties) > 20 else ""))
        answer = "\n\n".join(parts)
    else:
        support_answer = _capability_support_answer(question, schema_dict)
        if support_answer:
            answer = support_answer
        else:
            return None

    return {
        "answer": answer,
        "selected_query": "",
        "candidates": [],
        "schema_ranked": [],
        "learning_ranked": [],
        "metadata": {
            "metadata_help_route": True,
            "llm_skipped": True,
        },
        "errors": [],
        "prompt": "",
        "policy": "metadata_help",
        "entropy": 0.0,
        "selection_reason": "Deterministic metadata/capability help route selected before LLM fallback.",
        "used_ml": False,
        "effective_question": question,
        "selection_explanation": {
            "selected_policy": "metadata_help",
            "selection_reason": "Answered from schema, graph metadata, or capability registry.",
            "selected_query_valid": True,
            "selected_query_errors": [],
            "selected_execution_has_rows": None,
        },
        "answerability": {
            "status": "metadata_answer_available",
            "can_answer": True,
            "reason": "The request was answered deterministically from system metadata or supported capability definitions.",
        },
        "confidence_route": {
            "enabled": True,
            "route": "auto_answer",
            "score1": 0.97,
            "score2": 0.05,
            "margin": 0.92,
            "selected_query": "",
            "reason": "deterministic metadata/capability route; LLM skipped",
            "options": [],
            "safety_flags": [],
            "blocking_safety_flags": [],
        },
        "clarification": None,
        "request_clarification": None,
    }


def _report_html(markdown_report: str) -> str:
    # Small self-contained HTML document so the user can download and print it
    # without requiring a server-side PDF dependency.
    escaped = escape(markdown_report)
    body = escaped.replace("\n", "<br>")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>True Demand KG Overview</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #1f2937; margin: 40px auto; max-width: 900px; line-height: 1.5; }}
    button {{ margin-bottom: 20px; padding: 10px 14px; }}
    .report {{ white-space: normal; }}
    @media print {{ button {{ display: none; }} body {{ margin: 0; max-width: none; }} }}
  </style>
</head>
<body>
  <button onclick="window.print()">Print / Save as PDF</button>
  <div class="report">{body}</div>
</body>
</html>"""


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --kg-bg: #f7fafb;
            --kg-panel: #ffffff;
            --kg-panel-soft: #eef5f7;
            --kg-border: #d9e4e8;
            --kg-text: #17262b;
            --kg-muted: #60747b;
            --kg-accent: #00a99d;
            --kg-accent-dark: #007f78;
            --kg-blue: #2166a5;
            --kg-green: #2e9f6e;
            --kg-green-soft: #e8f7f0;
            --kg-accent-soft: #e4f7f5;
            --kg-success: #e8f7f0;
            --kg-warning: #fff6df;
            --kg-shadow: 0 16px 42px rgba(24, 54, 64, 0.08);
            --kg-shadow-soft: 0 8px 24px rgba(24, 54, 64, 0.055);
        }

        .stApp {
            background: var(--kg-bg);
            color: var(--kg-text);
        }
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            background: var(--kg-bg) !important;
        }
        [data-testid="stHeader"] {
            border-bottom: 1px solid var(--kg-border);
            box-shadow: 0 1px 8px rgba(24, 54, 64, 0.05);
        }
        .stale-element,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {
            opacity: 1 !important;
            filter: none !important;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--kg-border);
            box-shadow: 6px 0 24px rgba(24, 54, 64, 0.04);
        }
        [data-testid="stSidebar"] * {
            color: var(--kg-text);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--kg-muted);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--kg-text);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: var(--kg-text);
            letter-spacing: 0;
        }
        h1 {
            font-weight: 700;
        }
        h2 {
            margin-top: 1.2rem;
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--kg-muted);
        }
        textarea, input {
            background: var(--kg-panel) !important;
            color: var(--kg-text) !important;
            border: 1px solid var(--kg-border) !important;
        }
        textarea:focus, input:focus {
            border-color: var(--kg-accent) !important;
            box-shadow: 0 0 0 2px rgba(0, 169, 157, 0.14) !important;
        }
        button[kind="primary"] {
            background: var(--kg-accent) !important;
            border-color: var(--kg-accent) !important;
            color: #ffffff !important;
            font-weight: 600;
            border-radius: 8px !important;
            box-shadow: 0 8px 22px rgba(0, 169, 157, 0.2);
        }
        button[kind="secondary"] {
            border-color: var(--kg-border) !important;
            color: var(--kg-text) !important;
            background: var(--kg-panel) !important;
            border-radius: 8px !important;
        }
        button[kind="secondary"]:hover {
            border-color: var(--kg-accent) !important;
            color: var(--kg-accent-dark) !important;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--kg-border);
            box-shadow: none;
        }
        [data-testid="stAlert"] div {
            color: var(--kg-text);
        }
        [data-testid="stMetric"] {
            background: var(--kg-panel);
            border: 1px solid var(--kg-border);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 22px rgba(24, 54, 64, 0.05);
        }
        [data-testid="stExpander"] {
            background: var(--kg-panel);
            border: 1px solid var(--kg-border);
            border-radius: 8px;
            box-shadow: var(--kg-shadow-soft);
            overflow: hidden;
        }
        [data-testid="stExpander"] details summary {
            background: #ffffff;
            border-radius: 8px;
            min-height: 46px;
        }
        [data-testid="stTabs"] button {
            font-weight: 600;
            color: var(--kg-muted);
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--kg-accent-dark);
        }
        [data-testid="stDataFrame"],
        pre {
            border: 1px solid var(--kg-border);
            border-radius: 8px;
            background: var(--kg-panel) !important;
        }
        iframe {
            border-radius: 8px;
        }
        .kg-hero {
            background:
                linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(235,248,246,0.96) 100%),
                radial-gradient(circle at 78% 18%, rgba(0,169,157,0.14), transparent 34%);
            border: 1px solid var(--kg-border);
            border-left: 6px solid var(--kg-accent);
            border-radius: 8px;
            box-shadow: var(--kg-shadow);
            margin-bottom: 1.35rem;
            padding: 1.45rem 1.65rem;
        }
        .kg-hero h1 {
            margin: 0.1rem 0 0.35rem;
            font-size: clamp(2.1rem, 4vw, 3.1rem);
            line-height: 1.05;
        }
        .kg-kicker {
            color: var(--kg-accent-dark);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .kg-page-copy {
            color: var(--kg-muted);
            max-width: 48rem;
            font-size: 1rem;
            line-height: 1.6;
        }
        .kg-python-badge {
            align-items: center;
            background: var(--kg-accent-soft);
            border: 1px solid #c8e9e5;
            border-radius: 999px;
            color: var(--kg-accent-dark);
            display: inline-flex;
            font-size: 0.76rem;
            font-weight: 700;
            gap: 0.35rem;
            margin-top: 0.9rem;
            padding: 0.32rem 0.7rem;
            text-transform: uppercase;
        }
        .kg-sidebar-note {
            background: var(--kg-accent-soft);
            border: 1px solid #c8e9e5;
            border-radius: 8px;
            color: var(--kg-text);
            padding: 0.75rem;
            margin: 0.5rem 0 1rem;
        }
        .kg-side-legend {
            background: var(--kg-panel);
            border: 1px solid var(--kg-border);
            border-radius: 8px;
            min-height: 240px;
            padding: 1rem;
            box-shadow: var(--kg-shadow);
        }
        .kg-side-kicker {
            color: var(--kg-blue);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .kg-side-section {
            border-top: 1px solid var(--kg-border);
            margin-top: 0.9rem;
            padding-top: 0.9rem;
        }
        .kg-side-title {
            color: var(--kg-text);
            font-weight: 600;
            margin-bottom: 0.45rem;
        }
        .kg-side-row {
            align-items: center;
            color: var(--kg-muted);
            display: flex;
            font-size: 0.82rem;
            gap: 0.55rem;
            margin: 0.45rem 0;
        }
        .kg-side-dot {
            background: var(--kg-accent-soft);
            border: 2px solid var(--kg-accent);
            border-radius: 999px;
            display: inline-block;
            height: 12px;
            width: 12px;
        }
        .kg-side-dot.muted {
            background: #e6edf1;
            border-color: #91a4ad;
        }
        .kg-side-line {
            border-top: 2px solid var(--kg-blue);
            display: inline-block;
            width: 24px;
        }
        .kg-side-copy {
            color: var(--kg-muted);
            font-size: 0.82rem;
            line-height: 1.6;
        }
        .kg-evidence-row {
            border-top: 1px solid var(--kg-border);
            display: grid;
            gap: 0.18rem;
            padding: 0.45rem 0;
        }
        .kg-evidence-row span {
            color: var(--kg-muted);
            font-size: 0.74rem;
            text-transform: capitalize;
        }
        .kg-evidence-row strong {
            color: var(--kg-text);
            font-size: 0.82rem;
            font-weight: 600;
            overflow-wrap: anywhere;
        }
        .stSuccess {
            background: var(--kg-success) !important;
        }
        .stWarning {
            background: var(--kg-warning) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_graph_overview(
    schema_path: str,
    graph_path: str,
    *,
    ontology_path: str = "",
    graph_height: int = 760,
    full_graph_limit: int = 3000,
    subgraph_hops: int = 1,
    subgraph_edge_limit: int = 1200,
) -> None:
    try:
        raw_schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Schema load failed: {exc}")
        return

    try:
        with st.spinner("Reading graph statistics..."):
            graph_stats = _graph_data_stats(graph_path)
    except Exception:
        graph_stats = {}
    report = _graph_overview_report(raw_schema, graph_stats, include_inventory=False)
    full_report = _graph_overview_report(raw_schema, graph_stats, include_inventory=True)
    html_report = _report_html(full_report)

    st.title("True Demand KG Overview")
    st.caption("A compact guide to what the graph contains and what users can ask.")
    col1, col2 = st.columns([1, 1])
    col1.download_button(
        "Download report (.md)",
        data=full_report,
        file_name="infineon_kg_overview.md",
        mime="text/markdown",
        use_container_width=True,
    )
    col2.download_button(
        "Download printable report (.html)",
        data=html_report,
        file_name="infineon_kg_overview.html",
        mime="text/html",
        use_container_width=True,
    )
    st.caption("Open the downloaded HTML report and use its `Print / Save as PDF` button for a PDF copy.")

    st.markdown(report)
    st.info(
        "This page describes the graph, schema, and ontology assets. "
        "Accuracy, routing, cost, and failure analysis are shown in the "
        "KGQA Confidence Routing Dashboard."
    )

    with st.expander("Complete schema inventory", expanded=False):
        st.markdown("#### All classes")
        st.write(sorted(raw_schema.get("classes") or []))
        st.markdown("#### All predicates")
        st.write(sorted(raw_schema.get("predicates") or []))
        st.markdown("#### All properties")
        st.write(sorted(raw_schema.get("properties") or []))
        st.markdown("#### All declared relationships")
        st.dataframe(list(raw_schema.get("relationships") or []), width="stretch")

    st.divider()
    _render_interactive_graph_explorer(
        schema_path=schema_path,
        graph_path=graph_path,
        graph_height=graph_height,
        full_graph_limit=full_graph_limit,
        subgraph_hops=subgraph_hops,
        subgraph_edge_limit=subgraph_edge_limit,
    )


def _render_interactive_graph_explorer(
    schema_path: str,
    graph_path: str,
    graph_height: int,
    full_graph_limit: int,
    subgraph_hops: int,
    subgraph_edge_limit: int,
) -> None:
    st.subheader("Interactive Graph Explorer")
    if not _graph_backend_available(graph_path):
        st.warning("Graph backend unavailable. Set a valid graph path or Fuseki query endpoint.")
        return

    tab_schema, tab_question, tab_raw = st.tabs(["Ontology Schema", "Question Subgraph", "Raw Data Triples"])

    with tab_schema:
        st.caption(
            "Ontology-level view built from declared class-to-class relationships. "
            "This is the cleaner schema graph, not a random sample of data instances."
        )
        try:
            raw_schema = _load_schema_dict_cached(schema_path)
            triples = _overview_relationship_triples(raw_schema)
        except Exception as exc:
            triples = []
            st.warning(f"Could not load schema relationships: {exc}")
        if not triples:
            st.warning("No declared ontology relationships available for visualization.")
        else:
            graph_nodes = {node for s, _p, o in triples for node in (s, o)}
            st.caption(f"Showing {len(graph_nodes)} ontology nodes and {len(triples)} declared relationships.")
            html = build_graph_html(
                triples,
                height_px=int(graph_height),
                heading="True Demand Ontology Schema",
                max_nodes=90,
                max_edges=120,
            )
            components.html(
                html,
                height=int(graph_height) + 40,
                scrolling=True,
            )

    with tab_question:
        st.caption("Visualize the graph area related to the last selected query.")
        last_query = str(st.session_state.get("last_selected_query", "") or "").strip()
        if not last_query:
            st.info("Ask a question first to create a selected query.")
        else:
            st.code(last_query, language="sparql")
            if st.button("Visualize Question Subgraph", key="load_question_subgraph_btn"):
                with st.spinner("Building question-focused subgraph..."):
                    graph = _load_active_graph(graph_path)
                    rows = st.session_state.get("last_graph_rows") or []
                    triples, meta = collect_query_subgraph_triples(
                        graph=graph,
                        query=last_query,
                        result_rows=rows,
                        hops=int(subgraph_hops),
                        limit=int(subgraph_edge_limit),
                    )
                if not triples:
                    st.warning("Could not extract a non-empty subgraph for this query.")
                else:
                    html = build_graph_html(
                        triples,
                        height_px=int(graph_height),
                        heading="True Demand KG (Question Subgraph)",
                    )
                    st.caption(
                        f"Seeds: {meta.get('seed_count', 0)} | "
                        f"Edges shown: {meta.get('edge_count', 0)}"
                    )
                    components.html(
                        html,
                        height=int(graph_height) + 40,
                        scrolling=True,
                    )

    with tab_raw:
        st.caption(
            "Debug view over raw RDF data triples. This can look noisy because it includes "
            "instances, observations, sample values, and literals. It is not the ontology map."
        )
        if full_graph_limit == 0:
            st.warning("Raw full graph without limit may be very heavy in browser.")
        if st.button("Load Raw Data Triples", key="load_raw_graph_btn"):
            with st.spinner("Loading raw data triples and building visualization..."):
                graph = _load_active_graph(graph_path)
                triples, total = collect_full_graph_triples(graph, limit=int(full_graph_limit))
                if not triples:
                    st.warning("No triples available for visualization.")
                else:
                    html = build_graph_html(
                        triples,
                        height_px=int(graph_height),
                        heading="True Demand Raw Data Triples",
                    )
                    st.caption(f"Showing {len(triples)} raw triples out of total {total}.")
                    components.html(
                        html,
                        height=int(graph_height) + 40,
                        scrolling=True,
                    )


def _default_confidence_routing_report_path() -> str:
    for path in DEFAULT_CONFIDENCE_ROUTING_REPORT_PATHS:
        if path.exists():
            return str(path)
    return str(DEFAULT_CONFIDENCE_ROUTING_REPORT_PATHS[0])


def _load_confidence_routing_report(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Routing report must be a JSON object.")
    return payload


def _dashboard_policy_bucket(report: Dict[str, object], name: str) -> Dict[str, object]:
    return dict(dict(report.get("policy_buckets") or {}).get(name) or {})


def _dashboard_fmt_pct(value: object) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _dashboard_score(row: Dict[str, object], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _dashboard_distribution_rows(values: object) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in values or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            rows.append({"value": item[0], "count": item[1]})
    return rows


def _final_eval_pct(key: str) -> str:
    return f"{100 * float(FINAL_SYSTEM_EVALUATION[key]):.1f}%"


def _render_final_system_evaluation_dashboard() -> None:
    eval_data = FINAL_SYSTEM_EVALUATION
    st.subheader("Final System Evaluation")
    st.caption(
        "Audited answer-level results from the strict deterministic v2 full-system benchmark. "
        "These are user-facing final-answer metrics, not only raw candidate-selection metrics."
    )

    cols = st.columns(4)
    cols[0].metric(
        "Overall audited accuracy",
        _final_eval_pct("overall_accuracy"),
        f"{eval_data['correct_answers']}/{eval_data['benchmark_questions']} correct",
    )
    cols[1].metric(
        "Deterministic route",
        _final_eval_pct("deterministic_accuracy"),
        f"{eval_data['deterministic_questions']} questions",
    )
    cols[2].metric(
        "LLM fallback",
        _final_eval_pct("llm_fallback_accuracy"),
        f"{eval_data['llm_fallback_questions']} questions",
    )
    cols[3].metric(
        "LLM-call reduction",
        _final_eval_pct("llm_call_reduction"),
        f"EUR {eval_data['estimated_savings_eur']:.2f} saved est.",
    )

    st.markdown("##### Benchmark composition and route performance")
    st.dataframe(
        [
            {
                "Layer": "KG analytics",
                "Questions": eval_data["kg_questions"],
                "Correct": int(round(eval_data["kg_questions"] * eval_data["kg_accuracy"])),
                "Accuracy": _final_eval_pct("kg_accuracy"),
                "Role": "True Demand analytical questions over RDF graph data",
            },
            {
                "Layer": "DR ontology definitions",
                "Questions": eval_data["ontology_definition_questions"],
                "Correct": int(round(eval_data["ontology_definition_questions"] * eval_data["ontology_accuracy"])),
                "Accuracy": _final_eval_pct("ontology_accuracy"),
                "Role": "Deterministic concept/property definitions and ontology-model lookup",
            },
            {
                "Layer": "Advisory questions",
                "Questions": eval_data["advisory_questions"],
                "Correct": int(round(eval_data["advisory_questions"] * eval_data["advisory_accuracy"])),
                "Accuracy": _final_eval_pct("advisory_accuracy"),
                "Role": "Conservative graph-grounded planning guidance from query outputs",
            },
        ],
        width="stretch",
        hide_index=True,
    )
    st.dataframe(
        [
            {
                "Route": "Deterministic / auto-answer",
                "Questions": eval_data["deterministic_questions"],
                "Correct": eval_data["deterministic_correct"],
                "Incorrect": eval_data["deterministic_incorrect"],
                "Accuracy": _final_eval_pct("deterministic_accuracy"),
                "When used": "Known graph/ontology/advisory path with evidence",
            },
            {
                "Route": "LLM fallback + ranking",
                "Questions": eval_data["llm_fallback_questions"],
                "Correct": eval_data["llm_fallback_correct"],
                "Incorrect": eval_data["llm_fallback_incorrect"],
                "Accuracy": _final_eval_pct("llm_fallback_accuracy"),
                "When used": "Unsupported, unsafe, or genuinely ambiguous deterministic route",
            },
        ],
        width="stretch",
        hide_index=True,
    )


def _render_final_cost_and_failure_dashboard() -> None:
    eval_data = FINAL_SYSTEM_EVALUATION
    st.subheader("Cost, Failure Families, and Human Difficulty")
    cost_cols = st.columns(4)
    cost_cols[0].metric("Cold LLM calls", str(eval_data["llm_calls"]), f"baseline {eval_data['benchmark_questions']}")
    cost_cols[1].metric("Cold estimate", f"EUR {eval_data['estimated_cost_eur']:.2f}")
    cost_cols[2].metric("All-LLM baseline", f"EUR {eval_data['all_llm_baseline_cost_eur']:.2f}")
    cost_cols[3].metric("Estimated saving", f"EUR {eval_data['estimated_savings_eur']:.2f}")
    st.caption(
        "The final benchmark was also run with cache reuse. The cost claim shown here is the cold-run estimate: "
        f"{eval_data['llm_calls']} fallback LLM calls at EUR 0.20 each instead of sending all "
        f"{eval_data['benchmark_questions']} questions to the LLM."
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown("##### Remaining incorrect answers by human SPARQL difficulty")
        st.dataframe(
            [
                {"Difficulty": "Easy", "Incorrect": eval_data["failure_easy"], "Meaning": "A schema-aware human would likely write the correct query quickly."},
                {"Difficulty": "Medium", "Incorrect": eval_data["failure_medium"], "Meaning": "Requires schema awareness and careful metric/scope mapping."},
                {"Difficulty": "Hard", "Incorrect": eval_data["failure_hard"], "Meaning": "Requires graph exploration, indirect joins, or iterative SPARQL testing."},
            ],
            width="stretch",
            hide_index=True,
        )
    with cols[1]:
        st.markdown("##### Main remaining failure families")
        st.dataframe(
            [
                {"Family": "Autonomous-driving complex grouping", "Incorrect": eval_data["failure_autonomous_driving"]},
                {"Family": "Current-demand baseline / scope", "Incorrect": eval_data["failure_current_demand"]},
                {"Family": "Vehicle-sales metric / dimension", "Incorrect": eval_data["failure_vehicle_sales"]},
                {"Family": "Future-demand complex dimensions", "Incorrect": eval_data["failure_future_demand"]},
            ],
            width="stretch",
            hide_index=True,
        )


def _render_user_testing_log_panel() -> None:
    st.subheader("User Testing Logs")
    st.caption(
        "Each submitted question is saved for later review. The JSONL file is easy to inspect; "
        "the SQLite database is better for querying/filtering repeated internal testing sessions."
    )
    cols = st.columns(2)
    jsonl_exists = USER_AUDIT_LOG_PATH.exists()
    db_exists = USER_AUDIT_DB_PATH.exists()
    cols[0].metric("JSONL audit log", "available" if jsonl_exists else "not created yet")
    cols[1].metric("SQLite audit DB", "available" if db_exists else "not created yet")

    dl_cols = st.columns(2)
    if jsonl_exists:
        dl_cols[0].download_button(
            "Download user audit JSONL",
            data=USER_AUDIT_LOG_PATH.read_text(encoding="utf-8", errors="ignore"),
            file_name=USER_AUDIT_LOG_PATH.name,
            mime="application/jsonl",
            use_container_width=True,
        )
    if db_exists:
        dl_cols[1].download_button(
            "Download user audit SQLite",
            data=USER_AUDIT_DB_PATH.read_bytes(),
            file_name=USER_AUDIT_DB_PATH.name,
            mime="application/vnd.sqlite3",
            use_container_width=True,
        )

    if jsonl_exists:
        try:
            rows = []
            with USER_AUDIT_LOG_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            recent = rows[-10:]
            if recent:
                st.caption(f"Recent questions shown: {len(recent)} / {len(rows)} logged")
                st.dataframe(
                    [
                        {
                            "time": row.get("timestamp_utc"),
                            "question": row.get("question"),
                            "route": row.get("route_family"),
                            "confidence": _confidence_percent(float(row.get("confidence_index") or 0.0)),
                            "rows": row.get("graph_row_count"),
                            "llm_calls": row.get("llm_estimated_calls"),
                        }
                        for row in recent
                    ],
                    width="stretch",
                    hide_index=True,
                )
        except Exception as exc:
            st.caption(f"Could not preview user audit log: {exc}")


def _dashboard_cases(report: Dict[str, object], case_set: str) -> List[Dict[str, object]]:
    if case_set == "low_confidence_examples":
        return [dict(row) for row in report.get("low_confidence_examples") or []]
    if case_set == "high_confidence_wrong_examples":
        return [dict(row) for row in report.get("high_confidence_wrong_examples") or []]
    return []


def _dashboard_all_flags(cases: List[Dict[str, object]]) -> List[str]:
    flags = set()
    for row in cases:
        for flag in row.get("safety_flags") or []:
            flags.add(str(flag))
    return sorted(flags)


def _dashboard_top3_table(row: Dict[str, object]) -> List[Dict[str, object]]:
    table: List[Dict[str, object]] = []
    for candidate in row.get("top3") or []:
        if not isinstance(candidate, dict):
            continue
        table.append(
            {
                "rank": candidate.get("rank"),
                "score": round(float(candidate.get("score") or 0.0), 4),
                "label": candidate.get("label"),
                "source": candidate.get("source"),
                "query": candidate.get("query"),
            }
        )
    return table


def _render_confidence_dashboard_summary(report: Dict[str, object]) -> None:
    summary = dict(report.get("summary") or {})
    inputs = dict(report.get("inputs") or {})
    auto = _dashboard_policy_bucket(report, "auto_answer")
    clarification = _dashboard_policy_bucket(report, "clarification")

    st.subheader("Candidate Selection / Confidence Report")
    st.caption(
        "This section comes from a confidence-routing JSON report. "
        "It describes candidate selection and confidence buckets, not the final mixed-system audit."
    )
    cols = st.columns(5)
    cols[0].metric(
        "Forced Top1",
        f"{summary.get('forced_top1_correct', 0)}/{summary.get('total', 0)}",
        _dashboard_fmt_pct(summary.get("forced_top1_accuracy")),
    )
    cols[1].metric(
        "Any Correct",
        f"{summary.get('any_correct', 0)}/{summary.get('total', 0)}",
        _dashboard_fmt_pct(summary.get("any_correct_rate")),
    )
    cols[2].metric(
        "Auto-answer",
        f"{auto.get('correct', 0)}/{auto.get('count', 0)}",
        _dashboard_fmt_pct(auto.get("accuracy")),
    )
    total = max(1, int(summary.get("total") or 1))
    cols[3].metric(
        "Auto coverage",
        str(auto.get("count", 0)),
        _dashboard_fmt_pct((auto.get("count", 0) or 0) / total),
    )
    cols[4].metric(
        "Clarification Any",
        f"{clarification.get('any_correct', 0)}/{clarification.get('count', 0)}",
        _dashboard_fmt_pct(clarification.get("any_correct_rate")),
    )

    st.caption(
        "Policy: "
        f"score >= {float(inputs.get('policy_min_score') or 0.0):.2f}, "
        f"margin >= {float(inputs.get('policy_min_margin') or 0.0):.2f}, "
        f"safety guard = {inputs.get('enable_safety_guard')}"
    )


def _render_confidence_dashboard_distributions(report: Dict[str, object]) -> None:
    st.subheader("Bucket Composition")
    bucket_name = st.radio(
        "Bucket",
        ["auto_answer", "clarification"],
        horizontal=True,
        key="confidence_dashboard_bucket",
    )
    bucket = _dashboard_policy_bucket(report, str(bucket_name))
    dimensions = [
        ("families", "Families"),
        ("aggregation", "Aggregation"),
        ("scopes", "Scopes"),
        ("dimensions", "Dimensions"),
        ("answer_shape", "Answer Shape"),
        ("safety_flags", "Safety Flags"),
    ]
    cols = st.columns(2)
    for idx, (key, label) in enumerate(dimensions):
        with cols[idx % 2]:
            st.markdown(f"**{label}**")
            st.dataframe(_dashboard_distribution_rows(bucket.get(key)), width="stretch", hide_index=True)


def _render_confidence_dashboard_cases(report: Dict[str, object]) -> None:
    st.subheader("Case Browser")
    case_set = st.selectbox(
        "Case set",
        ["high_confidence_wrong_examples", "low_confidence_examples"],
        index=0,
    )
    cases = _dashboard_cases(report, case_set)
    if not cases:
        st.info("No cases available in this report.")
        return

    family_options = sorted({str(row.get("family") or "unknown") for row in cases})
    selected_families = st.multiselect("Families", family_options, default=family_options)
    only_wrong = st.checkbox("Only top1 wrong", value=(case_set == "high_confidence_wrong_examples"))
    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
    selected_flags = st.multiselect("Safety flags", _dashboard_all_flags(cases))

    filtered: List[Dict[str, object]] = []
    for row in cases:
        if str(row.get("family") or "unknown") not in selected_families:
            continue
        if only_wrong and row.get("top1_correct"):
            continue
        if _dashboard_score(row, "score1") < min_score:
            continue
        row_flags = {str(flag) for flag in row.get("safety_flags") or []}
        if selected_flags and not row_flags.intersection(selected_flags):
            continue
        filtered.append(row)

    st.caption(f"Showing {len(filtered)} / {len(cases)} cases")
    for row in filtered:
        title = (
            f"{row.get('id')} | "
            f"score={_dashboard_score(row, 'score1'):.3f} | "
            f"margin={_dashboard_score(row, 'margin'):.3f}"
        )
        with st.expander(title, expanded=False):
            st.markdown(f"**Question:** {row.get('question')}")
            cols = st.columns(4)
            cols[0].metric("Score1", f"{_dashboard_score(row, 'score1'):.4f}")
            cols[1].metric("Score2", f"{_dashboard_score(row, 'score2'):.4f}")
            cols[2].metric("Margin", f"{_dashboard_score(row, 'margin'):.4f}")
            cols[3].metric("Top1 correct", str(bool(row.get("top1_correct"))))
            st.markdown(f"**Safety flags:** `{row.get('safety_flags') or []}`")
            st.markdown("**Top interpretations**")
            st.dataframe(_dashboard_top3_table(row), width="stretch", hide_index=True)


def _render_confidence_routing_dashboard() -> None:
    st.title("KGQA Confidence Routing Dashboard")
    st.caption(
        "Evaluation-focused page: final audited metrics, deterministic-vs-LLM routing, "
        "cost estimates, error families, and optional confidence-routing case analysis."
    )

    _render_final_system_evaluation_dashboard()
    st.divider()
    _render_final_cost_and_failure_dashboard()
    st.divider()
    _render_user_testing_log_panel()
    st.divider()

    st.subheader("Optional Confidence Routing Report")
    report_path = st.text_input(
        "Routing report JSON",
        value=_default_confidence_routing_report_path(),
        help=(
            "Optional diagnostic JSON for score/margin buckets and high-confidence/low-confidence cases. "
            "Final system metrics above do not depend on this file."
        ),
    )
    if not report_path.strip():
        st.info("Enter a confidence routing JSON report path to inspect candidate-selection buckets.")
        return
    try:
        report = _load_confidence_routing_report(report_path.strip())
    except Exception as exc:
        st.warning(f"Could not load optional confidence report: {exc}")
        return

    _render_confidence_dashboard_summary(report)
    st.divider()
    _render_confidence_dashboard_distributions(report)
    st.divider()
    _render_confidence_dashboard_cases(report)


st.set_page_config(page_title="True Demand KG QA", layout="wide")
_inject_app_styles()

with st.sidebar:
    page = st.radio("Page", ["Ask", "Graph Overview", "Confidence Routing Dashboard"])
    st.markdown(
        '<div class="kg-sidebar-note">Ask questions over the True Demand KG or open the overview report.</div>',
        unsafe_allow_html=True,
    )
    developer_mode_enabled = (
        os.getenv("TRUE_DEMAND_ENABLE_DEVELOPER_MODE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    developer_mode = (
        st.checkbox("Developer mode", value=False)
        if developer_mode_enabled
        else False
    )

    llm_backend = (
        os.environ.get("LLM_BACKEND")
        or os.environ.get("KGQA_LLM_BACKEND")
        or "infineon"
    ).strip().lower()
    if llm_backend in {"litellm", "lite_llm"}:
        default_url = (
            os.environ.get("LITELLM_BASE_URL")
            or os.environ.get("LITE_LLM_BASE_URL")
            or os.environ.get("BASE_URL")
            or os.environ.get("INFINEON_API_URL", "")
        )
        default_model = (
            os.environ.get("LITELLM_MODEL")
            or os.environ.get("LITE_LLM_MODEL")
            or os.environ.get("INFINEON_MODEL", "gpt-4o")
        )
        default_endpoint = (
            os.environ.get("LITELLM_CHAT_ENDPOINT")
            or os.environ.get("LITE_LLM_CHAT_ENDPOINT")
            or os.environ.get("INFINEON_CHAT_ENDPOINT", "/chat/completions")
        )
        default_key = (
            os.environ.get("LITELLM_API_KEY")
            or os.environ.get("LITE_LLM_TOKEN")
            or os.environ.get("LITE_LLM_API_KEY")
            or os.environ.get("INFINEON_API_KEY", "")
        )
    else:
        default_url = os.environ.get("INFINEON_API_URL", "https://gpt4ifx.icp.infineon.com")
        default_model = os.environ.get("INFINEON_MODEL", "gpt-4o")
        default_endpoint = os.environ.get("INFINEON_CHAT_ENDPOINT", "/chat/completions")
        default_key = os.environ.get("INFINEON_API_KEY", "")

    api_url = default_url
    api_endpoint = default_endpoint
    model_name = default_model
    api_key = default_key
    temperature = 0.2
    schema_path = str(DEFAULT_SCHEMA_PATH)
    graph_path = str(DEFAULT_GRAPH_PATH)
    ontology_path = os.getenv("TRUE_DEMAND_ONTOLOGY_PATH", str(DEFAULT_ONTOLOGY_PATH)).strip()
    dr_ontology_path = os.getenv("TRUE_DEMAND_DR_ONTOLOGY_PATH", str(DEFAULT_DR_ONTOLOGY_PATH)).strip()
    fuseki_query_url = os.getenv("FUSEKI_QUERY_URL", DEFAULT_FUSEKI_QUERY_URL).strip()
    use_ml_ranking = True
    ml_policy = "all"
    ml_model_path = _default_ml_model_path()
    ambiguity_config_path = _default_ambiguity_config_path()
    family_schema_routing_enabled = (
        os.getenv("INFINEON_ENABLE_SCHEMA_SLICING", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    try:
        family_schema_routing_max = int(os.getenv("INFINEON_SCHEMA_SLICING_MAX_FAMILIES", "3") or 3)
    except Exception:
        family_schema_routing_max = 3
    family_schema_routing_fallback = (
        os.getenv("INFINEON_SCHEMA_SLICING_FULL_FALLBACK", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    confidence_routing_enabled = True
    confidence_min_score = 0.95
    confidence_min_margin = 0.00
    confidence_safety_guard = True
    confidence_sort_by_score = True
    fast_interactive_mode = True
    cost_aware_direct_answers = True
    llm_candidate_cache_enabled = True
    show_prompt = False
    show_candidates = False
    show_candidate_diagnostics = False
    show_answer_evidence_graph = True
    execute_selected = True
    max_preview_rows = 200
    full_graph_limit = 3000
    subgraph_edge_limit = 1200
    subgraph_hops = 1
    graph_height = 760

    if developer_mode:
        with st.expander("Developer settings", expanded=True):
            st.markdown("##### Recommended demo settings")
            _render_recommended_settings()
            st.divider()

            st.subheader("Backend")
            llm_backend = st.selectbox(
                "LLM backend",
                options=["infineon", "litellm"],
                index=1 if llm_backend in {"litellm", "lite_llm"} else 0,
                help="Use litellm for the OpenAI-compatible LiteLLM gateway.",
            )
            api_url = st.text_input("LLM base URL", value=default_url)
            api_endpoint = st.text_input("LLM chat endpoint", value=default_endpoint)
            model_name = st.text_input("LLM model", value=default_model)
            api_key = st.text_input("LLM API key", value=default_key, type="password")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

            st.subheader("Data")
            schema_path = st.text_input("Schema path", value=str(DEFAULT_SCHEMA_PATH))
            graph_path = st.text_input("Graph path", value=str(DEFAULT_GRAPH_PATH))
            ontology_path = st.text_input(
                "Ontology path",
                value=ontology_path or str(DEFAULT_ONTOLOGY_PATH),
                help="Ontology/schema file used by the built-in graph overview.",
            )
            dr_ontology_path = st.text_input(
                "DR ontology path",
                value=dr_ontology_path or str(DEFAULT_DR_ONTOLOGY_PATH),
                help="Digital Reference ontology used for deterministic definition questions such as 'What is True Demand?'.",
            )
            fuseki_query_url = st.text_input(
                "Fuseki query endpoint",
                value=fuseki_query_url,
                placeholder=DEFAULT_FUSEKI_QUERY_URL,
                help=(
                    "Default graph backend. Keep Fuseki running at this endpoint for the "
                    "fastest interactive execution path. Clear only if you intentionally "
                    "want to fall back to local graph.ttl with RDFLib."
                ),
            )

            st.subheader("Ranking")
            use_ml_ranking = st.checkbox("Use ML ranking", value=True)
            ml_policy_options = ["auto", "all", "mid", "off"]
            ml_policy = st.selectbox(
                "ML policy",
                options=ml_policy_options,
                index=ml_policy_options.index(ml_policy) if ml_policy in ml_policy_options else 1,
                help=(
                    "all matches the final evaluated setup: direct graph-supported templates first, "
                    "then ML reranking for LLM-needed questions. auto/mid are experimental "
                    "ambiguity-routing modes."
                ),
            )
            ml_model_path = st.text_input("ML model path", value=_default_ml_model_path())
            ambiguity_config_path = st.text_input(
                "Ambiguity config path",
                value=_default_ambiguity_config_path(),
                help="Used when ML policy is auto/mid for runtime ambiguity regime prediction.",
            )
            if not use_ml_ranking:
                ml_policy = "off"

            st.subheader("Schema Routing")
            family_schema_routing_enabled = st.checkbox(
                "Family-aware schema routing",
                value=family_schema_routing_enabled,
                help=(
                    "Routes the LLM prompt to one or more ontology families. "
                    "SPARQL still executes against the full graph/Fuseki dataset."
                ),
            )
            family_schema_routing_max = st.number_input(
                "Max routed families",
                min_value=1,
                max_value=4,
                value=max(1, min(4, int(family_schema_routing_max))),
                step=1,
            )
            family_schema_routing_fallback = st.checkbox(
                "Retry full schema after sliced prompt",
                value=family_schema_routing_fallback,
                help=(
                    "More robust but slower because it can add a second LLM call. "
                    "Keep off for fast interactive testing."
                ),
            )

            st.subheader("Confidence Routing")
            confidence_routing_enabled = st.checkbox("Ask for clarification when confidence is low", value=True)
            confidence_min_score = st.slider(
                "Auto-answer minimum score",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
            )
            confidence_min_margin = st.slider(
                "Auto-answer minimum margin",
                min_value=-0.25,
                max_value=1.0,
                value=0.00,
                step=0.01,
            )
            confidence_safety_guard = st.checkbox("Enable safety guard", value=True)
            confidence_sort_by_score = st.checkbox(
                "Route by ML score order",
                value=True,
                help="Uses the ranker score order for confidence routing and top-3 clarification options.",
            )
            fast_interactive_mode = st.checkbox(
                "Fast interactive mode",
                value=True,
                help=(
                    "Skips expensive legacy clarification/answerability profiling inside the pipeline. "
                    "The UI still executes the selected or clarified query before showing an answer."
                ),
            )
            cost_aware_direct_answers = st.checkbox(
                "Skip LLM for single supported capability",
                value=True,
                help=(
                    "If the graph capability inventory resolves exactly one supported interpretation "
                    "and it returns rows, execute it directly before calling the LLM."
                ),
            )
            llm_candidate_cache_enabled = st.checkbox(
                "Cache identical LLM candidate prompts",
                value=True,
                help=(
                    "Reuses candidate-generation results only when the prompt/model/settings hash "
                    "matches exactly. This reduces repeated demo cost without changing ranking."
                ),
            )

            st.subheader("Diagnostics")
            show_prompt = st.checkbox("Show candidate prompt", value=False)
            show_candidates = st.checkbox("Show candidates", value=False)
            show_candidate_diagnostics = st.checkbox(
                "Run candidate diagnostics",
                value=False,
                help=(
                    "Runs graph execution checks for every candidate. Useful for debugging, "
                    "but slower on large graph slices."
                ),
            )
            show_answer_evidence_graph = st.checkbox(
                "Show answer evidence graph",
                value=True,
                help=(
                    "Builds a small graph visualization after the answer. Useful for demos, "
                    "Turn off only when measuring the fastest possible response time."
                ),
            )
            execute_selected = st.checkbox("Execute selected query on graph", value=True)
            max_preview_rows = st.number_input(
                "Max preview rows",
                min_value=10,
                max_value=1000,
                value=200,
                step=10,
            )

            st.subheader("Graph Explorer")
            full_graph_limit = st.number_input(
                "Full graph triple limit (0 = all)",
                min_value=0,
                max_value=100000,
                value=3000,
                step=500,
            )
            subgraph_edge_limit = st.number_input(
                "Question subgraph edge limit",
                min_value=50,
                max_value=20000,
                value=1200,
                step=50,
            )
            subgraph_hops = st.slider("Question subgraph hops", min_value=1, max_value=3, value=1, step=1)
            graph_height = st.slider("Graph canvas height (px)", min_value=400, max_value=1200, value=760, step=20)

    usable_fuseki_query_url = _usable_fuseki_query_url(fuseki_query_url)
    if usable_fuseki_query_url:
        os.environ["FUSEKI_QUERY_URL"] = usable_fuseki_query_url
    else:
        os.environ.pop("FUSEKI_QUERY_URL", None)
    if ontology_path.strip():
        os.environ["TRUE_DEMAND_ONTOLOGY_PATH"] = ontology_path.strip()
    if dr_ontology_path.strip():
        os.environ["TRUE_DEMAND_DR_ONTOLOGY_PATH"] = dr_ontology_path.strip()
    os.environ["INFINEON_ENABLE_SCHEMA_SLICING"] = "1" if family_schema_routing_enabled else "0"
    os.environ["INFINEON_SCHEMA_SLICING_MAX_FAMILIES"] = str(int(family_schema_routing_max))
    os.environ["INFINEON_SCHEMA_SLICING_FULL_FALLBACK"] = "1" if family_schema_routing_fallback else "0"
    os.environ["INFINEON_ENABLE_LLM_CACHE"] = "1" if llm_candidate_cache_enabled else "0"

    with st.expander("System status", expanded=False):
        _render_system_status(
            schema_path=schema_path,
            graph_path=graph_path,
            fuseki_query_url=_active_fuseki_query_url(),
            model_path=ml_model_path,
            api_url=api_url,
            api_key=api_key,
        )

if page == "Graph Overview":
    _render_graph_overview(
        schema_path,
        graph_path,
        ontology_path=ontology_path,
        graph_height=int(graph_height),
        full_graph_limit=int(full_graph_limit),
        subgraph_hops=int(subgraph_hops),
        subgraph_edge_limit=int(subgraph_edge_limit),
    )
    st.stop()

if page == "Confidence Routing Dashboard":
    _render_confidence_routing_dashboard()
    st.stop()

st.markdown(
    """
    <section class="kg-hero">
      <div class="kg-kicker">Knowledge graph assistant</div>
      <h1>True Demand KG QA</h1>
      <div class="kg-page-copy">Ask a natural-language question and receive a graph-grounded answer. When the request is genuinely ambiguous, the system asks for the intended interpretation first.</div>
      <div class="kg-python-badge">Python-only Streamlit UI · no Node.js required</div>
    </section>
    """,
    unsafe_allow_html=True,
)

question = _render_kg_autocomplete_input(schema_path, graph_path, _active_fuseki_query_url())
asked = st.button("Ask", type="primary")

_render_question_guidance(graph_path, _active_fuseki_query_url())

if "last_qa_result" not in st.session_state:
    st.session_state["last_qa_result"] = None
if "last_graph_rows" not in st.session_state:
    st.session_state["last_graph_rows"] = []
if "last_selected_query" not in st.session_state:
    st.session_state["last_selected_query"] = ""
if "last_graph_answer" not in st.session_state:
    st.session_state["last_graph_answer"] = ""
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""
if "clarification_choice_id" not in st.session_state:
    st.session_state["clarification_choice_id"] = None
if "confidence_clarification_choice_id" not in st.session_state:
    st.session_state["confidence_clarification_choice_id"] = None

clarification_rendered = False

if asked:
    guided_query = _active_guided_query(question)
    dr_definition = route_dr_ontology_definition(question)
    advisory_plan = resolve_advisory_plan(question)
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        metadata_help = _metadata_help_result(question, schema_path, graph_path, dr_ontology_path)
        if metadata_help is not None and not guided_query and dr_definition is None and advisory_plan is None:
            request_id = uuid.uuid4().hex
            st.session_state["last_qa_result"] = metadata_help
            st.session_state["last_graph_rows"] = []
            st.session_state["last_selected_query"] = ""
            st.session_state["last_graph_answer"] = str(metadata_help.get("answer") or "")
            st.session_state["last_question"] = question
            st.session_state["last_request_id"] = request_id
            st.session_state["last_latency_s"] = 0.0
            st.session_state["last_latency_breakdown"] = {}
            try:
                _write_user_audit_record(
                    _user_audit_payload(
                        request_id=request_id,
                        question=question,
                        result=metadata_help,
                        selected_query="",
                        graph_rows=[],
                        graph_exec_error="",
                        graph_answer=str(metadata_help.get("answer") or ""),
                        latency_s=0.0,
                    )
                )
            except Exception:
                pass
            route = metadata_help.get("confidence_route")
            if isinstance(route, dict):
                _render_confidence_route_badge(route)
            _render_answer_block(
                answer_text=str(metadata_help.get("answer") or ""),
                selected_query="",
                graph_rows=[],
                graph_exec_error="",
                execute_selected=False,
                answerability=metadata_help.get("answerability"),
            )
            _render_compact_explainability(metadata_help)
            st.stop()

        out_of_scope, out_of_scope_reason = _is_out_of_scope_question(question)
        if (
            out_of_scope
            and not guided_query
            and dr_definition is None
            and advisory_plan is None
        ):
            request_id = uuid.uuid4().hex
            result = {
                "answer": "Out-of-scope request.",
                "selected_query": "",
                "candidates": [],
                "metadata": {"llm_skipped": True, "out_of_scope": True},
                "policy": "out_of_scope_guard",
                "selection_reason": out_of_scope_reason,
                "confidence_route": {
                    "enabled": True,
                    "route": "controlled_no_answer",
                    "score1": 0.98,
                    "score2": 0.0,
                    "margin": 0.98,
                    "selected_query": "",
                    "reason": out_of_scope_reason,
                    "options": [],
                    "safety_flags": ["out_of_scope"],
                    "blocking_safety_flags": ["out_of_scope"],
                },
                "answerability": {
                    "status": "out_of_scope",
                    "can_answer": False,
                    "reason": out_of_scope_reason,
                },
            }
            st.session_state["last_qa_result"] = result
            st.session_state["last_graph_rows"] = []
            st.session_state["last_selected_query"] = ""
            st.session_state["last_graph_answer"] = ""
            st.session_state["last_question"] = question
            st.session_state["last_request_id"] = request_id
            try:
                _write_user_audit_record(
                    _user_audit_payload(
                        request_id=request_id,
                        question=question,
                        result=result,
                        selected_query="",
                        graph_rows=[],
                        graph_exec_error="",
                        graph_answer="Out-of-scope request.",
                        latency_s=0.0,
                    )
                )
            except Exception:
                pass
            _render_out_of_scope_message(out_of_scope_reason)
            st.stop()

        unsupported_time, unsupported_time_reason = _is_unsupported_relative_time_question(question)
        if (
            unsupported_time
            and not guided_query
            and dr_definition is None
            and advisory_plan is None
        ):
            approximate_question, approximate_query, approximate_note = _relative_time_approximation_query(question)
            if approximate_query:
                st.info(approximate_note)
                question = approximate_question
                guided_query = approximate_query
            else:
                request_id = uuid.uuid4().hex
                result = {
                    "answer": "Unsupported relative-time request.",
                    "selected_query": "",
                    "candidates": [],
                    "metadata": {"llm_skipped": True, "unsupported_relative_time": True},
                    "policy": "unsupported_relative_time_guard",
                    "selection_reason": unsupported_time_reason,
                    "confidence_route": {
                        "enabled": True,
                        "route": "controlled_no_answer",
                        "score1": 0.96,
                        "score2": 0.0,
                        "margin": 0.96,
                        "selected_query": "",
                        "reason": unsupported_time_reason,
                        "options": [],
                        "safety_flags": ["unsupported_relative_time"],
                        "blocking_safety_flags": ["unsupported_relative_time"],
                    },
                    "answerability": {
                        "status": "unsupported_relative_time",
                        "can_answer": False,
                        "reason": unsupported_time_reason,
                    },
                }
                st.session_state["last_qa_result"] = result
                st.session_state["last_graph_rows"] = []
                st.session_state["last_selected_query"] = ""
                st.session_state["last_graph_answer"] = "Unsupported relative-time request."
                st.session_state["last_question"] = question
                st.session_state["last_request_id"] = request_id
                try:
                    _write_user_audit_record(
                        _user_audit_payload(
                            request_id=request_id,
                            question=question,
                            result=result,
                            selected_query="",
                            graph_rows=[],
                            graph_exec_error="",
                            graph_answer="Unsupported relative-time request.",
                            latency_s=0.0,
                        )
                    )
                except Exception:
                    pass
                _render_unsupported_time_message(unsupported_time_reason)
                st.stop()

        if not guided_query and dr_definition is None and advisory_plan is None:
            quarter_question, quarter_query, quarter_note = _semiconductor_demand_quarter_guided_query(question)
            if quarter_query:
                st.info(quarter_note)
                question = quarter_question
                guided_query = quarter_query

        if (
            _is_advisory_like_question(question)
            and not guided_query
            and dr_definition is None
            and advisory_plan is None
        ):
            request_id = uuid.uuid4().hex
            result = _advisory_request_clarification_result(question)
            st.session_state["last_qa_result"] = result
            st.session_state["last_graph_rows"] = []
            st.session_state["last_selected_query"] = ""
            st.session_state["last_graph_answer"] = ""
            st.session_state["last_question"] = question
            st.session_state["last_request_id"] = request_id
            st.session_state["last_latency_s"] = 0.0
            st.session_state["last_latency_breakdown"] = {}
            try:
                _write_user_audit_record(
                    _user_audit_payload(
                        request_id=request_id,
                        question=question,
                        result=result,
                        selected_query="",
                        graph_rows=[],
                        graph_exec_error="",
                        graph_answer=str(result.get("answer") or ""),
                        latency_s=0.0,
                    )
                )
            except Exception:
                pass
            request_clarification = result.get("request_clarification")
            if isinstance(request_clarification, dict):
                _render_request_clarification(
                    request_clarification,
                    graph_path=graph_path,
                    fuseki_query_url=_active_fuseki_query_url(),
                )
            st.stop()

        try:
            schema = _load_schema_from_path(schema_path)
        except Exception as exc:
            st.error(f"Schema load failed: {exc}")
            st.stop()

        request_id = uuid.uuid4().hex
        request_started = time.perf_counter()
        direct_capability_option = None
        if (
            cost_aware_direct_answers
            and not guided_query
            and dr_definition is None
            and execute_selected
            and _graph_backend_available(graph_path)
            and not _interactive_budget_exceeded(request_started, reserve_s=2.0)
        ):
            direct_capability_option = _single_direct_capability_option(
                question=question,
                graph_path=graph_path,
                fuseki_query_url=_active_fuseki_query_url(),
            )

        if (
            not guided_query
            and dr_definition is None
            and advisory_plan is None
            and not direct_capability_option
            and not api_key.strip()
            and not os.environ.get("LITELLM_API_KEY")
            and not (os.environ.get("USER_LLM") or os.environ.get("INFINEON_API_USER"))
        ):
            st.error("Missing API key or token-refresh credentials.")
            st.stop()

        if show_prompt and not guided_query and dr_definition is None and advisory_plan is None and not direct_capability_option:
            prompt = generate_candidate_prompt(question, schema, k=5)
            st.subheader("Candidate Generation Prompt")
            st.code(prompt, language="text")

        try:
            if dr_definition is not None:
                result = {
                    "answer": str(dr_definition.get("answer") or ""),
                    "selected_query": "",
                    "candidates": [],
                    "schema_ranked": [],
                    "learning_ranked": [],
                    "metadata": {
                        "dr_ontology_route": True,
                        "llm_skipped": True,
                        "matched_term": dr_definition.get("matched_term"),
                        "term_kind": dr_definition.get("term_kind"),
                        "term_uri": dr_definition.get("term_uri"),
                        "ontology_path": dr_definition.get("ontology_path"),
                    },
                    "errors": [],
                    "prompt": "",
                    "policy": "dr_ontology_definition",
                    "entropy": 0.0,
                    "selection_reason": str(dr_definition.get("reason") or "Deterministic DR ontology definition selected."),
                    "used_ml": False,
                    "effective_question": question,
                    "selection_explanation": {
                        "selected_policy": "dr_ontology_definition",
                        "selection_reason": str(dr_definition.get("reason") or "Deterministic DR ontology definition selected."),
                        "selected_query_valid": True,
                        "selected_query_errors": [],
                        "selected_execution_has_rows": None,
                    },
                    "answerability": {
                        "status": "ontology_definition_answered",
                        "can_answer": True,
                        "reason": "The answer was retrieved deterministically from the Digital Reference ontology.",
                    },
                    "confidence_route": {
                        "enabled": True,
                        "route": "auto_answer",
                        "score1": 0.98,
                        "score2": 0.05,
                        "margin": 0.93,
                        "selected_query": "",
                        "reason": "deterministic Digital Reference ontology definition route",
                        "safety_flags": [],
                        "blocking_safety_flags": [],
                    },
                    "clarification": None,
                    "request_clarification": None,
                }
            elif guided_query:
                result = {
                    "answer": "Validated guided query selected.",
                    "selected_query": guided_query,
                    "candidates": [{"query": guided_query, "source": "guided"}],
                    "schema_ranked": [],
                    "learning_ranked": [],
                    "metadata": {"guided_query": True},
                    "errors": [],
                    "prompt": "",
                    "policy": "guided",
                    "entropy": 0.0,
                    "selection_reason": "Validated guided query pattern selected.",
                    "used_ml": False,
                    "effective_question": question,
                    "selection_explanation": {
                        "selected_policy": "guided",
                        "selection_reason": "Validated guided query pattern selected.",
                        "selected_query_valid": True,
                        "selected_query_errors": [],
                        "selected_execution_has_rows": None,
                    },
                    "answerability": {
                        "status": "guided_pending_execution",
                        "can_answer": None,
                        "reason": "A validated guided query was selected and will be executed directly.",
                    },
                    "clarification": None,
                    "request_clarification": None,
                }
            elif advisory_plan is not None:
                result = {
                    "answer": "Graph-grounded advisory template selected.",
                    "selected_query": advisory_plan.query,
                    "candidates": [{"query": advisory_plan.query, "source": "advisory"}],
                    "schema_ranked": [],
                    "learning_ranked": [],
                    "metadata": {
                        "advisory_route": True,
                        "advisory_plan_id": advisory_plan.plan_id,
                        "advisory_title": advisory_plan.title,
                        "llm_skipped": True,
                    },
                    "errors": [],
                    "prompt": "",
                    "policy": "advisory",
                    "entropy": 0.0,
                    "selection_reason": "Deterministic graph-grounded advisory template selected.",
                    "used_ml": False,
                    "effective_question": question,
                    "selection_explanation": {
                        "selected_policy": "advisory",
                        "selection_reason": "Deterministic advisory template selected.",
                        "selected_query_valid": True,
                        "selected_query_errors": [],
                        "selected_execution_has_rows": None,
                    },
                    "answerability": {
                        "status": "advisory_pending_execution",
                        "can_answer": None,
                        "reason": (
                            "A deterministic advisory query was selected and will be executed "
                            "against the graph. The output is analytical guidance, not an "
                            "autonomous business decision."
                        ),
                    },
                    "confidence_route": {
                        "enabled": True,
                        "route": "auto_answer",
                        "score1": 0.91,
                        "score2": 0.24,
                        "margin": 0.67,
                        "selected_query": advisory_plan.query,
                        "reason": "deterministic graph-grounded advisory route",
                        "safety_flags": ["graph_grounded_advisory_not_business_decision"],
                    },
                    "clarification": None,
                    "request_clarification": None,
                }
            elif direct_capability_option:
                result = _direct_capability_result(question, direct_capability_option)
            else:
                if not api_url.strip():
                    st.error("Missing API URL.")
                    st.stop()
                os.environ["INFINEON_REQUEST_TIMEOUT_SEC"] = os.environ.get(
                    "KGQA_INTERACTIVE_LLM_TIMEOUT_SEC",
                    str(int(DEFAULT_INTERACTIVE_LLM_TIMEOUT_SEC)),
                )
                os.environ["INFINEON_MAX_RETRIES"] = os.environ.get(
                    "KGQA_INTERACTIVE_LLM_MAX_RETRIES",
                    "0",
                )
                os.environ["INFINEON_RETRY_BACKOFF_SEC"] = os.environ.get(
                    "KGQA_INTERACTIVE_LLM_RETRY_BACKOFF_SEC",
                    "0.25",
                )
                os.environ["LLM_BACKEND"] = llm_backend.strip() or "infineon"
                if llm_backend in {"litellm", "lite_llm"}:
                    os.environ["LITELLM_BASE_URL"] = api_url.strip()
                    os.environ["LITELLM_CHAT_ENDPOINT"] = api_endpoint.strip() or "/chat/completions"
                    os.environ["LITELLM_MODEL"] = model_name.strip() or default_model
                    if api_key.strip():
                        os.environ["LITELLM_API_KEY"] = api_key.strip()
                    os.environ.setdefault("BASE_URL", api_url.strip())
                    os.environ.setdefault("LITE_LLM_TOKEN", api_key.strip())
                    os.environ["INFINEON_AUTO_REFRESH_TOKEN"] = "0"
                os.environ["INFINEON_CHAT_ENDPOINT"] = api_endpoint.strip() or "/chat/completions"
                client = InfineonGPTClient(
                    model=model_name.strip() or None,
                    base_url=api_url.strip() or None,
                    api_key=api_key.strip() or None,
                    temperature=float(temperature),
                )
                def _run_llm_pipeline() -> Dict[str, Any]:
                    with _temporary_socket_timeout(float(os.environ["INFINEON_REQUEST_TIMEOUT_SEC"])):
                        return answer_question(
                            question,
                            schema,
                            llm_client=client,
                            enable_entity_linking=True,
                            use_ml_ranking=bool(use_ml_ranking),
                            ml_policy=ml_policy,
                            ml_model_path=ml_model_path.strip() or None,
                            ml_ambiguity_config_path=ambiguity_config_path.strip() or None,
                            include_candidate_diagnostics=bool(show_candidate_diagnostics),
                            enable_clarification=not bool(fast_interactive_mode),
                            enable_answerability_assessment=not bool(fast_interactive_mode),
                        )

                result = _run_with_timeout(
                    _run_llm_pipeline,
                    min(
                        _interactive_remaining_sec(request_started, reserve_s=1.0),
                        max(1.0, float(os.environ["INFINEON_REQUEST_TIMEOUT_SEC"]) + 1.0),
                    ),
                    label="LLM candidate generation and ranking",
                )
        except LLMAuthError as exc:
            st.error("Infineon GPT authentication failed.")
            st.write(str(exc))
            st.info(
                "Refresh or replace INFINEON_API_KEY, then restart Streamlit. "
                "Opening the browser SSO page alone does not necessarily refresh the API token "
                "used by this local process."
            )
            st.stop()
        except LLMClientError as exc:
            st.error(f"LLM error: {exc}")
            st.stop()
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            elapsed = time.perf_counter() - request_started
            result = _interactive_timeout_result(question, elapsed, reason=str(exc))
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()
        request_elapsed = time.perf_counter() - request_started
        if _interactive_budget_exceeded(request_started, reserve_s=0.5):
            result = _interactive_timeout_result(question, request_elapsed)

        effective_question = str(result.get("effective_question", "")).strip()
        if effective_question and effective_question != question.strip():
            st.info(f"Canonicalized question: {effective_question}")

        is_direct_capability_route = bool(
            isinstance(result.get("metadata"), dict)
            and result["metadata"].get("direct_capability_route")
        )
        is_dr_ontology_route = bool(
            isinstance(result.get("metadata"), dict)
            and result["metadata"].get("dr_ontology_route")
        )
        is_advisory_route = bool(
            isinstance(result.get("metadata"), dict)
            and result["metadata"].get("advisory_route")
        )

        if (
            execute_selected
            and not guided_query
            and not is_dr_ontology_route
            and not is_direct_capability_route
            and not is_advisory_route
            and _graph_backend_available(graph_path)
            and not _interactive_budget_exceeded(request_started, reserve_s=2.0)
        ):
            execution_override = _execution_aware_selected_query_override(
                result,
                question=effective_question or question,
                graph_path=graph_path,
                fuseki_query_url=_active_fuseki_query_url(),
                sort_by_score=bool(confidence_sort_by_score),
                max_candidates=5,
            )
            if execution_override is not None:
                metadata = result.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["execution_aware_selection"] = execution_override
                replacement_query = str(execution_override.get("to_query") or "").strip()
                if replacement_query:
                    result["selected_query"] = replacement_query
                    result["selection_reason"] = (
                        str(result.get("selection_reason") or "").strip()
                        + "; execution-aware selected non-empty/shape-compatible alternative"
                    ).strip("; ")

        confidence_route = result.get("confidence_route") if (is_dr_ontology_route or is_direct_capability_route or is_advisory_route) else None
        if confidence_routing_enabled and not guided_query and not is_dr_ontology_route and not is_direct_capability_route and not is_advisory_route:
            confidence_route = _build_confidence_route(
                result,
                question=effective_question or question,
                schema_path=schema_path,
                min_score=float(confidence_min_score),
                min_margin=float(confidence_min_margin),
                enable_safety_guard=bool(confidence_safety_guard),
                sort_by_score=bool(confidence_sort_by_score),
            )
            if confidence_route is not None:
                result["confidence_route"] = confidence_route

        direct_option = None
        if (
            execute_selected
            and not guided_query
            and not is_dr_ontology_route
            and not is_direct_capability_route
            and not is_advisory_route
            and _graph_backend_available(graph_path)
            and not _interactive_budget_exceeded(request_started, reserve_s=2.0)
        ):
            direct_option = _single_direct_capability_option(
                question=effective_question or question,
                graph_path=graph_path,
                fuseki_query_url=_active_fuseki_query_url(),
            )
        if direct_option:
            direct_query = str(direct_option.get("query", "") or "").strip()
            if direct_query:
                result["selected_query"] = direct_query
                result["errors"] = []
                if not isinstance(confidence_route, dict):
                    confidence_route = {"enabled": True}
                confidence_route.update(
                    {
                        "route": "auto_answer",
                        "selected_query": direct_query,
                        "reason": (
                            "single graph-supported capability interpretation "
                            "resolved from capability, dimension, and intent"
                        ),
                        "score1": 0.96,
                        "score2": 0.18,
                        "margin": 0.78,
                        "options": [direct_option],
                        "safety_flags": [],
                        "blocking_safety_flags": [],
                    }
                )
                result["confidence_route"] = confidence_route

        selected_query = str(result.get("selected_query") or "").strip()
        if isinstance(confidence_route, dict) and confidence_route.get("route") == "auto_answer":
            selected_query = str(confidence_route.get("selected_query") or selected_query).strip()
            result["selected_query"] = selected_query
        graph_rows: List[Dict[str, str]] = []
        graph_rows_truncated = False
        graph_exec_error = ""
        graph_answer = ""
        graph_load_elapsed = 0.0
        graph_query_elapsed = 0.0
        answer_synthesis_elapsed = 0.0
        route_needs_clarification = bool(
            isinstance(confidence_route, dict)
            and confidence_route.get("route") == "clarification"
        )
        if (
            execute_selected
            and selected_query
            and _graph_backend_available(graph_path)
            and not route_needs_clarification
            and not _interactive_budget_exceeded(request_started, reserve_s=1.0)
        ):
            try:
                graph_load_started = time.perf_counter()
                graph = _run_with_timeout(
                    lambda: _load_active_graph(graph_path),
                    _interactive_remaining_sec(request_started, reserve_s=1.0),
                    label="graph backend load",
                )
                graph_load_elapsed = time.perf_counter() - graph_load_started
                graph_query_started = time.perf_counter()
                print("\n=== FINAL EXECUTION ===")
                print("GRAPH PATH:", graph_path)
                print("QUERY:")
                print(selected_query)
                graph_rows, graph_rows_truncated = _execute_query_preview(
                    graph,
                    selected_query,
                    max_rows=int(max_preview_rows),
                )
                graph_query_elapsed = time.perf_counter() - graph_query_started
                answer_synthesis_started = time.perf_counter()
                result_metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                if result_metadata.get("advisory_route") and advisory_plan is not None:
                    graph_answer = synthesize_advisory_answer(question, advisory_plan, graph_rows)
                else:
                    graph_answer = synthesize_answer(
                        question,
                        selected_query,
                        {
                            "rows": graph_rows,
                            "matched_question_id": None,
                            "error": None,
                        },
                        None if graph_rows else result.get("errors") or None,
                    )
                answer_synthesis_elapsed = time.perf_counter() - answer_synthesis_started
                if result_metadata.get("advisory_route"):
                    result["answerability"] = _clarified_answerability(graph_rows)
                elif guided_query:
                    result["answerability"] = _guided_answerability(graph_rows)
            except Exception as exc:
                graph_exec_error = str(exc)
                if guided_query:
                    result["answerability"] = _guided_answerability([], graph_exec_error)
        if (
            isinstance(confidence_route, dict)
            and confidence_route.get("route") == "auto_answer"
            and execute_selected
            and selected_query
            and not graph_exec_error
            and not graph_rows
            and not _interactive_budget_exceeded(request_started, reserve_s=2.0)
        ):
            fallback_options = _capability_backed_clarification_options(
                question=effective_question or question,
                graph_path=graph_path,
                fuseki_query_url=_active_fuseki_query_url(),
                max_options=3,
            )
            if len(fallback_options) == 1 and str(fallback_options[0].get("id", "")).startswith("capability_direct"):
                fallback_query = str(fallback_options[0].get("query", "") or "").strip()
                if fallback_query:
                    try:
                        graph_load_started = time.perf_counter()
                        graph = _run_with_timeout(
                            lambda: _load_active_graph(graph_path),
                            _interactive_remaining_sec(request_started, reserve_s=1.0),
                            label="graph backend load",
                        )
                        graph_load_elapsed = time.perf_counter() - graph_load_started
                        graph_query_started = time.perf_counter()
                        graph_rows, graph_rows_truncated = _execute_query_preview(
                            graph,
                            fallback_query,
                            max_rows=int(max_preview_rows),
                        )
                        graph_query_elapsed = time.perf_counter() - graph_query_started
                        if graph_rows:
                            selected_query = fallback_query
                            result["selected_query"] = selected_query
                            confidence_route["route"] = "auto_answer"
                            confidence_route["selected_query"] = selected_query
                            confidence_route["reason"] = (
                                "single graph-supported capability interpretation "
                                "resolved from capability and dimension"
                            )
                            confidence_route["options"] = fallback_options
                            result["confidence_route"] = confidence_route
                            route_needs_clarification = False
                            answer_synthesis_started = time.perf_counter()
                            graph_answer = synthesize_answer(
                                question,
                                selected_query,
                                {
                                    "rows": graph_rows,
                                    "matched_question_id": None,
                                    "error": None,
                                },
                                None if graph_rows else result.get("errors") or None,
                            )
                            answer_synthesis_elapsed = time.perf_counter() - answer_synthesis_started
                    except Exception as exc:
                        graph_exec_error = str(exc)
            elif len(fallback_options) >= 2:
                confidence_route["route"] = "clarification"
                confidence_route["reason"] = (
                    "The selected high-confidence query executed successfully but returned 0 rows. "
                    "The system found alternative graph-supported interpretations that should be checked by the user."
                )
                confidence_route["options"] = fallback_options
                result["confidence_route"] = confidence_route
                route_needs_clarification = True
            else:
                confidence_route["route"] = "controlled_no_answer"
                confidence_route["reason"] = (
                    "The selected interpretation returned 0 rows and no alternative answerable "
                    "interpretations were available for clarification."
                )
                confidence_route["options"] = []
                result["confidence_route"] = confidence_route
                result["answerability"] = {
                    "status": "no_rows_for_generated_queries",
                    "can_answer": False,
                    "reason": (
                        "The selected graph interpretation returned no rows, and the system did not "
                        "find multiple answerable alternatives to show for clarification."
                    ),
                }
                route_needs_clarification = False
        if selected_query and graph_rows and not graph_exec_error:
            result["selected_query"] = selected_query
            result["errors"] = []
            result["answerability"] = _clarified_answerability(graph_rows)
        total_elapsed = time.perf_counter() - request_started
        latency_breakdown = {
            "pipeline_s": request_elapsed,
            "graph_load_s": graph_load_elapsed,
            "graph_query_s": graph_query_elapsed,
            "answer_format_s": answer_synthesis_elapsed,
            "total_s": total_elapsed,
        }
        try:
            _append_jsonl(
                SESSION_LOG_PATH,
                _session_log_payload(
                    request_id=request_id,
                    question=question,
                    result=result,
                    selected_query=selected_query,
                    graph_rows=graph_rows,
                    graph_exec_error=graph_exec_error,
                    latency_s=total_elapsed,
                    latency_breakdown=latency_breakdown,
                ),
            )
            _write_user_audit_record(
                _user_audit_payload(
                    request_id=request_id,
                    question=question,
                    result=result,
                    selected_query=selected_query,
                    graph_rows=graph_rows,
                    graph_exec_error=graph_exec_error,
                    graph_answer=graph_answer,
                    latency_s=total_elapsed,
                )
            )
        except Exception:
            pass

        st.session_state["last_qa_result"] = result
        st.session_state["last_graph_rows"] = graph_rows
        st.session_state["last_selected_query"] = selected_query
        st.session_state["last_graph_answer"] = graph_answer
        st.session_state["last_question"] = question
        st.session_state["last_request_id"] = request_id
        st.session_state["last_latency_s"] = total_elapsed
        st.session_state["last_latency_breakdown"] = latency_breakdown
        st.session_state["clarification_choice_id"] = None
        st.session_state["confidence_clarification_choice_id"] = None

        clarification = result.get("clarification")
        request_clarification = result.get("request_clarification")
        needs_request_clarification = bool(
            isinstance(request_clarification, dict)
            and request_clarification.get("needs_clarification")
        )
        needs_clarification = bool(
            isinstance(clarification, dict) and clarification.get("needs_clarification")
        )
        confidence_auto_answer = bool(
            isinstance(confidence_route, dict)
            and confidence_route.get("route") == "auto_answer"
        )
        if isinstance(confidence_route, dict):
            _render_confidence_route_badge(confidence_route)
        if needs_request_clarification:
            _render_request_clarification(
                request_clarification,
                graph_path=graph_path,
                fuseki_query_url=_active_fuseki_query_url(),
            )
            clarification_rendered = True
        elif route_needs_clarification:
            _render_confidence_clarification(
                confidence_route,
                execute_selected=bool(execute_selected),
                graph_path=graph_path,
                max_preview_rows=int(max_preview_rows),
            )
            clarification_rendered = True
            if st.session_state.get("confidence_clarification_choice_id"):
                _render_answer_block(
                    answer_text=str(st.session_state.get("last_graph_answer", "")),
                    selected_query=str(st.session_state.get("last_selected_query", "")),
                    graph_rows=list(st.session_state.get("last_graph_rows") or []),
                    graph_exec_error="",
                    execute_selected=bool(execute_selected),
                    answerability=_clarified_answerability(
                        list(st.session_state.get("last_graph_rows") or [])
                    ),
                )
                _render_compact_explainability(result)
                if show_answer_evidence_graph:
                    _render_answer_subgraph(
                        selected_query=str(st.session_state.get("last_selected_query", "")),
                        graph_rows=list(st.session_state.get("last_graph_rows") or []),
                        graph_path=graph_path,
                    )
        elif needs_clarification and not confidence_auto_answer:
            _render_clarification(
                clarification,
                execute_selected=bool(execute_selected),
                graph_path=graph_path,
                max_preview_rows=int(max_preview_rows),
            )
            clarification_rendered = True
            if st.session_state.get("clarification_choice_id"):
                _render_answer_block(
                    answer_text=str(st.session_state.get("last_graph_answer", "")),
                    selected_query=str(st.session_state.get("last_selected_query", "")),
                    graph_rows=list(st.session_state.get("last_graph_rows") or []),
                    graph_exec_error="",
                    execute_selected=bool(execute_selected),
                    answerability=_clarified_answerability(
                        list(st.session_state.get("last_graph_rows") or [])
                    ),
                )
                _render_compact_explainability(result)
                if show_answer_evidence_graph:
                    _render_answer_subgraph(
                        selected_query=str(st.session_state.get("last_selected_query", "")),
                        graph_rows=list(st.session_state.get("last_graph_rows") or []),
                        graph_path=graph_path,
                    )

        if (
            not needs_request_clarification
            and (not needs_clarification or confidence_auto_answer)
            and not route_needs_clarification
        ):
            _render_answer_block(
                answer_text=graph_answer or str(result.get("answer", "")),
                selected_query=selected_query,
                graph_rows=graph_rows,
                graph_exec_error=graph_exec_error,
                execute_selected=bool(execute_selected),
                answerability=result.get("answerability") if isinstance(result, dict) else None,
            )
            _render_compact_explainability(result)
            if show_answer_evidence_graph:
                _render_answer_subgraph(
                    selected_query=selected_query,
                    graph_rows=graph_rows,
                    graph_path=graph_path,
                )

        if developer_mode:
            with st.expander("Technical details", expanded=False):
                if selected_query:
                    st.subheader("Selected Query")
                    st.code(_format_sparql_for_display(selected_query), language="sparql")
                else:
                    st.warning("No selected query.")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Policy", str(result.get("policy", "unknown")))
                col2.metric("Query-plan ML", "yes" if result.get("query_plan_ml_used") else "no")
                col3.metric("ML ranker", "yes" if result.get("ml_ranker_applied") else "no")
                col4.metric("Candidates", str(len(result.get("candidates", []))))
                pred_regime = result.get("predicted_regime")
                pred_entropy = result.get("predicted_entropy")
                if pred_regime is not None:
                    st.caption(
                        f"Predicted ambiguity regime: {pred_regime}"
                        + (
                            f" (entropy={float(pred_entropy):.3f})"
                            if pred_entropy is not None
                            else ""
                        )
                    )
                st.caption(
                    f"ML policy setting: {result.get('ml_policy', ml_policy)} | "
                    f"Model: {result.get('ml_model_path', ml_model_path)}"
                )
                st.caption(f"Pipeline time before graph preview: {request_elapsed:.2f}s")
                removed = int(result.get("candidate_duplicates_removed") or 0)
                if removed:
                    st.caption(f"Candidate deduplication removed {removed} duplicate query candidate(s).")
                gen_meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                if gen_meta:
                    routed = ", ".join(str(x) for x in gen_meta.get("schema_slice_names", []) or [])
                    if gen_meta.get("llm_skipped"):
                        st.caption("LLM generation: skipped by direct capability route")
                    elif gen_meta.get("llm_cache_enabled"):
                        st.caption(
                            "LLM generation cache: "
                            f"{'hit' if gen_meta.get('llm_cache_hit') else 'miss'}"
                        )
                    st.caption(
                        "Schema routing: "
                        f"{'sliced' if gen_meta.get('schema_slicing_applied') else 'full'}"
                        f" | confidence={gen_meta.get('schema_slice_confidence', 'n/a')}"
                        + (f" | families={routed}" if routed else "")
                    )
                answerability = result.get("answerability")
                if isinstance(answerability, dict):
                    st.subheader("Answerability")
                    st.json(answerability)
                _render_selection_explainability(result)

                if execute_selected and selected_query:
                    if not _graph_backend_available(graph_path):
                        st.error("Graph backend unavailable. Set a valid graph path or FUSEKI_QUERY_URL.")
                    else:
                        st.subheader("Graph Result Rows")
                        if graph_exec_error:
                            st.error(f"Selected query execution failed: {graph_exec_error}")
                        elif graph_rows:
                            st.dataframe(graph_rows, width="stretch")
                            if graph_rows_truncated:
                                st.caption(f"Preview truncated at {int(max_preview_rows)} rows.")
                        else:
                            st.write("No rows returned.")

                if show_candidates:
                    st.subheader("Candidates")
                    candidates = result.get("candidates", [])
                    if not candidates:
                        st.write("No candidates returned.")
                    else:
                        for idx, item in enumerate(candidates, start=1):
                            source = item.get("source", "unknown")
                            st.caption(f"Candidate {idx} ({source})")
                            st.code(_format_sparql_for_display(str(item.get("query", ""))), language="sparql")

if not clarification_rendered:
    last_result = st.session_state.get("last_qa_result")
    request_clarification = (
        (last_result or {}).get("request_clarification")
        if isinstance(last_result, dict)
        else None
    )
    if isinstance(request_clarification, dict) and request_clarification.get("needs_clarification"):
        _render_request_clarification(
            request_clarification,
            graph_path=graph_path,
            fuseki_query_url=_active_fuseki_query_url(),
        )
        clarification_rendered = True
    confidence_route = (last_result or {}).get("confidence_route") if isinstance(last_result, dict) else None
    confidence_auto_answer = bool(
        isinstance(confidence_route, dict)
        and confidence_route.get("route") == "auto_answer"
    )
    if (
        not clarification_rendered
        and isinstance(confidence_route, dict)
        and confidence_route.get("route") == "clarification"
    ):
        _render_confidence_route_badge(confidence_route)
        _render_confidence_clarification(
            confidence_route,
            execute_selected=bool(execute_selected),
            graph_path=graph_path,
            max_preview_rows=int(max_preview_rows),
        )
        clarification_rendered = True
        if st.session_state.get("confidence_clarification_choice_id"):
            _render_answer_block(
                answer_text=str(st.session_state.get("last_graph_answer", "")),
                selected_query=str(st.session_state.get("last_selected_query", "")),
                graph_rows=list(st.session_state.get("last_graph_rows") or []),
                graph_exec_error="",
                execute_selected=bool(execute_selected),
                answerability=_clarified_answerability(
                    list(st.session_state.get("last_graph_rows") or [])
                ),
            )
            if isinstance(last_result, dict):
                _render_compact_explainability(last_result)
                if show_answer_evidence_graph:
                    _render_answer_subgraph(
                        selected_query=str(st.session_state.get("last_selected_query", "")),
                        graph_rows=list(st.session_state.get("last_graph_rows") or []),
                        graph_path=graph_path,
                    )
    clarification = (last_result or {}).get("clarification") if isinstance(last_result, dict) else None
    if (
        not clarification_rendered
        and not confidence_auto_answer
        and isinstance(clarification, dict)
        and clarification.get("needs_clarification")
    ):
        _render_clarification(
            clarification,
            execute_selected=bool(execute_selected),
            graph_path=graph_path,
            max_preview_rows=int(max_preview_rows),
        )
        if st.session_state.get("clarification_choice_id"):
            _render_answer_block(
                answer_text=str(st.session_state.get("last_graph_answer", "")),
                selected_query=str(st.session_state.get("last_selected_query", "")),
                graph_rows=list(st.session_state.get("last_graph_rows") or []),
                graph_exec_error="",
                execute_selected=bool(execute_selected),
                answerability=_clarified_answerability(
                    list(st.session_state.get("last_graph_rows") or [])
                ),
            )
            if isinstance(last_result, dict):
                _render_compact_explainability(last_result)
                if show_answer_evidence_graph:
                    _render_answer_subgraph(
                        selected_query=str(st.session_state.get("last_selected_query", "")),
                        graph_rows=list(st.session_state.get("last_graph_rows") or []),
                        graph_path=graph_path,
                    )

if (
    not asked
    and not clarification_rendered
    and st.session_state.get("last_selected_query")
    and (
        st.session_state.get("last_graph_answer")
        or st.session_state.get("last_graph_rows")
    )
):
    last_result = st.session_state.get("last_qa_result")
    if isinstance(last_result, dict):
        confidence_route = last_result.get("confidence_route")
        if isinstance(confidence_route, dict):
            _render_confidence_route_badge(confidence_route)
    _render_answer_block(
        answer_text=str(st.session_state.get("last_graph_answer", "")),
        selected_query=str(st.session_state.get("last_selected_query", "")),
        graph_rows=list(st.session_state.get("last_graph_rows") or []),
        graph_exec_error="",
        execute_selected=bool(execute_selected),
        answerability=(
            last_result.get("answerability")
            if isinstance(last_result, dict)
            else _clarified_answerability(list(st.session_state.get("last_graph_rows") or []))
        ),
    )
    if isinstance(last_result, dict):
        _render_compact_explainability(last_result)
        if show_answer_evidence_graph:
            _render_answer_subgraph(
                selected_query=str(st.session_state.get("last_selected_query", "")),
                graph_rows=list(st.session_state.get("last_graph_rows") or []),
                graph_path=graph_path,
            )

if (
    st.session_state.get("last_selected_query")
    and (
        st.session_state.get("last_graph_answer")
        or st.session_state.get("last_graph_rows")
    )
):
    latency_value = st.session_state.get("last_latency_s")
    if isinstance(latency_value, (int, float)) and latency_value > 0:
        st.caption(f"End-to-end response time: {float(latency_value):.1f}s")
    latency_breakdown = st.session_state.get("last_latency_breakdown")
    if isinstance(latency_breakdown, dict):
        st.caption(
            "Timing: "
            f"LLM/ranking {float(latency_breakdown.get('pipeline_s') or 0.0):.1f}s | "
            f"graph load {float(latency_breakdown.get('graph_load_s') or 0.0):.1f}s | "
            f"query {float(latency_breakdown.get('graph_query_s') or 0.0):.1f}s | "
            f"answer formatting {float(latency_breakdown.get('answer_format_s') or 0.0):.1f}s"
        )
    _render_feedback_panel()
