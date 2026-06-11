import os
import time
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple, Any

import streamlit as st
import streamlit.components.v1 as components
from rdflib import Graph, BNode, URIRef
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from kg.schema import load_schema
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
DEFAULT_FUSEKI_QUERY_URL = "http://localhost:3030/infineon/sparql"
APP_LOG_DIR = PROJECT_ROOT / "logs"
SESSION_LOG_PATH = APP_LOG_DIR / "kgqa_sessions.jsonl"
FEEDBACK_LOG_PATH = APP_LOG_DIR / "kgqa_feedback.jsonl"
DEFAULT_ML_MODEL_PATHS = [
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


def _graph_backend_available(graph_path: str) -> bool:
    return bool(_active_fuseki_query_url()) or bool(graph_path and os.path.exists(graph_path))


@st.cache_resource(show_spinner=False)
def _load_graph_cached(graph_path: str, fuseki_query_url: str = "") -> Graph:
    if fuseki_query_url.strip():
        return Graph(store=SPARQLStore(fuseki_query_url.strip()))
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


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
        "schema_route": {
            "applied": bool(metadata.get("schema_slicing_applied")),
            "confidence": metadata.get("schema_slice_confidence"),
            "families": metadata.get("schema_slice_names") or [],
        },
    }


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
            graph = Graph(store=SPARQLStore(fuseki_query_url))
            list(graph.query("SELECT * WHERE { ?s ?p ?o } LIMIT 1"))
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
    col_ok, col_bad = st.columns(2)
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
        left, right = st.columns([1.2, 3])
        left.metric("Decision", route)
        right.write("The selected query passed the confidence policy and safety checks.")
        flags = confidence_route.get("safety_flags") or []
        if flags:
            st.caption("Technical safety notes are available in Developer mode.")
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
    left.metric("Confidence", confidence)
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
        graph = _load_active_graph(graph_path)
        triples, meta = collect_answer_evidence_triples(
            graph=graph,
            query=selected_query,
            limit=18,
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
    options = [
        option
        for option in list(clarification.get("options") or [])
        if option.get("row_count") is None or int(option.get("row_count") or 0) > 0
    ]
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
                        graph = _load_active_graph(graph_path)
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


def _render_request_clarification(clarification: Dict[str, Any]) -> None:
    st.subheader("Clarify Request")
    st.write(str(clarification.get("reason", "The requested task is not specific enough yet.")))
    st.write(str(clarification.get("question", "What do you want to know?")))
    for option in list(clarification.get("options") or []):
        cols = st.columns([3, 1])
        cols[0].write(str(option.get("label", "Option")))
        cols[1].button(
            "Use",
            key=f"request_clarify_{option.get('id')}",
            use_container_width=True,
            on_click=_set_question_input,
            args=(str(option.get("rewritten_question", "")),),
        )


def _set_question_input(value: str) -> None:
    st.session_state["question_input"] = value
    st.session_state["guided_query_override_question"] = ""
    st.session_state["guided_query_override"] = ""


def _set_guided_question_input(value: str, query: str) -> None:
    st.session_state["question_input"] = value
    st.session_state["guided_query_override_question"] = value
    st.session_state["guided_query_override"] = query


def _append_question_input(current: str, phrase: str) -> None:
    base = str(current or "").strip()
    addition = str(phrase or "").strip()
    if not addition:
        return
    if not base:
        st.session_state["question_input"] = addition
        return
    if addition.lower() in base.lower():
        return
    separator = " " if base.endswith((" ", "-", "/", ",")) else " "
    st.session_state["question_input"] = f"{base}{separator}{addition}".strip()


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
            return
    separator = "" if text.endswith((" ", "\n", "\t", "-", "/", ",")) or not text else " "
    st.session_state["question_input"] = f"{text}{separator}{completion} "


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

    return sorted(set(flags))


def _blocking_confidence_safety_flags(flags: List[str]) -> List[str]:
    blocking_prefixes = (
        "list_count_conflict",
        "scalar_grouping_conflict",
        "scope_missing:",
        "contract_conflict:",
        "class_missing:",
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


def _render_confidence_route_badge(route: Dict[str, object]) -> None:
    if route.get("route") == "auto_answer":
        st.success("High-confidence graph answer.")
    else:
        st.warning(
            "The system found multiple plausible interpretations. "
            "Please choose one before it answers."
        )


def _render_confidence_clarification(
    route: Dict[str, object],
    *,
    execute_selected: bool,
    graph_path: str,
    max_preview_rows: int,
) -> None:
    st.subheader("Clarify Interpretation")
    st.write("I found multiple possible interpretations.")
    st.markdown("**Did you mean...**")
    options = list(route.get("options") or [])
    if not options:
        st.warning("No candidate interpretations are available for clarification.")
        return
    for option in options[:3]:
        with st.container(border=True):
            st.markdown(f"**{str(option.get('label') or 'Interpretation')}**")
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
            ):
                chosen_query = str(option.get("query", "") or "").strip()
                st.session_state["last_selected_query"] = chosen_query
                st.session_state["confidence_clarification_choice_id"] = option.get("id")
                if execute_selected and chosen_query and _graph_backend_available(graph_path):
                    try:
                        graph = _load_active_graph(graph_path)
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

    chosen_id = st.session_state.get("confidence_clarification_choice_id")
    if chosen_id:
        chosen = next((opt for opt in options if opt.get("id") == chosen_id), None)
        if chosen is not None:
            st.success(f"Using clarified interpretation: {chosen.get('label')}")


def _normalize_question_key(question: str) -> str:
    return " ".join(str(question or "").strip().lower().rstrip("?.!").split())


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
def _guided_query_row_count(graph_path: str, query: str) -> Tuple[int, str]:
    if not query.strip() or not graph_path or not _graph_backend_available(graph_path):
        return 0, "graph_backend_or_query_missing"
    try:
        graph = _load_active_graph(graph_path)
        rows, _truncated = _execute_query_preview(graph, query, max_rows=1)
        return len(rows), ""
    except Exception as exc:
        return 0, str(exc)


def _answerable_guided_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    answerable: List[Dict[str, str]] = []
    for row in rows:
        row_count, query_error = _guided_query_row_count(str(DEFAULT_GRAPH_PATH), str(row.get("query", "")))
        if row_count <= 0:
            continue
        enriched = dict(row)
        enriched["row_count"] = str(row_count)
        enriched["query_error"] = query_error
        answerable.append(enriched)
    return answerable


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

SMART_QUERY_CONTINUATIONS = [
    ("by", "keyword"),
    ("for", "keyword"),
    ("in", "keyword"),
    ("compared to", "comparison"),
    ("in detail", "detail level"),
    ("for each", "breakdown"),
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
    continuation_context = bool(str(question or "").endswith((" ", "\n", "\t"))) and _smart_query_has_selected_concept(question)
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
    elif continuation_context:
        for label, suggestion_type in SMART_QUERY_CONTINUATIONS:
            suggestions.append({"label": label, "insert": label, "type": suggestion_type, "raw": ""})
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


@st.cache_data(show_spinner=False)
def _kg_autocomplete_entries(schema_path: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for term in SMART_QUERY_DOMAIN_TERMS:
        label = str(term.get("label", "")).strip()
        if not label:
            continue
        entries.append(
            {
                "label": label,
                "insert": str(term.get("insert", label)),
                "type": str(term.get("type", "concept")),
                "raw": label.upper().replace(" ", "_").replace("-", "_"),
                "aliases": list(term.get("aliases") or []),
            }
        )
    for dimension in SMART_QUERY_DIMENSIONS:
        entries.append(
            {
                "label": dimension,
                "insert": dimension,
                "type": "dimension",
                "raw": dimension.upper().replace(" ", "_").replace("-", "_"),
                "aliases": _smart_query_tokens(dimension),
            }
        )
    for label, suggestion_type in SMART_QUERY_CONTINUATIONS:
        entries.append(
            {
                "label": label,
                "insert": label,
                "type": suggestion_type,
                "raw": "",
                "aliases": _smart_query_tokens(label),
            }
        )

    schema_dict = _load_schema_dict_cached(schema_path)
    schema_values: List[Tuple[str, str]] = []
    schema_values.extend((str(v), "class") for v in list(schema_dict.get("classes") or []))
    schema_values.extend((str(v), "relationship") for v in list(schema_dict.get("predicates") or []))
    schema_values.extend((str(v), "property") for v in list(schema_dict.get("properties") or []))
    schema_values.extend((str(v.get("type", "")), "relationship") for v in list(schema_dict.get("relationships") or []) if isinstance(v, dict))
    for value, value_type in schema_values:
        human = _humanize_axis_value(str(value))
        if not human or len(human) > 48:
            continue
        entries.append(
            {
                "label": human,
                "insert": human,
                "type": value_type,
                "raw": str(value),
                "aliases": _smart_query_tokens(human) + _smart_query_tokens(str(value)),
            }
        )
    deduped = []
    seen = set()
    for entry in entries:
        key = (str(entry.get("label", "")).lower(), str(entry.get("type", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


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


def _render_kg_autocomplete_input(schema_path: str) -> str:
    current = str(st.session_state.get("question_input", "") or "")
    result = KG_AUTOCOMPLETE_COMPONENT(
        label="Your question",
        value=current,
        entries=_kg_autocomplete_entries(schema_path),
        key="kg_question_autocomplete",
        default={"text": current},
    )
    if isinstance(result, dict):
        text = str(result.get("text", "") or "")
    else:
        text = current
    st.session_state["question_input"] = text
    return text


def _render_question_guidance() -> None:
    with st.expander("Question guide", expanded=False):
        st.caption(
            "Use the question box above for free text, or pick an example/builder option here."
        )
        tabs = st.tabs(["Examples", "Guided builder", "Available topics"])
        with tabs[0]:
            query_lookup = _load_guided_query_lookup()
            example_options = [
                example
                for example in EXAMPLE_QUESTIONS
                if query_lookup.get(_normalize_question_key(example), "")
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
                st.caption("Examples use validated graph queries directly. Press Ask after selecting one.")
        with tabs[1]:
            validated_patterns = _validated_guided_patterns()
            topic_options = _unique_preserving_order([row["topic"] for row in validated_patterns])
            if not topic_options:
                st.warning("No validated guided patterns are available.")
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
                st.caption("This question comes from the validated graph-query library.")
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
            validated_patterns = _validated_guided_patterns()
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
    graph = _load_active_graph(graph_path)
    if _active_fuseki_query_url():
        try:
            triples = list(graph.query("SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"))
            subjects = list(graph.query("SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE { ?s ?p ?o }"))
            resources = list(
                graph.query(
                    """
                    SELECT (COUNT(DISTINCT ?x) AS ?count)
                    WHERE {
                      { ?x ?p ?o }
                      UNION
                      { ?s ?p ?x FILTER(isIRI(?x) || isBlank(?x)) }
                    }
                    """
                )
            )
            return {
                "triples": int(triples[0][0].toPython()) if triples else 0,
                "resource_nodes": int(resources[0][0].toPython()) if resources else 0,
                "subject_entities": int(subjects[0][0].toPython()) if subjects else 0,
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
This knowledge graph describes survey and analytical data around semiconductor and automotive demand. It connects regional demand, current- and future-demand analyses, vehicle sales, autonomous-driving development, order-cancellation responses, shortages, inventory trends, technology categories, vehicle types, companies, components, survey origins, and time periods. The graph is designed to support structured questions over business measures such as demand, percentage change, participant counts, sales units, shortage values, inventory trends, and autonomous-driving percentages.

## What the graph contains
{topic_lines}

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
Use precise wording when you know the intended calculation, such as **average**, **total**, **count**, **highest**, **by month**, or **by technology category**. If a question leaves the intended interpretation open, the QA system may ask for clarification before answering.
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
            --kg-bg: #f6f9fb;
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
            --kg-shadow: 0 12px 32px rgba(24, 54, 64, 0.08);
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
            max-width: 1160px;
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
            box-shadow: 0 6px 18px rgba(24, 54, 64, 0.04);
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
            background: linear-gradient(135deg, #ffffff 0%, #eef8f7 100%);
            border: 1px solid var(--kg-border);
            border-left: 6px solid var(--kg-accent);
            border-radius: 8px;
            box-shadow: var(--kg-shadow);
            margin-bottom: 1.25rem;
            padding: 1.35rem 1.5rem;
        }
        .kg-hero h1 {
            margin: 0.1rem 0 0.35rem;
        }
        .kg-kicker {
            color: var(--kg-blue);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .kg-page-copy {
            color: var(--kg-muted);
            max-width: 48rem;
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


def _render_graph_overview(schema_path: str, graph_path: str) -> None:
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

    with st.expander("Complete schema inventory", expanded=False):
        st.markdown("#### All classes")
        st.write(sorted(raw_schema.get("classes") or []))
        st.markdown("#### All predicates")
        st.write(sorted(raw_schema.get("predicates") or []))
        st.markdown("#### All properties")
        st.write(sorted(raw_schema.get("properties") or []))
        st.markdown("#### All declared relationships")
        st.dataframe(list(raw_schema.get("relationships") or []), width="stretch")

    triples = _overview_relationship_triples(raw_schema)
    if triples:
        st.subheader("High-level graph structure")
        st.caption("Core class-to-class relationships declared by the ontology.")
        graph_nodes = {node for s, _p, o in triples for node in (s, o)}
        graph_col, legend_col = st.columns([4.2, 1.2], gap="large")
        with graph_col:
            html = build_graph_html(
                triples,
                height_px=620,
                heading="Schema Relationship Graph",
                max_nodes=140,
                max_edges=180,
            )
            components.html(html, height=660, scrolling=True)
        with legend_col:
            _render_graph_side_panel(
                node_count=len(graph_nodes),
                edge_count=len(triples),
                relationship_label="Declared relationship",
                has_entity_nodes=any(isinstance(node, URIRef) for node in graph_nodes),
                has_literal_nodes=any(not isinstance(node, URIRef) for node in graph_nodes),
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

    st.subheader("System Metrics")
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
    report_path = st.text_input(
        "Routing report JSON",
        value=_default_confidence_routing_report_path(),
    )
    if not report_path.strip():
        st.info("Enter a confidence routing JSON report path.")
        return
    try:
        report = _load_confidence_routing_report(report_path.strip())
    except Exception as exc:
        st.error(f"Could not load report: {exc}")
        return

    _render_confidence_dashboard_summary(report)
    st.divider()
    _render_confidence_dashboard_distributions(report)
    st.divider()
    _render_confidence_dashboard_cases(report)


st.set_page_config(page_title="True Demand KG QA", layout="wide")
_inject_app_styles()

with st.sidebar:
    page = st.radio("Page", ["Ask", "Confidence Routing Dashboard", "Graph Overview"])
    st.markdown(
        '<div class="kg-sidebar-note">Ask questions over the True Demand KG or open the overview report.</div>',
        unsafe_allow_html=True,
    )
    developer_mode = st.checkbox("Developer mode", value=False)

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
    confidence_min_score = 0.90
    confidence_min_margin = 0.00
    confidence_safety_guard = True
    confidence_sort_by_score = True
    fast_interactive_mode = True
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
            api_url = st.text_input("INFINEON_API_URL", value=default_url)
            api_endpoint = st.text_input("INFINEON_CHAT_ENDPOINT", value=default_endpoint)
            model_name = st.text_input("INFINEON_MODEL", value=default_model)
            api_key = st.text_input("INFINEON_API_KEY", value=default_key, type="password")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

            st.subheader("Data")
            schema_path = st.text_input("Schema path", value=str(DEFAULT_SCHEMA_PATH))
            graph_path = st.text_input("Graph path", value=str(DEFAULT_GRAPH_PATH))
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
            ml_policy = st.selectbox(
                "ML policy",
                options=["auto", "all", "mid", "off"],
                index=0,
                help="auto: predict ambiguity and route automatically, all: always ML, mid: ML only for medium ambiguity, off: schema-only.",
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
                value=0.90,
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

    if fuseki_query_url.strip():
        os.environ["FUSEKI_QUERY_URL"] = fuseki_query_url.strip()
    else:
        os.environ.pop("FUSEKI_QUERY_URL", None)
    os.environ["INFINEON_ENABLE_SCHEMA_SLICING"] = "1" if family_schema_routing_enabled else "0"
    os.environ["INFINEON_SCHEMA_SLICING_MAX_FAMILIES"] = str(int(family_schema_routing_max))
    os.environ["INFINEON_SCHEMA_SLICING_FULL_FALLBACK"] = "1" if family_schema_routing_fallback else "0"

    with st.expander("System status", expanded=False):
        _render_system_status(
            schema_path=schema_path,
            graph_path=graph_path,
            fuseki_query_url=fuseki_query_url.strip(),
            model_path=ml_model_path,
            api_url=api_url,
            api_key=api_key,
        )

if page == "Graph Overview":
    _render_graph_overview(schema_path, graph_path)
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
    </section>
    """,
    unsafe_allow_html=True,
)

question = _render_kg_autocomplete_input(schema_path)
asked = st.button("Ask", type="primary")

_render_question_guidance()

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
    if not question.strip():
        st.warning("Please enter a question.")
    elif not api_url.strip():
        st.error("Missing API URL.")
    elif not guided_query and not api_key.strip() and not (
        os.environ.get("USER_LLM")
        or os.environ.get("INFINEON_API_USER")
    ):
        st.error("Missing API key or token-refresh credentials.")
    else:
        try:
            schema = _load_schema_from_path(schema_path)
        except Exception as exc:
            st.error(f"Schema load failed: {exc}")
            st.stop()

        if show_prompt and not guided_query:
            prompt = generate_candidate_prompt(question, schema, k=5)
            st.subheader("Candidate Generation Prompt")
            st.code(prompt, language="text")

        request_id = uuid.uuid4().hex
        request_started = time.perf_counter()
        try:
            if guided_query:
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
            else:
                os.environ["INFINEON_CHAT_ENDPOINT"] = api_endpoint.strip() or "/chat/completions"
                client = InfineonGPTClient(
                    model=model_name.strip() or None,
                    base_url=api_url.strip() or None,
                    api_key=api_key.strip() or None,
                    temperature=float(temperature),
                )
                result = answer_question(
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
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()
        request_elapsed = time.perf_counter() - request_started

        effective_question = str(result.get("effective_question", "")).strip()
        if effective_question and effective_question != question.strip():
            st.info(f"Canonicalized question: {effective_question}")

        confidence_route = None
        if confidence_routing_enabled and not guided_query:
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
        if execute_selected and selected_query and _graph_backend_available(graph_path) and not route_needs_clarification:
            try:
                graph_load_started = time.perf_counter()
                graph = _load_active_graph(graph_path)
                graph_load_elapsed = time.perf_counter() - graph_load_started
                graph_query_started = time.perf_counter()
                graph_rows, graph_rows_truncated = _execute_query_preview(
                    graph,
                    selected_query,
                    max_rows=int(max_preview_rows),
                )
                graph_query_elapsed = time.perf_counter() - graph_query_started
                answer_synthesis_started = time.perf_counter()
                graph_answer = synthesize_answer(
                    question,
                    selected_query,
                    {
                        "rows": graph_rows,
                        "matched_question_id": None,
                        "error": None,
                    },
                    result.get("errors") or None,
                )
                answer_synthesis_elapsed = time.perf_counter() - answer_synthesis_started
                if guided_query:
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
        ):
            confidence_route["route"] = "clarification"
            confidence_route["reason"] = (
                "The selected high-confidence query executed successfully but returned 0 rows. "
                "The system should not auto-answer without graph evidence."
            )
            result["confidence_route"] = confidence_route
            route_needs_clarification = True
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
            _render_request_clarification(request_clarification)
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
        _render_request_clarification(request_clarification)
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

if developer_mode:
    st.divider()
    st.subheader("Interactive Graph Explorer")

if developer_mode and not _graph_backend_available(graph_path):
    st.warning("Graph backend unavailable. Set a valid graph path or Fuseki query endpoint.")
elif developer_mode:
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
                heading="Infineon Ontology Schema",
                max_nodes=160,
                max_edges=220,
            )
            components.html(
                html,
                height=int(graph_height) + 40,
                scrolling=True,
            )

    with tab_question:
        st.caption(
            "Visualize the graph area related to the last selected query."
        )
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
                    st.warning(
                        "Could not extract a non-empty subgraph for this query."
                    )
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
                        heading="Infineon Raw Data Triples",
                    )
                    st.caption(
                        f"Showing {len(triples)} raw triples out of total {total}."
                    )
                    components.html(
                        html,
                        height=int(graph_height) + 40,
                        scrolling=True,
                    )
