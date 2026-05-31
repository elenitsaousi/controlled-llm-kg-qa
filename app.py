import os
import time
import json
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple, Any

import streamlit as st
import streamlit.components.v1 as components
from rdflib import Graph, BNode, URIRef

from kg.schema import load_schema
from llm.answer_synthesis import synthesize_answer
from llm.candidate_generation import generate_candidate_prompt
from llm.client import InfineonGPTClient, LLMAuthError, LLMClientError
from pipeline.qa import answer_question
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
DEFAULT_ML_MODEL_PATHS = [
    PROJECT_ROOT / "ranking" / "models" / "infineon_np_tfidf_ranker_entitylink.json",
    PROJECT_ROOT / "ranking" / "models" / "infineon_np_tfidf_ranker.json",
    PROJECT_ROOT / "ranking" / "models" / "infineon_ranker.joblib",
]
DEFAULT_AMBIGUITY_CONFIG_PATHS = [
    PROJECT_ROOT / "ranking" / "models" / "infineon_ambiguity_config_500.json",
    PROJECT_ROOT / "ranking" / "models" / "infineon_ambiguity_config.json",
]
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


@st.cache_resource(show_spinner=False)
def _load_graph_cached(graph_path: str) -> Graph:
    g = Graph()
    g.parse(graph_path, format="turtle")
    return g


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
            st.success(f"Returned {len(graph_rows)} rows from Infineon graph.")
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
            "data was found for that exact interpretation, or the generated graph path "
            "is still too narrow."
        ),
    }


def _render_answer_subgraph(
    *,
    selected_query: str,
    graph_rows: List[Dict[str, str]],
    graph_path: str,
) -> None:
    if not selected_query or not graph_path or not os.path.exists(graph_path):
        return
    if not graph_rows:
        st.subheader("Inspected Query Path")
        st.warning(
            "The selected query returned no graph rows, so no answer evidence graph is shown."
        )
        return
    try:
        graph = _load_graph_cached(graph_path)
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
                elif execute_selected and chosen_query and os.path.exists(graph_path):
                    try:
                        graph = _load_graph_cached(graph_path)
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


def _active_guided_query(question: str) -> str:
    if str(st.session_state.get("guided_query_override_question", "")).strip() == (question or "").strip():
        return str(st.session_state.get("guided_query_override", "") or "").strip()
    return ""


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
    if not query.strip() or not graph_path or not os.path.exists(graph_path):
        return 0, "graph_or_query_missing"
    try:
        graph = _load_graph_cached(graph_path)
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
    if not graph_path or not os.path.exists(graph_path):
        return {}
    graph = _load_graph_cached(graph_path)
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
    summary = f"""# Infineon Knowledge Graph Overview

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
  <title>Infineon Knowledge Graph Overview</title>
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
            --kg-bg: #101719;
            --kg-panel: #172124;
            --kg-panel-soft: #1d2a2d;
            --kg-border: #26373a;
            --kg-text: #edf4f3;
            --kg-muted: #91a4a4;
            --kg-accent: #19d6c6;
            --kg-green: #31b67a;
            --kg-green-soft: rgba(49, 182, 122, 0.16);
            --kg-accent-soft: rgba(25, 214, 198, 0.16);
            --kg-success: #153a34;
            --kg-warning: #3a3118;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 8%, rgba(25, 214, 198, 0.10), transparent 28rem),
                radial-gradient(circle at 8% 96%, rgba(49, 182, 122, 0.16), transparent 24rem),
                linear-gradient(180deg, #111719 0%, #101719 100%);
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
        }
        [data-testid="stSidebar"] {
            background: #070a0b;
            border-right: 1px solid var(--kg-border);
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
            padding-top: 3.75rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: var(--kg-text);
            letter-spacing: 0;
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
            box-shadow: 0 0 0 1px var(--kg-accent) !important;
        }
        button[kind="primary"] {
            background: var(--kg-accent) !important;
            border-color: var(--kg-accent) !important;
            color: #041514 !important;
            font-weight: 600;
        }
        button[kind="secondary"] {
            border-color: var(--kg-border) !important;
            color: var(--kg-text) !important;
            background: var(--kg-panel) !important;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--kg-border);
        }
        [data-testid="stAlert"] div {
            color: var(--kg-text);
        }
        [data-testid="stMetric"] {
            background: var(--kg-panel);
            border: 1px solid var(--kg-border);
            border-radius: 8px;
            padding: 0.8rem 1rem;
        }
        [data-testid="stExpander"] {
            background: var(--kg-panel);
            border: 1px solid var(--kg-border);
            border-radius: 8px;
        }
        [data-testid="stDataFrame"],
        pre {
            border: 1px solid var(--kg-border);
            border-radius: 8px;
        }
        iframe {
            border-radius: 14px;
        }
        .kg-hero {
            isolation: isolate;
            overflow: hidden;
            position: relative;
            padding: 0.2rem 0 1rem;
        }
        .kg-hero::before {
            background:
                radial-gradient(circle at 13% 130%, rgba(49, 182, 122, 0.46), transparent 18rem),
                radial-gradient(circle at 23% 140%, rgba(25, 214, 198, 0.38), transparent 21rem);
            border-radius: 999px;
            bottom: -11rem;
            content: "";
            height: 24rem;
            left: -6rem;
            position: absolute;
            width: 42rem;
            z-index: -1;
        }
        .kg-hero::after {
            background: linear-gradient(
                118deg,
                transparent 0%,
                rgba(49, 182, 122, 0.08) 34%,
                rgba(25, 214, 198, 0.20) 74%,
                transparent 100%
            );
            border-radius: 999px;
            bottom: -12rem;
            content: "";
            height: 18rem;
            left: -2rem;
            position: absolute;
            transform: rotate(10deg);
            width: 34rem;
            z-index: -1;
        }
        .kg-hero h1 {
            margin-bottom: 0.25rem;
        }
        .kg-kicker {
            color: var(--kg-accent);
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .kg-page-copy {
            color: var(--kg-muted);
            max-width: 42rem;
        }
        .kg-sidebar-note {
            background: linear-gradient(135deg, var(--kg-green-soft), var(--kg-accent-soft));
            border: 1px solid rgba(49, 182, 122, 0.34);
            border-radius: 8px;
            color: var(--kg-text);
            padding: 0.75rem;
            margin: 0.5rem 0 1rem;
        }
        .kg-side-legend {
            background: rgba(11, 28, 40, 0.88);
            border: 1px solid rgba(25, 214, 198, 0.48);
            border-radius: 14px;
            min-height: 240px;
            padding: 1rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
        }
        .kg-side-kicker {
            color: #f5d96c;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .kg-side-section {
            border-top: 1px solid rgba(25, 214, 198, 0.2);
            margin-top: 0.9rem;
            padding-top: 0.9rem;
        }
        .kg-side-title {
            color: #f5d96c;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }
        .kg-side-row {
            align-items: center;
            color: #d7e4e4;
            display: flex;
            font-size: 0.82rem;
            gap: 0.55rem;
            margin: 0.45rem 0;
        }
        .kg-side-dot {
            background: #b9f2f2;
            border: 2px solid #7de4df;
            border-radius: 999px;
            display: inline-block;
            height: 12px;
            width: 12px;
        }
        .kg-side-dot.muted {
            background: #54656d;
            border-color: #70838b;
        }
        .kg-side-line {
            border-top: 2px solid #6a8a8f;
            display: inline-block;
            width: 24px;
        }
        .kg-side-copy {
            color: var(--kg-muted);
            font-size: 0.82rem;
            line-height: 1.6;
        }
        .kg-evidence-row {
            border-top: 1px solid rgba(148, 163, 184, 0.18);
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

    st.title("Infineon Knowledge Graph Overview")
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


st.set_page_config(page_title="Infineon KG QA", layout="wide")
_inject_app_styles()

with st.sidebar:
    page = st.radio("Page", ["Ask", "Graph Overview"])
    st.markdown(
        '<div class="kg-sidebar-note">Ask questions over the Infineon knowledge graph or open the overview report.</div>',
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
    use_ml_ranking = True
    ml_policy = "auto"
    ml_model_path = _default_ml_model_path()
    ambiguity_config_path = _default_ambiguity_config_path()
    show_prompt = False
    show_candidates = False
    show_candidate_diagnostics = False
    execute_selected = True
    max_preview_rows = 200
    full_graph_limit = 3000
    subgraph_edge_limit = 1200
    subgraph_hops = 1
    graph_height = 760

    if developer_mode:
        with st.expander("Developer settings", expanded=True):
            st.subheader("Backend")
            api_url = st.text_input("INFINEON_API_URL", value=default_url)
            api_endpoint = st.text_input("INFINEON_CHAT_ENDPOINT", value=default_endpoint)
            model_name = st.text_input("INFINEON_MODEL", value=default_model)
            api_key = st.text_input("INFINEON_API_KEY", value=default_key, type="password")
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

            st.subheader("Data")
            schema_path = st.text_input("Schema path", value=str(DEFAULT_SCHEMA_PATH))
            graph_path = st.text_input("Graph path", value=str(DEFAULT_GRAPH_PATH))

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

if page == "Graph Overview":
    _render_graph_overview(schema_path, graph_path)
    st.stop()

st.markdown(
    """
    <section class="kg-hero">
      <div class="kg-kicker">Knowledge graph assistant</div>
      <h1>Infineon KG QA</h1>
      <div class="kg-page-copy">Ask a natural-language question and receive a graph-grounded answer. When the request is genuinely ambiguous, the system asks for the intended interpretation first.</div>
    </section>
    """,
    unsafe_allow_html=True,
)

question = st.text_area(
    "Your question",
    placeholder="e.g., How does semiconductor future demand evolve across technology categories and quarters?",
    height=120,
    key="question_input",
)
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

        selected_query = str(result.get("selected_query") or "").strip()
        graph_rows: List[Dict[str, str]] = []
        graph_rows_truncated = False
        graph_exec_error = ""
        graph_answer = ""
        if execute_selected and selected_query and os.path.exists(graph_path):
            try:
                graph = _load_graph_cached(graph_path)
                graph_rows, graph_rows_truncated = _execute_query_preview(
                    graph,
                    selected_query,
                    max_rows=int(max_preview_rows),
                )
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
                if guided_query:
                    result["answerability"] = _guided_answerability(graph_rows)
            except Exception as exc:
                graph_exec_error = str(exc)
                if guided_query:
                    result["answerability"] = _guided_answerability([], graph_exec_error)

        st.session_state["last_qa_result"] = result
        st.session_state["last_graph_rows"] = graph_rows
        st.session_state["last_selected_query"] = selected_query
        st.session_state["last_graph_answer"] = graph_answer
        st.session_state["last_question"] = question
        st.session_state["clarification_choice_id"] = None

        clarification = result.get("clarification")
        request_clarification = result.get("request_clarification")
        needs_request_clarification = bool(
            isinstance(request_clarification, dict)
            and request_clarification.get("needs_clarification")
        )
        needs_clarification = bool(
            isinstance(clarification, dict) and clarification.get("needs_clarification")
        )
        if needs_request_clarification:
            _render_request_clarification(request_clarification)
            clarification_rendered = True
        elif needs_clarification:
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
                _render_answer_subgraph(
                    selected_query=str(st.session_state.get("last_selected_query", "")),
                    graph_rows=list(st.session_state.get("last_graph_rows") or []),
                    graph_path=graph_path,
                )

        if not needs_request_clarification and not needs_clarification:
            _render_answer_block(
                answer_text=graph_answer or str(result.get("answer", "")),
                selected_query=selected_query,
                graph_rows=graph_rows,
                graph_exec_error=graph_exec_error,
                execute_selected=bool(execute_selected),
                answerability=result.get("answerability") if isinstance(result, dict) else None,
            )
            _render_compact_explainability(result)
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
                answerability = result.get("answerability")
                if isinstance(answerability, dict):
                    st.subheader("Answerability")
                    st.json(answerability)
                _render_selection_explainability(result)

                if execute_selected and selected_query:
                    if not os.path.exists(graph_path):
                        st.error(f"Graph path not found: {graph_path}")
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
    clarification = (last_result or {}).get("clarification") if isinstance(last_result, dict) else None
    if not clarification_rendered and isinstance(clarification, dict) and clarification.get("needs_clarification"):
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
                _render_answer_subgraph(
                    selected_query=str(st.session_state.get("last_selected_query", "")),
                    graph_rows=list(st.session_state.get("last_graph_rows") or []),
                    graph_path=graph_path,
                )

if developer_mode:
    st.divider()
    st.subheader("Interactive Graph Explorer")

if developer_mode and not os.path.exists(graph_path):
    st.warning(f"Graph path not found: {graph_path}")
elif developer_mode:
    tab_full, tab_question = st.tabs(["Full Graph", "Question Subgraph"])

    with tab_full:
        st.caption(
            "Visualize entities/relationships with zoom, drag and node click. "
            "Use a triple limit to keep the browser responsive."
        )
        if full_graph_limit == 0:
            st.warning("Full graph without limit may be very heavy in browser.")
        if st.button("Load Full Graph", key="load_full_graph_btn"):
            with st.spinner("Loading graph and building visualization..."):
                graph = _load_graph_cached(graph_path)
                triples, total = collect_full_graph_triples(graph, limit=int(full_graph_limit))
                if not triples:
                    st.warning("No triples available for visualization.")
                else:
                    html = build_graph_html(
                        triples,
                        height_px=int(graph_height),
                        heading="Infineon Graph (Full View)",
                    )
                    st.caption(
                        f"Showing {len(triples)} triples out of total {total}."
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
                    graph = _load_graph_cached(graph_path)
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
                            heading="Infineon Graph (Question Subgraph)",
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
