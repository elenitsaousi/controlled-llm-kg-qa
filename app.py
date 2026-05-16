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
from llm.client import InfineonGPTClient, LLMClientError
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
) -> None:
    st.subheader("Answer")
    if graph_exec_error:
        st.error(f"Query execution error: {graph_exec_error}")
        st.write(answer_text or "No answer.")
    elif execute_selected and selected_query:
        if graph_rows:
            st.success(f"Returned {len(graph_rows)} rows from Infineon graph.")
            st.write(answer_text)
        else:
            st.warning("Selected query returned 0 rows from Infineon graph.")
            st.write(answer_text or "No results were found for this question.")
    else:
        st.write(answer_text or "No answer.")


def _render_answer_subgraph(
    *,
    selected_query: str,
    graph_rows: List[Dict[str, str]],
    graph_path: str,
) -> None:
    if not selected_query or not graph_path or not os.path.exists(graph_path):
        return
    try:
        graph = _load_graph_cached(graph_path)
        triples, meta = collect_answer_evidence_triples(
            graph=graph,
            query=selected_query,
            limit=24,
        )
    except Exception:
        return
    if not triples:
        return

    st.subheader("Relevant Graph")
    st.caption(
        "The business relationships used by the selected query. "
        f"Predicates: {meta.get('predicate_count', 0)} | Edges shown: {meta.get('edge_count', 0)}"
    )
    graph_nodes = {node for s, _p, o in triples for node in (s, o)}
    graph_col, legend_col = st.columns([4.2, 1.2], gap="large")
    with graph_col:
        html = build_graph_html(
            triples,
            height_px=520,
            heading="Answer Evidence Graph",
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
    for option in options:
        cols = st.columns([3, 1])
        cols[0].write(str(option.get("label", "Interpretation")))
        if cols[1].button(
            "Use",
            key=f"clarify_{option.get('id')}",
            use_container_width=True,
        ):
            chosen_query = str(option.get("query", "") or "").strip()
            st.session_state["last_selected_query"] = chosen_query
            st.session_state["clarification_choice_id"] = option.get("id")
            if execute_selected and chosen_query and os.path.exists(graph_path):
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
) -> None:
    node_rows = []
    if has_entity_nodes:
        node_rows.append('<div class="kg-side-row"><span class="kg-side-dot"></span> Classes / entities</div>')
    if has_literal_nodes:
        node_rows.append('<div class="kg-side-row"><span class="kg-side-dot muted"></span> Literal / value nodes</div>')
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
        </aside>
        """,
        unsafe_allow_html=True,
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
            --kg-accent-soft: rgba(25, 214, 198, 0.16);
            --kg-success: #153a34;
            --kg-warning: #3a3118;
        }

        .stApp {
            background:
                radial-gradient(circle at 85% 8%, rgba(25, 214, 198, 0.10), transparent 28rem),
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
            padding: 0.2rem 0 1rem;
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
            background: var(--kg-accent-soft);
            border: 1px solid rgba(25, 214, 198, 0.28);
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

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""

question = st.text_area(
    "Your question",
    placeholder="e.g., How does semiconductor future demand evolve across technology categories and quarters?",
    height=120,
    key="question_input",
)

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

asked = st.button("Ask", type="primary")
clarification_rendered = False

if asked:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not api_url.strip() or not api_key.strip():
        st.error("Missing API URL or API key.")
    else:
        try:
            schema = _load_schema_from_path(schema_path)
        except Exception as exc:
            st.error(f"Schema load failed: {exc}")
            st.stop()

        if show_prompt:
            prompt = generate_candidate_prompt(question, schema, k=5)
            st.subheader("Candidate Generation Prompt")
            st.code(prompt, language="text")

        request_started = time.perf_counter()
        try:
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
            except Exception as exc:
                graph_exec_error = str(exc)

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
