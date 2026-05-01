import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import streamlit as st
import streamlit.components.v1 as components
from rdflib import Graph

from kg.schema import load_schema
from llm.candidate_generation import generate_candidate_prompt
from llm.client import InfineonGPTClient, LLMClientError
from pipeline.qa import answer_question
from visualization.interactive_graph import (
    build_graph_html,
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
            rows.append({str(k): str(v) for k, v in rd.items()})
        else:
            rows.append({f"col{j + 1}": str(v) for j, v in enumerate(row)})
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


st.set_page_config(page_title="Infineon KG QA", layout="wide")

st.title("Infineon KG QA")
st.caption("Ask a natural-language question and inspect generated SPARQL plus graph results.")

with st.sidebar:
    st.subheader("Backend")
    default_url = os.environ.get("INFINEON_API_URL", "https://gpt4ifx.icp.infineon.com")
    default_model = os.environ.get("INFINEON_MODEL", "gpt-4o")
    default_endpoint = os.environ.get("INFINEON_CHAT_ENDPOINT", "/chat/completions")
    default_key = os.environ.get("INFINEON_API_KEY", "")

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

    st.subheader("Display")
    show_prompt = st.checkbox("Show candidate prompt", value=False)
    show_candidates = st.checkbox("Show candidates", value=True)
    execute_selected = st.checkbox("Execute selected query on graph", value=True)
    max_preview_rows = st.number_input("Max preview rows", min_value=10, max_value=1000, value=200, step=10)

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

question = st.text_area(
    "Your question",
    placeholder="e.g., How does semiconductor future demand evolve across technology categories and quarters?",
    height=120,
)

if "last_qa_result" not in st.session_state:
    st.session_state["last_qa_result"] = None
if "last_graph_rows" not in st.session_state:
    st.session_state["last_graph_rows"] = []
if "last_selected_query" not in st.session_state:
    st.session_state["last_selected_query"] = ""

if st.button("Ask", type="primary"):
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
            )
        except LLMClientError as exc:
            st.error(f"LLM error: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        effective_question = str(result.get("effective_question", "")).strip()
        if effective_question and effective_question != question.strip():
            st.info(f"Canonicalized question: {effective_question}")

        selected_query = str(result.get("selected_query") or "").strip()
        graph_rows: List[Dict[str, str]] = []
        graph_rows_truncated = False
        graph_exec_error = ""
        if execute_selected and selected_query and os.path.exists(graph_path):
            try:
                graph = _load_graph_cached(graph_path)
                graph_rows, graph_rows_truncated = _execute_query_preview(
                    graph,
                    selected_query,
                    max_rows=int(max_preview_rows),
                )
            except Exception as exc:
                graph_exec_error = str(exc)

        st.session_state["last_qa_result"] = result
        st.session_state["last_graph_rows"] = graph_rows
        st.session_state["last_selected_query"] = selected_query

        st.subheader("Answer")
        if graph_exec_error:
            st.error(f"Query execution error: {graph_exec_error}")
            st.write(result.get("answer", "No answer."))
        elif execute_selected and selected_query:
            if graph_rows:
                st.success(f"Returned {len(graph_rows)} rows from Infineon graph.")
            else:
                st.warning("Selected query returned 0 rows from Infineon graph.")
        else:
            st.write(result.get("answer", "No answer."))

        if selected_query:
            st.subheader("Selected Query")
            st.code(selected_query, language="sparql")
        else:
            st.warning("No selected query.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Policy", str(result.get("policy", "unknown")))
        col2.metric("Used ML", "yes" if result.get("used_ml") else "no")
        col3.metric("Candidates", str(len(result.get("candidates", []))))
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
                    st.code(str(item.get("query", "")), language="sparql")

st.divider()
st.subheader("Interactive Graph Explorer")

if not os.path.exists(graph_path):
    st.warning(f"Graph path not found: {graph_path}")
else:
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
