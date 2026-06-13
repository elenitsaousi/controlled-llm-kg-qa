"""Run or estimate a cost-aware KGQA efficiency question set.

Default mode is intentionally conservative and cheap: it executes only
direct capability queries and estimates that every unresolved question would
need one LLM call. Use --call-llm only when you explicitly want to spend LLM
calls for non-direct questions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg.capabilities import DEFAULT_REGISTRY
from kg.schema import load_schema
from llm.client import InfineonGPTClient
from pipeline.qa import answer_question


DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query


def _format_value(value: object) -> str:
    text = str(value)
    ns = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"
    if text.startswith(ns):
        return text[len(ns) :]
    return text


def _load_questions(path: str) -> List[Dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Question file must contain a JSON list.")
    rows = []
    for idx, item in enumerate(payload, start=1):
        if isinstance(item, str):
            rows.append({"id": f"Q{idx:04d}", "question": item})
        elif isinstance(item, dict) and item.get("question"):
            rows.append({str(k): str(v) for k, v in item.items()})
    return rows


def _load_graph(graph_path: str, fuseki_query_url: str) -> Graph:
    if fuseki_query_url.strip():
        return Graph(store=SPARQLStore(fuseki_query_url.strip()))
    graph = Graph()
    graph.parse(graph_path, format="turtle")
    return graph


def _preview_rows(graph: Graph, query: str, max_rows: int = 3) -> Tuple[List[Dict[str, str]], str]:
    try:
        results = graph.query(_ensure_prefixes(query))
        rows: List[Dict[str, str]] = []
        for idx, row in enumerate(results):
            if idx >= max_rows:
                break
            if hasattr(row, "asdict"):
                rows.append({str(k): _format_value(v) for k, v in row.asdict().items()})
            else:
                rows.append({f"col{j + 1}": _format_value(v) for j, v in enumerate(row)})
        return rows, ""
    except Exception as exc:
        return [], str(exc)


def _estimated_llm_calls(metadata: Dict[str, Any]) -> int:
    if metadata.get("llm_skipped") or metadata.get("guided_query"):
        return 0
    calls = 0 if (metadata.get("llm_cache_enabled") and metadata.get("llm_cache_hit")) else 1
    if metadata.get("full_schema_generation_attempted"):
        calls += 0 if (metadata.get("llm_cache_enabled") and metadata.get("full_schema_llm_cache_hit")) else 1
    return calls


def _direct_row(
    *,
    request_id: str,
    question: str,
    query: str,
    graph_rows: List[Dict[str, str]],
    latency_s: float,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "route": "auto_answer",
        "score1": 1.0,
        "score2": 0.0,
        "margin": 1.0,
        "selected_source": "capability_inventory",
        "selected_query": query,
        "graph_row_count": len(graph_rows),
        "graph_rows_preview": graph_rows,
        "graph_error": "",
        "latency_s": round(latency_s, 3),
        "latency_breakdown": {"total_s": round(latency_s, 3)},
        "candidate_count": 1,
        "llm": {
            "skipped": True,
            "cache_enabled": False,
            "cache_hit": False,
            "full_schema_generation_attempted": False,
            "full_schema_cache_hit": False,
            "estimated_calls": 0,
        },
        "schema_route": {"applied": False, "confidence": "direct_capability", "families": []},
    }


def _estimated_llm_needed_row(
    *,
    request_id: str,
    question: str,
    latency_s: float,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "route": "llm_required_estimate",
        "score1": None,
        "score2": None,
        "margin": None,
        "selected_source": "",
        "selected_query": "",
        "graph_row_count": 0,
        "graph_rows_preview": [],
        "graph_error": "",
        "latency_s": round(latency_s, 3),
        "latency_breakdown": {"total_s": round(latency_s, 3)},
        "candidate_count": 0,
        "llm": {
            "skipped": False,
            "cache_enabled": False,
            "cache_hit": False,
            "full_schema_generation_attempted": False,
            "full_schema_cache_hit": False,
            "estimated_calls": 1,
        },
        "schema_route": {"applied": False, "confidence": "not_direct", "families": []},
    }


def _full_llm_row(
    *,
    request_id: str,
    question: str,
    result: Dict[str, Any],
    graph_rows: List[Dict[str, str]],
    graph_error: str,
    latency_s: float,
) -> Dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    selected_query = str(result.get("selected_query") or "")
    selected_source = ""
    selected_key = " ".join(selected_query.split()).lower()
    for candidate in result.get("candidates") or []:
        query_key = " ".join(str(candidate.get("query") or "").split()).lower()
        if selected_key and query_key == selected_key:
            selected_source = str(candidate.get("source") or "")
            break
    return {
        "request_id": request_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "route": "llm_ranking",
        "score1": None,
        "score2": None,
        "margin": None,
        "selected_source": selected_source,
        "selected_query": selected_query,
        "graph_row_count": len(graph_rows),
        "graph_rows_preview": graph_rows,
        "graph_error": graph_error,
        "latency_s": round(latency_s, 3),
        "latency_breakdown": {"total_s": round(latency_s, 3)},
        "candidate_count": len(result.get("candidates") or []),
        "llm": {
            "skipped": bool(metadata.get("llm_skipped")),
            "cache_enabled": bool(metadata.get("llm_cache_enabled")),
            "cache_hit": bool(metadata.get("llm_cache_hit")),
            "full_schema_generation_attempted": bool(metadata.get("full_schema_generation_attempted")),
            "full_schema_cache_hit": bool(metadata.get("full_schema_llm_cache_hit")),
            "estimated_calls": _estimated_llm_calls(metadata),
        },
        "schema_route": {
            "applied": bool(metadata.get("schema_slicing_applied")),
            "confidence": metadata.get("schema_slice_confidence"),
            "families": metadata.get("schema_slice_names") or [],
        },
    }


def run(
    *,
    questions_path: str,
    out_log: str,
    graph_path: str,
    fuseki_query_url: str,
    schema_path: str,
    call_llm: bool,
    limit: int | None,
    enable_llm_cache: bool,
) -> Dict[str, int]:
    questions = _load_questions(questions_path)
    if limit is not None:
        questions = questions[: max(0, int(limit))]

    os.environ["FUSEKI_QUERY_URL"] = fuseki_query_url.strip()
    os.environ["INFINEON_ENABLE_LLM_CACHE"] = "1" if enable_llm_cache else "0"

    graph = _load_graph(graph_path, fuseki_query_url)
    schema = load_schema(schema_path) if call_llm else None
    client = InfineonGPTClient() if call_llm else None

    out_path = Path(out_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "direct": 0, "llm_estimated": 0, "llm_called": 0, "direct_empty": 0}
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(questions, start=1):
            question = str(row.get("question") or "").strip()
            if not question:
                continue
            request_id = str(row.get("id") or f"EFF{idx:04d}")
            started = time.perf_counter()

            report = DEFAULT_REGISTRY.resolve(question)
            direct_query = DEFAULT_REGISTRY.direct_query_for(report)
            if direct_query:
                graph_rows, graph_error = _preview_rows(graph, direct_query, max_rows=3)
                if graph_rows and not graph_error:
                    payload = _direct_row(
                        request_id=request_id,
                        question=question,
                        query=direct_query,
                        graph_rows=graph_rows,
                        latency_s=time.perf_counter() - started,
                    )
                    counts["direct"] += 1
                    counts["total"] += 1
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    print(f"[{idx}/{len(questions)}] direct {request_id}: {question}")
                    continue
                counts["direct_empty"] += 1

            if not call_llm:
                payload = _estimated_llm_needed_row(
                    request_id=request_id,
                    question=question,
                    latency_s=time.perf_counter() - started,
                )
                counts["llm_estimated"] += 1
                counts["total"] += 1
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(f"[{idx}/{len(questions)}] estimated LLM {request_id}: {question}")
                continue

            assert schema is not None and client is not None
            result = answer_question(
                question,
                schema,
                llm_client=client,
                enable_entity_linking=True,
                use_ml_ranking=True,
                ml_policy="all",
                include_candidate_diagnostics=False,
                enable_clarification=False,
                enable_answerability_assessment=False,
            )
            selected_query = str(result.get("selected_query") or "")
            graph_rows, graph_error = _preview_rows(graph, selected_query, max_rows=3) if selected_query else ([], "")
            payload = _full_llm_row(
                request_id=request_id,
                question=question,
                result=result,
                graph_rows=graph_rows,
                graph_error=graph_error,
                latency_s=time.perf_counter() - started,
            )
            counts["llm_called"] += int(payload["llm"]["estimated_calls"] > 0)
            counts["total"] += 1
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"[{idx}/{len(questions)}] LLM {request_id}: {question}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/estimate cost-aware KGQA efficiency over a question set.")
    parser.add_argument("--questions", default="evaluation/question_sets/true_demand_efficiency_500.json")
    parser.add_argument("--out-log", default="logs/kgqa_efficiency_500.jsonl")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--fuseki-query-url", default=os.getenv("FUSEKI_QUERY_URL", ""))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--call-llm", action="store_true", help="Actually call the LLM for non-direct questions. This costs money.")
    parser.add_argument("--enable-llm-cache", action="store_true")
    args = parser.parse_args()

    counts = run(
        questions_path=args.questions,
        out_log=args.out_log,
        graph_path=args.graph,
        fuseki_query_url=args.fuseki_query_url,
        schema_path=args.schema,
        call_llm=args.call_llm,
        limit=args.limit,
        enable_llm_cache=args.enable_llm_cache,
    )
    print("===== KGQA EFFICIENCY QUESTION SET =====")
    print(f"Questions processed: {counts['total']}")
    print(f"Direct graph-supported: {counts['direct']}")
    print(f"Estimated LLM-needed: {counts['llm_estimated']}")
    print(f"Actual LLM-called rows: {counts['llm_called']}")
    print(f"Direct templates with no rows: {counts['direct_empty']}")
    print(f"Log: {args.out_log}")


if __name__ == "__main__":
    main()
