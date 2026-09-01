"""Run Philipp's True Demand / Digital Reference question set end-to-end
against a live graph (Fuseki by default) and report route/answer/timing per
question.

This exercises pipeline.qa.answer_question() directly -- the same function
the standard "Ask" button in app.py calls -- so it is a good complement to
manual spot-checking in the Streamlit UI, but it does NOT go through
app.py's own guided-query shortcuts (the dynamic time-window/region fixes
live there). A handful of questions that those shortcuts handle in the live
UI may therefore show up here as unsupported/low-confidence even though the
app answers them correctly -- that's a known gap in this runner, not a bug
in the app.

Requires the same LLM credentials the Streamlit app uses (e.g. INFINEON_API_KEY
and an API URL, via env vars or a .env file) -- candidate generation
constructs a client even for template-only candidates, so questions on the
kg_query route will error out with "Missing API URL." without them. Pure
definition-route questions (category 12) work with no credentials at all,
which is a useful way to sanity-check the script itself.

Usage:
    python evaluation/run_philipp_question_set.py --fuseki-query-url http://localhost:3030/infineon/sparql
    python evaluation/run_philipp_question_set.py --graph data/infineon/graph.ttl   # no live Fuseki
    python evaluation/run_philipp_question_set.py --call-llm                       # also spend LLM calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from rdflib import Graph

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg.fuseki import make_sparql_store
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


def _load_graph(graph_path: str, fuseki_query_url: str) -> Graph:
    if fuseki_query_url.strip():
        return Graph(store=make_sparql_store(fuseki_query_url.strip()))
    graph = Graph()
    graph.parse(graph_path, format="turtle")
    return graph


def _load_questions(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def _run_one(
    item: Dict[str, Any],
    graph: Graph,
    schema: Any,
    llm_client: Any,
    max_rows: int,
) -> Dict[str, Any]:
    question = item["question"]
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "expected_route": item.get("expected_route"),
    }
    try:
        result = answer_question(
            question,
            schema,
            llm_client=llm_client,
            enable_entity_linking=True,
            use_ml_ranking=False,
            enable_clarification=True,
            enable_answerability_assessment=True,
        )
        request_route = result.get("request_route")
        record["actual_route"] = (
            request_route.get("route") if isinstance(request_route, dict) else None
        )
        record["route_matches_expected"] = record["actual_route"] == record["expected_route"] or (
            record["expected_route"] == "kg_query" and bool(result.get("selected_query"))
        )
        selected_query = str(result.get("selected_query") or "")
        record["selected_query"] = selected_query
        if selected_query:
            try:
                rows = []
                for idx, row in enumerate(graph.query(_ensure_prefixes(selected_query))):
                    if idx >= max_rows:
                        break
                    rows.append(row.asdict() if hasattr(row, "asdict") else dict(enumerate(row)))
                record["row_count"] = len(rows)
                record["sample_rows"] = [
                    {str(k): str(v) for k, v in r.items()} for r in rows[:3]
                ]
            except Exception as exc:  # noqa: BLE001
                record["execution_error"] = str(exc)
                record["row_count"] = 0
        else:
            record["row_count"] = 0
        request_clarification = result.get("request_clarification")
        record["needs_clarification"] = bool(
            isinstance(request_clarification, dict) and request_clarification.get("needs_clarification")
        )
        record["answer"] = str(result.get("answer") or "")[:500]
    except Exception as exc:  # noqa: BLE001
        record["error"] = str(exc)
        record["row_count"] = 0
    record["elapsed_s"] = round(time.perf_counter() - started, 3)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Philipp's True Demand / Digital Reference question set against a live graph."
    )
    parser.add_argument(
        "--questions",
        default="evaluation/question_sets/philipp_true_demand_dr_test_questions.json",
    )
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--fuseki-query-url", default=os.getenv("FUSEKI_QUERY_URL", ""))
    parser.add_argument("--out", default="results/philipp_question_set_run.json")
    parser.add_argument("--limit", type=int, help="Only run the first N questions.")
    parser.add_argument("--category", help="Only run questions from this category key.")
    parser.add_argument(
        "--call-llm",
        action="store_true",
        help="Also let candidate generation use the LLM (costs money/time). Default is templates + retrieval only.",
    )
    parser.add_argument("--max-rows", type=int, default=20)
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[: args.limit]

    print(f"Loading graph ({'Fuseki: ' + args.fuseki_query_url if args.fuseki_query_url else args.graph}) ...")
    graph = _load_graph(args.graph, args.fuseki_query_url)
    schema = load_schema(args.schema)
    llm_client = InfineonGPTClient() if args.call_llm else None

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(questions, start=1):
        record = _run_one(item, graph, schema, llm_client, args.max_rows)
        results.append(record)
        status = "OK" if not record.get("error") else "ERROR"
        rows = record.get("row_count", 0)
        print(
            f"[{idx}/{len(questions)}] {record['id']} ({record['category']}) "
            f"route={record.get('actual_route')} rows={rows} {status} "
            f"({record['elapsed_s']}s)"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions_run": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "zero_row_answers": sum(
            1 for r in results if not r.get("error") and r.get("row_count", 0) == 0
        ),
        "route_mismatches": sum(
            1 for r in results if r.get("route_matches_expected") is False
        ),
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} results to {out_path}")
    print(
        f"errors={summary['errors']} zero_row_answers={summary['zero_row_answers']} "
        f"route_mismatches={summary['route_mismatches']}"
    )


if __name__ == "__main__":
    main()
