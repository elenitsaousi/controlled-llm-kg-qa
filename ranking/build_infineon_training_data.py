#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdflib import Graph

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    load_dotenv()
except Exception:
    pass

from kg.schema import load_schema
from llm.candidate_generation import generate_candidates, repair_candidate_query
from llm.client import InfineonGPTClient
from ranking.feature_extraction import extract_features


PREFIX = (
    "PREFIX survey: "
    "<http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)


VAR_RE = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
SINGLE_QUOTE_STR_RE = re.compile(r"'[^']*'")
DOUBLE_QUOTE_STR_RE = re.compile(r'"[^"]*"')
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _query_family_signature(query: str) -> str:
    q = " ".join(query.strip().split())
    q = SINGLE_QUOTE_STR_RE.sub("'STR'", q)
    q = DOUBLE_QUOTE_STR_RE.sub('"STR"', q)
    q = NUMBER_RE.sub("NUM", q)
    q = VAR_RE.sub("?VAR", q)
    import hashlib

    return "fam_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:16]


def _result_signature(rows) -> Set[Tuple[str, ...]]:
    return set(tuple(str(v) for v in row) for row in rows)


def _ensure_prefixes(query: str) -> str:
    return query if "PREFIX" in query.upper() else (PREFIX + query)


def _resolve_backend() -> str:
    backend = (
        os.environ.get("LLM_PROVIDER")
        or os.environ.get("LLM_BACKEND")
        or "infineon"
    ).strip().lower()
    if backend == "infiineon":
        backend = "infineon"
    return backend


def _validate_backend_env(allow_gold_only: bool) -> None:
    backend = _resolve_backend()
    if backend != "infineon":
        raise RuntimeError(
            f"Unsupported LLM backend '{backend}'. Supported backend: infineon."
        )
    if allow_gold_only:
        return
    missing = []
    if not os.environ.get("INFINEON_API_URL"):
        missing.append("INFINEON_API_URL")
    if not os.environ.get("INFINEON_API_KEY"):
        missing.append("INFINEON_API_KEY")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing {names}. Configure env/.env before building training data. "
            "Use --allow-gold-only only for debug runs."
        )


def _try_query_signature(graph: Graph, query: str) -> Tuple[int, Set[Tuple[str, ...]], str]:
    full_query = _ensure_prefixes(query)
    try:
        sig = _result_signature(graph.query(full_query))
        return 1, sig, ""
    except Exception as exc:
        return 0, set(), str(exc)


def _repair_candidates_for_run(
    question: str,
    schema,
    candidates: List[Dict[str, str]],
    graph: Graph,
    llm_client: Optional[object],
    max_candidates: int,
    max_attempts: int,
) -> Tuple[List[Dict[str, str]], int, int]:
    if not candidates or max_candidates <= 0 or max_attempts <= 0:
        return candidates, 0, 0

    repaired_attempted = 0
    repaired_succeeded = 0
    updated: List[Dict[str, str]] = []

    for cand in candidates:
        query = str(cand.get("query", "")).strip()
        if not query:
            continue

        is_valid, _, error_message = _try_query_signature(graph, query)
        if is_valid:
            updated.append(cand)
            continue

        if repaired_attempted >= max_candidates:
            updated.append(cand)
            continue

        repaired_attempted += 1
        current_error = error_message or "unknown query error"
        repaired_query = None

        for _ in range(max_attempts):
            try:
                proposal = repair_candidate_query(
                    question=question,
                    schema=schema,
                    invalid_query=query,
                    error_message=current_error,
                    llm_client=llm_client,
                )
            except Exception as exc:
                proposal = None
                current_error = str(exc)

            if not proposal:
                break

            is_valid_after, _, error_after = _try_query_signature(graph, proposal)
            if is_valid_after:
                repaired_query = proposal
                repaired_succeeded += 1
                break
            current_error = error_after or "unknown query error"

        if repaired_query:
            c = dict(cand)
            c["query"] = repaired_query
            updated.append(c)
        else:
            updated.append(cand)

    # Stable dedup after repair.
    deduped: List[Dict[str, str]] = []
    seen = set()
    for cand in updated:
        query = str(cand.get("query", "")).strip()
        if not query:
            continue
        key = " ".join(query.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)

    return deduped, repaired_attempted, repaired_succeeded


def build_training_data(
    dataset_path: str,
    graph_path: str,
    schema_path: str,
    output_path: str,
    k: int = 5,
    n_runs: int = 3,
    allow_gold_only: bool = False,
    repair_invalid: bool = True,
    repair_max_candidates: int = 2,
    repair_attempts: int = 1,
) -> None:
    _validate_backend_env(allow_gold_only=allow_gold_only)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)
    schema = load_schema(schema_path)

    g = Graph()
    g.parse(graph_path, format="turtle")
    print(f"Graph loaded: {len(g)} triples")
    print(f"Benchmark questions: {len(dataset)}")

    training_data: Dict[str, List[Dict[str, object]]] = {}
    total_candidates = 0
    total_correct = 0
    total_llm_candidates = 0
    total_generation_failures = 0
    total_generation_runs = 0
    total_repair_attempts = 0
    total_repair_successes = 0

    llm_client: Optional[object] = None
    if os.environ.get("INFINEON_API_URL") and os.environ.get("INFINEON_API_KEY"):
        try:
            llm_client = InfineonGPTClient()
        except Exception as exc:
            if not allow_gold_only:
                raise
            print(f"⚠️ Could not initialize Infineon client once upfront: {exc}")

    for i, item in enumerate(dataset, start=1):
        qid = item["id"]
        question = item["question"]
        gold_query = item["query"]
        ambiguity = str(item.get("ambiguity_label", "unknown")).lower()
        topic = str(item.get("topic", "unknown"))
        family = str(item.get("family", "")) or _query_family_signature(gold_query)

        print(f"\n[{i}/{len(dataset)}] {qid} ({ambiguity})")
        print(f"Q: {question}")

        try:
            gold_rows = _result_signature(g.query(_ensure_prefixes(gold_query)))
        except Exception as exc:
            print(f"  ❌ Gold query failed: {exc}")
            continue

        if not gold_rows:
            print("  ⚠️ Gold query returned empty set; skipping question")
            continue

        seen_queries = set()
        rows: List[Dict[str, object]] = []

        # Always include gold query as guaranteed positive.
        try:
            gold_feats = extract_features(question, gold_query, schema_dict)
        except Exception:
            gold_feats = {}
        rows.append(
            {
                "query_id": f"{qid}_GOLD",
                "question": question,
                "query": gold_query,
                "gold_query": gold_query,
                "is_correct": 1,
                "is_valid": 1,
                "features": gold_feats,
                "ambiguity_label": ambiguity,
                "topic": topic,
                "family": family,
                "source": "gold",
                "run_index": -1,
            }
        )
        seen_queries.add(gold_query.strip())
        total_candidates += 1
        total_correct += 1

        for run_idx in range(n_runs):
            print(f"  Generation run {run_idx + 1}/{n_runs} ...")
            total_generation_runs += 1
            try:
                generated = generate_candidates(question, schema, k=k, llm_client=llm_client)
                candidates = generated.get("candidates", [])
            except Exception as exc:
                print(f"    ❌ Candidate generation failed: {exc}")
                total_generation_failures += 1
                continue

            if repair_invalid and candidates:
                candidates, rep_attempted, rep_succeeded = _repair_candidates_for_run(
                    question=question,
                    schema=schema,
                    candidates=candidates,
                    graph=g,
                    llm_client=llm_client,
                    max_candidates=repair_max_candidates,
                    max_attempts=repair_attempts,
                )
                total_repair_attempts += rep_attempted
                total_repair_successes += rep_succeeded

            for cand_idx, cand in enumerate(candidates):
                query = str(cand.get("query", "")).strip()
                if not query:
                    continue
                if query in seen_queries:
                    continue
                seen_queries.add(query)

                is_valid, sig, _ = _try_query_signature(g, query)
                is_correct = int(is_valid == 1 and sig == gold_rows and len(sig) > 0)

                try:
                    feats = extract_features(question, query, schema_dict)
                except Exception:
                    feats = {}

                rows.append(
                    {
                        "query_id": f"{qid}_R{run_idx}_C{cand_idx}",
                        "question": question,
                        "query": query,
                        "gold_query": gold_query,
                        "is_correct": is_correct,
                        "is_valid": is_valid,
                        "features": feats,
                        "ambiguity_label": ambiguity,
                        "topic": topic,
                        "family": family,
                        "source": "llm",
                        "run_index": run_idx,
                    }
                )
                total_candidates += 1
                total_correct += is_correct
                total_llm_candidates += 1

        correct_in_q = sum(1 for r in rows if int(r["is_correct"]) == 1)
        llm_in_q = sum(1 for r in rows if str(r.get("source")) == "llm")
        print(f"  candidates={len(rows)} correct={correct_in_q} llm_candidates={llm_in_q}")
        training_data[qid] = rows

    if total_llm_candidates == 0 and not allow_gold_only:
        raise RuntimeError(
            "No LLM-generated candidates were produced. "
            "Refusing to save a gold-only training dataset. "
            "Check INFINEON_API_URL / INFINEON_API_KEY and rerun."
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n===== TRAINING DATA SUMMARY =====")
    print(f"Questions included: {len(training_data)}")
    print(f"Total candidates:   {total_candidates}")
    print(f"LLM candidates:     {total_llm_candidates}")
    print(f"LLM run failures:   {total_generation_failures}/{total_generation_runs}")
    print(
        f"Repairs:            enabled={repair_invalid} "
        f"attempted={total_repair_attempts} succeeded={total_repair_successes}"
    )
    if total_candidates > 0:
        print(
            f"Correct candidates: {total_correct} "
            f"({(total_correct / total_candidates) * 100:.1f}%)"
        )
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build candidate-level labeled training data for Infineon ranker."
    )
    parser.add_argument(
        "--dataset",
        default="data/infineon/infineon_dataset_100.json",
        help="Benchmark dataset with question/query/ambiguity_label.",
    )
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument(
        "--out",
        default="ranking/infineon_training_data_100.json",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument(
        "--allow-gold-only",
        action="store_true",
        help="Allow writing output even when zero LLM candidates were generated (debug only).",
    )
    parser.add_argument(
        "--no-repair-invalid",
        action="store_true",
        help="Disable execution-guided repair for invalid generated candidates.",
    )
    parser.add_argument(
        "--repair-max-candidates",
        type=int,
        default=2,
        help="Maximum invalid candidates to repair per generation run.",
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="Repair attempts per invalid candidate.",
    )
    args = parser.parse_args()

    build_training_data(
        dataset_path=args.dataset,
        graph_path=args.graph,
        schema_path=args.schema,
        output_path=args.out,
        k=args.k,
        n_runs=args.n_runs,
        allow_gold_only=args.allow_gold_only,
        repair_invalid=not args.no_repair_invalid,
        repair_max_candidates=max(0, int(args.repair_max_candidates)),
        repair_attempts=max(0, int(args.repair_attempts)),
    )


if __name__ == "__main__":
    main()
