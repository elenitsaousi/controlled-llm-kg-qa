#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
from llm.candidate_generation import generate_candidates
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


def build_training_data(
    dataset_path: str,
    graph_path: str,
    schema_path: str,
    output_path: str,
    k: int = 5,
    n_runs: int = 3,
    allow_gold_only: bool = False,
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
                generated = generate_candidates(question, schema, k=k)
                candidates = generated.get("candidates", [])
            except Exception as exc:
                print(f"    ❌ Candidate generation failed: {exc}")
                total_generation_failures += 1
                continue

            for cand_idx, cand in enumerate(candidates):
                query = str(cand.get("query", "")).strip()
                if not query:
                    continue
                if query in seen_queries:
                    continue
                seen_queries.add(query)

                full_query = _ensure_prefixes(query)
                try:
                    sig = _result_signature(g.query(full_query))
                    is_valid = 1
                    is_correct = int(sig == gold_rows and len(sig) > 0)
                except Exception:
                    is_valid = 0
                    is_correct = 0

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
    args = parser.parse_args()

    build_training_data(
        dataset_path=args.dataset,
        graph_path=args.graph,
        schema_path=args.schema,
        output_path=args.out,
        k=args.k,
        n_runs=args.n_runs,
        allow_gold_only=args.allow_gold_only,
    )


if __name__ == "__main__":
    main()
