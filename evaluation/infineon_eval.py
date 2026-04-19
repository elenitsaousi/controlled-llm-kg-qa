import argparse
import json
import os
import signal
import sys
from collections import Counter
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
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

from kg.schema import KGSchema, load_default_schema, load_schema
from llm.candidate_generation import generate_candidates
from llm.client import InfineonGPTClient

DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

NP_MODEL_TYPE = "np_tfidf_logreg_v1"
_RANKER_CACHE: Dict[str, object] = {}

def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query

def _strip_comments(query: str) -> str:
    cleaned_lines = []
    for line in query.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()

def _normalize_term(term) -> str:
    if term is None:
        return "NULL"
    try:
        return term.n3()
    except Exception:
        return str(term)

def _result_signature(rows: List[Tuple]) -> Counter:
    return Counter(tuple(_normalize_term(v) for v in row) for row in rows)

class QueryTimeout(Exception):
    pass

@contextmanager
def _time_limit(seconds: Optional[float]):
    if not seconds or seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    def _handler(signum, frame):
        raise QueryTimeout(f"Query exceeded {seconds} seconds")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

def _run_query(graph: Graph, query: str, timeout_s: Optional[float]) -> Counter:
    with _time_limit(timeout_s):
        results = graph.query(query)
        return _result_signature(list(results))

def _load_questions(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_llm_client(choice: str, temperature: float):
    choice = (choice or "auto").lower()
    if choice == "auto":
        choice = (os.environ.get("LLM_PROVIDER") or os.environ.get("LLM_BACKEND") or "infineon").strip().lower()
    if choice == "infiineon":
        choice = "infineon"
    if choice == "infineon":
        if not os.environ.get("INFINEON_API_URL") or not os.environ.get("INFINEON_API_KEY"):
            raise RuntimeError("Missing INFINEON_API_URL or INFINEON_API_KEY.")
        return InfineonGPTClient(temperature=temperature)
    if choice == "none":
        return None
    raise RuntimeError(
        f"Unsupported LLM backend '{choice}'. Supported backend: infineon."
    )


def _is_np_model_file(model_path: str) -> bool:
    if not model_path.lower().endswith(".json"):
        return False
    if not os.path.exists(model_path):
        return False
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("model_type") == NP_MODEL_TYPE
    except Exception:
        return False


def _load_np_ranker(model_path: str):
    ranker = _RANKER_CACHE.get(model_path)
    if ranker is not None:
        return ranker
    from ranking.np_tfidf_ranker import NPTfidfRanker
    ranker = NPTfidfRanker.load(model_path)
    _RANKER_CACHE[model_path] = ranker
    return ranker


def _ml_dependencies_available(model_path: str) -> bool:
    if _is_np_model_file(model_path):
        try:
            _load_np_ranker(model_path)
            return True
        except Exception:
            return False
    try:
        from ranking.runtime_ranker import LogisticRanker
        from ranking.feature_extraction import extract_features  # noqa: F401
        ranker = _RANKER_CACHE.get(model_path)
        if ranker is None:
            ranker = LogisticRanker(model_path)
            _RANKER_CACHE[model_path] = ranker
        return True
    except Exception:
        return False


def _ml_rank_candidates(
    candidates: List[Dict],
    question: str,
    schema_dict: dict,
    model_path: str
) -> List[Dict]:
    """Rank candidates using ML model. Returns reranked candidates."""
    if _is_np_model_file(model_path):
        try:
            from ranking.np_tfidf_ranker import rank_candidates_with_model
            ranker = _load_np_ranker(model_path)
            return rank_candidates_with_model(
                ranker,
                question,
                candidates,
                schema_dict,
            )
        except Exception:
            return candidates

    try:
        from ranking.runtime_ranker import LogisticRanker
        from ranking.feature_extraction import extract_features

        ranker = _RANKER_CACHE.get(model_path)
        if ranker is None:
            ranker = LogisticRanker(model_path)
            _RANKER_CACHE[model_path] = ranker
        feature_dicts = []
        for c in candidates:
            try:
                feats = extract_features(question, c["query"], schema_dict)
            except Exception:
                feats = {name: 0.0 for name in ranker.feature_names}
            feature_dicts.append(feats)

        if not feature_dicts:
            return candidates

        scores = ranker.score(feature_dicts)
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        return [c for _, c in ranked]
    except Exception:
        return candidates  # fallback to original order

def evaluate(
    dataset_path: str,
    graph_path: str,
    k: int = 3,
    schema: Optional[KGSchema] = None,
    schema_path: Optional[str] = None,
    out_path: Optional[str] = None,
    llm: str = "auto",
    temperature: float = 0.2,
    progress: bool = False,
    query_timeout: Optional[float] = None,
    use_ml_ranking: bool = True,
    ml_model_path: str = "ranking/models/infineon_ranker.joblib",
) -> Dict[str, object]:

    g = Graph()
    g.parse(graph_path, format="turtle")
    questions = _load_questions(dataset_path)

    if schema is None:
        if schema_path:
            schema = load_schema(schema_path)
        else:
            schema = load_default_schema()

    # Load schema dict for ML ranking
    _schema_path = schema_path or "data/infineon/schema.json"
    with open(_schema_path) as f:
        schema_dict = json.load(f)

    # Check ML ranking availability
    ml_ranking_enabled = (
        use_ml_ranking
        and os.path.exists(ml_model_path)
        and _ml_dependencies_available(ml_model_path)
    )
    if progress:
        print(f"ML Ranking: {'✅ Enabled' if ml_ranking_enabled else '❌ Disabled'}")

    llm_client = _build_llm_client(llm, temperature)
    if llm_client is None:
        raise RuntimeError("No LLM client available.")

    summary = {
        "total": 0,
        "gold_invalid": 0,
        "gold_timeout": 0,
        "top1_correct": 0,
        "top1_valid_wrong": 0,
        "top1_invalid": 0,
        "any_correct": 0,
        "total_candidates": 0,
        "correct_candidates": 0,
        "valid_wrong_candidates": 0,
        "invalid_candidates": 0,
        "candidate_timeouts": 0,
        "all_invalid": 0,
        "all_valid_wrong": 0,
        "llm": llm,
        "temperature": temperature,
        "schema_path": schema_path,
        "query_timeout": query_timeout,
        "ml_ranking": ml_ranking_enabled,
        "per_ambiguity": {},
    }

    details = []
    total_questions = len(questions)

    for idx, item in enumerate(questions):
        if progress:
            qid = item.get("id", "")
            question = item.get("question", "")
            print(f"[{idx + 1}/{total_questions}] {qid} - {question}", flush=True)

        summary["total"] += 1
        qid = item.get("id", "")
        question = item.get("question", "")
        gold_query = _strip_comments(str(item.get("query", "")).strip())
        ambiguity_label = str(item.get("ambiguity_label", "")).strip().lower()
        if not ambiguity_label:
            ambiguity_label = None

        amb_summary = None
        if ambiguity_label is not None:
            amb_summary = summary["per_ambiguity"].setdefault(
                ambiguity_label,
                {
                    "total": 0,
                    "gold_invalid": 0,
                    "gold_timeout": 0,
                    "top1_correct": 0,
                    "any_correct": 0,
                },
            )
            amb_summary["total"] += 1

        gold_full = _ensure_prefixes(gold_query)

        try:
            gold_sig = _run_query(g, gold_full, query_timeout)
        except QueryTimeout as exc:
            summary["gold_timeout"] += 1
            if amb_summary is not None:
                amb_summary["gold_timeout"] += 1
            details.append(
                {
                    "id": qid,
                    "question": question,
                    "ambiguity_label": ambiguity_label,
                    "gold_error": str(exc),
                }
            )
            continue
        except Exception as exc:
            summary["gold_invalid"] += 1
            if amb_summary is not None:
                amb_summary["gold_invalid"] += 1
            details.append(
                {
                    "id": qid,
                    "question": question,
                    "ambiguity_label": ambiguity_label,
                    "gold_error": str(exc),
                }
            )
            continue

        # Generate candidates
        generated = generate_candidates(question, schema, k=k, llm_client=llm_client)
        candidates = generated.get("candidates", [])

        # ML Ranking
        if ml_ranking_enabled and candidates:
            candidates = _ml_rank_candidates(
                candidates, question, schema_dict, ml_model_path
            )

        top1_correct = False
        any_correct = False
        any_valid = False
        candidate_results = []

        for c_idx, c in enumerate(candidates):
            summary["total_candidates"] += 1
            cand_query = _strip_comments(str(c.get("query", "")).strip())
            cand_full = _ensure_prefixes(cand_query)

            try:
                cand_sig = _run_query(g, cand_full, query_timeout)
                any_valid = True
                is_correct = cand_sig == gold_sig
            except QueryTimeout as exc:
                summary["invalid_candidates"] += 1
                summary["candidate_timeouts"] += 1
                candidate_results.append({
                    "index": c_idx,
                    "label": "timeout",
                    "error": str(exc),
                    "query": cand_query,
                })
                continue
            except Exception as exc:
                summary["invalid_candidates"] += 1
                candidate_results.append({
                    "index": c_idx,
                    "label": "invalid",
                    "error": str(exc),
                    "query": cand_query,
                })
                continue

            if is_correct:
                summary["correct_candidates"] += 1
                any_correct = True
                if c_idx == 0:
                    top1_correct = True
                label = "correct"
            else:
                summary["valid_wrong_candidates"] += 1
                label = "valid_wrong"

            candidate_results.append({
                "index": c_idx,
                "label": label,
                "query": cand_query,
            })

        if candidate_results:
            top1_label = candidate_results[0]["label"]
            if top1_label in ("invalid", "timeout"):
                summary["top1_invalid"] += 1
            elif top1_label == "valid_wrong":
                summary["top1_valid_wrong"] += 1

        if top1_correct:
            summary["top1_correct"] += 1
        if any_correct:
            summary["any_correct"] += 1
        if amb_summary is not None:
            amb_summary["top1_correct"] += int(top1_correct)
            amb_summary["any_correct"] += int(any_correct)
        if not any_valid:
            summary["all_invalid"] += 1
        if any_valid and not any_correct:
            summary["all_valid_wrong"] += 1

        details.append({
            "id": qid,
            "question": question,
            "ambiguity_label": ambiguity_label,
            "top1_correct": top1_correct,
            "any_correct": any_correct,
            "candidates": candidate_results,
        })

    summary["top1_correct_rate"] = summary["top1_correct"] / summary["total"]
    summary["any_correct_rate"] = summary["any_correct"] / summary["total"]
    if summary["total_candidates"] > 0:
        summary["candidate_correct_rate"] = (
            summary["correct_candidates"] / summary["total_candidates"]
        )
        summary["candidate_invalid_rate"] = (
            summary["invalid_candidates"] / summary["total_candidates"]
        )
    else:
        summary["candidate_correct_rate"] = 0.0
        summary["candidate_invalid_rate"] = 0.0

    for label, stats in summary["per_ambiguity"].items():
        denom = stats["total"] if stats["total"] > 0 else 1
        stats["top1_correct_rate"] = stats["top1_correct"] / denom
        stats["any_correct_rate"] = stats["any_correct"] / denom

    payload = {"summary": summary, "details": details}

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Infineon questions."
    )
    parser.add_argument("--dataset", default="data/infineon/infineon_dataset_30.json")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--llm", default="auto",
                        choices=["auto", "infineon"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--query-timeout", type=float, default=None)
    parser.add_argument("--out", default="results/infineon_eval.json")
    parser.add_argument("--no-ml-ranking", action="store_true",
                        help="Disable ML ranking")
    parser.add_argument("--ml-model",
                        default="ranking/models/infineon_ranker.joblib",
                        help="Path to ML ranking model")
    args = parser.parse_args()

    results = evaluate(
        dataset_path=args.dataset,
        graph_path=args.graph,
        k=args.k,
        schema_path=args.schema,
        out_path=args.out,
        llm=args.llm,
        temperature=args.temperature,
        progress=args.progress,
        query_timeout=args.query_timeout,
        use_ml_ranking=not args.no_ml_ranking,
        ml_model_path=args.ml_model,
    )

    summary = results["summary"]
    print("===== SUMMARY =====")
    print(f"Total: {summary['total']}")
    print(f"ML Ranking: {summary['ml_ranking']}")
    print(f"Gold invalid: {summary['gold_invalid']}")
    print(f"Gold timeout: {summary['gold_timeout']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.2%})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.2%})")
    print(f"All invalid: {summary['all_invalid']}")
    print(f"All valid wrong: {summary['all_valid_wrong']}")
    print(f"Saved: {args.out}")
    print("\n===== CANDIDATE STATS =====")
    print(f"Total candidates: {summary['total_candidates']}")
    print(f"Correct: {summary['correct_candidates']}")
    print(f"Valid wrong: {summary['valid_wrong_candidates']}")
    print(f"Invalid: {summary['invalid_candidates']}")
    print(f"Timeout: {summary['candidate_timeouts']}")
    print("\n===== TOP1 BEHAVIOR =====")
    print(f"Top1 correct: {summary['top1_correct']}")
    print(f"Top1 valid wrong: {summary['top1_valid_wrong']}")
    print(f"Top1 invalid: {summary['top1_invalid']}")


if __name__ == "__main__":
    main()
