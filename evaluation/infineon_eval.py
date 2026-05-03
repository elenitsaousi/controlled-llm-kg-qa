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
from kg.entity_linking import (
    build_entity_alias_index,
    canonicalize_question_with_index,
)
from llm.candidate_generation import generate_candidates, repair_candidate_query
from llm.client import InfineonGPTClient
from pipeline.qa import _rerank_with_semantic_coverage
from ranking.ambiguity_policy import (
    AmbiguityConfig,
    load_ambiguity_config,
    predict_regime,
)
from ranking.ranker import rank_candidates as rank_schema_candidates
from validation.semantic import rank_candidates_by_semantic_judge

DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""

NP_MODEL_TYPE = "np_tfidf_logreg_v1"
DEFAULT_QUERY_PLAN_MODEL = os.path.join(
    PROJECT_ROOT, "ranking", "models", "infineon_query_plan_predictor.json"
)
_RANKER_CACHE: Dict[str, object] = {}
_QUERY_PLAN_PREDICTOR_CACHE: Dict[str, object] = {}

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


def _query_cache_key(query: str, timeout_s: Optional[float]) -> Tuple[Optional[float], str]:
    return timeout_s, " ".join(_ensure_prefixes(_strip_comments(query)).split()).lower()


def _run_query_cached(
    graph: Graph,
    query: str,
    timeout_s: Optional[float],
    cache: Optional[Dict[Tuple[Optional[float], str], Tuple[str, object]]] = None,
) -> Counter:
    if cache is None:
        return _run_query(graph, query, timeout_s)

    key = _query_cache_key(query, timeout_s)
    cached = cache.get(key)
    if cached is not None:
        status, payload = cached
        if status == "ok":
            return Counter(payload)  # defensive copy
        if status == "timeout":
            raise QueryTimeout(str(payload))
        raise RuntimeError(str(payload))

    try:
        result = _run_query(graph, query, timeout_s)
    except QueryTimeout as exc:
        cache[key] = ("timeout", str(exc))
        raise
    except Exception as exc:
        cache[key] = ("error", str(exc))
        raise

    cache[key] = ("ok", Counter(result))
    return Counter(result)


def _dedupe_candidates(candidates: List[Dict]) -> Tuple[List[Dict], int]:
    seen = set()
    deduped: List[Dict] = []
    duplicate_count = 0
    for cand in candidates:
        query = _strip_comments(str(cand.get("query", "")).strip())
        if not query:
            continue
        key = " ".join(query.split()).lower()
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        item = dict(cand)
        item["query"] = query
        deduped.append(item)
    return deduped, duplicate_count

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


def _normalize_amb_label(label: str) -> str:
    x = (label or "").strip().lower()
    if x == "medium":
        return "mid"
    return x


def _parse_amb_regimes(text: Optional[str]) -> List[str]:
    if not text:
        return []
    allowed = {"low", "mid", "high"}
    out: List[str] = []
    for tok in text.split(","):
        lab = _normalize_amb_label(tok)
        if not lab:
            continue
        if lab not in allowed:
            raise ValueError(f"Invalid ambiguity label '{tok}'. Allowed: low,mid,high")
        if lab not in out:
            out.append(lab)
    return out


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


def _load_query_plan_predictor(model_path: str = DEFAULT_QUERY_PLAN_MODEL):
    if not model_path or not os.path.exists(model_path):
        return None
    cached = _QUERY_PLAN_PREDICTOR_CACHE.get(model_path)
    if cached is not None:
        return cached
    try:
        from ranking.np_tfidf_ranker import QueryPlanPredictor
        predictor = QueryPlanPredictor.load(model_path)
    except Exception:
        return None
    _QUERY_PLAN_PREDICTOR_CACHE[model_path] = predictor
    return predictor


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


def _schema_intent_rank_candidates(
    candidates: List[Dict],
    question: str,
    schema: KGSchema,
) -> List[Dict]:
    """Rank candidates with the same schema+intent ordering used by the UI."""
    try:
        ranked = _rerank_with_semantic_coverage(
            question,
            rank_schema_candidates(candidates, schema),
        )
    except Exception:
        return candidates
    by_key = {
        " ".join(str(c.get("query", "")).split()).lower(): c
        for c in candidates
    }
    ordered: List[Dict] = []
    seen = set()
    for row in ranked:
        key = " ".join(str(row.get("query", "")).split()).lower()
        original = by_key.get(key)
        if original is None or key in seen:
            continue
        item = dict(original)
        item["schema_intent_score"] = float(row.get("score", 0.0))
        item["schema_score"] = float(row.get("base_score", row.get("score", 0.0)))
        item["intent_score"] = float(row.get("intent_score", 0.0))
        item["intent_matched"] = list(row.get("intent_matched", []))
        item["intent_missing"] = list(row.get("intent_missing", []))
        ordered.append(item)
        seen.add(key)
    for item in candidates:
        key = " ".join(str(item.get("query", "")).split()).lower()
        if key not in seen:
            ordered.append(item)
            seen.add(key)
    return ordered


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(min_value, value)


def _validate_candidate_query(
    graph: Graph,
    query: str,
    timeout_s: Optional[float],
    query_cache: Optional[Dict[Tuple[Optional[float], str], Tuple[str, object]]] = None,
) -> Tuple[bool, Optional[str]]:
    try:
        _run_query_cached(graph, _ensure_prefixes(query), timeout_s, query_cache)
        return True, None
    except QueryTimeout as exc:
        return False, f"timeout: {exc}"
    except Exception as exc:
        return False, str(exc)


def _repair_invalid_candidates(
    question: str,
    schema: KGSchema,
    candidates: List[Dict],
    llm_client: object,
    graph: Graph,
    query_timeout: Optional[float],
    query_cache: Optional[Dict[Tuple[Optional[float], str], Tuple[str, object]]],
    max_repaired_candidates: int,
    max_attempts_per_candidate: int,
) -> Tuple[List[Dict], int, int]:
    if not candidates or max_repaired_candidates <= 0 or max_attempts_per_candidate <= 0:
        return candidates, 0, 0

    repaired_attempted = 0
    repaired_succeeded = 0
    updated: List[Dict] = []

    for cand in candidates:
        query = _strip_comments(str(cand.get("query", "")).strip())
        if not query:
            continue

        is_valid, err = _validate_candidate_query(graph, query, query_timeout, query_cache)
        if is_valid:
            c = dict(cand)
            c["query"] = query
            updated.append(c)
            continue

        if repaired_attempted >= max_repaired_candidates:
            c = dict(cand)
            c["query"] = query
            updated.append(c)
            continue

        repaired_attempted += 1
        current_error = err or "unknown query error"
        repaired_query: Optional[str] = None

        for _ in range(max_attempts_per_candidate):
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

            proposal = _strip_comments(proposal.strip())
            valid_after, err_after = _validate_candidate_query(
                graph, proposal, query_timeout, query_cache
            )
            if valid_after:
                repaired_query = proposal
                repaired_succeeded += 1
                break
            current_error = err_after or "unknown query error"

        c = dict(cand)
        c["query"] = repaired_query or query
        updated.append(c)

    deduped, _ = _dedupe_candidates(updated)
    return deduped, repaired_attempted, repaired_succeeded


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
    generation_runs: int = 1,
    use_ml_ranking: bool = True,
    use_schema_ranking: bool = False,
    use_semantic_selection: bool = False,
    semantic_selection_margin: float = 1.25,
    ml_model_path: str = "ranking/models/infineon_ranker.joblib",
    ml_ambiguity_regimes: Optional[List[str]] = None,
    ambiguity_config_path: Optional[str] = None,
    enable_entity_linking: bool = True,
    entity_link_max_matches: int = 5,
) -> Dict[str, object]:
    allowed_regimes = {"low", "mid", "high"}
    normalized_regimes: List[str] = []
    for lab in (ml_ambiguity_regimes or []):
        nlab = _normalize_amb_label(str(lab))
        if not nlab:
            continue
        if nlab not in allowed_regimes:
            raise ValueError(f"Invalid ambiguity regime '{lab}'. Allowed: low,mid,high")
        if nlab not in normalized_regimes:
            normalized_regimes.append(nlab)

    g = Graph()
    g.parse(graph_path, format="turtle")
    questions = _load_questions(dataset_path)

    entity_alias_index = None
    if enable_entity_linking:
        entity_alias_index = build_entity_alias_index(g)

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

    ambiguity_config: Optional[AmbiguityConfig] = None
    ambiguity_model = None
    if ambiguity_config_path:
        ambiguity_config = load_ambiguity_config(ambiguity_config_path)
        if not normalized_regimes:
            normalized_regimes = list(ambiguity_config.ml_regimes)
        if ambiguity_config.entropy_source == "ml" and _is_np_model_file(ml_model_path):
            try:
                ambiguity_model = _load_np_ranker(ml_model_path)
            except Exception:
                ambiguity_model = None

    if progress:
        print(f"ML Ranking: {'✅ Enabled' if ml_ranking_enabled else '❌ Disabled'}")
        print(f"Schema/intent Ranking: {'✅ Enabled' if use_schema_ranking else '❌ Disabled'}")

    llm_client = _build_llm_client(llm, temperature)
    if llm_client is None:
        raise RuntimeError("No LLM client available.")

    repair_enabled = _env_bool("INFINEON_ENABLE_REPAIR", True)
    repair_max_candidates = _env_int("INFINEON_REPAIR_MAX_CANDIDATES", 2, min_value=0)
    repair_attempts = _env_int("INFINEON_REPAIR_ATTEMPTS", 1, min_value=0)
    query_plan_predictor = _load_query_plan_predictor()

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
        "llm_generation_failures": 0,
        "generation_runs_requested": int(max(1, generation_runs)),
        "repair_enabled": repair_enabled,
        "repair_max_candidates": repair_max_candidates,
        "repair_attempts_per_candidate": repair_attempts,
        "repair_candidates_attempted": 0,
        "repair_candidates_succeeded": 0,
        "llm": llm,
        "temperature": temperature,
        "schema_path": schema_path,
        "query_timeout": query_timeout,
        "ml_ranking": ml_ranking_enabled,
        "schema_ranking": bool(use_schema_ranking),
        "semantic_selection": bool(use_semantic_selection),
        "semantic_selection_margin": float(semantic_selection_margin),
        "ml_ambiguity_regimes": normalized_regimes,
        "ambiguity_config_path": ambiguity_config_path,
        "predicted_regime_counts": {},
        "per_ambiguity": {},
        "entity_linking_enabled": bool(enable_entity_linking),
        "entity_link_max_matches": int(max(1, entity_link_max_matches)),
        "entity_linked_questions": 0,
        "query_plan_predictor": bool(query_plan_predictor),
        "execution_cache_entries": 0,
        "candidate_duplicates_removed": 0,
    }

    details = []
    total_questions = len(questions)
    query_cache: Dict[Tuple[Optional[float], str], Tuple[str, object]] = {}
    candidate_duplicates_removed = 0

    for idx, item in enumerate(questions):
        if progress:
            qid = item.get("id", "")
            question = item.get("question", "")
            print(f"[{idx + 1}/{total_questions}] {qid} - {question}", flush=True)

        summary["total"] += 1
        qid = item.get("id", "")
        question = item.get("question", "")
        canon = canonicalize_question_with_index(
            question,
            index=entity_alias_index,
            max_matches=max(1, int(entity_link_max_matches)),
        )
        effective_question = canon.effective_question or question
        entity_mappings = canon.mappings
        if canon.changed:
            summary["entity_linked_questions"] += 1
        gold_query = _strip_comments(str(item.get("query", "")).strip())
        ambiguity_label = _normalize_amb_label(str(item.get("ambiguity_label", "")))
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
            gold_sig = _run_query_cached(g, gold_full, query_timeout, query_cache)
        except QueryTimeout as exc:
            summary["gold_timeout"] += 1
            if amb_summary is not None:
                amb_summary["gold_timeout"] += 1
            details.append(
                {
                    "id": qid,
                    "question": question,
                    "effective_question": effective_question,
                    "entity_mappings": entity_mappings,
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
                    "effective_question": effective_question,
                    "entity_mappings": entity_mappings,
                    "ambiguity_label": ambiguity_label,
                    "gold_error": str(exc),
                }
            )
            continue

        # Generate candidates (optionally multiple generation runs, merged uniquely).
        candidates = []
        generation_errors = []
        seen_gen = set()
        ranking_question = effective_question
        for gen_run in range(max(1, int(generation_runs))):
            try:
                generated = generate_candidates(
                    question,
                    schema,
                    k=k,
                    llm_client=llm_client,
                    entity_alias_index=entity_alias_index,
                    max_entity_links=max(1, int(entity_link_max_matches)),
                    query_plan_predictor=query_plan_predictor,
                )
                batch = generated.get("candidates", [])
                metadata = generated.get("metadata", {})
                q_eff = str(metadata.get("effective_question", "")).strip()
                if q_eff:
                    ranking_question = q_eff
            except Exception as exc:
                summary["llm_generation_failures"] += 1
                generation_errors.append(str(exc))
                continue

            for cand in batch:
                qtext = _strip_comments(str(cand.get("query", "")).strip())
                if not qtext:
                    continue
                key = " ".join(qtext.split()).lower()
                if key in seen_gen:
                    continue
                seen_gen.add(key)
                c = dict(cand)
                c["query"] = qtext
                candidates.append(c)

        if not candidates:
            summary["top1_invalid"] += 1
            summary["all_invalid"] += 1
            details.append(
                {
                    "id": qid,
                    "question": question,
                    "effective_question": ranking_question,
                    "entity_mappings": entity_mappings,
                    "ambiguity_label": ambiguity_label,
                    "generation_error": (
                        "; ".join(generation_errors)
                        if generation_errors
                        else "No candidates generated"
                    ),
                    "candidates": [],
                    "top1_correct": False,
                    "any_correct": False,
                }
            )
            continue

        if repair_enabled and candidates:
            candidates, rep_attempted, rep_succeeded = _repair_invalid_candidates(
                question=ranking_question,
                schema=schema,
                candidates=candidates,
                llm_client=llm_client,
                graph=g,
                query_timeout=query_timeout,
                query_cache=query_cache,
                max_repaired_candidates=repair_max_candidates,
                max_attempts_per_candidate=repair_attempts,
            )
            summary["repair_candidates_attempted"] += rep_attempted
            summary["repair_candidates_succeeded"] += rep_succeeded
        candidates, duplicates_removed = _dedupe_candidates(candidates)
        candidate_duplicates_removed += duplicates_removed

        predicted_regime = None
        predicted_entropy = None
        regime_for_policy = ambiguity_label
        if ambiguity_config is not None:
            try:
                cand_payload = []
                for c in candidates:
                    row = {"query": str(c.get("query", ""))}
                    feats = c.get("features")
                    if isinstance(feats, dict) and feats:
                        row["features"] = feats
                    cand_payload.append(row)
                predicted_regime, predicted_entropy = predict_regime(
                    question=ranking_question,
                    candidates=cand_payload,
                    config=ambiguity_config,
                    schema_dict=schema_dict,
                    model=ambiguity_model,
                    graph=g if ambiguity_config.entropy_source == "agreement" else None,
                )
                regime_for_policy = predicted_regime
                summary["predicted_regime_counts"][predicted_regime] = (
                    int(summary["predicted_regime_counts"].get(predicted_regime, 0)) + 1
                )
            except Exception:
                predicted_regime = None
                predicted_entropy = None

        # ML Ranking (all, label-gated, or config-gated)
        apply_ml_ranking = ml_ranking_enabled
        if apply_ml_ranking and normalized_regimes:
            apply_ml_ranking = regime_for_policy in set(normalized_regimes)

        if apply_ml_ranking and candidates:
            candidates = _ml_rank_candidates(
                candidates, ranking_question, schema_dict, ml_model_path
            )
        elif use_schema_ranking and candidates:
            candidates = _schema_intent_rank_candidates(candidates, ranking_question, schema)
        elif use_semantic_selection and candidates:
            candidates = rank_candidates_by_semantic_judge(
                ranking_question,
                candidates,
                min_margin=semantic_selection_margin,
            )

        top1_correct = False
        any_correct = False
        any_valid = False
        candidate_results = []

        from pipeline.qa import _select_best_candidate_semantic

        if use_semantic_selection and candidates:
            best = _select_best_candidate_semantic(candidates, ranking_question)

            if best:
                best_query = best.get("query")
                # Βάζουμε το best πρώτο
                candidates = [best] + [
                    c for c in candidates
                    if str(c.get("query")).strip() != best_query
                ]


        for c_idx, c in enumerate(candidates):
            summary["total_candidates"] += 1
            cand_query = _strip_comments(str(c.get("query", "")).strip())
            cand_full = _ensure_prefixes(cand_query)

            try:
                cand_sig = _run_query_cached(g, cand_full, query_timeout, query_cache)
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
                "ml_score": c.get("ml_score"),
                "semantic_judge_score": c.get("semantic_judge_score"),
                "semantic_judge_report": c.get("semantic_judge_report"),
                "source": c.get("source"),
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
            "effective_question": ranking_question,
            "entity_mappings": entity_mappings,
            "ambiguity_label": ambiguity_label,
            "policy_regime": regime_for_policy,
            "predicted_regime": predicted_regime,
            "predicted_entropy": predicted_entropy,
            "ml_applied": apply_ml_ranking,
            "schema_ranking_applied": bool(use_schema_ranking and not apply_ml_ranking),
            "semantic_selection_applied": bool(
                use_semantic_selection and not apply_ml_ranking and not use_schema_ranking
            ),
            "top1_correct": top1_correct,
            "any_correct": any_correct,
            "candidates": candidate_results,
        })

    summary["top1_correct_rate"] = summary["top1_correct"] / summary["total"]
    summary["any_correct_rate"] = summary["any_correct"] / summary["total"]
    summary["execution_cache_entries"] = len(query_cache)
    summary["candidate_duplicates_removed"] = candidate_duplicates_removed

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
    parser.add_argument("--dataset", default="data/infineon/infineon_test_final.json")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--llm", default="auto",
                        choices=["auto", "infineon"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--query-timeout", type=float, default=None)
    parser.add_argument("--generation-runs", type=int, default=1)
    parser.add_argument("--out", default="results/infineon_eval.json")
    parser.add_argument("--no-ml-ranking", action="store_true",
                        help="Disable ML ranking")
    parser.add_argument(
        "--use-schema-ranking",
        action="store_true",
        help="Enable schema/intent candidate ranking when ML ranking is not applied.",
    )
    parser.add_argument(
        "--use-semantic-selection",
        action="store_true",
        help="Enable conservative semantic candidate selection when ML/schema ranking is not applied.",
    )
    parser.add_argument(
        "--semantic-selection-margin",
        type=float,
        default=1.25,
        help="Minimum semantic score margin required to override the first candidate.",
    )
    parser.add_argument("--ml-model",
                        default="ranking/models/infineon_ranker.joblib",
                        help="Path to ML ranking model")
    parser.add_argument(
        "--ml-ambiguity-regimes",
        default="",
        help="Comma-separated ambiguity labels where ML ranking is used (e.g. mid or low,mid).",
    )
    parser.add_argument(
        "--ambiguity-config",
        default="",
        help="Optional ambiguity config JSON for runtime regime prediction.",
    )
    parser.add_argument(
        "--no-entity-linking",
        action="store_true",
        help="Disable entity canonicalization before candidate generation.",
    )
    parser.add_argument(
        "--entity-link-max-matches",
        type=int,
        default=5,
        help="Maximum entity mentions to canonicalize per question.",
    )
    args = parser.parse_args()

    regimes = _parse_amb_regimes(args.ml_ambiguity_regimes)
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
        generation_runs=max(1, int(args.generation_runs)),
        use_ml_ranking=not args.no_ml_ranking,
        use_schema_ranking=args.use_schema_ranking,
        use_semantic_selection=args.use_semantic_selection,
        semantic_selection_margin=args.semantic_selection_margin,
        ml_model_path=args.ml_model,
        ml_ambiguity_regimes=regimes,
        ambiguity_config_path=(args.ambiguity_config or None),
        enable_entity_linking=not args.no_entity_linking,
        entity_link_max_matches=max(1, int(args.entity_link_max_matches)),
    )

    summary = results["summary"]
    print("===== SUMMARY =====")
    print(f"Total: {summary['total']}")
    print(f"ML Ranking: {summary['ml_ranking']}")
    print(f"Schema Ranking: {summary.get('schema_ranking')}")
    print(f"Semantic Selection: {summary.get('semantic_selection')}")
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
    print(f"Execution cache entries: {summary.get('execution_cache_entries', 0)}")
    print(f"Duplicate candidates removed: {summary.get('candidate_duplicates_removed', 0)}")
    print("\n===== TOP1 BEHAVIOR =====")
    print(f"Top1 correct: {summary['top1_correct']}")
    print(f"Top1 valid wrong: {summary['top1_valid_wrong']}")
    print(f"Top1 invalid: {summary['top1_invalid']}")


if __name__ == "__main__":
    main()
