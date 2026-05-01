# pipeline/qa.py

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery

from kg.executor import execute_query_stub
from kg.entity_linking import build_entity_alias_index
from kg.schema import KGSchema, load_default_schema
from kg.sparql_matching import is_relaxed_correct
from kg.sparql_normalization import normalize_sparql
from llm.answer_synthesis import synthesize_answer
from llm.candidate_generation import generate_candidates
from ranking.ambiguity_policy import (
    AmbiguityConfig,
    load_ambiguity_config,
    predict_regime,
)
from ranking.feature_extraction import extract_features
from ranking.ranker import rank_candidates as rank_schema_candidates
from validation.semantic import semantic_coverage_report, validate_query_semantic
from validation.syntax import validate_query_syntax
from visualization.ambiguity_metrics import ambiguity_entropy

BASE = Path(__file__).resolve().parents[1]
DEFAULT_LOGISTIC_MODEL = BASE / "ranking" / "models" / "logistic_ranker.joblib"
DEFAULT_INFINEON_JOBLIB_MODEL = BASE / "ranking" / "models" / "infineon_ranker.joblib"
DEFAULT_INFINEON_NP_MODEL = BASE / "ranking" / "models" / "infineon_np_tfidf_ranker_entitylink.json"
DEFAULT_INFINEON_NP_MODEL_FALLBACK = BASE / "ranking" / "models" / "infineon_np_tfidf_ranker.json"
DEFAULT_AMBIGUITY_CONFIG = BASE / "ranking" / "models" / "infineon_ambiguity_config.json"
DEFAULT_AMBIGUITY_CONFIG_500 = BASE / "ranking" / "models" / "infineon_ambiguity_config_500.json"
DEFAULT_QUERY_PLAN_MODEL = BASE / "ranking" / "models" / "infineon_query_plan_predictor.json"
DEFAULT_INFINEON_SCHEMA_PATH = BASE / "data" / "infineon" / "schema.json"
GATED_THRESHOLDS = BASE / "ranking" / "models" / "gated_thresholds.json"
EXPERIMENTS_DIR = BASE / "experiments"
DEFAULT_INFINEON_GRAPH = BASE / "data" / "infineon" / "graph.ttl"
DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""
_ENTITY_ALIAS_INDEX = None
_DEFAULT_GRAPH_CACHE: Optional[Graph] = None
_RANKER_CACHE: Dict[str, object] = {}
_AMBIGUITY_CONFIG_CACHE: Dict[str, AmbiguityConfig] = {}
_QUERY_PLAN_PREDICTOR_CACHE: Dict[str, object] = {}

NP_MODEL_TYPE = "np_tfidf_logreg_v1"


def _get_default_entity_alias_index():
    global _ENTITY_ALIAS_INDEX
    if _ENTITY_ALIAS_INDEX is not None:
        return _ENTITY_ALIAS_INDEX
    try:
        g = _get_default_graph()
        if g is None:
            _ENTITY_ALIAS_INDEX = None
            return None
        _ENTITY_ALIAS_INDEX = build_entity_alias_index(g)
    except Exception:
        _ENTITY_ALIAS_INDEX = None
    return _ENTITY_ALIAS_INDEX


def _get_default_graph() -> Optional[Graph]:
    global _DEFAULT_GRAPH_CACHE
    if _DEFAULT_GRAPH_CACHE is not None:
        return _DEFAULT_GRAPH_CACHE
    if not DEFAULT_INFINEON_GRAPH.exists():
        _DEFAULT_GRAPH_CACHE = None
        return None
    try:
        g = Graph()
        g.parse(str(DEFAULT_INFINEON_GRAPH), format="turtle")
        _DEFAULT_GRAPH_CACHE = g
        return g
    except Exception:
        _DEFAULT_GRAPH_CACHE = None
        return None


def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query


def _schema_to_dict(schema: KGSchema) -> Dict[str, object]:
    return {
        "description": schema.description,
        "classes": schema.classes,
        "predicates": schema.predicates,
        "properties": schema.properties,
        "labels": schema.labels,
        "relationships": schema.relationships,
        "allowed_property_filters": schema.allowed_property_filters,
        "notes": schema.notes,
    }


def _load_schema_dict_for_ranking(schema: KGSchema) -> Dict[str, object]:
    # Prefer explicit Infineon schema file when available.
    if DEFAULT_INFINEON_SCHEMA_PATH.exists():
        try:
            with open(DEFAULT_INFINEON_SCHEMA_PATH, "r", encoding="utf-8") as f:
                return dict(json.load(f))
        except Exception:
            pass
    return _schema_to_dict(schema)


def _is_np_model_file(model_path: Path) -> bool:
    if model_path.suffix.lower() != ".json":
        return False
    if not model_path.exists():
        return False
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("model_type") == NP_MODEL_TYPE
    except Exception:
        return False


def _resolve_learning_model_path(explicit_path: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            DEFAULT_INFINEON_NP_MODEL,
            DEFAULT_INFINEON_NP_MODEL_FALLBACK,
            DEFAULT_INFINEON_JOBLIB_MODEL,
            DEFAULT_LOGISTIC_MODEL,
        ]
    )
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_ambiguity_config_path(explicit_path: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = (os.getenv("INFINEON_AMBIGUITY_CONFIG") or "").strip()
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            DEFAULT_AMBIGUITY_CONFIG_500,
            DEFAULT_AMBIGUITY_CONFIG,
        ]
    )
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_ranker_cached(model_path: Path):
    key = str(model_path)
    cached = _RANKER_CACHE.get(key)
    if cached is not None:
        return cached

    if _is_np_model_file(model_path):
        from ranking.np_tfidf_ranker import NPTfidfRanker

        ranker = NPTfidfRanker.load(str(model_path))
        _RANKER_CACHE[key] = ranker
        return ranker

    from ranking.runtime_ranker import LogisticRanker

    ranker = LogisticRanker(str(model_path))
    _RANKER_CACHE[key] = ranker
    return ranker


def _load_query_plan_predictor_cached(model_path: Path = DEFAULT_QUERY_PLAN_MODEL):
    key = str(model_path)
    cached = _QUERY_PLAN_PREDICTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if not model_path.exists():
        return None
    try:
        from ranking.np_tfidf_ranker import QueryPlanPredictor

        predictor = QueryPlanPredictor.load(str(model_path))
    except Exception:
        return None
    _QUERY_PLAN_PREDICTOR_CACHE[key] = predictor
    return predictor


def _normalize_query(text: str) -> str:
    return normalize_sparql(text)


def _load_questions(questions_path: Path) -> List[Dict[str, object]]:
    with open(questions_path, "r", encoding="utf-8") as f:
        return list(json.load(f))


def _rank_learning_candidates(
    question: str,
    candidates: List[Dict[str, str]],
    schema: KGSchema,
    model_path: Optional[Path] = None,
) -> List[Dict[str, object]]:
    if not candidates:
        return []

    resolved_model_path = model_path or _resolve_learning_model_path(None)
    if resolved_model_path is None or not resolved_model_path.exists():
        return []

    try:
        ranker = _load_ranker_cached(resolved_model_path)
    except Exception:
        return []

    schema_dict = _load_schema_dict_for_ranking(schema)
    feature_dicts: List[Dict[str, float]] = []
    rows: List[Dict[str, object]] = []

    for cand in candidates:
        query = str(cand.get("query", ""))
        try:
            feats = extract_features(question, query, schema_dict)
        except Exception:
            feats = {}
        feature_dicts.append(feats)
        rows.append(
            {
                "query": query,
                "source": cand.get("source", "unknown"),
            }
        )

    try:
        if _is_np_model_file(resolved_model_path):
            queries = [str(c.get("query", "")) for c in candidates]
            scores = ranker.score_question_candidates(
                question,
                queries,
                feature_dicts,
            )
        else:
            # Logistic/XGB runtime rankers.
            from ranking.feature_config import FEATURE_NAMES
            normalized_features = []
            for feats in feature_dicts:
                normalized_features.append(
                    {name: float(feats.get(name, 0.0)) for name in FEATURE_NAMES}
                )
            scores = ranker.score(normalized_features)
    except Exception:
        return []

    for row, score in zip(rows, scores):
        row["score"] = float(score)
        row["model_path"] = str(resolved_model_path)

    rows.sort(key=lambda x: x.get("score", float("-inf")), reverse=True)
    return rows


def _compute_entropy(ranked: List[Dict[str, object]]) -> float:
    if not ranked:
        return 0.0
    scores = np.array(
        [float(c.get("score", 0.0)) for c in ranked], dtype=float
    )
    if scores.size == 0:
        return 0.0
    return float(ambiguity_entropy(scores))


def _rerank_with_semantic_coverage(
    question: str,
    ranked: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    adjusted: List[Dict[str, object]] = []
    for row in ranked:
        query = str(row.get("query", "") or "")
        report = semantic_coverage_report(question, query)
        coverage = float(report.get("coverage_score", 1.0))
        missing_count = int(report.get("missing_count", 0))
        base_score = float(row.get("score", 0.0))
        updated = dict(row)
        updated["base_score"] = base_score
        updated["coverage_score"] = coverage
        updated["coverage_missing_count"] = missing_count
        updated["coverage_missing"] = list(report.get("missing", []))
        updated["coverage_required"] = list(report.get("required", []))
        # Coverage is a hard semantic signal: valid-but-partial queries should
        # lose to candidates that cover the requested business concepts.
        updated["score"] = base_score + (2.0 * coverage) - (1.5 * missing_count)
        adjusted.append(updated)
    adjusted.sort(key=lambda x: x.get("score", float("-inf")), reverse=True)
    return adjusted


def _load_ambiguity_config_cached(path: Path) -> Optional[AmbiguityConfig]:
    key = str(path)
    cached = _AMBIGUITY_CONFIG_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        cfg = load_ambiguity_config(key)
    except Exception:
        return None
    _AMBIGUITY_CONFIG_CACHE[key] = cfg
    return cfg


def _predict_runtime_regime(
    question: str,
    candidates: List[Dict[str, str]],
    schema: KGSchema,
    model_path: Optional[Path],
    ambiguity_config: Optional[AmbiguityConfig],
) -> Tuple[Optional[str], Optional[float]]:
    if ambiguity_config is None or not candidates:
        return None, None

    schema_dict = _load_schema_dict_for_ranking(schema)
    ambiguity_model = None
    if (
        ambiguity_config.entropy_source == "ml"
        and model_path is not None
        and _is_np_model_file(model_path)
    ):
        try:
            ambiguity_model = _load_ranker_cached(model_path)
        except Exception:
            ambiguity_model = None

    graph = _get_default_graph() if ambiguity_config.entropy_source == "agreement" else None
    payload = [{"query": str(c.get("query", ""))} for c in candidates]
    try:
        regime, entropy = predict_regime(
            question=question,
            candidates=payload,
            config=ambiguity_config,
            schema_dict=schema_dict,
            model=ambiguity_model,
            graph=graph,
        )
        return regime, float(entropy)
    except Exception:
        return None, None


def _ordered_candidate_queries(
    primary: List[Dict[str, object]],
    fallback: List[Dict[str, object]],
) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for row in list(primary) + list(fallback):
        q = str(row.get("query", "")).strip()
        if not q:
            continue
        key = " ".join(q.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(q)
    return ordered


def _candidate_key(query: str) -> str:
    return " ".join((query or "").split()).lower()


def _runtime_validate_query(query: str) -> List[Dict[str, str]]:
    errors = []
    errors.extend(validate_query_syntax(query))
    errors.extend(validate_query_semantic(query))
    if errors:
        return errors
    try:
        parseQuery(_ensure_prefixes(query))
    except Exception as exc:
        errors.append({"type": "syntax", "message": str(exc)})
    return errors


def _query_runtime_profile(query: str, max_rows: int = 25) -> Dict[str, object]:
    graph = _get_default_graph()
    if graph is None:
        return {
            "has_rows": None,
            "row_count": 0,
            "unbound_vars": [],
            "error": "graph_unavailable",
        }
    try:
        results = graph.query(_ensure_prefixes(query))
        vars_seen = [str(v) for v in getattr(results, "vars", [])]
        bound_counts = {v: 0 for v in vars_seen}
        row_count = 0
        for row in results:
            if row_count >= max_rows:
                break
            row_count += 1
            if hasattr(row, "asdict"):
                rd = row.asdict()
                for var in vars_seen:
                    if rd.get(var) is not None:
                        bound_counts[var] = bound_counts.get(var, 0) + 1
            else:
                for var, value in zip(vars_seen, row):
                    if value is not None:
                        bound_counts[var] = bound_counts.get(var, 0) + 1
        unbound_vars = [
            var for var in vars_seen if row_count > 0 and bound_counts.get(var, 0) == 0
        ]
        return {
            "has_rows": row_count > 0,
            "row_count": row_count,
            "unbound_vars": unbound_vars,
            "error": None,
        }
    except Exception as exc:
        return {
            "has_rows": None,
            "row_count": 0,
            "unbound_vars": [],
            "error": str(exc),
        }


def _query_has_runtime_rows(query: str) -> Tuple[Optional[bool], Optional[str]]:
    profile = _query_runtime_profile(query)
    return profile.get("has_rows"), profile.get("error")


def _select_best_valid_query(
    ordered_queries: List[str],
) -> Tuple[Optional[str], List[Dict[str, str]], int]:
    if not ordered_queries:
        return None, [], -1

    first_valid: Optional[Tuple[str, int]] = None
    first_nonempty: Optional[Tuple[str, int]] = None
    first_errors: List[Dict[str, str]] = []
    for idx, query in enumerate(ordered_queries):
        errs = _runtime_validate_query(query)
        if errs:
            if idx == 0:
                first_errors = errs
            continue
        if first_valid is None:
            first_valid = (query, idx)
        profile = _query_runtime_profile(query)
        has_rows = profile.get("has_rows")
        exec_error = profile.get("error")
        unbound_vars = profile.get("unbound_vars") or []
        if has_rows is True and first_nonempty is None:
            first_nonempty = (query, idx)
        if has_rows is True and not unbound_vars:
            return query, [], idx
        if exec_error and idx == 0:
            first_errors = [{"type": "execution", "message": exec_error}]

    if first_nonempty is not None:
        return first_nonempty[0], [], first_nonempty[1]

    if first_valid is not None:
        return first_valid[0], [], first_valid[1]

    return ordered_queries[0], first_errors, 0


def _build_selection_explanation(
    question: str,
    effective_question: str,
    policy_mode: str,
    selected_policy: str,
    selection_reason: str,
    predicted_regime: Optional[str],
    predicted_entropy: Optional[float],
    selected_query: Optional[str],
    selected_from: Optional[str],
    selected_rank: Optional[int],
    selected_errors: Optional[List[Dict[str, str]]],
    candidates: List[Dict[str, str]],
    schema_ranked: List[Dict[str, object]],
    learning_ranked: List[Dict[str, object]],
) -> Dict[str, object]:
    schema_scores = {
        _candidate_key(str(r.get("query", ""))): float(r.get("score", 0.0))
        for r in schema_ranked
    }
    learning_scores = {
        _candidate_key(str(r.get("query", ""))): float(r.get("score", 0.0))
        for r in learning_ranked
    }
    schema_rank_pos = {
        _candidate_key(str(r.get("query", ""))): i + 1
        for i, r in enumerate(schema_ranked)
    }
    learning_rank_pos = {
        _candidate_key(str(r.get("query", ""))): i + 1
        for i, r in enumerate(learning_ranked)
    }
    selected_key = _candidate_key(selected_query or "")
    selected_coverage = (
        semantic_coverage_report(effective_question, selected_query or "")
        if selected_query
        else None
    )
    selected_has_rows, selected_execution_error = (
        _query_has_runtime_rows(selected_query or "") if selected_query else (None, None)
    )
    selected_profile = _query_runtime_profile(selected_query or "") if selected_query else {}

    rows: List[Dict[str, object]] = []
    for idx, cand in enumerate(candidates):
        query = str(cand.get("query", "")).strip()
        if not query:
            continue
        ckey = _candidate_key(query)
        errs = _runtime_validate_query(query)
        coverage = semantic_coverage_report(effective_question, query)
        has_rows, exec_error = (
            _query_has_runtime_rows(query) if not errs else (None, None)
        )
        exec_profile = _query_runtime_profile(query) if not errs else {}
        rows.append(
            {
                "candidate_index": idx + 1,
                "is_selected": bool(selected_query and ckey == selected_key),
                "is_valid": len(errs) == 0,
                "error_count": len(errs),
                "execution_has_rows": has_rows,
                "execution_error": exec_error,
                "execution_unbound_vars": ", ".join(
                    map(str, exec_profile.get("unbound_vars") or [])
                ),
                "coverage_score": coverage.get("coverage_score"),
                "coverage_missing": ", ".join(coverage.get("missing", [])),
                "schema_score": schema_scores.get(ckey),
                "schema_rank": schema_rank_pos.get(ckey),
                "ml_score": learning_scores.get(ckey),
                "ml_rank": learning_rank_pos.get(ckey),
                "query_preview": (query[:180] + "...") if len(query) > 180 else query,
            }
        )

    valid_count = sum(1 for r in rows if bool(r.get("is_valid")))
    return {
        "question": question,
        "effective_question": effective_question,
        "policy_mode": policy_mode,
        "selected_policy": selected_policy,
        "selection_reason": selection_reason,
        "predicted_regime": predicted_regime,
        "predicted_entropy": predicted_entropy,
        "selected_from": selected_from,
        "selected_rank_in_preference_order": selected_rank,
        "selected_query_valid": (
            bool(selected_query) and not bool(selected_errors)
        ),
        "selected_query_error_count": len(selected_errors or []),
        "selected_query_errors": list(selected_errors or []),
        "selected_coverage": selected_coverage,
        "selected_execution_has_rows": selected_has_rows,
        "selected_execution_error": selected_execution_error,
        "selected_execution_unbound_vars": list(selected_profile.get("unbound_vars") or []),
        "candidate_count": len(rows),
        "valid_candidate_count": valid_count,
        "invalid_candidate_count": max(0, len(rows) - valid_count),
        "candidates": rows,
    }


@dataclass
class PolicyDecision:
    type: str
    query: Optional[str]
    reason: str


class AmbiguityGatedPolicy:
    def __init__(
        self,
        thresholds_path: Optional[Path] = None,
        mode: str = "all",
    ) -> None:
        self.thresholds_path = thresholds_path or GATED_THRESHOLDS
        self.mode = (mode or "all").strip().lower()
        if self.mode not in {"off", "all", "mid"}:
            self.mode = "all"
        self.H1: Optional[float] = None
        self.H2: Optional[float] = None

        if self.mode == "mid" and self.thresholds_path.exists():
            try:
                data = json_load(self.thresholds_path)
                self.H1 = float(data.get("H1")) if "H1" in data else None
                self.H2 = float(data.get("H2")) if "H2" in data else None
            except Exception:
                self.H1 = None
                self.H2 = None

    def select(
        self,
        schema_ranked: List[Dict[str, object]],
        learning_ranked: List[Dict[str, object]],
        entropy: float,
    ) -> PolicyDecision:
        if not schema_ranked and not learning_ranked:
            return PolicyDecision("abstain", None, "no_candidates")

        if self.mode == "off":
            use_learning = False
            reason = "policy=off"
        elif self.mode == "all":
            use_learning = True
            reason = "policy=all"
        else:
            use_learning = False
            reason = "policy=mid; default_schema"
            if self.H1 is not None and self.H2 is not None:
                use_learning = self.H1 <= entropy <= self.H2
                reason = (
                    f"policy=mid; entropy {entropy:.4f} "
                    f"{'within' if use_learning else 'outside'} "
                    f"[{self.H1:.4f}, {self.H2:.4f}]"
                )

        primary = learning_ranked if use_learning else schema_ranked
        fallback = schema_ranked if use_learning else learning_ranked
        chosen = primary
        chosen_type = "learning" if use_learning else "schema"

        if not chosen:
            chosen = fallback
            chosen_type = "schema" if use_learning else "learning"
            reason = f"{reason}; fallback"

        if not chosen:
            return PolicyDecision("abstain", None, "no_ranked_candidates")

        best = max(chosen, key=lambda c: c.get("score", float("-inf")))
        return PolicyDecision(chosen_type, best.get("query"), reason)


class QAResult:
    """
    Unified result object returned to UI / CLI.
    """

    def __init__(
        self,
        answer: str,
        explanation: Dict[str, Any],
        raw_result: Any = None,
    ):
        self.answer = answer
        self.explanation = explanation
        self.raw_result = raw_result


def answer_question(
    question: str,
    schema: KGSchema,
    llm_client: Optional[object] = None,
    questions_path: Optional[str] = None,
    enable_entity_linking: bool = True,
    use_ml_ranking: bool = True,
    ml_policy: str = "auto",
    ml_model_path: Optional[str] = None,
    ml_ambiguity_config_path: Optional[str] = None,
) -> Dict[str, object]:
    alias_index = _get_default_entity_alias_index() if enable_entity_linking else None
    query_plan_predictor = _load_query_plan_predictor_cached()
    generation = generate_candidates(
        question,
        schema,
        llm_client=llm_client,
        entity_alias_index=alias_index,
        query_plan_predictor=query_plan_predictor,
    )
    candidates = generation.get("candidates", [])
    metadata = generation.get("metadata", {})
    prompt = generation.get("prompt", "")
    effective_question = str(metadata.get("effective_question", "")).strip() or question
    policy_mode = (ml_policy or "auto").strip().lower()
    if policy_mode not in {"off", "all", "mid", "auto"}:
        policy_mode = "auto"
    if not use_ml_ranking:
        policy_mode = "off"
    resolved_model_path = _resolve_learning_model_path(ml_model_path)
    resolved_ambiguity_config_path = _resolve_ambiguity_config_path(ml_ambiguity_config_path)
    ambiguity_config = (
        _load_ambiguity_config_cached(resolved_ambiguity_config_path)
        if resolved_ambiguity_config_path is not None
        else None
    )

    if not candidates:
        selection_explanation = _build_selection_explanation(
            question=question,
            effective_question=effective_question,
            policy_mode=policy_mode,
            selected_policy="abstain",
            selection_reason="no_candidates",
            predicted_regime=None,
            predicted_entropy=None,
            selected_query=None,
            selected_from=None,
            selected_rank=None,
            selected_errors=None,
            candidates=[],
            schema_ranked=[],
            learning_ranked=[],
        )
        return {
            "answer": "I could not generate any valid query candidates.",
            "selected_query": None,
            "candidates": [],
            "schema_ranked": [],
            "learning_ranked": [],
            "metadata": metadata,
            "errors": [],
            "prompt": prompt,
            "policy": "abstain",
            "entropy": 0.0,
            "selection_reason": "no_candidates",
            "used_ml": False,
            "effective_question": effective_question,
            "ml_policy": policy_mode,
            "ml_model_path": str(resolved_model_path) if resolved_model_path else None,
            "predicted_regime": None,
            "predicted_entropy": None,
            "ambiguity_config_path": (
                str(resolved_ambiguity_config_path)
                if resolved_ambiguity_config_path
                else None
            ),
            "selection_explanation": selection_explanation,
        }

    schema_ranked = _rerank_with_semantic_coverage(
        effective_question,
        rank_schema_candidates(candidates, schema),
    )
    learning_ranked = (
        _rerank_with_semantic_coverage(
            effective_question,
            _rank_learning_candidates(
                effective_question,
                candidates,
                schema,
                model_path=resolved_model_path,
            ),
        )
        if policy_mode != "off"
        else []
    )
    entropy = _compute_entropy(schema_ranked)
    predicted_regime, predicted_entropy = _predict_runtime_regime(
        question=effective_question,
        candidates=candidates,
        schema=schema,
        model_path=resolved_model_path,
        ambiguity_config=ambiguity_config,
    )

    if policy_mode == "off":
        use_learning = False
        selection_reason = "policy=off"
    elif policy_mode == "all":
        use_learning = True
        selection_reason = "policy=all"
    elif policy_mode == "mid":
        if predicted_regime is not None:
            use_learning = predicted_regime == "mid"
            selection_reason = f"policy=mid; predicted_regime={predicted_regime}"
        else:
            fallback_policy = AmbiguityGatedPolicy(mode="mid")
            fallback_decision = fallback_policy.select(
                schema_ranked=schema_ranked,
                learning_ranked=learning_ranked,
                entropy=entropy,
            )
            use_learning = fallback_decision.type == "learning"
            selection_reason = f"{fallback_decision.reason}; predictor_unavailable"
    else:  # auto
        if predicted_regime is not None and ambiguity_config is not None:
            allowed = set(ambiguity_config.ml_regimes or ["mid"])
            use_learning = predicted_regime in allowed
            selection_reason = (
                "policy=auto; "
                f"predicted_regime={predicted_regime}; "
                f"ml_regimes={','.join(sorted(allowed))}"
            )
        else:
            fallback_policy = AmbiguityGatedPolicy(mode="mid")
            fallback_decision = fallback_policy.select(
                schema_ranked=schema_ranked,
                learning_ranked=learning_ranked,
                entropy=entropy,
            )
            use_learning = fallback_decision.type == "learning"
            selection_reason = f"{fallback_decision.reason}; auto_fallback=entropy_mid"

    primary = learning_ranked if use_learning else schema_ranked
    fallback = schema_ranked if use_learning else learning_ranked
    selected_policy = "learning" if use_learning else "schema"
    if not primary and fallback:
        primary, fallback = fallback, []
        selected_policy = "schema" if use_learning else "learning"
        selection_reason = f"{selection_reason}; fallback=no_primary_ranked"

    if not primary:
        selection_explanation = _build_selection_explanation(
            question=question,
            effective_question=effective_question,
            policy_mode=policy_mode,
            selected_policy="abstain",
            selection_reason="no_ranked_candidates",
            predicted_regime=predicted_regime,
            predicted_entropy=predicted_entropy,
            selected_query=None,
            selected_from=None,
            selected_rank=None,
            selected_errors=None,
            candidates=candidates,
            schema_ranked=schema_ranked,
            learning_ranked=learning_ranked,
        )
        return {
            "answer": (
                "Multiple structurally valid interpretations were detected. "
                "Please refine your question."
            ),
            "selected_query": None,
            "candidates": candidates,
            "schema_ranked": schema_ranked,
            "learning_ranked": learning_ranked,
            "metadata": metadata,
            "errors": [],
            "prompt": prompt,
            "policy": "abstain",
            "entropy": entropy,
            "selection_reason": "no_ranked_candidates",
            "used_ml": False,
            "effective_question": effective_question,
            "ml_policy": policy_mode,
            "ml_model_path": str(resolved_model_path) if resolved_model_path else None,
            "predicted_regime": predicted_regime,
            "predicted_entropy": predicted_entropy,
            "ambiguity_config_path": (
                str(resolved_ambiguity_config_path)
                if resolved_ambiguity_config_path
                else None
            ),
            "selection_explanation": selection_explanation,
        }

    ordered_queries = _ordered_candidate_queries(primary, fallback)
    selected_query, errors, selected_rank = _select_best_valid_query(ordered_queries)
    if selected_query is None:
        selection_explanation = _build_selection_explanation(
            question=question,
            effective_question=effective_question,
            policy_mode=policy_mode,
            selected_policy="abstain",
            selection_reason=f"{selection_reason}; no_ordered_queries",
            predicted_regime=predicted_regime,
            predicted_entropy=predicted_entropy,
            selected_query=None,
            selected_from=None,
            selected_rank=None,
            selected_errors=None,
            candidates=candidates,
            schema_ranked=schema_ranked,
            learning_ranked=learning_ranked,
        )
        return {
            "answer": (
                "Multiple structurally valid interpretations were detected. "
                "Please refine your question."
            ),
            "selected_query": None,
            "candidates": candidates,
            "schema_ranked": schema_ranked,
            "learning_ranked": learning_ranked,
            "metadata": metadata,
            "errors": [],
            "prompt": prompt,
            "policy": "abstain",
            "entropy": entropy,
            "selection_reason": f"{selection_reason}; no_ordered_queries",
            "used_ml": False,
            "effective_question": effective_question,
            "ml_policy": policy_mode,
            "ml_model_path": str(resolved_model_path) if resolved_model_path else None,
            "predicted_regime": predicted_regime,
            "predicted_entropy": predicted_entropy,
            "ambiguity_config_path": (
                str(resolved_ambiguity_config_path)
                if resolved_ambiguity_config_path
                else None
            ),
            "selection_explanation": selection_explanation,
        }

    selected_key = " ".join(selected_query.split()).lower()
    primary_keys = {" ".join(str(r.get("query", "")).split()).lower() for r in primary}
    selected_from = "primary" if selected_key in primary_keys else "fallback"
    used_ml = selected_policy == "learning" and selected_from == "primary"

    if errors:
        results = {
            "rows": [],
            "matched_question_id": None,
            "error": "validation_failed",
        }
    else:
        results = execute_query_stub(
            selected_query,
            questions_path=questions_path,
            question=question,
        )

    answer = synthesize_answer(
        question, selected_query, results, errors or None
    )
    selection_explanation = _build_selection_explanation(
        question=question,
        effective_question=effective_question,
        policy_mode=policy_mode,
        selected_policy=selected_policy,
        selection_reason=selection_reason,
        predicted_regime=predicted_regime,
        predicted_entropy=predicted_entropy,
        selected_query=selected_query,
        selected_from=selected_from,
        selected_rank=selected_rank,
        selected_errors=errors,
        candidates=candidates,
        schema_ranked=schema_ranked,
        learning_ranked=learning_ranked,
    )

    return {
        "answer": answer,
        "selected_query": selected_query,
        "candidates": candidates,
        "schema_ranked": schema_ranked,
        "learning_ranked": learning_ranked,
        "metadata": metadata,
        "errors": errors,
        "prompt": prompt,
        "policy": selected_policy,
        "entropy": entropy,
        "selection_reason": selection_reason,
        "used_ml": used_ml,
        "raw_result": results,
        "effective_question": effective_question,
        "ml_policy": policy_mode,
        "ml_model_path": str(resolved_model_path) if resolved_model_path else None,
        "predicted_regime": predicted_regime,
        "predicted_entropy": predicted_entropy,
        "ambiguity_config_path": (
            str(resolved_ambiguity_config_path)
            if resolved_ambiguity_config_path
            else None
        ),
        "selected_query_rank": selected_rank,
        "selected_query_from": selected_from,
        "selection_explanation": selection_explanation,
    }


def run_qa(nl_question: str) -> QAResult:
    schema = load_default_schema()
    result = answer_question(nl_question, schema)
    explanation = {
        "policy": result.get("policy"),
        "entropy": result.get("entropy"),
        "selection_reason": result.get("selection_reason"),
        "errors": result.get("errors"),
        "num_candidates": len(result.get("candidates", [])),
    }
    return QAResult(
        answer=result.get("answer", ""),
        explanation=explanation,
        raw_result=result.get("raw_result"),
    )


def run_ambiguity_experiment(
    questions_path: Optional[str] = None,
    output_path: Optional[str] = None,
    llm_client: Optional[object] = None,
) -> List[Dict[str, object]]:
    schema = load_default_schema()
    qpath = (
        Path(questions_path)
        if questions_path
        else BASE / "data" / "infineon" / "infineon_dataset_100.json"
    )
    questions = _load_questions(qpath)

    summary: List[Dict[str, object]] = []
    for item in questions:
        question_text = str(item.get("question", ""))
        gold_query = str(item.get("gold_query") or item.get("query") or "")

        result = answer_question(
            question_text,
            schema,
            llm_client=llm_client,
            questions_path=str(qpath),
        )

        selected_query = result.get("selected_query") or ""
        correct = bool(selected_query) and is_relaxed_correct(
            selected_query, gold_query
        )

        summary.append(
            {
                "question": question_text,
                "entropy": float(result.get("entropy", 0.0) or 0.0),
                "correct": bool(correct),
                "used_ml": bool(result.get("used_ml", False)),
            }
        )

    out_path = (
        Path(output_path)
        if output_path
        else EXPERIMENTS_DIR / "ambiguity_results.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    return summary


def json_load(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))
