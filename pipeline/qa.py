# pipeline/qa.py

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import signal

import numpy as np
from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery

from kg.executor import execute_query_stub
from kg.entity_linking import build_entity_alias_index
from kg.fuseki import make_sparql_store
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
from ranking.clarification import build_clarification_payload
from pipeline.request_routing import route_request
from ranking.feature_extraction import extract_features, extract_query_plan
from ranking.query_contract import (
    compare_contracts,
    extract_query_contract,
    extract_question_contract,
)
from ranking.ranker import rank_candidates as rank_schema_candidates
from validation.semantic import (
    semantic_coverage_report,
    semantic_judge_report,
    validate_query_semantic,
)
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
_RUNTIME_PROFILE_CACHE: Dict[Tuple[int, int, str], Dict[str, object]] = {}
_VALIDATED_QUERY_PAIR_CACHE: Optional[List[Dict[str, str]]] = None
_LOCAL_GRAPH_CACHE: Optional[Graph] = None

NP_MODEL_TYPE = "np_tfidf_logreg_v1"
INTENT_RANKING_WEIGHT = float(os.getenv("INFINEON_INTENT_RANKING_WEIGHT", "0") or 0)
SEMANTIC_JUDGE_WEIGHT = float(os.getenv("INFINEON_SEMANTIC_JUDGE_WEIGHT", "0") or 0)
VALIDATED_RETRIEVAL_SOURCES = (
    BASE / "data" / "infineon" / "kgqa_seed_expansion_round1.json",
    BASE / "data" / "infineon" / "infineon_dev.json",
    BASE / "data" / "infineon" / "infineon_train.json",
)
VALIDATED_RETRIEVAL_DOMAIN_TOKENS = {
    "actual",
    "autonomous",
    "baseline",
    "bl1",
    "bl2",
    "cancellation",
    "company",
    "component",
    "current",
    "demand",
    "forecast",
    "future",
    "inventory",
    "month",
    "oem",
    "order",
    "quarter",
    "region",
    "sae",
    "sales",
    "semiconductor",
    "shortage",
    "survey",
    "technology",
    "tier1",
    "vehicle",
    "year",
}
VALIDATED_RETRIEVAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "list",
    "me",
    "of",
    "on",
    "our",
    "show",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def _get_default_entity_alias_index():
    global _ENTITY_ALIAS_INDEX
    if _ENTITY_ALIAS_INDEX is not None:
        return _ENTITY_ALIAS_INDEX
    try:
        g = _get_local_default_graph()
        if g is None:
            _ENTITY_ALIAS_INDEX = None
            return None
        _ENTITY_ALIAS_INDEX = build_entity_alias_index(g)
    except Exception:
        _ENTITY_ALIAS_INDEX = None
    return _ENTITY_ALIAS_INDEX


def _get_local_default_graph() -> Optional[Graph]:
    global _LOCAL_GRAPH_CACHE
    if _LOCAL_GRAPH_CACHE is not None:
        return _LOCAL_GRAPH_CACHE
    if not DEFAULT_INFINEON_GRAPH.exists():
        _LOCAL_GRAPH_CACHE = None
        return None
    try:
        g = Graph()
        g.parse(str(DEFAULT_INFINEON_GRAPH), format="turtle")
        _LOCAL_GRAPH_CACHE = g
        return g
    except Exception:
        _LOCAL_GRAPH_CACHE = None
        return None


def _get_default_graph() -> Optional[Graph]:
    global _DEFAULT_GRAPH_CACHE
    if _DEFAULT_GRAPH_CACHE is not None:
        return _DEFAULT_GRAPH_CACHE
    fuseki_query_url = os.getenv("FUSEKI_QUERY_URL", "").strip()
    if fuseki_query_url:
        try:
            g = Graph(store=make_sparql_store(fuseki_query_url))
            _DEFAULT_GRAPH_CACHE = g
            return g
        except Exception:
            _DEFAULT_GRAPH_CACHE = None
            return None
    _DEFAULT_GRAPH_CACHE = _get_local_default_graph()
    return _DEFAULT_GRAPH_CACHE


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


def _question_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    normalized: set[str] = set()
    for token in tokens:
        if token in VALIDATED_RETRIEVAL_STOPWORDS or len(token) < 2:
            continue
        if token == "regions":
            token = "region"
        elif token == "quarters":
            token = "quarter"
        elif token == "months":
            token = "month"
        elif token == "companies":
            token = "company"
        elif token == "vehicles":
            token = "vehicle"
        elif token == "technologies":
            token = "technology"
        elif token == "responses":
            token = "response"
        normalized.add(token)
    return normalized


def _load_validated_query_pairs() -> List[Dict[str, str]]:
    global _VALIDATED_QUERY_PAIR_CACHE
    if _VALIDATED_QUERY_PAIR_CACHE is not None:
        return list(_VALIDATED_QUERY_PAIR_CACHE)

    rows: List[Dict[str, str]] = []
    seen = set()
    for path in VALIDATED_RETRIEVAL_SOURCES:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "") or "").strip()
            query = str(item.get("query", "") or "").strip()
            if not question or not query:
                continue
            key = (_candidate_key(query), " ".join(question.lower().split()))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"question": question, "query": query, "source": path.name})

    _VALIDATED_QUERY_PAIR_CACHE = rows
    return list(rows)


def _validated_candidate_score(question: str, validated_question: str) -> float:
    q_tokens = _question_tokens(question)
    v_tokens = _question_tokens(validated_question)
    if not q_tokens or not v_tokens:
        return 0.0

    shared = q_tokens & v_tokens
    shared_domain = shared & VALIDATED_RETRIEVAL_DOMAIN_TOKENS
    if not shared_domain:
        return 0.0

    # Penalize candidates that introduce a concrete survey scope when the user
    # explicitly asked for another one. If the user gave no scope, keep the
    # candidate as a possible interpretation for clarification.
    for exclusive in ({"oem", "tier1"}, {"oem", "semiconductor"}, {"tier1", "semiconductor"}):
        q_has = q_tokens & exclusive
        v_has = v_tokens & exclusive
        if q_has and v_has and q_has != v_has:
            return 0.0

    overlap = len(shared) / max(1, len(q_tokens))
    jaccard = len(shared) / max(1, len(q_tokens | v_tokens))
    score = overlap + 0.45 * jaccard + 0.1 * len(shared_domain)

    ranking_terms = {"highest", "largest", "lowest", "top", "rank", "ranks", "ranking"}
    if v_tokens & ranking_terms and not q_tokens & ranking_terms:
        score -= 0.45

    q_norm = " ".join(question.lower().split())
    v_norm = " ".join(validated_question.lower().split())
    required_phrase_groups = [
        (("by month", "monthly", "each month", "per month"), ("month", "monthly")),
        (("by year", "yearly", "annual", "annually", "each year", "per year"), ("year", "yearly", "annual")),
        (("by quarter", "each quarter", "per quarter"), ("quarter",)),
        (("by region", "each region", "per region"), ("region",)),
        (("technology category", "technology categories"), ("technology category", "technology categories")),
        (("vehicle type", "vehicle category"), ("vehicle type", "vehicle category")),
        (("sae level",), ("sae level",)),
        (("response type",), ("response type",)),
        (("inventory trend", "by trend"), ("trend", "inventory trend")),
        (("by component", "per component"), ("component",)),
        (("vehicle sales", "vehicles sold", "sales volume"), ("vehicle sales", "vehicles sold", "sales volume", "sold")),
        (("actual", "actuals", "actual data"), ("actual", "actuals", "actual data")),
        (("forecast", "forecasted", "forecast data"), ("forecast", "forecasted", "forecast data")),
        (("future",), ("future",)),
        (("current",), ("current",)),
    ]
    for required_terms, accepted_terms in required_phrase_groups:
        if any(term in q_norm for term in required_terms) and not any(
            term in v_norm for term in accepted_terms
        ):
            score -= 0.85
    if (
        ("which compan" in q_norm or "list compan" in q_norm or "show compan" in q_norm)
        and any(term in v_norm for term in ["how many", "count", "number of"])
    ):
        score -= 1.1
    if ("which compan" in q_norm or "list compan" in q_norm or "show compan" in q_norm):
        if any(term in v_norm for term in ["split", "status", "yes/no", "versus", "group"]):
            score -= 1.0
        if any(term in v_norm for term in ["list compan", "which compan", "reported semiconductor shortage"]):
            score += 0.45
    if any(term in q_norm for term in ["sales", "sold", "vehicle units"]) and any(
        term in v_norm for term in ["autonomous", "sae level", "driving development"]
    ):
        score -= 1.1

    for phrase in (
        "by region",
        "by quarter",
        "by month",
        "by year",
        "technology category",
        "vehicle type",
        "sae level",
        "actual",
        "forecast",
        "shortage",
        "inventory",
        "order cancellation",
    ):
        if phrase in q_norm and phrase in v_norm:
            score += 0.25

    return score


def _retrieve_validated_candidates(question: str, limit: int = 8) -> List[Dict[str, str]]:
    rows: List[Tuple[float, Dict[str, str]]] = []
    for pair in _load_validated_query_pairs():
        score = _validated_candidate_score(question, pair["question"])
        if score < 0.72:
            continue
        rows.append((score, pair))

    rows.sort(key=lambda item: item[0], reverse=True)
    candidates: List[Dict[str, str]] = []
    seen_queries = set()
    for score, pair in rows:
        key = _candidate_key(pair["query"])
        if key in seen_queries:
            continue
        seen_queries.add(key)
        candidates.append(
            {
                "query": pair["query"],
                "source": "validated_retrieval",
                "validated_question": pair["question"],
                "validated_source": pair["source"],
                "validated_retrieval_score": f"{score:.4f}",
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


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
            sources = [str(c.get("source", "llm") or "llm") for c in candidates]
            scores = ranker.score_question_candidates(
                question,
                queries,
                feature_dicts,
                candidate_sources=sources,
                schema_dict=schema_dict,
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


def _has_word(text: str, *terms: str) -> bool:
    for term in terms:
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", text):
            return True
    return False


def _query_has(query: str, *terms: str) -> bool:
    q = query.lower()
    return any(term.lower() in q for term in terms)


def _add_intent_check(
    *,
    checks: List[Dict[str, object]],
    name: str,
    matched: bool,
    weight: float,
    reason: str,
) -> None:
    checks.append(
        {
            "name": name,
            "matched": bool(matched),
            "weight": float(weight),
            "reason": reason,
        }
    )


def _intent_alignment_report(question: str, query: str) -> Dict[str, object]:
    """Heuristic NL intent alignment used only for candidate ordering.

    The checks are intentionally generic: they reward structural agreement
    between the requested operation/dimensions and the generated SPARQL.
    They do not use evaluation IDs, gold queries, or final-test labels.
    """
    q = (question or "").lower()
    sparql = query or ""
    checks: List[Dict[str, object]] = []

    wants_avg = _has_word(q, "average", "avg", "mean")
    wants_total = (
        "total demand" in q
        or _has_word(q, "total", "totals", "sum", "aggregate", "aggregated")
    )
    wants_count = _has_word(q, "count", "counts") or "how many" in q
    wants_participant_total = _has_word(
        q, "participant", "participants", "response", "responses"
    )
    wants_max = _has_word(q, "highest", "largest", "maximum", "max", "strongest", "top")
    wants_delta = _has_word(q, "difference", "delta") or "bl1-bl2" in q

    if wants_avg:
        _add_intent_check(
            checks=checks,
            name="aggregation_avg",
            matched=_query_has(sparql, "avg("),
            weight=1.25,
            reason="question asks for average/mean",
        )
    if wants_count and not wants_participant_total:
        _add_intent_check(
            checks=checks,
            name="aggregation_count",
            matched=_query_has(sparql, "count("),
            weight=1.10,
            reason="question asks for counts/how many",
        )
    if wants_count and wants_participant_total:
        _add_intent_check(
            checks=checks,
            name="participant_count_sum",
            matched=_query_has(sparql, "sum(", "participantcount", "participants"),
            weight=1.10,
            reason="question asks for participant/response counts",
        )
    if wants_total and not wants_avg and not wants_count:
        _add_intent_check(
            checks=checks,
            name="aggregation_sum",
            matched=_query_has(sparql, "sum("),
            weight=1.00,
            reason="question asks for totals/aggregation",
        )
    if wants_max:
        _add_intent_check(
            checks=checks,
            name="max_or_desc_limit",
            matched=(
                _query_has(sparql, "max(")
                or bool(re.search(r"\border\s+by\s+desc\s*\(", sparql, flags=re.IGNORECASE))
            ),
            weight=0.90,
            reason="question asks for highest/largest/top",
        )

    if _query_has(q, "region", "regional"):
        _add_intent_check(
            checks=checks,
            name="dimension_region",
            matched=_query_has(sparql, "regionname", "inregion"),
            weight=0.90,
            reason="question asks for region grouping/filtering",
        )
    if _query_has(q, "technology", "tech", "nm"):
        _add_intent_check(
            checks=checks,
            name="dimension_technology",
            matched=_query_has(
                sparql,
                "analyzestechnologycategory",
                "fortechnologycategory",
                "technologycategory",
                "techlabel",
            ),
            weight=0.90,
            reason="question asks for technology category",
        )
    if _has_word(q, "quarter", "quarters", "quarterly", "q1", "q2", "q3", "q4"):
        _add_intent_check(
            checks=checks,
            name="dimension_quarter",
            matched=_query_has(sparql, "fortimeperiod", "periodlabel", "quarter"),
            weight=0.90,
            reason="question asks for quarter/time-period grouping",
        )
    asks_vehicle_type = (
        "vehicle type" in q
        or "vehicle category" in q
        or _has_word(q, "bev", "behv", "ice")
        or ("autonomous" in q and "vehicle" in q)
    )
    if asks_vehicle_type:
        _add_intent_check(
            checks=checks,
            name="dimension_vehicle",
            matched=_query_has(sparql, "hasvehicletype", "analyzesvehicletype", "vehicletype"),
            weight=0.90,
            reason="question asks for vehicle type",
        )
    if _has_word(q, "year", "years", "yearly"):
        _add_intent_check(
            checks=checks,
            name="dimension_year",
            matched=_query_has(sparql, "hasyear", "?year", " year"),
            weight=0.80,
            reason="question asks for yearly grouping",
        )

    if _has_word(q, "actual"):
        _add_intent_check(
            checks=checks,
            name="actual_data_filter",
            matched=_query_has(sparql, "isactualdata true"),
            weight=1.20,
            reason="question asks for actual data only",
        )
    if _has_word(q, "forecast", "forecasted"):
        _add_intent_check(
            checks=checks,
            name="forecast_data_filter",
            matched=_query_has(sparql, "isforecastdata true"),
            weight=1.20,
            reason="question asks for forecast data",
        )

    requested_origins = [
        ("tier1", "tier1_survey"),
        ("oem", "oem_survey"),
        ("semiconductor", "semiconductor_survey"),
    ]
    origin_mentions = [needle for word, needle in requested_origins if _query_has(q, word)]
    if origin_mentions:
        _add_intent_check(
            checks=checks,
            name="requested_survey_origins",
            matched=all(_query_has(sparql, needle) for needle in origin_mentions),
            weight=1.20,
            reason="question names survey-origin groups",
        )
    if len(origin_mentions) >= 2 and (
        _query_has(q, "group", "bucket", "separate", "split", "break down", "origin")
        or "survey group" in q
        or "survey-origin" in q
        or "survey origin" in q
    ):
        raw_type_projection = bool(
            re.search(r"select\s+\?[a-z0-9_]*surveytype", sparql, flags=re.IGNORECASE)
            and re.search(r"\?[a-z0-9_]*\s+a\s+\?[a-z0-9_]*surveytype", sparql, flags=re.IGNORECASE)
        )
        explicit_labels = (
            _query_has(sparql, "values")
            and _query_has(sparql, "'oem'", "'tier1'", "'semiconductor", '"oem"', '"tier1"', '"semiconductor')
        )
        _add_intent_check(
            checks=checks,
            name="survey_origin_labeling",
            matched=explicit_labels and not raw_type_projection,
            weight=1.35,
            reason="question asks for named survey-origin buckets",
        )

    mentions_bl1 = _has_word(q, "bl1")
    mentions_bl2 = _has_word(q, "bl2")
    if mentions_bl1 and mentions_bl2:
        returns_baseline_values = (
            _query_has(sparql, "?baseline")
            or (_query_has(sparql, "changebl1") and _query_has(sparql, "changebl2"))
            or (_query_has(sparql, " bl1") and _query_has(sparql, " bl2"))
        )
        if wants_delta:
            matched = _query_has(sparql, "delta", "bl1bl2") or (
                _query_has(sparql, "if(") and _query_has(sparql, "bl1") and _query_has(sparql, "bl2")
            )
        else:
            matched = returns_baseline_values and not (
                _query_has(sparql, "deltabl1bl2") and "select ((" in " ".join(sparql.lower().split())
            )
        _add_intent_check(
            checks=checks,
            name="bl1_bl2_structure",
            matched=matched,
            weight=1.35,
            reason="question asks for BL1/BL2 comparison or values",
        )
        if _query_has(q, "automotive"):
            _add_intent_check(
                checks=checks,
                name="automotive_filter",
                matched=_query_has(sparql, "automotive"),
                weight=0.90,
                reason="question asks for Automotive market segment",
            )

    matched_weight = sum(float(c["weight"]) for c in checks if bool(c["matched"]))
    missing_weight = sum(float(c["weight"]) for c in checks if not bool(c["matched"]))
    score = matched_weight - missing_weight
    return {
        "score": float(score),
        "matched": [c["name"] for c in checks if bool(c["matched"])],
        "missing": [c["name"] for c in checks if not bool(c["matched"])],
        "checks": checks,
    }


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
        intent = _intent_alignment_report(question, query)
        intent_score = float(intent.get("score", 0.0))
        intent_score_applied = intent_score * INTENT_RANKING_WEIGHT
        judge = semantic_judge_report(question, query)
        semantic_judge_score = float(judge.get("score", 0.0))
        semantic_judge_score_applied = semantic_judge_score * SEMANTIC_JUDGE_WEIGHT
        base_score = float(row.get("score", 0.0))
        updated = dict(row)
        updated["base_score"] = base_score
        updated["coverage_score"] = coverage
        updated["coverage_missing_count"] = missing_count
        updated["coverage_missing"] = list(report.get("missing", []))
        updated["coverage_required"] = list(report.get("required", []))
        updated["intent_score"] = intent_score
        updated["intent_score_applied"] = intent_score_applied
        updated["intent_weight"] = INTENT_RANKING_WEIGHT
        updated["intent_matched"] = list(intent.get("matched", []))
        updated["intent_missing"] = list(intent.get("missing", []))
        updated["semantic_judge_score"] = semantic_judge_score
        updated["semantic_judge_score_applied"] = semantic_judge_score_applied
        updated["semantic_judge_weight"] = SEMANTIC_JUDGE_WEIGHT
        updated["semantic_judge_penalties"] = ", ".join(judge.get("penalties", []))
        updated["semantic_judge_rewards"] = ", ".join(judge.get("rewards", []))
        # Coverage is a hard semantic signal: valid-but-partial queries should
        # lose to candidates that cover the requested business concepts.
        updated["score"] = (
            base_score
            + (2.0 * coverage)
            - (1.5 * missing_count)
            + intent_score_applied
            + semantic_judge_score_applied
        )
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


def _dedupe_candidates(candidates: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    seen = set()
    deduped: List[Dict[str, str]] = []
    duplicate_count = 0
    for cand in candidates:
        query = str(cand.get("query", "")).strip()
        if not query:
            continue
        key = _candidate_key(query)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(dict(cand))
    return deduped, duplicate_count


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

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()

def _safe_graph_query(graph, query, timeout=2):
    if not hasattr(signal, "SIGALRM"):
        try:
            return graph.query(query), None
        except Exception as e:
            return None, str(e)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        results = graph.query(query)
        signal.alarm(0)
        return results, None
    except TimeoutException:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


def _format_runtime_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if hasattr(value, "toPython"):
            return str(value.toPython())
    except Exception:
        pass
    text = str(value)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    return text


def _query_runtime_profile(query: str, max_rows: int = 25) -> Dict[str, object]:
    graph = _get_default_graph()
    if graph is None:
        return {
            "has_rows": None,
            "row_count": 0,
            "preview_rows": [],
            "unbound_vars": [],
            "error": "graph_unavailable",
        }
    query_key = _candidate_key(_ensure_prefixes(query))
    cache_key = (id(graph), int(max_rows), query_key)
    cached = _RUNTIME_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    try:
        print("\n=== SELECTION EXECUTION ===")
        print("QUERY:")
        print(query)
        results, error = _safe_graph_query(graph, _ensure_prefixes(query))

        if error:
            profile = {
                "has_rows": None if error == "timeout" else False,
                "row_count": 0,
                "preview_rows": [],
                "unbound_vars": [],
                "error": error,
            }
            _RUNTIME_PROFILE_CACHE[cache_key] = dict(profile)
            return profile
        
        vars_seen = [str(v) for v in getattr(results, "vars", [])]
        bound_counts = {v: 0 for v in vars_seen}
        row_count = 0
        preview_rows: List[Dict[str, str]] = []
        for row in results:
            if row_count >= max_rows:
                break
            row_count += 1
            if hasattr(row, "asdict"):
                rd = row.asdict()
                preview_rows.append(
                    {str(k): _format_runtime_value(v) for k, v in rd.items()}
                )
                for var in vars_seen:
                    if rd.get(var) is not None:
                        bound_counts[var] = bound_counts.get(var, 0) + 1
            else:
                preview_rows.append(
                    {
                        (vars_seen[idx] if idx < len(vars_seen) else f"col{idx + 1}"): _format_runtime_value(value)
                        for idx, value in enumerate(row)
                    }
                )
                for var, value in zip(vars_seen, row):
                    if value is not None:
                        bound_counts[var] = bound_counts.get(var, 0) + 1
        unbound_vars = [
            var for var in vars_seen if row_count > 0 and bound_counts.get(var, 0) == 0
        ]
        profile = {
            "has_rows": row_count > 0,
            "row_count": row_count,
            "preview_rows": preview_rows,
            "unbound_vars": unbound_vars,
            "error": None,
        }
        _RUNTIME_PROFILE_CACHE[cache_key] = dict(profile)
        return profile
    except Exception as exc:
        profile = {
            "has_rows": None,
            "row_count": 0,
            "preview_rows": [],
            "unbound_vars": [],
            "error": str(exc),
        }
        _RUNTIME_PROFILE_CACHE[cache_key] = dict(profile)
        return profile


def _query_has_runtime_rows(query: str) -> Tuple[Optional[bool], Optional[str]]:
    profile = _query_runtime_profile(query)
    return profile.get("has_rows"), profile.get("error")


def _strip_answer_prefix(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if cleaned.lower().startswith("answer:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned


def _candidate_interpretation_label(
    question: str,
    query: str,
    candidate: Optional[Dict[str, object]] = None,
) -> str:
    try:
        plan = extract_query_plan(query)
    except Exception:
        plan = {}
    aggregations = list(plan.get("aggregations") or [])
    surveys = list(plan.get("survey_origins") or [])
    classes = [
        c
        for c in list(plan.get("classes") or [])
        if c not in {"OEM_Survey", "Tier1_Survey", "Semiconductor_Survey"}
    ]
    group_by = list(plan.get("group_by_predicates") or []) or list(plan.get("group_by_vars") or [])

    if aggregations:
        prefix = {
            "AVG": "Average",
            "SUM": "Total",
            "COUNT": "Count of",
            "MAX": "Maximum",
            "MIN": "Minimum",
        }.get(str(aggregations[0]).upper(), str(aggregations[0]).upper())
    else:
        prefix = "Values for"

    topic = " / ".join(surveys[:1] + classes[:2]).strip(" /")
    if not topic:
        source = str((candidate or {}).get("source", "") or "").strip()
        topic = source if source else "graph interpretation"
    label = f"{prefix} {topic}".strip()
    if group_by:
        label += " by " + " and ".join(str(x).replace("_", " ") for x in group_by[:2])
    return label


def _answerable_option_preview(
    *,
    question: str,
    query: str,
    candidate: Dict[str, object],
    rank: int,
    profile: Dict[str, object],
) -> Dict[str, object]:
    preview_rows = list(profile.get("preview_rows") or [])
    answer_preview = synthesize_answer(
        question,
        query,
        {"rows": preview_rows[:5], "matched_question_id": None, "error": None},
        None,
    )
    answer_preview = _strip_answer_prefix(answer_preview)
    if len(answer_preview) > 320:
        answer_preview = answer_preview[:317].rstrip() + "..."
    return {
        "id": f"answerable_{rank}",
        "rank": rank,
        "label": _candidate_interpretation_label(question, query, candidate),
        "preview": answer_preview,
        "row_count": profile.get("row_count"),
        "query": query,
        "source": candidate.get("source"),
        "preview_rows": preview_rows[:5],
        "query_preview": (query[:180] + "...") if len(query) > 180 else query,
    }


def _candidate_is_user_answerable(question: str, query: str) -> Tuple[bool, List[str]]:
    """Guard user-facing options against non-empty but semantically wrong paths."""
    q = (question or "").lower()
    sparql = (query or "").lower()
    reasons: List[str] = []

    asks_sales = any(token in q for token in ("sales", "sold", "units sold", "vehicle units"))
    asks_demand = "demand" in q and not asks_sales
    asks_company_list = (
        "which compan" in q
        or "list compan" in q
        or ("show" in q and "compan" in q)
    )
    asks_count = any(
        token in q
        for token in ("how many", "number of", "count", "counts", "records", "entries")
    )
    if asks_sales and "autonomousdrivingdevelopment" in sparql:
        reasons.append("autonomous_query_for_sales_question")
    if asks_company_list and re.search(r"\bCOUNT\s*\(", query or "", re.IGNORECASE):
        reasons.append("count_query_for_company_list_question")
    if asks_demand and any(
        token in sparql
        for token in ("vehiclesalesobservation", "yearlysalesdata", "isforecast")
    ):
        reasons.append("sales_query_for_demand_question")
    if asks_demand and not asks_count and re.search(r"\bCOUNT\s*\(", query or "", re.IGNORECASE):
        reasons.append("count_query_for_demand_amount_question")
    if any(token in q for token in ("total", "sum", "summed", "overall")) and re.search(
        r"\bAVG\s*\(", query or "", re.IGNORECASE
    ):
        reasons.append("average_query_for_total_question")
    if any(token in q for token in ("average", "avg", "mean")) and re.search(
        r"\bSUM\s*\(", query or "", re.IGNORECASE
    ):
        reasons.append("sum_query_for_average_question")

    asks_inventory = "inventory" in q or "stock" in q
    if asks_inventory and not any(token in sparql for token in ("inventory", "component")):
        reasons.append("non_inventory_query_for_inventory_question")

    requested_region_literals = {
        "americas": "americas",
        "america": "americas",
        "europe": "europe",
        "japan": "japan",
        "china": "china",
        "all other": "all other",
        "asia pacific": "asia pacific",
    }
    for region_hint, required_literal in requested_region_literals.items():
        if region_hint in q and required_literal not in sparql:
            reasons.append(f"missing_region_filter:{region_hint}")

    asks_cancellation = "cancellation" in q or "cancel" in q
    if asks_cancellation and "ordercancellation" not in sparql:
        reasons.append("non_cancellation_query_for_cancellation_question")

    asks_autonomous = "autonomous" in q or "sae" in q or "adas" in q
    if asks_autonomous and "autonomousdrivingdevelopment" not in sparql:
        reasons.append("non_autonomous_query_for_autonomous_question")

    coverage = semantic_coverage_report(question, query)
    missing_required = [str(x) for x in coverage.get("missing", [])]
    if asks_demand and "demandforregion" in sparql and any(
        token in sparql for token in ("totaldemand", "totaldemandpercentagechange")
    ):
        missing_required = [x for x in missing_required if x != "CurrentDemandAnalysis"]
    if asks_sales and "yearlysalesdata" in sparql and "analyzesvehicletype" in sparql:
        missing_required = [x for x in missing_required if x != "VehicleType"]
    if missing_required and float(coverage.get("coverage_score", 0.0)) < 0.75:
        reasons.extend(f"missing_required:{x}" for x in missing_required)

    judge = semantic_judge_report(question, query)
    severe_penalties = [
        p
        for p in list(judge.get("penalties") or [])
        if str(p).startswith(
            (
                "missing_origin:",
                "missing_filter:",
                "missing_dimension:",
                "wrong_or_missing_aggregation:",
            )
        )
    ]
    if severe_penalties and float(judge.get("score", 0.0)) < -1.5:
        reasons.extend(str(p) for p in severe_penalties)

    return not reasons, reasons


def _prefer_answerable_selected_candidate(
    selected: Optional[Dict[str, object]],
    ordered_candidates: Sequence[Dict[str, object]],
    question: str,
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    if selected is None:
        return selected, None

    selected_query = str(selected.get("query", "") or "").strip()
    selected_ok, selected_reasons = _candidate_is_user_answerable(question, selected_query)
    selected["semantic_answerable"] = bool(selected_ok)
    selected["answerability_reject_reasons"] = list(selected_reasons)
    if selected_ok:
        return selected, None

    answerable_candidates: List[Dict[str, object]] = []
    for rank, candidate in enumerate(ordered_candidates[:8], start=1):
        query = str(candidate.get("query", "") or "").strip()
        if not query or _runtime_validate_query(query):
            continue
        ok, reasons = _candidate_is_user_answerable(question, query)
        candidate["semantic_answerable"] = bool(ok)
        candidate["answerability_reject_reasons"] = list(reasons)
        if not ok:
            continue
        profile = _query_runtime_profile(query)
        candidate["execution_has_rows"] = profile.get("has_rows")
        candidate["execution_error"] = profile.get("error")
        candidate["execution_row_count"] = profile.get("row_count")
        candidate["execution_unbound_vars"] = list(profile.get("unbound_vars") or [])
        if profile.get("has_rows") is True:
            candidate["selection_original_rank"] = int(candidate.get("selection_original_rank", rank - 1))
            answerable_candidates.append(candidate)

    if answerable_candidates:
        replacement = _select_best_candidate_semantic(answerable_candidates, question) or answerable_candidates[0]
        return replacement, {
            "status": "selected_semantic_replaced",
            "rejected_reasons": selected_reasons,
            "replacement_count": len(answerable_candidates),
        }

    return selected, {
        "status": "selected_semantic_rejected_no_replacement",
        "rejected_reasons": selected_reasons,
        "replacement_count": 0,
    }


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


def _answerability_assessment(
    *,
    question: str,
    selected_query: Optional[str],
    selected_errors: Optional[List[Dict[str, str]]],
    ordered_candidates: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Classify why a selected query can or cannot produce an answer.

    This is intentionally diagnostic: it does not claim that the graph has no
    possible answer unless all generated valid alternatives are also empty.
    """
    if not selected_query:
        return {
            "status": "generation_failure",
            "can_answer": False,
            "reason": "No query candidate was selected.",
            "selected_has_rows": None,
            "selected_error": None,
            "alternative_nonempty_count": 0,
            "valid_candidate_count": 0,
        }

    if selected_errors:
        return {
            "status": "query_invalid",
            "can_answer": False,
            "reason": "The selected query failed syntax or semantic validation.",
            "selected_has_rows": None,
            "selected_error": "; ".join(str(e.get("message", e)) for e in selected_errors),
            "alternative_nonempty_count": 0,
            "valid_candidate_count": 0,
        }

    selected_ok, selected_reject_reasons = _candidate_is_user_answerable(question, selected_query)
    selected_profile = _query_runtime_profile(selected_query)
    selected_has_rows = selected_profile.get("has_rows")
    selected_error = selected_profile.get("error")
    if selected_error:
        return {
            "status": "query_execution_error",
            "can_answer": False,
            "reason": "The selected query could not be executed against the graph.",
            "selected_has_rows": selected_has_rows,
            "selected_error": selected_error,
            "alternative_nonempty_count": 0,
            "valid_candidate_count": 0,
        }

    valid_count = 0
    nonempty_alternatives: List[Dict[str, object]] = []
    selected_key = _candidate_key(selected_query)
    for idx, candidate in enumerate(ordered_candidates[:8]):
        query = str(candidate.get("query", "") or "").strip()
        if not query:
            continue
        if _runtime_validate_query(query):
            continue
        valid_count += 1
        if _candidate_key(query) == selected_key:
            continue
        profile = _query_runtime_profile(query)
        if profile.get("has_rows") is True:
            is_answerable, reject_reasons = _candidate_is_user_answerable(question, query)
            if not is_answerable:
                candidate["answerability_reject_reasons"] = reject_reasons
                continue
            nonempty_alternatives.append(
                _answerable_option_preview(
                    question=question,
                    query=query,
                    candidate=candidate,
                    rank=idx + 1,
                    profile=profile,
                )
            )

    if not selected_ok:
        if nonempty_alternatives:
            return {
                "status": "selected_query_semantic_mismatch_but_alternatives_exist",
                "can_answer": False,
                "reason": (
                    "The selected query returned graph rows, but they do not match the "
                    "requested meaning. Other answerable interpretations were found."
                ),
                "selected_has_rows": selected_has_rows,
                "selected_error": None,
                "selected_row_count": selected_profile.get("row_count"),
                "selected_reject_reasons": selected_reject_reasons,
                "alternative_nonempty_count": len(nonempty_alternatives),
                "valid_candidate_count": valid_count,
                "nonempty_alternatives": nonempty_alternatives[:3],
            }
        return {
            "status": "selected_query_semantic_mismatch",
            "can_answer": False,
            "reason": (
                "The selected query returned graph rows, but the graph path does not "
                "match the requested meaning closely enough."
            ),
            "selected_has_rows": selected_has_rows,
            "selected_error": None,
            "selected_row_count": selected_profile.get("row_count"),
            "selected_reject_reasons": selected_reject_reasons,
            "alternative_nonempty_count": 0,
            "valid_candidate_count": valid_count,
            "nonempty_alternatives": [],
        }

    if selected_has_rows is True:
        return {
            "status": "answer_available",
            "can_answer": True,
            "reason": "The selected query executed and returned graph rows.",
            "selected_has_rows": True,
            "selected_error": None,
            "selected_row_count": selected_profile.get("row_count"),
            "alternative_nonempty_count": len(nonempty_alternatives),
            "valid_candidate_count": valid_count,
            "nonempty_alternatives": nonempty_alternatives[:3],
        }

    if nonempty_alternatives:
        return {
            "status": "selected_query_empty_but_alternatives_exist",
            "can_answer": False,
            "reason": (
                "The selected query returned 0 rows, but other valid generated "
                "interpretations returned rows. This usually indicates a wrong "
                "interpretation, over-filtering, or ranking/selection failure."
            ),
            "selected_has_rows": False,
            "selected_error": None,
            "selected_row_count": selected_profile.get("row_count"),
            "alternative_nonempty_count": len(nonempty_alternatives),
            "valid_candidate_count": valid_count,
            "nonempty_alternatives": nonempty_alternatives[:3],
        }

    return {
        "status": "no_rows_for_generated_queries",
        "can_answer": False,
        "reason": (
            "The selected query and the checked valid alternatives returned 0 rows. "
            "This may mean no matching data exists for the requested interpretation, "
            "or that the system did not generate the correct graph path."
        ),
        "selected_has_rows": False,
        "selected_error": None,
        "selected_row_count": selected_profile.get("row_count"),
        "alternative_nonempty_count": 0,
        "valid_candidate_count": valid_count,
        "nonempty_alternatives": [],
    }


def _candidate_shape_score(question: str, query: str, profile: Dict[str, object]) -> Tuple[float, List[str]]:
    q = (question or "").lower()
    sparql = (query or "").lower()
    score = 0.0
    reasons: List[str] = []

    order_cancellation_response_summary = (
        "order cancellation" in q
        and "technology" in q
        and ("response" in q or "responses" in q or "response type" in q)
    )
    if _has_word(q, "participant", "participants") or order_cancellation_response_summary:
        if "participantcount" in sparql and "sum(" in sparql:
            score += 2.5
            reasons.append("participant_count_sum")
        if "count(?entry" in sparql or "count( ?entry" in sparql:
            score -= 2.5
            reasons.append("participant_count_counts_rows")

    grouped_percentage = (
        (
            "percentage change" in q
            or "percentage changes" in q
            or "future-demand percentages" in q
            or _has_word(q, "percentages")
        )
        and (
            " by " in q
            or " across " in q
            or " over " in q
            or _has_word(q, "differ", "differs")
            or _has_word(q, "matrix", "table", "view", "quarter", "quarters", "technology", "technologies", "vehicle")
        )
    )
    if grouped_percentage:
        if "avg(" in sparql and "percentagechange" in sparql:
            score += 2.0
            reasons.append("grouped_percentage_avg")
        elif "sum(" in sparql and "percentagechange" in sparql:
            score -= 1.2
            reasons.append("grouped_percentage_sum")
        elif re.search(r"select\b(?:(?!where).)*\?(pct|percentage)\b", sparql, flags=re.DOTALL):
            score -= 1.4
            reasons.append("grouped_percentage_raw_rows")
        if _has_word(q, "technology", "technologies") and (
            "?techlabel" in sparql or "?technologycategory" in sparql
        ):
            score += 0.35
            reasons.append("technology_label_shape")
        if _has_word(q, "quarter", "quarters") and "?quarterlabel" in sparql:
            score += 0.35
            reasons.append("quarter_label_shape")

    if ("bl1" in q and "bl2" in q) and "current" in q and "demand" in q:
        if "sum(?pct" in sparql and "?baseline" in sparql:
            score += 2.2
            reasons.append("baseline_pct_sum")
        if "count(?entry" in sparql or "count( ?entry" in sparql:
            score -= 2.2
            reasons.append("baseline_count_entries")

    if _has_word(q, "month", "monthly") and ("total" in q or "aggregate" in q or "sum" in q):
        if "bind(replace(str(?month" in sparql and "?monthlabel" in sparql:
            score += 0.45
            reasons.append("month_label_bound_from_uri")
        if "survey:monthlabel ?monthlabel" in sparql:
            score += 0.15
            reasons.append("month_label_property")

    vehicle_sales_time_period = (
        ("time period" in q or "time periods" in q)
        and (
            "vehicle-sales" in q
            or "vehicle sales" in q
            or "sold units" in q
            or "units sold" in q
            or _has_word(q, "actual", "forecast", "forecasted")
        )
    )
    if vehicle_sales_time_period:
        if "?periodlabel" in sparql or "?timelabel" in sparql:
            score += 0.35
            reasons.append("time_period_label_shape")
        if re.search(r"select\b(?:(?!where).)*\?timeperiod\b", sparql, flags=re.DOTALL):
            score -= 0.35
            reasons.append("raw_time_period_output")

    return float(score), reasons


def _query_plan_alignment_score(question: str, query: str) -> Tuple[float, List[str]]:
    q = " ".join((question or "").lower().replace("tier 1", "tier1").split())
    sparql = (query or "").lower()
    score = 0.0
    reasons: List[str] = []
    try:
        plan = extract_query_plan(query)
    except Exception:
        plan = {}

    labels = {str(label) for label in plan.get("labels", [])}
    classes = {str(value).lower() for value in plan.get("classes", [])}
    predicates = {str(value).lower() for value in plan.get("predicates", [])}
    aggregations = {str(value).upper() for value in plan.get("aggregations", [])}
    query_types = {str(value).lower() for value in plan.get("query_types", [])}

    def has_query_any(*terms: str) -> bool:
        return any(term in q for term in terms)

    def has_plan_any(*terms: str) -> bool:
        haystack = " ".join(sorted(labels)).lower().replace("_", " ")
        return any(term.lower().replace("_", " ") in haystack for term in terms)

    asks_count_records = has_query_any(
        "number of records",
        "count of records",
        "record count",
        "records are there",
        "entries are there",
        "number of entries",
        "count of entries",
        "observations",
    )
    asks_count_companies = "how many" in q and ("compan" in q or "oem" in q or "supplier" in q)
    asks_count = asks_count_records or asks_count_companies
    asks_average = has_query_any("average", "avg", "mean")
    asks_ranking = has_query_any("highest", "largest", "top", "most", "maximum", "peak", "lowest", "smallest")
    asks_sum = has_query_any("total", "sum", "summed", "combined", "overall") and not asks_count

    if asks_average:
        if "AVG" in aggregations:
            score += 2.0
            reasons.append("plan_avg_match")
        elif aggregations & {"SUM", "COUNT", "MAX", "MIN"}:
            score -= 2.4
            reasons.append("plan_avg_mismatch")
    if asks_count:
        if "COUNT" in aggregations:
            score += 2.0
            reasons.append("plan_count_match")
        elif aggregations & {"SUM", "AVG"}:
            score -= 1.8
            reasons.append("plan_count_mismatch")
    if asks_sum:
        if "SUM" in aggregations:
            score += 1.8
            reasons.append("plan_sum_match")
        elif "AVG" in aggregations or "COUNT" in aggregations:
            score -= 1.6
            reasons.append("plan_sum_mismatch")
    if asks_ranking:
        if query_types & {"ranking", "limited"} or "limit 1" in sparql or "order by desc" in sparql:
            score += 1.4
            reasons.append("plan_ranking_match")
        else:
            score -= 0.8
            reasons.append("plan_ranking_missing")

    requested_origins = []
    if "oem" in q:
        requested_origins.append(("oem", "survey:OEM_Survey"))
    if "tier1" in q:
        requested_origins.append(("tier1", "survey:Tier1_Survey"))
    if "semiconductor" in q or "semi" in q:
        requested_origins.append(("semiconductor", "survey:Semiconductor_Survey"))
    for name, label in requested_origins:
        if label in labels or name in sparql:
            score += 1.4
            reasons.append(f"plan_origin_match:{name}")
        else:
            score -= 2.0
            reasons.append(f"plan_origin_missing:{name}")
    if not requested_origins and sum(origin in labels for origin in {"survey:OEM_Survey", "survey:Tier1_Survey", "survey:Semiconductor_Survey"}) > 0:
        score -= 0.15
        reasons.append("plan_unrequested_origin_scope")

    dimension_checks = [
        ("region", ("region", "inregion", "regionname")),
        ("quarter", ("quarter", "quarterlabel", "fortimeperiod")),
        ("month", ("month", "monthlabel", "fortimeperiod")),
        ("year", ("year", "foryear", "hasyear")),
        ("technology category", ("technologycategory", "analyzestechnologycategory", "fortechnologycategory")),
        ("vehicle type", ("vehicletype", "analyzesvehicletype", "hasvehicletype", "forvehicletype")),
        ("sae level", ("saelevel", "hassaelevel")),
        ("component", ("component", "forcomponent")),
        ("trend", ("trend", "inventorytrend", "hasinventorytrend")),
        ("response type", ("responsetype", "hasresponsetype")),
        ("baseline", ("baseline", "baselinetype")),
    ]
    for question_term, plan_terms in dimension_checks:
        if question_term not in q:
            continue
        if any(term in predicates or term in sparql for term in plan_terms):
            score += 0.9
            reasons.append(f"plan_dimension_match:{question_term}")
        else:
            score -= 1.5
            reasons.append(f"plan_dimension_missing:{question_term}")

    if has_query_any("actual", "actuals", "actual data", "actually sold"):
        if "isactualdata" in sparql:
            score += 1.1
            reasons.append("plan_actual_filter_match")
        elif "isforecastdata" in sparql:
            score -= 2.0
            reasons.append("plan_actual_filter_mismatch")
    if has_query_any("forecast", "forecasted", "projected"):
        if "isforecastdata" in sparql:
            score += 1.1
            reasons.append("plan_forecast_filter_match")
        elif "isactualdata" in sparql:
            score -= 2.0
            reasons.append("plan_forecast_filter_mismatch")

    asks_sales = has_query_any("vehicle sales", "vehicles sold", "sales volume", "units sold")
    asks_inventory = "inventory" in q
    asks_cancellation = "cancellation" in q or "order cancellation" in q
    asks_autonomous = "autonomous" in q or "sae" in q
    asks_demand = "demand" in q and not asks_sales

    if asks_sales:
        if "yearlysalesdata" in sparql or "vehiclesalesobservation" in sparql:
            score += 1.2
            reasons.append("plan_metric_sales_match")
        elif "autonomousdrivingdevelopment" in sparql or "demandforregion" in sparql:
            score -= 2.0
            reasons.append("plan_metric_sales_mismatch")
    if asks_inventory:
        if "inventorydevelopment" in sparql:
            score += 1.2
            reasons.append("plan_metric_inventory_match")
        else:
            score -= 1.8
            reasons.append("plan_metric_inventory_mismatch")
    if asks_cancellation:
        if "ordercancellation" in sparql:
            score += 1.2
            reasons.append("plan_metric_cancellation_match")
        else:
            score -= 1.8
            reasons.append("plan_metric_cancellation_mismatch")
    if asks_autonomous:
        if "autonomousdrivingdevelopment" in sparql:
            score += 1.2
            reasons.append("plan_metric_autonomous_match")
        else:
            score -= 1.8
            reasons.append("plan_metric_autonomous_mismatch")
    if asks_demand:
        if any(term in sparql for term in ("demandforregion", "currentdemandanalysis", "futuredemandanalysis", "aggregateddemand")):
            score += 0.9
            reasons.append("plan_metric_demand_match")
        elif any(term in sparql for term in ("yearlysalesdata", "vehiclesalesobservation", "inventorydevelopment")):
            score -= 1.8
            reasons.append("plan_metric_demand_mismatch")

    return float(score), reasons


def _select_best_candidate_semantic(candidates, question):
    scored: List[Dict[str, object]] = []
    runtime_profile_enabled = (
        os.getenv("INFINEON_ENABLE_SELECTION_RUNTIME_PROFILE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    contract_score_enabled = (
        os.getenv("INFINEON_ENABLE_QUERY_CONTRACT_SCORE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    contract_override_enabled = (
        os.getenv("INFINEON_ENABLE_CONTRACT_SELECTION_OVERRIDE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    validated_source_rescue_enabled = (
        os.getenv("INFINEON_ENABLE_VALIDATED_SOURCE_RESCUE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    contract_weight = float(os.getenv("INFINEON_QUERY_CONTRACT_WEIGHT", "0.75") or 0.75)
    question_contract = (
        extract_question_contract(question)
        if (contract_score_enabled or contract_override_enabled)
        else None
    )

    for original_rank, cand in enumerate(candidates):
        query = str(cand.get("query", ""))

        judge = cand.get("semantic_judge_report")
        if not isinstance(judge, dict):
            judge = semantic_judge_report(question, query)
        semantic_score = float(judge.get("score", cand.get("semantic_judge_score") or 0.0))
        coverage = semantic_coverage_report(question, query)
        coverage_score = float(coverage.get("coverage_score", 0.0))
        missing_coverage = int(coverage.get("missing_count", 0))
        ml_score = float(cand.get("ml_score") or 0.0)
        ml_score = ml_score / (1 + abs(ml_score))  # normalize

        profile = (
            _query_runtime_profile(query)
            if runtime_profile_enabled
            else {
                "has_rows": None,
                "row_count": 0,
                "unbound_vars": [],
                "error": None,
            }
        )
        has_rows = profile.get("has_rows")
        execution_error = profile.get("error")
        unbound_vars = list(profile.get("unbound_vars") or [])

        execution_component = 0.0
        if has_rows is True:
            execution_component += 0.85
        elif has_rows is False:
            execution_component -= 0.65
        if execution_error:
            execution_component -= 1.0
        if unbound_vars:
            execution_component -= 1.0
        shape_component, shape_reasons = _candidate_shape_score(question, query, profile)
        plan_component, plan_reasons = _query_plan_alignment_score(question, query)
        contract_component = 0.0
        contract_reasons: List[str] = []
        contract_report: Dict[str, object] = {}
        query_contract_report: Dict[str, object] = {}
        if question_contract is not None:
            query_contract = extract_query_contract(query)
            comparison = compare_contracts(question_contract, query_contract)
            if contract_score_enabled:
                contract_component = contract_weight * float(comparison.score)
            contract_reasons = list(comparison.reasons)
            contract_report = comparison.to_dict()
            query_contract_report = query_contract.to_dict()
        source = str(cand.get("source", "") or "").strip().lower()
        source_component = 0.0
        if source == "validated_retrieval":
            source_component += 0.25
            try:
                retrieval_score = float(cand.get("validated_retrieval_score") or 0.0)
            except (TypeError, ValueError):
                retrieval_score = 0.0
            if retrieval_score > 0:
                source_component += min(0.25, max(0.0, retrieval_score - 0.70))
        elif source == "template":
            source_component += 0.15

        semantic_component = 1.35 * semantic_score
        coverage_component = (8.0 * coverage_score) - (2.5 * missing_coverage)
        ml_component = 0.20 * ml_score
        order_component = -0.03 * original_rank
        score = (
            semantic_component
            + coverage_component
            + execution_component
            + shape_component
            + plan_component
            + contract_component
            + source_component
            + ml_component
            + order_component
        )

        cand["selection_score"] = float(score)
        cand["selection_score_breakdown"] = {
            "semantic_component": float(semantic_component),
            "coverage_component": float(coverage_component),
            "execution_component": float(execution_component),
            "shape_component": float(shape_component),
            "plan_component": float(plan_component),
            "contract_component": float(contract_component),
            "source_component": float(source_component),
            "ml_component": float(ml_component),
            "order_component": float(order_component),
        }
        cand["selection_shape_reasons"] = shape_reasons + plan_reasons + contract_reasons
        if question_contract is not None:
            cand["question_contract"] = question_contract.to_dict()
            cand["query_contract"] = query_contract_report
            cand["query_contract_report"] = contract_report
        cand["coverage_score"] = coverage_score
        cand["coverage_missing"] = list(coverage.get("missing", []))
        cand["coverage_required"] = list(coverage.get("required", []))
        cand["semantic_judge_score"] = semantic_score
        cand["semantic_judge_report"] = judge
        cand["execution_has_rows"] = has_rows
        cand["execution_error"] = execution_error
        cand["execution_row_count"] = profile.get("row_count")
        cand["execution_unbound_vars"] = unbound_vars
        cand["selection_original_rank"] = original_rank

        scored.append(cand)

    if not scored:
        return None

    first = scored[0]
    if validated_source_rescue_enabled:
        source_choice = _validated_source_rescue_selection(scored, first)
        if source_choice is not None and source_choice is not first:
            return source_choice

    if contract_override_enabled and question_contract is not None:
        contract_choice = _contract_override_selection(scored, first)
        if contract_choice is not None and contract_choice is not first:
            return contract_choice

    best = max(
        scored,
        key=lambda row: (
            float(row.get("selection_score", float("-inf"))),
            -int(row.get("semantic_judge_original_rank", 0)),
        ),
    )
    if best is first:
        return first

    first_score = float(first.get("selection_score", float("-inf")))
    best_score = float(best.get("selection_score", float("-inf")))
    first_semantic = float(first.get("semantic_judge_score", 0.0))
    best_semantic = float(best.get("semantic_judge_score", 0.0))
    first_coverage = float(first.get("coverage_score", 0.0))
    best_coverage = float(best.get("coverage_score", 0.0))
    first_missing = len(first.get("coverage_missing") or [])
    best_missing = len(best.get("coverage_missing") or [])
    first_exec_error = bool(first.get("execution_error") or first.get("execution_unbound_vars"))
    best_exec_error = bool(best.get("execution_error") or best.get("execution_unbound_vars"))
    best_rank = int(best.get("selection_original_rank", 99))
    first_report = first.get("semantic_judge_report") or {}
    best_report = best.get("semantic_judge_report") or {}
    first_penalties = {str(p) for p in first_report.get("penalties", [])}
    first_shape_reasons = {str(r) for r in first.get("selection_shape_reasons", [])}
    best_shape_reasons = {str(r) for r in best.get("selection_shape_reasons", [])}
    first_extra_filters = len(first_report.get("extra_filters") or [])
    best_extra_filters = len(best_report.get("extra_filters") or [])
    best_aggregation_match = bool(best_report.get("aggregation_match"))
    first_aggregation_mismatch = any(
        p.startswith("wrong_or_missing_aggregation") or "used_for" in p
        for p in first_penalties
    )
    best_penalties = {str(p) for p in best_report.get("penalties", [])}
    first_plan_component = float(
        (first.get("selection_score_breakdown") or {}).get("plan_component", 0.0)
    )
    best_plan_component = float(
        (best.get("selection_score_breakdown") or {}).get("plan_component", 0.0)
    )
    first_contract_component = float(
        (first.get("selection_score_breakdown") or {}).get("contract_component", 0.0)
    )
    best_contract_component = float(
        (best.get("selection_score_breakdown") or {}).get("contract_component", 0.0)
    )
    plan_override_enabled = (
        os.getenv("INFINEON_ENABLE_PLAN_SELECTION_OVERRIDE", "1").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    # Keep ML/top-order stable unless the rule-based selector has a clear,
    # structured reason to override it. This prevents coverage/execution noise
    # from degrading already-good top candidates.
    override_margin = float(os.getenv("INFINEON_SELECTION_OVERRIDE_MARGIN", "1.25") or 1.25)
    clear_score_win = best_score >= first_score + override_margin
    coverage_not_worse = best_coverage >= first_coverage and best_missing <= first_missing
    semantic_not_worse = best_semantic >= first_semantic
    critical_shape_fix = bool(
        best_shape_reasons
        & {
            "participant_count_sum",
            "grouped_percentage_avg",
            "month_label_bound_from_uri",
            "technology_label_shape",
            "quarter_label_shape",
            "time_period_label_shape",
            "baseline_pct_sum",
        }
    )
    first_has_structural_defect = bool(first_penalties or first_missing)
    fixes_bad_first = (
        first_exec_error
        or first_missing > best_missing
        or first_semantic < 0.0
    )

    if (
        clear_score_win
        and coverage_not_worse
        and (semantic_not_worse or fixes_bad_first)
        and (first_has_structural_defect or critical_shape_fix)
    ):
        return best
    if fixes_bad_first and best_score >= first_score + 2.0 and coverage_not_worse:
        return best
    first_has_rows = first.get("execution_has_rows")
    best_has_rows = best.get("execution_has_rows")
    if (
        first_has_rows is not True
        and best_has_rows is True
        and best_score >= first_score + 0.35
        and coverage_not_worse
        and not best_exec_error
    ):
        return best
    high_confidence_semantic_win = (
        best_score >= first_score + override_margin
        and coverage_not_worse
        and best_semantic >= first_semantic + 0.25
        and not best_exec_error
    )
    if high_confidence_semantic_win:
        return best
    decisive_plan_win = (
        plan_override_enabled
        and best_score >= first_score + 1.35
        and best_plan_component >= first_plan_component + 1.8
        and coverage_not_worse
        and not best_exec_error
        and (
            semantic_not_worse
            or best_semantic >= first_semantic - 0.10
        )
    )
    if decisive_plan_win:
        return best
    contract_reasons_enabled = (
        os.getenv("INFINEON_ENABLE_QUERY_CONTRACT_SCORE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    query_contract_override_enabled = (
        os.getenv("INFINEON_ENABLE_QUERY_CONTRACT_OVERRIDE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    first_contract_missing = sum(
        1
        for reason in first_shape_reasons
        if reason.startswith("contract_") and "_missing:" in reason
    )
    best_contract_missing = sum(
        1
        for reason in best_shape_reasons
        if reason.startswith("contract_") and "_missing:" in reason
    )
    first_contract_conflicts = sum(
        1
        for reason in first_shape_reasons
        if reason.startswith("contract_") and "_conflict:" in reason
    )
    best_contract_conflicts = sum(
        1
        for reason in best_shape_reasons
        if reason.startswith("contract_") and "_conflict:" in reason
    )
    best_contract_matches = sum(
        1
        for reason in best_shape_reasons
        if reason.startswith("contract_") and "_match:" in reason
    )
    decisive_contract_win = (
        contract_reasons_enabled
        and query_contract_override_enabled
        and best_score >= first_score + 1.10
        and best_contract_component >= first_contract_component + 1.35
        and best_contract_matches >= 2
        and best_contract_missing <= first_contract_missing
        and best_contract_conflicts <= first_contract_conflicts
        and (best_contract_missing + best_contract_conflicts)
        < (first_contract_missing + first_contract_conflicts)
        and coverage_not_worse
        and not best_exec_error
        and best_semantic >= first_semantic - 0.15
    )
    if decisive_contract_win:
        return best
    contract_override_enabled = (
        os.getenv("INFINEON_ENABLE_CONTRACT_SELECTION_OVERRIDE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    first_contract_matches = sum(
        1
        for reason in first_shape_reasons
        if reason.startswith("plan_origin_match:")
        or reason.startswith("plan_metric_")
        or reason in {"plan_sum_match", "plan_avg_match", "plan_count_match", "plan_ranking_match"}
    )
    best_contract_matches = sum(
        1
        for reason in best_shape_reasons
        if reason.startswith("plan_origin_match:")
        or reason.startswith("plan_metric_")
        or reason in {"plan_sum_match", "plan_avg_match", "plan_count_match", "plan_ranking_match"}
    )
    best_has_origin_match = any(r.startswith("plan_origin_match:") for r in best_shape_reasons)
    best_has_metric_match = any(r.startswith("plan_metric_") and r.endswith("_match") for r in best_shape_reasons)
    contract_plan_win = (
        contract_override_enabled
        and best_score >= first_score + 1.45
        and best_contract_matches >= first_contract_matches + 1
        and best_has_origin_match
        and best_has_metric_match
        and coverage_not_worse
        and not best_exec_error
        and best_semantic >= first_semantic - 0.15
    )
    if contract_plan_win:
        return best
    shape_defect_fixed = (
        "answer_shape_missing_expected_columns" in first_penalties
        and "answer_shape_missing_expected_columns" not in best_penalties
    )
    clean_low_margin_win = (
        best_score >= first_score + 0.35
        and coverage_not_worse
        and not best_exec_error
        and (
            semantic_not_worse
            or critical_shape_fix
            or shape_defect_fixed
            or (first_aggregation_mismatch and best_aggregation_match)
        )
        and (
            first_has_structural_defect
            or critical_shape_fix
            or shape_defect_fixed
        )
    )
    if clean_low_margin_win:
        return best
    critical_shape_low_margin_win = (
        best_score >= first_score + 0.2
        and coverage_not_worse
        and not best_exec_error
        and critical_shape_fix
    )
    if critical_shape_low_margin_win:
        return best
    targeted_rescue_enabled = (
        os.getenv("INFINEON_ENABLE_TARGETED_SELECTION_RESCUE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    near_top_candidate = best_rank <= 2
    if targeted_rescue_enabled and near_top_candidate and not best_exec_error:
        fixes_missing_required = first_missing > best_missing and best_coverage > first_coverage
        fixes_over_filtering = first_extra_filters > best_extra_filters
        fixes_aggregation = first_aggregation_mismatch and best_aggregation_match
        targeted_score_win = best_score >= first_score + 0.75
        if targeted_score_win and coverage_not_worse and (
            fixes_missing_required or fixes_over_filtering or fixes_aggregation
        ):
            return best
    return first


def _contract_issue_count(report: Dict[str, object], key: str) -> int:
    payload = report.get(key)
    if not isinstance(payload, dict):
        return 0
    return sum(len(values or []) for values in payload.values())


def _contract_match_count(report: Dict[str, object]) -> int:
    payload = report.get("matched")
    if not isinstance(payload, dict):
        return 0
    return sum(len(values or []) for values in payload.values())


def _contract_axis_values(report: Dict[str, object], section: str, axis: str) -> set:
    payload = report.get(section)
    if not isinstance(payload, dict):
        return set()
    values = payload.get(axis) or []
    return {str(v) for v in values}


def _contract_override_selection(
    scored: List[Dict[str, object]], first: Dict[str, object]
) -> Optional[Dict[str, object]]:
    """Promote only candidates that clearly satisfy requested query constraints.

    This is deliberately stricter than adding the contract score to every
    candidate. A candidate can override the first row only when it fixes a
    concrete requested axis such as metric, aggregation, scope, dimension, or
    filter, while not becoming substantially worse on semantic/coverage score.
    """

    first_report = first.get("query_contract_report")
    if not isinstance(first_report, dict):
        return None

    first_score = float(first.get("selection_score", 0.0))
    first_contract_score = float(first_report.get("score", 0.0))
    first_semantic = float(first.get("semantic_judge_score", 0.0))
    first_coverage = float(first.get("coverage_score", 0.0))
    first_missing_coverage = len(first.get("coverage_missing") or [])
    first_missing = _contract_issue_count(first_report, "missing")
    first_conflicts = _contract_issue_count(first_report, "conflicts")
    first_matches = _contract_match_count(first_report)

    min_delta = float(os.getenv("INFINEON_CONTRACT_OVERRIDE_MIN_DELTA", "2.0") or 2.0)
    max_score_drop = float(os.getenv("INFINEON_CONTRACT_OVERRIDE_MAX_SCORE_DROP", "2.25") or 2.25)
    max_semantic_drop = float(os.getenv("INFINEON_CONTRACT_OVERRIDE_MAX_SEMANTIC_DROP", "0.35") or 0.35)
    max_coverage_drop = float(os.getenv("INFINEON_CONTRACT_OVERRIDE_MAX_COVERAGE_DROP", "0.10") or 0.10)

    best_candidate: Optional[Dict[str, object]] = None
    best_key: Tuple[float, float, float, int] = (float("-inf"), float("-inf"), float("-inf"), -999)

    for cand in scored[1:]:
        report = cand.get("query_contract_report")
        if not isinstance(report, dict):
            continue
        cand_contract_score = float(report.get("score", 0.0))
        contract_delta = cand_contract_score - first_contract_score
        if contract_delta < min_delta:
            continue

        cand_conflicts = _contract_issue_count(report, "conflicts")
        cand_missing = _contract_issue_count(report, "missing")
        cand_matches = _contract_match_count(report)
        if cand_conflicts > first_conflicts:
            continue
        if cand_missing >= first_missing and cand_matches <= first_matches:
            continue

        fixes_axis = False
        for axis in ("metrics", "aggregation", "scopes", "dimensions", "filters"):
            first_bad = _contract_axis_values(first_report, "missing", axis) | _contract_axis_values(
                first_report, "conflicts", axis
            )
            cand_good = _contract_axis_values(report, "matched", axis)
            if first_bad & cand_good:
                fixes_axis = True
                break
        if not fixes_axis:
            continue

        cand_score = float(cand.get("selection_score", 0.0))
        cand_semantic = float(cand.get("semantic_judge_score", 0.0))
        cand_coverage = float(cand.get("coverage_score", 0.0))
        cand_missing_coverage = len(cand.get("coverage_missing") or [])
        if cand_score < first_score - max_score_drop:
            continue
        if cand_semantic < first_semantic - max_semantic_drop:
            continue
        if cand_coverage < first_coverage - max_coverage_drop:
            continue
        if cand_missing_coverage > first_missing_coverage + 1:
            continue
        if cand.get("execution_error") or cand.get("execution_unbound_vars"):
            continue

        cand["contract_override_reason"] = {
            "first_contract_score": first_contract_score,
            "candidate_contract_score": cand_contract_score,
            "contract_delta": contract_delta,
            "first_missing": first_missing,
            "candidate_missing": cand_missing,
            "first_conflicts": first_conflicts,
            "candidate_conflicts": cand_conflicts,
        }
        key = (
            contract_delta,
            cand_score,
            cand_semantic,
            -int(cand.get("selection_original_rank", 99)),
        )
        if key > best_key:
            best_key = key
            best_candidate = cand

    return best_candidate


def _source_rank(source: str) -> int:
    source = str(source or "").strip().lower()
    if source == "validated_retrieval":
        return 3
    if source == "template":
        return 2
    if source in {"infineon", "llm", "generated"}:
        return 1
    return 0


def _normalized_ml_score(row: Dict[str, object]) -> float:
    try:
        raw = float(row.get("ml_score") or 0.0)
    except (TypeError, ValueError):
        raw = 0.0
    return raw / (1 + abs(raw))


def _validated_source_rescue_selection(
    scored: List[Dict[str, object]], first: Dict[str, object]
) -> Optional[Dict[str, object]]:
    """Conservatively promote validated/template candidates over generated top1.

    Diagnostics showed many ranking failures where the top candidate is an LLM
    generated query while the first correct candidate comes from validated
    retrieval. This rescue only fires when the higher-trust source also has a
    better ML signal and is not worse on coverage/contract safety.
    """

    first_source_rank = _source_rank(str(first.get("source") or ""))
    if first_source_rank >= 3:
        return None

    first_ml = _normalized_ml_score(first)
    first_coverage = float(first.get("coverage_score", 0.0))
    first_missing = len(first.get("coverage_missing") or [])
    first_semantic = float(first.get("semantic_judge_score", 0.0))
    first_score = float(first.get("selection_score", 0.0))
    first_contract = first.get("query_contract_report")
    first_conflicts = _contract_issue_count(first_contract, "conflicts") if isinstance(first_contract, dict) else 0

    min_ml_delta = float(os.getenv("INFINEON_VALIDATED_RESCUE_MIN_ML_DELTA", "0.04") or 0.04)
    max_score_drop = float(os.getenv("INFINEON_VALIDATED_RESCUE_MAX_SCORE_DROP", "3.5") or 3.5)
    max_semantic_drop = float(os.getenv("INFINEON_VALIDATED_RESCUE_MAX_SEMANTIC_DROP", "0.55") or 0.55)
    max_coverage_drop = float(os.getenv("INFINEON_VALIDATED_RESCUE_MAX_COVERAGE_DROP", "0.10") or 0.10)
    max_rank = int(os.getenv("INFINEON_VALIDATED_RESCUE_MAX_RANK", "8") or 8)

    best_candidate: Optional[Dict[str, object]] = None
    best_key: Tuple[float, int, float, float, int] = (float("-inf"), -99, float("-inf"), float("-inf"), -999)

    for cand in scored[1:]:
        cand_rank = int(cand.get("selection_original_rank", 99))
        if cand_rank >= max_rank:
            continue
        cand_source_rank = _source_rank(str(cand.get("source") or ""))
        if cand_source_rank <= first_source_rank:
            continue

        cand_ml = _normalized_ml_score(cand)
        ml_delta = cand_ml - first_ml
        if ml_delta < min_ml_delta:
            continue

        cand_coverage = float(cand.get("coverage_score", 0.0))
        cand_missing = len(cand.get("coverage_missing") or [])
        if cand_coverage < first_coverage - max_coverage_drop:
            continue
        if cand_missing > first_missing:
            continue

        cand_semantic = float(cand.get("semantic_judge_score", 0.0))
        if cand_semantic < first_semantic - max_semantic_drop:
            continue

        cand_score = float(cand.get("selection_score", 0.0))
        if cand_score < first_score - max_score_drop:
            continue
        if cand.get("execution_error") or cand.get("execution_unbound_vars"):
            continue

        cand_contract = cand.get("query_contract_report")
        cand_conflicts = _contract_issue_count(cand_contract, "conflicts") if isinstance(cand_contract, dict) else 0
        if cand_conflicts > first_conflicts:
            continue

        cand["validated_source_rescue_reason"] = {
            "first_source": first.get("source"),
            "candidate_source": cand.get("source"),
            "first_ml": first_ml,
            "candidate_ml": cand_ml,
            "ml_delta": ml_delta,
            "first_selection_score": first_score,
            "candidate_selection_score": cand_score,
        }
        key = (
            ml_delta,
            cand_source_rank,
            cand_coverage,
            cand_semantic,
            -cand_rank,
        )
        if key > best_key:
            best_key = key
            best_candidate = cand

    return best_candidate


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
    include_candidate_diagnostics: bool = True,
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
    semantic_judge_scores = {
        _candidate_key(str(r.get("query", ""))): float(r.get("semantic_judge_score", 0.0))
        for r in list(schema_ranked) + list(learning_ranked)
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
    valid_count = 0
    invalid_count = 0
    for idx, cand in enumerate(candidates):
        query = str(cand.get("query", "")).strip()
        if not query:
            continue
        ckey = _candidate_key(query)
        errs = _runtime_validate_query(query)
        is_valid = len(errs) == 0
        valid_count += int(is_valid)
        invalid_count += int(not is_valid)
        if not include_candidate_diagnostics:
            continue
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
                "semantic_judge_score": semantic_judge_scores.get(ckey),
                "query_preview": (query[:180] + "...") if len(query) > 180 else query,
            }
        )

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
        "candidate_diagnostics_included": bool(include_candidate_diagnostics),
        "candidate_count": len(candidates),
        "valid_candidate_count": valid_count,
        "invalid_candidate_count": invalid_count,
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
    include_candidate_diagnostics: bool = True,
    enable_clarification: bool = True,
    clarification_candidate_window: int = 8,
    enable_answerability_assessment: bool = True,
) -> Dict[str, object]:
    alias_index = None
    request_route = route_request(question, schema=schema, alias_index=alias_index)
    if request_route.get("route") == "kg_query" and enable_entity_linking:
        alias_index = _get_default_entity_alias_index()
        request_route = route_request(question, schema=schema, alias_index=alias_index)
    if request_route.get("route") != "kg_query":
        answer = str(request_route.get("answer", ""))
        if request_route.get("route") == "general_definition":
            term = str(request_route.get("term", "")).strip()
            if llm_client is not None and hasattr(llm_client, "generate_text"):
                answer = str(
                    llm_client.generate_text(
                        "Answer in one concise sentence. "
                        f"What is {term}?"
                    )
                ).strip()
            else:
                answer = f'I do not have a graph-backed definition for "{term}".'
        return {
            "answer": answer,
            "selected_query": None,
            "candidates": [],
            "schema_ranked": [],
            "learning_ranked": [],
            "metadata": {},
            "errors": [],
            "prompt": "",
            "policy": str(request_route.get("route")),
            "entropy": 0.0,
            "selection_reason": str(request_route.get("reason", request_route.get("route"))),
            "used_ml": False,
            "effective_question": question,
            "ml_policy": (ml_policy or "auto").strip().lower(),
            "ml_model_path": None,
            "predicted_regime": None,
            "predicted_entropy": None,
            "ambiguity_config_path": None,
            "selection_explanation": None,
            "query_plan_ml_used": False,
            "ml_ranker_applied": False,
            "candidate_duplicates_removed": 0,
            "request_route": request_route,
            "request_clarification": request_route.get("request_clarification"),
            "clarification": None,
            "answerability": {
                "status": "non_kg_request",
                "can_answer": bool(answer) and request_route.get("route") != "controlled_no_answer",
                "reason": str(request_route.get("reason", request_route.get("route"))),
                "selected_has_rows": None,
                "selected_error": None,
                "alternative_nonempty_count": 0,
                "valid_candidate_count": 0,
            },
        }

    query_plan_predictor = _load_query_plan_predictor_cached()
    generation = generate_candidates(
        question,
        schema,
        llm_client=llm_client,
        entity_alias_index=alias_index,
        entity_profile_graph=_get_default_graph() if enable_entity_linking else None,
        query_plan_predictor=query_plan_predictor,
    )
    raw_candidates = generation.get("candidates", [])
    validated_candidates = _retrieve_validated_candidates(question)
    candidates, duplicate_candidates_removed = _dedupe_candidates(
        list(validated_candidates) + list(raw_candidates)
    )
    metadata = generation.get("metadata", {})
    metadata["candidate_count_raw"] = len(raw_candidates)
    metadata["validated_retrieval_count"] = len(validated_candidates)
    metadata["candidate_duplicates_removed"] = duplicate_candidates_removed
    metadata["candidate_count_deduped"] = len(candidates)
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
            include_candidate_diagnostics=include_candidate_diagnostics,
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
            "query_plan_ml_used": bool(metadata.get("query_plan_predictor_applied")),
            "ml_ranker_applied": False,
            "candidate_duplicates_removed": duplicate_candidates_removed,
            "answerability": {
                "status": "generation_failure",
                "can_answer": False,
                "reason": "No query candidates were generated.",
                "selected_has_rows": None,
                "selected_error": None,
                "alternative_nonempty_count": 0,
                "valid_candidate_count": 0,
            },
        }

    schema_ranked = _rerank_with_semantic_coverage(
        effective_question,
        rank_schema_candidates(candidates, schema),
    )
    learning_ranked: List[Dict[str, object]] = []
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
                learning_ranked=[],
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
                learning_ranked=[],
                entropy=entropy,
            )
            use_learning = fallback_decision.type == "learning"
            selection_reason = f"{fallback_decision.reason}; auto_fallback=entropy_mid"

    if use_learning:
        learning_ranked = _rerank_with_semantic_coverage(
            effective_question,
            _rank_learning_candidates(
                effective_question,
                candidates,
                schema,
                model_path=resolved_model_path,
            ),
        )

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
            include_candidate_diagnostics=include_candidate_diagnostics,
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
            "query_plan_ml_used": bool(metadata.get("query_plan_predictor_applied")),
            "ml_ranker_applied": False,
            "candidate_duplicates_removed": duplicate_candidates_removed,
            "answerability": {
                "status": "generation_failure",
                "can_answer": False,
                "reason": "No ranked query candidate was available after generation.",
                "selected_has_rows": None,
                "selected_error": None,
                "alternative_nonempty_count": 0,
                "valid_candidate_count": 0,
            },
        }

    ordered_candidates = primary + fallback

    selected = _select_best_candidate_semantic(
        ordered_candidates,
        effective_question
    )
    clarification = None
    if enable_clarification and int(clarification_candidate_window) > 0:
        # Profile the top plan neighborhood before asking for clarification.
        # This is useful for offline diagnostics and safe alternatives, but can
        # be expensive in the interactive UI over a large RDFLib graph.
        clarification_window = max(1, int(clarification_candidate_window))
        for candidate_rank, candidate in enumerate(ordered_candidates[:clarification_window], start=1):
            query = str(candidate.get("query", "") or "").strip()
            if not query:
                continue
            profile = _query_runtime_profile(query)
            candidate["execution_has_rows"] = profile.get("has_rows")
            candidate["execution_error"] = profile.get("error")
            candidate["execution_row_count"] = profile.get("row_count")
            candidate["execution_unbound_vars"] = list(profile.get("unbound_vars") or [])
            is_answerable, reject_reasons = _candidate_is_user_answerable(effective_question, query)
            candidate["semantic_answerable"] = bool(is_answerable)
            candidate["answerability_reject_reasons"] = list(reject_reasons)
            if profile.get("has_rows") is True and is_answerable:
                option_preview = _answerable_option_preview(
                    question=effective_question,
                    query=query,
                    candidate=candidate,
                    rank=candidate_rank,
                    profile=profile,
                )
                candidate["answer_preview"] = option_preview.get("preview")
                candidate["preview_rows"] = option_preview.get("preview_rows")

        clarification = build_clarification_payload(
            effective_question,
            ordered_candidates,
            schema_dict=_load_schema_dict_for_ranking(schema),
            max_candidates=clarification_window,
        )
        if clarification is not None and selected and selected.get("execution_has_rows") is False:
            nonempty_candidates = [
                candidate
                for candidate in ordered_candidates[:clarification_window]
                if candidate.get("execution_has_rows") is True
                and candidate.get("semantic_answerable") is not False
            ]
            if nonempty_candidates:
                selected = _select_best_candidate_semantic(
                    nonempty_candidates,
                    effective_question,
                )

    selected, selected_answerability_override = _prefer_answerable_selected_candidate(
        selected,
        ordered_candidates,
        effective_question,
    )
    if selected_answerability_override is not None:
        metadata["selected_answerability_override"] = selected_answerability_override
    selected_query = selected.get("query") if selected else None
    errors = _runtime_validate_query(selected_query) if selected_query else []
    selected_rank = None

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
            include_candidate_diagnostics=include_candidate_diagnostics,
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
            "query_plan_ml_used": bool(metadata.get("query_plan_predictor_applied")),
            "ml_ranker_applied": False,
            "candidate_duplicates_removed": duplicate_candidates_removed,
            "answerability": {
                "status": "generation_failure",
                "can_answer": False,
                "reason": "No ordered query candidate was selected.",
                "selected_has_rows": None,
                "selected_error": None,
                "alternative_nonempty_count": 0,
                "valid_candidate_count": 0,
            },
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

    if enable_answerability_assessment:
        answerability = _answerability_assessment(
            question=effective_question,
            selected_query=selected_query,
            selected_errors=errors,
            ordered_candidates=ordered_candidates,
        )
    else:
        answerability = {
            "status": "not_checked_fast_mode",
            "can_answer": None,
            "reason": (
                "Interactive fast mode skipped expensive runtime answerability "
                "checks. The selected query is executed by the UI before rendering."
            ),
            "selected_has_rows": None,
            "selected_error": None,
            "alternative_nonempty_count": 0,
            "valid_candidate_count": 0,
            "nonempty_alternatives": [],
        }

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
        include_candidate_diagnostics=include_candidate_diagnostics,
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
        "query_plan_ml_used": bool(metadata.get("query_plan_predictor_applied")),
        "ml_ranker_applied": used_ml,
        "candidate_duplicates_removed": duplicate_candidates_removed,
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
        "answerability": answerability,
        "clarification": clarification,
        "request_route": request_route,
        "request_clarification": None,
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
        else BASE / "data" / "infineon" / "infineon_train.json"
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
