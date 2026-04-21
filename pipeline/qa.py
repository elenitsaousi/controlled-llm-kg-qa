# pipeline/qa.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from rdflib import Graph

from kg.executor import execute_query_stub
from kg.entity_linking import build_entity_alias_index
from kg.schema import KGSchema, load_default_schema
from kg.sparql_matching import is_relaxed_correct
from kg.sparql_normalization import normalize_sparql
from llm.answer_synthesis import synthesize_answer
from llm.candidate_generation import generate_candidates
from ranking.feature_extraction import extract_features
from ranking.ranker import rank_candidates as rank_schema_candidates
from validation.semantic import validate_query_semantic
from validation.syntax import validate_query_syntax
from visualization.ambiguity_metrics import ambiguity_entropy

BASE = Path(__file__).resolve().parents[1]
DEFAULT_LOGISTIC_MODEL = BASE / "ranking" / "models" / "logistic_ranker.joblib"
DEFAULT_INFINEON_JOBLIB_MODEL = BASE / "ranking" / "models" / "infineon_ranker.joblib"
DEFAULT_INFINEON_NP_MODEL = BASE / "ranking" / "models" / "infineon_np_tfidf_ranker_entitylink.json"
DEFAULT_INFINEON_NP_MODEL_FALLBACK = BASE / "ranking" / "models" / "infineon_np_tfidf_ranker.json"
DEFAULT_INFINEON_SCHEMA_PATH = BASE / "data" / "infineon" / "schema.json"
GATED_THRESHOLDS = BASE / "analysis_outputs" / "gated_thresholds.json"
EXPERIMENTS_DIR = BASE / "experiments"
DEFAULT_INFINEON_GRAPH = BASE / "data" / "infineon" / "graph.ttl"
_ENTITY_ALIAS_INDEX = None
_RANKER_CACHE: Dict[str, object] = {}

NP_MODEL_TYPE = "np_tfidf_logreg_v1"


def _get_default_entity_alias_index():
    global _ENTITY_ALIAS_INDEX
    if _ENTITY_ALIAS_INDEX is not None:
        return _ENTITY_ALIAS_INDEX
    if not DEFAULT_INFINEON_GRAPH.exists():
        _ENTITY_ALIAS_INDEX = None
        return None
    try:
        g = Graph()
        g.parse(str(DEFAULT_INFINEON_GRAPH), format="turtle")
        _ENTITY_ALIAS_INDEX = build_entity_alias_index(g)
    except Exception:
        _ENTITY_ALIAS_INDEX = None
    return _ENTITY_ALIAS_INDEX


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
    ml_policy: str = "all",
    ml_model_path: Optional[str] = None,
) -> Dict[str, object]:
    alias_index = _get_default_entity_alias_index() if enable_entity_linking else None
    generation = generate_candidates(
        question,
        schema,
        llm_client=llm_client,
        entity_alias_index=alias_index,
    )
    candidates = generation.get("candidates", [])
    metadata = generation.get("metadata", {})
    prompt = generation.get("prompt", "")
    effective_question = str(metadata.get("effective_question", "")).strip() or question
    policy_mode = (ml_policy or "all").strip().lower()
    if policy_mode not in {"off", "all", "mid"}:
        policy_mode = "all"
    if not use_ml_ranking:
        policy_mode = "off"
    resolved_model_path = _resolve_learning_model_path(ml_model_path)

    if not candidates:
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
        }

    schema_ranked = rank_schema_candidates(candidates, schema)
    learning_ranked = (
        _rank_learning_candidates(
            effective_question,
            candidates,
            schema,
            model_path=resolved_model_path,
        )
        if policy_mode != "off"
        else []
    )
    entropy = _compute_entropy(schema_ranked)

    policy = AmbiguityGatedPolicy(mode=policy_mode)
    decision = policy.select(schema_ranked, learning_ranked, entropy)
    used_ml = decision.type == "learning"

    if decision.type == "abstain" or not decision.query:
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
            "policy": decision.type,
            "entropy": entropy,
            "selection_reason": decision.reason,
            "used_ml": used_ml,
            "effective_question": effective_question,
            "ml_policy": policy_mode,
            "ml_model_path": str(resolved_model_path) if resolved_model_path else None,
        }

    selected_query = decision.query

    syntax_errors = validate_query_syntax(selected_query)
    semantic_errors = validate_query_semantic(selected_query)
    errors = syntax_errors + semantic_errors

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

    return {
        "answer": answer,
        "selected_query": selected_query,
        "candidates": candidates,
        "schema_ranked": schema_ranked,
        "learning_ranked": learning_ranked,
        "metadata": metadata,
        "errors": errors,
        "prompt": prompt,
        "policy": decision.type,
        "entropy": entropy,
        "selection_reason": decision.reason,
        "used_ml": used_ml,
        "raw_result": results,
        "effective_question": effective_question,
        "ml_policy": policy_mode,
        "ml_model_path": str(resolved_model_path) if resolved_model_path else None,
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
        else BASE / "data" / "toy_kg" / "questions" / "questions.json"
    )
    questions = _load_questions(qpath)

    summary: List[Dict[str, object]] = []
    for item in questions:
        question_text = str(item.get("question", ""))
        gold_query = str(item.get("gold_query", ""))

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
