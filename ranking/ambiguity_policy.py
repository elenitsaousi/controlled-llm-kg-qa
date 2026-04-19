import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rdflib import Graph
except Exception:  # pragma: no cover
    Graph = None  # type: ignore

from ranking.feature_config import FEATURE_NAMES
from ranking.feature_extraction import extract_features


DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


@dataclass
class AmbiguityConfig:
    entropy_source: str
    tau1: float
    tau2: float
    ml_regimes: List[str]
    agreement_top_n: int = 3
    agreement_invalid_penalty: float = 0.20

    def to_dict(self) -> Dict[str, object]:
        return {
            "entropy_source": self.entropy_source,
            "tau1": float(self.tau1),
            "tau2": float(self.tau2),
            "ml_regimes": list(self.ml_regimes),
            "agreement_top_n": int(self.agreement_top_n),
            "agreement_invalid_penalty": float(self.agreement_invalid_penalty),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "AmbiguityConfig":
        return cls(
            entropy_source=str(payload.get("entropy_source", "schema")).strip().lower(),
            tau1=float(payload.get("tau1", 0.33)),
            tau2=float(payload.get("tau2", 0.66)),
            ml_regimes=[normalize_label(x) for x in payload.get("ml_regimes", ["mid"])],
            agreement_top_n=int(payload.get("agreement_top_n", 3)),
            agreement_invalid_penalty=float(payload.get("agreement_invalid_penalty", 0.20)),
        )


def normalize_label(label: str) -> str:
    x = (label or "").strip().lower()
    if x == "medium":
        return "mid"
    return x


def regime_from_entropy(h: float, tau1: float, tau2: float) -> str:
    if h <= tau1:
        return "low"
    if h <= tau2:
        return "mid"
    return "high"


def entropy_from_scores(scores: Sequence[float], normalize: bool = True) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    if arr.size <= 1:
        return 0.0
    arr = arr - np.max(arr)
    ex = np.exp(arr)
    probs = ex / np.maximum(ex.sum(), 1e-12)
    h = float(-np.sum(probs * np.log(probs + 1e-12)))
    if normalize and arr.size > 1:
        h /= math.log(float(arr.size))
    return h


def schema_signal(features: Dict[str, float]) -> float:
    return (
        2.0 * float(features.get("entity_coverage", 0.0))
        + 1.5 * float(features.get("relation_coverage", 0.0))
        + 1.0 * float(features.get("expected_intermediate_coverage", 0.0))
        + 0.3 * float(features.get("has_where", 0.0))
        + 0.3 * float(features.get("has_type", 0.0))
        + 0.2 * float(features.get("has_aggregation", 0.0))
        - 1.2 * float(features.get("unexpected_label_ratio", 0.0))
        - 0.8 * float(features.get("invalid_predicate_count", 0.0))
        - 0.8 * float(features.get("unused_select_vars", 0.0))
        - 0.05 * float(features.get("rel_count", 0.0))
    )


def ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query


def _result_signature(rows: Iterable[Tuple]) -> frozenset[Tuple[str, ...]]:
    return frozenset(tuple(str(v) for v in row) for row in rows)


def _jaccard(a: frozenset[Tuple[str, ...]], b: frozenset[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return float(len(a & b) / u)


def _candidate_features(
    question: str,
    query: str,
    candidate: Dict[str, object],
    schema_dict: Optional[Dict[str, object]],
) -> Dict[str, float]:
    features = candidate.get("features")
    if isinstance(features, dict):
        return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}
    if schema_dict is None:
        return {name: 0.0 for name in FEATURE_NAMES}
    try:
        return extract_features(question, query, schema_dict)
    except Exception:
        return {name: 0.0 for name in FEATURE_NAMES}


def _agreement_entropy(
    question: str,
    candidates: Sequence[Dict[str, object]],
    schema_dict: Optional[Dict[str, object]],
    graph: Optional[Graph],
    top_n: int,
    invalid_penalty: float,
) -> float:
    if graph is None:
        return 1.0
    ranked = sorted(
        candidates,
        key=lambda c: schema_signal(
            _candidate_features(
                question=question,
                query=str(c.get("query", "")),
                candidate=c,
                schema_dict=schema_dict,
            )
        ),
        reverse=True,
    )
    selected = ranked[: max(1, int(top_n))]
    signatures: List[Optional[frozenset[Tuple[str, ...]]]] = []
    invalid = 0

    for cand in selected:
        query = str(cand.get("query", "")).strip()
        if not query:
            signatures.append(None)
            invalid += 1
            continue
        try:
            rows = graph.query(ensure_prefixes(query))
            signatures.append(_result_signature(rows))
        except Exception:
            signatures.append(None)
            invalid += 1

    valid = [s for s in signatures if s is not None]
    if len(valid) >= 2:
        sims = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                sims.append(_jaccard(valid[i], valid[j]))
        mean_sim = float(np.mean(sims)) if sims else 1.0
        entropy = 1.0 - mean_sim
    elif len(valid) == 1:
        entropy = 0.0
    else:
        entropy = 1.0

    if selected:
        entropy = min(1.0, entropy + invalid_penalty * (invalid / len(selected)))
    return float(max(0.0, entropy))


def estimate_entropy(
    question: str,
    candidates: Sequence[Dict[str, object]],
    source: str,
    schema_dict: Optional[Dict[str, object]] = None,
    model: Optional[object] = None,
    graph: Optional[Graph] = None,
    agreement_top_n: int = 3,
    agreement_invalid_penalty: float = 0.20,
) -> float:
    source = (source or "schema").strip().lower()
    if not candidates:
        return 1.0

    if source == "agreement":
        return _agreement_entropy(
            question=question,
            candidates=candidates,
            schema_dict=schema_dict,
            graph=graph,
            top_n=agreement_top_n,
            invalid_penalty=agreement_invalid_penalty,
        )

    if source == "ml":
        if model is None:
            return 1.0
        queries = [str(c.get("query", "")) for c in candidates]
        feats = [
            _candidate_features(question, q, c, schema_dict)
            for c, q in zip(candidates, queries)
        ]
        try:
            scores = model.score_question_candidates(question, queries, feats)
            return entropy_from_scores(scores.tolist(), normalize=True)
        except Exception:
            return 1.0

    # schema default
    schema_scores = [
        schema_signal(
            _candidate_features(
                question=question,
                query=str(c.get("query", "")),
                candidate=c,
                schema_dict=schema_dict,
            )
        )
        for c in candidates
    ]
    return entropy_from_scores(schema_scores, normalize=True)


def predict_regime(
    question: str,
    candidates: Sequence[Dict[str, object]],
    config: AmbiguityConfig,
    schema_dict: Optional[Dict[str, object]] = None,
    model: Optional[object] = None,
    graph: Optional[Graph] = None,
) -> Tuple[str, float]:
    h = estimate_entropy(
        question=question,
        candidates=candidates,
        source=config.entropy_source,
        schema_dict=schema_dict,
        model=model,
        graph=graph,
        agreement_top_n=config.agreement_top_n,
        agreement_invalid_penalty=config.agreement_invalid_penalty,
    )
    reg = regime_from_entropy(h, config.tau1, config.tau2)
    return reg, h


def save_ambiguity_config(path: str, config: AmbiguityConfig, metadata: Optional[Dict[str, object]] = None) -> None:
    payload = config.to_dict()
    payload["metadata"] = metadata or {}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_ambiguity_config(path: str) -> AmbiguityConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return AmbiguityConfig.from_dict(payload)
