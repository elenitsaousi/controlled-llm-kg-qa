import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ranking.feature_config import FEATURE_NAMES
from ranking.feature_extraction import extract_features, extract_query_plan
from ranking.query_contract import (
    compare_contracts,
    extract_query_contract,
    extract_question_contract,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9_:%<>.=/-]+")
VAR_RE = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
SINGLE_QUOTE_STR_RE = re.compile(r"'[^']*'")
DOUBLE_QUOTE_STR_RE = re.compile(r'"[^"]*"')
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")

EXTRA_FEATURE_NAMES = [
    "candidate_order_score",
    "source_is_llm",
    "source_is_template",
    "source_is_validated_retrieval",
    "source_is_gold",
    "query_plan_label_count_log",
    "query_plan_token_overlap",
    "question_aggregation_match",
    "question_origin_match",
    "question_dimension_match",
    "contract_score",
    "contract_matched_count",
    "contract_missing_count",
    "contract_conflict_count",
    "contract_metric_match_count",
    "contract_metric_missing_count",
    "contract_metric_conflict_count",
    "contract_aggregation_match",
    "contract_aggregation_missing",
    "contract_aggregation_conflict",
    "contract_scope_match_count",
    "contract_scope_missing_count",
    "contract_dimension_match_count",
    "contract_dimension_missing_count",
    "contract_filter_match_count",
    "contract_filter_missing_count",
    "contract_filter_conflict_count",
    "contract_shape_match",
    "contract_shape_conflict",
    "grouped_requested_dimension_count",
    "grouped_dimension_match_count",
    "grouped_dimension_missing_count",
    "grouped_dimension_match_ratio",
    "query_is_grouped_when_requested",
    "query_is_ranked_when_grouped_requested",
    "query_is_filtered_when_grouped_requested",
    "shape_ranking_requested",
    "shape_grouped_requested",
    "shape_filtered_requested",
    "shape_query_is_ranking",
    "shape_query_is_grouped",
    "shape_query_is_filtered",
    "shape_ranking_match",
    "shape_ranking_missing",
    "shape_ranking_extra",
    "shape_grouped_match",
    "shape_grouped_missing",
    "shape_grouped_extra",
    "shape_filtered_match",
    "shape_filtered_missing",
    "shape_filtered_extra",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]


def _label_tokens(labels: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for label in labels:
        raw = str(label or "")
        if ":" in raw:
            raw = raw.split(":", 1)[1]
        raw = raw.replace("_", " ").replace("-", " ")
        raw = CAMEL_RE.sub(" ", raw)
        tokens.extend(_tokenize(raw))
    return tokens


def _candidate_order_score(position: int) -> float:
    return 1.0 / float(max(1, int(position) + 1))


def _query_plan_token_overlap(question: str, labels: Sequence[str]) -> float:
    q_tokens = set(_tokenize(question))
    label_tokens = set(_label_tokens(labels))
    if not q_tokens or not label_tokens:
        return 0.0
    return len(q_tokens & label_tokens) / float(len(label_tokens))


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _question_aggregation_match(question: str, labels: Sequence[str]) -> float:
    q = " ".join(str(question or "").lower().split())
    label_set = {str(label) for label in labels}
    if _contains_any(q, ["average", "avg", "mean"]):
        return 1.0 if "aggregation:AVG" in label_set else -1.0
    if _contains_any(q, ["how many", "number of", "count", "records", "entries"]):
        return 1.0 if "aggregation:COUNT" in label_set else -0.8
    if _contains_any(q, ["total", "sum", "summed", "combined", "overall"]):
        return 1.0 if "aggregation:SUM" in label_set else -0.7
    if _contains_any(q, ["highest", "largest", "top", "most", "maximum", "lowest", "smallest"]):
        return 1.0 if ("query_type:ranking" in label_set or "query_type:limited" in label_set) else -0.5
    return 0.0


def _question_origin_match(question: str, labels: Sequence[str]) -> float:
    q = " ".join(str(question or "").lower().replace("tier 1", "tier1").split())
    label_set = {str(label) for label in labels}
    requested = []
    if "oem" in q:
        requested.append("survey:OEM_Survey")
    if "tier1" in q:
        requested.append("survey:Tier1_Survey")
    if "semiconductor" in q or "semi" in q:
        requested.append("survey:Semiconductor_Survey")
    if not requested:
        return 0.0
    hits = sum(1 for label in requested if label in label_set)
    if hits == len(requested):
        return 1.0
    if hits:
        return 0.25
    return -1.0


def _question_dimension_match(question: str, labels: Sequence[str]) -> float:
    q = " ".join(str(question or "").lower().split())
    label_text = " ".join(str(label).lower().replace("_", " ") for label in labels)
    required_groups = [
        (["region"], ["region", "inregion"]),
        (["quarter"], ["quarter", "timeperiod", "for time period", "quarterlabel"]),
        (["month", "monthly"], ["month", "timeperiod", "for time period", "monthlabel"]),
        (["year", "yearly", "annual"], ["year", "foryear", "hasyear"]),
        (["technology category", "technology categories"], ["technologycategory", "technology category", "analyzestechnologycategory", "fortechnologycategory"]),
        (["vehicle type", "vehicle category"], ["vehicletype", "vehicle type", "analyzesvehicletype", "hasvehicletype"]),
        (["sae level", "sae"], ["saelevel", "sae level", "hassaelevel"]),
        (["component"], ["component", "forcomponent"]),
        (["trend", "inventory trend"], ["trend", "inventorytrend", "hasinventorytrend"]),
        (["response type"], ["responsetype", "response type", "hasresponsetype"]),
    ]
    requested = 0
    matched = 0
    for q_terms, label_terms in required_groups:
        if not any(term in q for term in q_terms):
            continue
        requested += 1
        if any(term in label_text for term in label_terms):
            matched += 1
    if requested == 0:
        return 0.0
    return (2.0 * matched / requested) - 1.0


def _axis_count(report: Dict[str, object], section: str, axis: str | None = None) -> float:
    payload = report.get(section)
    if not isinstance(payload, dict):
        return 0.0
    if axis is not None:
        values = payload.get(axis) or []
        return float(len(values))
    return float(sum(len(values or []) for values in payload.values()))


def _contract_feature_values(question: str, query: str) -> List[float]:
    try:
        report = compare_contracts(
            extract_question_contract(question),
            extract_query_contract(query),
        ).to_dict()
    except Exception:
        return [0.0] * 19

    return [
        float(report.get("score") or 0.0),
        _axis_count(report, "matched"),
        _axis_count(report, "missing"),
        _axis_count(report, "conflicts"),
        _axis_count(report, "matched", "metrics"),
        _axis_count(report, "missing", "metrics"),
        _axis_count(report, "conflicts", "metrics"),
        _axis_count(report, "matched", "aggregation"),
        _axis_count(report, "missing", "aggregation"),
        _axis_count(report, "conflicts", "aggregation"),
        _axis_count(report, "matched", "scopes"),
        _axis_count(report, "missing", "scopes"),
        _axis_count(report, "matched", "dimensions"),
        _axis_count(report, "missing", "dimensions"),
        _axis_count(report, "matched", "filters"),
        _axis_count(report, "missing", "filters"),
        _axis_count(report, "conflicts", "filters"),
        _axis_count(report, "matched", "answer_shape"),
        _axis_count(report, "conflicts", "answer_shape"),
    ]


def _plan_values(plan: Dict[str, object], key: str) -> List[str]:
    values = plan.get(key) or []
    if isinstance(values, (str, bytes)):
        return [str(values)]
    try:
        return [str(v) for v in values]
    except TypeError:
        return []


def _extract_plan_for_grouped_features(
    query: str,
    labels: Sequence[str],
    schema_dict: Dict[str, object] | None,
) -> Dict[str, object]:
    try:
        return extract_query_plan(query, schema_dict) if schema_dict is not None else extract_query_plan(query)
    except Exception:
        return {"labels": list(labels or [])}


def _dimension_present_in_grouped_output(plan: Dict[str, object], dimension: str) -> bool:
    text = " ".join(
        value.lower().replace("_", " ")
        for key in ("group_by_vars", "group_by_predicates", "select_vars", "labels")
        for value in _plan_values(plan, key)
    )
    compact = text.replace(" ", "")
    patterns = {
        "region": ("region", "inregion", "regionname"),
        "quarter": ("quarter", "periodlabel", "fortimeperiod", "timeperiod"),
        "month": ("month", "monthlabel", "periodlabel", "fortimeperiod", "timeperiod"),
        "year": ("year", "foryear", "hasyear"),
        "technology_category": ("technologycategory", "technology category", "category", "techlabel"),
        "vehicle_type": ("vehicletype", "vehicle type", "vehicle"),
        "sae_level": ("saelevel", "sae level", "sae"),
        "component": ("component", "componentlabel", "forcomponent"),
        "trend": ("trend", "inventorytrend", "hasinventorytrend"),
        "response_type": ("responsetype", "response type", "category"),
        "baseline": ("baseline", "baselinetype"),
        "survey": ("survey", "hassurveyorigin", "origin"),
    }
    return any(pattern in text or pattern in compact for pattern in patterns.get(dimension, (dimension,)))


def _grouped_dimension_feature_values(
    question: str,
    query: str,
    labels: Sequence[str],
    schema_dict: Dict[str, object] | None = None,
) -> List[float]:
    try:
        question_contract = extract_question_contract(question)
    except Exception:
        return [0.0] * 7

    if question_contract.answer_shape != "grouped_table" or not question_contract.dimensions:
        return [0.0] * 7

    plan = _extract_plan_for_grouped_features(query, labels, schema_dict)
    requested = set(question_contract.dimensions)
    matched = {
        dimension
        for dimension in requested
        if _dimension_present_in_grouped_output(plan, dimension)
    }
    missing = requested - matched
    query_types = {value.lower() for value in _plan_values(plan, "query_types")}
    is_grouped = bool(plan.get("group_by_vars") or plan.get("group_by_predicates") or "grouped" in query_types)
    is_ranked = bool({"ranking", "limited"} & query_types)
    is_filtered = "filtered" in query_types
    requested_count = float(len(requested))
    matched_count = float(len(matched))
    missing_count = float(len(missing))
    return [
        requested_count,
        matched_count,
        missing_count,
        matched_count / requested_count if requested_count else 0.0,
        1.0 if is_grouped else -1.0,
        -1.0 if is_ranked else 0.0,
        -0.5 if is_filtered and missing else 0.0,
    ]


def _extract_plan_labels_from_query(
    query: str,
    schema_dict: Dict[str, object] | None = None,
) -> List[str]:
    if schema_dict is None:
        return []
    try:
        return list(extract_query_plan(query, schema_dict).get("labels", []))
    except Exception:
        return []


def _query_shape_requested(question: str) -> Dict[str, bool]:
    q = " ".join(str(question or "").lower().replace("tier 1", "tier1").split())
    ranking_terms = [
        "highest",
        "largest",
        "lowest",
        "smallest",
        "top",
        "most",
        "least",
        "maximum",
        "minimum",
        "greatest",
    ]
    grouped_terms = [
        " by ",
        "broken down",
        "segmented",
        "grouped",
        "for each",
        "per ",
        "across",
        "compare",
        "distribution",
        "vary",
        "varies",
    ]
    filtered_terms = [
        "only",
        "restricted to",
        "according to",
        "based on",
        "oem",
        "tier1",
        "semiconductor",
        "actual",
        "forecast",
        "option1",
        "option2",
        "option3",
        "bl1",
        "bl2",
    ]
    return {
        "ranking": _contains_any(q, ranking_terms),
        "grouped": _contains_any(f" {q} ", grouped_terms),
        "filtered": _contains_any(q, filtered_terms),
    }


def _query_shape_feature_values(
    question: str,
    query: str,
    labels: Sequence[str],
    schema_dict: Dict[str, object] | None = None,
) -> List[float]:
    requested = _query_shape_requested(question)
    plan = _extract_plan_for_grouped_features(query, labels, schema_dict)
    query_types = {value.lower() for value in _plan_values(plan, "query_types")}
    is_ranking = bool({"ranking", "limited", "ordered"} & query_types)
    is_grouped = bool(plan.get("group_by_vars") or plan.get("group_by_predicates") or "grouped" in query_types)
    is_filtered = "filtered" in query_types

    def _match(req: bool, present: bool) -> float:
        if req and present:
            return 1.0
        if req and not present:
            return -1.0
        return 0.0

    def _missing(req: bool, present: bool) -> float:
        return 1.0 if req and not present else 0.0

    def _extra(req: bool, present: bool) -> float:
        return 1.0 if present and not req else 0.0

    return [
        1.0 if requested["ranking"] else 0.0,
        1.0 if requested["grouped"] else 0.0,
        1.0 if requested["filtered"] else 0.0,
        1.0 if is_ranking else 0.0,
        1.0 if is_grouped else 0.0,
        1.0 if is_filtered else 0.0,
        _match(requested["ranking"], is_ranking),
        _missing(requested["ranking"], is_ranking),
        _extra(requested["ranking"], is_ranking),
        _match(requested["grouped"], is_grouped),
        _missing(requested["grouped"], is_grouped),
        _extra(requested["grouped"], is_grouped),
        _match(requested["filtered"], is_filtered),
        _missing(requested["filtered"], is_filtered),
        _extra(requested["filtered"], is_filtered),
    ]


def _extra_feature_values(
    question: str,
    query: str,
    query_plan_labels: Sequence[str] | None,
    source: str,
    position: int,
    schema_dict: Dict[str, object] | None = None,
) -> List[float]:
    labels = list(query_plan_labels or [])
    if not labels:
        labels = _extract_plan_labels_from_query(query, schema_dict)
    source_norm = (source or "").strip().lower()
    return [
        _candidate_order_score(position),
        1.0 if source_norm == "llm" else 0.0,
        1.0 if source_norm == "template" else 0.0,
        1.0 if source_norm == "validated_retrieval" else 0.0,
        1.0 if source_norm == "gold" else 0.0,
        math.log1p(float(len(labels))),
        _query_plan_token_overlap(question, labels),
        _question_aggregation_match(question, labels),
        _question_origin_match(question, labels),
        _question_dimension_match(question, labels),
    ] + _contract_feature_values(question, query) + _grouped_dimension_feature_values(
        question,
        query,
        labels,
        schema_dict,
    ) + _query_shape_feature_values(
        question,
        query,
        labels,
        schema_dict,
    )


def _query_family_signature(query: str) -> str:
    q = " ".join(query.strip().split())
    q = SINGLE_QUOTE_STR_RE.sub("'STR'", q)
    q = DOUBLE_QUOTE_STR_RE.sub('"STR"', q)
    q = NUMBER_RE.sub("NUM", q)
    q = VAR_RE.sub("?VAR", q)
    digest = hashlib.md5(q.encode("utf-8")).hexdigest()[:16]
    return f"fam_{digest}"


@dataclass
class QuestionCandidate:
    query_id: str
    query: str
    is_correct: int
    is_valid: int
    features: Dict[str, float]
    source: str = "llm"
    query_plan_labels: List[str] | None = None


@dataclass
class QuestionItem:
    qid: str
    question: str
    ambiguity_label: str
    family: str
    candidates: List[QuestionCandidate]


def load_training_data(
    path: str,
    include_gold: bool = False,
    min_candidates: int = 1,
) -> Dict[str, QuestionItem]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed: Dict[str, QuestionItem] = {}
    for qid, rows in raw.items():
        if not rows:
            continue

        question = str(rows[0].get("question", "")).strip()
        ambiguity = str(rows[0].get("ambiguity_label", "")).strip().lower() or "unknown"
        family = str(rows[0].get("family", "")).strip()
        if not family:
            family = _query_family_signature(str(rows[0].get("gold_query", "")))
            if family == "fam_d41d8cd98f00b204":
                family = qid

        candidates: List[QuestionCandidate] = []
        for row in rows:
            source = str(row.get("source", "")).strip().lower()
            if source == "gold" and not include_gold:
                continue
            candidates.append(
                QuestionCandidate(
                    query_id=str(row.get("query_id", "")),
                    query=str(row.get("query", "")),
                    is_correct=int(row.get("is_correct", 0)),
                    is_valid=int(row.get("is_valid", 0)),
                    features={k: float(v) for k, v in row.get("features", {}).items()},
                    source=source or "llm",
                    query_plan_labels=list(row.get("query_plan_labels", []) or []),
                )
            )

        if len(candidates) < max(1, int(min_candidates)):
            continue

        parsed[qid] = QuestionItem(
            qid=qid,
            question=question,
            ambiguity_label=ambiguity,
            family=family,
            candidates=candidates,
        )
    return parsed


class SimpleTfidf:
    def __init__(self, idf: Dict[str, float] | None = None):
        self.idf: Dict[str, float] = idf or {}

    def fit(self, texts: Sequence[str]) -> None:
        df = Counter()
        n_docs = 0
        for text in texts:
            tokens = set(_tokenize(text))
            if not tokens:
                continue
            n_docs += 1
            for t in tokens:
                df[t] += 1

        self.idf = {}
        if n_docs == 0:
            return

        for tok, freq in df.items():
            self.idf[tok] = math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0

    def transform(self, text: str) -> Dict[str, float]:
        counts = Counter(_tokenize(text))
        total = float(sum(counts.values()))
        if total <= 0.0:
            return {}
        vec: Dict[str, float] = {}
        for tok, cnt in counts.items():
            idf = self.idf.get(tok)
            if idf is None:
                continue
            tf = cnt / total
            vec[tok] = tf * idf
        return vec

    @staticmethod
    def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        dot = 0.0
        for k, v in a.items():
            dot += v * b.get(k, 0.0)
        if dot == 0.0:
            return 0.0
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def similarity(self, text_a: str, text_b: str) -> float:
        return self.cosine(self.transform(text_a), self.transform(text_b))


def compose_feature_names(base_feature_names: Sequence[str] = FEATURE_NAMES) -> List[str]:
    return list(base_feature_names) + ["tfidf_similarity"] + list(EXTRA_FEATURE_NAMES)


def _build_feature_row(
    base_features: Dict[str, float],
    tfidf_similarity: float,
    extra_features: Sequence[float] | None = None,
    base_feature_names: Sequence[str] = FEATURE_NAMES,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> List[float]:
    row = [float(base_features.get(name, 0.0)) for name in base_feature_names]
    row.append(float(tfidf_similarity))
    row.extend(float(v) for v in (extra_features or [0.0] * len(EXTRA_FEATURE_NAMES)))
    disabled_names = {str(name) for name in (disabled_feature_names or [])}
    disabled_prefixes = tuple(str(prefix) for prefix in (disabled_feature_prefixes or []) if str(prefix))
    if disabled_names or disabled_prefixes:
        for idx, name in enumerate(compose_feature_names(base_feature_names)):
            if name in disabled_names or any(name.startswith(prefix) for prefix in disabled_prefixes):
                row[idx] = 0.0
    return row


def _fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _scale(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    reg: float = 0.02,
    epochs: int = 2500,
) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pos_weight = float(neg / pos) if pos > 0 else 1.0

    for _ in range(epochs):
        logits = X @ w + b
        probs = _sigmoid(logits)
        weights = np.where(y == 1, pos_weight, 1.0)
        err = (probs - y) * weights
        grad_w = (X.T @ err) / n + reg * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


class NPTfidfRanker:
    MODEL_TYPE = "np_tfidf_logreg_v1"

    def __init__(
        self,
        feature_names: Sequence[str],
        weights: np.ndarray,
        bias: float,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        idf: Dict[str, float],
        disabled_feature_names: Sequence[str] | None = None,
        disabled_feature_prefixes: Sequence[str] | None = None,
    ):
        self.feature_names = list(feature_names)
        self.weights = np.array(weights, dtype=float)
        self.bias = float(bias)
        self.scaler_mean = np.array(scaler_mean, dtype=float)
        self.scaler_std = np.array(scaler_std, dtype=float)
        self.vectorizer = SimpleTfidf(idf=idf)
        self.disabled_feature_names = list(disabled_feature_names or [])
        self.disabled_feature_prefixes = list(disabled_feature_prefixes or [])

    def score_rows(self, rows: np.ndarray) -> np.ndarray:
        rows_s = _scale(rows, self.scaler_mean, self.scaler_std)
        return _sigmoid(rows_s @ self.weights + self.bias)

    def score_question_candidates(
        self,
        question: str,
        candidate_queries: Sequence[str],
        candidate_base_features: Sequence[Dict[str, float]],
        candidate_query_plan_labels: Sequence[Sequence[str]] | None = None,
        candidate_sources: Sequence[str] | None = None,
        schema_dict: Dict[str, object] | None = None,
    ) -> np.ndarray:
        rows = []
        if candidate_query_plan_labels is None:
            candidate_query_plan_labels = [[] for _ in candidate_queries]
        if candidate_sources is None:
            candidate_sources = ["llm" for _ in candidate_queries]
        for position, (query, base_features, labels, source) in enumerate(
            zip(
                candidate_queries,
                candidate_base_features,
                candidate_query_plan_labels,
                candidate_sources,
            )
        ):
            sim = self.vectorizer.similarity(question, query)
            extra = _extra_feature_values(
                question=question,
                query=query,
                query_plan_labels=labels,
                source=source,
                position=position,
                schema_dict=schema_dict,
            )
            rows.append(
                _build_feature_row(
                    base_features,
                    sim,
                    extra,
                    FEATURE_NAMES,
                    disabled_feature_names=self.disabled_feature_names,
                    disabled_feature_prefixes=self.disabled_feature_prefixes,
                )
            )
        if not rows:
            return np.array([], dtype=float)
        X = np.array(rows, dtype=float)
        return self.score_rows(X)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.MODEL_TYPE,
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "idf": self.vectorizer.idf,
            "disabled_feature_names": list(self.disabled_feature_names),
            "disabled_feature_prefixes": list(self.disabled_feature_prefixes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "NPTfidfRanker":
        if data.get("model_type") != cls.MODEL_TYPE:
            raise ValueError(f"Unsupported model_type: {data.get('model_type')}")
        return cls(
            feature_names=data["feature_names"],
            weights=np.array(data["weights"], dtype=float),
            bias=float(data["bias"]),
            scaler_mean=np.array(data["scaler_mean"], dtype=float),
            scaler_std=np.array(data["scaler_std"], dtype=float),
            idf={k: float(v) for k, v in data.get("idf", {}).items()},
            disabled_feature_names=list(data.get("disabled_feature_names") or []),
            disabled_feature_prefixes=list(data.get("disabled_feature_prefixes") or []),
        )

    @classmethod
    def load(cls, path: str) -> "NPTfidfRanker":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str, metadata: Dict[str, object] | None = None) -> None:
        payload = self.to_dict()
        payload["metadata"] = metadata or {}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")


def _build_rows_for_qids(
    data: Dict[str, QuestionItem],
    qids: Iterable[str],
    vectorizer: SimpleTfidf,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]], Dict[str, bool]]:
    rows: List[List[float]] = []
    y: List[int] = []
    meta: List[Tuple[str, str]] = []
    any_correct: Dict[str, bool] = {}

    for qid in qids:
        item = data[qid]
        any_correct[qid] = any(c.is_correct == 1 for c in item.candidates)
        for position, cand in enumerate(item.candidates):
            sim = vectorizer.similarity(item.question, cand.query)
            extra = _extra_feature_values(
                question=item.question,
                query=cand.query,
                query_plan_labels=cand.query_plan_labels,
                source=cand.source,
                position=position,
            )
            rows.append(
                _build_feature_row(
                    cand.features,
                    sim,
                    extra,
                    FEATURE_NAMES,
                    disabled_feature_names=disabled_feature_names,
                    disabled_feature_prefixes=disabled_feature_prefixes,
                )
            )
            y.append(int(cand.is_correct))
            meta.append((qid, cand.query_id))

    X = np.array(rows, dtype=float) if rows else np.zeros((0, len(compose_feature_names())))
    y_arr = np.array(y, dtype=int) if y else np.zeros((0,), dtype=int)
    return X, y_arr, meta, any_correct


def _candidate_scores_by_question(
    data: Dict[str, QuestionItem],
    qids: Sequence[str],
    model: NPTfidfRanker,
) -> Dict[str, List[Tuple[str, float, int]]]:
    out: Dict[str, List[Tuple[str, float, int]]] = {}
    for qid in qids:
        item = data[qid]
        base_features = [c.features for c in item.candidates]
        queries = [c.query for c in item.candidates]
        labels = [c.query_plan_labels or [] for c in item.candidates]
        sources = [c.source for c in item.candidates]
        scores = model.score_question_candidates(
            item.question,
            queries,
            base_features,
            candidate_query_plan_labels=labels,
            candidate_sources=sources,
        )
        rows = []
        for cand, score in zip(item.candidates, scores):
            rows.append((cand.query_id, float(score), int(cand.is_correct)))
        out[qid] = rows
    return out


def _bootstrap_ci(
    values: Sequence[int],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(sample.mean()))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def build_grouped_stratified_folds(
    data: Dict[str, QuestionItem],
    n_folds: int = 5,
    seed: int = 42,
) -> List[List[str]]:
    # Group by family to reduce near-duplicate leakage.
    families: Dict[str, List[str]] = defaultdict(list)
    for qid, item in data.items():
        families[item.family].append(qid)

    labels = sorted({item.ambiguity_label for item in data.values()})
    total_per_label = Counter(item.ambiguity_label for item in data.values())
    target_per_fold = {
        lab: total_per_label[lab] / float(n_folds) for lab in labels
    }
    total_target = len(data) / float(n_folds)

    fam_records = []
    for fam, qids in families.items():
        c = Counter(data[qid].ambiguity_label for qid in qids)
        fam_records.append((fam, qids, c))

    rng = np.random.default_rng(seed)
    rng.shuffle(fam_records)
    fam_records.sort(key=lambda x: len(x[1]), reverse=True)

    folds: List[List[str]] = [[] for _ in range(n_folds)]
    fold_counts = [Counter() for _ in range(n_folds)]
    fold_totals = [0 for _ in range(n_folds)]

    for _, qids, fam_counter in fam_records:
        best_idx = 0
        best_score = None
        for i in range(n_folds):
            score = 0.0
            new_total = fold_totals[i] + len(qids)
            score += 0.2 * ((new_total - total_target) ** 2)
            for lab in labels:
                new_lab = fold_counts[i][lab] + fam_counter.get(lab, 0)
                score += (new_lab - target_per_fold[lab]) ** 2
            if best_score is None or score < best_score:
                best_score = score
                best_idx = i

        folds[best_idx].extend(qids)
        fold_totals[best_idx] += len(qids)
        for lab, c in fam_counter.items():
            fold_counts[best_idx][lab] += c

    return [sorted(fold) for fold in folds]


def cross_validate_ranker(
    data: Dict[str, QuestionItem],
    n_folds: int = 5,
    seed: int = 42,
    lr: float = 0.05,
    reg: float = 0.02,
    epochs: int = 2500,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> Dict[str, object]:
    folds = build_grouped_stratified_folds(data, n_folds=n_folds, seed=seed)
    all_qids = sorted(data.keys())
    oof_rows = []
    fold_summaries = []

    for fold_idx, test_qids in enumerate(folds, start=1):
        test_set = set(test_qids)
        train_qids = [qid for qid in all_qids if qid not in test_set]

        # Fit TF-IDF only on training texts -> no leakage.
        train_texts = []
        for qid in train_qids:
            item = data[qid]
            train_texts.append(item.question)
            train_texts.extend(c.query for c in item.candidates)
        vectorizer = SimpleTfidf()
        vectorizer.fit(train_texts)

        X_train, y_train, _, _ = _build_rows_for_qids(
            data,
            train_qids,
            vectorizer,
            disabled_feature_names=disabled_feature_names,
            disabled_feature_prefixes=disabled_feature_prefixes,
        )
        mean, std = _fit_scaler(X_train)
        X_train_s = _scale(X_train, mean, std)
        w, b = train_logistic(X_train_s, y_train, lr=lr, reg=reg, epochs=epochs)

        fold_model = NPTfidfRanker(
            feature_names=compose_feature_names(),
            weights=w,
            bias=b,
            scaler_mean=mean,
            scaler_std=std,
            idf=vectorizer.idf,
            disabled_feature_names=disabled_feature_names,
            disabled_feature_prefixes=disabled_feature_prefixes,
        )

        score_rows = _candidate_scores_by_question(data, test_qids, fold_model)

        fold_top1 = []
        fold_any = []
        fold_baseline = []

        for qid in test_qids:
            item = data[qid]
            rows = score_rows[qid]
            if not rows:
                continue
            ranked = sorted(rows, key=lambda x: x[1], reverse=True)
            top1_correct = int(ranked[0][2] == 1)
            any_correct = int(any(c.is_correct == 1 for c in item.candidates))
            baseline_correct = int(item.candidates[0].is_correct == 1)

            fold_top1.append(top1_correct)
            fold_any.append(any_correct)
            fold_baseline.append(baseline_correct)

            oof_rows.append(
                {
                    "qid": qid,
                    "ambiguity_label": item.ambiguity_label,
                    "family": item.family,
                    "fold": fold_idx,
                    "top1_correct": top1_correct,
                    "any_correct": any_correct,
                    "baseline_top1_correct": baseline_correct,
                    "top_query_id": ranked[0][0],
                    "top_score": float(ranked[0][1]),
                }
            )

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_questions": len(train_qids),
                "test_questions": len(test_qids),
                "top1_rate": float(np.mean(fold_top1)) if fold_top1 else 0.0,
                "any_rate": float(np.mean(fold_any)) if fold_any else 0.0,
                "baseline_top1_rate": float(np.mean(fold_baseline)) if fold_baseline else 0.0,
            }
        )

    oof_rows.sort(key=lambda r: r["qid"])
    top1_values = [r["top1_correct"] for r in oof_rows]
    any_values = [r["any_correct"] for r in oof_rows]
    base_values = [r["baseline_top1_correct"] for r in oof_rows]

    overall = {
        "n_questions": len(oof_rows),
        "top1_correct": int(sum(top1_values)),
        "any_correct": int(sum(any_values)),
        "baseline_top1_correct": int(sum(base_values)),
        "top1_rate": float(np.mean(top1_values)) if top1_values else 0.0,
        "any_rate": float(np.mean(any_values)) if any_values else 0.0,
        "baseline_top1_rate": float(np.mean(base_values)) if base_values else 0.0,
        "top1_ci95": _bootstrap_ci(top1_values, seed=seed),
        "any_ci95": _bootstrap_ci(any_values, seed=seed + 1),
        "baseline_ci95": _bootstrap_ci(base_values, seed=seed + 2),
    }

    per_ambiguity = {}
    labels = sorted(set(r["ambiguity_label"] for r in oof_rows))
    for lab in labels:
        rows = [r for r in oof_rows if r["ambiguity_label"] == lab]
        t = [r["top1_correct"] for r in rows]
        a = [r["any_correct"] for r in rows]
        b = [r["baseline_top1_correct"] for r in rows]
        per_ambiguity[lab] = {
            "n_questions": len(rows),
            "top1_correct": int(sum(t)),
            "any_correct": int(sum(a)),
            "baseline_top1_correct": int(sum(b)),
            "top1_rate": float(np.mean(t)) if t else 0.0,
            "any_rate": float(np.mean(a)) if a else 0.0,
            "baseline_top1_rate": float(np.mean(b)) if b else 0.0,
        }

    return {
        "config": {
            "n_folds": n_folds,
            "seed": seed,
            "lr": lr,
            "reg": reg,
            "epochs": epochs,
            "feature_names": compose_feature_names(),
            "disabled_feature_names": list(disabled_feature_names or []),
            "disabled_feature_prefixes": list(disabled_feature_prefixes or []),
        },
        "overall": overall,
        "per_ambiguity": per_ambiguity,
        "folds": fold_summaries,
        "oof_predictions": oof_rows,
    }


def train_final_ranker(
    data: Dict[str, QuestionItem],
    lr: float = 0.05,
    reg: float = 0.02,
    epochs: int = 2500,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> NPTfidfRanker:
    qids = sorted(data.keys())

    texts = []
    for qid in qids:
        item = data[qid]
        texts.append(item.question)
        texts.extend(c.query for c in item.candidates)

    vectorizer = SimpleTfidf()
    vectorizer.fit(texts)

    X, y, _, _ = _build_rows_for_qids(
        data,
        qids,
        vectorizer,
        disabled_feature_names=disabled_feature_names,
        disabled_feature_prefixes=disabled_feature_prefixes,
    )
    mean, std = _fit_scaler(X)
    X_s = _scale(X, mean, std)
    w, b = train_logistic(X_s, y, lr=lr, reg=reg, epochs=epochs)

    return NPTfidfRanker(
        feature_names=compose_feature_names(),
        weights=w,
        bias=b,
        scaler_mean=mean,
        scaler_std=std,
        idf=vectorizer.idf,
        disabled_feature_names=disabled_feature_names,
        disabled_feature_prefixes=disabled_feature_prefixes,
    )


def rank_candidates_with_model(
    model: NPTfidfRanker,
    question: str,
    candidates: Sequence[Dict[str, str]],
    schema_dict: Dict[str, object],
) -> List[Dict[str, str]]:
    queries = [str(c.get("query", "")) for c in candidates]
    base_features = []
    query_plan_labels = []
    sources = []
    for query in queries:
        try:
            feats = extract_features(question, query, schema_dict)
        except Exception:
            feats = {name: 0.0 for name in FEATURE_NAMES}
        base_features.append(feats)
        query_plan_labels.append(_extract_plan_labels_from_query(query, schema_dict))
    for cand in candidates:
        sources.append(str(cand.get("source", "llm") or "llm"))

    scores = model.score_question_candidates(
        question,
        queries,
        base_features,
        candidate_query_plan_labels=query_plan_labels,
        candidate_sources=sources,
        schema_dict=schema_dict,
    )
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    out: List[Dict[str, str]] = []
    for score, cand in ranked:
        row = dict(cand)
        row["ml_score"] = float(score)
        out.append(row)
    return out


def _tfidf_matrix(vectorizer: SimpleTfidf, texts: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    rows = []
    for text in texts:
        vec = vectorizer.transform(text)
        rows.append([float(vec.get(tok, 0.0)) for tok in vocab])
    if not rows:
        return np.zeros((0, len(vocab)), dtype=float)
    return np.array(rows, dtype=float)


class QueryPlanPredictor:
    MODEL_TYPE = "query_plan_ovr_logreg_v1"

    def __init__(
        self,
        labels: Sequence[str],
        vocab: Sequence[str],
        weights: np.ndarray,
        bias: np.ndarray,
        idf: Dict[str, float],
        threshold: float = 0.35,
        top_k: int = 24,
    ) -> None:
        self.labels = list(labels)
        self.vocab = list(vocab)
        self.weights = np.array(weights, dtype=float)
        self.bias = np.array(bias, dtype=float)
        self.vectorizer = SimpleTfidf(idf={k: float(v) for k, v in idf.items()})
        self.threshold = float(threshold)
        self.top_k = int(top_k)

    def score(self, questions: Sequence[str]) -> np.ndarray:
        X = _tfidf_matrix(self.vectorizer, questions, self.vocab)
        if X.size == 0:
            return np.zeros((0, len(self.labels)), dtype=float)
        return _sigmoid((X @ self.weights.T) + self.bias)

    def predict_labels(self, question: str, threshold: float | None = None, top_k: int | None = None) -> List[str]:
        scores = self.score([question])
        if scores.size == 0:
            return []
        th = self.threshold if threshold is None else float(threshold)
        k = self.top_k if top_k is None else int(top_k)
        row = scores[0]
        selected = [self.labels[i] for i, s in enumerate(row) if float(s) >= th]
        if not selected and len(row):
            selected = [self.labels[int(row.argmax())]]
        selected.sort(key=lambda lab: float(row[self.labels.index(lab)]), reverse=True)
        return selected[: max(1, k)]

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.MODEL_TYPE,
            "labels": self.labels,
            "vocab": self.vocab,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "idf": self.vectorizer.idf,
            "threshold": self.threshold,
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "QueryPlanPredictor":
        if data.get("model_type") != cls.MODEL_TYPE:
            raise ValueError(f"Unsupported model_type: {data.get('model_type')}")
        return cls(
            labels=list(data.get("labels", [])),
            vocab=list(data.get("vocab", [])),
            weights=np.array(data.get("weights", []), dtype=float),
            bias=np.array(data.get("bias", []), dtype=float),
            idf={k: float(v) for k, v in dict(data.get("idf", {})).items()},
            threshold=float(data.get("threshold", 0.35)),
            top_k=int(data.get("top_k", 24)),
        )

    @classmethod
    def load(cls, path: str) -> "QueryPlanPredictor":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str, metadata: Dict[str, object] | None = None) -> None:
        payload = self.to_dict()
        payload["metadata"] = metadata or {}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")


def load_query_plan_training_rows(dataset_path: str, schema_path: str) -> List[Dict[str, object]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    rows: List[Dict[str, object]] = []
    for item in dataset:
        question = str(item.get("question", "")).strip()
        query = str(item.get("query", "")).strip()
        if not question or not query:
            continue
        plan = extract_query_plan(query, schema)
        labels = list(plan.get("labels", []))
        if not labels:
            continue
        rows.append(
            {
                "id": str(item.get("id", "")),
                "question": question,
                "query": query,
                "labels": labels,
                "query_plan": plan,
            }
        )
    return rows


def train_query_plan_predictor(
    rows: Sequence[Dict[str, object]],
    min_label_count: int = 2,
    threshold: float = 0.35,
    top_k: int = 24,
    lr: float = 0.2,
    reg: float = 0.001,
    epochs: int = 800,
) -> QueryPlanPredictor:
    if not rows:
        raise RuntimeError("No query-plan training rows.")

    label_counts = Counter()
    for row in rows:
        label_counts.update(str(l) for l in row.get("labels", []))
    labels = sorted(
        lab for lab, count in label_counts.items()
        if int(count) >= max(1, int(min_label_count))
    )
    if not labels:
        raise RuntimeError("No labels meet min_label_count.")

    questions = [str(row.get("question", "")) for row in rows]
    vectorizer = SimpleTfidf()
    vectorizer.fit(questions)
    vocab = sorted(vectorizer.idf.keys())
    X = _tfidf_matrix(vectorizer, questions, vocab)

    weights = []
    biases = []
    for label in labels:
        y = np.array(
            [1 if label in set(map(str, row.get("labels", []))) else 0 for row in rows],
            dtype=int,
        )
        if int(y.sum()) == 0:
            weights.append(np.zeros((len(vocab),), dtype=float))
            biases.append(-10.0)
            continue
        if int(y.sum()) == len(y):
            weights.append(np.zeros((len(vocab),), dtype=float))
            biases.append(10.0)
            continue
        w, b = train_logistic(X, y, lr=lr, reg=reg, epochs=epochs)
        weights.append(w)
        biases.append(b)

    return QueryPlanPredictor(
        labels=labels,
        vocab=vocab,
        weights=np.array(weights, dtype=float),
        bias=np.array(biases, dtype=float),
        idf=vectorizer.idf,
        threshold=threshold,
        top_k=top_k,
    )


def evaluate_query_plan_predictor(
    model: QueryPlanPredictor,
    rows: Sequence[Dict[str, object]],
    threshold: float | None = None,
    top_k: int | None = None,
) -> Dict[str, object]:
    exact = 0
    f1s = []
    details = []
    for row in rows:
        gold = set(str(l) for l in row.get("labels", []) if str(l) in model.labels)
        pred = set(model.predict_labels(str(row.get("question", "")), threshold=threshold, top_k=top_k))
        tp = len(gold & pred)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
        exact += int(gold == pred)
        details.append(
            {
                "id": row.get("id", ""),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "gold_count": len(gold),
                "pred_count": len(pred),
            }
        )
    n = max(1, len(rows))
    return {
        "questions": len(rows),
        "labels": len(model.labels),
        "exact_match_rate": exact / n,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "details": details,
    }
