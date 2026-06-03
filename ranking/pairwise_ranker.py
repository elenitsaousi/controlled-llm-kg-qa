import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ranking.feature_config import FEATURE_NAMES
from ranking.np_tfidf_ranker import (
    QuestionItem,
    SimpleTfidf,
    _build_feature_row,
    _extra_feature_values,
    _fit_scaler,
    _scale,
    _sigmoid,
    compose_feature_names,
    load_training_data,
)


def _candidate_feature_rows(
    item: QuestionItem,
    vectorizer: SimpleTfidf,
) -> np.ndarray:
    rows: List[List[float]] = []
    for position, cand in enumerate(item.candidates):
        sim = vectorizer.similarity(item.question, cand.query)
        extra = _extra_feature_values(
            question=item.question,
            query=cand.query,
            query_plan_labels=cand.query_plan_labels,
            source=cand.source,
            position=position,
        )
        rows.append(_build_feature_row(cand.features, sim, extra, FEATURE_NAMES))
    if not rows:
        return np.zeros((0, len(compose_feature_names())), dtype=float)
    return np.array(rows, dtype=float)


def _fit_vectorizer(data: Dict[str, QuestionItem], qids: Sequence[str]) -> SimpleTfidf:
    texts: List[str] = []
    for qid in qids:
        item = data[qid]
        texts.append(item.question)
        texts.extend(c.query for c in item.candidates)
    vectorizer = SimpleTfidf()
    vectorizer.fit(texts)
    return vectorizer


def _candidate_rows_for_qids(
    data: Dict[str, QuestionItem],
    qids: Sequence[str],
    vectorizer: SimpleTfidf,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    all_rows: List[np.ndarray] = []
    by_qid: Dict[str, np.ndarray] = {}
    for qid in qids:
        rows = _candidate_feature_rows(data[qid], vectorizer)
        by_qid[qid] = rows
        if len(rows):
            all_rows.append(rows)
    if not all_rows:
        return np.zeros((0, len(compose_feature_names())), dtype=float), by_qid
    return np.vstack(all_rows), by_qid


def _pairwise_training_rows(
    data: Dict[str, QuestionItem],
    qids: Sequence[str],
    rows_by_qid_scaled: Dict[str, np.ndarray],
    max_pairs_per_question: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    pair_rows: List[np.ndarray] = []
    labels: List[int] = []
    rng = np.random.default_rng(42)
    for qid in qids:
        item = data[qid]
        rows = rows_by_qid_scaled.get(qid)
        if rows is None or len(rows) == 0:
            continue
        correct_idx = [idx for idx, cand in enumerate(item.candidates) if cand.is_correct == 1]
        wrong_idx = [idx for idx, cand in enumerate(item.candidates) if cand.is_correct != 1]
        pairs = [(cidx, widx) for cidx in correct_idx for widx in wrong_idx]
        if not pairs:
            continue
        if len(pairs) > max_pairs_per_question:
            selected = rng.choice(len(pairs), size=max_pairs_per_question, replace=False)
            pairs = [pairs[int(i)] for i in selected]
        for cidx, widx in pairs:
            diff = rows[cidx] - rows[widx]
            pair_rows.append(diff)
            labels.append(1)
            pair_rows.append(-diff)
            labels.append(0)
    if not pair_rows:
        return np.zeros((0, len(compose_feature_names())), dtype=float), np.zeros((0,), dtype=int)
    return np.array(pair_rows, dtype=float), np.array(labels, dtype=int)


def train_pairwise_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    reg: float = 0.02,
    epochs: int = 2500,
) -> np.ndarray:
    if X.size == 0:
        return np.zeros((X.shape[1] if X.ndim == 2 else len(compose_feature_names())), dtype=float)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    for _ in range(int(epochs)):
        logits = X @ w
        probs = _sigmoid(logits)
        err = probs - y
        grad = (X.T @ err) / float(n) + float(reg) * w
        w -= float(lr) * grad
    return w


class PairwiseRanker:
    MODEL_TYPE = "pairwise_np_logreg_v1"

    def __init__(
        self,
        feature_names: Sequence[str],
        weights: np.ndarray,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        idf: Dict[str, float],
    ):
        self.feature_names = list(feature_names)
        self.weights = np.array(weights, dtype=float)
        self.scaler_mean = np.array(scaler_mean, dtype=float)
        self.scaler_std = np.array(scaler_std, dtype=float)
        self.vectorizer = SimpleTfidf(idf=idf)

    def score_question_candidates(
        self,
        question: str,
        candidate_queries: Sequence[str],
        candidate_base_features: Sequence[Dict[str, float]],
        candidate_query_plan_labels: Sequence[Sequence[str]] | None = None,
        candidate_sources: Sequence[str] | None = None,
    ) -> np.ndarray:
        if candidate_query_plan_labels is None:
            candidate_query_plan_labels = [[] for _ in candidate_queries]
        if candidate_sources is None:
            candidate_sources = ["llm" for _ in candidate_queries]
        rows = []
        for position, (query, base_features, labels, source) in enumerate(
            zip(candidate_queries, candidate_base_features, candidate_query_plan_labels, candidate_sources)
        ):
            sim = self.vectorizer.similarity(question, query)
            extra = _extra_feature_values(
                question=question,
                query=query,
                query_plan_labels=labels,
                source=source,
                position=position,
            )
            rows.append(_build_feature_row(base_features, sim, extra, FEATURE_NAMES))
        if not rows:
            return np.array([], dtype=float)
        X = _scale(np.array(rows, dtype=float), self.scaler_mean, self.scaler_std)
        return X @ self.weights

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.MODEL_TYPE,
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "idf": self.vectorizer.idf,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PairwiseRanker":
        if data.get("model_type") != cls.MODEL_TYPE:
            raise ValueError(f"Unsupported model_type: {data.get('model_type')}")
        return cls(
            feature_names=list(data["feature_names"]),
            weights=np.array(data["weights"], dtype=float),
            scaler_mean=np.array(data["scaler_mean"], dtype=float),
            scaler_std=np.array(data["scaler_std"], dtype=float),
            idf={k: float(v) for k, v in data.get("idf", {}).items()},
        )

    @classmethod
    def load(cls, path: str) -> "PairwiseRanker":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str, metadata: Dict[str, object] | None = None) -> None:
        payload = self.to_dict()
        payload["metadata"] = metadata or {}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")


def train_final_pairwise_ranker(
    data: Dict[str, QuestionItem],
    lr: float = 0.05,
    reg: float = 0.02,
    epochs: int = 2500,
    max_pairs_per_question: int = 64,
) -> PairwiseRanker:
    qids = sorted(data.keys())
    vectorizer = _fit_vectorizer(data, qids)
    X_all, rows_by_qid = _candidate_rows_for_qids(data, qids, vectorizer)
    mean, std = _fit_scaler(X_all)
    rows_by_qid_scaled = {qid: _scale(rows, mean, std) for qid, rows in rows_by_qid.items()}
    X_pair, y_pair = _pairwise_training_rows(
        data,
        qids,
        rows_by_qid_scaled,
        max_pairs_per_question=max_pairs_per_question,
    )
    weights = train_pairwise_logistic(X_pair, y_pair, lr=lr, reg=reg, epochs=epochs)
    return PairwiseRanker(
        feature_names=compose_feature_names(),
        weights=weights,
        scaler_mean=mean,
        scaler_std=std,
        idf=vectorizer.idf,
    )


def score_items(
    data: Dict[str, QuestionItem],
    qids: Sequence[str],
    model: PairwiseRanker,
) -> Dict[str, List[Tuple[str, float, int]]]:
    out: Dict[str, List[Tuple[str, float, int]]] = {}
    for qid in qids:
        item = data[qid]
        scores = model.score_question_candidates(
            item.question,
            [cand.query for cand in item.candidates],
            [cand.features for cand in item.candidates],
            candidate_query_plan_labels=[cand.query_plan_labels or [] for cand in item.candidates],
            candidate_sources=[cand.source for cand in item.candidates],
        )
        out[qid] = [
            (cand.query_id, float(score), int(cand.is_correct))
            for cand, score in zip(item.candidates, scores)
        ]
    return out
