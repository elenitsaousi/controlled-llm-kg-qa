import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ranking.feature_config import FEATURE_NAMES
from ranking.np_tfidf_ranker import (
    NPTfidfRanker,
    QuestionItem,
    SimpleTfidf,
    _build_feature_row,
    _extra_feature_values,
    build_grouped_stratified_folds,
    compose_feature_names,
    load_training_data,
)


class XGBoostCandidateRanker:
    MODEL_TYPE = "xgboost_candidate_classifier_v1"

    def __init__(
        self,
        classifier,
        idf: Dict[str, float],
        feature_names: Sequence[str],
        disabled_feature_names: Sequence[str] | None = None,
        disabled_feature_prefixes: Sequence[str] | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> None:
        self.classifier = classifier
        self.vectorizer = SimpleTfidf(idf=idf)
        self.feature_names = list(feature_names)
        self.disabled_feature_names = list(disabled_feature_names or [])
        self.disabled_feature_prefixes = list(disabled_feature_prefixes or [])
        self.metadata = metadata or {}
        # Compatibility with evaluation/apply_ml_ranker_to_results.py feature check.
        self.scaler_mean = np.zeros(len(self.feature_names), dtype=float)

    def _row(
        self,
        question: str,
        query: str,
        base_features: Dict[str, float],
        query_plan_labels: Sequence[str],
        source: str,
        position: int,
        schema_dict: Dict[str, object] | None = None,
    ) -> List[float]:
        sim = self.vectorizer.similarity(question, query)
        extra = _extra_feature_values(
            question=question,
            query=query,
            query_plan_labels=query_plan_labels,
            source=source,
            position=position,
            schema_dict=schema_dict,
        )
        return _build_feature_row(
            base_features,
            sim,
            extra,
            FEATURE_NAMES,
            disabled_feature_names=self.disabled_feature_names,
            disabled_feature_prefixes=self.disabled_feature_prefixes,
        )

    def score_question_candidates(
        self,
        question: str,
        candidate_queries: Sequence[str],
        candidate_base_features: Sequence[Dict[str, float]],
        candidate_query_plan_labels: Sequence[Sequence[str]] | None = None,
        candidate_sources: Sequence[str] | None = None,
        schema_dict: Dict[str, object] | None = None,
    ) -> np.ndarray:
        if candidate_query_plan_labels is None:
            candidate_query_plan_labels = [[] for _ in candidate_queries]
        if candidate_sources is None:
            candidate_sources = ["llm" for _ in candidate_queries]
        rows = [
            self._row(question, query, base_features, labels, source, position, schema_dict)
            for position, (query, base_features, labels, source) in enumerate(
                zip(
                    candidate_queries,
                    candidate_base_features,
                    candidate_query_plan_labels,
                    candidate_sources,
                )
            )
        ]
        if not rows:
            return np.array([], dtype=float)
        X = np.asarray(rows, dtype=float)
        if hasattr(self.classifier, "predict_proba"):
            return np.asarray(self.classifier.predict_proba(X)[:, 1], dtype=float)
        return np.asarray(self.classifier.predict(X), dtype=float)

    def save(self, path: str, metadata: Dict[str, object] | None = None) -> None:
        payload = {
            "model_type": self.MODEL_TYPE,
            "classifier": self.classifier,
            "idf": self.vectorizer.idf,
            "feature_names": self.feature_names,
            "disabled_feature_names": self.disabled_feature_names,
            "disabled_feature_prefixes": self.disabled_feature_prefixes,
            "metadata": metadata or self.metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "XGBoostCandidateRanker":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("model_type") != cls.MODEL_TYPE:
            raise ValueError(f"Unsupported model_type: {payload.get('model_type')}")
        return cls(
            classifier=payload["classifier"],
            idf={k: float(v) for k, v in payload.get("idf", {}).items()},
            feature_names=list(payload.get("feature_names") or compose_feature_names()),
            disabled_feature_names=list(payload.get("disabled_feature_names") or []),
            disabled_feature_prefixes=list(payload.get("disabled_feature_prefixes") or []),
            metadata=dict(payload.get("metadata") or {}),
        )


def _fit_vectorizer(data: Dict[str, QuestionItem], qids: Sequence[str]) -> SimpleTfidf:
    texts: List[str] = []
    for qid in qids:
        item = data[qid]
        texts.append(item.question)
        texts.extend(cand.query for cand in item.candidates)
    vectorizer = SimpleTfidf()
    vectorizer.fit(texts)
    return vectorizer


def _ltr_trainable_qids(data: Dict[str, QuestionItem], qids: Iterable[str]) -> List[str]:
    trainable: List[str] = []
    for qid in qids:
        item = data[qid]
        labels = [int(cand.is_correct) for cand in item.candidates]
        if any(label == 1 for label in labels) and any(label == 0 for label in labels):
            trainable.append(qid)
    return trainable


def _build_rows_for_qids(
    data: Dict[str, QuestionItem],
    qids: Iterable[str],
    vectorizer: SimpleTfidf,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[Tuple[str, int]]]]:
    rows: List[List[float]] = []
    labels: List[int] = []
    by_qid: Dict[str, List[Tuple[str, int]]] = {}
    for qid in qids:
        item = data[qid]
        by_qid[qid] = []
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
            labels.append(int(cand.is_correct))
            by_qid[qid].append((cand.query_id, int(cand.is_correct)))
    X = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(compose_feature_names())))
    y = np.asarray(labels, dtype=int) if labels else np.zeros((0,), dtype=int)
    return X, y, by_qid


def _build_rows_groups_for_qids(
    data: Dict[str, QuestionItem],
    qids: Iterable[str],
    vectorizer: SimpleTfidf,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, List[Tuple[str, int]]]]:
    rows: List[List[float]] = []
    labels: List[int] = []
    groups: List[int] = []
    by_qid: Dict[str, List[Tuple[str, int]]] = {}
    for qid in qids:
        item = data[qid]
        by_qid[qid] = []
        group_size = 0
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
            labels.append(int(cand.is_correct))
            by_qid[qid].append((cand.query_id, int(cand.is_correct)))
            group_size += 1
        if group_size:
            groups.append(group_size)
    X = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(compose_feature_names())))
    y = np.asarray(labels, dtype=int) if labels else np.zeros((0,), dtype=int)
    return X, y, groups, by_qid


def _make_classifier(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    random_state: int,
    scale_pos_weight: float,
):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "xgboost is not available in this Python environment. Install it with "
            "`python -m pip install xgboost` or use the existing logistic ranker."
        ) from exc

    return XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(random_state),
        scale_pos_weight=float(scale_pos_weight),
        n_jobs=1,
    )


def _train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    seed: int,
):
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    scale_pos_weight = float(neg / pos) if pos else 1.0
    clf = _make_classifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X, y)
    return clf


def _make_ltr_ranker(
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    objective: str,
    random_state: int,
):
    try:
        from xgboost import XGBRanker
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "xgboost is not available in this Python environment. Install it with "
            "`python -m pip install xgboost` or use the existing logistic ranker."
        ) from exc

    return XGBRanker(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        objective=str(objective),
        eval_metric="ndcg",
        tree_method="hist",
        random_state=int(random_state),
        n_jobs=1,
    )


def _train_ltr_ranker(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[int],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    objective: str,
    seed: int,
):
    ranker = _make_ltr_ranker(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective=objective,
        random_state=seed,
    )
    ranker.fit(X, y, group=list(groups))
    return ranker


def train_final_xgboost_ranker(
    data: Dict[str, QuestionItem],
    *,
    n_estimators: int = 80,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 5.0,
    seed: int = 42,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> XGBoostCandidateRanker:
    qids = sorted(data.keys())
    vectorizer = _fit_vectorizer(data, qids)
    X, y, _ = _build_rows_for_qids(
        data,
        qids,
        vectorizer,
        disabled_feature_names=disabled_feature_names,
        disabled_feature_prefixes=disabled_feature_prefixes,
    )
    clf = _train_classifier(
        X,
        y,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        seed=seed,
    )
    return XGBoostCandidateRanker(
        classifier=clf,
        idf=vectorizer.idf,
        feature_names=compose_feature_names(),
        disabled_feature_names=disabled_feature_names,
        disabled_feature_prefixes=disabled_feature_prefixes,
    )


def train_final_xgboost_ltr_ranker(
    data: Dict[str, QuestionItem],
    *,
    n_estimators: int = 120,
    max_depth: int = 2,
    learning_rate: float = 0.03,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 10.0,
    objective: str = "rank:pairwise",
    seed: int = 42,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> XGBoostCandidateRanker:
    qids = _ltr_trainable_qids(data, sorted(data.keys()))
    if not qids:
        raise RuntimeError("No trainable LTR question groups found.")
    vectorizer = _fit_vectorizer(data, qids)
    X, y, groups, _ = _build_rows_groups_for_qids(
        data,
        qids,
        vectorizer,
        disabled_feature_names=disabled_feature_names,
        disabled_feature_prefixes=disabled_feature_prefixes,
    )
    ranker = _train_ltr_ranker(
        X,
        y,
        groups,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective=objective,
        seed=seed,
    )
    return XGBoostCandidateRanker(
        classifier=ranker,
        idf=vectorizer.idf,
        feature_names=compose_feature_names(),
        disabled_feature_names=disabled_feature_names,
        disabled_feature_prefixes=disabled_feature_prefixes,
        metadata={
            "training_mode": "learning_to_rank",
            "objective": objective,
            "trainable_question_groups": len(qids),
        },
    )


def cross_validate_xgboost_ranker(
    data: Dict[str, QuestionItem],
    *,
    n_folds: int = 5,
    seed: int = 42,
    n_estimators: int = 80,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 5.0,
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> Dict[str, object]:
    folds = build_grouped_stratified_folds(data, n_folds=n_folds, seed=seed)
    all_qids = sorted(data.keys())
    oof_rows: List[Dict[str, object]] = []
    fold_summaries: List[Dict[str, object]] = []

    for fold_idx, test_qids in enumerate(folds, start=1):
        test_set = set(test_qids)
        train_qids = [qid for qid in all_qids if qid not in test_set]
        vectorizer = _fit_vectorizer(data, train_qids)
        X_train, y_train, _ = _build_rows_for_qids(
            data,
            train_qids,
            vectorizer,
            disabled_feature_names=disabled_feature_names,
            disabled_feature_prefixes=disabled_feature_prefixes,
        )
        clf = _train_classifier(
            X_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            seed=seed + fold_idx,
        )
        model = XGBoostCandidateRanker(
            classifier=clf,
            idf=vectorizer.idf,
            feature_names=compose_feature_names(),
            disabled_feature_names=disabled_feature_names,
            disabled_feature_prefixes=disabled_feature_prefixes,
        )

        fold_top1: List[int] = []
        fold_any: List[int] = []
        fold_baseline: List[int] = []
        for qid in test_qids:
            item = data[qid]
            scores = model.score_question_candidates(
                item.question,
                [cand.query for cand in item.candidates],
                [cand.features for cand in item.candidates],
                candidate_query_plan_labels=[cand.query_plan_labels or [] for cand in item.candidates],
                candidate_sources=[cand.source for cand in item.candidates],
            )
            ranked = sorted(
                zip(item.candidates, scores),
                key=lambda row: float(row[1]),
                reverse=True,
            )
            if not ranked:
                continue
            top1_correct = int(ranked[0][0].is_correct == 1)
            any_correct = int(any(cand.is_correct == 1 for cand in item.candidates))
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
                    "top_query_id": ranked[0][0].query_id,
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

    top1_values = [int(row["top1_correct"]) for row in oof_rows]
    any_values = [int(row["any_correct"]) for row in oof_rows]
    base_values = [int(row["baseline_top1_correct"]) for row in oof_rows]
    return {
        "config": {
            "model": XGBoostCandidateRanker.MODEL_TYPE,
            "n_folds": n_folds,
            "seed": seed,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "feature_names": compose_feature_names(),
            "disabled_feature_names": list(disabled_feature_names or []),
            "disabled_feature_prefixes": list(disabled_feature_prefixes or []),
        },
        "overall": {
            "n_questions": len(oof_rows),
            "top1_correct": int(sum(top1_values)),
            "any_correct": int(sum(any_values)),
            "baseline_top1_correct": int(sum(base_values)),
            "top1_rate": float(np.mean(top1_values)) if top1_values else 0.0,
            "any_rate": float(np.mean(any_values)) if any_values else 0.0,
            "baseline_top1_rate": float(np.mean(base_values)) if base_values else 0.0,
        },
        "folds": fold_summaries,
        "oof_predictions": sorted(oof_rows, key=lambda row: str(row["qid"])),
    }


def cross_validate_xgboost_ltr_ranker(
    data: Dict[str, QuestionItem],
    *,
    n_folds: int = 5,
    seed: int = 42,
    n_estimators: int = 120,
    max_depth: int = 2,
    learning_rate: float = 0.03,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 10.0,
    objective: str = "rank:pairwise",
    disabled_feature_names: Sequence[str] | None = None,
    disabled_feature_prefixes: Sequence[str] | None = None,
) -> Dict[str, object]:
    folds = build_grouped_stratified_folds(data, n_folds=n_folds, seed=seed)
    all_qids = sorted(data.keys())
    oof_rows: List[Dict[str, object]] = []
    fold_summaries: List[Dict[str, object]] = []

    for fold_idx, test_qids in enumerate(folds, start=1):
        test_set = set(test_qids)
        train_qids = _ltr_trainable_qids(
            data,
            [qid for qid in all_qids if qid not in test_set],
        )
        if not train_qids:
            raise RuntimeError(f"No trainable LTR question groups found for fold {fold_idx}.")
        vectorizer = _fit_vectorizer(data, train_qids)
        X_train, y_train, groups, _ = _build_rows_groups_for_qids(
            data,
            train_qids,
            vectorizer,
            disabled_feature_names=disabled_feature_names,
            disabled_feature_prefixes=disabled_feature_prefixes,
        )
        ranker = _train_ltr_ranker(
            X_train,
            y_train,
            groups,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective=objective,
            seed=seed + fold_idx,
        )
        model = XGBoostCandidateRanker(
            classifier=ranker,
            idf=vectorizer.idf,
            feature_names=compose_feature_names(),
            disabled_feature_names=disabled_feature_names,
            disabled_feature_prefixes=disabled_feature_prefixes,
            metadata={
                "training_mode": "learning_to_rank",
                "objective": objective,
                "trainable_question_groups": len(train_qids),
            },
        )

        fold_top1: List[int] = []
        fold_any: List[int] = []
        fold_baseline: List[int] = []
        for qid in test_qids:
            item = data[qid]
            scores = model.score_question_candidates(
                item.question,
                [cand.query for cand in item.candidates],
                [cand.features for cand in item.candidates],
                candidate_query_plan_labels=[cand.query_plan_labels or [] for cand in item.candidates],
                candidate_sources=[cand.source for cand in item.candidates],
            )
            ranked = sorted(
                zip(item.candidates, scores),
                key=lambda row: float(row[1]),
                reverse=True,
            )
            if not ranked:
                continue
            top1_correct = int(ranked[0][0].is_correct == 1)
            any_correct = int(any(cand.is_correct == 1 for cand in item.candidates))
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
                    "top_query_id": ranked[0][0].query_id,
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

    top1_values = [int(row["top1_correct"]) for row in oof_rows]
    any_values = [int(row["any_correct"]) for row in oof_rows]
    base_values = [int(row["baseline_top1_correct"]) for row in oof_rows]
    return {
        "config": {
            "model": "xgboost_learning_to_rank_v1",
            "wrapper_model": XGBoostCandidateRanker.MODEL_TYPE,
            "objective": objective,
            "n_folds": n_folds,
            "seed": seed,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "feature_names": compose_feature_names(),
            "disabled_feature_names": list(disabled_feature_names or []),
            "disabled_feature_prefixes": list(disabled_feature_prefixes or []),
        },
        "overall": {
            "n_questions": len(oof_rows),
            "top1_correct": int(sum(top1_values)),
            "any_correct": int(sum(any_values)),
            "baseline_top1_correct": int(sum(base_values)),
            "top1_rate": float(np.mean(top1_values)) if top1_values else 0.0,
            "any_rate": float(np.mean(any_values)) if any_values else 0.0,
            "baseline_top1_rate": float(np.mean(base_values)) if base_values else 0.0,
        },
        "folds": fold_summaries,
        "oof_predictions": sorted(oof_rows, key=lambda row: str(row["qid"])),
    }


__all__ = [
    "XGBoostCandidateRanker",
    "cross_validate_xgboost_ltr_ranker",
    "cross_validate_xgboost_ranker",
    "load_training_data",
    "train_final_xgboost_ltr_ranker",
    "train_final_xgboost_ranker",
]
