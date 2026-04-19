#!/usr/bin/env python3
"""
Validate entropy-based ambiguity definition and gated query-selection policy
for the Infineon benchmark using grouped cross-validation.

Key properties:
- Grouped folds by query family (reduces near-duplicate leakage).
- Thresholds (tau1, tau2) tuned on train fold only.
- Test fold kept strictly holdout.
- Compares three policies:
  1) no_ml  : keep original candidate order (LLM-first baseline)
  2) ml_all : always rank by ML model score
  3) gated  : apply ML only on selected ambiguity regimes (default: mid)
- Reports ambiguity-classification quality against dataset labels (if present).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.feature_config import FEATURE_NAMES
from ranking.np_tfidf_ranker import (
    NPTfidfRanker,
    QuestionCandidate,
    QuestionItem,
    SimpleTfidf,
    build_grouped_stratified_folds,
    train_logistic,
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
    return "fam_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:16]


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_label(label: str) -> str:
    x = (label or "").strip().lower()
    if x == "medium":
        return "mid"
    return x


@dataclass
class FoldQuestionResult:
    qid: str
    fold: int
    true_ambiguity: str
    predicted_regime: str
    entropy: float
    no_ml_top1: int
    ml_top1: int
    gated_top1: int
    any_correct: int
    n_candidates: int
    ml_used: int


def _load_questions(
    path: str,
    include_gold: bool,
    min_candidates: int,
) -> Tuple[Dict[str, QuestionItem], Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    questions: Dict[str, QuestionItem] = {}
    dropped = Counter()

    for qid, rows in raw.items():
        if not rows:
            dropped["empty_rows"] += 1
            continue

        q0 = rows[0]
        question = str(q0.get("question", "")).strip()
        ambiguity = _normalize_label(str(q0.get("ambiguity_label", "unknown"))) or "unknown"

        family = str(q0.get("family", "")).strip()
        if not family:
            family = _query_family_signature(
                str(q0.get("gold_query", "")).strip() or str(q0.get("query", "")).strip()
            )
            if family == "fam_d41d8cd98f00b204":
                family = qid

        candidates: List[QuestionCandidate] = []
        for idx, row in enumerate(rows):
            source = str(row.get("source", "")).strip().lower()
            if source == "gold" and not include_gold:
                continue

            query = str(row.get("query", "")).strip()
            if not query:
                continue

            query_id = str(row.get("query_id", "")).strip() or f"{qid}_cand_{idx}"
            features_raw = row.get("features", {})
            features = {
                name: _to_float(features_raw.get(name, 0.0))
                for name in FEATURE_NAMES
            }

            candidates.append(
                QuestionCandidate(
                    query_id=query_id,
                    query=query,
                    is_correct=int(row.get("is_correct", 0)),
                    is_valid=int(row.get("is_valid", 0)),
                    features=features,
                )
            )

        if len(candidates) < min_candidates:
            dropped["too_few_candidates"] += 1
            continue

        questions[qid] = QuestionItem(
            qid=qid,
            question=question,
            ambiguity_label=ambiguity,
            family=family,
            candidates=candidates,
        )

    return questions, dict(dropped)


def _fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _scale(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _build_train_matrix(
    data: Dict[str, QuestionItem],
    qids: Sequence[str],
    vectorizer: SimpleTfidf,
) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    labels: List[int] = []

    for qid in qids:
        item = data[qid]
        for cand in item.candidates:
            sim = vectorizer.similarity(item.question, cand.query)
            feat_row = [float(cand.features.get(name, 0.0)) for name in FEATURE_NAMES]
            feat_row.append(sim)
            rows.append(feat_row)
            labels.append(int(cand.is_correct))

    if not rows:
        return np.zeros((0, len(FEATURE_NAMES) + 1), dtype=float), np.zeros((0,), dtype=int)

    return np.array(rows, dtype=float), np.array(labels, dtype=int)


def _fit_fold_model(
    data: Dict[str, QuestionItem],
    train_qids: Sequence[str],
    lr: float,
    reg: float,
    epochs: int,
) -> NPTfidfRanker:
    texts: List[str] = []
    for qid in train_qids:
        item = data[qid]
        texts.append(item.question)
        texts.extend(c.query for c in item.candidates)

    vectorizer = SimpleTfidf()
    vectorizer.fit(texts)

    X_train, y_train = _build_train_matrix(data, train_qids, vectorizer)
    if X_train.shape[0] == 0:
        # Degenerate fallback model (never expected in normal runs).
        nfeat = len(FEATURE_NAMES) + 1
        return NPTfidfRanker(
            feature_names=list(FEATURE_NAMES) + ["tfidf_similarity"],
            weights=np.zeros(nfeat, dtype=float),
            bias=0.0,
            scaler_mean=np.zeros(nfeat, dtype=float),
            scaler_std=np.ones(nfeat, dtype=float),
            idf=vectorizer.idf,
        )

    mean, std = _fit_scaler(X_train)
    Xs = _scale(X_train, mean, std)
    w, b = train_logistic(Xs, y_train, lr=lr, reg=reg, epochs=epochs)

    return NPTfidfRanker(
        feature_names=list(FEATURE_NAMES) + ["tfidf_similarity"],
        weights=w,
        bias=b,
        scaler_mean=mean,
        scaler_std=std,
        idf=vectorizer.idf,
    )


def _schema_signal(features: Dict[str, float]) -> float:
    # Lightweight structural relevance score (non-ML), used for ambiguity estimation
    # when entropy_source=schema.
    return (
        2.0 * features.get("entity_coverage", 0.0)
        + 1.5 * features.get("relation_coverage", 0.0)
        + 1.0 * features.get("expected_intermediate_coverage", 0.0)
        + 0.3 * features.get("has_where", 0.0)
        + 0.3 * features.get("has_type", 0.0)
        + 0.2 * features.get("has_aggregation", 0.0)
        - 1.2 * features.get("unexpected_label_ratio", 0.0)
        - 0.8 * features.get("invalid_predicate_count", 0.0)
        - 0.8 * features.get("unused_select_vars", 0.0)
        - 0.05 * features.get("rel_count", 0.0)
    )


def _entropy_from_scores(scores: Sequence[float], normalize: bool = True) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    if arr.size <= 1:
        return 0.0
    arr = arr - np.max(arr)
    exp_scores = np.exp(arr)
    probs = exp_scores / np.maximum(exp_scores.sum(), 1e-12)
    h = float(-np.sum(probs * np.log(probs + 1e-12)))
    if normalize and arr.size > 1:
        h /= math.log(float(arr.size))
    return h


def _regime_from_entropy(h: float, tau1: float, tau2: float) -> str:
    if h <= tau1:
        return "low"
    if h <= tau2:
        return "mid"
    return "high"


def _evaluate_question(
    item: QuestionItem,
    model: NPTfidfRanker,
    entropy_source: str,
) -> Dict[str, object]:
    queries = [c.query for c in item.candidates]
    base_features = [c.features for c in item.candidates]
    ml_scores = model.score_question_candidates(item.question, queries, base_features)

    if ml_scores.size == 0:
        # Degenerate case; treat as no-confidence scores.
        ml_scores = np.zeros(len(item.candidates), dtype=float)

    schema_scores = np.array([_schema_signal(c.features) for c in item.candidates], dtype=float)

    if entropy_source == "schema":
        entropy = _entropy_from_scores(schema_scores.tolist(), normalize=True)
    elif entropy_source == "ml":
        entropy = _entropy_from_scores(ml_scores.tolist(), normalize=True)
    else:
        raise ValueError(f"Unsupported entropy_source={entropy_source}")

    no_ml_idx = 0
    ml_idx = int(np.argmax(ml_scores))

    no_ml_top1 = int(item.candidates[no_ml_idx].is_correct == 1)
    ml_top1 = int(item.candidates[ml_idx].is_correct == 1)
    any_correct = int(any(c.is_correct == 1 for c in item.candidates))

    return {
        "entropy": entropy,
        "no_ml_top1": no_ml_top1,
        "ml_top1": ml_top1,
        "any_correct": any_correct,
        "n_candidates": len(item.candidates),
    }


def _parse_float_csv(text: str) -> List[float]:
    values = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        values.append(float(tok))
    return values


def _parse_label_set(text: str) -> List[str]:
    allowed = {"low", "mid", "high"}
    labels = []
    for tok in text.split(","):
        lab = _normalize_label(tok)
        if not lab:
            continue
        if lab not in allowed:
            raise ValueError(
                f"Invalid label '{tok}'. Allowed labels: low,mid,high"
            )
        if lab not in labels:
            labels.append(lab)
    if not labels:
        raise ValueError("At least one label is required for --ml-regimes")
    return labels


def _tune_thresholds(
    train_rows: Sequence[Dict[str, object]],
    q1_grid: Sequence[float],
    q2_grid: Sequence[float],
    ml_regimes: Sequence[str],
) -> Dict[str, float]:
    ent = np.array([float(r["entropy"]) for r in train_rows], dtype=float)
    if ent.size == 0:
        return {
            "tau1": 0.33,
            "tau2": 0.66,
            "train_gated_top1": 0.0,
            "train_ml_usage": 0.0,
        }

    best = None
    eps = 1e-12

    for q1 in q1_grid:
        tau1 = float(np.quantile(ent, q1))
        for q2 in q2_grid:
            if q2 <= q1:
                continue
            tau2 = float(np.quantile(ent, q2))

            gated_values = []
            ml_usage = []
            for r in train_rows:
                reg = _regime_from_entropy(float(r["entropy"]), tau1, tau2)
                use_ml = int(reg in ml_regimes)
                val = int(r["ml_top1"]) if use_ml else int(r["no_ml_top1"])
                gated_values.append(val)
                ml_usage.append(use_ml)

            score = float(np.mean(gated_values)) if gated_values else 0.0
            usage = float(np.mean(ml_usage)) if ml_usage else 0.0

            cand = {
                "tau1": tau1,
                "tau2": tau2,
                "train_gated_top1": score,
                "train_ml_usage": usage,
            }

            if best is None:
                best = cand
                continue

            # Primary: maximize train gated top1.
            # Tie-breaker: lower ML usage (more conservative policy).
            # Secondary tie-breaker: wider middle band stability (tau2-tau1).
            if score > best["train_gated_top1"] + eps:
                best = cand
            elif abs(score - best["train_gated_top1"]) <= eps:
                if usage < best["train_ml_usage"] - eps:
                    best = cand
                elif abs(usage - best["train_ml_usage"]) <= eps:
                    if (tau2 - tau1) > (best["tau2"] - best["tau1"]):
                        best = cand

    assert best is not None
    return best


def _bootstrap_ci(values: Sequence[int], seed: int, n_boot: int = 2000) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(sample.mean()))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = ("low", "mid", "high"),
) -> Dict[str, object]:
    assert len(y_true) == len(y_pred)
    n = len(y_true)

    confusion: Dict[str, Dict[str, int]] = {
        t: {p: 0 for p in labels} for t in labels
    }

    valid_idx = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t not in labels:
            continue
        if p not in labels:
            continue
        confusion[t][p] += 1
        valid_idx.append(i)

    valid_n = len(valid_idx)
    if valid_n == 0:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_label": {},
            "confusion_matrix": confusion,
        }

    acc = float(sum(confusion[l][l] for l in labels) / valid_n)

    per_label = {}
    f1s = []
    for lab in labels:
        tp = confusion[lab][lab]
        fp = sum(confusion[t][lab] for t in labels if t != lab)
        fn = sum(confusion[lab][p] for p in labels if p != lab)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

        per_label[lab] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(sum(confusion[lab].values())),
        }

    return {
        "n": valid_n,
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def _aggregate_policy(
    rows: Sequence[FoldQuestionResult],
    attr: str,
) -> Dict[str, object]:
    vals = [int(getattr(r, attr)) for r in rows]
    return {
        "correct": int(sum(vals)),
        "total": len(vals),
        "rate": float(np.mean(vals)) if vals else 0.0,
        "ci95": _bootstrap_ci(vals, seed=42),
    }


def _group_summary(rows: Sequence[FoldQuestionResult], key: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    groups: Dict[str, List[FoldQuestionResult]] = defaultdict(list)
    for r in rows:
        groups[getattr(r, key)].append(r)

    for g, rs in sorted(groups.items()):
        out[g] = {
            "n": len(rs),
            "no_ml_top1_rate": float(np.mean([x.no_ml_top1 for x in rs])) if rs else 0.0,
            "ml_top1_rate": float(np.mean([x.ml_top1 for x in rs])) if rs else 0.0,
            "gated_top1_rate": float(np.mean([x.gated_top1 for x in rs])) if rs else 0.0,
            "any_correct_rate": float(np.mean([x.any_correct for x in rs])) if rs else 0.0,
            "ml_usage_rate": float(np.mean([x.ml_used for x in rs])) if rs else 0.0,
        }
    return out


def run_validation(args: argparse.Namespace) -> Dict[str, object]:
    questions, dropped = _load_questions(
        path=args.training_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )

    if not questions:
        raise RuntimeError(
            "No usable questions found. Check training data and --include-gold/--min-candidates settings."
        )

    folds = build_grouped_stratified_folds(questions, n_folds=args.folds, seed=args.seed)
    all_qids = sorted(questions.keys())

    q1_grid = _parse_float_csv(args.q1_grid)
    q2_grid = _parse_float_csv(args.q2_grid)
    ml_regimes = _parse_label_set(args.ml_regimes)

    fold_summaries = []
    all_rows: List[FoldQuestionResult] = []

    for fold_idx, test_qids in enumerate(folds, start=1):
        test_set = set(test_qids)
        train_qids = [qid for qid in all_qids if qid not in test_set]

        model = _fit_fold_model(
            questions,
            train_qids,
            lr=args.lr,
            reg=args.reg,
            epochs=args.epochs,
        )

        train_rows = []
        for qid in train_qids:
            item = questions[qid]
            ev = _evaluate_question(item, model, entropy_source=args.entropy_source)
            train_rows.append(ev)

        tuned = _tune_thresholds(
            train_rows,
            q1_grid=q1_grid,
            q2_grid=q2_grid,
            ml_regimes=ml_regimes,
        )
        tau1 = tuned["tau1"]
        tau2 = tuned["tau2"]

        fold_out = []
        for qid in test_qids:
            item = questions[qid]
            ev = _evaluate_question(item, model, entropy_source=args.entropy_source)
            reg = _regime_from_entropy(float(ev["entropy"]), tau1, tau2)
            use_ml = int(reg in ml_regimes)
            gated = int(ev["ml_top1"]) if use_ml else int(ev["no_ml_top1"])

            row = FoldQuestionResult(
                qid=qid,
                fold=fold_idx,
                true_ambiguity=_normalize_label(item.ambiguity_label) or "unknown",
                predicted_regime=reg,
                entropy=float(ev["entropy"]),
                no_ml_top1=int(ev["no_ml_top1"]),
                ml_top1=int(ev["ml_top1"]),
                gated_top1=gated,
                any_correct=int(ev["any_correct"]),
                n_candidates=int(ev["n_candidates"]),
                ml_used=use_ml,
            )
            fold_out.append(row)
            all_rows.append(row)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_questions": len(train_qids),
                "test_questions": len(test_qids),
                "tau1": tau1,
                "tau2": tau2,
                "train_gated_top1": tuned["train_gated_top1"],
                "train_ml_usage": tuned["train_ml_usage"],
                "test_no_ml_top1": float(np.mean([r.no_ml_top1 for r in fold_out])) if fold_out else 0.0,
                "test_ml_top1": float(np.mean([r.ml_top1 for r in fold_out])) if fold_out else 0.0,
                "test_gated_top1": float(np.mean([r.gated_top1 for r in fold_out])) if fold_out else 0.0,
                "test_any_correct": float(np.mean([r.any_correct for r in fold_out])) if fold_out else 0.0,
                "test_ml_usage": float(np.mean([r.ml_used for r in fold_out])) if fold_out else 0.0,
            }
        )

    # Overall policy metrics
    no_ml = _aggregate_policy(all_rows, "no_ml_top1")
    ml_all = _aggregate_policy(all_rows, "ml_top1")
    gated = _aggregate_policy(all_rows, "gated_top1")
    any_correct = _aggregate_policy(all_rows, "any_correct")

    # Ambiguity classification metrics (against dataset labels, if present)
    y_true = [r.true_ambiguity for r in all_rows]
    y_pred = [r.predicted_regime for r in all_rows]
    clf = _classification_metrics(y_true, y_pred)

    # Group summaries
    by_true = _group_summary(all_rows, "true_ambiguity")
    by_pred = _group_summary(all_rows, "predicted_regime")

    overall = {
        "questions_total": len(all_rows),
        "dropped": dropped,
        "no_ml_top1": no_ml,
        "ml_all_top1": ml_all,
        "gated_top1": gated,
        "any_correct": any_correct,
        "delta_gated_vs_no_ml": gated["rate"] - no_ml["rate"],
        "delta_gated_vs_ml_all": gated["rate"] - ml_all["rate"],
    }

    result = {
        "config": {
            "training_data": args.training_data,
            "folds": args.folds,
            "seed": args.seed,
            "include_gold": args.include_gold,
            "min_candidates": args.min_candidates,
            "lr": args.lr,
            "reg": args.reg,
            "epochs": args.epochs,
            "entropy_source": args.entropy_source,
            "ml_regimes": ml_regimes,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
        },
        "overall": overall,
        "ambiguity_classification": clf,
        "per_true_ambiguity": by_true,
        "per_predicted_regime": by_pred,
        "folds": fold_summaries,
        "per_question": [
            {
                "qid": r.qid,
                "fold": r.fold,
                "true_ambiguity": r.true_ambiguity,
                "predicted_regime": r.predicted_regime,
                "entropy": r.entropy,
                "n_candidates": r.n_candidates,
                "no_ml_top1": r.no_ml_top1,
                "ml_top1": r.ml_top1,
                "gated_top1": r.gated_top1,
                "any_correct": r.any_correct,
                "ml_used": r.ml_used,
            }
            for r in all_rows
        ],
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate entropy-based ambiguity definition and gated policy with grouped CV."
    )
    parser.add_argument(
        "--training-data",
        default="ranking/infineon_training_data_100.json",
        help="Candidate-level labeled training data JSON.",
    )
    parser.add_argument(
        "--out",
        default="results/infineon_ambiguity_gated_cv_100.json",
        help="Where to write CV validation report JSON.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=2500)

    parser.add_argument(
        "--entropy-source",
        choices=["schema", "ml"],
        default="schema",
        help="Score source used to compute ambiguity entropy.",
    )
    parser.add_argument(
        "--ml-regimes",
        default="mid",
        help="Comma-separated regimes where ML is enabled (e.g., mid or low,mid).",
    )

    parser.add_argument(
        "--q1-grid",
        default="0.10,0.20,0.25,0.30,0.33,0.40",
        help="Quantile grid for tau1 tuning.",
    )
    parser.add_argument(
        "--q2-grid",
        default="0.60,0.66,0.70,0.75,0.80,0.90",
        help="Quantile grid for tau2 tuning.",
    )

    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Include gold candidate rows in evaluation (not recommended for unbiased validation).",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=1,
        help="Minimum candidate count required per question after filtering.",
    )

    args = parser.parse_args()

    result = run_validation(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    overall = result["overall"]
    clf = result["ambiguity_classification"]

    print("===== INFINEON AMBIGUITY CV VALIDATION =====")
    print(f"Questions used: {overall['questions_total']}")
    print(f"Dropped: {overall['dropped']}")
    print(
        f"No-ML Top1:  {overall['no_ml_top1']['correct']}/{overall['no_ml_top1']['total']} "
        f"({overall['no_ml_top1']['rate']:.3f})"
    )
    print(
        f"ML-all Top1: {overall['ml_all_top1']['correct']}/{overall['ml_all_top1']['total']} "
        f"({overall['ml_all_top1']['rate']:.3f})"
    )
    print(
        f"Gated Top1:  {overall['gated_top1']['correct']}/{overall['gated_top1']['total']} "
        f"({overall['gated_top1']['rate']:.3f})"
    )
    print(
        f"Any-correct upper bound: {overall['any_correct']['correct']}/{overall['any_correct']['total']} "
        f"({overall['any_correct']['rate']:.3f})"
    )
    print(f"Δ Gated vs No-ML:  {overall['delta_gated_vs_no_ml']:+.3f}")
    print(f"Δ Gated vs ML-all: {overall['delta_gated_vs_ml_all']:+.3f}")

    if clf["n"]:
        print("\nAmbiguity classification vs dataset labels:")
        print(f"  n={clf['n']} accuracy={clf['accuracy']:.3f} macro_f1={clf['macro_f1']:.3f}")

    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
