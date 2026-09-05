#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.np_tfidf_ranker import build_grouped_stratified_folds, load_training_data
from ranking.pairwise_ranker import (
    _candidate_rows_for_qids,
    _fit_vectorizer,
    _pairwise_training_rows,
    compose_pairwise_feature_names,
    score_items,
    train_final_pairwise_ranker,
    train_pairwise_logistic,
    PairwiseRanker,
)
from ranking.np_tfidf_ranker import _fit_scaler, _scale


def _evaluate_scores(data, qids, score_rows):
    rows = []
    top1 = []
    any_correct = []
    baseline = []
    for qid in qids:
        item = data[qid]
        scored = score_rows.get(qid) or []
        if not scored:
            continue
        ranked = sorted(scored, key=lambda row: row[1], reverse=True)
        top = int(ranked[0][2] == 1)
        anyc = int(any(c.is_correct == 1 for c in item.candidates))
        base = int(item.candidates[0].is_correct == 1)
        top1.append(top)
        any_correct.append(anyc)
        baseline.append(base)
        rows.append(
            {
                "qid": qid,
                "top1_correct": top,
                "any_correct": anyc,
                "baseline_top1_correct": base,
                "top_query_id": ranked[0][0],
                "top_score": float(ranked[0][1]),
            }
        )
    return {
        "n_questions": len(rows),
        "top1_correct": int(sum(top1)),
        "any_correct": int(sum(any_correct)),
        "baseline_top1_correct": int(sum(baseline)),
        "top1_rate": float(np.mean(top1)) if top1 else 0.0,
        "any_rate": float(np.mean(any_correct)) if any_correct else 0.0,
        "baseline_top1_rate": float(np.mean(baseline)) if baseline else 0.0,
        "rows": rows,
    }


def cross_validate_pairwise(data, folds=5, seed=42, lr=0.05, reg=0.02, epochs=2500, max_pairs_per_question=64):
    fold_qids = build_grouped_stratified_folds(data, n_folds=folds, seed=seed)
    all_qids = sorted(data.keys())
    all_rows = []
    summaries = []
    for fold_idx, test_qids in enumerate(fold_qids, start=1):
        test_set = set(test_qids)
        train_qids = [qid for qid in all_qids if qid not in test_set]
        vectorizer = _fit_vectorizer(data, train_qids)
        X_all, train_rows_by_qid = _candidate_rows_for_qids(data, train_qids, vectorizer)
        mean, std = _fit_scaler(X_all)
        train_rows_by_qid_scaled = {
            qid: _scale(rows, mean, std) for qid, rows in train_rows_by_qid.items()
        }
        X_pair, y_pair = _pairwise_training_rows(
            data,
            train_qids,
            train_rows_by_qid_scaled,
            max_pairs_per_question=max_pairs_per_question,
        )
        weights = train_pairwise_logistic(X_pair, y_pair, lr=lr, reg=reg, epochs=epochs)
        model = PairwiseRanker(
            feature_names=compose_pairwise_feature_names(),
            weights=weights,
            scaler_mean=mean,
            scaler_std=std,
            idf=vectorizer.idf,
        )
        score_rows = score_items(data, test_qids, model)
        eval_payload = _evaluate_scores(data, test_qids, score_rows)
        for row in eval_payload["rows"]:
            row["fold"] = fold_idx
            all_rows.append(row)
        summaries.append(
            {
                "fold": fold_idx,
                "train_questions": len(train_qids),
                "test_questions": len(test_qids),
                "top1_rate": eval_payload["top1_rate"],
                "any_rate": eval_payload["any_rate"],
                "baseline_top1_rate": eval_payload["baseline_top1_rate"],
            }
        )

    top1 = [row["top1_correct"] for row in all_rows]
    anyc = [row["any_correct"] for row in all_rows]
    base = [row["baseline_top1_correct"] for row in all_rows]
    return {
        "config": {
            "folds": folds,
            "seed": seed,
            "lr": lr,
            "reg": reg,
            "epochs": epochs,
            "max_pairs_per_question": max_pairs_per_question,
            "feature_names": compose_pairwise_feature_names(),
        },
        "overall": {
            "n_questions": len(all_rows),
            "top1_correct": int(sum(top1)),
            "any_correct": int(sum(anyc)),
            "baseline_top1_correct": int(sum(base)),
            "top1_rate": float(np.mean(top1)) if top1 else 0.0,
            "any_rate": float(np.mean(anyc)) if anyc else 0.0,
            "baseline_top1_rate": float(np.mean(base)) if base else 0.0,
        },
        "folds": summaries,
        "oof_predictions": sorted(all_rows, key=lambda row: row["qid"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pairwise candidate reranker.")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--cv-out", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--max-pairs-per-question", type=int, default=64)
    parser.add_argument("--include-gold", action="store_true")
    parser.add_argument("--min-candidates", type=int, default=1)
    args = parser.parse_args()

    data = load_training_data(
        args.training_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )
    if not data:
        raise RuntimeError(f"No usable training data found in {args.training_data}")

    cv = cross_validate_pairwise(
        data,
        folds=args.folds,
        seed=args.seed,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        max_pairs_per_question=args.max_pairs_per_question,
    )
    Path(args.cv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cv_out, "w", encoding="utf-8") as f:
        json.dump(cv, f, indent=2, ensure_ascii=False)
        f.write("\n")

    model = train_final_pairwise_ranker(
        data,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        max_pairs_per_question=args.max_pairs_per_question,
    )
    model.save(
        args.model_out,
        metadata={
            "training_data": args.training_data,
            "questions": len(data),
            "folds_cv": args.folds,
            "seed": args.seed,
            "lr": args.lr,
            "reg": args.reg,
            "epochs": args.epochs,
            "max_pairs_per_question": args.max_pairs_per_question,
            "include_gold": args.include_gold,
        },
    )

    overall = cv["overall"]
    print("===== PAIRWISE RANKER CV =====")
    print(f"Questions: {overall['n_questions']}")
    print(f"Top1: {overall['top1_correct']}/{overall['n_questions']} ({overall['top1_rate']:.3f})")
    print(f"Any:  {overall['any_correct']}/{overall['n_questions']} ({overall['any_rate']:.3f})")
    print(
        f"Baseline top1: {overall['baseline_top1_correct']}/{overall['n_questions']} "
        f"({overall['baseline_top1_rate']:.3f})"
    )
    print(f"Saved CV report: {args.cv_out}")
    print(f"Saved model:     {args.model_out}")


if __name__ == "__main__":
    main()
