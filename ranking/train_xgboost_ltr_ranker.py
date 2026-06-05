#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.xgboost_ranker import (
    cross_validate_xgboost_ltr_ranker,
    load_training_data,
    train_final_xgboost_ltr_ranker,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an XGBoost learning-to-rank selector for KGQA reranking."
    )
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--cv-out", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=10.0)
    parser.add_argument(
        "--objective",
        default="rank:pairwise",
        choices=["rank:pairwise", "rank:ndcg", "rank:map"],
    )
    parser.add_argument("--include-gold", action="store_true")
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--disable-feature", action="append", default=[])
    parser.add_argument("--disable-feature-prefix", action="append", default=[])
    args = parser.parse_args()

    data = load_training_data(
        args.training_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )
    if not data:
        raise RuntimeError(f"No usable training data found in {args.training_data}.")

    cv = cross_validate_xgboost_ltr_ranker(
        data,
        n_folds=args.folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        objective=args.objective,
        disabled_feature_names=args.disable_feature,
        disabled_feature_prefixes=args.disable_feature_prefix,
    )
    Path(args.cv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cv_out, "w", encoding="utf-8") as f:
        json.dump(cv, f, indent=2, ensure_ascii=False)
        f.write("\n")

    model = train_final_xgboost_ltr_ranker(
        data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        objective=args.objective,
        seed=args.seed,
        disabled_feature_names=args.disable_feature,
        disabled_feature_prefixes=args.disable_feature_prefix,
    )
    model.save(
        args.model_out,
        metadata={
            "training_mode": "learning_to_rank",
            "training_data": args.training_data,
            "n_questions": len(data),
            "folds_cv": args.folds,
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.reg_lambda,
            "objective": args.objective,
            "include_gold": args.include_gold,
            "min_candidates": args.min_candidates,
            "disabled_feature_names": list(args.disable_feature),
            "disabled_feature_prefixes": list(args.disable_feature_prefix),
        },
    )

    overall = cv["overall"]
    print("===== XGBOOST LTR CV SUMMARY (Out-of-Fold) =====")
    print(f"Questions: {overall['n_questions']}")
    print(
        f"Top1: {overall['top1_correct']}/{overall['n_questions']} "
        f"({overall['top1_rate']:.3f})"
    )
    print(
        f"Any:  {overall['any_correct']}/{overall['n_questions']} "
        f"({overall['any_rate']:.3f})"
    )
    print(
        f"Baseline top1: {overall['baseline_top1_correct']}/{overall['n_questions']} "
        f"({overall['baseline_top1_rate']:.3f})"
    )
    print(f"Objective: {args.objective}")
    print(f"Saved CV report: {args.cv_out}")
    print(f"Saved model:     {args.model_out}")


if __name__ == "__main__":
    main()
