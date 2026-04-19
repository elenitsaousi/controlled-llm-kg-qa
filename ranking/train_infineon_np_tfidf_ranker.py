#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.np_tfidf_ranker import (
    cross_validate_ranker,
    load_training_data,
    train_final_ranker,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate numpy TF-IDF ranker for Infineon benchmark."
    )
    parser.add_argument(
        "--training-data",
        default="ranking/infineon_training_data_100.json",
        help="Candidate-level labeled training data JSON.",
    )
    parser.add_argument(
        "--cv-out",
        default="results/infineon_ranker_cv_100.json",
        help="Where to save CV (out-of-fold) evaluation JSON.",
    )
    parser.add_argument(
        "--model-out",
        default="ranking/models/infineon_np_tfidf_ranker.json",
        help="Where to save final deployment model.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Include gold candidates in training/CV (debug only, can cause leakage).",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=1,
        help="Minimum candidate count per question after filtering.",
    )
    args = parser.parse_args()

    data = load_training_data(
        args.training_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )
    if not data:
        raise RuntimeError(
            f"No usable training data found in {args.training_data}. "
            "If your file contains only gold rows, rebuild training data with working LLM generation "
            "or run with --include-gold for debug-only diagnostics."
        )

    cv = cross_validate_ranker(
        data,
        n_folds=args.folds,
        seed=args.seed,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
    )
    Path(args.cv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cv_out, "w", encoding="utf-8") as f:
        json.dump(cv, f, indent=2, ensure_ascii=False)
        f.write("\n")

    model = train_final_ranker(
        data,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
    )
    model.save(
        args.model_out,
        metadata={
            "training_data": args.training_data,
            "n_questions": len(data),
            "folds_cv": args.folds,
            "seed": args.seed,
            "lr": args.lr,
            "reg": args.reg,
            "epochs": args.epochs,
            "include_gold": args.include_gold,
            "min_candidates": args.min_candidates,
        },
    )

    overall = cv["overall"]
    print("===== CV SUMMARY (Out-of-Fold) =====")
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
    print(f"Include gold rows: {args.include_gold}")
    print(f"Saved CV report: {args.cv_out}")
    print(f"Saved model:     {args.model_out}")


if __name__ == "__main__":
    main()
