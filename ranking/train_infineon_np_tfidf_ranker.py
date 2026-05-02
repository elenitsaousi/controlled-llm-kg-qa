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
    evaluate_query_plan_predictor,
    load_training_data,
    load_query_plan_training_rows,
    train_final_ranker,
    train_query_plan_predictor,
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
    parser.add_argument(
        "--train-query-plan",
        action="store_true",
        help="Train the question -> query-plan-label predictor from gold SPARQL.",
    )
    parser.add_argument(
        "--query-plan-data",
        default="data/infineon/infineon_train.json",
        help="Gold question/SPARQL dataset for query-plan predictor.",
    )
    parser.add_argument(
        "--schema",
        default="data/infineon/schema.json",
        help="Schema path used to extract query-plan labels.",
    )
    parser.add_argument(
        "--query-plan-model-out",
        default="ranking/models/infineon_query_plan_predictor.json",
        help="Where to save the query-plan predictor.",
    )
    parser.add_argument(
        "--query-plan-report-out",
        default="results/infineon_query_plan_predictor_eval.json",
        help="Where to save query-plan predictor training-set diagnostics.",
    )
    parser.add_argument("--query-plan-min-label-count", type=int, default=2)
    parser.add_argument("--query-plan-threshold", type=float, default=0.35)
    parser.add_argument("--query-plan-top-k", type=int, default=24)
    args = parser.parse_args()

    if args.train_query_plan:
        rows = load_query_plan_training_rows(args.query_plan_data, args.schema)
        model = train_query_plan_predictor(
            rows,
            min_label_count=args.query_plan_min_label_count,
            threshold=args.query_plan_threshold,
            top_k=args.query_plan_top_k,
        )
        report = evaluate_query_plan_predictor(
            model,
            rows,
            threshold=args.query_plan_threshold,
            top_k=args.query_plan_top_k,
        )
        Path(args.query_plan_report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.query_plan_report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")
        model.save(
            args.query_plan_model_out,
            metadata={
                "training_data": args.query_plan_data,
                "schema": args.schema,
                "min_label_count": args.query_plan_min_label_count,
                "threshold": args.query_plan_threshold,
                "top_k": args.query_plan_top_k,
                "rows": len(rows),
            },
        )
        print("===== QUERY PLAN PREDICTOR =====")
        print(f"Questions: {report['questions']}")
        print(f"Labels:    {report['labels']}")
        print(f"Macro F1:  {report['macro_f1']:.3f}")
        print(f"Exact:     {report['exact_match_rate']:.3f}")
        print(f"Saved report: {args.query_plan_report_out}")
        print(f"Saved model:  {args.query_plan_model_out}")
        return

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
