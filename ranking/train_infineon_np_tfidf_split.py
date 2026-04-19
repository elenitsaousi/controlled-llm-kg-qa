#!/usr/bin/env python3
"""
Train NP TF-IDF ranker on train split and report clean dev/test KPIs.

Uses pre-generated candidate pools (no online LLM calls), so policy comparisons
are done on the exact same candidate set.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rdflib import Graph

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.ambiguity_policy import (
    AmbiguityConfig,
    load_ambiguity_config,
    normalize_label,
    predict_regime,
)
from ranking.np_tfidf_ranker import (
    NPTfidfRanker,
    load_training_data,
    train_final_ranker,
)


def _parse_ml_regimes(text: str) -> List[str]:
    allowed = {"low", "mid", "high"}
    out = []
    for tok in text.split(","):
        lab = normalize_label(tok)
        if not lab:
            continue
        if lab not in allowed:
            raise ValueError(f"Invalid label '{tok}'. Allowed: low,mid,high")
        if lab not in out:
            out.append(lab)
    if not out:
        out = ["mid"]
    return out


def _evaluate_split(
    data: Dict[str, object],
    model: NPTfidfRanker,
    ml_regimes: Sequence[str],
    ambiguity_config: Optional[AmbiguityConfig] = None,
    schema_dict: Optional[Dict[str, object]] = None,
    agreement_graph: Optional[Graph] = None,
) -> Dict[str, object]:
    per_label = defaultdict(lambda: {
        "questions": 0,
        "generation_recall": 0,
        "no_ml_top1": 0,
        "ml_top1": 0,
        "gated_top1": 0,
    })
    per_predicted = defaultdict(lambda: {"questions": 0, "ml_used": 0})

    summary = {
        "questions": 0,
        "candidate_total": 0,
        "candidate_correct": 0,
        "candidate_invalid": 0,
        "generation_recall": 0,   # any-correct in candidate pool
        "no_ml_top1": 0,
        "ml_top1": 0,
        "gated_top1": 0,
        "ml_used_count": 0,
    }

    details = []
    for qid, item in sorted(data.items()):
        candidates = item.candidates
        if not candidates:
            continue

        summary["questions"] += 1
        summary["candidate_total"] += len(candidates)
        summary["candidate_correct"] += sum(int(c.is_correct == 1) for c in candidates)
        summary["candidate_invalid"] += sum(int(c.is_valid == 0) for c in candidates)

        question = item.question
        label = normalize_label(item.ambiguity_label)
        any_correct = int(any(c.is_correct == 1 for c in candidates))

        queries = [c.query for c in candidates]
        base_features = [c.features for c in candidates]
        scores = model.score_question_candidates(question, queries, base_features)
        ml_idx = int(scores.argmax()) if len(scores) else 0

        no_ml_top1 = int(candidates[0].is_correct == 1)
        ml_top1 = int(candidates[ml_idx].is_correct == 1)

        if ambiguity_config is not None:
            regime, entropy = predict_regime(
                question=question,
                candidates=[
                    {"query": c.query, "features": c.features}
                    for c in candidates
                ],
                config=ambiguity_config,
                schema_dict=schema_dict,
                model=model,
                graph=agreement_graph,
            )
        else:
            regime, entropy = label, None

        use_ml = int(regime in ml_regimes)
        gated_top1 = ml_top1 if use_ml else no_ml_top1

        summary["generation_recall"] += any_correct
        summary["no_ml_top1"] += no_ml_top1
        summary["ml_top1"] += ml_top1
        summary["gated_top1"] += gated_top1
        summary["ml_used_count"] += use_ml

        pl = per_label[label]
        pl["questions"] += 1
        pl["generation_recall"] += any_correct
        pl["no_ml_top1"] += no_ml_top1
        pl["ml_top1"] += ml_top1
        pl["gated_top1"] += gated_top1

        pp = per_predicted[regime]
        pp["questions"] += 1
        pp["ml_used"] += use_ml

        details.append(
            {
                "qid": qid,
                "ambiguity_label": label,
                "predicted_regime": regime,
                "entropy": entropy,
                "any_correct": any_correct,
                "no_ml_top1": no_ml_top1,
                "ml_top1": ml_top1,
                "gated_top1": gated_top1,
                "ml_used": use_ml,
                "n_candidates": len(candidates),
            }
        )

    qn = max(1, summary["questions"])
    cn = max(1, summary["candidate_total"])
    summary["generation_recall_rate"] = summary["generation_recall"] / qn
    summary["no_ml_top1_rate"] = summary["no_ml_top1"] / qn
    summary["ml_top1_rate"] = summary["ml_top1"] / qn
    summary["gated_top1_rate"] = summary["gated_top1"] / qn
    summary["ml_usage_rate"] = summary["ml_used_count"] / qn
    summary["candidate_correct_rate"] = summary["candidate_correct"] / cn
    summary["candidate_invalid_rate"] = summary["candidate_invalid"] / cn
    summary["delta_gated_vs_no_ml"] = summary["gated_top1_rate"] - summary["no_ml_top1_rate"]
    summary["delta_gated_vs_ml"] = summary["gated_top1_rate"] - summary["ml_top1_rate"]

    per_label_out = {}
    for lab, s in sorted(per_label.items()):
        n = max(1, s["questions"])
        per_label_out[lab] = {
            "questions": s["questions"],
            "generation_recall_rate": s["generation_recall"] / n,
            "no_ml_top1_rate": s["no_ml_top1"] / n,
            "ml_top1_rate": s["ml_top1"] / n,
            "gated_top1_rate": s["gated_top1"] / n,
        }

    per_predicted_out = {}
    for reg, s in sorted(per_predicted.items()):
        n = max(1, s["questions"])
        per_predicted_out[reg] = {
            "questions": s["questions"],
            "ml_usage_rate": s["ml_used"] / n,
        }

    return {
        "summary": summary,
        "per_ambiguity_label": per_label_out,
        "per_predicted_regime": per_predicted_out,
        "details": details,
    }


def _print_split_report(name: str, rep: Dict[str, object]) -> None:
    s = rep["summary"]
    print(f"\n===== {name.upper()} =====")
    print(f"Questions: {s['questions']}")
    print(f"Generation recall: {s['generation_recall']}/{s['questions']} ({s['generation_recall_rate']:.3f})")
    print(f"No-ML top1: {s['no_ml_top1']}/{s['questions']} ({s['no_ml_top1_rate']:.3f})")
    print(f"ML-all top1: {s['ml_top1']}/{s['questions']} ({s['ml_top1_rate']:.3f})")
    print(f"Gated top1: {s['gated_top1']}/{s['questions']} ({s['gated_top1_rate']:.3f})")
    print(f"Δ gated vs no-ML: {s['delta_gated_vs_no_ml']:+.3f}")
    print(f"Δ gated vs ML-all: {s['delta_gated_vs_ml']:+.3f}")
    print(f"ML usage: {s['ml_usage_rate']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Infineon NP TF-IDF ranker on train split and report dev/test KPIs."
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--model-out", default="ranking/models/infineon_np_tfidf_ranker_split.json")
    parser.add_argument("--report-out", default="results/infineon_split_kpi_report.json")
    parser.add_argument("--ml-regimes", default="mid")
    parser.add_argument("--ambiguity-config", default="")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--include-gold", action="store_true")
    parser.add_argument("--min-candidates", type=int, default=1)
    args = parser.parse_args()

    train = load_training_data(
        args.train_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )
    dev = load_training_data(
        args.dev_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )
    test = load_training_data(
        args.test_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )

    if not train:
        raise RuntimeError("Train split has no usable questions.")
    if not dev:
        raise RuntimeError("Dev split has no usable questions.")
    if not test:
        raise RuntimeError("Test split has no usable questions.")

    model = train_final_ranker(
        train,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
    )
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(
        args.model_out,
        metadata={
            "train_data": args.train_data,
            "dev_data": args.dev_data,
            "test_data": args.test_data,
            "lr": args.lr,
            "reg": args.reg,
            "epochs": args.epochs,
            "include_gold": args.include_gold,
            "min_candidates": args.min_candidates,
        },
    )

    ml_regimes = _parse_ml_regimes(args.ml_regimes)
    ambiguity_config = None
    schema_dict = None
    agreement_graph = None
    if args.ambiguity_config:
        ambiguity_config = load_ambiguity_config(args.ambiguity_config)
        if ambiguity_config.entropy_source in {"schema", "agreement", "ml"}:
            with open(args.schema, "r", encoding="utf-8") as f:
                schema_dict = json.load(f)
        if ambiguity_config.entropy_source == "agreement":
            agreement_graph = Graph()
            agreement_graph.parse(args.graph, format="turtle")

    train_rep = _evaluate_split(
        train,
        model,
        ml_regimes=ml_regimes,
        ambiguity_config=ambiguity_config,
        schema_dict=schema_dict,
        agreement_graph=agreement_graph,
    )
    dev_rep = _evaluate_split(
        dev,
        model,
        ml_regimes=ml_regimes,
        ambiguity_config=ambiguity_config,
        schema_dict=schema_dict,
        agreement_graph=agreement_graph,
    )
    test_rep = _evaluate_split(
        test,
        model,
        ml_regimes=ml_regimes,
        ambiguity_config=ambiguity_config,
        schema_dict=schema_dict,
        agreement_graph=agreement_graph,
    )

    report = {
        "config": {
            "train_data": args.train_data,
            "dev_data": args.dev_data,
            "test_data": args.test_data,
            "model_out": args.model_out,
            "ml_regimes": ml_regimes,
            "ambiguity_config": args.ambiguity_config or None,
            "lr": args.lr,
            "reg": args.reg,
            "epochs": args.epochs,
            "include_gold": args.include_gold,
            "min_candidates": args.min_candidates,
        },
        "train": train_rep,
        "dev": dev_rep,
        "test": test_rep,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    _print_split_report("train", train_rep)
    _print_split_report("dev", dev_rep)
    _print_split_report("test", test_rep)
    print(f"\nSaved model: {args.model_out}")
    print(f"Saved KPI report: {args.report_out}")


if __name__ == "__main__":
    main()
