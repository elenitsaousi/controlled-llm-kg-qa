#!/usr/bin/env python3
"""
Fit ambiguity-threshold policy (tau1/tau2) for gated ML ranking.

Input: pre-generated candidate pools + trained ranker model.
Output: ambiguity config JSON for runtime/test use without dataset ambiguity labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from rdflib import Graph

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.ambiguity_policy import (
    AmbiguityConfig,
    estimate_entropy,
    normalize_label,
    regime_from_entropy,
    save_ambiguity_config,
)
from ranking.np_tfidf_ranker import NPTfidfRanker, load_training_data


def _parse_float_grid(text: str) -> List[float]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("Empty quantile grid")
    return vals


def _parse_labels(text: str) -> List[str]:
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


def _collect_rows(
    data: Dict[str, object],
    model: NPTfidfRanker,
    entropy_source: str,
    schema_dict: Dict[str, object],
    graph: Graph | None,
    agreement_top_n: int,
    agreement_invalid_penalty: float,
) -> List[Dict[str, object]]:
    rows = []
    for qid, item in sorted(data.items()):
        candidates = item.candidates
        if not candidates:
            continue
        question = item.question
        queries = [c.query for c in candidates]
        feats = [c.features for c in candidates]
        scores = model.score_question_candidates(question, queries, feats)
        ml_idx = int(scores.argmax()) if len(scores) else 0

        entropy = estimate_entropy(
            question=question,
            candidates=[{"query": c.query, "features": c.features} for c in candidates],
            source=entropy_source,
            schema_dict=schema_dict,
            model=model,
            graph=graph,
            agreement_top_n=agreement_top_n,
            agreement_invalid_penalty=agreement_invalid_penalty,
        )

        rows.append(
            {
                "qid": qid,
                "entropy": float(entropy),
                "ambiguity_label": normalize_label(item.ambiguity_label),
                "no_ml_top1": int(candidates[0].is_correct == 1),
                "ml_top1": int(candidates[ml_idx].is_correct == 1),
                "any_correct": int(any(c.is_correct == 1 for c in candidates)),
            }
        )
    return rows


def _tune_thresholds(
    rows: Sequence[Dict[str, object]],
    ml_regimes: Sequence[str],
    q1_grid: Sequence[float],
    q2_grid: Sequence[float],
) -> Dict[str, float]:
    ent = np.array([float(r["entropy"]) for r in rows], dtype=float)
    if ent.size == 0:
        return {"tau1": 0.33, "tau2": 0.66, "gated_top1": 0.0, "ml_usage": 0.0}

    best = None
    eps = 1e-12
    for q1 in q1_grid:
        tau1 = float(np.quantile(ent, q1))
        for q2 in q2_grid:
            if q2 <= q1:
                continue
            tau2 = float(np.quantile(ent, q2))
            gated_vals = []
            ml_usage = []
            for r in rows:
                reg = regime_from_entropy(float(r["entropy"]), tau1, tau2)
                use_ml = int(reg in ml_regimes)
                val = int(r["ml_top1"]) if use_ml else int(r["no_ml_top1"])
                gated_vals.append(val)
                ml_usage.append(use_ml)
            gated = float(np.mean(gated_vals)) if gated_vals else 0.0
            usage = float(np.mean(ml_usage)) if ml_usage else 0.0

            cand = {
                "tau1": tau1,
                "tau2": tau2,
                "gated_top1": gated,
                "ml_usage": usage,
            }

            if best is None:
                best = cand
                continue
            if gated > best["gated_top1"] + eps:
                best = cand
            elif abs(gated - best["gated_top1"]) <= eps:
                if usage < best["ml_usage"] - eps:
                    best = cand
                elif abs(usage - best["ml_usage"]) <= eps:
                    if (tau2 - tau1) > (best["tau2"] - best["tau1"]):
                        best = cand
    assert best is not None
    return best


def _evaluate_thresholds(
    rows: Sequence[Dict[str, object]],
    tau1: float,
    tau2: float,
    ml_regimes: Sequence[str],
) -> Dict[str, object]:
    no_ml = [int(r["no_ml_top1"]) for r in rows]
    ml = [int(r["ml_top1"]) for r in rows]
    gated = []
    any_correct = [int(r["any_correct"]) for r in rows]
    y_true = []
    y_pred = []
    ml_used = []

    for r in rows:
        reg = regime_from_entropy(float(r["entropy"]), tau1, tau2)
        use_ml = int(reg in ml_regimes)
        gated_val = int(r["ml_top1"]) if use_ml else int(r["no_ml_top1"])
        gated.append(gated_val)
        ml_used.append(use_ml)
        y_true.append(normalize_label(str(r.get("ambiguity_label", ""))))
        y_pred.append(reg)

    n = max(1, len(rows))
    labels = ["low", "mid", "high"]
    conf = {t: {p: 0 for p in labels} for t in labels}
    valid = 0
    for t, p in zip(y_true, y_pred):
        if t not in labels or p not in labels:
            continue
        conf[t][p] += 1
        valid += 1
    acc = None
    if valid > 0:
        acc = float(sum(conf[l][l] for l in labels) / valid)

    return {
        "questions": len(rows),
        "no_ml_top1_rate": float(np.mean(no_ml)) if no_ml else 0.0,
        "ml_top1_rate": float(np.mean(ml)) if ml else 0.0,
        "gated_top1_rate": float(np.mean(gated)) if gated else 0.0,
        "any_correct_rate": float(np.mean(any_correct)) if any_correct else 0.0,
        "ml_usage_rate": float(np.mean(ml_used)) if ml_used else 0.0,
        "delta_gated_vs_no_ml": (float(np.mean(gated)) - float(np.mean(no_ml))) if no_ml else 0.0,
        "delta_gated_vs_ml": (float(np.mean(gated)) - float(np.mean(ml))) if ml else 0.0,
        "label_accuracy": acc,
        "label_confusion": conf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit ambiguity-threshold policy for gated ML ranking."
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--calib-data", required=True, help="Usually dev split candidate pools.")
    parser.add_argument("--model", required=True, help="Trained NP TF-IDF ranker JSON.")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--entropy-source", choices=["schema", "ml", "agreement"], default="agreement")
    parser.add_argument("--agreement-top-n", type=int, default=3)
    parser.add_argument("--agreement-invalid-penalty", type=float, default=0.20)
    parser.add_argument("--ml-regimes", default="mid")
    parser.add_argument("--q1-grid", default="0.10,0.20,0.25,0.30,0.33,0.40")
    parser.add_argument("--q2-grid", default="0.60,0.66,0.70,0.75,0.80,0.90")
    parser.add_argument("--include-gold", action="store_true")
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--out-config", default="ranking/models/infineon_ambiguity_config.json")
    parser.add_argument("--out-report", default="results/infineon_ambiguity_calibration.json")
    args = parser.parse_args()

    with open(args.schema, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)

    graph = None
    if args.entropy_source == "agreement":
        graph = Graph()
        graph.parse(args.graph, format="turtle")

    train = load_training_data(
        args.train_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )
    calib = load_training_data(
        args.calib_data,
        include_gold=args.include_gold,
        min_candidates=args.min_candidates,
    )
    if not train:
        raise RuntimeError("train-data has no usable questions.")
    if not calib:
        raise RuntimeError("calib-data has no usable questions.")

    model = NPTfidfRanker.load(args.model)
    ml_regimes = _parse_labels(args.ml_regimes)
    q1_grid = _parse_float_grid(args.q1_grid)
    q2_grid = _parse_float_grid(args.q2_grid)

    train_rows = _collect_rows(
        data=train,
        model=model,
        entropy_source=args.entropy_source,
        schema_dict=schema_dict,
        graph=graph,
        agreement_top_n=args.agreement_top_n,
        agreement_invalid_penalty=args.agreement_invalid_penalty,
    )
    calib_rows = _collect_rows(
        data=calib,
        model=model,
        entropy_source=args.entropy_source,
        schema_dict=schema_dict,
        graph=graph,
        agreement_top_n=args.agreement_top_n,
        agreement_invalid_penalty=args.agreement_invalid_penalty,
    )

    tuned = _tune_thresholds(
        rows=calib_rows,
        ml_regimes=ml_regimes,
        q1_grid=q1_grid,
        q2_grid=q2_grid,
    )
    tau1 = float(tuned["tau1"])
    tau2 = float(tuned["tau2"])

    config = AmbiguityConfig(
        entropy_source=args.entropy_source,
        tau1=tau1,
        tau2=tau2,
        ml_regimes=ml_regimes,
        agreement_top_n=args.agreement_top_n,
        agreement_invalid_penalty=args.agreement_invalid_penalty,
    )
    Path(args.out_config).parent.mkdir(parents=True, exist_ok=True)
    save_ambiguity_config(
        args.out_config,
        config,
        metadata={
            "train_data": args.train_data,
            "calib_data": args.calib_data,
            "model": args.model,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
            "include_gold": args.include_gold,
            "min_candidates": args.min_candidates,
        },
    )

    train_eval = _evaluate_thresholds(train_rows, tau1, tau2, ml_regimes=ml_regimes)
    calib_eval = _evaluate_thresholds(calib_rows, tau1, tau2, ml_regimes=ml_regimes)
    report = {
        "config": {
            "entropy_source": args.entropy_source,
            "ml_regimes": ml_regimes,
            "agreement_top_n": args.agreement_top_n,
            "agreement_invalid_penalty": args.agreement_invalid_penalty,
            "tau1": tau1,
            "tau2": tau2,
        },
        "tuning": tuned,
        "train_eval": train_eval,
        "calibration_eval": calib_eval,
        "rows": {
            "train": len(train_rows),
            "calib": len(calib_rows),
            "calib_label_distribution": dict(Counter(r["ambiguity_label"] for r in calib_rows)),
        },
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("===== AMBIGUITY CALIBRATION =====")
    print(f"Entropy source: {args.entropy_source}")
    print(f"tau1={tau1:.6f} tau2={tau2:.6f}")
    print(f"ML regimes: {','.join(ml_regimes)}")
    print(f"Calibration gated top1: {calib_eval['gated_top1_rate']:.3f}")
    print(f"Calibration no-ML top1: {calib_eval['no_ml_top1_rate']:.3f}")
    print(f"Calibration ML-all top1: {calib_eval['ml_top1_rate']:.3f}")
    print(f"Saved ambiguity config: {args.out_config}")
    print(f"Saved calibration report: {args.out_report}")


if __name__ == "__main__":
    main()
