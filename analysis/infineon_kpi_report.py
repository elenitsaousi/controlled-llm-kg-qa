#!/usr/bin/env python3
"""
Build a concise KPI report for Infineon experiments.

Supported inputs:
1) split-training report from ranking/train_infineon_np_tfidf_split.py
2) optional runtime evaluation JSONs from evaluation/run_infineon_100_eval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def _load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _runtime_kpis(payload: Dict) -> Dict[str, object]:
    s = payload.get("summary", {})
    return {
        "total_questions": int(s.get("total", 0)),
        "top1_rate": float(s.get("top1_correct_rate", 0.0)),
        "any_rate": float(s.get("any_correct_rate", 0.0)),
        "candidate_correct_rate": float(s.get("candidate_correct_rate", 0.0)),
        "candidate_invalid_rate": float(s.get("candidate_invalid_rate", 0.0)),
        "llm_generation_failures": int(s.get("llm_generation_failures", 0)),
        "ml_ranking": bool(s.get("ml_ranking", False)),
        "ml_ambiguity_regimes": s.get("ml_ambiguity_regimes", []),
    }


def _split_kpis(payload: Dict) -> Dict[str, object]:
    out = {}
    for split in ("train", "dev", "test"):
        s = payload.get(split, {}).get("summary", {})
        out[split] = {
            "questions": int(s.get("questions", 0)),
            "generation_recall_rate": float(s.get("generation_recall_rate", 0.0)),
            "no_ml_top1_rate": float(s.get("no_ml_top1_rate", 0.0)),
            "ml_top1_rate": float(s.get("ml_top1_rate", 0.0)),
            "gated_top1_rate": float(s.get("gated_top1_rate", 0.0)),
            "delta_gated_vs_no_ml": float(s.get("delta_gated_vs_no_ml", 0.0)),
            "delta_gated_vs_ml": float(s.get("delta_gated_vs_ml", 0.0)),
            "candidate_correct_rate": float(s.get("candidate_correct_rate", 0.0)),
            "candidate_invalid_rate": float(s.get("candidate_invalid_rate", 0.0)),
            "ml_usage_rate": float(s.get("ml_usage_rate", 0.0)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Infineon KPI results.")
    parser.add_argument("--split-report", default="", help="Output JSON from train_infineon_np_tfidf_split.py")
    parser.add_argument("--eval-no-ml", default="", help="Runtime eval JSON (no ML).")
    parser.add_argument("--eval-ml-all", default="", help="Runtime eval JSON (ML all).")
    parser.add_argument("--eval-gated", default="", help="Runtime eval JSON (ML gated).")
    parser.add_argument("--out", default="results/infineon_kpi_summary.json")
    args = parser.parse_args()

    split_payload = _load_json(args.split_report)
    eval_no_ml = _load_json(args.eval_no_ml)
    eval_ml_all = _load_json(args.eval_ml_all)
    eval_gated = _load_json(args.eval_gated)

    report = {
        "split_experiment": _split_kpis(split_payload) if split_payload else None,
        "runtime_eval": {
            "no_ml": _runtime_kpis(eval_no_ml) if eval_no_ml else None,
            "ml_all": _runtime_kpis(eval_ml_all) if eval_ml_all else None,
            "gated": _runtime_kpis(eval_gated) if eval_gated else None,
        },
    }

    # Optional runtime deltas
    if report["runtime_eval"]["no_ml"] and report["runtime_eval"]["gated"]:
        report["runtime_eval"]["delta_gated_vs_no_ml"] = (
            report["runtime_eval"]["gated"]["top1_rate"]
            - report["runtime_eval"]["no_ml"]["top1_rate"]
        )
    if report["runtime_eval"]["ml_all"] and report["runtime_eval"]["gated"]:
        report["runtime_eval"]["delta_gated_vs_ml_all"] = (
            report["runtime_eval"]["gated"]["top1_rate"]
            - report["runtime_eval"]["ml_all"]["top1_rate"]
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("===== KPI SUMMARY =====")
    if split_payload:
        t = report["split_experiment"]["test"]
        print(
            "Split test: "
            f"recall={t['generation_recall_rate']:.3f} "
            f"no-ml={t['no_ml_top1_rate']:.3f} "
            f"ml-all={t['ml_top1_rate']:.3f} "
            f"gated={t['gated_top1_rate']:.3f}"
        )
    if report["runtime_eval"]["no_ml"]:
        n = report["runtime_eval"]["no_ml"]
        print(f"Runtime no-ML: top1={n['top1_rate']:.3f} any={n['any_rate']:.3f}")
    if report["runtime_eval"]["ml_all"]:
        m = report["runtime_eval"]["ml_all"]
        print(f"Runtime ML-all: top1={m['top1_rate']:.3f} any={m['any_rate']:.3f}")
    if report["runtime_eval"]["gated"]:
        g = report["runtime_eval"]["gated"]
        print(f"Runtime gated: top1={g['top1_rate']:.3f} any={g['any_rate']:.3f}")
    if "delta_gated_vs_no_ml" in report["runtime_eval"]:
        print(f"Δ gated vs no-ML: {report['runtime_eval']['delta_gated_vs_no_ml']:+.3f}")
    if "delta_gated_vs_ml_all" in report["runtime_eval"]:
        print(f"Δ gated vs ML-all: {report['runtime_eval']['delta_gated_vs_ml_all']:+.3f}")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
