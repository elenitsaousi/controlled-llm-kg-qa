#!/usr/bin/env python3
"""Explain why schema-only selection can fail while ML reranking helps.

The script combines three kinds of evidence:

* headline selection metrics from analysis JSON files;
* switch counts between raw/schema/ML result files when detailed candidate
  files are available;
* absolute ranker weights grouped into interpretable feature families.

It is intended for thesis reporting, not for training.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_json(path: str | None) -> Dict[str, object]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary(payload: Dict[str, object]) -> Dict[str, object]:
    summary = dict(payload.get("summary") or payload)
    total = int(summary.get("total") or summary.get("questions") or 0)
    top1 = int(summary.get("top1_correct") or summary.get("correct") or 0)
    any_correct = int(summary.get("any_correct") or 0)
    return {
        "total": total,
        "top1_correct": top1,
        "top1_accuracy": top1 / total if total else 0.0,
        "any_correct": any_correct,
        "any_accuracy": any_correct / total if total else 0.0,
        "ranking_failures": int(summary.get("ranking_failures") or max(any_correct - top1, 0)),
        "generation_failures": int(summary.get("generation_failures") or max(total - any_correct, 0)),
    }


def _details(payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    rows = payload.get("details") or payload.get("rows") or []
    if not isinstance(rows, list):
        return {}
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("id") or row.get("question_id") or row.get("request_id") or "")
        if qid:
            out[qid] = row
    return out


def _top_label(row: Dict[str, object]) -> str:
    candidates = row.get("candidates") or []
    if isinstance(candidates, list) and candidates:
        return str(candidates[0].get("label") or "").strip().lower()
    return str(row.get("top_label") or row.get("top1_label") or "").strip().lower()


def _is_correct(row: Dict[str, object]) -> bool:
    value = row.get("top1_correct")
    if isinstance(value, bool):
        return value
    return _top_label(row) == "correct"


def _feature_family(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("invalid", "schema", "predicate", "relation", "entity", "coverage", "precision")):
        return "schema/semantic coverage"
    if any(x in n for x in ("aggregation", "group", "select", "distinct", "limit", "order", "rank")):
        return "answer shape and query form"
    if any(x in n for x in ("scope", "origin", "survey", "oem", "tier", "semiconductor")):
        return "survey scope and provenance"
    if any(x in n for x in ("row", "empty", "count", "result", "source", "template")):
        return "execution/candidate source"
    if any(x in n for x in ("node", "rel", "triple", "optional", "where", "exists")):
        return "structural complexity"
    if n.startswith("tfidf") or "token" in n:
        return "lexical alignment"
    return "other"


def _ranker_features(model_path: str | None, limit: int = 20) -> Dict[str, object]:
    model = _load_json(model_path)
    names = model.get("feature_names") or []
    weights = model.get("weights") or []
    if not isinstance(names, list) or not isinstance(weights, list):
        return {"available": False, "reason": "model has no feature_names/weights arrays"}
    rows = []
    family_weight = Counter()
    for name, weight in zip(names, weights):
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        family = _feature_family(str(name))
        rows.append({"feature": str(name), "weight": w, "abs_weight": abs(w), "family": family})
        family_weight[family] += abs(w)
    rows.sort(key=lambda row: (-float(row["abs_weight"]), str(row["feature"])))
    return {
        "available": True,
        "model": model_path,
        "top_features": rows[:limit],
        "family_abs_weight": [
            {"family": family, "abs_weight": weight}
            for family, weight in family_weight.most_common()
        ],
    }


def _switches(before: Dict[str, Dict[str, object]], after: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    counts = Counter()
    for qid, b in before.items():
        a = after.get(qid)
        if not a:
            continue
        b_ok = _is_correct(b)
        a_ok = _is_correct(a)
        if b_ok and a_ok:
            counts["correct_to_correct"] += 1
        elif b_ok and not a_ok:
            counts["lost_correct"] += 1
        elif not b_ok and a_ok:
            counts["rescued"] += 1
        else:
            counts["wrong_to_wrong"] += 1
    return dict(counts)


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    raw_analysis = _summary(_load_json(args.raw_analysis))
    schema_analysis = _summary(_load_json(args.schema_analysis))
    ml_analysis = _summary(_load_json(args.ml_analysis))

    raw_details = _details(_load_json(args.raw_results))
    schema_details = _details(_load_json(args.schema_results))
    ml_details = _details(_load_json(args.ml_results))

    switches = {}
    if raw_details and schema_details:
        switches["schema_vs_raw"] = _switches(raw_details, schema_details)
    if raw_details and ml_details:
        switches["ml_vs_raw"] = _switches(raw_details, ml_details)
    if schema_details and ml_details:
        switches["ml_vs_schema"] = _switches(schema_details, ml_details)

    return {
        "inputs": vars(args),
        "metrics": {
            "raw": raw_analysis,
            "schema_only": schema_analysis,
            "guarded_ml": ml_analysis,
        },
        "switches": switches,
        "ranker_feature_evidence": _ranker_features(args.model, args.feature_limit),
        "interpretation": (
            "Schema validity is useful as a feature but insufficient as a decision rule. "
            "A structurally valid query can still use the wrong metric, scope, aggregation, "
            "or answer shape. The ML reranker can improve over schema-only selection because "
            "it combines schema signals with query-form, provenance, lexical, candidate-source, "
            "and safety features instead of treating schema validity as the final objective."
        ),
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _write_md(report: Dict[str, object], out_md: str) -> None:
    lines = [
        "# Schema-Only Selection vs Guarded ML",
        "",
        "## Selection Metrics",
        "",
        "| Mode | Questions | Top-1 | Any Correct | Ranking Failures | Generation Failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [("raw", "Raw LLM order"), ("schema_only", "Schema-only"), ("guarded_ml", "Guarded ML")]:
        row = report["metrics"][key]
        lines.append(
            f"| {label} | {row['total']} | {row['top1_correct']} ({_pct(row['top1_accuracy'])}) | "
            f"{row['any_correct']} ({_pct(row['any_accuracy'])}) | {row['ranking_failures']} | {row['generation_failures']} |"
        )
    lines.extend(["", "## Interpretation", "", str(report["interpretation"]), ""])

    if report["switches"]:
        lines.extend(["## Selection Switch Evidence", ""])
        for name, counts in report["switches"].items():
            lines.append(f"- `{name}`: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        lines.append("")

    feature_info = report["ranker_feature_evidence"]
    if feature_info.get("available"):
        lines.extend(["## Ranker Feature Evidence", "", "| Feature family | Sum abs. weight |", "|---|---:|"])
        for row in feature_info["family_abs_weight"]:
            lines.append(f"| {row['family']} | {float(row['abs_weight']):.3f} |")
        lines.extend(["", "Top individual absolute weights:", ""])
        for row in feature_info["top_features"]:
            lines.append(f"- `{row['feature']}` ({row['family']}): weight={float(row['weight']):.3f}")
    else:
        lines.extend(["## Ranker Feature Evidence", "", f"Not available: {feature_info.get('reason')}."])

    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze schema-only failure and guarded ML feature evidence.")
    parser.add_argument("--raw-analysis", required=True)
    parser.add_argument("--schema-analysis", required=True)
    parser.add_argument("--ml-analysis", required=True)
    parser.add_argument("--raw-results")
    parser.add_argument("--schema-results")
    parser.add_argument("--ml-results")
    parser.add_argument("--model")
    parser.add_argument("--feature-limit", type=int, default=20)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(args)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(report, args.out_md)

    print("===== SCHEMA VS ML SIGNAL ANALYSIS =====")
    for key, label in [("raw", "Raw"), ("schema_only", "Schema"), ("guarded_ml", "GuardedML")]:
        row = report["metrics"][key]
        print(f"{label}: {row['top1_correct']}/{row['total']} ({row['top1_accuracy']:.3f}), any={row['any_accuracy']:.3f}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
