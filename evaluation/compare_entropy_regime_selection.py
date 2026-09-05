#!/usr/bin/env python3
"""Compare baseline vs ML selection accuracy within entropy regimes."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _load_results(path: str) -> Dict[str, Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Results must be a JSON object: {path}")
    details = payload.get("details")
    if not isinstance(details, list):
        raise ValueError(f"Results must contain a details list: {path}")
    out: Dict[str, Dict[str, object]] = {}
    for idx, detail in enumerate(details):
        if not isinstance(detail, dict):
            continue
        qid = str(detail.get("id") or f"row_{idx}")
        out[qid] = detail
    return out


def _load_dataset(path: str) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return {}
    return {str(row.get("id", "")): row for row in rows if isinstance(row, dict)}


def _label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "") or "").strip().lower()


def _score(candidate: Dict[str, object], preferred: str) -> Optional[float]:
    keys = [preferred]
    for key in ("ml_score", "selection_score", "score", "semantic_judge_score"):
        if key not in keys:
            keys.append(key)
    for key in keys:
        value = candidate.get(key)
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return score
    return None


def _candidates(detail: Dict[str, object]) -> List[Dict[str, object]]:
    return [c for c in list(detail.get("candidates") or []) if isinstance(c, dict)]


def _top1_correct(detail: Dict[str, object]) -> bool:
    value = detail.get("top1_correct")
    if isinstance(value, bool):
        return value
    candidates = _candidates(detail)
    return bool(candidates and _label(candidates[0]) == "correct")


def _any_correct(detail: Dict[str, object]) -> bool:
    value = detail.get("any_correct")
    if isinstance(value, bool):
        return value
    return any(_label(candidate) == "correct" for candidate in _candidates(detail))


def _first_correct_rank(detail: Dict[str, object]) -> Optional[int]:
    for idx, candidate in enumerate(_candidates(detail), start=1):
        if _label(candidate) == "correct":
            return idx
    return None


def _ranked_for_entropy(
    detail: Dict[str, object],
    *,
    score_key: str,
    sort_by_score: bool,
) -> Tuple[List[Dict[str, object]], List[float]]:
    rows = []
    for candidate in _candidates(detail):
        score = _score(candidate, score_key)
        if score is None:
            continue
        rows.append((candidate, score))
    if sort_by_score:
        rows.sort(key=lambda item: item[1], reverse=True)
    return [row[0] for row in rows], [row[1] for row in rows]


def _probabilities(scores: Sequence[float], method: str, temperature: float) -> List[float]:
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    if method == "auto":
        method = "positive" if min(scores) >= 0 and sum(scores) > 0 else "softmax"
    if method == "positive":
        min_score = min(scores)
        values = list(scores)
        if min_score <= 0:
            values = [score - min_score + 1e-12 for score in scores]
        total = sum(values)
        return [value / total for value in values] if total > 0 else [1.0 / len(scores)] * len(scores)
    if method == "minmax":
        min_score = min(scores)
        max_score = max(scores)
        if math.isclose(min_score, max_score):
            return [1.0 / len(scores)] * len(scores)
        values = [(score - min_score) + 1e-12 for score in scores]
        total = sum(values)
        return [value / total for value in values]
    if method == "softmax":
        temp = max(float(temperature), 1e-9)
        max_score = max(scores)
        values = [math.exp((score - max_score) / temp) for score in scores]
        total = sum(values)
        return [value / total for value in values] if total > 0 else [1.0 / len(scores)] * len(scores)
    raise ValueError(f"Unsupported normalization: {method}")


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def _normalized_entropy(probs: Sequence[float]) -> float:
    return 0.0 if len(probs) <= 1 else _entropy(probs) / math.log(len(probs))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _regime(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def _summarize(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(rows)
    total = len(rows)
    baseline_correct = sum(1 for row in rows if row["baseline_top1_correct"])
    ml_correct = sum(1 for row in rows if row["ml_top1_correct"])
    both_correct = sum(1 for row in rows if row["baseline_top1_correct"] and row["ml_top1_correct"])
    ml_gain = sum(1 for row in rows if not row["baseline_top1_correct"] and row["ml_top1_correct"])
    ml_loss = sum(1 for row in rows if row["baseline_top1_correct"] and not row["ml_top1_correct"])
    return {
        "count": total,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / total if total else 0.0,
        "ml_correct": ml_correct,
        "ml_accuracy": ml_correct / total if total else 0.0,
        "delta_correct": ml_correct - baseline_correct,
        "delta_accuracy": (ml_correct - baseline_correct) / total if total else 0.0,
        "both_correct": both_correct,
        "ml_gain": ml_gain,
        "ml_loss": ml_loss,
        "same_wrong": sum(1 for row in rows if not row["baseline_top1_correct"] and not row["ml_top1_correct"]),
        "any_correct": sum(1 for row in rows if row["any_correct"]),
        "any_correct_rate": sum(1 for row in rows if row["any_correct"]) / total if total else 0.0,
        "avg_normalized_entropy": mean([float(row["normalized_entropy"]) for row in rows]) if rows else 0.0,
        "avg_margin": mean([float(row["margin"]) for row in rows]) if rows else 0.0,
    }


def _dist(rows: Iterable[Dict[str, object]], key: str, limit: int = 20) -> List[Tuple[str, int]]:
    counter: Counter = Counter(str(row.get(key) or "unknown") for row in rows)
    return counter.most_common(limit)


def _one_line(query: object, limit: int = 260) -> str:
    text = " ".join(str(query or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def analyze(
    *,
    baseline_results_path: str,
    ml_results_path: str,
    dataset_path: str,
    entropy_source: str,
    score_key: str,
    sort_by_score: bool,
    normalization: str,
    temperature: float,
    bucket_mode: str,
    low_threshold: float,
    high_threshold: float,
    example_limit: int,
) -> Dict[str, object]:
    baseline = _load_results(baseline_results_path)
    ml = _load_results(ml_results_path)
    dataset = _load_dataset(dataset_path)
    source_map = ml if entropy_source == "ml" else baseline
    shared_ids = [qid for qid in ml if qid in baseline]

    rows: List[Dict[str, object]] = []
    skipped_without_scores: List[str] = []
    for qid in shared_ids:
        source_detail = source_map.get(qid, {})
        ranked, scores = _ranked_for_entropy(source_detail, score_key=score_key, sort_by_score=sort_by_score)
        if len(scores) < 2:
            skipped_without_scores.append(qid)
            continue
        probs = _probabilities(scores, normalization, temperature)
        h_norm = _normalized_entropy(probs)
        dataset_row = dataset.get(qid, {})
        baseline_detail = baseline[qid]
        ml_detail = ml[qid]
        rows.append(
            {
                "id": qid,
                "question": str(
                    ml_detail.get("effective_question")
                    or ml_detail.get("question")
                    or baseline_detail.get("question")
                    or ""
                ),
                "family": str(
                    dataset_row.get("topic")
                    or dataset_row.get("family")
                    or ml_detail.get("family")
                    or baseline_detail.get("family")
                    or "unknown"
                ),
                "ambiguity_label": str(
                    dataset_row.get("ambiguity_label")
                    or dataset_row.get("complexity_label")
                    or ml_detail.get("ambiguity_label")
                    or baseline_detail.get("ambiguity_label")
                    or "unknown"
                ),
                "entropy": _entropy(probs),
                "normalized_entropy": h_norm,
                "score1": scores[0],
                "score2": scores[1],
                "margin": scores[0] - scores[1],
                "baseline_top1_correct": _top1_correct(baseline_detail),
                "ml_top1_correct": _top1_correct(ml_detail),
                "any_correct": _any_correct(ml_detail) or _any_correct(baseline_detail),
                "baseline_first_correct_rank": _first_correct_rank(baseline_detail),
                "ml_first_correct_rank": _first_correct_rank(ml_detail),
                "baseline_top_query": _one_line((_candidates(baseline_detail) or [{}])[0].get("query") if _candidates(baseline_detail) else ""),
                "ml_top_query": _one_line((_candidates(ml_detail) or [{}])[0].get("query") if _candidates(ml_detail) else ""),
                "entropy_top_candidates": [
                    {
                        "rank": idx + 1,
                        "score": score,
                        "probability": probs[idx] if idx < len(probs) else None,
                        "label": _label(candidate),
                        "source": candidate.get("source"),
                        "query": _one_line(candidate.get("query")),
                    }
                    for idx, (candidate, score) in enumerate(zip(ranked[:5], scores[:5]))
                ],
            }
        )

    if bucket_mode == "quantiles":
        low_threshold = _quantile([float(row["normalized_entropy"]) for row in rows], 1 / 3)
        high_threshold = _quantile([float(row["normalized_entropy"]) for row in rows], 2 / 3)

    by_regime: Dict[str, List[Dict[str, object]]] = {"low": [], "medium": [], "high": []}
    for row in rows:
        regime = _regime(float(row["normalized_entropy"]), low_threshold, high_threshold)
        row["entropy_regime"] = regime
        by_regime[regime].append(row)

    examples = {
        "ml_gains": [
            row for row in rows if not row["baseline_top1_correct"] and row["ml_top1_correct"]
        ][:example_limit],
        "ml_losses": [
            row for row in rows if row["baseline_top1_correct"] and not row["ml_top1_correct"]
        ][:example_limit],
        "high_entropy_ml_losses": [
            row
            for row in sorted(rows, key=lambda item: float(item["normalized_entropy"]), reverse=True)
            if row["baseline_top1_correct"] and not row["ml_top1_correct"]
        ][:example_limit],
    }

    return {
        "inputs": {
            "baseline_results": baseline_results_path,
            "ml_results": ml_results_path,
            "dataset": dataset_path,
            "entropy_source": entropy_source,
            "score_key": score_key,
            "sort_by_score": sort_by_score,
            "normalization": normalization,
            "temperature": temperature,
            "bucket_mode": bucket_mode,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
        "summary": {
            "shared_questions": len(shared_ids),
            "total_with_scores": len(rows),
            "skipped_without_scores": len(skipped_without_scores),
            **_summarize(rows),
        },
        "by_entropy_regime": [
            {"regime": regime, **_summarize(by_regime[regime]), "families": _dist(by_regime[regime], "family")}
            for regime in ("low", "medium", "high")
        ],
        "examples": examples,
        "cases": rows,
        "skipped_ids": skipped_without_scores[:100],
    }


def _pct(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def write_markdown(report: Dict[str, object], out_path: str) -> None:
    inputs = dict(report.get("inputs") or {})
    summary = dict(report.get("summary") or {})
    lines: List[str] = []
    lines.append("# Entropy Regime Baseline vs ML Selection")
    lines.append("")
    lines.append(f"- Baseline results: `{inputs.get('baseline_results')}`")
    lines.append(f"- ML results: `{inputs.get('ml_results')}`")
    lines.append(f"- Dataset: `{inputs.get('dataset')}`")
    lines.append(f"- Entropy source: `{inputs.get('entropy_source')}`")
    lines.append(f"- Score key: `{inputs.get('score_key')}`")
    lines.append(f"- Normalization: `{inputs.get('normalization')}`")
    lines.append(
        f"- Thresholds: low <= {float(inputs.get('low_threshold') or 0.0):.3f}, "
        f"medium <= {float(inputs.get('high_threshold') or 0.0):.3f}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Shared questions | {summary.get('shared_questions', 0)} |")
    lines.append(f"| Total with entropy scores | {summary.get('total_with_scores', 0)} |")
    lines.append(f"| Baseline Top-1 | {summary.get('baseline_correct', 0)}/{summary.get('total_with_scores', 0)} ({_pct(summary.get('baseline_accuracy'))}) |")
    lines.append(f"| ML Top-1 | {summary.get('ml_correct', 0)}/{summary.get('total_with_scores', 0)} ({_pct(summary.get('ml_accuracy'))}) |")
    lines.append(f"| Delta | {summary.get('delta_correct', 0)} ({_pct(summary.get('delta_accuracy'))}) |")
    lines.append(f"| ML gains | {summary.get('ml_gain', 0)} |")
    lines.append(f"| ML losses | {summary.get('ml_loss', 0)} |")
    lines.append(f"| Any Correct | {summary.get('any_correct', 0)}/{summary.get('total_with_scores', 0)} ({_pct(summary.get('any_correct_rate'))}) |")
    lines.append("")
    lines.append("## Baseline vs ML By Entropy Regime")
    lines.append("")
    lines.append("| Regime | Count | Baseline Top-1 | ML Top-1 | Delta | ML gains | ML losses | Any correct | Avg H_norm | Avg margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report.get("by_entropy_regime") or []:
        lines.append(
            f"| `{row['regime']}` | {row['count']} | "
            f"{row['baseline_correct']} ({_pct(row['baseline_accuracy'])}) | "
            f"{row['ml_correct']} ({_pct(row['ml_accuracy'])}) | "
            f"{row['delta_correct']} ({_pct(row['delta_accuracy'])}) | "
            f"{row['ml_gain']} | {row['ml_loss']} | "
            f"{row['any_correct']} ({_pct(row['any_correct_rate'])}) | "
            f"{_pct(row['avg_normalized_entropy'])} | {_pct(row['avg_margin'])} |"
        )
    lines.append("")
    for title, key in (
        ("ML Gains", "ml_gains"),
        ("ML Losses", "ml_losses"),
        ("High-Entropy ML Losses", "high_entropy_ml_losses"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for row in dict(report.get("examples") or {}).get(key) or []:
            lines.append(f"### {row.get('id')}")
            lines.append("")
            lines.append(f"Question: {row.get('question')}")
            lines.append("")
            lines.append(
                f"- regime={row.get('entropy_regime')}, H_norm={float(row.get('normalized_entropy', 0.0)):.4f}, "
                f"margin={float(row.get('margin', 0.0)):.4f}"
            )
            lines.append(f"- Baseline top query: `{row.get('baseline_top_query')}`")
            lines.append(f"- ML top query: `{row.get('ml_top_query')}`")
            lines.append("")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and ML query selection by entropy regime.")
    parser.add_argument("--baseline-results", required=True, help="Results before ML reranking or with ML disabled.")
    parser.add_argument("--ml-results", required=True, help="Results after ML reranking/selection.")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--entropy-source", default="ml", choices=["baseline", "ml"])
    parser.add_argument("--score-key", default="ml_score", choices=["ml_score", "selection_score", "score", "semantic_judge_score"])
    parser.add_argument("--sort-by-score", action="store_true")
    parser.add_argument("--normalization", default="softmax", choices=["auto", "positive", "softmax", "minmax"])
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--bucket-mode", default="quantiles", choices=["thresholds", "quantiles"])
    parser.add_argument("--low-threshold", type=float, default=0.33)
    parser.add_argument("--high-threshold", type=float, default=0.66)
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(
        baseline_results_path=args.baseline_results,
        ml_results_path=args.ml_results,
        dataset_path=args.dataset,
        entropy_source=args.entropy_source,
        score_key=args.score_key,
        sort_by_score=args.sort_by_score,
        normalization=args.normalization,
        temperature=args.temperature,
        bucket_mode=args.bucket_mode,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        example_limit=args.example_limit,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_markdown(report, args.out_md)

    summary = report["summary"]
    print("===== ENTROPY REGIME BASELINE VS ML =====")
    print(f"Baseline: {args.baseline_results}")
    print(f"ML:       {args.ml_results}")
    print(f"Total with scores: {summary['total_with_scores']}")
    print(
        f"Baseline Top1: {summary['baseline_correct']}/{summary['total_with_scores']} "
        f"({summary['baseline_accuracy']:.3f})"
    )
    print(
        f"ML Top1:       {summary['ml_correct']}/{summary['total_with_scores']} "
        f"({summary['ml_accuracy']:.3f})"
    )
    print(f"Delta:         {summary['delta_correct']} ({summary['delta_accuracy']:.3f})")
    print("By entropy regime:")
    for row in report["by_entropy_regime"]:
        print(
            f"  {row['regime']}: baseline={row['baseline_correct']}/{row['count']} "
            f"({row['baseline_accuracy']:.3f}), ml={row['ml_correct']}/{row['count']} "
            f"({row['ml_accuracy']:.3f}), delta={row['delta_correct']} "
            f"({row['delta_accuracy']:.3f}), gains={row['ml_gain']}, losses={row['ml_loss']}"
        )
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
