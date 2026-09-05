#!/usr/bin/env python3
"""Analyze candidate-set ambiguity with entropy over query candidate scores."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Results JSON must be an object with a details list.")
    return payload


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
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _question(detail: Dict[str, object]) -> str:
    return str(detail.get("effective_question") or detail.get("question") or "")


def _ranked_candidates(
    detail: Dict[str, object],
    score_key: str,
    *,
    sort_by_score: bool,
) -> List[Dict[str, object]]:
    candidates = [c for c in list(detail.get("candidates") or []) if isinstance(c, dict)]
    if not sort_by_score:
        return candidates
    return sorted(candidates, key=lambda c: (_score(c, score_key) is not None, _score(c, score_key) or -math.inf), reverse=True)


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
        if total <= 0:
            return [1.0 / len(scores)] * len(scores)
        return [value / total for value in values]

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
        if total <= 0:
            return [1.0 / len(scores)] * len(scores)
        return [value / total for value in values]

    raise ValueError(f"Unsupported normalization method: {method}")


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def _normalized_entropy(probs: Sequence[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    return _entropy(probs) / math.log(len(probs))


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


def _regime(value: float, low_threshold: float, high_threshold: float) -> str:
    if value <= low_threshold:
        return "low"
    if value <= high_threshold:
        return "medium"
    return "high"


def _summarize(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(rows)
    total = len(rows)
    correct = sum(1 for row in rows if row.get("top1_correct"))
    any_correct = sum(1 for row in rows if row.get("any_correct"))
    return {
        "count": total,
        "top1_correct": correct,
        "top1_accuracy": (correct / total) if total else 0.0,
        "any_correct": any_correct,
        "any_correct_rate": (any_correct / total) if total else 0.0,
        "avg_entropy": mean([float(row["entropy"]) for row in rows]) if rows else 0.0,
        "avg_normalized_entropy": mean([float(row["normalized_entropy"]) for row in rows]) if rows else 0.0,
        "avg_margin": mean([float(row["margin"]) for row in rows]) if rows else 0.0,
        "avg_score1": mean([float(row["score1"]) for row in rows]) if rows else 0.0,
    }


def _distribution(rows: Iterable[Dict[str, object]], key: str, limit: int = 20) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for row in rows:
        counter[str(row.get(key) or "unknown")] += 1
    return counter.most_common(limit)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _top_candidates(
    candidates: Sequence[Dict[str, object]],
    score_key: str,
    probs: Sequence[float],
    limit: int = 5,
) -> List[Dict[str, object]]:
    out = []
    for idx, candidate in enumerate(candidates[:limit]):
        out.append(
            {
                "rank": idx + 1,
                "label": _label(candidate),
                "score": _score(candidate, score_key),
                "probability": probs[idx] if idx < len(probs) else None,
                "source": candidate.get("source"),
                "query": " ".join(str(candidate.get("query") or "").split())[:320],
            }
        )
    return out


def _case_rows(
    results: Dict[str, object],
    dataset_by_id: Dict[str, Dict[str, object]],
    *,
    score_key: str,
    sort_by_score: bool,
    normalization: str,
    temperature: float,
) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for detail in results.get("details") or []:
        if not isinstance(detail, dict):
            continue
        qid = str(detail.get("id", "") or "")
        candidates = _ranked_candidates(detail, score_key, sort_by_score=sort_by_score)
        scores: List[float] = []
        usable_candidates: List[Dict[str, object]] = []
        for candidate in candidates:
            score = _score(candidate, score_key)
            if score is None:
                continue
            scores.append(score)
            usable_candidates.append(candidate)
        if len(scores) < 2:
            skipped.append(qid or f"row_{len(skipped) + 1}")
            continue
        probs = _probabilities(scores, normalization, temperature)
        entropy = _entropy(probs)
        normalized_entropy = _normalized_entropy(probs)
        top1 = usable_candidates[0]
        top2 = usable_candidates[1]
        dataset_row = dataset_by_id.get(qid, {})
        rows.append(
            {
                "id": qid,
                "question": _question(detail),
                "family": str(
                    dataset_row.get("topic")
                    or dataset_row.get("family")
                    or detail.get("family")
                    or "unknown"
                ),
                "ambiguity_label": str(
                    dataset_row.get("ambiguity_label")
                    or dataset_row.get("complexity_label")
                    or detail.get("ambiguity_label")
                    or "unknown"
                ),
                "candidate_count": len(usable_candidates),
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "score1": scores[0],
                "score2": scores[1],
                "margin": scores[0] - scores[1],
                "top1_correct": _label(top1) == "correct",
                "any_correct": any(_label(candidate) == "correct" for candidate in usable_candidates),
                "first_correct_rank": next(
                    (idx + 1 for idx, candidate in enumerate(usable_candidates) if _label(candidate) == "correct"),
                    None,
                ),
                "top_candidates": _top_candidates(usable_candidates, score_key, probs),
            }
        )
    return rows, skipped


def analyze(
    *,
    results_path: str,
    dataset_path: str,
    score_key: str,
    sort_by_score: bool,
    normalization: str,
    temperature: float,
    bucket_mode: str,
    low_threshold: float,
    high_threshold: float,
    example_limit: int,
) -> Dict[str, object]:
    results = _load_json(results_path)
    dataset_by_id = _load_dataset(dataset_path)
    rows, skipped = _case_rows(
        results,
        dataset_by_id,
        score_key=score_key,
        sort_by_score=sort_by_score,
        normalization=normalization,
        temperature=temperature,
    )
    if bucket_mode == "quantiles":
        low_threshold = _quantile([float(row["normalized_entropy"]) for row in rows], 1 / 3)
        high_threshold = _quantile([float(row["normalized_entropy"]) for row in rows], 2 / 3)

    by_regime: Dict[str, List[Dict[str, object]]] = {"low": [], "medium": [], "high": []}
    for row in rows:
        regime = _regime(float(row["normalized_entropy"]), low_threshold, high_threshold)
        row["entropy_regime"] = regime
        by_regime[regime].append(row)

    total = len(rows)
    correct = sum(1 for row in rows if row["top1_correct"])
    any_correct = sum(1 for row in rows if row["any_correct"])
    norm_entropy_values = [float(row["normalized_entropy"]) for row in rows]
    margin_values = [float(row["margin"]) for row in rows]
    score_values = [float(row["score1"]) for row in rows]
    correct_values = [1.0 if row["top1_correct"] else 0.0 for row in rows]

    high_entropy_wrong = [
        row
        for row in sorted(rows, key=lambda item: float(item["normalized_entropy"]), reverse=True)
        if not row["top1_correct"]
    ][:example_limit]
    low_entropy_wrong = [
        row
        for row in sorted(rows, key=lambda item: float(item["normalized_entropy"]))
        if not row["top1_correct"]
    ][:example_limit]

    return {
        "inputs": {
            "results": results_path,
            "dataset": dataset_path,
            "score_key": score_key,
            "sort_by_score": sort_by_score,
            "normalization": normalization,
            "temperature": temperature,
            "bucket_mode": bucket_mode,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
        "summary": {
            "total_with_scores": total,
            "skipped_without_scores": len(skipped),
            "forced_top1_correct": correct,
            "forced_top1_accuracy": (correct / total) if total else 0.0,
            "any_correct": any_correct,
            "any_correct_rate": (any_correct / total) if total else 0.0,
            "avg_entropy": mean([float(row["entropy"]) for row in rows]) if rows else 0.0,
            "avg_normalized_entropy": mean(norm_entropy_values) if rows else 0.0,
            "entropy_margin_pearson": _pearson(norm_entropy_values, margin_values),
            "entropy_score1_pearson": _pearson(norm_entropy_values, score_values),
            "entropy_correct_pearson": _pearson(norm_entropy_values, correct_values),
        },
        "accuracy_by_entropy_regime": [
            {"regime": name, **_summarize(by_regime[name])}
            for name in ("low", "medium", "high")
        ],
        "family_by_entropy_regime": {
            name: _distribution(by_regime[name], "family")
            for name in ("low", "medium", "high")
        },
        "existing_ambiguity_labels": _distribution(rows, "ambiguity_label"),
        "high_entropy_wrong_examples": high_entropy_wrong,
        "low_entropy_wrong_examples": low_entropy_wrong,
        "cases": rows,
        "skipped_ids": skipped[:100],
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
    lines.append("# Entropy-Based Ambiguity Analysis")
    lines.append("")
    lines.append(f"- Results: `{inputs.get('results')}`")
    lines.append(f"- Dataset: `{inputs.get('dataset')}`")
    lines.append(f"- Score key: `{inputs.get('score_key')}`")
    lines.append(f"- Sort by score: `{inputs.get('sort_by_score')}`")
    lines.append(f"- Normalization: `{inputs.get('normalization')}`")
    lines.append(f"- Bucket mode: `{inputs.get('bucket_mode')}`")
    lines.append(
        f"- Entropy thresholds: low <= {float(inputs.get('low_threshold') or 0.0):.3f}, "
        f"medium <= {float(inputs.get('high_threshold') or 0.0):.3f}, high above"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total with scores | {summary.get('total_with_scores', 0)} |")
    lines.append(f"| Skipped without scores | {summary.get('skipped_without_scores', 0)} |")
    lines.append(
        f"| Forced Top-1 | {summary.get('forced_top1_correct', 0)}/"
        f"{summary.get('total_with_scores', 0)} ({_pct(summary.get('forced_top1_accuracy'))}) |"
    )
    lines.append(
        f"| Any Correct | {summary.get('any_correct', 0)}/"
        f"{summary.get('total_with_scores', 0)} ({_pct(summary.get('any_correct_rate'))}) |"
    )
    lines.append(f"| Avg normalized entropy | {_pct(summary.get('avg_normalized_entropy'))} |")
    lines.append(f"| Corr(entropy, margin) | {_pct(summary.get('entropy_margin_pearson'))} |")
    lines.append(f"| Corr(entropy, score1) | {_pct(summary.get('entropy_score1_pearson'))} |")
    lines.append(f"| Corr(entropy, correct) | {_pct(summary.get('entropy_correct_pearson'))} |")
    lines.append("")
    lines.append("## Accuracy By Entropy Regime")
    lines.append("")
    lines.append("| Regime | Count | Top-1 correct | Top-1 accuracy | Any correct | Any-correct rate | Avg H_norm | Avg margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report.get("accuracy_by_entropy_regime") or []:
        lines.append(
            f"| `{row['regime']}` | {row['count']} | {row['top1_correct']} | "
            f"{_pct(row['top1_accuracy'])} | {row['any_correct']} | "
            f"{_pct(row['any_correct_rate'])} | {_pct(row['avg_normalized_entropy'])} | "
            f"{_pct(row['avg_margin'])} |"
        )
    lines.append("")
    lines.append("## Family Composition By Regime")
    lines.append("")
    for regime, values in dict(report.get("family_by_entropy_regime") or {}).items():
        lines.append(f"### {str(regime).title()}")
        lines.append("")
        lines.append("| Family | Count |")
        lines.append("|---|---:|")
        for family, count in values:
            lines.append(f"| `{family}` | {count} |")
        lines.append("")
    for title, key in (
        ("High-Entropy Wrong Examples", "high_entropy_wrong_examples"),
        ("Low-Entropy Wrong Examples", "low_entropy_wrong_examples"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for row in report.get(key) or []:
            lines.append(f"### {row.get('id')}")
            lines.append("")
            lines.append(f"Question: {row.get('question')}")
            lines.append("")
            lines.append(
                f"- H={float(row.get('entropy', 0.0)):.4f}, "
                f"H_norm={float(row.get('normalized_entropy', 0.0)):.4f}, "
                f"margin={float(row.get('margin', 0.0)):.4f}, "
                f"first_correct_rank={row.get('first_correct_rank')}"
            )
            lines.append("- Top candidates:")
            for candidate in row.get("top_candidates") or []:
                lines.append(
                    f"  {candidate['rank']}. p={float(candidate.get('probability') or 0.0):.3f}, "
                    f"score={float(candidate.get('score') or 0.0):.3f}, "
                    f"label={candidate.get('label')}, source={candidate.get('source')}: "
                    f"`{candidate.get('query')}`"
                )
            lines.append("")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute entropy-based ambiguity regimes from KGQA candidate scores.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--score-key", default="ml_score", choices=["ml_score", "selection_score", "score", "semantic_judge_score"])
    parser.add_argument("--sort-by-score", action="store_true", help="Sort candidates by score before computing Top-1 and entropy examples.")
    parser.add_argument("--normalization", default="auto", choices=["auto", "positive", "softmax", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature when --normalization softmax is used.")
    parser.add_argument("--bucket-mode", default="thresholds", choices=["thresholds", "quantiles"])
    parser.add_argument("--low-threshold", type=float, default=0.33, help="Low/medium boundary for normalized entropy.")
    parser.add_argument("--high-threshold", type=float, default=0.66, help="Medium/high boundary for normalized entropy.")
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(
        results_path=args.results,
        dataset_path=args.dataset,
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
    print("===== ENTROPY AMBIGUITY ANALYSIS =====")
    print(f"Results: {args.results}")
    print(f"Score key: {args.score_key}")
    print(f"Normalization: {args.normalization}")
    print(
        f"Thresholds: low<={report['inputs']['low_threshold']:.3f}, "
        f"medium<={report['inputs']['high_threshold']:.3f}"
    )
    print(f"Total with scores: {summary['total_with_scores']}")
    print(f"Skipped without scores: {summary['skipped_without_scores']}")
    print(
        f"Forced Top1: {summary['forced_top1_correct']}/{summary['total_with_scores']} "
        f"({summary['forced_top1_accuracy']:.3f})"
    )
    print(
        f"Any Correct: {summary['any_correct']}/{summary['total_with_scores']} "
        f"({summary['any_correct_rate']:.3f})"
    )
    print(f"Average normalized entropy: {summary['avg_normalized_entropy']:.3f}")
    print("Accuracy by entropy regime:")
    for row in report["accuracy_by_entropy_regime"]:
        print(
            f"  {row['regime']}: {row['top1_correct']}/{row['count']} "
            f"({row['top1_accuracy']:.3f}), any={row['any_correct_rate']:.3f}, "
            f"avg_H_norm={row['avg_normalized_entropy']:.3f}, avg_margin={row['avg_margin']:.3f}"
        )
    print("Correlations:")
    print(f"  entropy vs margin: {summary['entropy_margin_pearson']}")
    print(f"  entropy vs score1: {summary['entropy_score1_pearson']}")
    print(f"  entropy vs correct: {summary['entropy_correct_pearson']}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
