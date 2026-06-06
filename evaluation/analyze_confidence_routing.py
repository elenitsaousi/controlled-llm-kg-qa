#!/usr/bin/env python3
"""Evaluate confidence-aware answer/clarification routing for KGQA results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Results JSON must be an object.")
    return payload


def _label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "") or "").strip().lower()


def _score(candidate: Dict[str, object], preferred: str) -> float:
    keys = [preferred]
    if preferred != "ml_score":
        keys.append("ml_score")
    if preferred != "selection_score":
        keys.append("selection_score")
    keys.extend(["score", "semantic_judge_score"])
    for key in keys:
        value = candidate.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _question(detail: Dict[str, object]) -> str:
    return str(
        detail.get("effective_question")
        or detail.get("question")
        or ""
    )


def _one_line_query(query: str, limit: int = 260) -> str:
    text = " ".join(str(query or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _bucket(value: float, buckets: List[Tuple[str, Optional[float], Optional[float]]]) -> str:
    for name, low, high in buckets:
        if low is not None and value <= low:
            continue
        if high is not None and value > high:
            continue
        return name
    return "unbucketed"


def _summarize_bucket(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(rows)
    total = len(rows)
    correct = sum(1 for row in rows if row["top1_correct"])
    return {
        "count": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


def _ranked_candidates(detail: Dict[str, object], score_key: str) -> List[Dict[str, object]]:
    candidates = [c for c in list(detail.get("candidates") or []) if isinstance(c, dict)]
    # The results are normally already reranked. Sorting here only makes the
    # analysis robust when a result file stores candidate scores but not order.
    return sorted(candidates, key=lambda c: _score(c, score_key), reverse=True)


def _case_rows(results: Dict[str, object], score_key: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for detail in results.get("details") or []:
        if not isinstance(detail, dict):
            continue
        candidates = _ranked_candidates(detail, score_key)
        if not candidates:
            continue
        top1 = candidates[0]
        top2 = candidates[1] if len(candidates) > 1 else {}
        score1 = _score(top1, score_key)
        score2 = _score(top2, score_key) if top2 else 0.0
        top3 = []
        for idx, candidate in enumerate(candidates[:3], start=1):
            top3.append(
                {
                    "rank": idx,
                    "score": _score(candidate, score_key),
                    "label": _label(candidate),
                    "source": candidate.get("source"),
                    "query": _one_line_query(str(candidate.get("query", "") or "")),
                }
            )
        rows.append(
            {
                "id": detail.get("id"),
                "question": _question(detail),
                "score1": score1,
                "score2": score2,
                "margin": score1 - score2,
                "top1_correct": _label(top1) == "correct",
                "any_correct": any(_label(candidate) == "correct" for candidate in candidates),
                "top3": top3,
            }
        )
    return rows


def _threshold_sweep(
    rows: List[Dict[str, object]],
    *,
    margin_thresholds: List[float],
    score_thresholds: List[float],
) -> List[Dict[str, object]]:
    total = len(rows)
    report: List[Dict[str, object]] = []
    for min_margin in margin_thresholds:
        for min_score in score_thresholds:
            answered = [
                row
                for row in rows
                if float(row["margin"]) >= min_margin and float(row["score1"]) >= min_score
            ]
            correct = sum(1 for row in answered if row["top1_correct"])
            report.append(
                {
                    "min_margin": min_margin,
                    "min_score": min_score,
                    "answered": len(answered),
                    "clarified": total - len(answered),
                    "coverage": (len(answered) / total) if total else 0.0,
                    "correct": correct,
                    "accuracy_when_answered": (correct / len(answered)) if answered else 0.0,
                }
            )
    return sorted(
        report,
        key=lambda row: (
            float(row["accuracy_when_answered"]),
            float(row["coverage"]),
            int(row["answered"]),
        ),
        reverse=True,
    )


def analyze(
    *,
    results_path: str,
    score_key: str,
    margin_thresholds: List[float],
    score_thresholds: List[float],
    low_confidence_limit: int,
) -> Dict[str, object]:
    results = _load_json(results_path)
    rows = _case_rows(results, score_key)
    total = len(rows)
    forced_correct = sum(1 for row in rows if row["top1_correct"])
    any_correct = sum(1 for row in rows if row["any_correct"])

    margin_buckets = [
        ("margin_gt_0.50", 0.50, None),
        ("margin_0.30_to_0.50", 0.30, 0.50),
        ("margin_0.10_to_0.30", 0.10, 0.30),
        ("margin_lte_0.10", None, 0.10),
    ]
    score_buckets = [
        ("score_gt_0.80", 0.80, None),
        ("score_0.60_to_0.80", 0.60, 0.80),
        ("score_lte_0.60", None, 0.60),
    ]

    margin_rows: Dict[str, List[Dict[str, object]]] = {name: [] for name, _, _ in margin_buckets}
    score_rows: Dict[str, List[Dict[str, object]]] = {name: [] for name, _, _ in score_buckets}
    for row in rows:
        margin_rows[_bucket(float(row["margin"]), margin_buckets)].append(row)
        score_rows[_bucket(float(row["score1"]), score_buckets)].append(row)

    low_confidence = sorted(rows, key=lambda row: (float(row["margin"]), float(row["score1"])))[:low_confidence_limit]

    return {
        "inputs": {
            "results": results_path,
            "score_key": score_key,
        },
        "summary": {
            "total": total,
            "forced_top1_correct": forced_correct,
            "forced_top1_accuracy": (forced_correct / total) if total else 0.0,
            "any_correct": any_correct,
            "any_correct_rate": (any_correct / total) if total else 0.0,
        },
        "accuracy_by_margin": [
            {"bucket": name, **_summarize_bucket(margin_rows[name])}
            for name, _, _ in margin_buckets
        ],
        "accuracy_by_score": [
            {"bucket": name, **_summarize_bucket(score_rows[name])}
            for name, _, _ in score_buckets
        ],
        "threshold_sweep": _threshold_sweep(
            rows,
            margin_thresholds=margin_thresholds,
            score_thresholds=score_thresholds,
        ),
        "low_confidence_examples": low_confidence,
    }


def _format_pct(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def write_markdown(report: Dict[str, object], out_path: str) -> None:
    summary = dict(report.get("summary") or {})
    lines: List[str] = []
    lines.append("# Confidence-Aware Routing Analysis")
    lines.append("")
    lines.append(f"- Results: `{dict(report.get('inputs') or {}).get('results')}`")
    lines.append(f"- Score key: `{dict(report.get('inputs') or {}).get('score_key')}`")
    lines.append(f"- Total: {summary.get('total', 0)}")
    lines.append(
        f"- Forced Top1: {summary.get('forced_top1_correct', 0)}/{summary.get('total', 0)} "
        f"({_format_pct(summary.get('forced_top1_accuracy'))})"
    )
    lines.append(
        f"- Any Correct: {summary.get('any_correct', 0)}/{summary.get('total', 0)} "
        f"({_format_pct(summary.get('any_correct_rate'))})"
    )
    lines.append("")
    lines.append("## Accuracy by Margin")
    lines.append("")
    lines.append("| Margin bucket | Count | Correct | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for row in report.get("accuracy_by_margin") or []:
        lines.append(
            f"| `{row['bucket']}` | {row['count']} | {row['correct']} | {_format_pct(row['accuracy'])} |"
        )
    lines.append("")
    lines.append("## Accuracy by Score")
    lines.append("")
    lines.append("| Score bucket | Count | Correct | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for row in report.get("accuracy_by_score") or []:
        lines.append(
            f"| `{row['bucket']}` | {row['count']} | {row['correct']} | {_format_pct(row['accuracy'])} |"
        )
    lines.append("")
    lines.append("## Threshold Sweep")
    lines.append("")
    lines.append("| Min margin | Min score | Answered | Coverage | Correct | Accuracy when answered | Clarified |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in (report.get("threshold_sweep") or [])[:30]:
        lines.append(
            f"| {row['min_margin']:.2f} | {row['min_score']:.2f} | {row['answered']} | "
            f"{_format_pct(row['coverage'])} | {row['correct']} | "
            f"{_format_pct(row['accuracy_when_answered'])} | {row['clarified']} |"
        )
    lines.append("")
    lines.append("## Low-Confidence Examples")
    lines.append("")
    for row in report.get("low_confidence_examples") or []:
        lines.append(f"### {row.get('id')}")
        lines.append("")
        lines.append(f"Question: {row.get('question')}")
        lines.append("")
        lines.append(
            f"- score1={float(row.get('score1', 0.0)):.4f}, "
            f"score2={float(row.get('score2', 0.0)):.4f}, "
            f"margin={float(row.get('margin', 0.0)):.4f}, "
            f"top1_correct={row.get('top1_correct')}"
        )
        lines.append("- Top interpretations:")
        for candidate in row.get("top3") or []:
            lines.append(
                f"  {candidate['rank']}. score={float(candidate['score']):.4f}, "
                f"label={candidate['label']}, source={candidate.get('source')}: "
                f"`{candidate['query']}`"
            )
        lines.append("")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze selective answer/clarification routing from candidate scores.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--score-key", default="ml_score", choices=["ml_score", "selection_score"])
    parser.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--score-thresholds", default="0.00,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--low-confidence-limit", type=int, default=12)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(
        results_path=args.results,
        score_key=args.score_key,
        margin_thresholds=_parse_float_list(args.margin_thresholds),
        score_thresholds=_parse_float_list(args.score_thresholds),
        low_confidence_limit=args.low_confidence_limit,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_markdown(report, args.out_md)

    summary = report["summary"]
    print("===== CONFIDENCE ROUTING ANALYSIS =====")
    print(f"Results: {args.results}")
    print(
        f"Forced Top1: {summary['forced_top1_correct']}/{summary['total']} "
        f"({summary['forced_top1_accuracy']:.3f})"
    )
    print(
        f"Any Correct: {summary['any_correct']}/{summary['total']} "
        f"({summary['any_correct_rate']:.3f})"
    )
    print("Accuracy by margin:")
    for row in report["accuracy_by_margin"]:
        print(f"  {row['bucket']}: {row['correct']}/{row['count']} ({row['accuracy']:.3f})")
    print("Accuracy by score:")
    for row in report["accuracy_by_score"]:
        print(f"  {row['bucket']}: {row['correct']}/{row['count']} ({row['accuracy']:.3f})")
    print("Best threshold policies:")
    for row in report["threshold_sweep"][:10]:
        print(
            "  "
            f"margin>={row['min_margin']:.2f}, score>={row['min_score']:.2f}: "
            f"answered={row['answered']} coverage={row['coverage']:.3f} "
            f"accuracy={row['accuracy_when_answered']:.3f}"
        )
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
